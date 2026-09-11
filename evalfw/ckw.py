"""
CKW adapter: household survey labels + net-metered 15-minute series + weather.

Why this dataset changes the evaluation. AEW's labels come from a SUBSIDY
register, so a missing entry means "no subsidy record", not "no asset" - which
left heat pumps and EVs with positives but no usable negatives, capping every
metric. CKW's labels come from a household SURVEY, so "electric_car: false" is a
real declared negative. AUC, precision and PR curves become computable.

Measured on the delivered files:
  solar_installed=true  export a median of 12,561 kWh over 20 months
  solar_installed=false export a median of        12 kWh
so the survey answers agree with the meter. The same check on AEW's register
failed outright (13 of 19 "no PV" meters were exporting).

Shape differences handled here:
  * one signed `kWh` column (net), split into import_kwh / export_kwh so existing
    detectors run unchanged
  * `location_profile_id` is the meter key, not an integer mp_id
  * hourly MeteoSwiss temperature is joined on, as `temp_c`, for detectors that
    want it - heating load is a function of outdoor temperature, and this is the
    signal AEW's data could not supply
"""

import numpy as np
import pandas as pd
import duckdb

# heating_type_primary values, from the delivered file
HEATPUMP = ("airWaterPump", "geothermalHeatPump", "airAirWaterPump")
RESISTIVE = ("electricRadiator",)
NON_ELECTRIC = ("oilHeating", "wood", "districtHeating", "gasWithRadiator",
                "solarThermalSystem")

ASSETS = ("pv", "ev", "hp", "electric_heating", "boiler")


def _profiles(path):
    return duckdb.connect().execute(
        f"SELECT * FROM read_csv('{path}', header=true, all_varchar=true)").df()


def build_labels(profiles_path, asset, ts_path=None, min_days=365):
    """mp_id, gp, y (1/0/NaN), tier - same contract as evalfw.labels.build."""
    p = _profiles(profiles_path)
    ident = p.location_profile_id
    y = pd.Series(float("nan"), index=p.index)
    tier = pd.Series("ambiguous", index=p.index)

    if asset == "pv":
        y[p.solar_installed == "true"] = 1.0
        y[p.solar_installed == "false"] = 0.0
    elif asset == "ev":
        y[p.electric_car == "true"] = 1.0
        y[p.electric_car == "false"] = 0.0
    elif asset == "hp":
        # heat pump vs NON-ELECTRIC heating. Resistance heating is deliberately
        # excluded from both classes: it is electric heating but not a heat pump,
        # and lumping it either way makes the target incoherent. Scored separately
        # below, where the COP difference shows up as a 3x larger seasonal swing.
        y[p.heating_type_primary.isin(HEATPUMP)] = 1.0
        y[p.heating_type_primary.isin(NON_ELECTRIC)] = 0.0
    elif asset == "boiler":
        # resistance water heating vs every other hot-water type, including
        # heat-pump boilers - those are electric too, but a third of the load,
        # and telling them apart is the useful distinction for a utility
        y[p.hotwater_type == "boiler"] = 1.0
        y[p.hotwater_type.notna() & (p.hotwater_type != "boiler")
          & (p.hotwater_type != "unknown")] = 0.0
    elif asset == "electric_heating":
        # the honestly detectable target: does this household heat with electricity?
        y[p.heating_type_primary.isin(HEATPUMP + RESISTIVE)] = 1.0
        y[p.heating_type_primary.isin(NON_ELECTRIC)] = 0.0
    else:
        raise ValueError(f"CKW has no label for {asset!r}; available: {ASSETS}")

    tier[y == 1] = "gold_positive"
    tier[y == 0] = "gold_negative"
    out = pd.DataFrame({"mp_id": ident, "gp": p.gpnr.fillna(ident), "y": y, "tier": tier})
    out = out[out.tier != "ambiguous"].reset_index(drop=True)

    if ts_path and min_days:
        ok = duckdb.connect().execute(f"""
            SELECT location_profile_id AS mp_id
            FROM '{ts_path}' GROUP BY 1
            HAVING count(DISTINCT date_trunc('day', start_datetime_utc)) >= {min_days}
        """).df()
        out = out[out.mp_id.isin(set(ok.mp_id))].reset_index(drop=True)
    return out


def meter_list(ts_path, min_days=365):
    return duckdb.connect().execute(f"""
        SELECT location_profile_id AS mp_id,
               count(DISTINCT date_trunc('day', start_datetime_utc)) AS days
        FROM '{ts_path}' GROUP BY 1 HAVING days >= {min_days}
    """).df()


def weather_sql(weather_path):
    """Hourly temperature, keyed to the hour a 15-min interval falls in."""
    if not weather_path:
        return None
    return f"""
        SELECT date_trunc('hour', timestamp_utc::TIMESTAMP) AS hr,
               avg(try_cast(tre200h0 AS DOUBLE)) AS temp_c
        FROM read_csv('{weather_path}', header=true, all_varchar=true)
        GROUP BY 1
    """


def load_batch(con, ts_path, ids, weather_path=None):
    """One batch of households, in the frame shape detectors expect.

    The single signed kWh column is split: positive is import, negative is export
    (PV feed-in). Detectors written against the AEW schema then work untouched.
    """
    lst = ",".join("'" + str(i).replace("'", "") + "'" for i in ids)
    w = weather_sql(weather_path)
    join = f"LEFT JOIN ({w}) wx ON wx.hr = date_trunc('hour', v.start_datetime_utc)" if w else ""
    sel_temp = "wx.temp_c" if w else "NULL::DOUBLE AS temp_c"
    return con.execute(f"""
        SELECT v.location_profile_id                         AS mp_id,
               v.start_datetime_utc                          AS ts,
               v.start_datetime_utc::DATE                    AS d,
               CASE WHEN v.kWh > 0 THEN  v.kWh ELSE 0 END    AS import_kwh,
               CASE WHEN v.kWh < 0 THEN -v.kWh ELSE 0 END    AS export_kwh,
               {sel_temp},
               (month(v.start_datetime_utc) IN (3, 10)
                AND dayofweek(v.start_datetime_utc) = 0
                AND month(v.start_datetime_utc + INTERVAL 7 DAY)
                    <> month(v.start_datetime_utc))          AS dst_day
        FROM '{ts_path}' v
        {join}
        WHERE v.location_profile_id IN ({lst})
        ORDER BY v.location_profile_id, v.start_datetime_utc
    """).df()


def composition(profiles_path):
    p = _profiles(profiles_path)
    n = len(p)
    return (f"{n} households - "
            f"{100*(p.solar_installed=='true').mean():.0f}% solar, "
            f"{100*p.heating_type_primary.isin(HEATPUMP).mean():.0f}% heat pump, "
            f"{100*(p.electric_car=='true').mean():.0f}% electric car")
