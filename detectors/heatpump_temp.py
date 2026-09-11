"""Heat-pump detection from temperature response - needs the CKW weather join."""

import numpy as np
from .base import Detector

BASE_TEMP_C = 15.0          # Swiss convention for the heating threshold


class HPTempSlope(Detector):
    """Regress overnight consumption on heating degree hours.

    Electric heating makes consumption a function of outdoor temperature. The
    slope is physically interpretable - roughly the building's heat loss divided
    by the system's COP - so a heat pump and a resistance heater differ by a
    factor of about three for the same house, and a household with neither shows
    almost no slope at all.

    Overnight only (21:00-05:00): no PV contributes, so the signal is load rather
    than net generation. This is the feature AEW's data could not support, having
    no weather to join.
    """
    name, asset = "hp_temp_slope", "hp"
    night_based = True
    description = "slope of overnight consumption against heating degree hours"

    def score(self, mp_id, df):
        d = df[~df.dst_day]
        d = d[(d.hour >= 21) | (d.hour < 5)]
        if d.temp_c.isna().all():
            return float("nan"), {"reason": "no weather joined - pass --weather"}
        g = d.groupby("d").agg(kwh=("import_kwh", "sum"), temp=("temp_c", "mean"))
        g = g.dropna()
        if len(g) < 180:
            return float("nan"), {"reason": "fewer than 180 usable days"}
        hdd = np.clip(BASE_TEMP_C - g.temp.to_numpy(), 0, None)
        if hdd.std() == 0:
            return float("nan"), {}
        slope, intercept = np.polyfit(hdd, g.kwh.to_numpy(), 1)
        r = float(np.corrcoef(hdd, g.kwh.to_numpy())[0, 1])
        return float(slope), {"kwh_per_degree_hour": round(float(slope), 3),
                              "base_load_kwh": round(float(intercept), 2),
                              "fit_r": round(r, 3), "days": int(len(g))}


class ElectricHeatingTempSlope(HPTempSlope):
    """Same feature against the honest target: does this household heat with
    electricity at all? Heat pump vs resistance is a COP question, and the slope
    separates those two by magnitude rather than by presence."""
    name, asset = "eh_temp_slope", "electric_heating"
