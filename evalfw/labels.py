"""
Label tiers for asset detection.

Ground truth quality differs per asset, and pretending otherwise is the main way
an evaluation lies to you:

  gold_positive   measurement-verified present   (PV: sustained export)
  gold_negative   measurement-verified absent    (PV: never exports over >=1y)
  weak_positive   register flag, asset live, meter unambiguously attributable
  unlabelled      register dash - NOT a negative. The AEW register records
                  SUBSIDISED installations, so a dash means "no subsidy record".
                  Measured on 2023-09, 13 of 19 meters flagged "no PV" were
                  exporting. Scoring those as negatives invents false positives.

Counts on training_set_v2 (>=1y of data, one meter per partner):
  PV   206 gold positive / 75 gold negative      <- real ground truth
  HP    68 weak positive / 140 unlabelled
  EV    48 weak positive / 160 unlabelled
"""

import duckdb
import pandas as pd

ASSETS = ["pv", "hp", "ev", "battery"]

# PV export thresholds, kWh over the meter's whole history
PV_POS_KWH = 100.0     # unambiguously producing
PV_NEG_KWH = 1.0       # unambiguously not


def meter_summary(ts_path, min_days=365, unambiguous_only=True):
    """One row per meter: coverage, register flags, and crude energy totals."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT mp_id,
               any_value(gp)              AS gp,
               any_value(n_meters_for_gp) AS n_meters_for_gp,
               count(DISTINCT d)          AS days,
               min(d) AS first_day, max(d) AS last_day,
               any_value(has_pv) AS has_pv, any_value(has_hp) AS has_hp,
               any_value(has_ev) AS has_ev, any_value(has_battery) AS has_battery,
               any_value(pv_live) AS pv_live, any_value(hp_live) AS hp_live,
               any_value(ev_live) AS ev_live, any_value(battery_live) AS battery_live,
               sum(export_kwh) AS export_kwh_total,
               sum(import_kwh) AS import_kwh_total
        FROM read_parquet('{ts_path}')
        GROUP BY mp_id
    """).df()
    df = df[df.days >= min_days]
    if unambiguous_only:
        # a partner owning several meters tells us the HOUSEHOLD has the asset,
        # not which meter carries it - unusable as a per-meter label
        df = df[df.n_meters_for_gp == 1]
    return df.reset_index(drop=True)


def build(ts_path, asset, min_days=365, unambiguous_only=True):
    """Return mp_id, gp, y (1/0/NaN), tier for one asset.

    y is NaN for unlabelled meters. Metrics that need negatives must skip them;
    the flag rate over them is reported separately as an upper bound on the
    false-positive rate.
    """
    if asset not in ASSETS:
        raise ValueError(f"unknown asset {asset!r}")
    s = meter_summary(ts_path, min_days, unambiguous_only)

    if asset == "pv":
        # measured export beats the register outright: it is objective, it covers
        # every meter, and it dates commissioning to within days
        tier = pd.Series("ambiguous", index=s.index)
        tier[s.export_kwh_total >= PV_POS_KWH] = "gold_positive"
        tier[s.export_kwh_total < PV_NEG_KWH] = "gold_negative"
        y = pd.Series(float("nan"), index=s.index)
        y[tier == "gold_positive"] = 1.0
        y[tier == "gold_negative"] = 0.0
    else:
        flag, live = s[f"has_{asset}"].fillna(False), s[f"{asset}_live"].fillna(False)
        tier = pd.Series("unlabelled", index=s.index)
        tier[flag & live] = "weak_positive"
        tier[flag & ~live] = "ambiguous"      # registered but not yet commissioned
        y = pd.Series(float("nan"), index=s.index)
        y[tier == "weak_positive"] = 1.0
        if asset == "battery":
            # Battery is the one register flag whose negatives survive an audit.
            # Among PV-owning meters flagged "no battery", 29 of 40 show no trace
            # of the physical signature (near-zero net import through dark summer
            # hours), 4 are ambiguous and 7 look like undeclared batteries - about
            # 18% contamination, against near-total for heat pumps and EVs. Usable,
            # so measured precision is a LOWER BOUND: some false positives are
            # probably real batteries with no subsidy record.
            tier[~flag] = "silver_negative"
            y[~flag] = 0.0
        # otherwise deliberately NOT setting y=0 for unlabelled

    out = pd.DataFrame({"mp_id": s.mp_id, "gp": s.gp, "y": y, "tier": tier})
    return out.reset_index(drop=True)


def describe(labels, asset):
    c = labels.tier.value_counts().to_dict()
    parts = [f"{k}={v}" for k, v in sorted(c.items())]
    has_neg = (labels.y == 0).sum() > 0
    if asset == "battery" and has_neg:
        note = "  [silver negatives, ~18% contaminated - precision is a LOWER bound]"
    elif has_neg:
        note = "  [gold negatives available]"
    else:
        note = "  [NO trustworthy negatives - precision is not computable]"
    return f"{asset}: " + ", ".join(parts) + note
