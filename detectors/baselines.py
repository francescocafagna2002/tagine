"""Baselines. Two of these exist to validate the harness, not to win."""

import numpy as np
from .base import Detector

WINTER, SUMMER = (12, 1, 2), (6, 7, 8)


class RandomScore(Detector):
    """Sanity check: must land at AUC ~0.5. If it doesn't, the harness is
    misaligning scores and labels and every other number is untrustworthy."""
    description = "uniform random score, seeded per meter"

    def __init__(self, asset):
        self.asset = asset
        self.name = f"random_{asset}"

    def score(self, mp_id, df):
        # hash rather than int(): CKW meter ids are hex strings, not integers
        seed = abs(hash(str(mp_id))) % (2 ** 32)
        return float(np.random.default_rng(seed).random()), {}


class PVExportTotal(Detector):
    """The floor for PV: total energy exported. Any cleverer PV approach that
    fails to beat this is not worth presenting."""
    name, asset = "pv_export_total", "pv"
    description = "sum of export_kwh over the meter's history"
    # The PV gold labels are themselves derived from export, so scoring this
    # against them is tautological - it will read 1.000 and mean nothing. Its
    # real job is to MANUFACTURE labels for the fleet, not to compete.
    circular = True

    def score(self, mp_id, df):
        tot = float(df.export_kwh.fillna(0).sum())
        peak = float(df.export_kwh.fillna(0).max())
        return tot, {"export_kwh_total": round(tot, 1), "peak_kwh_15min": round(peak, 3)}


class PVMiddayDip(Detector):
    """PV without relying on export at all - catches self-consumption-only systems
    that never feed in, which the export baseline is blind to by construction.

    Behind-the-meter generation suppresses midday IMPORT relative to the morning
    and evening shoulders, so the ratio drops below what the same household shows
    in winter.
    """
    name, asset = "pv_midday_dip", "pv"
    description = "summer midday import suppression vs morning/evening shoulders"

    def score(self, mp_id, df):
        d = df[~df.dst_day]
        summer = d[d.ts.dt.month.isin(SUMMER)]
        if len(summer) < 96 * 30:
            return float("nan"), {"reason": "insufficient summer data"}
        midday = summer[summer.hour.between(10, 14)].import_kwh.mean()
        shoulder = summer[summer.hour.isin([6, 7, 8, 17, 18, 19])].import_kwh.mean()
        if not shoulder or np.isnan(shoulder) or shoulder <= 0:
            return float("nan"), {"reason": "no shoulder load"}
        ratio = midday / shoulder
        return float(1.0 - ratio), {"midday_kwh": round(float(midday), 4),
                                    "shoulder_kwh": round(float(shoulder), 4),
                                    "ratio": round(float(ratio), 3)}
