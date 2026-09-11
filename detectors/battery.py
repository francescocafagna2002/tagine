"""
Home-battery detection.

The clearest signature in either dataset. A battery charged from the day's PV
carries the household through the evening, so net import sits at essentially
ZERO after dark - not merely low, but flat at zero for hours, which no ordinary
household reproduces. Measured on AEW meters that own PV, in a fully dark summer
window (22:00-24:00, June-July):

    battery owners      92.4% of slots below 0.02 kWh
    non-owners           0.0%
    AUC 0.917

Grid-charging is NOT worth a second detector. A battery charged from the cheap
night tariff draws a large flat block instead of holding import at zero - the
inverse signature - so it seemed like a gap worth filling. Measured on AEW's
labelled meters: tariff-window charge blocks reach AUC 0.613 alone, and combining
them with the discharge signature moves 0.935 to 0.943, inside the noise on 116
meters. Even among meters with no PV the discharge test wins (0.779 against
0.562). Individual meters may well arbitrage the tariff; the population does not
need a detector for it.

The window matters. An evening window starting at 18:00 is contaminated: Swiss
midsummer sunsets are near 21:30, so PV alone suppresses import and any PV owner
looks like a battery owner. Restricting to full darkness removes generation from
the picture entirely.
"""

import numpy as np
from .base import Detector

DARK_HOURS = (22, 23)            # fully dark in Swiss summer
SUMMER, WINTER = (6, 7), (12, 1)
ZERO_KWH = 0.02                  # a 15-min slot this small is not a real load


def _zero_frac(d, months):
    s = d[d.ts.dt.month.isin(months) & d.hour.isin(DARK_HOURS)]
    if len(s) < 100:
        return float("nan")
    return float((s.import_kwh.fillna(0) < ZERO_KWH).mean())


class BatteryDarkZero(Detector):
    """Fraction of dark summer slots at near-zero import."""
    name, asset = "battery_dark_zero", "battery"
    description = "share of dark summer slots with net import below 0.02 kWh"

    def score(self, mp_id, df):
        d = df[~df.dst_day]
        z = _zero_frac(d, SUMMER)
        if np.isnan(z):
            return float("nan"), {"reason": "insufficient summer night data"}
        return z, {"summer_dark_zero_frac": round(z, 3)}


class BatterySeasonalGap(Detector):
    """Summer minus winter dark-hour zero-fraction.

    A battery is full on summer nights and empty on winter ones, so the gap is
    large. Subtracting winter also removes households that simply have almost no
    load at night - they read zero in both seasons and score nothing here, where
    the raw summer fraction would flag them.
    """
    name, asset = "battery_seasonal_gap", "battery"
    description = "summer minus winter dark-hour zero-fraction"

    def score(self, mp_id, df):
        d = df[~df.dst_day]
        s, w = _zero_frac(d, SUMMER), _zero_frac(d, WINTER)
        if np.isnan(s) or np.isnan(w):
            return float("nan"), {"reason": "needs a full summer and winter"}
        return s - w, {"summer_zero_frac": round(s, 3), "winter_zero_frac": round(w, 3)}
