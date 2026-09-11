"""
Electric water-heater (Elektroboiler) detection.

A resistance boiler is the weakest of the assets here, and it is worth
understanding why before reading any score. Expectation says a block of a couple
of hours on the night tariff, at 3-6 kW: Swiss 300 L Elektroboiler are sold at
3.0, 4.0 and 6.0 kW, the larger ratings on 400 V three-phase, heating a full tank
in 3 to 6 hours. That range overlaps the EV wallbox band almost entirely, which is
why the EV detector needs its start-time irregularity term to tell them apart -
power alone cannot. What the data shows, on CKW households
with declared hot-water type and no PV to mask the load:

    boiler households consume ~2x more than non-electric ones at 22:00-02:00
    ...and the same as them at 15:00-17:00

so the tariff-window contrast is real in AGGREGATE. But only 2.3% of night slots
exceed 1.8 kW, where a genuine 2-hour nightly block would give roughly 28%. The
population-level difference is a ~0.35 kW shift in the mean, not a rectangle, and
within-household variance swamps it: every single-feature cut tops out near 0.63.

Combining features reaches ~0.73, so the signal is real but diffuse - exactly the
case where a model earns its place, and the opposite of battery detection where
one physical feature reaches 0.94.

BATTERY MASKING. Everything here is measured in dark hours, which is precisely
where a home battery holds net import at zero. Transferring this rule to AEW
meters gives an implied boiler rate of 48.5% among meters without a battery and
29.8% among those with one - on a population that is 68% battery. The bias is
downward and grows as storage spreads, so a boiler rate quoted over a mixed
population is an underestimate of unknown size. Detect batteries first (AUC 0.940,
the easiest asset here), then either exclude those meters or report separately.
"""

import numpy as np
from .base import Detector
from ._blocks import find_blocks, block_features

SUMMER = (6, 7, 8)          # no space heating to confound the night load
BOILER_KWH = 0.45           # 1.8 kW in a 15-minute slot


class BoilerNightBand(Detector):
    """Share of summer-night slots drawing boiler-sized power."""
    name, asset = "boiler_night_band", "boiler"
    night_based = True
    description = "share of summer-night slots above 1.8 kW"

    def score(self, mp_id, df):
        d = df[~df.dst_day]
        d = d[d.ts.dt.month.isin(SUMMER) & ((d.hour < 6) | (d.hour >= 23))]
        if len(d) < 500:
            return float("nan"), {"reason": "insufficient summer night data"}
        f = float((d.import_kwh.fillna(0) > BOILER_KWH).mean())
        return f, {"night_frac_over_1_8kw": round(f, 4)}


class BoilerTariffContrast(Detector):
    """Night-tariff window against the late afternoon.

    Ripple-controlled boilers run on the cheap tariff, so consumption tips towards
    22:00-02:00 relative to 15:00-17:00. Scale-free, so house size cancels.
    """
    name, asset = "boiler_tariff_contrast", "boiler"
    night_based = True
    description = "summer 22:00-02:00 load relative to 15:00-17:00"

    def score(self, mp_id, df):
        d = df[~df.dst_day]
        d = d[d.ts.dt.month.isin(SUMMER)]
        if len(d) < 2000:
            return float("nan"), {"reason": "insufficient summer data"}
        night = d[(d.hour >= 22) | (d.hour < 2)].import_kwh.mean()
        aft = d[d.hour.between(15, 17)].import_kwh.mean()
        if not aft or np.isnan(aft) or aft <= 0:
            return float("nan"), {}
        return float(night / aft), {"tariff_window_kwh": round(float(night), 4),
                                    "afternoon_kwh": round(float(aft), 4)}
