"""Heat-pump detection from the seasonal signature."""

import numpy as np
from .base import Detector

WINTER, SUMMER = (12, 1, 2), (6, 7, 8)


class HPWinterRatio(Detector):
    """Winter-to-summer ratio, measured at NIGHT only.

    Space heating is the one asset with an unmistakable annual signature, which
    is why a single summer month could never have found it.

    Night-only is not a detail. Measured on the whole day this feature reads ~10
    for heat-pump and non-heat-pump meters alike, because PV self-consumption
    drives summer NET IMPORT towards zero and the ratio explodes. In a register
    that is 83% PV owners, the daytime version measures PV, not heating. Between
    22:00 and 06:00 there is no generation, so the ratio reflects load.

    Caveat worth stating in any write-up: resistance heating looks the same. This
    separates electric heating from none, not heat pump from direct electric.
    """
    name, asset = "hp_winter_ratio", "hp"
    night_based = True
    description = "winter/summer overnight import ratio (PV cannot contaminate 22:00-06:00)"

    def score(self, mp_id, df):
        d = df[~df.dst_day]
        night = d[(d.hour >= 22) | (d.hour < 6)]
        w = night[night.ts.dt.month.isin(WINTER)]
        s = night[night.ts.dt.month.isin(SUMMER)]
        if len(w) < 32 * 20 or len(s) < 32 * 20:
            return float("nan"), {"reason": "needs a full winter and summer"}
        wd = w.groupby("d").import_kwh.sum().mean()
        sd = s.groupby("d").import_kwh.sum().mean()
        if not sd or np.isnan(sd) or sd <= 0:
            return float("nan"), {}
        return float(wd / sd), {"winter_night_kwh": round(float(wd), 2),
                                "summer_night_kwh": round(float(sd), 2)}


class HPNightBlocks(Detector):
    """Heat pumps run overnight on cheap tariff and cycle. Scores the winter
    night-time share of consumption relative to summer."""
    name, asset = "hp_night_share", "hp"
    night_based = True
    description = "winter overnight consumption share, minus the summer baseline"

    def score(self, mp_id, df):
        d = df[~df.dst_day]
        w = d[d.ts.dt.month.isin(WINTER)]
        s = d[d.ts.dt.month.isin(SUMMER)]
        if len(w) < 96 * 30 or len(s) < 96 * 30:
            return float("nan"), {"reason": "needs a full winter and summer"}
        def night_share(x):
            tot = x.import_kwh.sum()
            return float(x[x.hour < 6].import_kwh.sum() / tot) if tot > 0 else np.nan
        nw, ns = night_share(w), night_share(s)
        if np.isnan(nw) or np.isnan(ns):
            return float("nan"), {}
        return float(nw - ns), {"winter_night_share": round(nw, 3),
                                "summer_night_share": round(ns, 3)}
