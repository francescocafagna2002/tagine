"""
Gradient-boosted detector over shape features.

Deliberately SHAPE-ONLY - ratios, shares and power-band occupancy, nothing that
encodes how much electricity a household uses or whether it owns solar. Absolute
level features score slightly higher in-sample (heat pump AUC 0.859 vs 0.822) but
they encode house size and consumption scale, which differ between the labelled
survey sample and the fleet a detector is deployed on. The 0.04 is worth paying
for transfer.

Measured out-of-fold on CKW, against the best single-feature rule:
    EV          rule 0.753   ->  ~0.80
    heat pump   rule 0.658   ->  ~0.82   (the temperature-slope RULE gets 0.830)

That last comparison is the honest one: for heat pumps a single well-chosen
physical feature beats a model over a dozen weak ones. ML earns its place where
the signature is a conjunction of weak cues - EV charging is, space heating is
not.
"""

import numpy as np
from .base import TrainableDetector

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_SKLEARN = True
except ImportError:                       # the runner skips registration
    HAVE_SKLEARN = False

BASE_TEMP_C = 15.0
WINTER, SUMMER = (12, 1, 2), (6, 7, 8)


class GBMShape(TrainableDetector):
    description = "gradient boosting over scale-free shape features"

    def __init__(self, asset):
        self.asset = asset
        self.name = f"gbm_{asset}"
        self.model = None
        self.feature_names = []

    # ---------------------------------------------------------------- features
    def featurize(self, mp_id, df):
        d = df[~df.dst_day]
        imp = d.import_kwh.fillna(0)
        tot = float(imp.sum())
        if tot <= 0:
            return {}
        h, mon = d.hour, d.ts.dt.month
        win, sm = mon.isin(WINTER), mon.isin(SUMMER)
        night = (h >= 21) | (h < 5)
        wk = d.ts.dt.dayofweek >= 5

        def safe(a, b):
            return float(a / b) if b and not np.isnan(b) and b > 0 else np.nan

        f = {
            "share_night":   safe(imp[h < 6].sum(), tot),
            "share_midday":  safe(imp[h.between(10, 14)].sum(), tot),
            "share_evening": safe(imp[h.between(18, 23)].sum(), tot),
            "seas_ratio":       safe(imp[win].sum(), imp[sm].sum()),
            "seas_night_ratio": safe(imp[win & night].sum(), imp[sm & night].sum()),
            "weekend_ratio":    safe(imp[wk].mean(), imp[~wk].mean()),
            # occupancy of single/three-phase wallbox power bands
            "band_37": safe((imp.between(0.85, 1.10)).sum(), len(imp)),
            "band_74": safe((imp.between(1.70, 2.05)).sum(), len(imp)),
            "band_11": safe((imp.between(2.55, 3.05)).sum(), len(imp)),
            "frac_over_6kw":  safe((imp > 1.5).sum(), len(imp)),
            "frac_over_10kw": safe((imp > 2.5).sum(), len(imp)),
            # scale-free peakiness: how spiky, not how big
            "peak_ratio": safe(imp.quantile(0.999), imp.quantile(0.5)),
            "cv":         safe(imp.std(), imp.mean()),
            # night-tariff occupancy, for resistance water heating
            "night_over_1kw":   safe((imp[night] > 0.25).sum(), max(night.sum(), 1)),
            "night_over_1_8kw": safe((imp[night] > 0.45).sum(), max(night.sum(), 1)),
            "tariff_contrast":  safe(imp[(h >= 22) | (h < 2)].mean(),
                                     imp[h.between(15, 17)].mean()),
        }
        # temperature response, normalised so it does not smuggle in house size
        f["hdd_slope_norm"] = np.nan
        if "temp_c" in d.columns and not d.temp_c.isna().all():
            n = d[night]
            g = n.groupby("d").agg(kwh=("import_kwh", "sum"), t=("temp_c", "mean")).dropna()
            if len(g) >= 180:
                hdd = np.clip(BASE_TEMP_C - g.t.to_numpy(), 0, None)
                if hdd.std() > 0 and g.kwh.mean() > 0:
                    slope = float(np.polyfit(hdd, g.kwh.to_numpy(), 1)[0])
                    f["hdd_slope_norm"] = slope / float(g.kwh.mean())
        return f

    # ---------------------------------------------------------------- model
    def fit(self, X, y):
        self.model = HistGradientBoostingClassifier(
            max_iter=250, max_depth=3, learning_rate=0.06,
            l2_regularization=1.0, random_state=0).fit(X, y)

    def predict(self, X):
        if self.model is None:
            return np.full(len(X), np.nan)
        return self.model.predict_proba(X)[:, 1]
