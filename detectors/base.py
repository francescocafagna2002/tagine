"""
The detector contract.

Every approach - hand-written rule, fitted classifier, anything - implements the
same call, so the leaderboard compares like with like:

    score(mp_id, df) -> (float, evidence dict)

`df` is one meter's history, sorted by ts, with columns:
    ts (timestamp)  d (date)  hour (int)  import_kwh  export_kwh  dst_day (bool)

`score` must be CONTINUOUS and higher-means-more-likely. Threshold-free metrics
(ROC-AUC, average precision) need the ordering; the runner picks cutoffs itself
on training folds, so detectors must not hard-code one.

`evidence` is a small dict naming WHY - the days and hours that triggered it.
The challenge grades on explainability, and it makes two detectors disagreeing
about a meter debuggable instead of mysterious.
"""


class Detector:
    name = "unnamed"
    asset = "pv"          # pv | hp | ev | battery | boiler | electric_heating
    description = ""

    # Set True when the signal lives in dark hours. A home battery holds net
    # import flat at zero after sunset, so it does not merely add noise to such a
    # detector - it erases the evidence. Measured: applying the boiler rule to AEW
    # meters gives an implied rate of 48.5% without a battery and 29.8% with one,
    # on a population that is 68% battery. The runner reports these detectors
    # stratified by battery ownership wherever that label exists.
    night_based = False

    def score(self, mp_id, df):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name} asset={self.asset}>"


class TrainableDetector(Detector):
    """A detector that LEARNS parameters from labelled meters.

    Split into featurize / fit / predict rather than a single score(), because a
    model must never be fitted on the meters it is scored on. The runner computes
    features once per meter (deterministic, so no leakage), then for each
    training fold calls fit() and scores only the held-out fold. The leaderboard
    therefore compares out-of-fold model scores against rule scores computed on
    the same meters - which is the only way the two are comparable at all.

    Implement featurize / fit / predict; score() is not used.
    """
    trainable = True

    def featurize(self, mp_id, df):
        """-> {feature_name: float}. Must not depend on labels."""
        raise NotImplementedError

    def fit(self, X, y):
        """X: (n, d) float array in self.feature_names order. y: 0/1."""
        raise NotImplementedError

    def predict(self, X):
        """-> continuous score, higher means more likely."""
        raise NotImplementedError

    def score(self, mp_id, df):
        raise RuntimeError(f"{self.name} is trainable - the runner calls "
                           "featurize/fit/predict, not score()")
