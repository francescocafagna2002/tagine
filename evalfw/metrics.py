"""
Metrics with uncertainty, sized for very small label sets.

With 48 EV positives a recall of 0.75 carries a 95% CI of about +-0.12. Two
detectors less than ~12 points apart are statistically indistinguishable, so a
leaderboard without intervals invites the team to tune noise. Every headline
number here therefore comes with a bootstrap CI, and detector-vs-detector
comparison uses a PAIRED bootstrap on the same meters, which has far more power
than comparing two independent intervals.

No sklearn dependency - AUC and average precision are a few lines each and the
Renku image may not have it.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260911)


# ---------------------------------------------------------------- base metrics

def roc_auc(y, s):
    """Mann-Whitney U form, tie-aware."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~(np.isnan(y) | np.isnan(s))
    y, s = y[m], s[m]
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def average_precision(y, s):
    """Area under precision-recall, the step-wise (non-interpolated) estimate."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~(np.isnan(y) | np.isnan(s))
    y, s = y[m], s[m]
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y == 1)
    prec = tp / np.arange(1, len(y) + 1)
    return float(prec[y == 1].sum() / (y == 1).sum())


def recall_at(y, s, thr):
    y, s = np.asarray(y, float), np.asarray(s, float)
    pos = (y == 1) & ~np.isnan(s)
    return float((s[pos] >= thr).mean()) if pos.sum() else float("nan")


def precision_at(y, s, thr):
    """Only meaningful where gold negatives exist - callers must check."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~(np.isnan(y) | np.isnan(s))
    y, s = y[m], s[m]
    flagged = s >= thr
    return float((y[flagged] == 1).mean()) if flagged.sum() else float("nan")


def f1_at(y, s, thr):
    p, r = precision_at(y, s, thr), recall_at(y, s, thr)
    return float("nan") if (np.isnan(p) or np.isnan(r) or p + r == 0) else 2 * p * r / (p + r)


def lift_at(y, s, thr):
    """recall / flag-rate-on-unlabelled.

    The honest headline where negatives are untrustworthy. A detector that fires
    at random scores 1.0 by construction, so lift says how much better than
    chance the ranking is WITHOUT ever claiming a precision it cannot support.
    """
    y, s = np.asarray(y, float), np.asarray(s, float)
    pos, unl = (y == 1) & ~np.isnan(s), np.isnan(y) & ~np.isnan(s)
    if pos.sum() == 0 or unl.sum() == 0:
        return float("nan")
    r, f = (s[pos] >= thr).mean(), (s[unl] >= thr).mean()
    return float(r / f) if f > 0 else float("nan")


def threshold_for_prevalence(scores_unlabelled, prior):
    """Cutoff that flags `prior` of the unlabelled pool.

    Without negatives an F1-style search collapses to flagging everything. Pinning
    the operating point to an assumed prevalence instead makes recall comparable
    across detectors and interpretable: "at a threshold calibrated to 20% of
    meters having a heat pump, this recovers X% of the known ones."
    """
    s = np.asarray(scores_unlabelled, float)
    s = s[~np.isnan(s)]
    if len(s) == 0 or not (0 < prior < 1):
        return float("nan")
    return float(np.quantile(s, 1.0 - prior))


def flag_rate(scores, thr):
    """Share of a pool flagged. Over the unlabelled pool this is an upper bound
    on the false-positive rate AND an estimate of true prevalence among them."""
    s = np.asarray(scores, float)
    s = s[~np.isnan(s)]
    return float((s >= thr).mean()) if len(s) else float("nan")


# ---------------------------------------------------------------- thresholds

def youden_j(y, s, thr):
    """TPR - FPR. Balanced, so it does not collapse when one class dominates."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~(np.isnan(y) | np.isnan(s))
    y, s = y[m], s[m]
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    tpr = (s[y == 1] >= thr).mean()
    fpr = (s[y == 0] >= thr).mean()
    return float(tpr - fpr)


def best_threshold(y, s, objective="youden"):
    """Pick a cutoff on TRAINING data only. Candidates are the observed scores.

    Youden's J by default, not F1. These gold sets are heavily one-sided - 78% of
    CKW's heating labels are positive - and maximising F1 there just flags every
    meter, reporting precision equal to the base rate and recall of 1.000, which
    says nothing about the detector. J stays informative at any class balance.
    """
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~(np.isnan(y) | np.isnan(s))
    yv, sv = y[m], s[m]
    if len(sv) == 0:
        return float("nan")
    cands = np.unique(sv)
    if len(cands) > 200:
        cands = np.quantile(sv, np.linspace(0, 1, 200))
    if (yv == 0).sum() == 0:
        # no negatives: F1 is undefined, so target a fixed recall instead of
        # letting the threshold collapse to "flag everything"
        pos = sv[yv == 1]
        return float(np.quantile(pos, 0.20)) if len(pos) else float("nan")
    fn = youden_j if objective == "youden" else f1_at
    scores = [fn(yv, sv, t) for t in cands]
    best = int(np.nanargmax(scores)) if not np.all(np.isnan(scores)) else 0
    return float(cands[best])


def grouped_folds(groups, k=5, seed=0):
    """K folds split on `groups` (Geschäftspartner), so one partner's meters
    never straddle train and test."""
    g = pd.Series(groups).astype(str).to_numpy()
    uniq = pd.unique(g)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    assign = {u: i % k for i, u in enumerate(uniq)}
    fold = np.array([assign[x] for x in g])
    return [(np.where(fold != i)[0], np.where(fold == i)[0]) for i in range(k)]


# ---------------------------------------------------------------- uncertainty

def bootstrap_ci(fn, *arrays, n=2000, alpha=0.05):
    """Resample METERS with replacement; the meter is the independent unit."""
    arrays = [np.asarray(a) for a in arrays]
    n_obs = len(arrays[0])
    point = fn(*arrays)
    if n_obs == 0:
        return point, float("nan"), float("nan")
    vals = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, n_obs, n_obs)
        try:
            vals[i] = fn(*[a[idx] for a in arrays])
        except Exception:
            vals[i] = np.nan
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def paired_bootstrap(fn, y, s_a, s_b, n=2000, alpha=0.05):
    """Is B better than A on the SAME meters? Returns (diff, lo, hi, p_two_sided).

    Paired resampling removes between-meter variance, which is why it can resolve
    differences an unpaired comparison of two CIs never could.
    """
    y, s_a, s_b = np.asarray(y, float), np.asarray(s_a, float), np.asarray(s_b, float)
    n_obs = len(y)
    diff = fn(y, s_b) - fn(y, s_a)
    vals = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, n_obs, n_obs)
        try:
            vals[i] = fn(y[idx], s_b[idx]) - fn(y[idx], s_a[idx])
        except Exception:
            vals[i] = np.nan
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(diff), float(lo), float(hi), float(min(p, 1.0))


def fmt(point, lo, hi, digits=3):
    if np.isnan(point):
        return "    n/a      "
    return f"{point:.{digits}f} [{lo:.{digits}f},{hi:.{digits}f}]"
