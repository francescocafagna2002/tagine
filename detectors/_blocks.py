"""
Shared block extraction: sustained, flat runs of power, and how regular they are.

The key measurement separating an EV from a timed load is not power or duration -
those overlap - but WHEN the block starts. A ripple-controlled boiler starts at the
same clock time every night; a car is plugged in whenever the driver gets home.
Measured on CKW households: start-time concentration has AUC 0.311 for EV, i.e.
inverted - irregularity is the signal, at 0.689 on its own.
"""

import numpy as np

QUARTER = np.timedelta64(15, "m")


def find_blocks(ts, v, lo, hi, min_run=3, flat=0.30):
    """Sustained flat runs within a power band -> [(start_slot, length, mean_kwh)].

    Runs are cut at gaps in the series, so a missing day cannot glue two evenings
    into one implausible block. Runs MAY cross midnight, which matters: EV charging
    frequently does, and forcing per-day windows would chop those in half.
    """
    ts = np.asarray(ts, dtype="datetime64[ns]")
    v = np.nan_to_num(np.asarray(v, float))
    if len(v) < min_run:
        return []
    contiguous = np.r_[True, (np.diff(ts) == QUARTER)]
    above = (v >= lo) & contiguous
    # a discontinuity also terminates whatever run was open
    m = above.astype(np.int8)
    m[~contiguous] = 0
    edges = np.diff(np.concatenate([[0], m, [0]]))
    out = []
    slot = ((ts - ts.astype("datetime64[D]")).astype("timedelta64[m]").astype(int) // 15)
    for st, en in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        n = en - st
        if n < min_run:
            continue
        seg = v[st:en]
        mu = float(seg.mean())
        if not (lo <= mu <= hi):
            continue
        if seg.std() > flat * mu:          # a wallbox holds steady; cooking does not
            continue
        out.append((int(slot[st]), int(n), mu))
    return out


def _conc_topk(starts, n, k=2):
    """Share of blocks starting inside the k best non-overlapping 75-min windows.

    Top-TWO, not top-one, because households commonly run two timed loads - a
    night-tariff boiler plus an afternoon boost, say. A single-window measure
    reads such a meter as only ~0.5 concentrated and lets it pass as a car.
    Measured on CKW, top-2 separates far more cleanly than top-1:
    median 0.36 for EV households against 0.59 for the rest.
    """
    hist = np.bincount(starts, minlength=96).astype(float)
    win = np.convolve(np.r_[hist, hist[:4]], np.ones(5), "valid")[:96]
    total, taken = 0.0, np.zeros(96, bool)
    for _ in range(k):
        order = np.argsort(-win)
        pick = next((i for i in order if not taken[i] and win[i] > 0), None)
        if pick is None:
            break
        total += win[pick]
        for o in range(-4, 5):
            taken[(pick + o) % 96] = True
        win[pick] = 0
    return float(total / n)


def block_features(blocks, n_days, k=2):
    """freq, start-time concentration (top-k windows), duration variability."""
    if len(blocks) < 3 or n_days <= 0:
        return {"freq": 0.0, "conc": 1.0, "durcv": 0.0, "n": len(blocks),
                "modal_slot": None}
    starts = np.array([b[0] for b in blocks])
    durs = np.array([b[1] for b in blocks], float)
    hist = np.bincount(starts, minlength=96).astype(float)
    win = np.convolve(np.r_[hist, hist[:4]], np.ones(5), "valid")[:96]
    return {"freq": len(blocks) / n_days,
            "conc": _conc_topk(starts, len(blocks), k),
            "durcv": float(durs.std() / durs.mean()) if durs.mean() else 0.0,
            "n": len(blocks),
            "modal_slot": int(np.argmax(win))}
