"""EV charging detection from sustained rectangular draw."""

import numpy as np
import pandas as pd
from .base import Detector
from ._blocks import find_blocks, block_features

# kWh per 15 min for single/three-phase wallboxes: 3.7 / 7.4 / 11 kW
BAND_LO, BAND_HI = 0.85, 3.20          # ~3.4 kW to ~12.8 kW
MIN_RUN = 3                             # >=45 min of steady draw


def _runs_above(v, lo):
    """(start, length) of each consecutive stretch at or above `lo`."""
    m = (v >= lo).astype(np.int8)
    if m.sum() == 0:
        return []
    edges = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), (ends - starts).tolist()))


class EVSustainedBlocks(Detector):
    """Sustained wallbox-band blocks, weighted by how IRREGULARLY they start.

    Counting blocks alone confuses a car with a timer. A ripple-controlled boiler
    produces a sustained flat block in an overlapping power range every single
    night at the same minute; scoring on frequency alone, such a meter looks like
    someone charging a car daily. Measured on CKW:

        blocks per day alone                       AUC 0.677
        x single-window irregularity               AUC 0.706
        x top-2-window irregularity, squared       AUC 0.712

    Irregularity is the discriminating term, not a refinement: start-time
    concentration on its own runs AUC 0.311 - inverted - because clock-driven
    loads are exactly what a frequency count picks up by mistake.
    """
    name, asset = "ev_sustained_blocks", "ev"
    description = "wallbox-band blocks weighted by start-time irregularity"

    def score(self, mp_id, df):
        d = df[~df.dst_day]
        if d.empty:
            return float("nan"), {}
        n_days = int(d.d.nunique())
        bl = find_blocks(d.ts.to_numpy(), d.import_kwh.to_numpy(),
                         BAND_LO, BAND_HI, MIN_RUN)
        f = block_features(bl, n_days)
        if f["n"] < 3:
            return 0.0, {"blocks": f["n"], "days": n_days,
                         "note": "too few blocks to judge regularity"}
        score = f["freq"] * (1 - f["conc"]) ** 2 * (1 + f["durcv"])
        ex = [{"slot": b[0], "at": f"{b[0]//4:02d}:{15*(b[0]%4):02d}",
               "slots": b[1], "kW": round(b[2] * 4, 1)} for b in bl[:4]]
        return float(score), {
            "blocks": f["n"], "days": n_days,
            "blocks_per_day": round(f["freq"], 3),
            "start_concentration": round(f["conc"], 2),
            "reading": ("clock-driven — looks like a timer, not a car"
                        if f["conc"] > 0.55 else "irregular starts — consistent with a car"),
            "examples": ex}


class EVPeakBand(Detector):
    """Cruder alternative: does the meter's peak sit in a wallbox band at all?
    Included so the leaderboard has something to beat on EV besides random."""
    name, asset = "ev_peak_band", "ev"
    description = "closeness of the observed peak to 3.7 / 7.4 / 11 kW"
    # KEPT AS A LESSON, NOT A DETECTOR. Measured AUC 0.347 on CKW - worse than
    # chance, and reliably so. It rewards peaks NEAR a wallbox rating, but EV
    # households simply peak HIGHER (median 13.9 kW vs 8.9 kW), so proximity to a
    # band penalises exactly the households it should flag. Do not read its score
    # as evidence; per-meter reports mark it accordingly.
    anti_predictive = True

    def score(self, mp_id, df):
        peak_kw = float(df.import_kwh.fillna(0).max()) * 4
        if peak_kw <= 0:
            return float("nan"), {}
        near = min(abs(peak_kw - r) / r for r in (3.7, 7.4, 11.0))
        return float(max(0.0, 1.0 - near)), {"peak_kw": round(peak_kw, 1)}
