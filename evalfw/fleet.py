#!/usr/bin/env python3
"""
Fleet-wide prevalence check.

    python3 -m evalfw.fleet --raw '~/work/store/data/prepared/raw/*/*.parquet' \
        --sample 2000 --leaderboard results/leaderboard.csv --out results

Runs each detector at its leaderboard threshold over a RANDOM SAMPLE of the whole
metering fleet - not the register-covered meters - and reports the adoption rate
it implies.

Why this matters more than it looks. The labelled set has 37 EV positives, so an
in-sample recall carries a +-0.14 interval and a detector can look respectable by
memorising a handful of meters. Implied prevalence is a different kind of
evidence: if a detector says 40% of households own a wallbox, it is broken no
matter how well it scored, and no amount of cross-validation on 37 meters would
have said so. It is also the only check here computed on a population that
actually resembles the one the detector will be deployed against - the labelled
set is a subsidy register, 84% PV and 64% battery, which the fleet is not.

The sample is drawn from the raw (wide) layer, so this reads the full converted
history, not the labelled extract.
"""

import argparse
import os
import numpy as np
import pandas as pd
import duckdb

from .runner import discover, PRIORS

MAX_SLOTS = 100
SLOT_COLS = [f"v{i:03d}" for i in range(1, MAX_SLOTS + 1)]

# Rough Swiss single-family-household adoption, for a plausibility check only.
# REPLACE with AEW's own figures before quoting any of this - these are priors,
# not measurements, and the register cannot supply them (it sees only subsidies).
EXPECTED = {
    "pv":      (0.05, 0.25),
    "hp":      (0.10, 0.30),
    "ev":      (0.03, 0.15),
    "battery": (0.01, 0.10),
}


def sample_meters(raw_glob, n, seed=11, min_days=365):
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT mp_id, count(DISTINCT d) AS days
        FROM read_parquet('{raw_glob}', hive_partitioning=true)
        GROUP BY mp_id
        HAVING count(DISTINCT d) >= {min_days}
    """).df()
    if df.empty:
        return df, 0
    total = len(df)
    if len(df) > n:
        df = df.sample(n, random_state=seed)
    return df.reset_index(drop=True), total


def load_batch(con, raw_glob, mp_ids):
    """Wide raw layer -> the long per-meter frame detectors expect."""
    lst = ",".join(str(int(x)) for x in mp_ids)
    slots = ", ".join(SLOT_COLS)
    return con.execute(f"""
        WITH picked AS (
            SELECT * FROM read_parquet('{raw_glob}', hive_partitioning=true)
            WHERE mp_id IN ({lst})
        ), long AS (
            SELECT mp_id, obis, d,
                   unnest(range(1, {MAX_SLOTS + 1})) AS slot_idx,
                   unnest([{slots}])                 AS kwh
            FROM picked
        )
        SELECT mp_id,
               d + INTERVAL (15 * (slot_idx - 1)) MINUTE AS ts,
               d,
               max(kwh) FILTER (WHERE obis LIKE '1-1:1.29.0%') AS import_kwh,
               max(kwh) FILTER (WHERE obis LIKE '1-1:2.29.0%') AS export_kwh,
               (month(d) IN (3, 10) AND dayofweek(d) = 0
                AND month(d + INTERVAL 7 DAY) <> month(d))     AS dst_day
        FROM long
        WHERE kwh IS NOT NULL
        GROUP BY mp_id, ts, d, dst_day
        ORDER BY mp_id, ts
    """).df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="glob for the prepared raw parquet layer")
    ap.add_argument("--leaderboard", default="results/leaderboard.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--min-days", type=int, default=365)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--expect", default="", help="override, e.g. hp=0.12:0.28")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    raw = os.path.expanduser(args.raw)

    expected = dict(EXPECTED)
    for kv in filter(None, args.expect.split(",")):
        k, rng = kv.split("=")
        lo, hi = rng.split(":")
        expected[k.strip()] = (float(lo), float(hi))

    lb = pd.read_csv(args.leaderboard)
    lb = lb[lb.stratum == "all"] if "stratum" in lb.columns else lb
    thr = {(r.asset, r.detector): r.threshold for r in lb.itertuples()
           if not pd.isna(r.threshold)}
    circular = {r.detector for r in lb.itertuples() if getattr(r, "circular", False)}

    dets = [d for d in discover() if (d.asset, d.name) in thr]
    print(f"{len(dets)} detectors have a threshold on the leaderboard")

    meters, total = sample_meters(raw, args.sample, args.seed, args.min_days)
    if meters.empty:
        raise SystemExit(f"no meters with >={args.min_days} days under {raw}")
    print(f"fleet: {total} meters with >={args.min_days} days; "
          f"sampled {len(meters)}\n")

    con = duckdb.connect()
    rows = []
    ids = meters.mp_id.tolist()
    for i in range(0, len(ids), args.batch):
        df = load_batch(con, raw, ids[i:i + args.batch])
        if df.empty:
            continue
        df["hour"] = df.ts.dt.hour
        for mp, g in df.groupby("mp_id", sort=False):
            for det in dets:
                try:
                    sc, _ = det.score(mp, g)
                except Exception:
                    sc = float("nan")
                rows.append({"mp_id": mp, "detector": det.name, "asset": det.asset,
                             "score": sc})
        print(f"  {min(i+args.batch, len(ids))}/{len(ids)} meters", flush=True)

    sc = pd.DataFrame(rows)
    sc.to_parquet(os.path.join(args.out, "fleet_scores.parquet"), index=False)

    out = []
    print("\n== implied fleet prevalence ==")
    for det in dets:
        s = sc[sc.detector == det.name].score.to_numpy(float)
        scored = s[~np.isnan(s)]
        coverage = len(scored) / len(s) if len(s) else 0.0
        if len(scored) == 0:
            print(f"  {det.asset:8} {det.name:22} no meter could be scored")
            continue
        t = thr[(det.asset, det.name)]
        flagged = scored >= t
        p = float(flagged.mean())
        # Wilson interval - better behaved than normal approximation at small p
        n, z = len(scored), 1.96
        c = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        lo, hi = max(0.0, c - half), min(1.0, c + half)
        exp_lo, exp_hi = expected.get(det.asset, (0.0, 1.0))
        if det.name in circular:
            verdict = "circular - defines the label"
        elif lo > exp_hi:
            verdict = f"IMPLAUSIBLE, far above the {exp_lo:.0%}-{exp_hi:.0%} expectation"
        elif hi < exp_lo:
            verdict = f"IMPLAUSIBLE, far below the {exp_lo:.0%}-{exp_hi:.0%} expectation"
        else:
            verdict = "plausible"
        print(f"  {det.asset:8} {det.name:22} {p:6.1%} [{lo:.1%},{hi:.1%}]   "
              f"coverage {coverage:.0%}   {verdict}")
        out.append({"asset": det.asset, "detector": det.name, "threshold": t,
                    "n_scored": n, "coverage": round(coverage, 3),
                    "implied_prevalence": round(p, 4),
                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                    "expected_lo": exp_lo, "expected_hi": exp_hi, "verdict": verdict})

    p = os.path.join(args.out, "fleet_prevalence.csv")
    pd.DataFrame(out).to_csv(p, index=False)
    print(f"\n-> {p}")
    print("\nExpected ranges are ASSUMPTIONS (rough Swiss household shares), not")
    print("measurements. Replace them with AEW's own figures via --expect before")
    print("quoting any verdict: the register cannot supply them, since it sees")
    print("only subsidised installations.")
    print("\nA plausible rate is NECESSARY, NOT SUFFICIENT. Where no gold negatives")
    print("exist the threshold is pinned to an assumed prevalence, so any detector -")
    print("including random_* - reproduces that prevalence on the fleet by")
    print("construction. Read this together with lift on the leaderboard: this check")
    print("rules detectors OUT, it never rules one in.")


if __name__ == "__main__":
    main()
