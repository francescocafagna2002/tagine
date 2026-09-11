#!/usr/bin/env python3
"""
Per-customer asset report: run every detector on named meters and show the evidence.

    # by Geschaeftspartner, resolved through the mapping files
    python3 -m evalfw.explain --raw '~/work/store/data/prepared/raw/*/*.parquet' \
        --mapping-dir ~/work/aew-data/test-blob/input_data --gp <gp>,<gp>,<gp>

    # or straight by meter
    python3 -m evalfw.explain --raw '...' --mp-id <mp>,<mp>,<mp>

Prints, per meter: coverage, annual energy, and each detector's score with the
evidence that produced it. With --leaderboard it also states whether the score
clears the threshold that detector was calibrated at, and with --reference it
gives the percentile against a scored population, which is usually more
informative than a bare number.

This is the explainability output, not a classifier: it shows WHICH days and hours
drove each score so a human can agree or disagree. On a single customer that
matters more than any AUC - these detectors range from 0.94 (battery) to 0.64
(boiler), so several of the lines below are suggestive rather than conclusive, and
the report says which is which.
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import duckdb

from .runner import discover
from .timebase import localize

# rough guide to how much weight a line deserves, from the leaderboards
CONFIDENCE = {"battery": ("high", 0.94), "hp": ("moderate", 0.83),
              "electric_heating": ("moderate", 0.83), "pv": ("high", 0.90),
              "ev": ("low-moderate", 0.74), "boiler": ("low", 0.64)}


def resolve_gp(mapping_dir, gps):
    z = os.path.join(mapping_dir, "Zähler-GP.csv")
    m = os.path.join(mapping_dir, "mpid_zähler_mapping.csv")
    for p in (z, m):
        if not os.path.exists(p):
            sys.exit(f"missing {p} - point --mapping-dir at the folder holding the AEW CSVs")
    lst = ",".join("'" + g.strip() + "'" for g in gps)
    return duckdb.connect().execute(f"""
        SELECT trim(z."GPartner") AS gp, trim(z."Zählpunktbezeichnung") AS zp,
               try_cast(trim(m."MP ID") AS BIGINT) AS mp_id
        FROM read_csv('{z}', header=true, all_varchar=true) z
        LEFT JOIN read_csv('{m}', header=true, all_varchar=true) m
          ON trim(m."Zählpunktbezeichnung") = trim(z."Zählpunktbezeichnung")
        WHERE trim(z."GPartner") IN ({lst})
        ORDER BY gp
    """).df()


def load_meter(con, raw_glob, mp_id, max_slots=100):
    slots = ", ".join(f"v{i:03d}" for i in range(1, max_slots + 1))
    df = con.execute(f"""
        WITH picked AS (
            SELECT * FROM read_parquet('{raw_glob}', hive_partitioning=true)
            WHERE mp_id = {int(mp_id)}
        ), long AS (
            SELECT mp_id, obis, d,
                   unnest(range(1, {max_slots + 1})) AS slot_idx,
                   unnest([{slots}])                 AS kwh
            FROM picked
        )
        SELECT mp_id, d + INTERVAL (15 * (slot_idx - 1)) MINUTE AS ts, d,
               max(kwh) FILTER (WHERE obis LIKE '1-1:1.29.0%') AS import_kwh,
               max(kwh) FILTER (WHERE obis LIKE '1-1:2.29.0%') AS export_kwh,
               NULL::DOUBLE AS temp_c,
               (month(d) IN (3,10) AND dayofweek(d) = 0
                AND month(d + INTERVAL 7 DAY) <> month(d)) AS dst_day
        FROM long WHERE kwh IS NOT NULL
        GROUP BY mp_id, ts, d, dst_day ORDER BY ts
    """).df()
    if not df.empty:
        df = localize(df, source="CET")     # AEW stores fixed CET, not local time
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="glob for the prepared raw parquet layer")
    ap.add_argument("--gp", default="", help="comma-separated Geschaeftspartner numbers")
    ap.add_argument("--mp-id", default="", help="comma-separated MP IDs")
    ap.add_argument("--mapping-dir", default=None, help="folder with the AEW mapping CSVs")
    ap.add_argument("--leaderboard", default=None, help="use its thresholds for a verdict")
    ap.add_argument("--reference", default=None,
                    help="a scores.parquet to rank this meter against")
    args = ap.parse_args()
    raw = os.path.expanduser(args.raw)
    if not glob.glob(raw):
        sys.exit(f"no parquet matched {raw}")

    targets = []
    if args.gp:
        if not args.mapping_dir:
            sys.exit("--gp needs --mapping-dir")
        r = resolve_gp(os.path.expanduser(args.mapping_dir), args.gp.split(","))
        if r.empty:
            sys.exit("none of those partner numbers appear in Zähler-GP.csv")
        targets += [(row.gp, row.zp, row.mp_id) for row in r.itertuples()]
        for g in args.gp.split(","):
            if g.strip() not in set(r.gp):
                print(f"! partner {g.strip()} not found in the mapping")
    targets += [(None, None, int(x)) for x in args.mp_id.split(",") if x.strip()]

    thr = {}
    if args.leaderboard and os.path.exists(args.leaderboard):
        lb = pd.read_csv(args.leaderboard)
        lb = lb[lb.stratum == "all"] if "stratum" in lb.columns else lb
        thr = {r.detector: r.threshold for r in lb.itertuples() if not pd.isna(r.threshold)}
    ref = pd.read_parquet(args.reference) if args.reference else None

    dets = discover()
    con = duckdb.connect()
    for gp, zp, mp in targets:
        head = f"MP {mp}" + (f"   ·   GP {gp}   ·   {zp}" if gp else "")
        print("\n" + "=" * 78 + f"\n{head}\n" + "=" * 78)
        if mp is None or (isinstance(mp, float) and np.isnan(mp)):
            print("  no MP ID for this partner - the Zählpunkt has no mapping entry")
            continue
        df = load_meter(con, raw, mp)
        if df.empty:
            print("  NO DATA for this meter in the raw layer")
            continue
        days = df.d.nunique()
        span = (df.d.max() - df.d.min()).days + 1
        yrs = max(span / 365.25, 1e-6)
        print(f"  {df.d.min()} .. {df.d.max()}   {days} days of {span} "
              f"({100*days/span:.0f}% complete)")
        if span >= 180:      # annualising a one-month window is meaningless
            print(f"  import {df.import_kwh.sum()/yrs:,.0f} kWh/yr   "
                  f"export {df.export_kwh.sum()/yrs:,.0f} kWh/yr   "
                  f"peak {df.import_kwh.max()*4:.1f} kW")
        else:
            print(f"  import {df.import_kwh.sum():,.0f} kWh   "
                  f"export {df.export_kwh.sum():,.0f} kWh over the window   "
                  f"peak {df.import_kwh.max()*4:.1f} kW")
        print()
        for det in sorted(dets, key=lambda d: d.asset):
            if getattr(det, "trainable", False) or det.name.startswith("random_"):
                continue
            try:
                sc, ev = det.score(mp, df)
            except Exception as e:
                print(f"  {det.asset:17} {det.name:24} error: {str(e)[:40]}")
                continue
            if sc is None or (isinstance(sc, float) and np.isnan(sc)):
                why = ev.get("reason", "not applicable") if isinstance(ev, dict) else ""
                print(f"  {det.asset:17} {det.name:24} —        {why}")
                continue
            verdict = ""
            if det.name in thr:
                verdict = "ABOVE threshold" if sc >= thr[det.name] else "below threshold"
                verdict += f" ({thr[det.name]:.3g})"
            if ref is not None:
                pool = ref[ref.detector == det.name].score.dropna()
                if len(pool) > 20:
                    verdict += f"   p{100*(pool < sc).mean():.0f} of population"
            conf = CONFIDENCE.get(det.asset, ("unknown", 0))[0]
            if getattr(det, "anti_predictive", False):
                conf = "IGNORE - worse than chance"
            print(f"  {det.asset:17} {det.name:24} {sc:>9.3f}  [{conf}] {verdict}")
            if isinstance(ev, dict) and ev:
                bits = ", ".join(f"{k}={v}" for k, v in list(ev.items())[:4])
                print(f"  {'':17} {'':24}             {bits}")
    print("\nConfidence reflects each detector's measured AUC, not this meter: "
          "battery 0.94, PV ~0.90,\nheat pump 0.83, EV 0.74, boiler 0.64. Treat the "
          "low-confidence lines as hypotheses.")


if __name__ == "__main__":
    main()
