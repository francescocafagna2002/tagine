#!/usr/bin/env python3
"""
Build the modelling table: one Parquet holding the full 15-minute history of only
those meters we have register metadata for, with their labels attached.

    python3 make_training_set.py --prepared ~/work/prepared --out ~/work/training_set.parquet

Input  : the raw/ parquet layer written by prepare_meter_data.py, plus its
         labels.parquet (run that script with --steps labels)
Output : one row per (meter, timestamp) with import and export side by side, the
         asset flags, and per-asset commissioning dates.

Shape, and why:
  Import and export are pivoted into two columns rather than kept as two OBIS rows,
  which halves the row count and is the shape every model wants. Metadata is joined
  onto each row so the file is self-contained; the columns are low-cardinality per
  meter, so dictionary encoding makes them nearly free. A companion meters.parquet
  with one row per meter is written alongside for convenience.

Three traps this handles, all found in the AEW data:
  1. A meter can carry SEVERAL register rows (one per subsidised installation, in
     different years). Flags are OR-ed and each asset gets its OWN commissioning
     date, so PV from 2019 and a wallbox from 2023 do not collapse into one date.
  2. Labels attach to a Geschaeftspartner, not a meter, and some partners own
     several meters. For those the label is ambiguous - which meter has the heat
     pump? n_meters_for_gp exposes this; filter to = 1 for a clean training set.
  3. The register is a CURRENT snapshot. An asset is only visible in data recorded
     after it went live, so *_live booleans mark, per row, whether each asset
     actually existed at that timestamp. Train on those, not on the raw flag.
"""

import argparse
import os
import sys
import glob

try:
    import duckdb
except ImportError:
    sys.exit("duckdb missing. Install it with:  pip install --user duckdb")

MAX_SLOTS = 100
SLOT_COLS = [f"v{i:03d}" for i in range(1, MAX_SLOTS + 1)]
ASSETS = ["pv", "hp", "ev", "battery", "hp_boiler"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", "--stage1-out", dest="prepared", required=True,
                    help="prepare_meter_data.py output dir (holds raw/ and labels.parquet)")
    ap.add_argument("--out", required=True, help="destination .parquet")
    ap.add_argument("--memory", default="4GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--temp-dir", default=None, help="spill dir; set it when memory is tight")
    ap.add_argument("--unambiguous-only", action="store_true",
                    help="keep only meters whose Geschaeftspartner owns exactly one meter")
    ap.add_argument("--asset-date", choices=["last", "first"], default="last",
                    help="with several register rows for one asset: 'last' (default) uses "
                         "the most recent installation date, 'first' the earliest")
    ap.add_argument("--strict-dates", action="store_true",
                    help="use only real InBetrieb-Daten; treat signature-derived dates as unknown")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="drop meters whose share of non-null readings is below this (0-1)")
    args = ap.parse_args()

    raw_glob = os.path.join(args.prepared, "raw", "*", "*.parquet")
    labels = os.path.join(args.prepared, "labels.parquet")
    if not glob.glob(raw_glob):
        sys.exit(f"no raw parquet under {raw_glob} - run prepare_meter_data.py --steps convert first")
    if not os.path.exists(labels):
        sys.exit(f"{labels} not found - run prepare_meter_data.py --steps labels first")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    if args.temp_dir:
        os.makedirs(args.temp_dir, exist_ok=True)
        con.execute(f"SET temp_directory='{args.temp_dir}'")
    con.execute("SET preserve_insertion_order=false")   # lets the writer stream

    # ---- 1. one metadata row per meter -------------------------------------
    asset_cols = []
    for a in ASSETS:
        asset_cols.append(f"COALESCE(bool_or(has_{a}), FALSE) AS has_{a}")
        # in strict mode only real InBetrieb-Daten count; signature-derived dates run
        # early (median 145 days) and would mark an asset live before it existed
        keep = f"has_{a} AND NOT live_from_estimated" if args.strict_dates else f"has_{a}"
        agg = "max" if args.asset_date == "last" else "min"
        asset_cols.append(f"{agg}(asset_live_from) FILTER (WHERE {keep}) AS {a}_from")
        asset_cols.append(
            f"COALESCE(bool_or(live_from_estimated) FILTER (WHERE has_{a}), FALSE) "
            f"AS {a}_from_estimated")
    con.execute(f"""
        CREATE OR REPLACE TABLE meters AS
        WITH per_meter AS (
            SELECT mp_id, any_value(gp) AS gp, any_value(zp) AS zp,
                   any_value(plz) AS reg_plz, any_value(ort) AS ort,
                   max(pv_kwp) AS pv_kwp, count(*) AS n_register_rows,
                   {', '.join(asset_cols)}
            FROM '{labels}'
            GROUP BY mp_id
        ),
        gp_size AS (
            SELECT gp, count(DISTINCT mp_id) AS n_meters_for_gp
            FROM '{labels}' GROUP BY gp
        )
        SELECT p.*, g.n_meters_for_gp
        FROM per_meter p LEFT JOIN gp_size g USING (gp)
    """)
    n_meters = con.execute("SELECT count(*) FROM meters").fetchone()[0]
    print(f"register covers {n_meters} meters")

    if args.unambiguous_only:
        con.execute("DELETE FROM meters WHERE n_meters_for_gp <> 1")
        print(f"  -> {con.execute('SELECT count(*) FROM meters').fetchone()[0]} after "
              f"dropping multi-meter partners")

    # ---- 2. how many of them are actually in the measurements? -------------
    con.execute(f"""
        CREATE OR REPLACE TABLE present AS
        SELECT DISTINCT mp_id FROM read_parquet('{raw_glob}', hive_partitioning=true)
        WHERE mp_id IN (SELECT mp_id FROM meters)
    """)
    n_present = con.execute("SELECT count(*) FROM present").fetchone()[0]
    print(f"of those, {n_present} appear in the load data"
          f"{' - THIS is your usable label set' if n_present else ''}")
    if not n_present:
        sys.exit("no overlap between register and measurements - check the mapping files")

    # ---- 3. dedupe, unpivot, pivot import/export ---------------------------
    slots = ", ".join(SLOT_COLS)
    con.execute(f"""
        CREATE OR REPLACE VIEW picked AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (PARTITION BY mp_id, obis, d) AS rn
            FROM read_parquet('{raw_glob}', hive_partitioning=true)
            WHERE mp_id IN (SELECT mp_id FROM present)
        ) WHERE rn = 1
    """)
    dups = con.execute(f"""
        SELECT count(*) FROM (
            SELECT mp_id, obis, d, count(*) c
            FROM read_parquet('{raw_glob}', hive_partitioning=true)
            WHERE mp_id IN (SELECT mp_id FROM present)
            GROUP BY 1,2,3 HAVING c > 1)
    """).fetchone()[0]
    if dups:
        print(f"  ! {dups} duplicate (meter, obis, day) keys - keeping one row of each")

    con.execute(f"""
        CREATE OR REPLACE VIEW long AS
        SELECT mp_id, plz, obis,
               d + INTERVAL (15 * (slot_idx - 1)) MINUTE AS ts,
               d, slot_idx, kwh
        FROM (
            SELECT mp_id, plz, obis, d,
                   unnest(range(1, {MAX_SLOTS + 1})) AS slot_idx,
                   unnest([{slots}])                 AS kwh
            FROM picked
        )
        WHERE kwh IS NOT NULL
    """)

    live = ", ".join(
        f"(m.{a}_from IS NOT NULL AND l.d >= m.{a}_from) AS {a}_live" for a in ASSETS
    )
    flags = ", ".join(f"m.has_{a}" for a in ASSETS)
    est = ", ".join(f"m.{a}_from_estimated" for a in ASSETS)
    froms = ", ".join(f"m.{a}_from" for a in ASSETS)

    print("writing ...", flush=True)
    con.execute(f"""
        COPY (
            SELECT
                l.mp_id, m.zp, l.plz, l.ts, l.d, l.slot_idx,
                max(l.kwh) FILTER (WHERE l.obis LIKE '1-1:1.29.0%') AS import_kwh,
                max(l.kwh) FILTER (WHERE l.obis LIKE '1-1:2.29.0%') AS export_kwh,
                -- DST transition days: the source writes a fixed 96-slot grid, so the
                -- October fall-back hour is missing and March carries padding
                (month(l.d) IN (3, 10) AND dayofweek(l.d) = 0
                 AND month(l.d + INTERVAL 7 DAY) <> month(l.d)) AS dst_day,
                {flags}, {froms}, {live}, {est},
                m.pv_kwp, m.gp, m.n_meters_for_gp, m.n_register_rows, m.ort
            FROM long l JOIN meters m USING (mp_id)
            GROUP BY l.mp_id, m.zp, l.plz, l.ts, l.d, l.slot_idx,
                     {flags}, {froms}, {est}, m.pv_kwp, m.gp, m.n_meters_for_gp,
                     m.n_register_rows, m.ort
            ORDER BY l.mp_id, l.ts
        ) TO '{args.out}' (FORMAT parquet, COMPRESSION zstd)
    """)

    companion = os.path.join(os.path.dirname(os.path.abspath(args.out)), "meters.parquet")
    con.execute(f"COPY (SELECT * FROM meters WHERE mp_id IN (SELECT mp_id FROM present)) "
                f"TO '{companion}' (FORMAT parquet)")

    # ---- 4. report ---------------------------------------------------------
    s = con.execute(f"""
        SELECT count(*), count(DISTINCT mp_id), min(d), max(d),
               round(100.0 * count(import_kwh) / count(*), 1),
               round(100.0 * count(export_kwh) / count(*), 1)
        FROM '{args.out}'
    """).fetchone()
    sz = os.path.getsize(args.out)
    human = f"{sz/1e9:.2f} GB" if sz >= 1e9 else (f"{sz/1e6:.1f} MB" if sz >= 1e6 else f"{sz/1e3:.0f} KB")
    print(f"\n{args.out}  ({human})")
    print(f"  {s[0]:,} rows, {s[1]} meters, {s[2]} .. {s[3]}")
    print(f"  import present on {s[4]}% of rows, export on {s[5]}%")
    print(f"  {companion}  (one row per meter)")

    print("\nlabel counts (meters, and how many have data before/after commissioning):")
    for a in ASSETS:
        r = con.execute(f"""
            SELECT count(DISTINCT mp_id) FILTER (WHERE has_{a}),
                   count(DISTINCT mp_id) FILTER (WHERE {a}_live),
                   count(DISTINCT mp_id) FILTER (WHERE has_{a} AND NOT {a}_live)
            FROM '{args.out}'
        """).fetchone()
        print(f"  {a:10} {r[0]:4} meters   {r[1]:4} with post-commissioning data   "
              f"{r[2]:4} with pre-commissioning data (usable as before/after pairs)")


if __name__ == "__main__":
    main()
