#!/usr/bin/env python3
"""
Inspect the modelling table: shape, label counts, and the average day per asset
class - the quickest way to see whether a signal exists before modelling anything.

    python3 inspect_training_set.py --set training_set.parquet [--plots out_dir]

Every comparison is asset-LIVE vs not, never the raw register flag, because a
meter whose PV was commissioned after the measurement window carries the flag but
none of the behaviour.
"""

import argparse
import os
import sys

try:
    import duckdb
except ImportError:
    sys.exit("duckdb missing:  pip install --user duckdb")

ASSETS = ["pv", "hp", "ev", "battery"]


def rel(con, path):
    return f"read_parquet('{path}')"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="training_set.parquet")
    ap.add_argument("--plots", default=None, help="directory to write PNGs into")
    ap.add_argument("--memory", default="4GB")
    args = ap.parse_args()

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory}'")
    t = rel(con, args.set)

    print("== shape ==")
    con.sql(f"""
        SELECT count(*) AS rows, count(DISTINCT mp_id) AS meters,
               min(d) AS first_day, max(d) AS last_day,
               count(DISTINCT d) AS days,
               round(sum(import_kwh)/1000, 1) AS import_mwh,
               round(sum(export_kwh)/1000, 1) AS export_mwh
        FROM {t}
    """).show()

    print("== per-meter coverage (how complete is each meter's history?) ==")
    con.sql(f"""
        WITH per_meter AS (
            SELECT mp_id, count(*) AS n, count(DISTINCT d) AS days FROM {t} GROUP BY 1
        ), span AS (SELECT count(DISTINCT d) AS total_days FROM {t})
        SELECT CASE WHEN days >= 0.95*total_days THEN 'a. complete (>=95%)'
                    WHEN days >= 0.50*total_days THEN 'b. partial (50-95%)'
                    ELSE 'c. sparse (<50%)' END AS bucket,
               count(*) AS meters, round(avg(days)) AS avg_days
        FROM per_meter, span GROUP BY 1 ORDER BY 1
    """).show()

    print("== labels: meters per asset, and how much before/after data exists ==")
    parts = " UNION ALL ".join(f"""
        SELECT '{a}' AS asset,
               count(DISTINCT mp_id) FILTER (WHERE has_{a})                  AS flagged,
               count(DISTINCT mp_id) FILTER (WHERE {a}_live)                 AS live_somewhere,
               count(DISTINCT mp_id) FILTER (WHERE has_{a} AND NOT {a}_live) AS has_pre_period,
               count(DISTINCT mp_id) FILTER (WHERE NOT has_{a})              AS flagged_negative
        FROM {t}""" for a in ASSETS)
    con.sql(parts).show()
    print("  flagged_negative counts the register's '-', which means 'no subsidy record',")
    print("  NOT 'no asset'. Do not use it as a negative label without checking.\n")

    print("== average day by hour: asset live vs not (kWh per 15 min) ==")
    for a in ASSETS:
        print(f"-- {a} --")
        con.sql(f"""
            SELECT hour(ts) AS h,
                   round(avg(import_kwh) FILTER (WHERE {a}_live), 3) AS live,
                   round(avg(import_kwh) FILTER (WHERE NOT {a}_live), 3) AS not_live,
                   round(avg(import_kwh) FILTER (WHERE {a}_live)
                         - avg(import_kwh) FILTER (WHERE NOT {a}_live), 3) AS diff
            FROM {t} WHERE NOT dst_day GROUP BY 1 ORDER BY 1
        """).show(max_rows=30)

    print("== highest-power meters (one row each; EV charging shows up here) ==")
    con.sql(f"""
        SELECT mp_id, has_ev, ev_live, has_hp,
               round(max(import_kwh), 2) AS peak_kwh_15min,
               round(max(import_kwh) * 4, 1) AS implied_kw,
               round(sum(import_kwh)) AS total_kwh
        FROM {t} GROUP BY 1,2,3,4 ORDER BY peak_kwh_15min DESC LIMIT 15
    """).show()
    print("  implied_kw near 3.7 / 7.4 / 11 is a wallbox signature; much higher is")
    print("  usually a business or a heat pump with electric backup.\n")

    if args.plots:
        make_plots(con, t, args.plots)


def make_plots(con, t, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n! matplotlib not installed - skipping plots (pip install --user matplotlib)")
        return
    os.makedirs(outdir, exist_ok=True)

    # average daily shape per asset, live vs not
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, a in zip(axes.flat, ASSETS):
        df = con.sql(f"""
            SELECT slot_idx,
                   avg(import_kwh) FILTER (WHERE {a}_live)     AS live,
                   avg(import_kwh) FILTER (WHERE NOT {a}_live) AS not_live
            FROM {t} WHERE NOT dst_day AND slot_idx <= 96 GROUP BY 1 ORDER BY 1
        """).df()
        ax.plot(df.slot_idx / 4, df.live, label=f"{a} live")
        ax.plot(df.slot_idx / 4, df.not_live, label=f"no {a}", linestyle="--")
        ax.set_title(f"{a}: mean import per 15 min")
        ax.set_xlabel("hour"); ax.set_ylabel("kWh"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(outdir, "daily_shape_by_asset.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"  -> {p}")

    # PV: import vs export on an average day
    df = con.sql(f"""
        SELECT slot_idx, avg(import_kwh) AS imp, avg(export_kwh) AS exp
        FROM {t} WHERE pv_live AND NOT dst_day AND slot_idx <= 96 GROUP BY 1 ORDER BY 1
    """).df()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df.slot_idx / 4, df.imp, label="import")
    ax.plot(df.slot_idx / 4, df.exp, label="export")
    ax.set_title("PV-live meters: mean day"); ax.set_xlabel("hour"); ax.set_ylabel("kWh/15min")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    p = os.path.join(outdir, "pv_import_export.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"  -> {p}")

    # a few example meter-days with the largest sustained draw
    ex = con.sql(f"""
        WITH cand AS (
            SELECT mp_id, d, max(import_kwh) AS pk, any_value(has_ev) AS has_ev
            FROM {t} GROUP BY 1,2 ORDER BY pk DESC LIMIT 6
        ) SELECT * FROM cand
    """).df()
    if len(ex):
        fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
        for ax, (_, r) in zip(axes.flat, ex.iterrows()):
            df = con.sql(f"""
                SELECT slot_idx, import_kwh FROM {t}
                WHERE mp_id = {r.mp_id} AND d = DATE '{r.d}' ORDER BY slot_idx
            """).df()
            ax.plot(df.slot_idx / 4, df.import_kwh)
            ax.set_title(f"mp {r.mp_id}  {r.d}  ev={r.has_ev}", fontsize=9)
            ax.set_xlabel("hour"); ax.grid(alpha=.3)
        fig.tight_layout()
        p = os.path.join(outdir, "example_meter_days.png")
        fig.savefig(p, dpi=110); plt.close(fig)
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
