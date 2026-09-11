#!/usr/bin/env python3
"""
Stage 1 for the AEW "Energy Fingerprints" challenge (Energy Data Hackdays 2026).

Runs WHERE THE DATA LIVES (Renku session), turns the raw semicolon CSVs into
compact Parquet, and emits a small meter x day feature table that is the only
thing you actually need to carry back to a laptop.

    python3 prepare_meter_data.py --input DIR --out DIR [--steps profile,convert,labels,features]

Steps
  profile   classify every CSV, report shape / coverage / anomalies -> profile.json
  convert   load CSVs        -> out/raw/ym=YYYY-MM/*.parquet   (wide, ~8x smaller)
  labels    GIGI + mappings  -> out/labels.parquet             (asset ground truth)
  features  raw              -> out/features/ym=YYYY-MM/*.parquet (meter x day)
            plus out/coverage.parquet and out/pv_export.parquet
  validate  confront register labels against measured export -> label_check.parquet

Why the raw layer is stored "wide": one row per meter/day/direction with 96 slot
columns is ~8.6x smaller than the source CSV and ~5x smaller than the same data
in long format, and DuckDB prunes unread columns, so a query over one time-slot
across years touches megabytes. Use readings_view() below when you want long.

KNOWN DATA TRAPS this script handles (all verified against the 2023-09 extract):
  1. The header row has 4 metadata columns but data rows have 5 - the
     Zaehlpunktbezeichnung column is missing from the header. A naive
     pd.read_csv shifts every value by one column. We ignore the header and
     declare the schema explicitly.
  2. A trailing ';' on every row produces a phantom final column.
  3. Slot columns run 00:15..23:45 then 00:00 - the LAST column is the interval
     ending at midnight, not the first of the day. Slot i covers
     [(i-1)*15min, i*15min) after local midnight.
  4. ~7% of values are NULL, and they are day-shaped, not scattered: whole
     meter-days are missing and some meters are empty for a whole month.
     Coverage is therefore computed explicitly, never assumed.
  5. DST days have 92 (March) or 100 (October) intervals instead of 96. The
     schema is padded to 100 slots so those files parse, and the days are
     flagged via the dst_day column so you can drop them.
  6. OBIS 1-1:1.29.0*255 = import (consumption), 1-1:2.29.0*255 = export (PV
     feed-in). Verified: export peaks midday, ~0 at night.
"""

import argparse
import json
import os
import sys
import glob
import datetime as dt
import re

try:
    import duckdb
except ImportError:
    sys.exit("duckdb missing. Install it with:  pip install --user duckdb")

MAX_SLOTS = 100          # October DST day: 25h * 4
SLOT_COLS = [f"v{i:03d}" for i in range(1, MAX_SLOTS + 1)]
META = ["mp_id", "zp", "obis", "datum", "plz"]
OBIS_IMPORT = "1-1:1.29.0*255"
OBIS_EXPORT = "1-1:2.29.0*255"

# Slot index windows, 1-based. Slot i covers [(i-1)*15, i*15) minutes after midnight.
WINDOWS = {
    "night":     (1, 24),    # 00:00-06:00
    "morning":   (25, 40),   # 06:00-10:00
    "midday":    (41, 60),   # 10:00-15:00
    "afternoon": (61, 68),   # 15:00-17:00
    "evening":   (69, 88),   # 17:00-22:00
    "late":      (89, 96),   # 22:00-24:00
}

# kWh per 15 min for common single/three-phase EV charge rates (3.7 / 7.4 / 11 kW).
EV_BANDS = {"p37": 0.90, "p74": 1.80, "p11": 2.70}
RUN_THRESHOLD = 0.80     # kWh/15min (~3.2 kW) - sustained block => likely EV


# --------------------------------------------------------------------------- #
# file classification
# --------------------------------------------------------------------------- #

def sniff(path):
    """Identify a CSV by its header, so the script works whatever the files are named."""
    with open(path, "rb") as fh:
        head = fh.read(4096).decode("utf-8", "replace").lstrip("﻿")
    first = head.splitlines()[0] if head.splitlines() else ""
    low = first.lower()
    if "mp id" in low and "00:15" in low:
        return "load"
    if "mp id" in low and "zählpunkt" in low:
        return "mpid_map"
    if "gpartner" in low:
        return "zaehler_gp"
    if "gp-nr" in low:
        return "gigi"
    return "unknown"


def classify(input_dir):
    found = {"load": [], "mpid_map": [], "zaehler_gp": [], "gigi": [], "unknown": []}
    for p in sorted(glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)):
        found[sniff(p)].append(p)
    return found


def connect(args):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    if args.temp_dir:
        os.makedirs(args.temp_dir, exist_ok=True)
        con.execute(f"SET temp_directory='{args.temp_dir}'")
    return con


TIME_HDR = re.compile(r"^\d{1,2}:\d{2}$")
DATE_VAL = re.compile(r"^(\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2})$")
ZP_VAL = re.compile(r"^[A-Za-z]{2}[0-9A-Za-z]{10,}$")
OBIS_VAL = re.compile(r"^\d+-\d+:")

HDR_ALIASES = {
    "mpid": "mp_id", "zahlpunktbezeichnung": "zp", "obiscode": "obis",
    "datum": "datum", "plz": "plz",
}


def _norm_hdr(s):
    s = s.strip().lower()
    for a, b in (("ä", "a"), ("ö", "o"), ("ü", "u"), ("ß", "ss")):
        s = s.replace(a, b)
    return "".join(ch for ch in s if ch.isalnum())


def detect_layout(path, probe_lines=400):
    """Work out a load file's real column layout from its header AND its data.

    The header cannot be trusted on its own: the 2023 extract declares 4 metadata
    columns while its rows carry 5 (an unnamed Zaehlpunktbezeichnung), whereas the
    2026 exports genuinely have 4. Getting this wrong shifts every column, which
    shows up as dates that will not parse rather than as an error.

    The date column is the anchor - it is the only field with an unambiguous
    pattern - so its position in a real data row fixes how many metadata columns
    precede it, and the remaining width gives the slot count (92/96/100 for
    short/normal/long DST days).
    """
    with open(path, "rb") as fh:
        header = fh.readline().decode("utf-8", "replace").lstrip("\ufeff").rstrip("\r\n")
        first_row = None
        widths = set()
        for _ in range(probe_lines):
            line = fh.readline()
            if not line:
                break
            widths.add(line.count(b";") + 1)
            if first_row is None:
                first_row = line.decode("utf-8", "replace").rstrip("\r\n").split(";")

    size = os.path.getsize(path)
    with open(path, "rb") as fh:                     # also probe the middle and tail
        for offset in (size // 2, max(0, size - 4_000_000)):
            fh.seek(offset)
            fh.readline()
            for _ in range(probe_lines):
                line = fh.readline()
                if not line:
                    break
                widths.add(line.count(b";") + 1)

    if not widths or first_row is None:
        raise RuntimeError(f"{path}: no data rows found")

    hf = header.split(";")
    has_trailing = bool(hf) and hf[-1] == ""
    hf_eff = hf[:-1] if has_trailing else hf
    time_at = [i for i, x in enumerate(hf_eff) if TIME_HDR.match(x.strip())]
    if not time_at:
        raise RuntimeError(f"{path}: no HH:MM columns in the header - not a load file?")
    hdr_meta = [_norm_hdr(x) for x in hf_eff[:time_at[0]]]
    if "datum" not in hdr_meta:
        raise RuntimeError(f"{path}: no 'Datum' column in the header: {hf_eff[:time_at[0]]}")
    pos_datum_hdr = hdr_meta.index("datum")
    n_after_datum = len(hdr_meta) - pos_datum_hdr - 1

    # anchor on the date in a real row
    idx_datum = next((i for i, v in enumerate(first_row[:12]) if DATE_VAL.match(v.strip())), None)
    if idx_datum is None:
        raise RuntimeError(
            f"{path}: no date-looking field in the first data row: {first_row[:8]}. "
            f"If the date format changed, add it to DATE_VAL and to the strptime lists."
        )

    n_meta = idx_datum + 1 + n_after_datum
    n_fields = max(widths)
    n_slots = n_fields - n_meta - (1 if has_trailing else 0)
    if not 88 <= n_slots <= MAX_SLOTS:
        raise RuntimeError(
            f"{path}: derived {n_slots} slots from {n_fields} fields and {n_meta} metadata "
            f"columns, which is not a plausible day. Widths seen: {sorted(widths)}."
        )

    # Name the metadata columns from what the DATA looks like, not from header
    # order. The unnamed Zaehlpunkt in the 2023 extract sits between MP ID and
    # OBIS-Code, so appending unnamed columns after the named ones mislabels the
    # OBIS column - which parses fine and silently yields zero export readings.
    names = [None] * n_meta
    names[idx_datum] = "datum"
    for i in range(n_meta):
        if i == idx_datum or i >= len(first_row):
            continue
        v = first_row[i].strip()
        if ZP_VAL.match(v):
            names[i] = "zp"
        elif OBIS_VAL.match(v):
            names[i] = "obis"
    # the first still-unnamed field before the date is the meter id
    for i in range(idx_datum):
        if names[i] is None:
            names[i] = "mp_id"
            break
    # anything after the date follows the header (PLZ in every format seen so far)
    trailing = [HDR_ALIASES.get(h, h) for h in hdr_meta[pos_datum_hdr + 1:]]
    for i in range(idx_datum + 1, n_meta):
        names[i] = trailing.pop(0) if trailing else f"meta{i}"
    names = [n if n else f"meta{i}" for i, n in enumerate(names)]

    if "mp_id" not in names or "obis" not in names:
        raise RuntimeError(
            f"{path}: could not identify the mp_id/obis columns. Derived {names} from "
            f"first row {first_row[:6]}."
        )
    if len(names) != n_meta:
        raise RuntimeError(f"{path}: metadata naming produced {len(names)} of {n_meta}: {names}")
    return {"n_fields": n_fields, "n_slots": n_slots, "meta": names,
            "has_trailing": has_trailing, "widths": sorted(widths)}


def read_load_sql(path, layout):
    """read_csv over a load file, declaring the schema the layout detection derived."""
    cols = [f"'{c}': 'VARCHAR'" for c in layout["meta"]]
    cols += [f"'{c}': 'FLOAT'" for c in SLOT_COLS[:layout["n_slots"]]]
    if layout["has_trailing"]:
        cols.append("'junk': 'VARCHAR'")
    return (
        f"read_csv('{path}', delim=';', header=false, skip=1, "
        f"columns={{{', '.join(cols)}}}, null_padding=true, ignore_errors=false, "
        f"max_line_size=10000000)"
    )


def meta_select(layout):
    """Emit a fixed set of metadata columns whether or not the file carries them."""
    have = set(layout["meta"])
    return ", ".join([
        "CAST(mp_id AS BIGINT) AS mp_id",
        "zp" if "zp" in have else "NULL::VARCHAR AS zp",
        "obis" if "obis" in have else "NULL::VARCHAR AS obis",
        "plz" if "plz" in have else "NULL::VARCHAR AS plz",
    ])


def normalized_slots(n_slots):
    """Always emit v001..v100 so every partition shares one parquet schema."""
    return ", ".join(
        SLOT_COLS[:n_slots]
        + [f"NULL::FLOAT AS {c}" for c in SLOT_COLS[n_slots:]]
    )


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #

def step_profile(con, files, args):
    report = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "files": []}
    for kind, paths in files.items():
        for p in paths:
            entry = {"path": p, "kind": kind, "size_mb": round(os.path.getsize(p) / 1e6, 1)}
            if kind == "load":
                lay = detect_layout(p)
                n_slots = lay["n_slots"]
                entry.update(n_fields=lay["n_fields"], n_slots=n_slots,
                             meta_cols=lay["meta"], widths_seen=lay["widths"])
                src_sql = read_load_sql(p, lay)
                slots = ",".join(SLOT_COLS[:n_slots])
                present = f"list_count(list_filter([{slots}], x -> x IS NOT NULL))"
                q = con.execute(f"""
                    SELECT count(*), count(DISTINCT mp_id),
                           min(try_strptime(datum, ['%d.%m.%Y','%Y-%m-%d'])),
                           max(try_strptime(datum, ['%d.%m.%Y','%Y-%m-%d'])),
                           list(DISTINCT obis),
                           sum(CASE WHEN {present} = 0 THEN 1 ELSE 0 END),
                           sum(CASE WHEN {present} BETWEEN 1 AND {n_slots - 1} THEN 1 ELSE 0 END),
                           sum(CASE WHEN datum IS NULL OR mp_id IS NULL THEN 1 ELSE 0 END),
                           count(*) - count(DISTINCT concat_ws('|', mp_id, obis, datum))
                    FROM {src_sql}
                """).fetchone()
                entry.update(
                    rows=q[0], meters=q[1],
                    date_min=str(q[2]), date_max=str(q[3]), obis=q[4],
                    empty_rows=q[5], partial_rows=q[6], unparsable_rows=q[7],
                    duplicate_rows=q[8],
                )
                # (mp_id, obis, date) must be unique; duplicates double-count energy
                # in every daily total downstream, without any error being raised
                if q[8]:
                    print(f"  !! {os.path.basename(p)}: {q[8]} duplicate (meter, obis, date) "
                          f"rows. Deduplicate before trusting any sum.")
                entry["empty_pct"] = round(100.0 * q[5] / q[0], 2) if q[0] else None
                if n_slots != 96:
                    entry["note"] = f"{n_slots} slots/day - DST file, window features assume 96"
                # a shifted layout shows up as dates that will not parse, never as an
                # error, so refuse to go any further rather than convert garbage
                if q[4] is not None and len(q[4]) > 10:
                    entry["FATAL"] = "obis column holds many distinct values"
                    print(f"  !! {os.path.basename(p)}: the OBIS column has {len(q[4])} distinct "
                          f"values; expected 2. The column layout is mislabelled: {lay['meta']}")
                if q[2] is None or q[7]:
                    entry["FATAL"] = "date column did not parse"
                    print(f"  !! {os.path.basename(p)}: the Datum column did not parse.")
                    print(f"     Detected metadata columns: {lay['meta']}")
                    print(f"     If those look wrong the layout detection is off; if they look")
                    print(f"     right, the date format changed - extend DATE_VAL and strptime.")
            report["files"].append(entry)
            print(f"  [{kind}] {os.path.basename(p)}: {entry.get('rows', '-')} rows, "
                  f"{entry.get('meters', '-')} meters, "
                  f"{entry.get('date_min', '')}..{entry.get('date_max', '')}, "
                  f"{entry.get('n_slots', '-')} slots/day, "
                  f"{entry.get('empty_pct', '-')}% empty rows, "
                  f"{entry.get('duplicate_rows', '-')} dup rows")
    out = os.path.join(args.out, "profile.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"  -> {out}")
    return report


def step_convert(con, files, args):
    """Load CSVs -> partitioned wide Parquet. One file at a time to bound memory."""
    raw = os.path.join(args.out, "raw")
    done_dir = os.path.join(args.out, "_done")
    os.makedirs(done_dir, exist_ok=True)
    for p in files["load"]:
        marker = os.path.join(done_dir, os.path.basename(p) + ".done")
        if os.path.exists(marker) and not args.force:
            print(f"  skip (done): {os.path.basename(p)}")
            continue
        lay = detect_layout(p)
        n_slots = lay["n_slots"]
        bad = con.execute(f"""
            SELECT count(*) FILTER (WHERE try_strptime(datum, ['%d.%m.%Y','%Y-%m-%d']) IS NULL),
                   count(*)
            FROM {read_load_sql(p, lay)}
        """).fetchone()
        if bad[0]:
            raise RuntimeError(
                f"{p}: {bad[0]} of {bad[1]} rows have an unparsable date. Detected metadata "
                f"columns {lay['meta']} - the layout is probably wrong. Refusing to convert."
            )
        print(f"  converting {os.path.basename(p)} "
              f"({n_slots} slots/day, meta={lay['meta']}) ...", flush=True)
        con.execute(f"""
            COPY (
                SELECT
                    {meta_select(lay)},
                    try_strptime(datum, ['%d.%m.%Y','%Y-%m-%d'])::DATE      AS d,
                    strftime(try_strptime(datum, ['%d.%m.%Y','%Y-%m-%d']), '%Y-%m') AS ym,
                    {n_slots}                                               AS n_slots,
                    {normalized_slots(n_slots)}
                FROM {read_load_sql(p, lay)}
                WHERE mp_id IS NOT NULL AND datum IS NOT NULL
            ) TO '{raw}'
              (FORMAT parquet, PARTITION_BY (ym), COMPRESSION zstd,
               OVERWRITE_OR_IGNORE, FILENAME_PATTERN '{os.path.basename(p)[:40]}_{{i}}')
        """)
        open(marker, "w").close()
    print(f"  -> {raw}")


def resolve_cols(con, path, wanted):
    """Map logical names to the file's real header, tolerating whitespace/case/umlaut drift.

    The GIGI export has headers like ' PV' and 'PV-Leistung in kWp ' with stray
    spaces, and readers differ on whether they trim them. Match on a normalized
    form instead of hard-coding the exact string.
    """
    cols = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_csv('{path}', delim=';', header=true, all_varchar=true)"
    ).fetchall()]

    def norm(s):
        s = s.strip().lower()
        for a, b in (("ä", "a"), ("ö", "o"), ("ü", "u"), ("ß", "ss")):
            s = s.replace(a, b)
        return "".join(ch for ch in s if ch.isalnum())

    index = {norm(c): c for c in cols}
    out = {}
    for logical, candidates in wanted.items():
        hit = next((index[norm(c)] for c in candidates if norm(c) in index), None)
        if hit is None:
            hit = next((c for c in cols if any(norm(x) in norm(c) for x in candidates)), None)
        out[logical] = hit
        if hit is None:
            print(f"  ! column for '{logical}' not found in {os.path.basename(path)}; "
                  f"it will be NULL. Header was: {cols}")
    return out


def step_labels(con, files, args):
    """GIGI asset register joined down to MP ID, with validity dates."""
    if not (files["gigi"] and files["zaehler_gp"] and files["mpid_map"]):
        print("  ! missing one of GIGI / Zaehler-GP / mpid mapping - skipping labels")
        return
    gigi, zgp, mpm = files["gigi"][0], files["zaehler_gp"][0], files["mpid_map"][0]

    c = resolve_cols(con, gigi, {
        "gp":       ["GP-Nr", "GPartner", "GP Nr"],
        "hp":       ["WärmePumpe", "Wärmepumpe"],
        "pv":       ["PV"],
        "battery":  ["Batterie/Speicher", "Batterie"],
        "ev":       ["Ladestation für Elektrofahrzeuge", "Ladestation"],
        "hp_boil":  ["Wärmepumpenboiler"],
        "kwp":      ["PV-Leistung in kWp"],
        "inbetr":   ["InBetrieb-Datum", "Inbetriebnahme"],
        "signed":   ["Datum Unterschrift"],
        "baustart": ["geplanter Baustart"],
        "uebergabe": ["Übergabe"],
        "plz":      ["PLZ"],
        "ort":      ["Ort"],
    })
    z = resolve_cols(con, zgp, {"zp": ["Zählpunktbezeichnung"], "gp": ["GPartner"]})
    m = resolve_cols(con, mpm, {"zp": ["Zählpunktbezeichnung"], "mp": ["MP ID"]})

    def flag(key):
        # the register uses x / X / - / empty / stray junk. Only x|X asserts presence;
        # '-' asserts absence; anything else is genuinely unknown, so keep it NULL
        # rather than silently turning missing paperwork into a negative label.
        col = c[key]
        if not col:
            return "NULL"
        return (f'CASE WHEN lower(trim("{col}")) = \'x\' THEN TRUE '
                f'WHEN trim("{col}") = \'-\' THEN FALSE ELSE NULL END')

    def date(key):
        col = c[key]
        return f"try_strptime(trim(\"{col}\"), ['%d.%m.%Y','%d.%m.%y'])::DATE" if col else "NULL::DATE"

    def txt(key):
        return f'trim("{c[key]}")' if c[key] else "NULL"

    con.execute(f"""
        CREATE OR REPLACE TABLE labels AS
        WITH g AS (
            SELECT trim("{c['gp']}") AS gp,
                   {flag('hp')}      AS has_hp,
                   {flag('pv')}      AS has_pv,
                   {flag('battery')} AS has_battery,
                   {flag('ev')}      AS has_ev,
                   {flag('hp_boil')} AS has_hp_boiler,
                   try_cast(replace(trim("{c['kwp']}"), ',', '.') AS DOUBLE) AS pv_kwp,
                   {date('inbetr')}    AS commissioned,
                   {date('signed')}    AS signed,
                   {date('baustart')}  AS baustart,
                   {date('uebergabe')} AS uebergabe,
                   {txt('plz')}      AS plz,
                   {txt('ort')}      AS ort
            FROM read_csv('{gigi}', delim=';', header=true, all_varchar=true)
            WHERE trim("{c['gp']}") <> ''
        ),
        z AS (
            SELECT trim("{z['zp']}") AS zp, trim("{z['gp']}") AS gp
            FROM read_csv('{zgp}', delim=';', header=true, all_varchar=true)
        ),
        m AS (
            SELECT trim("{m['zp']}") AS zp, CAST(trim("{m['mp']}") AS BIGINT) AS mp_id
            FROM read_csv('{mpm}', delim=';', header=true, all_varchar=true)
        )
        SELECT m.mp_id, z.zp, g.*,
               -- the register is a CURRENT snapshot, with signatures running years past
               -- the measurement window. An asset is only observable in data recorded
               -- after it went live, so join on this, never on the flag alone.
               -- Take the LATEST available date, so an asset is never marked live
               -- before it existed. Measured against InBetrieb-Datum, the other
               -- columns sit at: Uebergabe +42d (after commissioning), geplanter
               -- Baustart -17d, Datum Unterschrift -145d. Coalescing to the
               -- signature date - the old behaviour - was the most optimistic
               -- choice available and marked assets live months too early.
               greatest(g.commissioned, g.uebergabe, g.baustart, g.signed) AS asset_live_from,
               -- true when no real InBetrieb-Datum backs this date
               (g.commissioned IS NULL) AS live_from_estimated,
               CASE WHEN g.commissioned IS NOT NULL THEN 'inbetrieb'
                    WHEN g.uebergabe   IS NOT NULL THEN 'uebergabe'
                    WHEN g.baustart    IS NOT NULL THEN 'baustart'
                    WHEN g.signed      IS NOT NULL THEN 'unterschrift'
                    ELSE 'none' END AS live_from_source
        FROM g JOIN z USING (gp) JOIN m USING (zp)
    """)
    out = os.path.join(args.out, "labels.parquet")
    con.execute(f"COPY labels TO '{out}' (FORMAT parquet)")
    n, mp, live = con.execute("""
        SELECT count(*), count(DISTINCT mp_id), count(*) FILTER (WHERE asset_live_from IS NOT NULL)
        FROM labels""").fetchone()
    print(f"  labels: {n} rows, {mp} distinct MP IDs ({live} with a validity date)")
    print(f"  -> {out}")


def _feature_sql():
    """Per meter/day/direction features, computed on the slot list (no unnest)."""
    day = ",".join(SLOT_COLS[:96])                       # slots 1..96 = the normal day
    l = f"list_transform([{day}], x -> COALESCE(x, 0.0))"
    parts = [
        "list_sum(l)                                        AS total",
        "list_count(list_filter(raw, x -> x IS NOT NULL))   AS n_present",
        "list_max(l)                                        AS vmax",
        "list_aggregate(l, 'avg')                           AS vmean",
        "list_aggregate(l, 'stddev_pop')                    AS vstd",
        "list_aggregate(l, 'quantile_cont', 0.1)            AS p10",
        "list_aggregate(l, 'quantile_cont', 0.5)            AS p50",
    ]
    for name, (a, b) in WINDOWS.items():
        parts.append(f"list_sum(l[{a}:{b}])                 AS {name}_sum")
    for name, thr in EV_BANDS.items():
        parts.append(f"list_count(list_filter(l, x -> x >= {thr})) AS n_{name}")
    # longest sustained block + number of blocks, via run-length on a 0/1 string
    runs = (f"regexp_extract_all(array_to_string("
            f"list_transform(l, x -> CASE WHEN x >= {RUN_THRESHOLD} THEN '1' ELSE '0' END), ''), '1+')")
    parts.append(f"COALESCE(list_max(list_transform({runs}, s -> length(s))), 0) AS max_run")
    parts.append(f"len({runs})                              AS n_runs")
    # sharpest step between consecutive slots - an appliance switching on
    parts.append("list_max(list_transform(range(2, 97), i -> l[i] - l[i-1])) AS step_up_max")
    parts.append("list_min(list_transform(range(2, 97), i -> l[i] - l[i-1])) AS step_dn_max")
    return l, day, parts


def step_features(con, args):
    raw = os.path.join(args.out, "raw", "*", "*.parquet")
    if not glob.glob(raw):
        print("  ! no raw parquet found - run the convert step first")
        return
    l, day, parts = _feature_sql()
    feats = ",\n                   ".join(parts)

    con.execute(f"""
        CREATE OR REPLACE VIEW day_feat AS
        WITH base AS (
            SELECT mp_id, zp, plz, d, ym, obis, n_slots,
                   [{day}] AS raw,
                   {l}     AS l
            FROM read_parquet('{raw}', hive_partitioning=true)
        )
        SELECT mp_id, zp, plz, d, ym, obis,
               dayofweek(d) AS dow, month(d) AS mon,
               -- DST days. AEW's exporter writes a fixed 96-slot grid year-round, so
               -- the fall-back hour is simply absent from the October transition day
               -- and the spring-forward day carries 4 padded slots - neither shows up
               -- as an unusual row width. Flag them by calendar (last Sunday of March
               -- and October) and drop them from anything time-of-day sensitive.
               (n_slots <> 96
                OR (month(d) IN (3, 10)
                    AND dayofweek(d) = 0
                    AND month(d + INTERVAL 7 DAY) <> month(d))) AS dst_day,
               {feats}
        FROM base
    """)

    # pivot import/export side by side: one row per meter-day
    imp = [p.split(" AS ")[-1].strip() for p in parts]
    sel_imp = ", ".join(f"max(CASE WHEN obis = '{OBIS_IMPORT}' THEN {c} END) AS imp_{c}" for c in imp)
    sel_exp = ", ".join(f"max(CASE WHEN obis = '{OBIS_EXPORT}' THEN {c} END) AS exp_{c}" for c in imp)

    out = os.path.join(args.out, "features")
    con.execute(f"""
        COPY (
            SELECT mp_id, any_value(zp) AS zp, any_value(plz) AS plz, d, ym,
                   any_value(dow) AS dow, any_value(mon) AS mon,
                   bool_or(dst_day) AS dst_day,
                   {sel_imp}, {sel_exp}
            FROM day_feat
            GROUP BY mp_id, d, ym
        ) TO '{out}'
          (FORMAT parquet, PARTITION_BY (ym), COMPRESSION zstd, OVERWRITE_OR_IGNORE)
    """)
    print(f"  -> {out}")

    # coverage per meter-month: which meters are usable at all
    cov = os.path.join(args.out, "coverage.parquet")
    con.execute(f"""
        COPY (
            SELECT mp_id, ym,
                   count(*) AS meter_days,
                   sum(imp_n_present) AS vals_present,
                   sum(CASE WHEN imp_n_present = 0 THEN 1 ELSE 0 END) AS empty_days,
                   round(sum(imp_n_present) / (96.0 * count(*)), 4) AS coverage
            FROM (SELECT * FROM read_parquet('{out}/*/*.parquet', hive_partitioning=true))
            GROUP BY mp_id, ym
        ) TO '{cov}' (FORMAT parquet)
    """)
    print(f"  -> {cov}")

    # PV ground truth for free: first month a meter ever exports = commissioning event
    pv = os.path.join(args.out, "pv_export.parquet")
    con.execute(f"""
        COPY (
            WITH m AS (
                SELECT mp_id, ym, sum(exp_total) AS export_kwh, max(exp_vmax) AS export_peak
                FROM read_parquet('{out}/*/*.parquet', hive_partitioning=true)
                GROUP BY mp_id, ym
            )
            SELECT mp_id,
                   min(ym) FILTER (WHERE export_kwh > 1.0) AS first_export_ym,
                   sum(export_kwh)                          AS export_kwh_total,
                   max(export_peak)                         AS export_peak_kwh_15min,
                   count(*) FILTER (WHERE export_kwh > 1.0) AS months_exporting
            FROM m GROUP BY mp_id
            HAVING sum(export_kwh) > 1.0
        ) TO '{pv}' (FORMAT parquet)
    """)
    n = con.execute(f"SELECT count(*) FROM '{pv}'").fetchone()[0]
    print(f"  -> {pv}  ({n} meters with measurable export = derived PV labels)")


def step_validate(con, args):
    """Confront the register against the measurements and report the disagreement.

    Run before anyone trains on these labels. On the 2023-09 extract, most meters
    the register marks '-' for PV are in fact exporting: the register records
    SUBSIDISED assets, so '-' means "no subsidy record", not "no asset". Treating
    it as a negative label poisons training and inflates every accuracy number.
    """
    feats = os.path.join(args.out, "features", "*", "*.parquet")
    labels = os.path.join(args.out, "labels.parquet")
    if not glob.glob(feats) or not os.path.exists(labels):
        print("  ! need the features and labels steps first")
        return

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE meter AS
        SELECT mp_id,
               sum(imp_total) AS imp_kwh,
               sum(exp_total) AS exp_kwh,
               sum(imp_n_present) / (96.0 * count(*)) AS coverage,
               count(*) AS days
        FROM read_parquet('{feats}', hive_partitioning=true)
        GROUP BY mp_id
    """)
    rows = con.execute(f"""
        SELECT l.has_pv,
               count(*) AS meters,
               sum(CASE WHEN m.exp_kwh > 1 THEN 1 ELSE 0 END) AS exporting
        FROM meter m JOIN '{labels}' l USING (mp_id)
        WHERE m.coverage > 0.5
          AND l.asset_live_from < (SELECT min(d) FROM read_parquet('{feats}', hive_partitioning=true))
          AND l.has_pv IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetchall()

    print("  register PV flag vs measured export (meters live before the window):")
    contradictions = 0
    for has_pv, n, exporting in rows:
        print(f"    has_pv={str(has_pv):5} {n:5} meters, {exporting:5} actually exporting")
        if has_pv is False and exporting:
            contradictions = exporting
    if contradictions:
        print(f"  !! {contradictions} meters flagged 'no PV' are exporting. Treat the register's")
        print("     '-' as UNKNOWN, not as a negative. Use measured export as PV ground truth.")

    out = os.path.join(args.out, "label_check.parquet")
    con.execute(f"""
        COPY (
            SELECT l.mp_id, l.has_pv, l.has_ev, l.has_hp, l.has_battery,
                   l.asset_live_from, l.pv_kwp,
                   m.imp_kwh, m.exp_kwh, m.coverage,
                   (l.has_pv = FALSE AND m.exp_kwh > 1) AS pv_flag_contradicted
            FROM meter m JOIN '{labels}' l USING (mp_id)
        ) TO '{out}' (FORMAT parquet)
    """)
    print(f"  -> {out}")


def readings_view(out_dir):
    """SQL that unpivots the wide raw layer to long (mp_id, ts, obis, kwh)."""
    cases = ", ".join(SLOT_COLS)
    return f"""
-- Long view over the wide raw layer. Slot i covers [(i-1)*15min, i*15min) after
-- local midnight, so ts is the interval START, and the source column named
-- '00:00' is the LAST interval of the day, not the first.
-- On DST days (n_slots <> 96) this mapping is approximate - filter them out for
-- anything time-of-day sensitive.
CREATE OR REPLACE VIEW readings AS
SELECT mp_id, zp, plz, obis, d, n_slots,
       d + INTERVAL (15 * (slot_idx - 1)) MINUTE AS ts,
       slot_idx, kwh
FROM (
    SELECT mp_id, zp, plz, obis, d, n_slots,
           unnest(range(1, {MAX_SLOTS + 1})) AS slot_idx,
           unnest([{cases}])                 AS kwh
    FROM read_parquet('{os.path.join(out_dir, "raw", "*", "*.parquet")}',
                      hive_partitioning=true)
)
WHERE kwh IS NOT NULL;
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="directory holding the raw CSVs (searched recursively)")
    ap.add_argument("--out", required=True, help="output directory for parquet + reports")
    ap.add_argument("--steps", default="profile,convert,labels,features,validate")
    ap.add_argument("--memory", default="4GB", help="DuckDB memory limit (keep below the session's RAM)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--temp-dir", default=None, help="spill directory; set it if memory is tight")
    ap.add_argument("--force", action="store_true", help="reconvert files already marked done")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    files = classify(args.input)
    print(f"input: {args.input}")
    for k, v in files.items():
        if v:
            print(f"  {k}: {len(v)} file(s)")
    if not files["load"]:
        sys.exit("no load-profile CSVs found (expected a header containing 'MP ID' and '00:15')")

    con = connect(args)
    for s in steps:
        print(f"\n== {s} ==")
        if s == "profile":
            step_profile(con, files, args)
        elif s == "convert":
            step_convert(con, files, args)
        elif s == "labels":
            step_labels(con, files, args)
        elif s == "features":
            step_features(con, args)
        elif s == "validate":
            step_validate(con, args)
        else:
            sys.exit(f"unknown step: {s}")

    with open(os.path.join(args.out, "readings_view.sql"), "w") as fh:
        fh.write(readings_view(args.out))
    print(f"\ndone. carry back: {args.out}/features, labels.parquet, coverage.parquet, pv_export.parquet")


if __name__ == "__main__":
    main()
