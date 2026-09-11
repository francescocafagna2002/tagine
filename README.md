# Energy Fingerprints

Energy Data Hackdays 2026, Brugg — [challenge page](https://www.energydatahackdays.ch/challenges/energy-fingerprints-what-can-you-learn-from-electricity-data)

Detecting household assets — PV, battery, heat pump, EV charger, electric boiler —
from 15-minute smart-meter data, and measuring honestly how well that works.

Two datasets: **AEW** (82,476 metering points, 3.5 years, labels from a subsidy
register) and **CKW** (1,981 households, 20 months, labels from a household survey
plus MeteoSwiss weather). Detectors are developed and validated on CKW, where a
declared "no" is a real negative, then applied to AEW, where the challenge lives.

## Repository

| path | what it is |
|---|---|
| `prepare_meter_data.py` | raw CSVs → Parquet, plus labels, daily features and a data-quality report. *Formerly `stage1.py`.* |
| `make_training_set.py` | the modelling table: one row per meter and timestamp, labels attached |
| `inspect_training_set.py` | shape, coverage, label counts and plots for that table |
| `evalfw/` | evaluation framework — label tiers, bootstrap intervals, leakage-safe thresholds, fleet prevalence, per-customer reports |
| `detectors/` | the five assets, rules and models behind one interface |
| `app.py` | Streamlit meter inspector — heatmap, 15-minute trace, verdicts vs declared labels |
| `energy_fingerprints_results.pdf` | **the results presentation** — 13 slides, built from `presentation.html` |
| `energy_fingerprints_results.pptx` | the same argument as PowerPoint, without the tool screenshots |
| `energy_fingerprints_label_coverage.pdf` / `.pptx` | the label-coverage analysis, for AEW |

`Data/`, `results*/` and the HTML deck are gitignored — see *What is not in this
repo* at the end.

## Layout in the Renku session

```
~/work/
  aew-data/test-blob/input_data/     # mounted data connector (read-only)
    2023/ 2024/ 2025/ 2026/          #   <Monat JJJJ>/LG_AIM2Hackerdays_kWh_*.csv
    HackDays2026 - GIGI.csv          #   asset register (the labels)
    mpid_zähler_mapping.csv          #   MP ID <-> Zählpunkt
    Zähler-GP.csv                    #   Zählpunkt <-> Geschäftspartner
  store/src/                         # scripts
  store/data/prepared/                 # pipeline output
```

## 1. Convert everything to Parquet

`profile` reads every file and reports its shape without writing bulk data. Run
it first — it is the safety net that catches layout changes.

```bash
python3 ~/work/store/src/prepare_meter_data.py \
  --input ~/work/aew-data/test-blob/input_data \
  --out   ~/work/store/data/prepared \
  --memory 4GB --threads 4 --steps profile
```

Then the conversion itself. Detach it so a session timeout cannot kill it:

```bash
nohup python3 ~/work/store/src/prepare_meter_data.py \
  --input ~/work/aew-data/test-blob/input_data \
  --out   ~/work/store/data/prepared \
  --memory 4GB --threads 4 --temp-dir ~/work/tmp \
  --steps profile,convert > ~/work/prepare.log 2>&1 &
```

```bash
tail -f ~/work/prepare.log
```

Idempotent: converted files are marked in `_done/` and skipped, so re-running the
identical command after a crash resumes where it stopped. Add `--force` to redo them.

Output: `store/data/prepared/raw/ym=YYYY-MM/*.parquet`, one row per
meter/day/direction with 96 slot columns. About **8.6× smaller than the CSV**
(1.1 GB → 128 MB on the 2023 file), and DuckDB prunes unread columns.

## 2. Build the labels

Seconds — it only reads the three small mapping CSVs. Must go to the **same**
`--out` directory as the conversion.

```bash
python3 ~/work/store/src/prepare_meter_data.py \
  --input ~/work/aew-data/test-blob/input_data \
  --out   ~/work/store/data/prepared \
  --steps labels
```

Output: `labels.parquet` — GIGI → GPartner → Zählpunkt → MP ID, one row per
register entry, with commissioning dates.

## 3. Build the modelling table

Full 15-minute history of only the meters the register covers, labels attached.

```bash
python3 ~/work/store/src/make_training_set.py \
  --prepared ~/work/store/data/prepared \
  --out        ~/work/store/data/training_set.parquet \
  --memory 4GB --threads 4 --temp-dir ~/work/tmp
```

Reads Parquet only — never re-parses the CSVs. Useful flags:

| flag | effect |
|---|---|
| `--unambiguous-only` | drop meters whose Geschäftspartner owns several meters (label can't be attributed) |
| `--min-coverage 0.8` | drop meters with sparse readings |

Output: one row per `(meter, timestamp)` with `import_kwh` / `export_kwh` side by
side, asset flags, per-asset commissioning dates, and `*_live` booleans saying
whether each asset existed **at that timestamp**. Plus `meters.parquet`, the
one-row-per-meter version.

## 4. Optional: fleet-wide daily features

Not needed for the labelled table; use it for unsupervised work across all meters.

```bash
python3 ~/work/store/src/prepare_meter_data.py --input ~/work/aew-data/test-blob/input_data \
  --out ~/work/store/data/prepared --steps features,validate
```

Output: `features/ym=*/`, `coverage.parquet`, `pv_export.parquet` (PV labels
derived from measured export), `label_check.parquet`.

## Inspecting a single meter

Two ways, both reading the prepared Parquet directly.

**Interactively** — a heatmap of hour-of-day against date over the whole history,
a 15-minute trace for any window, and every detector's verdict compared against
the declared label:

```bash
streamlit run app.py
```

Find a meter by MP ID, by Geschäftspartner number (resolved through the mapping
files), or browse. `EF_DEFAULT_MP` sets the meter it opens on. Editing anything
under `detectors/` needs the sidebar's **↻ reload detectors** button — Streamlit
reloads `app.py` on save but keeps detector objects cached.

**From the command line**, with the evidence behind each score:

```bash
python3 -m evalfw.explain --raw 'Data/raw/*/*.parquet' --mapping-dir Data --gp <gp>
```

Detectors that cannot apply say so (`needs a full winter and summer`) rather than
returning a misleading zero, and each line carries a confidence tag from that
detector's measured AUC.

## Viewing Parquet

```bash
cd ~/work/store/data/prepared && python3 -c 'import duckdb; duckdb.read_parquet("labels.parquet").show(max_rows=30)'
```

```bash
cd ~/work/store/data/prepared && python3 -c 'import duckdb; duckdb.read_parquet("labels.parquet").write_csv("labels.csv")'
```

`readings_view.sql` in the output dir creates a long `(mp_id, ts, obis, kwh)` view
over the wide raw layer.

---

# Data traps — all verified, not theoretical

**The metadata layout changed between exports.** The 2023 extract has 5 leading
columns (`MP ID; Zählpunktbezeichnung; OBIS-Code; Datum; PLZ`) but its header
names only 4 — the Zählpunkt column is missing from the header. The 2025/2026
exports genuinely have 4 (no Zählpunkt) and an honest header. A fixed schema
mislabels one or the other. `prepare_meter_data.py` anchors on the date column to derive the
metadata width, and names columns by data pattern (`CH1011…` = Zählpunkt,
`1-1:1.29.0*255` = OBIS). **A wrong layout does not raise an error** — it shows up
as dates that won't parse, or worse, as an OBIS column full of Zählpunkt strings
that silently yields zero export readings. `convert` refuses to run if any date
fails to parse.

**OBIS codes:** `1-1:1.29.0*255` = import (consumption), `1-1:2.29.0*255` = export
(PV feed-in). Verified: export peaks midday, ≈0 at night.

**Slot columns run `00:15 … 23:45` then `00:00`** — the last column is the interval
*ending* at midnight, not the first of the day. Slot *i* covers
`[(i-1)·15min, i·15min)` after local midnight.

**DST is lossy at source.** The exporter writes a fixed 96-slot grid year-round, so
on the October fall-back day one hour is simply absent, and the March
spring-forward day carries padding. Confirmed: 26 Oct 2025 rows have 101 fields
like every other day. Neither shows up as an unusual row width, so `dst_day` is
flagged **by calendar** (last Sunday of March/October). Drop those days from
anything time-of-day sensitive.

**Missing data is day-shaped, not scattered.** In 2023-09, 7.05% of values are
NULL — but they are whole missing meter-days: 1,611 meters (5%) had *no* readings
all month and 712 more were under 50% covered. Filter on `coverage.parquet`;
never assume a meter is complete. (März 2026: 4.08%, Okt 2025: 4.76%.)

**Duplicates exist.** März 2026 has 62 rows too many — exactly one meter-month
duplicated on `(mp_id, obis, datum)`. Duplicates double-count energy in every sum
without raising anything. `profile` counts them; `make_training_set.py` dedupes
and reports.

**The meter population grows over time** — 75,955 meters in Oct 2025, 82,476 in
Mar 2026 (smart-meter rollout). A meter appearing for the first time is **not** an
asset installation; any step-change detector must distinguish the two or it will
report thousands of phantom installations.

## The register is not ground truth

`HackDays2026 - GIGI.csv` records **subsidised** installations, so `-` means "no
subsidy record", not "no asset". Measured on 2023-09: of 19 meters flagged
`has_pv = FALSE` and live before the window, **13 were exporting** ~590 kWh that
month. Treat `-` as *unknown*, never as a negative label — `prepare_meter_data.py` keeps
`-` → `FALSE` and blank → `NULL` distinct so you can decide. For PV, **measured
export is better ground truth than the register**: 1,556 meters exported in
2023-09 alone.

It is also a *current* snapshot, with signature dates running to 2026 while the
measurements may be years earlier. Always gate on commissioning: of 59 PV-flagged
meters in 2023-09, 10 had no panels yet and produced essentially nothing. That is
what `*_live` is for.

Finally, **labels attach to a Geschäftspartner, not a meter.** Most own one meter,
but some own several (one owns 17) — for those, "has a heat pump" doesn't say
which meter. `n_meters_for_gp` exposes it; `--unambiguous-only` filters it.

## Heat pumps in multi-family buildings are invisible

A customer's meter can only show what that meter measures. In a block of flats the
central heat pump has **its own metering point, registered to the building owner**
— so a tenant's meter shows no heating at all, however the asset records describe
the property.

Worked through on one partner: their meter draws 2,148 kWh/yr, peaks at 4.2 kW and
uses *more* in summer than winter. Six units along the same block, on consecutive
Anlage numbers, sits a meter drawing 27,462 kWh/yr, peaking at 21.6 kW, with a
winter/summer ratio of **5.75** — 4,641 kWh in January against 897 in August, and
5–9 kW running through winter nights. That is the building's heat pump, and it
belongs to a different Geschäftspartner.

**8,742 of 73,470 meters (12%)** look like plant meters on this test — strongly
seasonal and above 15,000 kWh/yr. Every flat behind one of them is a household
whose meter can never show heating.

So *"does this customer have a heat pump"* is unanswerable from a tenant's meter,
and arguably the wrong question; the answerable version is building-level. The
Anlage numbering looks like the way in — a plant's Anlage sits inside the
contiguous run of its building's unit Anlagen.

This does **not** contaminate the evaluation set: of the 68 heat-pump-labelled
meters, none are too small to contain one and the median is 6,574 kWh/yr, because
subsidy recipients are owners of single-family homes rather than tenants. It is a
limit on fleet *coverage*, not on the measured numbers.

## Scale reference

| | value |
|---|---|
| 2023-09 extract | 33,188 meters, 1.99M rows, 1.1 GB CSV → 128 MB Parquet |
| 2025/2026 exports | ~76k–82k meters, ~5M rows/month |
| register coverage | 878 partners → 413 meters (85 present in the 2023-09 subset) |
| conversion speed | ~18 s per month (4 threads, 4 GB) |

---

# How many meters can we actually train on?

Short answer: **208 meters with usable labels**, out of 1,192 register rows. Here is
where each drop happens — all figures computed from the delivered files.

## Where it starts — the register lists installations, not households

| step | count | change | why |
|---|---:|---:|---|
| GIGI rows | 1,192 | | |
| with a GP-Nr | 1,167 | −25 | blank partner number |
| unique Geschäftspartner | **878** | −289 | 289 rows are *repeat installations* by the same customer (e.g. PV 2019 + battery 2023). One household, several rows — not a loss of information. |

The headline "1,192" is 1,192 subsidised installations belonging to **878 customers**.

## The bottleneck — linking customers to meters

| step | count | change | why |
|---|---:|---:|---|
| partners found in `Zähler-GP.csv` | **337** | **−541 (62%)** | see below |
| their Zählpunkte | 413 | +76 | some partners own several meters |
| with an MP ID | 413 | 0 | `mpid_zähler_mapping.csv` is complete |

**541 of 878 subsidised customers have no meter listed in `Zähler-GP.csv`.** This is
not a formatting problem: the unmatched partner numbers use the same 6-digit format
in the same numeric range (10179–920285) as the matched ones, and the file covers
76,718 partners. Those customers are simply absent from the extract.

Likely cause: AEW subsidises assets across a wider area than it operates the grid
for (energy supplier vs. distribution system operator), or the extract was limited
to smart-metered customers. **Recovering these links is the single highest-value
ask** — 2.6× the labels for zero modelling effort.

## The last cut — meeting the measurements

| step | count | change | why |
|---|---:|---:|---|
| present in the load data | 411 | −2 | |
| with ≥1 year of readings | 281 | −130 | smart-meter rollout — 45 meters first appear in March 2026, 196 cover under half the window |
| unambiguous (partner owns exactly 1 meter) | **208** | −73 | for multi-meter partners the register says the *household* has a heat pump, not which meter |

## What is trainable, per asset

Among those 208 meters:

| asset | positives | register-negatives |
|---|---:|---:|
| PV | 173 | 35 |
| battery | 142 | 66 |
| heat pump | 68 | 140 |
| EV charger | 48 | 160 |

Register-negatives are unreliable — `-` means "no subsidy record", not "no asset"
(see *The register is not ground truth* above).

## What this means for method

**Heat pumps (68) and EV chargers (48): validate, do not fit.** A classifier trained
on 48 EV meters memorises them. Build a physically-motivated detector — sustained
3.7 / 7.4 / 11 kW charging blocks, thermostat cycling — and use those 48 meters to
*measure* it. It generalises to all 82,476 meters instead of the 208, and it is a
better story than an overfitted accuracy number.

**PV: skip the register.** Measured export labels the whole fleet for free, and dates
commissioning to within days. 295 of these 411 meters export at some point; across
the full fleet it is thousands. The register only adds self-consumption-only systems
that never feed in — a real but small niche.

---

# Detector results

Five assets, two datasets, every number out-of-fold with a bootstrap 95% interval.
Detectors are compared against a random-scoring baseline that must land at AUC 0.5;
where it does not, the harness is misaligned and nothing else can be trusted.

## Where each asset is best measured

| asset | best detector | AUC | dataset | why there |
|---|---|---|---|---|
| **Battery** | `battery_dark_zero` (rule) | **0.939** [0.893, 0.974] | AEW | only AEW records batteries |
| **Electric heating** | `gbm_electric_heating` | **0.823** [0.787, 0.859] | CKW | needs declared heating type |
| **Heat pump** | `gbm_hp` | **0.817** [0.777, 0.855] | CKW | needs non-electric negatives |
| **PV** | `pv_export_total` (rule) | **0.814** [0.766, 0.858] | CKW | AEW's PV labels derive from export, so scoring there is circular |
| **EV** | `gbm_ev` | **0.728** [0.684, 0.771] | CKW | AEW has no usable EV negatives |
| **Boiler** | `gbm_boiler` | **0.692** [0.647, 0.734] | CKW | only CKW asks about hot water |

Random baselines: 0.487–0.545 everywhere. `ev_peak_band` sits at **0.349** — reliably
worse than chance, kept on the board as a lesson and flagged `anti_predictive` so no
report reads it as evidence (it rewards peaks *near* a wallbox rating, but EV
households simply peak higher).

## Rules against models

Paired bootstrap on the same meters, which is far more sensitive than comparing two
intervals:

| asset | rule | model | difference | verdict |
|---|---|---|---|---|
| PV | 0.814 | 0.797 | −0.017 [−0.022, +0.053] | not distinguishable |
| heat pump | 0.798 | 0.817 | +0.019 [−0.016, +0.052] | not distinguishable |
| electric heating | 0.802 | 0.823 | +0.021 [−0.009, +0.052] | not distinguishable |
| EV | 0.714 | 0.728 | +0.014 [−0.032, +0.063] | not distinguishable |
| battery | 0.939 | 0.840 | −0.099 | **rule wins** |
| boiler | 0.606 | 0.692 | +0.086 [+0.037, +0.142] | **model wins** |

**A physically-motivated rule matches a gradient-boosted model on four of six
targets, beats it on one, and loses only on the asset with no clean mechanism.**
That is the central methodological result. Where a mechanism exists — a battery
holding net import at zero after dark, heating load tracking outdoor temperature —
one feature derived from it is worth more than a dozen weak ones. Where it does not,
as with resistance water heating, only a model over many weak cues gets anywhere.

The EV row is the clearest illustration. An earlier version of `ev_sustained_blocks`
counted charging blocks and the model beat it significantly (+0.071, p=0.007).
Adding a start-time irregularity term — a car is plugged in when the driver gets
home, a timer fires at the same minute — closed the gap entirely.

## What each detector keys on

| detector | signal |
|---|---|
| `battery_dark_zero` | share of fully dark summer hours with net import under 0.02 kWh. Battery owners 92.4%, non-owners 0.0% |
| `hp_temp_slope` | overnight consumption regressed on heating degree hours; the slope is heat loss divided by COP |
| `pv_export_total` | measured export. Sustained export *is* generation |
| `ev_sustained_blocks` | wallbox-band blocks weighted by start-time **ir**regularity |
| `boiler_night_band` | share of summer-night slots above 1.8 kW |

## Caveats that belong with any number above

**The evaluation populations are not the fleet.** AEW's labelled meters are a
subsidy register — 84% PV, 64% battery. CKW's are consenting survey respondents —
72% solar, 77% heat pump. Both over-represent asset owners enormously. Run
`evalfw.fleet` before quoting a prevalence for a real network.

**Assets mask each other.** A battery holds net import at zero after dark, erasing
the evidence a boiler detector reads: the same boiler rule implies 48.5% prevalence
among AEW meters without a battery and 29.8% among those with one. The effect is
asset-specific — severe for boilers, absent for the heat-pump seasonal ratio, since
a 10 kWh battery covers a 0.3 kW overnight baseline but not a 3 kW heat pump.
Detection order is part of the method: identify batteries first, then the rest.

**Power alone cannot separate an EV from a boiler.** Swiss 300 L Elektroboiler ship
at 3.0, 4.0 and 6.0 kW, overlapping the wallbox band (3.4–12.8 kW) almost entirely.
Timing does the work, not magnitude.

**Time base matters.** AEW stores fixed CET year-round, CKW stores UTC; neither is
local wall-clock. Clock-driven loads therefore appear at two different slots across
the year, smearing every time-of-day feature. `evalfw/timebase.py` converts at load
time — on one meter it moved start-time concentration from 0.72 to 0.88.

---

# What is not in this repo

Three things are deliberately absent, all for the same reason: a multi-year
15-minute load curve is personally attributable. It reveals occupancy, working
hours, holidays and absence, and this challenge is explicitly about analysing it
*while preserving privacy*.

| excluded | why |
|---|---|
| `Data/` | the AEW and CKW extracts, 11 GB. Customer data. |
| `results*/` | per-meter scores keyed by real MP IDs — derived customer data, one step removed |
| `presentation.html`, `energy_fingerprints_results.pdf` | both embed screenshots of one real meter's 3.5-year profile beside its MP ID. Fine to present, or to hand to AEW and the jury directly; not something to put in a repo that may become public. Regenerate the screenshots from an anonymised meter to commit them. `energy_fingerprints_results.pptx` carries the same argument without screenshots and *is* committed. |

No meter, Zählpunkt or Geschäftspartner numbers appear anywhere in the source.
`app.py` defaults to whichever meter has the longest history; set `EF_DEFAULT_MP`
or `EF_DEFAULT_GP` for a demo.

To reproduce everything, point `prepare_meter_data.py` at the AEW input folder and
work forward through the sections above.
