# Evaluation framework

Compare asset-detection approaches on a common footing, with honest uncertainty.

```bash
python3 -m evalfw.runner --set Data/training_set_v2.parquet --out results
python3 -m evalfw.runner --set Data/training_set_v2.parquet --out results \
    --assets hp --stratify pv --reuse-scores
```

Outputs `results/leaderboard.csv` (one row per stratum × asset × detector) and
`results/scores.parquet` (per-meter scores and evidence, reusable via
`--reuse-scores` so you only rescore when a detector changes).

## Two datasets

```bash
# AEW - subsidy register labels
python3 -m evalfw.runner --set Data/training_set_v2.parquet --out results

# CKW - household survey labels + MeteoSwiss weather
python3 -m evalfw.runner --dataset ckw --set Data/DataCKW/15MinValues.parquet \
    --profiles Data/DataCKW/Homeprofiles.csv --weather Data/DataCKW/Weather_Data.csv \
    --out results_ckw
```

`--sample N` scores a subset for quick iteration. `--reuse-scores` reuses cached
per-meter scores and tops up any detector or meter the cache is missing.

The adapter (`evalfw/ckw.py`) splits CKW's single signed `kWh` into
`import_kwh` / `export_kwh` and joins hourly temperature as `temp_c`, so every
detector written against the AEW schema runs on CKW unchanged.

**Why CKW matters.** AEW's labels come from a subsidy register, so a missing entry
means "no subsidy record", not "no asset" — heat pumps and EVs had positives but
no usable negatives, which capped every metric at lift-only reporting. CKW's come
from a household survey, so `electric_car: false` is a real declared negative and
AUC, precision and PR curves become computable. The survey also checks out against
the meter: `solar_installed=true` households export a median 12,561 kWh over 20
months, `false` households 12 kWh.

CKW targets are `pv`, `ev`, `hp` and `electric_heating`. `hp` is heat pump vs
**non-electric** heating; resistance heating is excluded from both classes rather
than lumped in, and `electric_heating` is the combined, honestly-detectable
target. The seasonal swing separates them anyway — resistance heating shows a
winter/summer night ratio of 10.2 against a heat pump's 3.5, which is what a COP
of 3-4 predicts.

Note `circular` is a property of the detector *and* the label source:
`pv_export_total` is tautological against AEW's export-derived PV labels but a
legitimate independent signal against CKW's survey labels, so the flag is applied
per dataset.

## Results on CKW (600 households sampled)

| asset | detector | AUC | vs random |
|---|---|---|---|
| electric heating | **gbm_electric_heating** | **0.823** [0.787, 0.859] | 0.545 |
| heat pump | **gbm_hp** | **0.817** [0.777, 0.855] | 0.520 |
| heat pump | hp_temp_slope (rule) | 0.798 [0.753, 0.838] | |
| PV | **pv_export_total** (rule) | **0.814** [0.766, 0.858] | 0.505 |
| EV | **gbm_ev** | **0.728** [0.684, 0.771] | 0.496 |
| EV | ev_sustained_blocks (rule) | 0.714 [0.673, 0.754] | |
| boiler | **gbm_boiler** | **0.692** [0.647, 0.734] | 0.487 |
| boiler | boiler_night_band (rule) | 0.606 [0.554, 0.653] | |
| EV | ev_peak_band | 0.349 — *anti-predictive* | |

On AEW: `battery_dark_zero` **0.939** [0.893, 0.974] against a random 0.520.

Paired comparisons put rule and model level on PV, heat pump, electric heating and
EV; the rule wins outright on battery (0.939 vs 0.840); the model wins only on
boiler (+0.086 [+0.037, +0.142], p<0.001). The full table and its reading are in
the project README.

## Threshold objective

`best_threshold` maximises **Youden's J**, not F1. These gold sets are heavily
one-sided — 78% of CKW heating labels are positive — and maximising F1 there flags
every meter, reporting precision equal to the base rate and recall of 1.000, which
says nothing. J stays informative at any class balance.

## Adding a detector

Drop a class into `detectors/`:

```python
from .base import Detector

class MyDetector(Detector):
    name, asset = "ev_my_idea", "ev"
    def score(self, mp_id, df):
        return float(...), {"why": "..."}     # continuous, higher = more likely
```

`df` is one meter's history: `ts, d, hour, import_kwh, export_kwh, dst_day`. The
runner discovers it automatically. Return a **continuous** score — threshold-free
metrics need the ordering, and the runner picks cutoffs itself. `evidence` is a
small dict naming what triggered it; the challenge grades explainability, and it
makes two detectors disagreeing about a meter debuggable.

## Design decisions, and why

**Label quality differs per asset, so the metrics do too.** PV has real ground
truth from measured export (206 positive / 75 negative). Heat pumps and EVs have
register positives but no trustworthy negatives — a register dash means "no
subsidy record", not "no asset". So for those, precision and AUC are **not
printed at all**. The honest pair is recall (measurable) plus flag rate over the
unlabelled pool (an upper bound on false positives, and an estimate of prevalence
among them), combined as **lift = recall / flag-rate**. Random scores 1.0 by
construction, so lift says how much better than chance a ranking is without
claiming a precision that cannot be supported.

**Thresholds come from training folds only**, split by Geschäftspartner so one
partner's meters never straddle train and test. Where gold negatives exist the
cutoff maximises F1; where none do it is pinned to an assumed prevalence
(`PRIORS` in `runner.py`, override with `--prior hp=0.15`) — an F1 search over
contaminated labels just flags everything, which is how the first version of this
came to report a 75% flag rate.

**Every headline number carries a bootstrap CI.** With 35 EV positives a recall of
0.75 has a 95% interval of roughly ±0.14; two detectors less than ~14 points apart
are indistinguishable. Head-to-head comparison uses a **paired** bootstrap on the
same meters, which resolves differences two independent intervals never could.

**Two baselines exist to police the harness, not to win.** `random_*` must land at
AUC ≈ 0.5 and lift ≈ 1.0 — if it doesn't, scores and labels are misaligned and
every other number is worthless. `pv_export_total` is marked `circular = True`:
the PV labels are themselves built from export, so it reads 1.000 and means
nothing. Its real job is to manufacture labels for the fleet.

## Read the composition line before the metrics

The runner prints it first, and it bounds what everything below can mean:

> 411 meters — 84% PV, 64% battery, 32% heat pump

This is a **subsidy register, not a random sample of the fleet.** Consequences
found the hard way:

- A whole-day winter/summer import ratio reads ~10 for heat-pump and non-heat-pump
  meters alike, because PV self-consumption drives summer *net import* toward zero.
  In this population that feature measures PV, not heating. `hp_winter_ratio` now
  uses 22:00–06:00 only, where no generation contributes.
- Summer overnight import sits near 0.12 kW even for meters with no heat pump —
  home batteries discharging. Any night-time feature is partly measuring storage.

Use `--stratify pv` (or `battery`) to check a detector is not simply re-detecting
generation. `hp_winter_ratio` goes from lift 1.19 overall to 1.59 among PV owners.

## Current state of the baselines

All shipped detectors sit at or near chance for heat pumps and EVs. That is the
starting line, not a result: they are deliberately simple, and the framework
exists to tell you honestly whether something beats them. The PV side works —
`pv_midday_dip` reaches AUC 0.942 [0.911, 0.970] *without using export at all*,
which is the interesting case, since it can find self-consumption-only systems
the export baseline is blind to by construction.

## Fleet-wide prevalence check

```bash
python3 -m evalfw.fleet --raw '~/work/store/data/prepared/raw/*/*.parquet' \
    --sample 2000 --leaderboard results/leaderboard.csv --out results
```

Runs each detector at its leaderboard threshold over a random sample of the whole
fleet — read from the raw layer, so the full converted history, not the labelled
extract — and reports the adoption rate it implies, with a Wilson interval.

This is the only check computed on a population resembling the one a detector will
actually be deployed against. The labelled set is a subsidy register (84% PV, 64%
battery); the fleet is not. It caught `random_pv` implying **98.6%** PV adoption —
its threshold was fitted on a set where 83% of labels are positive, so on real
meters it flags nearly everything. Nothing in-sample would have revealed that.

`coverage` is reported alongside: a detector needing a full winter and summer
scores nothing on meters with short histories, and a prevalence computed from 5%
of the sample is not a prevalence.

**A plausible rate is necessary, not sufficient.** Where no gold negatives exist
the threshold is pinned to an assumed prevalence, so every detector — `random_*`
included — reproduces that prior on the fleet by construction. Read it together
with lift: this check rules detectors **out**, it never rules one in.

Expected ranges in `EXPECTED` are rough Swiss household shares, i.e. assumptions.
Override with `--expect hp=0.12:0.28` and replace them with AEW's own figures
before quoting a verdict — the register cannot supply them, since it only sees
subsidised installations.
