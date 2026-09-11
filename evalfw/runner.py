#!/usr/bin/env python3
"""
Run every registered detector over the labelled meters and produce a leaderboard.

    python3 -m evalfw.runner --set training_set_v2.parquet --out results

What it guarantees:
  * thresholds are chosen on TRAINING folds only, split by Geschaeftspartner so
    one partner's meters never straddle train and test
  * every headline number carries a bootstrap CI - with 48 EV positives, two
    detectors under ~12 recall points apart are indistinguishable
  * precision is printed ONLY where trustworthy negatives exist. For heat pumps
    and EVs the register's dashes are not negatives, so the honest pair is
    recall (measurable) plus flag rate over the unlabelled pool (an upper bound
    on the false-positive rate, and an estimate of prevalence among them)
"""

import argparse
import importlib
import inspect
import pkgutil
import os
import sys
import numpy as np
import pandas as pd
import duckdb

from . import labels as labelmod
from . import ckw as ckwmod
from .timebase import localize
from . import metrics as M

# Assumed prevalence among meters the register does not cover, used ONLY to fix a
# comparable operating point where no trustworthy negatives exist. These are
# assumptions, not measurements - override with --prior hp=0.15,ev=0.08 and say
# which you used when reporting. Rough Swiss household shares, 2026.
PRIORS = {"hp": 0.20, "ev": 0.10, "battery": 0.05, "pv": 0.15}

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def discover(package="detectors"):
    from detectors.base import Detector
    found = []
    pkg = importlib.import_module(package)
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name == "base":
            continue
        m = importlib.import_module(f"{package}.{mod.name}")
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if issubclass(obj, Detector) and obj is not Detector and obj.__module__ == m.__name__:
                try:
                    found.append(obj())               # no-arg detectors
                except TypeError:
                    pass                              # needs args, e.g. RandomScore
    from detectors.baselines import RandomScore
    from . import ckw as _ckw
    assets = sorted(set(labelmod.ASSETS) | set(_ckw.ASSETS))
    found += [RandomScore(a) for a in assets]
    try:
        from detectors.ml import GBMShape, HAVE_SKLEARN
        if HAVE_SKLEARN:
            found += [GBMShape(a) for a in assets]
        else:
            print("  (sklearn not installed - ML detectors skipped)")
    except ImportError:
        pass
    return found


def _aew_loader(ts_path):
    def load(con, ids):
        lst = ",".join(str(int(x)) for x in ids)
        return localize(con.execute(f"""
            SELECT mp_id, ts, d, import_kwh, export_kwh, NULL::DOUBLE AS temp_c, dst_day
            FROM read_parquet('{ts_path}')
            WHERE mp_id IN ({lst})
            ORDER BY mp_id, ts
        """).df(), source="CET")
    return load


def score_all(loader, detectors, meter_ids, batch=40):
    """Returns (scores, features). Rule detectors produce scores directly;
    trainable ones produce features, which the caller turns into out-of-fold
    scores after fitting per fold."""
    """Score every meter with every detector, streaming meters in batches so
    memory stays flat regardless of how long the history is."""
    con = duckdb.connect()
    rows, frows = [], []
    ids = list(meter_ids)
    for i in range(0, len(ids), batch):
        df = loader(con, ids[i:i + batch])
        if df.empty:
            continue
        df["hour"] = df.ts.dt.hour
        for mp, g in df.groupby("mp_id", sort=False):
            for det in detectors:
                if getattr(det, "trainable", False):
                    try:
                        fv = det.featurize(mp, g)
                    except Exception as e:
                        fv = {"_error": 1.0}
                        print(f"  ! {det.name} featurize failed on {mp}: {e}")
                    frows.append({"mp_id": mp, "detector": det.name,
                                  **{f"f_{k}": v for k, v in fv.items()}})
                    continue
                try:
                    sc, ev = det.score(mp, g)
                except Exception as e:                # a broken detector must not
                    sc, ev = float("nan"), {"error": str(e)[:80]}   # kill the run
                rows.append({"mp_id": mp, "detector": det.name, "asset": det.asset,
                             "score": sc, "evidence": str(ev)[:300]})
        print(f"  scored {min(i+batch, len(ids))}/{len(ids)} meters", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(frows)


def evaluate(lab, sc, prior=0.15, k=5, n_boot=2000):
    """One detector on one asset -> a row of metrics."""
    d = lab.merge(sc, on="mp_id", how="left")
    y, s, gp = d.y.to_numpy(float), d.score.to_numpy(float), d.gp.to_numpy()
    scored = ~np.isnan(s)
    has_neg = (y == 0).sum() > 0

    # Threshold from TRAINING folds only. With gold negatives, maximise F1; with
    # none, pin the operating point to an assumed prevalence instead - an F1
    # search over contaminated labels just flags everything.
    thrs = []
    for tr, _ in M.grouped_folds(gp, k=k, seed=7):
        m = tr[~np.isnan(s[tr])]
        if not len(m):
            continue
        if has_neg:
            thrs.append(M.best_threshold(y[m], s[m]))
        else:
            thrs.append(M.threshold_for_prevalence(s[m][np.isnan(y[m])], prior))
    thr = float(np.nanmedian(thrs)) if thrs else float("nan")

    ev = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
          "n_unlabelled": int(np.isnan(y).sum()),
          "coverage": round(float(scored.mean()), 3), "threshold": thr}

    lbl = ~np.isnan(y) & scored
    if has_neg:
        ev["auc"] = M.bootstrap_ci(M.roc_auc, y[lbl], s[lbl], n=n_boot)
        ev["ap"] = M.bootstrap_ci(M.average_precision, y[lbl], s[lbl], n=n_boot)
    else:
        ev["auc"] = ev["ap"] = (float("nan"),) * 3

    pos = (y == 1) & scored
    ev["recall"] = M.bootstrap_ci(lambda yy, ss: M.recall_at(yy, ss, thr),
                                  y[pos | (y == 0)], s[pos | (y == 0)], n=n_boot) \
        if pos.sum() else (float("nan"),) * 3
    ev["precision"] = (M.bootstrap_ci(lambda yy, ss: M.precision_at(yy, ss, thr),
                                      y[lbl], s[lbl], n=n_boot)
                       if has_neg else (float("nan"),) * 3)
    unl = np.isnan(y) & scored
    ev["flag_rate_unlabelled"] = M.flag_rate(s[unl], thr) if unl.sum() else float("nan")
    ev["lift"] = (M.bootstrap_ci(lambda yy, ss: M.lift_at(yy, ss, thr), y, s, n=n_boot)
                  if unl.sum() else (float("nan"),) * 3)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True,
                    help="AEW training_set parquet, or CKW 15MinValues parquet")
    ap.add_argument("--dataset", choices=["aew", "ckw"], default="aew")
    ap.add_argument("--profiles", default=None, help="CKW Homeprofiles.csv")
    ap.add_argument("--weather", default=None, help="CKW Weather_Data.csv (optional)")
    ap.add_argument("--sample", type=int, default=0,
                    help="score only this many meters (0 = all); for quick iteration")
    ap.add_argument("--out", default="results")
    ap.add_argument("--min-days", type=int, default=365)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--assets", default=",".join(labelmod.ASSETS))
    ap.add_argument("--stratify", default="",
                    help="also report within strata, e.g. 'pv' or 'battery'. The "
                         "labelled set is a SUBSIDY register - 83%% have PV, 46%% a "
                         "battery - so seasonal and night-time features measure "
                         "storage behaviour unless you control for it")
    ap.add_argument("--prior", default="",
                    help="override assumed prevalence, e.g. hp=0.15,ev=0.08")
    ap.add_argument("--reuse-scores", action="store_true",
                    help="reuse results/scores.parquet instead of rescoring")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    priors = dict(PRIORS)
    for kv in filter(None, args.prior.split(",")):
        k_, v_ = kv.split("=")
        priors[k_.strip()] = float(v_)
    ckw = args.dataset == "ckw"
    if ckw and not args.profiles:
        raise SystemExit("--dataset ckw needs --profiles Homeprofiles.csv")
    if args.assets == ",".join(labelmod.ASSETS) and ckw:
        args.assets = ",".join(ckwmod.ASSETS)          # CKW has different targets
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    dets = [d for d in discover() if d.asset in assets]
    print(f"{len(dets)} detectors: " + ", ".join(sorted(d.name for d in dets)))

    if ckw:
        lab_by_asset = {a: ckwmod.build_labels(args.profiles, a, args.set, args.min_days)
                        for a in assets}
        loader = lambda con, ids: localize(
            ckwmod.load_batch(con, args.set, ids, args.weather), source="UTC")
    else:
        lab_by_asset = {a: labelmod.build(args.set, a, args.min_days) for a in assets}
        loader = _aew_loader(args.set)
    for a in assets:
        print("  " + labelmod.describe(lab_by_asset[a], a))

    meters = sorted(set().union(*[set(l.mp_id) for l in lab_by_asset.values()]))
    if args.sample and len(meters) > args.sample:
        rng = np.random.default_rng(3)
        meters = sorted(rng.choice(meters, args.sample, replace=False).tolist())
        print(f"sampling {len(meters)} meters (--sample)")
    scores_path = os.path.join(args.out, "scores.parquet")
    feats_path = os.path.join(args.out, "features.parquet")
    if args.reuse_scores and os.path.exists(scores_path):
        scores = pd.read_parquet(scores_path)
        feats = (pd.read_parquet(feats_path) if os.path.exists(feats_path)
                 else pd.DataFrame(columns=["mp_id", "detector"]))
        # the cache is keyed by file, not by which detectors were in it - so a
        # reuse after adding a detector must top up rather than silently report
        # zero coverage for the new one
        cached = set(scores.detector)
        missing = [d for d in dets if d.name not in cached]
        have_meters = set(scores.mp_id)
        new_meters = [m for m in meters if m not in have_meters]
        print(f"reusing {scores_path} ({len(cached)} detectors cached)")
        if missing or new_meters:
            if missing:
                print(f"  scoring {len(missing)} uncached detector(s): "
                      + ", ".join(d.name for d in missing))
            add = []
            add, addf = [], []
            if missing:
                s_, f_ = score_all(loader, missing, meters, batch=25 if ckw else 40)
                add.append(s_); addf.append(f_)
            if new_meters:
                s_, f_ = score_all(loader, dets, new_meters, batch=25 if ckw else 40)
                add.append(s_); addf.append(f_)
            scores = pd.concat([scores] + [a for a in add if not a.empty],
                               ignore_index=True) \
                       .drop_duplicates(["mp_id", "detector"], keep="last")
            addf = [a for a in addf if not a.empty]
            if addf:
                feats = pd.concat([feats] + addf, ignore_index=True) \
                          .drop_duplicates(["mp_id", "detector"], keep="last")
                feats.to_parquet(feats_path, index=False)
            scores.to_parquet(scores_path, index=False)
    else:
        print(f"\nscoring {len(meters)} meters ...")
        scores, feats = score_all(loader, dets, meters, batch=25 if ckw else 40)
        scores.to_parquet(scores_path, index=False)
        if not feats.empty:
            feats.to_parquet(feats_path, index=False)

    # composition of the evaluation set - print it before any metric, because it
    # bounds what the metrics can be trusted to mean
    if ckw:
        print(f"\nevaluation set composition: {ckwmod.composition(args.profiles)}")
        print("  survey-declared labels, so negatives are REAL. Still a consenting,")
        print("  eco-engaged sample - check fleet prevalence before quoting a rate.\n")
        comp = None
    else:
      comp = duckdb.connect().execute(f"""
        SELECT count(*) AS meters,
               round(100.0*sum(has_pv::INT)/count(*),0)      AS pct_pv,
               round(100.0*sum(has_battery::INT)/count(*),0) AS pct_battery,
               round(100.0*sum(has_hp::INT)/count(*),0)      AS pct_hp
        FROM (SELECT DISTINCT mp_id, has_pv, has_battery, has_hp
              FROM read_parquet('{args.set}'))
      """).fetchone()
    if comp:
        print(f"\nevaluation set composition: {comp[0]} meters - "
              f"{comp[1]:.0f}% PV, {comp[2]:.0f}% battery, {comp[3]:.0f}% heat pump")
        print("  this is a SUBSIDY register, not a random sample of the fleet. Features")
        print("  that interact with generation or storage are confounded here; use")
        print("  --stratify pv to check a detector is not just re-detecting PV.\n")

    strata = [("all", None)]
    strat_col = args.stratify
    night_dets = [d for d in dets if getattr(d, "night_based", False)]
    auto = False
    if night_dets and not strat_col:
        # A battery holds net import flat at zero after dark, which is exactly
        # where these detectors read. It does not add noise, it erases evidence -
        # so never report a single pooled number for them over a mixed population.
        strat_col, auto = "battery", True
        print(f"  {len(night_dets)} night-based detector(s) present -> stratifying by "
              f"battery ownership")
        print("  (a battery zeroes net import after dark, erasing the signal these read)")
    if strat_col:
        try:
            if ckw:
                prof = ckwmod._profiles(args.profiles)
                if strat_col in prof.columns:
                    col = prof[strat_col].astype(str)
                    strata += [(f"{strat_col}=1", set(prof.location_profile_id[col == "true"])),
                               (f"{strat_col}=0", set(prof.location_profile_id[col == "false"]))]
                elif auto:
                    print(f"  (CKW has no '{strat_col}' field - reporting pooled only)")
                else:
                    print(f"  ! CKW has no survey field '{strat_col}'")
            else:
                c = f"has_{strat_col}"
                s = duckdb.connect().execute(
                    f"SELECT DISTINCT mp_id, {c} FROM read_parquet('{args.set}')").df()
                strata += [(f"{strat_col}=1", set(s[s[c] == True].mp_id)),
                           (f"{strat_col}=0", set(s[s[c] == False].mp_id))]
        except Exception as e:
            print(f"  ! could not stratify by {strat_col}: {e}")

    # --- trainable detectors: fit inside folds, score only held-out meters ---
    trainables = [d for d in dets if getattr(d, "trainable", False)]
    for det in trainables:
        F = feats[feats.detector == det.name] if not feats.empty else pd.DataFrame()
        if F.empty:
            print(f"  ! {det.name}: no features computed"); continue
        cols = [c for c in F.columns if c.startswith("f_")]
        lab = lab_by_asset[det.asset]
        m = lab.merge(F[["mp_id"] + cols], on="mp_id", how="inner")
        m = m[~m.y.isna()].reset_index(drop=True)
        if len(m) < 50 or m.y.nunique() < 2:
            print(f"  ! {det.name}: too few labelled meters to fit ({len(m)})"); continue
        X = m[cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        y = m.y.to_numpy(int)
        oof = np.full(len(m), np.nan)
        for tr, te in M.grouped_folds(m.gp, k=5, seed=7):
            if len(np.unique(y[tr])) < 2:
                continue
            det.feature_names = cols
            det.fit(X[tr], y[tr])
            oof[te] = det.predict(X[te])
        scores = pd.concat([scores, pd.DataFrame({
            "mp_id": m.mp_id, "detector": det.name, "asset": det.asset,
            "score": oof, "evidence": "out-of-fold, 5 grouped folds"})],
            ignore_index=True)
        print(f"  {det.name}: fitted on {len(m)} labelled meters, "
              f"{int((~np.isnan(oof)).sum())} out-of-fold scores")

    # Coverage must be measured against the meters actually ATTEMPTED, not the
    # whole label set - otherwise --sample makes every detector look like it
    # failed on 94% of meters.
    attempted = set(scores.mp_id)
    rows = []
    for stratum, keep in strata:
      for a in assets:
        lab = lab_by_asset[a]
        lab = lab[lab.mp_id.isin(attempted)]
        if keep is not None:
            lab = lab[lab.mp_id.isin(keep)]
            if (lab.y == 1).sum() < 8:
                continue
        for det in [d for d in dets if d.asset == a]:
            sc = scores[scores.detector == det.name][["mp_id", "score"]]
            r = evaluate(lab, sc, prior=priors.get(a, 0.15), n_boot=args.bootstrap)
            rows.append({"stratum": stratum, "asset": a, "detector": det.name, **{
                "n_pos": r["n_pos"], "n_neg": r["n_neg"], "n_unlabelled": r["n_unlabelled"],
                "coverage": r["coverage"], "threshold": round(r["threshold"], 4)
                            if not np.isnan(r["threshold"]) else None,
                "auc": r["auc"][0], "auc_lo": r["auc"][1], "auc_hi": r["auc"][2],
                "ap": r["ap"][0], "ap_lo": r["ap"][1], "ap_hi": r["ap"][2],
                "recall": r["recall"][0], "recall_lo": r["recall"][1], "recall_hi": r["recall"][2],
                "precision": r["precision"][0], "precision_lo": r["precision"][1],
                "precision_hi": r["precision"][2],
                "flag_rate_unlabelled": r["flag_rate_unlabelled"],
                "lift": r["lift"][0], "lift_lo": r["lift"][1], "lift_hi": r["lift"][2],
                # circular only against labels built from the same signal: AEW's PV
                # labels come from export, CKW's from a survey, so the export
                # baseline is tautological on one and legitimate on the other
                "circular": getattr(det, "circular", False) and not ckw,
            }})
            rr = rows[-1]
            print(f"\n[{stratum}][{a}] {det.name}")
            print(f"   n: {rr['n_pos']} pos / {rr['n_neg']} neg / {rr['n_unlabelled']} unlabelled"
                  f"   coverage {rr['coverage']:.2f}")
            if rr["n_neg"]:
                print(f"   AUC       {M.fmt(rr['auc'], rr['auc_lo'], rr['auc_hi'])}"
                      f"   AP {M.fmt(rr['ap'], rr['ap_lo'], rr['ap_hi'])}")
                print(f"   precision {M.fmt(rr['precision'], rr['precision_lo'], rr['precision_hi'])}", end="")
            else:
                print("   AUC/precision: not computable - no trustworthy negatives", end="")
            print(f"   recall {M.fmt(rr['recall'], rr['recall_lo'], rr['recall_hi'])}")
            if not np.isnan(rr["flag_rate_unlabelled"]):
                print(f"   flags {rr['flag_rate_unlabelled']:.1%} of unlabelled   "
                      f"lift {M.fmt(rr['lift'], rr['lift_lo'], rr['lift_hi'], 2)}")
            if rr["circular"]:
                print("   !! CIRCULAR: uses the same signal the labels are built from. "
                      "It defines the label, it cannot be scored against it.")

    out = pd.DataFrame(rows).sort_values(["stratum", "asset", "recall"],
                                        ascending=[True, True, False])
    p = os.path.join(args.out, "leaderboard.csv")
    out.to_csv(p, index=False)
    print(f"\n-> {p}")

    # how much does battery ownership cost the night-based detectors?
    if strat_col == "battery" and any(s[0] == "battery=1" for s in strata):
        print("\n== battery masking: night-based detectors, by battery ownership ==")
        any_row = False
        for det in night_dets:
            rows_d = {r_["stratum"]: r_ for r_ in rows if r_["detector"] == det.name}
            a, b = rows_d.get("battery=0"), rows_d.get("battery=1")
            if not a or not b:
                continue
            any_row = True
            metric = "auc" if not np.isnan(a.get("auc", np.nan)) else "recall"
            va, vb = a.get(metric), b.get(metric)
            if va is not None and vb is not None and not (np.isnan(va) or np.isnan(vb)):
                print(f"  {det.name:24} {metric:6} without battery {va:.3f}  "
                      f"with battery {vb:.3f}   delta {vb - va:+.3f}")
            # Flag rate over unlabelled meters is the more sensitive view, and the
            # only one available for an asset with no labels on this dataset. It is
            # where the effect showed up first: transferring the boiler rule to AEW
            # gave 48.5% without a battery against 29.8% with one.
            fa, fb = a.get("flag_rate_unlabelled"), b.get("flag_rate_unlabelled")
            if fa is not None and fb is not None and not (np.isnan(fa) or np.isnan(fb)):
                print(f"  {det.name:24} {'flags':6} without battery {fa:.1%}  "
                      f"with battery {fb:.1%}   delta {fb - fa:+.1%}")
        if any_row:
            print("  A negative delta is the battery erasing the dark-hour evidence, not")
            print("  the detector failing. Detect batteries first (they are the easiest")
            print("  asset here), then exclude or report those meters separately.")
        else:
            print("  (not enough labelled meters in both strata to compare)")

    # head-to-head, where negatives make AUC meaningful
    print("\n== paired comparison (same meters, so far more sensitive than two CIs) ==")
    for a in assets:
        lab = lab_by_asset[a]
        if (lab.y == 0).sum() == 0:
            print(f"  {a}: skipped - no gold negatives")
            continue
        sub = out[(out.asset == a) & (out.stratum == "all")].dropna(subset=["auc"]) \
                 .sort_values("auc", ascending=False)
        if len(sub) < 2:
            continue
        a1, a2 = sub.iloc[0].detector, sub.iloc[1].detector
        m = lab.merge(scores[scores.detector == a1][["mp_id", "score"]], on="mp_id") \
               .merge(scores[scores.detector == a2][["mp_id", "score"]], on="mp_id",
                      suffixes=("_a", "_b"))
        m = m[~m.y.isna() & ~m.score_a.isna() & ~m.score_b.isna()]
        if len(m) < 10:
            continue
        diff, lo, hi, p_ = M.paired_bootstrap(M.roc_auc, m.y.to_numpy(float),
                                              m.score_b.to_numpy(float),
                                              m.score_a.to_numpy(float), n=args.bootstrap)
        verdict = "significant" if (lo > 0 or hi < 0) else "NOT distinguishable"
        print(f"  {a}: {a1} vs {a2}  dAUC {diff:+.3f} [{lo:+.3f},{hi:+.3f}]  p={p_:.3f}  {verdict}")


if __name__ == "__main__":
    main()
