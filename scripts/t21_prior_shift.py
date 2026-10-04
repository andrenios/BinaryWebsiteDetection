#!/usr/bin/env python3
"""T21: label-free threshold transfer under prevalence shift (RQ4, method M3).

  python scripts/t21_prior_shift.py --dataset putra --variant S1 \
      --targets phreshphish-native,phreshphish-20k --adapt-dataset phreshphish-adapt [--instantiation twin-<model>]

No API calls (cached bank responses only). The lr combiner (q_direct + the
indicators, as T07) is fitted on the source training split; its out-of-fold
scores are calibrated on train (--calibration, default Platt; T13 feeds T21),
which gives posteriors under the source prior pi_s = 0.5 (Putra is balanced).
On every target sample:
  R0           the Putra threshold (best F1 on the calibrated train scores) applied as is
  prior_known  logit shift by log(pi/(1-pi)) - log(pi_s/(1-pi_s)) with the target's
               true prior; threshold re-derived without labels for the same
               false-positive rate as the Putra threshold has on Putra train,
               from the expected FPR of the shifted posteriors
               (sum_{p_i >= t}(1-p_i) / sum_i (1-p_i))
  prior_known_bayes  the same shift with the Putra posterior threshold kept
  prior_em     as prior_known, with the prior estimated by Saerens-Latinne-
               Decaestecker EM on the unlabelled target scores
  prior_em_bayes     as prior_known_bayes with the EM prior
  R1           threshold refitted on --r1 (200) labelled sites from --adapt-dataset
               (train split); without an adaptation set they are held out of the
               target sample (development only)
ECE before / after per-indicator isotonic maps fitted on the source train split
and applied to the target (combiner refitted on the mapped source features).

Output: results/t21_prior_shift.csv (paper/tables.md Table 17 columns first,
then target, score, n, split_source), per-site results/t21_<target>_scores.csv.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from _bootstrap import load_config, add_common_args, add_split_args, banner, REPO, decision_table, out_name
sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config  # noqa: E402
from jev.run import save_table  # noqa: E402
from questions.bank import load_bank, bank_questions  # noqa: E402
from eval.metrics import detection, best_f1_threshold, ece_equal_mass  # noqa: E402
from combine.combiners import make_lr, cv_scores, question_columns, Platt, Isotonic  # noqa: E402
from combine.prior_shift import prior_shift, em_prior, threshold_for_expected_fpr, source_fpr_at, PerIndicatorIsotonic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--targets", default="phreshphish-native,phreshphish-20k", help="comma list of dataset[:split][@variant]")
    ap.add_argument("--target-variant", default=None, help="state variant of the target tables (default: --variant)")
    ap.add_argument("--adapt-dataset", default=None, help="labelled adaptation sites for R1 (phreshphish-adapt)")
    ap.add_argument("--r1", type=int, default=200)
    ap.add_argument("--calibration", default="platt", choices=["none", "platt", "isotonic"])
    ap.add_argument("--source-prior", type=float, default=0.5)
    ap.add_argument("--instantiation", default="jev")
    args = ap.parse_args()
    cfg = load_config(args.config)
    inst, seed = args.instantiation, cfg["SEED"]
    client = client_from_config(cfg, f"t21_{args.dataset}", use_cache=not args.no_cache) if inst == "jev" else None
    bank = load_bank(cfg.bank_path)
    qs = bank_questions(bank, args.variant)
    real = args.dataset == "putra" and args.variant == "S1"

    tr, src_tr = decision_table(cfg, client, args.dataset, args.variant, "train", qs, f"t07_{args.dataset}_{args.variant}_train", inst, args.limit, args.workers, "t21 source train")
    args.split = "train"; banner("t21", args, tr, src_tr, client)
    if not len(tr):
        raise SystemExit("[t21] no source train table")
    cols = ["q_direct"] + question_columns(tr)
    ytr = tr.label.values.astype(int)
    oof = cv_scores(make_lr, tr[cols].astype(float), ytr, seed=seed)
    model = make_lr().fit(tr[cols].astype(float), ytr)
    cal_cls = {"none": None, "platt": Platt, "isotonic": Isotonic}[args.calibration]
    cal = cal_cls().fit(oof, ytr) if cal_cls else None
    calib = (lambda p: cal.predict(p)) if cal else (lambda p: np.asarray(p, float))
    p_tr = calib(oof)
    t_src = best_f1_threshold(ytr, p_tr)[0]
    fpr_src = source_fpr_at(p_tr, ytr, t_src)
    print(f"[t21] source threshold {t_src:.3f} (FPR on source train {fpr_src:.4f}), calibration {args.calibration}, pi_s {args.source_prior}")
    # per-indicator isotonic transfer (ECE after)
    iso = PerIndicatorIsotonic().fit(tr[cols], ytr)
    oof2 = cv_scores(make_lr, iso.transform(tr[cols]), ytr, seed=seed)
    model2 = make_lr().fit(iso.transform(tr[cols]), ytr)
    cal2 = cal_cls().fit(oof2, ytr) if cal_cls else None
    calib2 = (lambda p: cal2.predict(p)) if cal2 else (lambda p: np.asarray(p, float))

    adapt = None
    if args.adapt_dataset:
        adapt, _ = decision_table(cfg, client, args.adapt_dataset, args.variant, None, qs, f"t14_{args.adapt_dataset}_{args.variant}", inst, args.limit, args.workers, "t21 R1 adaptation")
        rng = np.random.default_rng(seed)
        adapt = adapt.iloc[rng.permutation(len(adapt))[:args.r1]]
        print(f"[t21] R1 adaptation sites from {args.adapt_dataset}: {len(adapt)}")

    rows = []
    for spec in args.targets.split(","):
        spec, _, tv = spec.partition("@")
        ds, _, sp = spec.partition(":")
        tv = tv or args.target_variant or args.variant
        try:
            tt, tsrc = decision_table(cfg, client, ds, tv, sp or None, bank_questions(bank, tv), f"t14_{ds}_{tv}", inst, args.limit, args.workers, "t21 target")
        except SystemExit as e:
            print(f"[t21] target {ds}: {e}; skipped"); continue
        if not len(tt):
            print(f"[t21] target {ds}: empty table; skipped"); continue
        tcols = [c for c in cols if c in tt.columns]
        if len(tcols) < len(cols):
            print(f"[t21] target {ds}: missing {set(cols) - set(tcols)}; skipped"); continue
        y = tt.label.values.astype(int)
        if len(set(y)) < 2:
            print(f"[t21] target {ds}: a single class ({len(y)} rows); skipped"); continue
        p = calib(model.predict_proba(tt[cols].astype(float))[:, 1])
        p_after = calib2(model2.predict_proba(iso.transform(tt[cols]))[:, 1])
        pi_true = float(y.mean())
        em = em_prior(p, args.source_prior)
        ece_before = ece_equal_mass(y, p)[0]; ece_after = ece_equal_mass(y, p_after)[0]
        # R1: labelled adaptation sites (adapt set) or a held-out slice of the target (development only)
        if adapt is not None:
            pa = calib(model.predict_proba(adapt[cols].astype(float))[:, 1]); ya = adapt.label.values.astype(int)
            ev = np.ones(len(y), bool)
        else:
            rng = np.random.default_rng(seed); perm = rng.permutation(len(y)); k = min(args.r1, len(y) // 2)
            pa, ya = p[perm[:k]], y[perm[:k]]; ev = np.zeros(len(y), bool); ev[perm[k:]] = True
        t_r1 = best_f1_threshold(ya, pa)[0] if len(set(ya)) == 2 else t_src
        methods = {
            "R0": (p, t_src, np.nan),
            "prior_known": (prior_shift(p, pi_true, args.source_prior), None, np.nan),
            "prior_known_bayes": (prior_shift(p, pi_true, args.source_prior), t_src, np.nan),
            "prior_em": (prior_shift(p, em["pi_hat"], args.source_prior), None, em["pi_hat"]),
            "prior_em_bayes": (prior_shift(p, em["pi_hat"], args.source_prior), t_src, em["pi_hat"]),
            "R1": (p, t_r1, np.nan),
        }
        per_site = pd.DataFrame({"site_id": tt.site_id, "label": y, "p_cal": p, "p_after_isotonic": p_after,
                                 "p_shift_known": methods["prior_known"][0], "p_shift_em": methods["prior_em"][0]})
        for name, (pp, t, pi_hat) in methods.items():
            if t is None:
                t = threshold_for_expected_fpr(pp, fpr_src)
            m = ev if name == "R1" else np.ones(len(y), bool)
            d = detection(y[m], pp[m], t)
            rows.append({"method": name, "prevalence_true": pi_true, "prevalence_estimated": pi_hat, "threshold": t,
                         "f1": d["f1"], "precision": d["precision"], "recall": d["recall"], "auroc": d["auroc"],
                         "ece_before": ece_before, "ece_after": ece_after,
                         "target": ds + (f":{sp}" if sp else ""), "score": f"lr_bank ({args.calibration})", "n": d["n"], "n_pos": d["n_pos"],
                         "n_adapt": len(ya) if name == "R1" else 0, "em_iterations": em["iterations"] if "em" in name else np.nan,
                         "target_fpr": fpr_src, "split_source": tsrc})
            per_site[f"thr_{name}"] = t
        save_table(cfg, per_site, out_name(f"t21_{ds}_scores", inst))
        r0, r1, pe = (next(r["f1"] for r in rows if r["target"].startswith(ds) and r["method"] == m) for m in ("R0", "R1", "prior_em"))
        share = (pe - r0) / (r1 - r0) if r1 != r0 else np.nan
        print(f"[t21] {ds}: pi_true {pi_true:.3f}, pi_em {em['pi_hat']:.3f} ({em['iterations']} it.), F1 R0 {r0:.3f} -> EM {pe:.3f} -> R1 {r1:.3f}; "
              f"share of the R1 gain recovered without labels {share:.2f}")
    out = pd.DataFrame(rows)
    name = "t21_prior_shift" if real else f"t21_{args.dataset}_{args.variant}_prior_shift"
    save_table(cfg, out, out_name(name, inst))
    if len(out):
        print(out[["target", "method", "prevalence_true", "prevalence_estimated", "threshold", "f1", "precision", "recall", "auroc", "ece_before", "ece_after"]]
              .round(3).to_string(index=False))
    if client:
        client.close()


if __name__ == "__main__":
    main()
