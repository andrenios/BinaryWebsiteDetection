#!/usr/bin/env python3
"""T13: calibration and low-prevalence operation.

  python scripts/t13_calibration.py --dataset putra --variant S1

Scores: q_direct (from the T07 train/test tables) and the LR-combined score.
Platt and isotonic fitted on train (out-of-fold for LR); reliability diagrams
(10 equal-mass bins), ECE, Brier on test. PPV at 0.1 / 1 / 5 % prevalence with
the paper-2 resampling protocol and bootstrap CIs; false positives per 1,000
benign at 95 % recall. Also accepts extra score tables with --extra name=path:col
(gate+Jev, cascades, baselines, paper-2 rows).
Outputs results/t13_<dataset>_{calibration,ppv,fp95,reliability_*}.csv, figures/t13_*.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from _bootstrap import load_config, add_common_args, REPO
sys.path.insert(0, str(REPO / "src"))
from eval.metrics import ece_equal_mass, brier, ppv_at_prevalence, fp_per_1000_benign_at_recall, best_f1_threshold, detection  # noqa: E402
from eval.plots import reliability  # noqa: E402
from combine.combiners import Platt, Isotonic, make_lr, cv_scores, fit_apply, question_columns  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dataset", default="bootstrap-phreshphish")
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--extra", nargs="*", default=[], help="name=path:col additional test scores (no calibration fit)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    R = cfg.results_root
    tag = f"t13_{args.dataset}_{args.variant}"
    tr = pd.read_csv(R / f"t07_{args.dataset}_{args.variant}_train_table.csv"); tr = tr[tr.ok]
    te = pd.read_csv(R / f"t07_{args.dataset}_{args.variant}_test_table.csv"); te = te[te.ok]
    ytr, yte = tr.label.values.astype(int), te.label.values.astype(int)
    cols = ["q_direct"] + [c for c in question_columns(tr) if c in te.columns]
    scores = {"q_direct": (tr.q_direct.values.astype(float), te.q_direct.values.astype(float))}
    oof = cv_scores(make_lr, tr[cols].astype(float), ytr, seed=cfg["SEED"])
    _, p_lr = fit_apply(make_lr, tr[cols].astype(float), ytr, te[cols].astype(float))
    scores["lr_bank"] = (oof, p_lr)
    extras = {}
    for e in args.extra:
        name, rest = e.split("="); path, col = rest.rsplit(":", 1)
        d = pd.read_csv(path).set_index("site_id").reindex(te.site_id)
        extras[name] = d[col].values.astype(float)

    cal_rows, ppv_rows, fp_rows = [], [], []
    for name, (ptr, pte) in scores.items():
        variants = {"raw": (None, pte), "platt": (Platt().fit(ptr, ytr), None), "isotonic": (Isotonic().fit(ptr, ytr), None)}
        for cname, (cal, p) in variants.items():
            p = p if p is not None else cal.predict(pte)
            ptr_c = ptr if cal is None else cal.predict(ptr)
            ece, table = ece_equal_mass(yte, p)
            table.to_csv(R / f"{tag}_reliability_{name}_{cname}.csv", index=False)
            reliability(table, cfg.figures_root / f"{tag}_reliability_{name}_{cname}", title=f"{name} {cname} ({args.dataset})")
            t_fit = best_f1_threshold(ytr, ptr_c)[0]
            det = detection(yte, p, t_fit)
            cal_rows.append({"score": name, "calibration": cname, "n_test": len(yte), "ece_10bins": ece, "brier": brier(yte, p),
                             "threshold_fit_on_train": t_fit, "f1_at_fit": det["f1"], "auroc": det["auroc"],
                             "mean_p_phish": float(p[yte == 1].mean()), "mean_p_benign": float(p[yte == 0].mean())})
            for prev in cfg.get("prevalences", [0.001, 0.01, 0.05]):
                ppv_rows.append({"score": name, "calibration": cname, **ppv_at_prevalence(yte, p, t_fit, prev, seed=cfg["SEED"])})
            fp_rows.append({"score": name, "calibration": cname, **fp_per_1000_benign_at_recall(yte, p, 0.95)})
    for name, p in extras.items():
        m = ~np.isnan(p)
        t_fit = 0.5
        ece, table = ece_equal_mass(yte[m], p[m])
        cal_rows.append({"score": name, "calibration": "as given", "n_test": int(m.sum()), "ece_10bins": ece, "brier": brier(yte[m], p[m]),
                         "threshold_fit_on_train": t_fit, **{k: v for k, v in detection(yte[m], p[m], t_fit).items() if k in ("f1", "auroc")},
                         "mean_p_phish": float(p[m][yte[m] == 1].mean()), "mean_p_benign": float(p[m][yte[m] == 0].mean())})
        for prev in cfg.get("prevalences", [0.001, 0.01, 0.05]):
            ppv_rows.append({"score": name, "calibration": "as given", **ppv_at_prevalence(yte[m], p[m], t_fit, prev, seed=cfg["SEED"])})
        fp_rows.append({"score": name, "calibration": "as given", **fp_per_1000_benign_at_recall(yte[m], p[m], 0.95)})
    src = te.split_source.iloc[0] if "split_source" in te else ""
    for nm, rows in (("calibration", cal_rows), ("ppv", ppv_rows), ("fp95", fp_rows)):
        d = pd.DataFrame(rows); d["split_source"] = src
        d.to_csv(R / f"{tag}_{nm}.csv", index=False)
        print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
