#!/usr/bin/env python3
"""T08 (a)–(c): classical baselines, fit on train, evaluate once on test.

  python scripts/t08_baselines.py --dataset putra

(a) URL-feature gradient boosting, (b) TF-IDF + LR over the serialised S1
state, (c) hand-written rule set mirroring the bank (no fitting; threshold on
the rule score fitted on train). Records training and inference time per site
on CPU. (d) ModernBERT is scripts/t08d_encoder.py (GPU).
Outputs results/t08_<dataset>_baselines.csv, results/t08_<dataset>_test_scores.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _bootstrap import load_config, add_common_args, add_split_args, sites_for, banner, REPO
sys.path.insert(0, str(REPO / "src"))
from jev.run import load_state  # noqa: E402
from eval.metrics import detection, best_f1_threshold  # noqa: E402
from baselines.classical import url_features, make_url_gbm, make_tfidf_lr, serialise_state, rules  # noqa: E402
from combine.combiners import cv_scores  # noqa: E402


def collect(cfg, dataset, split, variant, limit):
    sites, source = sites_for(cfg, dataset, split, limit=limit, reason="t08")
    rows = []
    for s in sites:
        st = load_state(cfg, dataset, variant, s.id)
        if st is None:
            continue
        rows.append({"site_id": s.id, "label": s.label, "url": s.url, "state": st})
    return pd.DataFrame(rows), source


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--variant", default="S1")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["SEED"]
    tr, src_tr = collect(cfg, args.dataset, "train", args.variant, args.limit)
    te, src_te = collect(cfg, args.dataset, "test", args.variant, args.limit)
    args.split = "train/test"; banner("t08", args, list(range(len(tr) + len(te))), src_te)
    ytr, yte = tr.label.values.astype(int), te.label.values.astype(int)
    out, scores = [], te[["site_id", "label"]].copy()

    def record(name, feats, fit_s, infer_s, p, oof=None):
        t_fit, _ = best_f1_threshold(ytr, oof) if oof is not None else (0.5, None)
        d = detection(yte, p, t_fit)
        out.append({"baseline": name, "features": feats, "n_train": len(ytr), "n_test": len(yte),
                    "cv_auroc_train": roc_auc_score(ytr, oof) if oof is not None else np.nan,
                    "threshold_fit_on_train": t_fit, "test_f1_at_fit": d["f1"], "test_precision": d["precision"],
                    "test_recall": d["recall"], "test_auroc": d["auroc"], "test_auprc": d["auprc"],
                    "train_seconds": fit_s, "infer_ms_per_site_cpu": 1000 * infer_s / max(1, len(yte)),
                    "usd_per_1000_sites": 0.0, "split_source": src_te})
        scores[f"p_{name}"] = p

    # (a) URL features + GBM
    Xtr = pd.DataFrame([url_features(u) for u in tr.url]); Xte = pd.DataFrame([url_features(u) for u in te.url])
    oof = cv_scores(lambda: make_url_gbm(seed), Xtr, ytr, seed=seed)
    t0 = time.perf_counter(); m = make_url_gbm(seed).fit(Xtr, ytr); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); p = m.predict_proba(Xte)[:, 1]; inf_s = time.perf_counter() - t0
    record("url_gbm", f"{Xtr.shape[1]} URL features", fit_s, inf_s, p, oof)

    # (b) TF-IDF over serialised S1 + LR
    dtr = [serialise_state(s) for s in tr.state]; dte = [serialise_state(s) for s in te.state]
    oof = cv_scores(lambda: make_tfidf_lr(seed), pd.Series(dtr), ytr, seed=seed) if len(ytr) >= 10 else None
    t0 = time.perf_counter(); m = make_tfidf_lr(seed).fit(dtr, ytr); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); p = m.predict_proba(dte)[:, 1]; inf_s = time.perf_counter() - t0
    record("tfidf_lr", f"word 1-2 grams of serialised {args.variant}", fit_s, inf_s, p, oof)

    # (c) rule set
    t0 = time.perf_counter(); rtr = pd.DataFrame([rules(s) for s in tr.state]); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); rte = pd.DataFrame([rules(s) for s in te.state]); inf_s = time.perf_counter() - t0
    record("rule_set", "13 deterministic rules mirroring bank_v1 (score = share of rules firing)", 0.0, inf_s,
           rte.rule_score.values, rtr.rule_score.values)
    per_rule = pd.DataFrame({"rule": [c for c in rte.columns if c.startswith("r_")],
                             "auroc_test": [roc_auc_score(yte, rte[c]) if rte[c].nunique() > 1 else np.nan for c in rte.columns if c.startswith("r_")],
                             "fire_rate_phish": [rte.loc[yte == 1, c].mean() for c in rte.columns if c.startswith("r_")],
                             "fire_rate_benign": [rte.loc[yte == 0, c].mean() for c in rte.columns if c.startswith("r_")]})
    res = pd.DataFrame(out)
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    res.to_csv(cfg.results_root / f"t08_{args.dataset}_baselines.csv", index=False)
    scores.to_csv(cfg.results_root / f"t08_{args.dataset}_test_scores.csv", index=False)
    per_rule.to_csv(cfg.results_root / f"t08_{args.dataset}_rules.csv", index=False)
    print(res.round(3).to_string(index=False)); print(per_rule.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
