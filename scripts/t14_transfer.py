#!/usr/bin/env python3
"""T14 transfer regimes on a PhreshPhish sample (S1, q_direct and bank_v1).

  python scripts/t14_transfer.py --dataset phreshphish-20k --putra-train results/t07_putra_S1_train_table.csv

R0 frozen: Putra-trained LR combiner and Putra-fitted thresholds applied as is.
R1 threshold-only recalibration on --r1 labelled sites (default 200).
R2 combiner refit on --r2 sites (500). R3 refit on --r3 sites (5,000).
v2.3: the adaptation sites come from --adapt-dataset (phreshphish-adapt, drawn
from the TRAIN split minus dev ids; the three sets are disjoint slices of a
seeded permutation) and the whole evaluation sample is scored. Without
--adapt-dataset (development only) they are held out of the sample itself.
Per-language breakdown for languages with >= --min-lang sites. Outputs
results/t14_<dataset>_<variant>_{table,regimes,per_language,lr_odds_ratios,scores}.csv
(odds ratios of the R3 combiner; per-site scores of q_direct and the R0 lr for T21/T_STATS).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _bootstrap import load_config, add_common_args, add_split_args, sites_for, banner, REPO
sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config  # noqa: E402
from jev.run import run_questions, save_table  # noqa: E402
from questions.bank import load_bank, bank_questions  # noqa: E402
from eval.metrics import detection, best_f1_threshold  # noqa: E402
from combine.combiners import make_lr, fit_apply, question_columns, lr_odds_ratios  # noqa: E402
from eval.timing import TimingLog  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.set_defaults(split="all")
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--putra-train", default=None, help="t07 train table of the source domain (for R0)")
    ap.add_argument("--r1", type=int, default=200); ap.add_argument("--r2", type=int, default=500); ap.add_argument("--r3", type=int, default=5000)
    ap.add_argument("--min-lang", type=int, default=200)
    ap.add_argument("--adapt-dataset", default=None, help="dataset holding the R1/R2/R3 adaptation sites (phreshphish-adapt)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sites, src = sites_for(cfg, args.dataset, None if args.split == "all" else args.split, limit=args.limit, reason="t14")
    tag = f"t14_{args.dataset}_{args.variant}"
    client = client_from_config(cfg, tag, use_cache=not args.no_cache)
    banner("t14", args, sites, src, client)
    bank = load_bank(cfg.bank_path)
    timing = TimingLog(cfg.timing_root / f"{tag}.csv")
    df = run_questions(cfg, client, sites, args.dataset, args.variant, bank_questions(bank, args.variant), tag, args.split, src,
                       workers=args.workers, timing=timing)
    save_table(cfg, df, f"{tag}_table")
    df = df[df.ok].reset_index(drop=True)
    lang = pd.Series({s.id: s.language for s in sites})
    df["language"] = df.site_id.map(lang).fillna("unk")
    cols = ["q_direct"] + question_columns(df)
    y = df.label.values.astype(int)
    rng = np.random.default_rng(cfg["SEED"])
    adapt_df = None
    if args.adapt_dataset:
        a_sites, a_src = sites_for(cfg, args.adapt_dataset, None, reason="t14 adaptation sets")
        adapt_df = run_questions(cfg, client, a_sites, args.adapt_dataset, args.variant, bank_questions(bank, args.variant),
                                 f"t14_{args.adapt_dataset}_{args.variant}", "all", a_src, workers=args.workers, timing=timing)
        save_table(cfg, adapt_df, f"t14_{args.adapt_dataset}_{args.variant}_table")
        adapt_df = adapt_df[adapt_df.ok].reset_index(drop=True)
        aperm = rng.permutation(len(adapt_df))
        print(f"[t14] adaptation sites from {args.adapt_dataset}: {len(adapt_df)} (R1/R2/R3 = disjoint slices of {args.r1}/{args.r2}/{args.r3})")
    else:
        print("[t14] no --adapt-dataset: adaptation sites are held out of the evaluation sample (development only)")
    perm = rng.permutation(len(df))

    rows, scores = [], {}
    # R0 frozen
    if args.putra_train and (REPO / args.putra_train).exists():
        src_tr = pd.read_csv(REPO / args.putra_train); src_tr = src_tr[src_tr.ok]
        scols = [c for c in cols if c in src_tr.columns]
        m, p = fit_apply(make_lr, src_tr[scols].astype(float), src_tr.label.values.astype(int), df[scols].astype(float))
        t_dir = best_f1_threshold(src_tr.label, src_tr.q_direct)[0]
        t_lr = best_f1_threshold(src_tr.label, m.predict_proba(src_tr[scols].astype(float))[:, 1])[0]
        for name, pp, t in (("q_direct", df.q_direct.values.astype(float), t_dir), ("lr_bank", p, t_lr)):
            rows.append({"regime": "R0 frozen", "score": name, "n_eval": len(df), "n_adapt": 0, "threshold": t, **detection(y, pp, t)})
            scores[("R0", name)] = pp
    else:
        print("[t14] no --putra-train table: R0 reported with threshold 0.5 and no source-trained combiner")
        rows.append({"regime": "R0 frozen (0.5)", "score": "q_direct", "n_eval": len(df), "n_adapt": 0, "threshold": 0.5, **detection(y, df.q_direct, 0.5)})
    # R1–R3
    offset = 0
    r3_model = None
    for name, k, refit in (("R1 threshold", args.r1, False), ("R2 refit", args.r2, True), ("R3 refit", args.r3, True)):
        if adapt_df is not None:
            k = min(k, max(0, len(adapt_df) - offset))
            if k < 10:
                continue
            A = adapt_df.iloc[aperm[offset:offset + k]]; offset += k
            E = df
        else:
            k = min(k, len(df) // 2)
            if k < 10:
                continue
            A, E = df.iloc[perm[:k]], df.iloc[perm[k:]]
        ya, ye = A.label.values.astype(int), E.label.values.astype(int)
        if len(set(ya)) < 2 or len(set(ye)) < 2:
            continue
        acols = [c for c in cols if c in A.columns]
        t = best_f1_threshold(ya, A.q_direct.values.astype(float))[0]
        rows.append({"regime": name, "score": "q_direct", "n_eval": len(E), "n_adapt": k, "threshold": t,
                     **detection(ye, E.q_direct.values.astype(float), t)})
        if refit:
            m, p = fit_apply(make_lr, A[acols].astype(float), ya, E[acols].astype(float))
            t = best_f1_threshold(ya, m.predict_proba(A[acols].astype(float))[:, 1])[0]
            rows.append({"regime": name, "score": "lr_bank", "n_eval": len(E), "n_adapt": k, "threshold": t, **detection(ye, p, t)})
            r3_model = (m, acols)
    reg = pd.DataFrame(rows); reg["split_source"] = src
    save_table(cfg, reg, f"{tag}_regimes")
    print(reg.round(3).to_string(index=False))
    if r3_model is not None:                      # indicator ordering on PhreshPhish (Table 9c)
        save_table(cfg, lr_odds_ratios(*r3_model), f"{tag}_lr_odds_ratios")
    sc = df[["site_id", "label", "language", "q_direct"]].copy()
    if ("R0", "lr_bank") in scores:
        sc["p_lr_R0"] = scores[("R0", "lr_bank")]
    sc["split_source"] = src
    save_table(cfg, sc, f"{tag}_scores")

    # per language (q_direct, threshold 0.5 and AUROC)
    pl = []
    for lg, g in df.groupby("language"):
        if len(g) < args.min_lang:
            continue
        pl.append({"language": lg, "n": len(g), "phishing": int(g.label.sum()),
                   "auroc_q_direct": roc_auc_score(g.label, g.q_direct) if g.label.nunique() == 2 else np.nan,
                   "f1_q_direct_at_0.5": detection(g.label, g.q_direct, 0.5)["f1"],
                   "mean_p_phish": float(g.q_direct[g.label == 1].mean()), "mean_p_benign": float(g.q_direct[g.label == 0].mean())})
    save_table(cfg, pd.DataFrame(pl), f"{tag}_per_language")
    if pl:
        print(pd.DataFrame(pl).round(3).to_string(index=False))
    timing.close(); client.close()


if __name__ == "__main__":
    main()
