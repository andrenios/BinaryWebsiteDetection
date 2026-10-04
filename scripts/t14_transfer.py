#!/usr/bin/env python3
"""T14 transfer regimes on a PhreshPhish sample (S1, q_direct and bank_v1).

  python scripts/t14_transfer.py --dataset phreshphish-20k --putra-train results/t07_putra_S1_train_table.csv

R0 frozen: Putra-trained LR combiner and Putra-fitted thresholds applied as is.
R1 threshold-only recalibration on --r1 labelled sites (default 200).
R2 combiner refit on --r2 sites (500). R3 refit on --r3 sites (5,000).
Recalibration/refit sites are drawn from the sample and excluded from the
evaluation rows. Per-language breakdown for languages with >= --min-lang sites.
Outputs results/t14_<dataset>_{table,regimes,per_language}.csv.
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
from combine.combiners import make_lr, fit_apply, question_columns  # noqa: E402
from eval.timing import TimingLog  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.set_defaults(split="all")
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--putra-train", default=None, help="t07 train table of the source domain (for R0)")
    ap.add_argument("--r1", type=int, default=200); ap.add_argument("--r2", type=int, default=500); ap.add_argument("--r3", type=int, default=5000)
    ap.add_argument("--min-lang", type=int, default=200)
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
    for name, k, refit in (("R1 threshold", args.r1, False), ("R2 refit", args.r2, True), ("R3 refit", args.r3, True)):
        k = min(k, len(df) // 2)
        if k < 10:
            continue
        adapt, ev = perm[:k], perm[k:]
        ya, ye = y[adapt], y[ev]
        if len(set(ya)) < 2 or len(set(ye)) < 2:
            continue
        pd_ = df.q_direct.values.astype(float)
        rows.append({"regime": name, "score": "q_direct", "n_eval": len(ev), "n_adapt": k,
                     "threshold": best_f1_threshold(ya, pd_[adapt])[0], **detection(ye, pd_[ev], best_f1_threshold(ya, pd_[adapt])[0])})
        if refit:
            m, p = fit_apply(make_lr, df.iloc[adapt][cols].astype(float), ya, df.iloc[ev][cols].astype(float))
            t = best_f1_threshold(ya, m.predict_proba(df.iloc[adapt][cols].astype(float))[:, 1])[0]
            rows.append({"regime": name, "score": "lr_bank", "n_eval": len(ev), "n_adapt": k, "threshold": t, **detection(ye, p, t)})
    reg = pd.DataFrame(rows); reg["split_source"] = src
    save_table(cfg, reg, f"{tag}_regimes")
    print(reg.round(3).to_string(index=False))

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
