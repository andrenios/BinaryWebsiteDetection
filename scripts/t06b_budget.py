#!/usr/bin/env python3
"""T06b: evidence budget sweep. q_direct and bank_v1 on S1-b for all budgets.

  python scripts/t06b_budget.py --dataset putra --split test      # the reported curve
  python scripts/t06b_budget.py --dataset putra --split train     # budget selection curve

Outputs results/t06b_<dataset>_<split>_curve.csv (per budget: mean tokens, USD
per 1,000, q_direct F1/AUROC, bank mean-vote AUROC, bank LR CV-AUROC) and
figures/t06b_<dataset>_<split>_{auroc,f1}_vs_tokens; reports the smallest
budget within 1 F1 point of the 2,000-token result.
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
from questions.bank import load_bank, direct_question, bank_questions  # noqa: E402
from eval.metrics import detection, usd_per_1000, latency_summary  # noqa: E402
from eval.plots import curve  # noqa: E402
from eval.timing import TimingLog  # noqa: E402
from combine.combiners import make_lr, cv_auroc, question_columns  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--budgets", default=None, help="comma list; default config summary_budgets")
    args = ap.parse_args()
    cfg = load_config(args.config)
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else cfg["summary_budgets"]
    sites, source = sites_for(cfg, args.dataset, args.split, limit=args.limit, offset=args.offset, reason="t06b")
    tag = f"t06b_{args.dataset}_{args.split}"
    client = client_from_config(cfg, tag, use_cache=not args.no_cache)
    banner("t06b", args, sites, source, client)
    bank = load_bank(cfg.bank_path)
    qd, qb = direct_question(bank), bank_questions(bank, "S1")
    thr = cfg.get("operating_threshold", 0.5)
    timing = TimingLog(cfg.timing_root / f"{tag}.csv")
    rows = []
    for b in budgets:
        v = f"S1b-{b}"
        d = run_questions(cfg, client, sites, args.dataset, v, qd, f"{tag}_{v}_direct", args.split, source,
                          workers=args.workers, timing=timing)
        k = run_questions(cfg, client, sites, args.dataset, v, qb, f"{tag}_{v}_bank", args.split, source,
                          workers=args.workers, timing=timing)
        if not len(d) or not len(k):
            print(f"[t06b] {v}: missing states"); continue
        save_table(cfg, d, f"{tag}_{v}_direct"); save_table(cfg, k, f"{tag}_{v}_bank")
        d, k = d[d.ok & d.q_direct.notna()], k[k.ok]
        det = detection(d.label, d.q_direct, thr)
        qcols = question_columns(k)
        y = k.label.values.astype(int)
        vote = k[qcols].astype(float).mean(axis=1)
        row = {"budget": b, "variant": v, "n": det["n"], "tokens_mean_direct": float(d.input_tokens.mean()),
               "tokens_mean_bank": float(k.input_tokens.mean()), "usd_per_1000_direct": usd_per_1000(d.cost_usd),
               "usd_per_1000_bank": usd_per_1000(k.cost_usd), "f1_direct": det["f1"], "auroc_direct": det["auroc"],
               "auroc_bank_vote": roc_auc_score(y, vote) if len(set(y)) == 2 else np.nan,
               "auroc_bank_lr_cv": cv_auroc(make_lr, k[qcols].astype(float), y, seed=cfg["SEED"]) if len(set(y)) == 2 and len(y) >= 10 else np.nan,
               **latency_summary(d.latency_ms), "http_failures": int((~d.ok).sum()), "split_source": source}
        rows.append(row)
        print(f"[t06b] budget {b}: tokens {row['tokens_mean_direct']:.0f}, F1 {det['f1']:.3f}, AUROC {det['auroc']:.3f}, "
              f"bank LR CV AUROC {row['auroc_bank_lr_cv']:.3f}")
    cur = pd.DataFrame(rows).sort_values("budget")
    if len(cur):
        ref = cur[cur.budget == cur.budget.max()].f1_direct.iloc[0]
        within = cur[cur.f1_direct >= ref - 0.01]
        cur["smallest_budget_within_1pt_f1"] = int(within.budget.min()) if len(within) else np.nan
        save_table(cfg, cur, f"{tag}_curve")
        curve(cur, "tokens_mean_direct", ["auroc_direct", "auroc_bank_vote", "auroc_bank_lr_cv"],
              cfg.figures_root / f"{tag}_auroc_vs_tokens", xlabel="mean input tokens per site", ylabel="AUROC", logx=True)
        curve(cur, "usd_per_1000_direct", ["f1_direct", "auroc_direct"], cfg.figures_root / f"{tag}_f1_vs_usd",
              xlabel="USD per 1,000 sites", ylabel="detection", logx=True)
        print(f"[t06b] smallest budget within 1 F1 point of {int(cur.budget.max())}: {cur.smallest_budget_within_1pt_f1.iloc[0]}")
    timing.close(); client.close()


if __name__ == "__main__":
    main()
