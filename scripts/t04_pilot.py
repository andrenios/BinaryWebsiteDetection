#!/usr/bin/env python3
"""T04: go/no-go pilot. bank_v1 (+ q_direct) on S1 for N balanced training sites.

  python scripts/t04_pilot.py --dataset putra --split train --n 300

Outputs results/t04_<dataset>_per_question.csv (AUROC, noul distribution by
label), results/t04_<dataset>_correlation.csv (Spearman), figures/t04_<dataset>_corr,
results/t04_<dataset>_summary.csv (cost, latency), results/t04_<dataset>_table.csv.
Site selection: balanced by label after a seeded shuffle within the split
(the paper's language x length stratification needs the full data; D12).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from _bootstrap import load_config, add_common_args, add_split_args, sites_for, banner, REPO
sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config  # noqa: E402
from jev.run import run_questions, summarise_run, save_table  # noqa: E402
from questions.bank import load_bank, bank_questions  # noqa: E402
from eval.timing import TimingLog  # noqa: E402
from eval.plots import heatmap  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.set_defaults(split="train")
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n = args.n or cfg.get("pilot_sites", 300)
    sites, source = sites_for(cfg, args.dataset, args.split, limit=n, balanced=True, reason="t04 pilot")
    exp = f"t04_{args.dataset}_{args.variant}"
    client = client_from_config(cfg, exp, use_cache=not args.no_cache)
    banner("t04", args, sites, source, client)
    bank = load_bank(cfg.bank_path)
    qs = bank_questions(bank, args.variant)          # includes q_direct
    timing = TimingLog(cfg.timing_root / f"{exp}.csv")
    df = run_questions(cfg, client, sites, args.dataset, args.variant, qs, exp, args.split, source,
                       workers=args.workers, timing=timing)
    save_table(cfg, df, f"{exp}_table")
    ok = df[df.ok]
    qcols = [q for q in qs if q in ok.columns]

    rows = []
    for q in qcols:
        p = ok[q].astype(float)
        rows.append({"question": q, "auroc": roc_auc_score(ok.label, p) if ok.label.nunique() == 2 else np.nan,
                     "mean_phish": p[ok.label == 1].mean(), "mean_benign": p[ok.label == 0].mean(),
                     "p10": p.quantile(.1), "p50": p.quantile(.5), "p90": p.quantile(.9),
                     "share_gt_0.5": float((p > 0.5).mean()), "share_in_0.3_0.7": float(p.between(0.3, 0.7).mean())})
    per_q = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    save_table(cfg, per_q, f"{exp}_per_question")
    corr = ok[qcols].astype(float).corr(method="spearman")
    corr.to_csv(cfg.results_root / f"{exp}_correlation.csv")
    heatmap(corr, cfg.figures_root / f"{exp}_corr", title=f"bank_v1 Spearman correlation ({args.dataset}, {args.split})")
    summ = {**summarise_run(df, client, exp, cfg["expected_response_model"]), "sites": len(sites),
            "split_source": source, "n_questions": len(qcols), "auroc_q_direct": per_q.set_index("question").auroc.get("q_direct")}
    pd.DataFrame([summ]).to_csv(cfg.results_root / f"{exp}_summary.csv", index=False)
    print(per_q.round(3).to_string(index=False)); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()})
    timing.close(); client.close()


if __name__ == "__main__":
    main()
