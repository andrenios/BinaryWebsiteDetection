#!/usr/bin/env python3
"""T06: q_direct on S0, S1, S2, S3 over the test split; option-order check on S1.

  python scripts/t06_direct.py --dataset putra --split test --variants S0,S1,S2,S3

Outputs results/t06_<dataset>_<split>_<variant>_direct.csv (decision tables),
results/t06_<dataset>_<split>_table.csv (one row per variant: F1/P/R/Acc@0.5,
AUROC, AUPRC, latency p50/p95, USD per 1,000 sites, HTTP failures, tokens),
results/t06_<dataset>_<split>_option_order.csv, and appends the paper-2
reference rows from data/paper2/reference_rows.csv when that file exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _bootstrap import load_config, add_common_args, add_split_args, sites_for, banner, REPO
sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config  # noqa: E402
from jev.run import run_questions, summarise_run, save_table  # noqa: E402
from questions.bank import load_bank, direct_question, structural_questions  # noqa: E402
from eval.metrics import detection, latency_summary, usd_per_1000, bootstrap_ci  # noqa: E402
from eval.timing import TimingLog  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--variants", default="S0,S1,S2,S3")
    ap.add_argument("--no-option-order", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sites, source = sites_for(cfg, args.dataset, args.split, limit=args.limit, offset=args.offset, reason="t06")
    tag = f"t06_{args.dataset}_{args.split}"
    client = client_from_config(cfg, tag, use_cache=not args.no_cache)
    banner("t06", args, sites, source, client)
    bank = load_bank(cfg.bank_path)
    qd = direct_question(bank)
    thr = cfg.get("operating_threshold", 0.5)
    timing = TimingLog(cfg.timing_root / f"{tag}.csv")

    rows = []
    for v in args.variants.split(","):
        exp = f"{tag}_{v}_direct"
        df = run_questions(cfg, client, sites, args.dataset, v, qd, exp, args.split, source,
                           workers=args.workers, timing=timing)
        if not len(df):
            print(f"[t06] {v}: no states"); continue
        save_table(cfg, df, exp)
        ok = df[df.ok & df.q_direct.notna()]
        det = detection(ok.label, ok.q_direct, thr)
        lo, hi = bootstrap_ci(ok.label, ok.q_direct, roc_auc_score) if det["n_pos"] and det["n_pos"] < det["n"] else (np.nan, np.nan)
        rows.append({"config": f"Jev q_direct {v}", "variant": v, "question": "q_direct", **det,
                     "auroc_ci_low": lo, "auroc_ci_high": hi, "http_failures": int((~df.ok).sum()),
                     "requests": len(df), "input_tokens_mean": float(ok.input_tokens.mean()),
                     "usd_per_1000": usd_per_1000(ok.cost_usd), **latency_summary(ok.latency_ms),
                     "split_source": source})
        print(f"[t06] {v}: n={det['n']} F1={det['f1']:.3f} AUROC={det['auroc']:.3f} fails={int((~df.ok).sum())} "
              f"tokens={ok.input_tokens.mean():.0f} USD/1000={usd_per_1000(ok.cost_usd):.3f}")

    # option-order check on S1: choice and reversed choice in separate requests
    if not args.no_option_order and "S1" in args.variants.split(","):
        st = structural_questions(bank)
        a = run_questions(cfg, client, sites, args.dataset, "S1", {"q_direct_choice": st["q_direct_choice"]},
                          f"{tag}_S1_choice", args.split, source, workers=args.workers, timing=timing)
        b = run_questions(cfg, client, sites, args.dataset, "S1", {"q_direct_choice_rev": st["q_direct_choice_rev"]},
                          f"{tag}_S1_choice_rev", args.split, source, workers=args.workers, timing=timing)
        save_table(cfg, a, f"{tag}_S1_choice"); save_table(cfg, b, f"{tag}_S1_choice_rev")
        m = a[a.ok].merge(b[b.ok], on=["site_id", "label"], suffixes=("_a", "_b"))
        pa, pb = m["q_direct_choice__p_phishing"].astype(float), m["q_direct_choice_rev__p_phishing"].astype(float)
        oo = {"n": len(m), "flip_rate": float((m.q_direct_choice != m.q_direct_choice_rev).mean()),
              "mean_abs_shift_p_phishing": float((pa - pb).abs().mean()),
              "mean_p_phishing_first_order": float(pa.mean()), "mean_p_phishing_reversed": float(pb.mean()),
              "auroc_first_order": roc_auc_score(m.label, pa) if m.label.nunique() == 2 else np.nan,
              "auroc_reversed": roc_auc_score(m.label, pb) if m.label.nunique() == 2 else np.nan,
              "auroc_average": roc_auc_score(m.label, (pa + pb) / 2) if m.label.nunique() == 2 else np.nan,
              "split_source": source}
        pd.DataFrame([oo]).to_csv(cfg.results_root / f"{tag}_option_order.csv", index=False)
        print(f"[t06] option order: flip rate {oo['flip_rate']:.3f}, mean |shift| {oo['mean_abs_shift_p_phishing']:.3f}, "
              f"AUROC {oo['auroc_first_order']:.3f} vs {oo['auroc_reversed']:.3f}")

    table = pd.DataFrame(rows)
    ref = Path(cfg.get("paper2_reference_csv", "")) if cfg.get("paper2_reference_csv") else None
    if ref and (REPO / ref).exists():
        r = pd.read_csv(REPO / ref)
        table = pd.concat([table, r.assign(config="paper2 " + r.model.astype(str) + " " + r["mode"].astype(str))],
                          ignore_index=True)
    else:
        print("[t06] paper-2 reference rows not present (data/paper2/reference_rows.csv); table has Jev rows only")
    save_table(cfg, table, f"{tag}_table")
    summ = summarise_run(pd.concat([pd.read_csv(cfg.results_root / f"{tag}_{v}_direct.csv")
                                    for v in args.variants.split(",")
                                    if (cfg.results_root / f"{tag}_{v}_direct.csv").exists()]), client, tag,
                         cfg["expected_response_model"])
    pd.DataFrame([summ]).to_csv(cfg.results_root / f"{tag}_summary.csv", index=False)
    print(summ)
    timing.close(); client.close()


if __name__ == "__main__":
    main()
