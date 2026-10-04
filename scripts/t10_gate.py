#!/usr/bin/env python3
"""T10: deterministic rule gate before any model call.

  python scripts/t10_gate.py --dataset putra --variant S1

Training split: fraction gated, phishing missed, prevalence shift, for every
candidate gate. Test split, once: gate followed by Jev q_direct (and the LR
combiner score when t07 test scores exist); detection, fraction of model calls
saved, class ratio seen by Jev. Outputs results/t10_<dataset>_{train_gates,test}.csv.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from _bootstrap import load_config, add_common_args, add_split_args, sites_for, banner, REPO
sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config  # noqa: E402
from jev.run import run_questions, save_table  # noqa: E402
from questions.bank import load_bank, direct_question  # noqa: E402
from gate.rules import GATES, gate_table, gate_stats, apply_gate  # noqa: E402
from eval.metrics import detection, best_f1_threshold, usd_per_1000  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--gate", default="no_forms_no_phishy_links", choices=list(GATES))
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = cfg.get("operating_threshold", 0.5)

    # training split: candidate gates
    tr, src = sites_for(cfg, args.dataset, "train", limit=args.limit, reason="t10")
    g = gate_table(cfg, args.dataset, args.variant, [s.id for s in tr]).merge(
        pd.DataFrame({"site_id": [s.id for s in tr], "label": [s.label for s in tr]}), on="site_id")
    train_rows = [{**gate_stats(g, name), "split": "train", "split_source": src} for name in GATES]
    save_table(cfg, pd.DataFrame(train_rows), f"t10_{args.dataset}_train_gates")
    print(pd.DataFrame(train_rows).round(3).to_string(index=False))

    # test split once: gate + Jev
    te, src_te = sites_for(cfg, args.dataset, "test", limit=args.limit, reason="t10 gate+Jev")
    args.split = "test"
    client = client_from_config(cfg, f"t10_{args.dataset}", use_cache=not args.no_cache)
    banner("t10", args, te, src_te, client)
    bank = load_bank(cfg.bank_path)
    df = run_questions(cfg, client, te, args.dataset, args.variant, direct_question(bank), f"t06_{args.dataset}_test_{args.variant}_direct",
                       "test", src_te, workers=args.workers)      # same experiment name as T06 -> cache hits
    df = df[df.ok].merge(gate_table(cfg, args.dataset, args.variant, list(df.site_id)), on="site_id")
    gated = df[f"gate_{args.gate}"]
    # threshold fitted on train for q_direct (train run is cached if T06/T07 ran on train; otherwise 0.5)
    trd = run_questions(cfg, client, tr, args.dataset, args.variant, direct_question(bank), f"t06_{args.dataset}_train_{args.variant}_direct",
                        "train", src, workers=args.workers)
    t_fit = best_f1_threshold(trd[trd.ok].label, trd[trd.ok].q_direct)[0] if len(trd) else thr
    rows = []
    for name, p, uses_gate in (("Jev q_direct, no gate", df.q_direct, False),
                               (f"gate[{args.gate}] + Jev q_direct", apply_gate(df.q_direct, gated), True)):
        d = detection(df.label, p, t_fit)
        calls = float((~gated).mean()) if uses_gate else 1.0
        rows.append({"config": name, **d, "threshold": t_fit, "fraction_model_calls": calls,
                     "model_calls_saved": 1 - calls, "usd_per_1000": usd_per_1000(df.cost_usd) * calls,
                     "prevalence_seen_by_jev": float(df.label[~gated].mean()) if uses_gate and (~gated).any() else float(df.label.mean()),
                     "phishing_blocked_by_gate": int((gated & (df.label == 1)).sum()) if uses_gate else 0,
                     "split_source": src_te})
    save_table(cfg, pd.DataFrame(rows), f"t10_{args.dataset}_test")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    client.close()


if __name__ == "__main__":
    main()
