#!/usr/bin/env python3
"""T15: adversarial steering on the test split's phishing pages + N benign.

  python scripts/t15_adversarial.py --dataset putra --benign 500 --questions direct,bank

Variants A1–A4 (src/adversarial/inject.py) plus the mitigation arm (text in
`untrusted_text` with a warning in the instructions). For each arm: noul shift
vs the clean state, flip rate at the operating threshold, AUROC.
Outputs results/t15_<dataset>_<questions>_{table,summary}.csv.
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
from jev.run import run_questions, load_state, save_table  # noqa: E402
from questions.bank import load_bank, direct_question, bank_questions  # noqa: E402
from summarise.state import build_s1  # noqa: E402
from adversarial.inject import VARIANTS, a4_padding_html, mitigation_state, mitigation_questions  # noqa: E402
from combine.combiners import question_columns  # noqa: E402
from eval.timing import TimingLog  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--benign", type=int, default=500)
    ap.add_argument("--questions", default="direct", help="direct | bank")
    ap.add_argument("--arms", default="clean,A1,A2,A3,A4,A1m,A2m")
    ap.add_argument("--threshold", type=float, default=None, help="operating threshold (default: config)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = args.threshold if args.threshold is not None else cfg.get("operating_threshold", 0.5)
    all_sites, src = sites_for(cfg, args.dataset, args.split, reason="t15")
    pos = [s for s in all_sites if s.label == 1]
    neg = [s for s in all_sites if s.label == 0][:args.benign]
    sites = sorted(pos + neg, key=lambda s: s.id)
    if args.limit:
        sites = sites[:args.limit]
    tag = f"t15_{args.dataset}_{args.questions}"
    client = client_from_config(cfg, tag, use_cache=not args.no_cache)
    banner("t15", args, sites, src, client)
    bank = load_bank(cfg.bank_path)
    qs = direct_question(bank) if args.questions == "direct" else bank_questions(bank, args.variant)
    timing = TimingLog(cfg.timing_root / f"{tag}.csv")

    clean = {s.id: load_state(cfg, args.dataset, args.variant, s.id) for s in sites}
    clean = {k: v for k, v in clean.items() if v is not None}
    tables = {}
    for arm in args.arms.split(","):
        mitig = arm.endswith("m")
        base = arm[:-1] if mitig else arm
        if base == "clean":
            states = clean
        elif base in VARIANTS:
            states = {k: VARIANTS[base](v) for k, v in clean.items()}
        elif base == "A4":
            states = {}
            for s in sites:
                if s.id in clean and s.html_path.exists():
                    st, _ = build_s1(a4_padding_html(s.html()), s.url, cfg["s1_budget"], cfg["token_counter_model"])
                    states[s.id] = st
        else:
            print(f"[t15] unknown arm {arm}"); continue
        q = qs
        if mitig:
            states = {k: mitigation_state(v) for k, v in states.items()}
            q = mitigation_questions(qs)
        df = run_questions(cfg, client, [s for s in sites if s.id in states], args.dataset, args.variant, q, f"{tag}_{arm}",
                           args.split, src, states=states, workers=args.workers, timing=timing, extra_meta={"arm": arm})
        df = df[df.ok]
        tables[arm] = df
        save_table(cfg, df, f"{tag}_{arm}")
        print(f"[t15] {arm}: {len(df)} ok, mean q_direct phish {df.q_direct[df.label == 1].mean():.3f} benign {df.q_direct[df.label == 0].mean():.3f}")

    c = tables["clean"].set_index("site_id")
    cols = ["q_direct"] + (question_columns(c) if args.questions == "bank" else [])
    rows = []
    for arm, df in tables.items():
        d = df.set_index("site_id").reindex(c.index)
        for col in cols:
            if col not in d:
                continue
            a, b = c[col].astype(float), d[col].astype(float)
            m = a.notna() & b.notna()
            y = c.label[m].astype(int)
            rows.append({"arm": arm, "question": col, "n": int(m.sum()),
                         "shift_mean_phish": float((b - a)[m & (c.label == 1)].mean()),
                         "shift_mean_benign": float((b - a)[m & (c.label == 0)].mean()),
                         "flip_rate_phish_to_benign": float(((a >= thr) & (b < thr))[m & (c.label == 1)].mean()),
                         "flip_rate_benign_to_phish": float(((a < thr) & (b >= thr))[m & (c.label == 0)].mean()),
                         "auroc_clean": roc_auc_score(y, a[m]) if y.nunique() == 2 else np.nan,
                         "auroc_arm": roc_auc_score(y, b[m]) if y.nunique() == 2 else np.nan,
                         "tokens_mean": float(d.input_tokens[m].mean()), "threshold": thr, "split_source": src})
    out = pd.DataFrame(rows)
    save_table(cfg, out, f"{tag}_summary")
    print(out[out.question == "q_direct"].round(3).to_string(index=False))
    timing.close(); client.close()


if __name__ == "__main__":
    main()
