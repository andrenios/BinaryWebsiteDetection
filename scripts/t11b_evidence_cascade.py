#!/usr/bin/env python3
"""T11b: evidence-acquisition cascade S0 -> S1 -> S2 (-> S3 optional).

  python scripts/t11b_evidence_cascade.py --dataset putra --stages S0,S1,S2 [--score lr]

Stage probabilities are q_direct (default) or, with --score lr, the LR
combiner fitted on the training split's bank run per stage. Per-stage
evidence-acquisition seconds come from config `evidence_seconds`
(PLACEHOLDERS until T17 measures them; the table says so). Bands are swept on
train and evaluated once on test. Outputs results/t11b_<dataset>_{sweep_train,test}.csv
and appends frontier points.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from _bootstrap import load_config, add_common_args, add_split_args, sites_for, banner, REPO
sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config  # noqa: E402
from jev.run import run_questions  # noqa: E402
from questions.bank import load_bank, direct_question, bank_questions  # noqa: E402
from cascade.band import Stage, run_cascade, evaluate_cascade, sweep_bands, pick_band  # noqa: E402
from combine.combiners import make_lr, fit_apply, cv_scores, question_columns  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--stages", default="S0,S1,S2")
    ap.add_argument("--score", default="direct", choices=["direct", "lr"])
    ap.add_argument("--objective", default="f1")
    ap.add_argument("--max-escalation", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = cfg.get("operating_threshold", 0.5)
    stages = args.stages.split(",")
    tag = f"t11b_{args.dataset}_{args.score}_{'-'.join(stages)}"
    client = client_from_config(cfg, tag, use_cache=not args.no_cache)
    bank = load_bank(cfg.bank_path)
    ev = cfg.get("evidence_seconds", {})

    probs = {"train": {}, "test": {}}; usd = {}; lat = {}
    labels = {}
    for split in ("train", "test"):
        sites, src = sites_for(cfg, args.dataset, split, limit=args.limit, reason="t11b")
        args.split = split; banner("t11b", args, sites, src, client)
        labels[split] = pd.Series({s.id: s.label for s in sites})
        for v in stages:
            if args.score == "direct":
                # reuse T06 experiment names so cached responses are shared
                df = run_questions(cfg, client, sites, args.dataset, v, direct_question(bank), f"t06_{args.dataset}_{split}_{v}_direct", split, src, workers=args.workers)
                if not len(df):
                    continue
                df = df[df.ok].set_index("site_id")
                probs[split][v] = df.q_direct.astype(float)
            else:
                exp = f"t07_{args.dataset}_{v}_{split}" if v != "S0" else f"t06_{args.dataset}_{split}_S0_direct"
                qs = bank_questions(bank, v) if v != "S0" else direct_question(bank)
                df = run_questions(cfg, client, sites, args.dataset, v, qs, exp, split, src, workers=args.workers)
                if not len(df):
                    continue
                df = df[df.ok].set_index("site_id")
                probs[split][v] = df   # fit later
            usd[v] = float(df.cost_usd.mean()); lat[v] = float(df.latency_ms.mean() / 1000)
    if args.score == "lr":
        for v in list(probs["train"]):
            tr, te = probs["train"][v], probs["test"].get(v)
            if te is None:
                continue
            cols = ["q_direct"] + question_columns(tr) if v != "S0" else ["q_direct"]
            cols = [c for c in cols if c in te.columns]
            y = tr.label.values.astype(int)
            probs["train"][v] = pd.Series(cv_scores(make_lr, tr[cols].astype(float), y, seed=cfg["SEED"]), index=tr.index)
            probs["test"][v] = pd.Series(fit_apply(make_lr, tr[cols].astype(float), y, te[cols].astype(float))[1], index=te.index)

    def build(split):
        idx = labels[split].index
        return [Stage(v, probs[split][v].reindex(idx), usd.get(v, 0.0), lat.get(v, 0.0) + float(ev.get(v, 0.0)))
                for v in stages if v in probs[split]]

    st_tr, st_te = build("train"), build("test")
    if len(st_te) < 2:
        raise SystemExit("[t11b] fewer than two stages have states/probabilities; nothing to cascade")
    sweep = sweep_bands(st_tr, labels["train"], threshold=thr)
    sweep["evidence_seconds_source"] = cfg.get("evidence_seconds_source", "placeholder")
    sweep.to_csv(cfg.results_root / f"{tag}_sweep_train.csv", index=False)
    band = pick_band(sweep, args.objective, args.max_escalation, stages[0])
    print(f"[t11b] band chosen on train: [{band.t_low}, {band.t_high}] {args.objective}={band[args.objective]:.3f}")

    rows = []
    for s in st_te:   # single-stage references
        rows.append({"config": f"{s.name} alone ({args.score})", **evaluate_cascade(run_cascade([s], 0, 0), labels["test"], thr)})
    res = run_cascade(st_te, band.t_low, band.t_high)
    rows.append({"config": f"cascade {'->'.join(s.name for s in st_te)} band [{band.t_low},{band.t_high}] ({args.score})",
                 **evaluate_cascade(res, labels["test"], thr), "t_low": band.t_low, "t_high": band.t_high})
    out = pd.DataFrame(rows)
    out["evidence_seconds_source"] = cfg.get("evidence_seconds_source", "placeholder")
    out["split_source"] = sites_for(cfg, args.dataset, "test", limit=1)[1]
    out.to_csv(cfg.results_root / f"{tag}_test.csv", index=False)
    cols = [c for c in ["config", "f1", "auroc", "ppv_at_1pct", "usd_per_1000", "mean_seconds_per_site"] + [f"stop_{s.name}" for s in st_te] if c in out]
    print(out[cols].round(3).to_string(index=False))
    fp = cfg.results_root / "frontier_points.csv"
    out[cols].rename(columns={"mean_seconds_per_site": "latency_s_per_site"}).assign(task="t11b", dataset=args.dataset) \
        .to_csv(fp, mode="a", header=not fp.exists(), index=False)
    client.close()


if __name__ == "__main__":
    main()
