#!/usr/bin/env python3
"""T11: escalation cascade Jev -> stored second-stage decision.

  python scripts/t11_cascade.py --dataset putra --first results/t07_putra_S1_test_scores.csv --first-col p_lr \
      --second data/paper2/res_gemma4_31b_tuned.csv --second-col decision --second-seconds 18 --second-usd 0.005

Sweeps the band on the TRAIN split (needs first-stage train scores with the
same column; pass --first-train) and evaluates the chosen band once on test.
Every configuration is appended to results/frontier_points.csv for the T12 frontier.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from _bootstrap import load_config, add_common_args, REPO
sys.path.insert(0, str(REPO / "src"))
from cascade.band import Stage, run_cascade, evaluate_cascade, sweep_bands, pick_band  # noqa: E402


def load_scores(path, col, usd_col=None, sec_col=None):
    df = pd.read_csv(path).set_index("site_id")
    return df, df[col].astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dataset", default="bootstrap-phreshphish")
    ap.add_argument("--name", default="cascade")
    ap.add_argument("--first", required=True, help="test decision table of the first stage (site_id,label,<col>)")
    ap.add_argument("--first-col", default="q_direct")
    ap.add_argument("--first-train", default=None, help="train decision table of the first stage (band sweep)")
    ap.add_argument("--first-usd", type=float, default=None, help="USD per site of stage 1 (default: mean cost_usd in table)")
    ap.add_argument("--first-seconds", type=float, default=None, help="seconds per site of stage 1 (default: latency_ms/1000)")
    ap.add_argument("--second", required=True, help="second-stage table (site_id,<col>); probability or 0/1 decision")
    ap.add_argument("--second-col", default="decision")
    ap.add_argument("--second-train", default=None)
    ap.add_argument("--second-usd", type=float, default=0.0)
    ap.add_argument("--second-seconds", type=float, default=0.0)
    ap.add_argument("--second-is-decision", action="store_true")
    ap.add_argument("--objective", default="f1")
    ap.add_argument("--max-escalation", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = cfg.get("operating_threshold", 0.5)

    f_te, p1_te = load_scores(args.first, args.first_col)
    s_te, p2_te = load_scores(args.second, args.second_col)
    usd1 = args.first_usd if args.first_usd is not None else float(f_te.cost_usd.mean()) if "cost_usd" in f_te else 0.0
    sec1 = args.first_seconds if args.first_seconds is not None else float(f_te.latency_ms.mean() / 1000) if "latency_ms" in f_te else 0.0
    y_te = f_te.label.astype(int)

    def stages(p1, p2):
        return [Stage("jev", p1, usd1, sec1), Stage("second", p2.reindex(p1.index), args.second_usd, args.second_seconds, args.second_is_decision)]

    if args.first_train and args.second_train:
        f_tr, p1_tr = load_scores(args.first_train, args.first_col)
        _, p2_tr = load_scores(args.second_train, args.second_col)
        sweep = sweep_bands(stages(p1_tr, p2_tr), f_tr.label.astype(int), threshold=thr)
        sweep.to_csv(cfg.results_root / f"t11_{args.dataset}_{args.name}_sweep_train.csv", index=False)
        band = pick_band(sweep, args.objective, args.max_escalation, "jev")
        print(f"[t11] band chosen on train: [{band.t_low}, {band.t_high}] ({args.objective}={band[args.objective]:.3f})")
        lows, highs = [band.t_low], [band.t_high]
        src = "train sweep"
    else:
        print("[t11] no train tables given: reporting the full band grid on test (exploratory, not a selected configuration)")
        lows = highs = None; src = "grid on test (exploratory)"

    st = stages(p1_te, p2_te)
    rows = []
    base = evaluate_cascade(run_cascade([st[0]], 0, 0), y_te, thr); rows.append({"config": f"{args.name}: Jev only", **base})
    sec_only = evaluate_cascade(run_cascade([Stage("second", p2_te.reindex(p1_te.index), args.second_usd, args.second_seconds, args.second_is_decision)], 0, 0), y_te, thr)
    rows.append({"config": f"{args.name}: second stage only", **sec_only})
    grid = sweep_bands(st, y_te, lows, highs, thr)
    grid["config"] = [f"{args.name}: band [{r.t_low},{r.t_high}]" for r in grid.itertuples()]
    out = pd.concat([pd.DataFrame(rows), grid], ignore_index=True)
    out["band_source"] = src
    out.to_csv(cfg.results_root / f"t11_{args.dataset}_{args.name}_test.csv", index=False)
    cols = [c for c in ["config", "f1", "auroc", "ppv_at_1pct", "usd_per_1000", "mean_seconds_per_site", "stop_jev", "stop_second"] if c in out]
    print(out[cols].round(3).to_string(index=False))
    # frontier points
    fp = cfg.results_root / "frontier_points.csv"
    pts = out[cols].rename(columns={"mean_seconds_per_site": "latency_s_per_site"}).assign(task="t11", dataset=args.dataset)
    pts.to_csv(fp, mode="a", header=not fp.exists(), index=False)


if __name__ == "__main__":
    main()
