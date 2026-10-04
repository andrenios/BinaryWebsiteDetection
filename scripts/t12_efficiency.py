#!/usr/bin/env python3
"""T12: cost, latency, throughput and stage breakdown per configuration.

  python scripts/t12_efficiency.py --dataset putra [--throughput]

Aggregates every results/t06*_direct.csv, t07 tables and timing CSVs for the
dataset into results/t12_<dataset>_configs.csv (tokens/site, USD per 1,000,
latency p50/p95, per-stage seconds), draws the cost-detection frontier from
results/frontier_points.csv + T06/T07 rows (figures/t12_<dataset>_frontier),
and with --throughput measures achieved requests per minute at concurrency
1, 4, 16 with live (uncached) q_direct calls on --n sites each.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import load_config, add_common_args, sites_for, REPO
sys.path.insert(0, str(REPO / "src"))
from eval.metrics import detection, latency_summary, usd_per_1000  # noqa: E402
from eval.plots import frontier  # noqa: E402
from jev.client import client_from_config  # noqa: E402
from jev.run import run_questions  # noqa: E402
from questions.bank import load_bank, direct_question  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dataset", default="bootstrap-phreshphish")
    ap.add_argument("--throughput", action="store_true")
    ap.add_argument("--n", type=int, default=48, help="sites per concurrency level for --throughput")
    ap.add_argument("--levels", default="1,4,16")
    args = ap.parse_args()
    cfg = load_config(args.config)
    R = cfg.results_root
    thr = cfg.get("operating_threshold", 0.5)
    rows = []
    # stage timing from t02
    tim = pd.concat([pd.read_csv(p) for p in glob.glob(str(R / "timing" / f"t02_{args.dataset}.csv"))], ignore_index=True) \
        if glob.glob(str(R / "timing" / f"t02_{args.dataset}.csv")) else pd.DataFrame()
    stage_s = tim.groupby(["variant", "stage"]).seconds.mean().unstack() if len(tim) else pd.DataFrame()
    for p in sorted(glob.glob(str(R / f"t06_{args.dataset}_test_*_direct.csv"))) + sorted(glob.glob(str(R / f"t07_{args.dataset}_*_test_table.csv"))):
        df = pd.read_csv(p); ok = df[df.ok]
        if not len(ok):
            continue
        v = ok.variant.iloc[0]
        qcol = "q_direct"
        det = detection(ok.label, ok[qcol], thr) if qcol in ok else {}
        name = Path(p).stem.replace(f"_{args.dataset}", "")
        r = {"config": name, "variant": v, "n": len(ok), "tokens_per_site": float(ok.input_tokens.mean()),
             "usd_per_1000": usd_per_1000(ok.cost_usd), **latency_summary(ok.latency_ms),
             "f1_q_direct_at_0.5": det.get("f1"), "auroc_q_direct": det.get("auroc"),
             "summarise_s_per_site": float(stage_s.loc[v, "summarise"]) if len(stage_s) and v in stage_s.index and "summarise" in stage_s else np.nan,
             "ocr_s_per_site": float(stage_s.loc["S2", "ocr"]) if len(stage_s) and "S2" in stage_s.index and "ocr" in stage_s and v == "S2" else 0.0,
             "decide_s_per_site": float(ok.latency_ms.mean() / 1000), "split_source": ok.split_source.iloc[0] if "split_source" in ok else ""}
        r["latency_s_per_site"] = (r["summarise_s_per_site"] if not np.isnan(r["summarise_s_per_site"]) else 0) + r["ocr_s_per_site"] + r["decide_s_per_site"]
        rows.append(r)
    cf = pd.DataFrame(rows)
    cf.to_csv(R / f"t12_{args.dataset}_configs.csv", index=False)
    print(cf.round(3).to_string(index=False) if len(cf) else "[t12] no T06/T07 tables found")

    # frontier
    pts = cf.rename(columns={"f1_q_direct_at_0.5": "f1"})[["config", "f1", "usd_per_1000", "latency_s_per_site"]] if len(cf) else pd.DataFrame()
    fp = R / "frontier_points.csv"
    if fp.exists():
        extra = pd.read_csv(fp)
        extra = extra[extra.dataset == args.dataset] if "dataset" in extra else extra
        pts = pd.concat([pts, extra[[c for c in ["config", "f1", "usd_per_1000", "latency_s_per_site"] if c in extra]]], ignore_index=True)
    ref = REPO / cfg.get("paper2_reference_csv", "x")
    if ref.exists():
        r = pd.read_csv(ref)
        pts = pd.concat([pts, pd.DataFrame({"config": "paper2 " + r.model.astype(str), "f1": r.f1,
                                            "usd_per_1000": r.usd_per_pass / 2.631, "latency_s_per_site": r.seconds_per_site})], ignore_index=True)
    pts = pts.dropna(subset=["f1", "usd_per_1000"])
    pts = pts[pts.usd_per_1000 > 0]
    if len(pts):
        pts.to_csv(R / f"t12_{args.dataset}_frontier_points.csv", index=False)
        frontier(pts, cfg.figures_root / f"t12_{args.dataset}_frontier", title=f"cost-detection frontier ({args.dataset})")

    if args.throughput:
        sites, src = sites_for(cfg, args.dataset, "all", limit=args.n, reason="t12 throughput")
        bank = load_bank(cfg.bank_path)
        client = client_from_config(cfg, f"t12_{args.dataset}_throughput", use_cache=False)
        trows = []
        for lvl in [int(x) for x in args.levels.split(",")]:
            t0 = time.time()
            df = run_questions(cfg, client, sites, args.dataset, "S1", direct_question(bank), f"t12_{args.dataset}_throughput",
                               "all", src, workers=lvl, extra_meta={"concurrency": lvl})
            wall = time.time() - t0
            ok = df[df.ok]
            trows.append({"concurrency": lvl, "requests": len(df), "ok": len(ok), "wall_s": wall,
                          "requests_per_minute": 60 * len(df) / wall, "sites_per_second": len(df) / wall,
                          **latency_summary(ok.latency_ms), "http_429_or_fail": int((~df.ok).sum()),
                          "cost_usd": float(df.cost_usd.sum())})
            print(trows[-1])
        pd.DataFrame(trows).to_csv(R / f"t12_{args.dataset}_throughput.csv", index=False)
        client.close()


if __name__ == "__main__":
    main()
