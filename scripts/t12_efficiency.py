#!/usr/bin/env python3
"""T12: cost, latency, throughput and stage breakdown per configuration.

  python scripts/t12_efficiency.py --dataset putra [--throughput]

Aggregates every results/t06*_direct.csv, t07 tables and timing CSVs for the
dataset into results/t12_<dataset>_configs.csv (tokens/site, USD per 1,000,
latency p50/p95, per-stage seconds), draws the cost-detection frontier from
results/frontier_points.csv + T06/T07 rows (figures/t12_<dataset>_frontier),
and with --throughput measures achieved requests per minute at concurrency
1, 4, 16 with live (uncached) q_direct calls on --n sites each.

v2.3 additions:
  --sustained   appends a sustained-throughput row computed from the persisted
                live requests of the T06 S1 test run (data/raw_responses/
                t06_<dataset>_test.jsonl, experiment t06_<dataset>_test_S1_direct,
                uncached records; wall = last completion - first start) to
                results/t12_<dataset>_throughput.csv (column source = "sustained ...").
  cost sensitivity  results/t12_<dataset>_cost_sensitivity.csv: whether the frontier
                ordering (configurations sorted by USD per 1,000 sites) changes
                under GPU rate x0.5 / x2, utilisation 50 / 100 %, and the CPU rate.
                Cost basis per point: API (Jev rows, unchanged), GPU (paper-2 rows,
                t08d encoder, t09 twin), CPU (t08 classical rows).
"""
from __future__ import annotations

import json

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
    ap.add_argument("--sustained", action="store_true", help="sustained-throughput row from the T06 S1 test run's persisted requests")
    ap.add_argument("--sustained-experiment", default=None, help="default t06_<dataset>_test_S1_direct")
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
    # local-model rows (GPU / CPU cost basis) for the frontier and the cost-sensitivity table
    gpu_rate = float(cfg.get("gpu_usd_per_hour", 0.99)); cpu_rate = float(cfg.get("cpu_usd_per_hour", 0.10))
    basis = pd.Series("api", index=pts.index) if len(pts) else pd.Series(dtype=str)
    if len(pts):
        basis[pts.config.astype(str).str.startswith("paper2 ")] = "gpu"
    for pat, col_f1, col_cost, b in ((f"t08d_{args.dataset}_encoder.csv", "test_f1_at_fit", "usd_per_1000_sites", "gpu"),
                                     (f"t09_{args.dataset}_S1_summary.csv", "test_f1_at_fit", "usd_per_1000", "gpu"),
                                     (f"t08_{args.dataset}_baselines.csv", "test_f1_at_fit", None, "cpu")):
        q = R / pat
        if not q.exists():
            continue
        d = pd.read_csv(q)
        if b == "cpu" and "infer_ms_per_site_cpu" in d:
            d["usd_per_1000"] = d.infer_ms_per_site_cpu / 1000 / 3600 * cpu_rate * 1000
            d["latency_s_per_site"] = d.infer_ms_per_site_cpu / 1000
        elif col_cost in d:
            d["usd_per_1000"] = d[col_cost]
        label_col = "baseline" if "baseline" in d else ("model" if "model" in d else "config")
        if col_f1 in d and "usd_per_1000" in d:
            add = pd.DataFrame({"config": f"{b} " + d[label_col].astype(str), "f1": d[col_f1], "usd_per_1000": d.usd_per_1000,
                                "latency_s_per_site": d.get("latency_s_per_site", np.nan)})
            pts = pd.concat([pts, add], ignore_index=True); basis = pd.concat([basis, pd.Series(b, index=add.index + (len(pts) - len(add)))])
    pts = pts.dropna(subset=["f1", "usd_per_1000"])
    basis = basis.reindex(pts.index).fillna("api")
    pts = pts[pts.usd_per_1000 > 0]; basis = basis.reindex(pts.index)
    if len(pts):
        pts.assign(cost_basis=basis.values).to_csv(R / f"t12_{args.dataset}_frontier_points.csv", index=False)
        frontier(pts, cfg.figures_root / f"t12_{args.dataset}_frontier", title=f"cost-detection frontier ({args.dataset})")
        # cost-accounting sensitivity (Appendix A5): does the ordering by USD per 1,000 change?
        def ordering(scale_gpu=1.0, scale_cpu=1.0):
            c = pts.usd_per_1000 * np.where(basis.values == "gpu", scale_gpu, np.where(basis.values == "cpu", scale_cpu, 1.0))
            return " < ".join(pts.config.astype(str).values[np.argsort(c.values, kind="stable")])
        base = ordering()
        srows = [{"assumption": "baseline", "value": f"GPU {gpu_rate} USD/h at recorded utilisation; CPU {cpu_rate} USD/h; API list price", "frontier_ordering_changed": False, "ordering": base}]
        for name, val, sg, sc in (("gpu_rate_x0.5", gpu_rate * 0.5, 0.5, 1.0), ("gpu_rate_x2", gpu_rate * 2, 2.0, 1.0),
                                  ("utilisation_50pct", 0.5, 2.0, 1.0), ("utilisation_100pct", 1.0, 1.0, 1.0),
                                  ("cpu_rate_x0.5", cpu_rate * 0.5, 1.0, 0.5), ("cpu_rate_x2", cpu_rate * 2, 1.0, 2.0)):
            o = ordering(sg, sc)
            srows.append({"assumption": name, "value": val, "frontier_ordering_changed": o != base, "ordering": o})
        pd.DataFrame(srows).to_csv(R / f"t12_{args.dataset}_cost_sensitivity.csv", index=False)
        print(pd.DataFrame(srows)[["assumption", "value", "frontier_ordering_changed"]].to_string(index=False))

    if args.sustained:
        exp = args.sustained_experiment or f"t06_{args.dataset}_test_S1_direct"
        logp = cfg.raw_responses_root / f"t06_{args.dataset}_test.jsonl"
        recs = []
        if logp.exists():
            with open(logp, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("experiment") == exp and not r.get("cached") and r.get("ts"):
                        recs.append(r)
        if recs:
            d = pd.DataFrame({"ts": [r["ts"] for r in recs], "latency_ms": [r.get("latency_ms") or 0.0 for r in recs],
                              "ok": [bool(r.get("ok")) for r in recs], "cost_usd": [r.get("cost_usd") or 0.0 for r in recs],
                              "workers": [r.get("workers") for r in recs]})
            start = (d.ts - d.latency_ms / 1000).min(); wall = float(d.ts.max() - start)
            row = {"concurrency": int(d.workers.dropna().mode().iloc[0]) if d.workers.notna().any() else cfg.get("concurrency"),
                   "requests": len(d), "ok": int(d.ok.sum()), "wall_s": wall, "requests_per_minute": 60 * len(d) / wall if wall > 0 else np.nan,
                   "sites_per_second": len(d) / wall if wall > 0 else np.nan, **latency_summary(d.latency_ms),
                   "http_429_or_fail": int((~d.ok).sum()), "cost_usd": float(d.cost_usd.sum()), "source": f"sustained ({exp}, live requests)"}
            tp = R / f"t12_{args.dataset}_throughput.csv"
            prev = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
            if len(prev) and "source" not in prev:
                prev["source"] = "burst (--throughput)"
            if len(prev):
                prev = prev[~prev.source.astype(str).str.startswith("sustained")]
            pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(tp, index=False)
            print(f"[t12] sustained: {row}")
        else:
            print(f"[t12] no live records for {exp} in {logp}; sustained row not written")

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
                          "cost_usd": float(df.cost_usd.sum()), "source": "burst (--throughput)"})
            print(trows[-1])
        tp = R / f"t12_{args.dataset}_throughput.csv"
        prev = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
        keep = prev[prev.source.astype(str).str.startswith("sustained")] if len(prev) and "source" in prev else pd.DataFrame()
        pd.concat([pd.DataFrame(trows), keep], ignore_index=True).to_csv(tp, index=False)
        client.close()


if __name__ == "__main__":
    main()
