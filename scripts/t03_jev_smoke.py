#!/usr/bin/env python3
"""T03: Jev client smoke test.

  python scripts/t03_jev_smoke.py --config config.yaml --dataset fixture --variant S1 --questions direct

Reads states built by t02 (data/derived/states/<dataset>/<variant>/), sends one
request per site, persists everything through JevClient, and writes
  results/t03_<dataset>_<variant>_<questions>.csv      one row per site: nouls, tokens, cost, latency, request id
  results/t03_<dataset>_<variant>_<questions>_summary.csv
Every response must carry model == expected_response_model (the client aborts otherwise).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from _bootstrap import load_config, add_common_args, open_dataset, select_sites, REPO

sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config, JevError  # noqa: E402
from questions.bank import load_bank, bank_questions, direct_question, structural_questions  # noqa: E402
from eval.timing import TimingLog  # noqa: E402


def load_state(cfg, dataset: str, variant: str, sid: str) -> dict | None:
    p = cfg.derived_root / "states" / dataset / variant / f"{sid}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pick_questions(bank, which: str, variant: str) -> dict:
    if which == "direct":
        return direct_question(bank)
    if which == "bank":
        return bank_questions(bank, "S2" if variant == "S2" else "S1")
    if which == "structural":
        return {**direct_question(bank), **structural_questions(bank)}
    raise SystemExit(f"unknown --questions {which}")


def flatten_answers(answers: dict) -> dict:
    out = {}
    for qid, a in answers.items():
        t = a.get("type")
        if t == "noul":
            out[qid] = a.get("noul")
        elif t == "choice":
            out[qid] = a.get("choice")
            for opt, p in (a.get("probabilities") or {}).items():
                out[f"{qid}__p_{opt}"] = p
            out[f"{qid}__confidence"] = a.get("confidence")
        elif t == "score":
            out[qid] = a.get("score")
            for lvl, p in (a.get("probabilities") or {}).items():
                out[f"{qid}__p{lvl}"] = p
            out[f"{qid}__confidence"] = a.get("confidence")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dataset", default="fixture")
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--questions", default="direct", choices=["direct", "bank", "structural"])
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = args.experiment or f"t03_{args.dataset}_{args.variant}_{args.questions}"
    ds = open_dataset(cfg, args.dataset)
    sites = select_sites(ds, args.limit, args.offset)
    bank = load_bank(cfg.bank_path)
    questions = pick_questions(bank, args.questions, args.variant)
    client = client_from_config(cfg, exp, use_cache=not args.no_cache)
    timing = TimingLog(cfg.timing_root / f"{exp}.csv")
    print(f"[t03] {exp}: {len(sites)} sites, {len(questions)} questions, model {cfg['model']}, "
          f"cumulative spend so far {client.cumulative_usd:.4f} USD (cap {cfg['MAX_USD']})")

    items, missing = [], []
    for s in sites:
        st = load_state(cfg, args.dataset, args.variant, s.id)
        if st is None:
            missing.append(s.id); continue
        items.append((st, questions, {"site_id": s.id, "label": s.label, "variant": args.variant,
                                      "dataset": args.dataset, "experiment": exp}))
    if missing:
        print(f"[t03] {len(missing)} sites without a {args.variant} state (run t02 first): {missing[:5]}...")

    rows = []

    def on_result(meta, res):
        timing.record(exp, meta["site_id"], args.variant, "decide",
                      (res.latency_ms or 0) / 1000, "cached" if res.cached else "")
        rows.append({"site_id": meta["site_id"], "label": meta["label"], "variant": args.variant,
                     "ok": res.ok, "cached": res.cached, "model": res.model_returned,
                     "request_id": res.request_id, "http_status": res.http_status,
                     "input_tokens": res.usage.get("input_tokens"), "output_tokens": res.usage.get("output_tokens"),
                     "cost_usd": res.cost_usd, "latency_ms": res.latency_ms, "error": res.error,
                     **flatten_answers(res.answers)})

    t0 = time.time()
    try:
        client.ask_many(items, workers=args.workers or cfg.get("concurrency", 4), on_result=on_result)
    except JevError as e:
        print(f"[t03] ABORT: {e}")
    wall = time.time() - t0

    df = pd.DataFrame(rows).sort_values("site_id") if rows else pd.DataFrame()
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    out = cfg.results_root / f"{exp}.csv"
    df.to_csv(out, index=False)
    st = client.session_stats()
    live = df[~df.cached] if len(df) else df
    summary = {"experiment": exp, "sites": len(sites), "requests": len(df), "ok": int(df.ok.sum()) if len(df) else 0,
               "http_failures": int((~df.ok).sum()) if len(df) else 0, "cached": int(df.cached.sum()) if len(df) else 0,
               "models_returned": ";".join(sorted(df.model.dropna().unique())) if len(df) else "",
               "all_model_match": bool(len(df) and (df.model.dropna() == cfg["expected_response_model"]).all()),
               "input_tokens_mean": float(df.input_tokens.mean()) if len(df) else None,
               "cost_usd_new_calls": round(st["session_usd"], 6), "cumulative_usd": round(st["cumulative_usd"], 6),
               "usd_per_1000_sites": round(1000 * df.cost_usd.mean(), 4) if len(df) else None,
               "latency_p50_ms": st.get("p50_ms"), "latency_p95_ms": st.get("p95_ms"), "wall_s": round(wall, 2),
               "workers": args.workers or cfg.get("concurrency", 4), "n_live_latency": st.get("n", 0)}
    pd.DataFrame([summary]).to_csv(cfg.results_root / f"{exp}_summary.csv", index=False)
    print(json.dumps(summary, indent=1))
    print(f"[t03] wrote {out}")
    timing.close(); client.close()


if __name__ == "__main__":
    main()
