#!/usr/bin/env python3
"""T05: determinism and structural invariance.

  python scripts/t05_determinism.py --config config.yaml --light            # 50 bootstrap PhreshPhish sites
  python scripts/t05_determinism.py --config config.yaml --dataset <train>   # 200 training sites (Phase 1 proper)

Part A (determinism): `q_direct` alone on S1, repeated --repeats times with the
cache bypassed; the per-request JSONL keeps every response. Reports the share
of sites whose probabilities are identical across repeats, mean/max absolute
pairwise difference.
Part B (structure): one extra request with q_direct + q_direct_choice +
q_direct_score; agreement of the thresholded answers and Spearman rank
correlations between q_direct, P(phishing) of the choice and the score
expectation. Also compares q_direct alone vs q_direct with companions.

Writes results/t05_<tag>_repeats.csv, results/t05_<tag>_structure.csv,
results/t05_<tag>_summary.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _bootstrap import load_config, add_common_args, open_dataset, select_sites, REPO

sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config, JevError  # noqa: E402
from questions.bank import load_bank, direct_question, structural_questions  # noqa: E402
from eval.timing import TimingLog  # noqa: E402


def load_state(cfg, dataset, variant, sid):
    p = cfg.derived_root / "states" / dataset / variant / f"{sid}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dataset", default="bootstrap-phreshphish")
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--light", action="store_true", help="50 balanced bootstrap sites")
    ap.add_argument("--n", type=int, default=None, help="number of sites (balanced); default 50 light / 200 full")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n = args.n or (50 if args.light else 200)
    tag = f"{'light_' if args.light else ''}{args.dataset}_{args.variant}"
    exp = f"t05_{tag}"
    ds = open_dataset(cfg, args.dataset)
    sites = select_sites(ds, n, balanced=True, seed=cfg["SEED"])
    bank = load_bank(cfg.bank_path)
    q_direct = direct_question(bank)
    q_struct = {**q_direct, **structural_questions(bank)}
    timing = TimingLog(cfg.timing_root / f"{exp}.csv")

    # Part A: repeats bypass the cache (otherwise repeat 2 and 3 would be served locally).
    client = client_from_config(cfg, exp, use_cache=False)
    print(f"[t05] {exp}: {len(sites)} sites x {args.repeats} repeats + 1 structural request; "
          f"cumulative spend so far {client.cumulative_usd:.4f} USD")

    done: dict[tuple[str, int], float] = {}
    if args.resume and client.log_path.exists():
        for line in open(client.log_path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("ok") and r.get("part") == "A":
                done[(r["site_id"], r["repeat"])] = r["response"]["answers"]["q_direct"]["noul"]
        print(f"[t05] resume: {len(done)} repeat responses already on disk")

    rep_rows, struct_rows = [], []
    t0 = time.time()
    try:
        items = []
        for s in sites:
            st = load_state(cfg, args.dataset, args.variant, s.id)
            if st is None:
                print(f"[t05] no state for {s.id}; run t02"); continue
            for k in range(args.repeats):
                if (s.id, k) in done:
                    rep_rows.append({"site_id": s.id, "label": s.label, "repeat": k, "q_direct": done[(s.id, k)], "cached_log": True})
                    continue
                items.append((st, q_direct, {"site_id": s.id, "label": s.label, "repeat": k, "part": "A",
                                             "variant": args.variant, "dataset": args.dataset}))
            items.append((st, q_struct, {"site_id": s.id, "label": s.label, "repeat": -1, "part": "B",
                                         "variant": args.variant, "dataset": args.dataset}))

        def on_result(meta, res):
            timing.record(exp, meta["site_id"], args.variant, "decide", (res.latency_ms or 0) / 1000, meta["part"])
            if not res.ok:
                print(f"[t05] HTTP failure {res.http_status} on {meta['site_id']}: {res.error}")
                return
            if meta["part"] == "A":
                rep_rows.append({"site_id": meta["site_id"], "label": meta["label"], "repeat": meta["repeat"],
                                 "q_direct": res.noul("q_direct"), "input_tokens": res.usage.get("input_tokens"),
                                 "latency_ms": res.latency_ms, "request_id": res.request_id})
            else:
                a = res.answers
                ch, sc = a.get("q_direct_choice", {}), a.get("q_direct_score", {})
                struct_rows.append({"site_id": meta["site_id"], "label": meta["label"],
                                    "q_direct_with_companions": a.get("q_direct", {}).get("noul"),
                                    "choice": ch.get("choice"), "choice_p_phishing": (ch.get("probabilities") or {}).get("phishing"),
                                    "choice_confidence": ch.get("confidence"),
                                    "score": sc.get("score"), "score_confidence": sc.get("confidence"),
                                    **{f"score_p{l}": p for l, p in (sc.get("probabilities") or {}).items()},
                                    "input_tokens": res.usage.get("input_tokens"), "request_id": res.request_id})

        client.ask_many(items, workers=cfg.get("concurrency", 4), on_result=on_result)
    except JevError as e:
        print(f"[t05] ABORT: {e}")

    cfg.results_root.mkdir(parents=True, exist_ok=True)
    rep = pd.DataFrame(rep_rows)
    st = pd.DataFrame(struct_rows)
    rep.to_csv(cfg.results_root / f"{exp}_repeats.csv", index=False)
    st.to_csv(cfg.results_root / f"{exp}_structure.csv", index=False)

    summary = {"experiment": exp, "sites": len(sites), "repeats": args.repeats}
    if len(rep):
        wide = rep.pivot_table(index="site_id", columns="repeat", values="q_direct")
        full = wide.dropna()
        diffs = [np.abs(full[a] - full[b]) for a, b in combinations(full.columns, 2)]
        pair = pd.concat(diffs, axis=1)
        summary.update({"sites_with_all_repeats": len(full),
                        "share_identical": float((full.nunique(axis=1) == 1).mean()),
                        "mean_abs_diff": float(pair.values.mean()), "max_abs_diff": float(pair.values.max()),
                        "share_diff_le_0.01": float((pair.max(axis=1) <= 0.01).mean()),
                        "share_flip_at_0.5": float(((full > 0.5).nunique(axis=1) > 1).mean())})
        if len(st):
            m = full.mean(axis=1).rename("q_direct_alone").reset_index().merge(st, on="site_id")
            m = m.dropna(subset=["q_direct_with_companions", "choice_p_phishing", "score"])
            summary.update({
                "n_structure": len(m),
                "agree_direct_vs_choice": float(((m.q_direct_alone > 0.5) == (m.choice == "phishing")).mean()),
                "agree_direct_vs_score_ge2": float(((m.q_direct_alone > 0.5) == (m.score >= 2.0)).mean()),
                "spearman_direct_choice": float(spearmanr(m.q_direct_alone, m.choice_p_phishing).correlation),
                "spearman_direct_score": float(spearmanr(m.q_direct_alone, m.score).correlation),
                "spearman_choice_score": float(spearmanr(m.choice_p_phishing, m.score).correlation),
                "mean_abs_diff_alone_vs_with_companions": float((m.q_direct_alone - m.q_direct_with_companions).abs().mean()),
                "spearman_alone_vs_with_companions": float(spearmanr(m.q_direct_alone, m.q_direct_with_companions).correlation)})
    cs = client.session_stats()
    summary.update({"api_calls": cs["api_calls"], "cost_usd_new_calls": round(cs["session_usd"], 6),
                    "cumulative_usd": round(cs["cumulative_usd"], 6), "latency_p50_ms": cs.get("p50_ms"),
                    "latency_p95_ms": cs.get("p95_ms"), "wall_s": round(time.time() - t0, 1)})
    pd.DataFrame([summary]).to_csv(cfg.results_root / f"{exp}_summary.csv", index=False)
    print(json.dumps(summary, indent=1))
    timing.close(); client.close()


if __name__ == "__main__":
    main()
