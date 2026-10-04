"""Shared runner: send one question map to Jev for every site of a dataset on a
given state variant and return a decision table (one row per site).

Columns: site_id, label, dataset, variant, split, split_source, ok, cached,
model, request_id, http_status, input_tokens, output_tokens, cost_usd,
latency_ms, error, then one column per question (noul / choice / score with
their probabilities). Used by T04, T06, T06b, T07, T11b, T14, T15.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .client import JevClient, JevError, JevResult
from eval.timing import TimingLog


def load_state(cfg, dataset: str, variant: str, sid: str) -> dict | None:
    p = cfg.derived_root / "states" / dataset / variant / f"{sid}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def flatten_answers(answers: dict) -> dict:
    out: dict[str, Any] = {}
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


def run_questions(cfg, client: JevClient, sites, dataset: str, variant: str, questions: dict,
                  experiment: str, split: str | None = None, split_source: str = "all",
                  states: dict[str, dict] | None = None,
                  state_transform: Callable[[dict], dict] | None = None,
                  workers: int | None = None, timing: TimingLog | None = None,
                  extra_meta: dict | None = None) -> pd.DataFrame:
    """`states` overrides the on-disk states (used by the adversarial variants);
    `state_transform` is applied to each state before sending."""
    items, rows, missing = [], [], []
    for s in sites:
        st = states.get(s.id) if states is not None else load_state(cfg, dataset, variant, s.id)
        if st is None:
            missing.append(s.id)
            continue
        if state_transform:
            st = state_transform(st)
        meta = {"site_id": s.id, "label": s.label, "variant": variant, "dataset": dataset,
                "experiment": experiment, "split": split or "all",
                "workers": workers or cfg.get("concurrency", 4), **(extra_meta or {})}
        items.append((st, questions, meta))
    if missing:
        print(f"[run] {experiment}: {len(missing)} sites without a {variant} state (run t02 first)")

    def on_result(meta: dict, res: JevResult) -> None:
        if timing is not None:
            timing.record(experiment, meta["site_id"], variant, "decide", (res.latency_ms or 0) / 1000,
                          "cached" if res.cached else "")
        rows.append({"site_id": meta["site_id"], "label": meta["label"], "dataset": dataset,
                     "variant": variant, "split": split or "all", "split_source": split_source,
                     "ok": res.ok, "cached": res.cached, "model": res.model_returned,
                     "request_id": res.request_id, "http_status": res.http_status,
                     "input_tokens": res.usage.get("input_tokens"), "output_tokens": res.usage.get("output_tokens"),
                     "cost_usd": res.cost_usd, "latency_ms": res.latency_ms, "error": res.error,
                     **{k: v for k, v in (extra_meta or {}).items()},
                     **flatten_answers(res.answers)})

    t0 = time.time()
    try:
        client.ask_many(items, workers=workers or cfg.get("concurrency", 4), on_result=on_result)
    except JevError as e:
        print(f"[run] {experiment}: ABORT {e}")
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("site_id").reset_index(drop=True)
    df.attrs["wall_s"] = time.time() - t0
    return df


def summarise_run(df: pd.DataFrame, client: JevClient, experiment: str, expected_model: str) -> dict:
    st = client.session_stats()
    ok = df[df.ok] if len(df) else df
    return {"experiment": experiment, "requests": len(df), "ok": int(df.ok.sum()) if len(df) else 0,
            "http_failures": int((~df.ok).sum()) if len(df) else 0,
            "cached": int(df.cached.sum()) if len(df) else 0,
            "all_model_match": bool(len(ok) and (ok.model == expected_model).all()),
            "input_tokens_mean": float(ok.input_tokens.mean()) if len(ok) else None,
            "usd_per_1000_sites": float(1000 * ok.cost_usd.mean()) if len(ok) else None,
            "cost_usd_new_calls": round(st["session_usd"], 6), "cumulative_usd": round(st["cumulative_usd"], 6),
            "latency_p50_ms": st.get("p50_ms"), "latency_p95_ms": st.get("p95_ms"),
            "wall_s": round(df.attrs.get("wall_s", 0), 2)}


def save_table(cfg, df: pd.DataFrame, name: str) -> Path:
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    p = cfg.results_root / f"{name}.csv"
    df.to_csv(p, index=False)
    return p
