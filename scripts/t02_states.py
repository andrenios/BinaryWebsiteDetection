#!/usr/bin/env python3
"""T02: build the state variants for a dataset and time every stage.

  python scripts/t02_states.py --config config.yaml --dataset bootstrap-phreshphish [--limit N] [--variants S0,S1,S1b,S2,S3]

Writes
  data/derived/states/<dataset>/<variant>/<id>.json      one state per site/variant (S1b: S1b-<budget>)
  results/timing/t02_<dataset>.csv                       per site/stage wall seconds
  results/t02_<dataset>_summary.csv                      per variant: n, tokens (mean/p50/p95), truncation share, seconds
  results/t02_<dataset>_ocr.csv                          per site OCR chars/seconds/error (S2 only)
"""
from __future__ import annotations

import argparse
import json
import signal
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

from _bootstrap import load_config, add_common_args, open_dataset, select_sites, REPO

sys.path.insert(0, str(REPO / "src"))
from summarise.state import build_s0, build_s1, build_s3, build_s2, state_tokens  # noqa: E402
from summarise.ocr import ocr_image  # noqa: E402
from eval.timing import TimingLog  # noqa: E402


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def with_timeout(seconds: int, fn, *a, **kw):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return fn(*a, **kw)
    finally:
        signal.alarm(0)


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(p * (len(xs) - 1))))
    return xs[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dataset", default="bootstrap-phreshphish")
    ap.add_argument("--variants", default="S0,S1,S1b,S2,S3")
    ap.add_argument("--s3-timeout", type=int, default=180, help="seconds per site for the S3 truncation")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = open_dataset(cfg, args.dataset)
    sites = select_sites(ds, args.limit, args.offset)
    variants = args.variants.split(",")
    out_root = cfg.derived_root / "states" / args.dataset
    timing = TimingLog(cfg.timing_root / f"t02_{args.dataset}.csv")
    exp = f"t02_{args.dataset}"
    tm = cfg["token_counter_model"]
    budgets = cfg["summary_budgets"]

    stats: dict[str, dict[str, list]] = {}
    ocr_rows = []

    def put(variant: str, sid: str, state: dict, seconds: float, truncated: float | None = None):
        d = out_root / variant
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        st = stats.setdefault(variant, {"tokens": [], "seconds": [], "truncated": []})
        st["tokens"].append(state_tokens(state, tm))
        st["seconds"].append(seconds)
        if truncated is not None:
            st["truncated"].append(truncated)
        timing.record(exp, sid, variant, "summarise", seconds)

    t_start = time.time()
    for i, s in enumerate(sites, 1):
        if args.resume and all((out_root / v / f"{s.id}.json").exists() for v in variants if v not in ("S1b", "S2")):
            continue
        html = s.html()
        if "S0" in variants:
            st, t = build_s0(s.url); put("S0", s.id, st, t["summarise_s"])
        if "S1" in variants:
            st, t = build_s1(html, s.url, cfg["s1_budget"], tm)
            put("S1", s.id, st, t["summarise_s"], t["truncated"])
        if "S1b" in variants:
            for b in budgets:
                st, t = build_s1(html, s.url, b, tm)
                put(f"S1b-{b}", s.id, st, t["summarise_s"], t["truncated"])
        if "S2" in variants and s.screenshot_path is not None:
            text, secs, err = ocr_image(s.screenshot_path, cfg.get("tesseract_lang", "eng"))
            timing.record(exp, s.id, "S2", "ocr", secs, err or "")
            ocr_rows.append({"site_id": s.id, "label": s.label, "chars": len(text), "seconds": secs, "error": err or ""})
            st, t = build_s2(html, s.url, text, secs, cfg["s1_budget"], tm)
            put("S2", s.id, st, t["summarise_s"], t["truncated"])
        if "S3" in variants:
            t0 = time.perf_counter()
            try:
                st, t = with_timeout(args.s3_timeout, build_s3, html, s.url, cfg["s3_max_tokens"], tm)
                put("S3", s.id, st, t["summarise_s"])
            except Timeout:
                secs = time.perf_counter() - t0
                timing.record(exp, s.id, "S3", "summarise", secs, "timeout")
                stats.setdefault("S3", {"tokens": [], "seconds": [], "truncated": []})
                stats["S3"].setdefault("timeouts", []).append(s.id)
                print(f"[t02] S3 timeout after {secs:.0f}s on {s.id} ({len(html)} chars)")
        if i % 10 == 0 or i == len(sites):
            print(f"[t02] {i}/{len(sites)} sites, {time.time() - t_start:.0f}s elapsed", flush=True)

    rows = []
    for v, st in stats.items():
        tok, sec, tr = st["tokens"], st["seconds"], st["truncated"]
        rows.append({"variant": v, "n": len(tok), "tokens_mean": statistics.fmean(tok) if tok else None,
                     "tokens_p50": pct(tok, .5), "tokens_p95": pct(tok, .95), "tokens_max": max(tok) if tok else None,
                     "truncated_share": (sum(tr) / len(tr)) if tr else None,
                     "seconds_mean": statistics.fmean(sec) if sec else None, "seconds_p50": pct(sec, .5),
                     "seconds_p95": pct(sec, .95), "seconds_max": max(sec) if sec else None,
                     "timeouts": len(st.get("timeouts", []))})
    summ = pd.DataFrame(rows)
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    summ.to_csv(cfg.results_root / f"t02_{args.dataset}_summary.csv", index=False)
    print(summ.to_string())
    if ocr_rows:
        o = pd.DataFrame(ocr_rows)
        o.to_csv(cfg.results_root / f"t02_{args.dataset}_ocr.csv", index=False)
        print(f"[t02] OCR: {len(o)} screenshots, {int((o.chars >= 20).sum())} with >=20 chars, "
              f"{o.seconds.mean():.2f}s mean, {int((o.error != '').sum())} errors")
    timing.close()


if __name__ == "__main__":
    main()
