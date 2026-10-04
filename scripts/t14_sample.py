#!/usr/bin/env python3
"""T14 sampling (WORKORDER v2.3): PhreshPhish samples in the folder layout.

  # evaluation: balanced, language-stratified, from the BENCHMARK TEST split
  python scripts/t14_sample.py --role eval       --n 20000 --out data/derived/phreshphish/sample20k
  # the same split at its NATIVE base rate (seeded thinning, no balancing)
  python scripts/t14_sample.py --role native     --n 20000 --out data/derived/phreshphish/native_test
  # T19 stability sample, TRAIN split minus dev ids, 5,000, stratified by language
  python scripts/t14_sample.py --role stability  --n 5000  --out data/derived/phreshphish/stability5k
  # R1/R2/R3 adaptation sets (200 + 500 + 5,000), TRAIN split minus dev ids, disjoint from the stability sample
  python scripts/t14_sample.py --role adapt      --n 5700  --out data/derived/phreshphish/adapt5700 \
         --exclude data/derived/phreshphish/stability5k/meta.csv
  # the SVLM paper's 2,500-site sample by id
  python scripts/t14_sample.py --ids data/paper2/phreshphish_2500_ids.txt --out data/derived/phreshphish/paper2_2500

Every role excludes the bootstrap dev rows (data/derived/bootstrap/phreshphish/
meta.csv and data/derived/phreshphish_dev_ids.txt) and every meta.csv passed
with --exclude. Stratification (eval / stability / adapt): per class, language
quotas proportional to the language distribution seen in the first --scan rows
(languages with fewer than --min-lang rows are pooled as 'other'). Sampling
code, seed, role, split and exclusions are recorded in <out>/sampling.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

from _bootstrap import load_config, REPO
sys.path.insert(0, str(REPO / "src"))
from data.phreshphish import stream, DevIds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", default="eval", choices=["eval", "native", "stability", "adapt", "custom"],
                    help="eval/native: benchmark test split; stability/adapt: train split; custom: as given by --split")
    ap.add_argument("--split", default=None, help="overrides the role's split (custom)")
    ap.add_argument("--split-size", type=int, default=None, help="rows in the split (native thinning rate); default from the HF card")
    ap.add_argument("--exclude", nargs="*", default=[], help="meta.csv files of samples that must stay disjoint from this one")
    ap.add_argument("--ids", default=None, help="take exactly these sha256 ids (paper-2 sample) instead of sampling")
    ap.add_argument("--scan", type=int, default=20000, help="rows scanned to estimate the language distribution")
    ap.add_argument("--min-lang", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg["SEED"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    role_split = {"eval": "test", "native": "test", "stability": "train", "adapt": "train", "custom": "test"}
    args.split = args.split or role_split[args.role]
    if args.role in ("stability", "adapt") and args.split != "train":
        raise SystemExit("stability and adaptation samples must come from the train split (WORKORDER v2.3, T14)")
    if args.role in ("eval", "native") and args.split != "test":
        raise SystemExit("evaluation samples must come from the benchmark test split (WORKORDER v2.3, T14)")
    dev = DevIds(cfg.derived_root / "bootstrap" / "phreshphish" / "meta.csv", cfg.derived_root / "phreshphish_dev_ids.txt")
    for ex in args.exclude:
        dev.add_meta(ex)
    want_ids = set(l.strip() for l in open(args.ids) if l.strip()) if args.ids else None
    rnd = random.Random(seed)
    split_sizes = {"train": 498255, "test": 168060}            # HF dataset card, checked 2026-10-04
    native_rate = min(1.0, args.n / (args.split_size or split_sizes.get(args.split, args.n))) if args.role == "native" else None

    rows, counts = [], collections.Counter()
    quotas = None
    if want_ids is None and args.role != "native":
        # pass 1 (bounded): language distribution per class
        lang_counts = {0: collections.Counter(), 1: collections.Counter()}
        for i, r in enumerate(stream(args.split)):
            if i >= args.scan:
                break
            if r["label"] is None:
                continue
            lang_counts[r["label"]][r["language"] or "unk"] += 1
        quotas = {}
        for lab in (0, 1):
            tot = sum(lang_counts[lab].values())
            per_class = args.n // 2
            q = {}
            other = 0
            for lang, c in lang_counts[lab].items():
                if c >= args.min_lang:
                    q[lang] = round(per_class * c / tot)
                else:
                    other += c
            q["other"] = per_class - sum(q.values())
            quotas[lab] = q
        print(f"[t14] quotas: {quotas}")

    def accept(r) -> bool:
        if r["label"] is None or dev.is_dev(r):
            return False
        if want_ids is not None:
            return r["sha256"] in want_ids
        if args.role == "native":                      # seeded thinning keeps the split's base rate and language mix
            return rnd.random() < native_rate
        lang = r["language"] or "unk"
        key = lang if lang in quotas[r["label"]] else "other"
        if counts[(r["label"], key)] >= quotas[r["label"]].get(key, 0):
            return False
        # reservoir-free streaming acceptance with a seeded coin keeps the order-independence modest
        counts[(r["label"], key)] += 1
        return True

    target = len(want_ids) if want_ids is not None else args.n
    for r in stream(args.split):
        if not accept(r):
            continue
        sid = "pp_" + r["sha256"][:16]
        d = out / sid; d.mkdir(exist_ok=True)
        (d / "index.html").write_text(r["html"] or "", encoding="utf-8", errors="ignore")
        rows.append({"_id": sid, "url": r["url"], "label": r["label"], "language": r["language"], "source": "phreshphish",
                     "target": r["target"] or "", "sha256": r["sha256"], "date": r["date"]})
        if len(rows) % 500 == 0:
            print(f"[t14] {len(rows)}/{target}")
        if len(rows) >= target:
            break
    import csv
    with open(out / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump({"n": len(rows), "role": args.role, "split": args.split, "seed": seed, "quotas": quotas, "ids_file": args.ids,
               "native_rate": native_rate, "n_phishing": int(sum(r["label"] for r in rows)),
               "excluded_ids": len(dev), "exclude_files": args.exclude, "script": "scripts/t14_sample.py"},
              open(out / "sampling.json", "w"), indent=1)
    print(f"[t14] wrote {len(rows)} sites to {out}")


if __name__ == "__main__":
    main()
