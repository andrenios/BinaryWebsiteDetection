#!/usr/bin/env python3
"""T14 sampling: a balanced, language-stratified PhreshPhish sample that
excludes the bootstrap dev ids, written in the folder layout.

  python scripts/t14_sample.py --n 20000 --out data/derived/phreshphish/sample20k
  python scripts/t14_sample.py --ids data/paper2/phreshphish_2500_ids.txt --out data/derived/phreshphish/paper2_2500

Streams the `test` (benchmark) split by default: it is the realistic-base-rate
benchmark split and disjoint from the `train` split used for the bootstrap.
Stratification: per class, language quotas proportional to the language
distribution seen in the first --scan rows (languages with fewer than
--min-lang rows are pooled as 'other'). Sampling code and seed recorded in
<out>/sampling.json so the sample is reproducible.
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
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default=None, help="take exactly these sha256 ids (paper-2 sample) instead of sampling")
    ap.add_argument("--scan", type=int, default=20000, help="rows scanned to estimate the language distribution")
    ap.add_argument("--min-lang", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg["SEED"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = DevIds(cfg.derived_root / "bootstrap" / "phreshphish" / "meta.csv")
    want_ids = set(l.strip() for l in open(args.ids) if l.strip()) if args.ids else None
    rnd = random.Random(seed)

    rows, counts = [], collections.Counter()
    quotas = None
    if want_ids is None:
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
    json.dump({"n": len(rows), "split": args.split, "seed": seed, "quotas": quotas, "ids_file": args.ids,
               "dev_ids_excluded": len(dev), "script": "scripts/t14_sample.py"}, open(out / "sampling.json", "w"), indent=1)
    print(f"[t14] wrote {len(rows)} sites to {out}")


if __name__ == "__main__":
    main()
