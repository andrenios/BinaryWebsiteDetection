"""PhreshPhish (Hugging Face `phreshphish/phreshphish`).

Schema (checked 2026-10-04): sha256, url, label ("phish" | "benign"), target,
date, lang, lang_score, html. Splits: train 498,255 rows; test 168,060 rows
(the benchmark split with adjusted base rates, used by T14).

`dev_ids()` returns the bootstrap ids that every transfer sample must exclude
(WORKORDER.md 1.3b); `is_dev()` checks a row against them by sha256 and url.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

HF_NAME = "phreshphish/phreshphish"


def label_to_int(label: str) -> int | None:
    v = (label or "").strip().lower()
    if v == "phish":
        return 1
    if v == "benign":
        return 0
    return None


def stream(split: str = "train", cache_dir: str | Path | None = None) -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split=split, streaming=True,
                      cache_dir=str(cache_dir) if cache_dir else None)
    for ex in ds:
        yield {"sha256": ex["sha256"], "url": ex["url"], "label": label_to_int(ex["label"]),
               "target": ex.get("target"), "date": str(ex.get("date")),
               "language": ex.get("lang") or "", "html": ex["html"]}


class DevIds:
    """Ids that a PhreshPhish sample must exclude: the bootstrap dev rows
    (sha256 + url from the bootstrap meta.csv, or the short ids
    `pp_<sha256[:16]>` of data/derived/phreshphish_dev_ids.txt) and, via
    `add_meta`, any previously drawn sample (so that the evaluation, adaptation
    and stability samples are disjoint; WORKORDER v2.3, T14)."""

    def __init__(self, bootstrap_meta_csv: str | Path | None = None, ids_file: str | Path | None = None):
        self.sha = set()
        self.urls = set()
        self.short = set()
        if bootstrap_meta_csv:
            self.add_meta(bootstrap_meta_csv)
        if ids_file and Path(ids_file).exists():
            for line in open(ids_file, encoding="utf-8"):
                if line.strip():
                    self.short.add(line.strip())

    def add_meta(self, meta_csv: str | Path) -> "DevIds":
        p = Path(meta_csv)
        if p.exists():
            with open(p, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("sha256"):
                        self.sha.add(row["sha256"])
                    if row.get("url"):
                        self.urls.add(row["url"])
                    if row.get("_id"):
                        self.short.add(row["_id"])
        return self

    def is_dev(self, row: dict) -> bool:
        sha = row.get("sha256") or ""
        return sha in self.sha or row.get("url") in self.urls or ("pp_" + sha[:16]) in self.short

    def __len__(self) -> int:
        return len(self.sha | self.urls | self.short)
