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
    """Bootstrap ids (sha256 + url) that T14 must exclude."""

    def __init__(self, bootstrap_meta_csv: str | Path):
        self.sha = set()
        self.urls = set()
        p = Path(bootstrap_meta_csv)
        if p.exists():
            with open(p, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("sha256"):
                        self.sha.add(row["sha256"])
                    if row.get("url"):
                        self.urls.add(row["url"])

    def is_dev(self, row: dict) -> bool:
        return row.get("sha256") in self.sha or row.get("url") in self.urls

    def __len__(self) -> int:
        return len(self.sha | self.urls)
