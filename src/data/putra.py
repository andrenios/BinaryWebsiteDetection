"""Putra (2023) loaders.

Two shapes are supported:

1. The co-author's prepared copy (preferred, WORKORDER.md 1.2):
     <data_root>/paper2/train.csv, test.csv      (_id,label,filename)
     <data_root>/datasets/putra/<id>/index.html + <id>/*.jpg
2. The raw Zenodo record 8041387 unpacked (or the bootstrap sample): the two
   class CSVs `phishing.csv` / `not-phishing.csv` (columns incl. _id,url,
   language) plus site folders. A `meta.csv` in the folder layout of
   `folder.py` is produced from them by `zenodo_meta()`.

The test split is read-only before Phase 2; `load_split()` logs every access
to the test file into results/test_set_access.log (ground rule 6).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import pandas as pd

from .folder import FolderDataset, Site

csv.field_size_limit(sys.maxsize)

PUTRA_CSV_COLS = ["_id", "url", "language"]


def zenodo_meta(csv_dir: str | Path) -> pd.DataFrame:
    """_id,url,label,language,target from the two Zenodo class CSVs."""
    csv_dir = Path(csv_dir)
    frames = []
    for name, label in (("phishing.csv", 1), ("not-phishing.csv", 0)):
        p = csv_dir / name
        if not p.exists():
            continue
        cols = PUTRA_CSV_COLS + (["brands"] if label == 1 else [])
        df = pd.read_csv(p, usecols=cols, dtype=str, keep_default_na=False)
        df["label"] = label
        df["target"] = df["brands"] if "brands" in df else ""
        df["source"] = "putra"
        frames.append(df[["_id", "url", "label", "language", "source", "target"]])
    if not frames:
        raise FileNotFoundError(f"no phishing.csv / not-phishing.csv under {csv_dir}")
    return pd.concat(frames, ignore_index=True)


def write_meta_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_split(paper2_dir: str | Path, which: str, access_log: str | Path | None = None,
               reason: str = "") -> pd.DataFrame:
    """train.csv / test.csv (_id,label,filename). Every test read is logged."""
    p = Path(paper2_dir) / f"{which}.csv"
    if which == "test" and access_log is not None:
        Path(access_log).parent.mkdir(parents=True, exist_ok=True)
        with open(access_log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{p}\t{reason}\n")
    return pd.read_csv(p, dtype={"_id": str})


def load_folder(root: str | Path, meta_csv: str | Path | None = None) -> FolderDataset:
    return FolderDataset(root, meta_csv)
