"""Dataset registry and split handling shared by every task script.

A dataset is a folder in the shared layout (`folder.py`). Splits come from
`data/paper2/train.csv` / `test.csv` (`_id,label,filename`) when the dataset
declares a `split_dir`; otherwise a deterministic *pseudo* split (70/30,
stratified by label, SEED) is generated so that train-fit / test-evaluate code
paths run on the bootstrap samples. Every table produced from a pseudo split
carries `split_source = pseudo` and is development-only.

Reading the real test split is logged to results/test_set_access.log
(ground rule 6).
"""
from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from .folder import FolderDataset, Site


@dataclass
class DatasetSpec:
    name: str
    root: str                      # relative to data_root
    bootstrap: bool = False        # development only, never in a report table
    split_dir: str | None = None   # relative to data_root; holds train.csv / test.csv
    screenshots: bool = True
    notes: str = ""


REGISTRY: dict[str, DatasetSpec] = {
    "fixture": DatasetSpec("fixture", "fixture", bootstrap=True, notes="20 Putra sites, provisional (D7)"),
    "bootstrap-phreshphish": DatasetSpec("bootstrap-phreshphish", "derived/bootstrap/phreshphish",
                                         bootstrap=True, screenshots=False),
    "bootstrap-putra": DatasetSpec("bootstrap-putra", "derived/bootstrap/putra", bootstrap=True),
    # the real sets (folders to be created when the data arrives; see README)
    "putra": DatasetSpec("putra", "datasets/putra", split_dir="paper2",
                         notes="co-author's prepared copy, 8,791 sites; split from paper2/train.csv, test.csv"),
    "phreshphish-2500": DatasetSpec("phreshphish-2500", "derived/phreshphish/paper2_2500", screenshots=False,
                                    notes="the SVLM paper's 2,500-site sample (ids from paper2/)"),
    "phreshphish-20k": DatasetSpec("phreshphish-20k", "derived/phreshphish/sample20k", screenshots=False,
                                   notes="stratified 20,000-site transfer sample (scripts/t14_sample.py)"),
    "phishark": DatasetSpec("phishark", "datasets/phishark", notes="T16, controlled access"),
    "fresh": DatasetSpec("fresh", "datasets/fresh", notes="T17 own crawl"),
}


def spec(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise SystemExit(f"unknown dataset {name!r}; known: {', '.join(REGISTRY)}")
    return REGISTRY[name]


def open_dataset(cfg, name: str) -> FolderDataset:
    s = spec(name)
    root = cfg.data_root / s.root
    if not (root / "meta.csv").exists():
        raise SystemExit(f"dataset {name}: {root / 'meta.csv'} not found ({s.notes})")
    return FolderDataset(root)


def _pseudo_split(ds: FolderDataset, seed: int, test_frac: float = 0.30) -> dict[str, str]:
    """Deterministic, label-stratified: rank sites by sha1(seed|id) within each
    class and send the top test_frac to 'test'."""
    out: dict[str, str] = {}
    for label in (0, 1):
        ids = [s.id for s in ds if s.label == label]
        ids.sort(key=lambda i: hashlib.sha1(f"{seed}|{i}".encode()).hexdigest())
        n_test = round(len(ids) * test_frac)
        for k, i in enumerate(ids):
            out[i] = "test" if k < n_test else "train"
    return out


def split_map(cfg, name: str, reason: str = "") -> tuple[dict[str, str], str]:
    """Returns ({site_id: 'train'|'test'}, source) with source 'paper2' or 'pseudo'."""
    s = spec(name)
    if s.split_dir:
        d = cfg.data_root / s.split_dir
        tr, te = d / "train.csv", d / "test.csv"
        if tr.exists() and te.exists():
            out: dict[str, str] = {}
            for which, p in (("train", tr), ("test", te)):
                with open(p, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        out[row["_id"]] = which
            log = cfg.results_root / "test_set_access.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\tsplit_map\t{te}\t{reason}\n")
            return out, "paper2"
    ds = open_dataset(cfg, name)
    return _pseudo_split(ds, cfg["SEED"]), "pseudo"


def sites_for(cfg, name: str, split: str | None, limit: int | None = None, offset: int = 0,
              balanced: bool = False, seed: int | None = None, reason: str = "") -> tuple[list[Site], str]:
    """Sites of a dataset restricted to a split ('train' | 'test' | None = all),
    in a deterministic seeded order. Returns (sites, split_source)."""
    import random
    ds = open_dataset(cfg, name)
    sites = ds.existing()
    source = "all"
    if split in ("train", "test"):
        sm, source = split_map(cfg, name, reason=reason)
        sites = [s for s in sites if sm.get(s.id) == split]
    sites.sort(key=lambda s: s.id)
    random.Random(seed if seed is not None else cfg["SEED"]).shuffle(sites)
    if balanced and limit:
        per = limit // 2
        pos = [s for s in sites if s.label == 1][:per]
        neg = [s for s in sites if s.label == 0][:limit - per]
        sites = sorted(pos + neg, key=lambda s: s.id)
    else:
        sites = sites[offset:]
        if limit:
            sites = sites[:limit]
    return sites, source
