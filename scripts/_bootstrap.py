"""Shared helpers for the task scripts: sys.path, dataset selection."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config import load_config, add_common_args  # noqa: E402
from data.folder import FolderDataset  # noqa: E402

DATASETS = {
    # name -> (root relative to data_root, is_bootstrap)
    "fixture": ("fixture", False),
    "bootstrap-phreshphish": ("derived/bootstrap/phreshphish", True),
    "bootstrap-putra": ("derived/bootstrap/putra", True),
}


def open_dataset(cfg, name: str) -> FolderDataset:
    rel, _ = DATASETS[name]
    return FolderDataset(cfg.data_root / rel)


def is_bootstrap(name: str) -> bool:
    return DATASETS[name][1]


def select_sites(ds: FolderDataset, limit: int | None, offset: int = 0, balanced: bool = False,
                 seed: int = 2107):
    """Deterministic selection. balanced=True takes limit/2 per class in id order
    after a seeded shuffle."""
    import random
    sites = ds.existing()
    sites.sort(key=lambda s: s.id)
    rnd = random.Random(seed)
    rnd.shuffle(sites)
    if balanced and limit:
        per = limit // 2
        pos = [s for s in sites if s.label == 1][:per]
        neg = [s for s in sites if s.label == 0][:limit - per]
        sites = sorted(pos + neg, key=lambda s: s.id)
        return sites
    sites = sites[offset:]
    if limit:
        sites = sites[:limit]
    return sites
