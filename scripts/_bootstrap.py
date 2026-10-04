"""Shared helpers for the task scripts: sys.path, dataset selection, CLI."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config import load_config, add_common_args  # noqa: E402,F401
from data.folder import FolderDataset  # noqa: E402
from data.registry import REGISTRY, spec, open_dataset as _open, sites_for  # noqa: E402,F401

# kept for t02/t03/t05 (written before the registry); same semantics
DATASETS = {name: (s.root, s.bootstrap) for name, s in REGISTRY.items()}


def open_dataset(cfg, name: str) -> FolderDataset:
    return _open(cfg, name)


def is_bootstrap(name: str) -> bool:
    return spec(name).bootstrap


def select_sites(ds: FolderDataset, limit: int | None, offset: int = 0, balanced: bool = False,
                 seed: int = 2107):
    """Deterministic selection over a whole dataset (no split)."""
    import random
    sites = ds.existing()
    sites.sort(key=lambda s: s.id)
    random.Random(seed).shuffle(sites)
    if balanced and limit:
        per = limit // 2
        pos = [s for s in sites if s.label == 1][:per]
        neg = [s for s in sites if s.label == 0][:limit - per]
        return sorted(pos + neg, key=lambda s: s.id)
    sites = sites[offset:]
    return sites[:limit] if limit else sites


def add_split_args(ap) -> None:
    ap.add_argument("--dataset", default="bootstrap-phreshphish")
    ap.add_argument("--split", default="test", choices=["train", "test", "all"])
    ap.add_argument("--workers", type=int, default=None)


def banner(task: str, args, sites, source: str, client=None) -> None:
    extra = f", cumulative spend {client.cumulative_usd:.4f} USD (cap {client.max_usd})" if client else ""
    tag = " [PSEUDO SPLIT — development only]" if source == "pseudo" else ""
    boot = " [BOOTSTRAP — never in a report table]" if is_bootstrap(args.dataset) else ""
    print(f"[{task}] {args.dataset} split={args.split} ({source}){tag}{boot}: {len(sites)} sites{extra}", flush=True)
