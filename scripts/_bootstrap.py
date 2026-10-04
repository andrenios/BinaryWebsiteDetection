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


# ----------------------------------------------------------------------------- v2.3 helpers
def inst_suffix(instantiation: str) -> str:
    """Output-name suffix per instantiation: '' for the decision model (jev),
    '_twin-<model>' for the open-weight twin (T09)."""
    if not instantiation or instantiation == "jev":
        return ""
    return "_" + instantiation.replace(":", "-").replace("/", "-")


def out_name(base: str, instantiation: str = "jev") -> str:
    return f"{base}{inst_suffix(instantiation)}"


def decision_table(cfg, client, dataset: str, variant: str, split: str | None, questions: dict, experiment: str,
                   instantiation: str = "jev", limit: int | None = None, workers: int | None = None, reason: str = ""):
    """One decision table (site_id, label, q_* columns, ...) per dataset /
    variant / split for either instantiation. jev: the bank is sent through the
    cached Jev client (content-addressed cache; no API call when the responses
    exist). twin-<model>: the T09 table results/t09_<dataset>_<variant>_<model>_
    <split>_table.csv is read (T09 must have run). Returns (table, split_source)."""
    import pandas as pd
    from jev.run import run_questions
    sites, source = sites_for(cfg, dataset, split if split in ("train", "test") else None, limit=limit, reason=reason)
    if instantiation == "jev":
        df = run_questions(cfg, client, sites, dataset, variant, questions, experiment, split or "all", source, workers=workers)
        return (df[df.ok].reset_index(drop=True) if len(df) else df), source
    model = instantiation.split("-", 1)[1] if instantiation.startswith("twin-") else instantiation
    p = cfg.results_root / f"t09_{dataset}_{variant}_{model.replace(':', '_').replace('/', '_')}_{split or 'all'}_table.csv"
    if not p.exists():
        raise SystemExit(f"[{experiment}] twin table {p} not found; run scripts/t09_open_weight_twin.py first")
    df = pd.read_csv(p)
    want = {s.id for s in sites}
    df = df[df.site_id.isin(want)].reset_index(drop=True)
    return df, source
