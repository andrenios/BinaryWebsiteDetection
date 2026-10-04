#!/usr/bin/env python3
"""T01: environment record and data inventory.

Writes
  results/env.txt                    python, packages, tesseract, platform
  results/t01_data_inventory.csv     one row per dataset present on this machine
  results/t01_putra_csv_counts.csv   rows and language counts of the Zenodo CSVs
  results/t01_split_status.csv       paper-2 split files present? counts match?
"""
from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

from _bootstrap import load_config, FolderDataset, DATASETS, REPO

sys.path.insert(0, str(REPO / "src"))
from data.putra import zenodo_meta  # noqa: E402


def write_env(cfg) -> Path:
    out = cfg.results_root / "env.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}",
             f"machine {platform.machine()}", f"config {cfg.path.name} profile={cfg['profile']}",
             f"model {cfg['model']} (expected response model {cfg['expected_response_model']})"]
    try:
        lines.append("tesseract " + subprocess.run(["tesseract", "--version"], capture_output=True,
                                                   text=True).stdout.splitlines()[0])
    except Exception as e:  # noqa: BLE001
        lines.append(f"tesseract unavailable: {e}")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    lines.append("--- pip freeze ---")
    lines.extend(freeze.splitlines())
    out.write_text("\n".join(lines) + "\n")
    (REPO / "requirements.txt").write_text(freeze)
    return out


def inventory(cfg) -> pd.DataFrame:
    rows = []
    for name, (rel, boot) in DATASETS.items():
        root = cfg.data_root / rel
        if not (root / "meta.csv").exists():
            rows.append({"dataset": name, "path": str(root), "present": False, "bootstrap_only": boot})
            continue
        ds = FolderDataset(root)
        c = ds.counts()
        langs = pd.Series([s.language for s in ds]).value_counts()
        rows.append({"dataset": name, "path": str(root), "present": True, "bootstrap_only": boot,
                     **c, "languages": ";".join(f"{k}:{v}" for k, v in langs.head(8).items())})
    ids = cfg.derived_root / "phreshphish_dev_ids.txt"
    rows.append({"dataset": "phreshphish_dev_ids", "path": str(ids), "present": ids.exists(),
                 "n": sum(1 for _ in open(ids)) if ids.exists() else 0})
    return pd.DataFrame(rows)


def putra_csv_counts(cfg) -> pd.DataFrame | None:
    d = cfg.data_root / "datasets" / "putra_zenodo"
    if not (d / "phishing.csv").exists():
        return None
    meta = zenodo_meta(d)
    rows = []
    for label, g in meta.groupby("label"):
        top = g["language"].value_counts()
        rows.append({"class": "phishing" if label else "benign", "rows": len(g),
                     "n_languages": g["language"].nunique(),
                     "top_languages": ";".join(f"{k}:{v}" for k, v in top.head(10).items())})
    rows.append({"class": "total", "rows": len(meta), "n_languages": meta["language"].nunique(),
                 "top_languages": ""})
    return pd.DataFrame(rows)


def split_status(cfg) -> pd.DataFrame:
    p2 = cfg.data_root / "paper2"
    rows = []
    for which, expected in (("train", 6160), ("test", 2631)):
        f = p2 / f"{which}.csv"
        if f.exists():
            n = sum(1 for _ in open(f)) - 1
            rows.append({"file": str(f), "present": True, "rows": n, "expected": expected, "match": n == expected})
        else:
            rows.append({"file": str(f), "present": False, "rows": 0, "expected": expected, "match": False})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    env = write_env(cfg)
    print(f"[t01] wrote {env}")
    inv = inventory(cfg)
    inv.to_csv(cfg.results_root / "t01_data_inventory.csv", index=False)
    print(inv.to_string())
    pc = putra_csv_counts(cfg)
    if pc is not None:
        pc.to_csv(cfg.results_root / "t01_putra_csv_counts.csv", index=False)
        print(pc.to_string())
    ss = split_status(cfg)
    ss.to_csv(cfg.results_root / "t01_split_status.csv", index=False)
    print(ss.to_string())


if __name__ == "__main__":
    main()
