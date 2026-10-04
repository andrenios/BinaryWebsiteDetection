#!/usr/bin/env python3
"""Build data/fixture/ (20 sites: 10 phishing, 10 benign) from any Putra copy in
the folder layout. The ids are fixed in data/fixture/ids.txt (committed); when
that file does not exist yet it is created from the source with SEED.

  python scripts/make_fixture.py --config config.yaml --source data/derived/bootstrap/putra
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from _bootstrap import load_config, FolderDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default=None, help="Putra folder with meta.csv (default: bootstrap putra)")
    ap.add_argument("--per-class", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    src_root = Path(args.source) if args.source else cfg.data_root / "derived/bootstrap/putra"
    ds = FolderDataset(src_root)
    fx = cfg.fixture_root
    fx.mkdir(parents=True, exist_ok=True)
    ids_file = fx / "ids.txt"

    if ids_file.exists():
        ids = [l.strip() for l in ids_file.read_text().splitlines() if l.strip()]
        print(f"[fixture] using {len(ids)} fixed ids from {ids_file}")
    else:
        rnd = random.Random(cfg["SEED"])
        pos = sorted(s.id for s in ds.existing() if s.label == 1 and s.screenshot_path)
        neg = sorted(s.id for s in ds.existing() if s.label == 0 and s.screenshot_path)
        rnd.shuffle(pos); rnd.shuffle(neg)
        ids = sorted(pos[:args.per_class] + neg[:args.per_class])
        ids_file.write_text("\n".join(ids) + "\n")
        print(f"[fixture] wrote {len(ids)} new ids to {ids_file} (seed {cfg['SEED']})")

    rows = []
    missing = []
    for sid in ids:
        try:
            s = ds.by_id(sid)
        except KeyError:
            missing.append(sid); continue
        dst = fx / sid
        dst.mkdir(exist_ok=True)
        for name in ("index.html", "original.html", "screenshot.jpg"):
            p = s.dir / name
            if p.exists():
                shutil.copyfile(p, dst / name)
        rows.append({"_id": sid, "url": s.url, "label": s.label, "language": s.language, "source": s.source})
    with open(fx / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["_id", "url", "label", "language", "source"])
        w.writeheader(); w.writerows(rows)
    print(f"[fixture] {len(rows)} sites copied to {fx}; missing in source: {missing}")


if __name__ == "__main__":
    main()
