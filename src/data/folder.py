"""Loader for the shared folder layout used by Putra, the bootstrap samples and
the fixture:

    <root>/<id>/index.html
    <root>/<id>/screenshot.jpg      (optional)
    <root>/meta.csv                 columns: _id,url,label,language[,source]
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Site:
    id: str
    url: str
    label: int                 # 1 phishing, 0 benign
    language: str
    source: str
    dir: Path

    @property
    def html_path(self) -> Path:
        return self.dir / "index.html"

    @property
    def screenshot_path(self) -> Path | None:
        for name in ("screenshot.jpg", "screenshot.jpeg", "screenshot.png"):
            p = self.dir / name
            if p.exists():
                return p
        return None

    def html(self) -> str:
        return self.html_path.read_text(encoding="utf-8", errors="ignore")


class FolderDataset:
    def __init__(self, root: str | Path, meta_csv: str | Path | None = None):
        self.root = Path(root)
        self.meta_csv = Path(meta_csv) if meta_csv else self.root / "meta.csv"
        self._sites: list[Site] = []
        with open(self.meta_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row["_id"]
                self._sites.append(Site(id=sid, url=row.get("url", ""),
                                        label=int(row["label"]),
                                        language=row.get("language", "") or "",
                                        source=row.get("source", "") or "",
                                        dir=self.root / sid))

    def __len__(self) -> int:
        return len(self._sites)

    def __iter__(self) -> Iterator[Site]:
        return iter(self._sites)

    def __getitem__(self, i: int) -> Site:
        return self._sites[i]

    def ids(self) -> list[str]:
        return [s.id for s in self._sites]

    def by_id(self, sid: str) -> Site:
        for s in self._sites:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def subset(self, ids: list[str]) -> list[Site]:
        want = set(ids)
        return [s for s in self._sites if s.id in want]

    def existing(self) -> list[Site]:
        return [s for s in self._sites if s.html_path.exists()]

    def counts(self) -> dict[str, int]:
        return {"n": len(self._sites),
                "phishing": sum(s.label for s in self._sites),
                "benign": sum(1 - s.label for s in self._sites),
                "with_html": sum(s.html_path.exists() for s in self._sites),
                "with_screenshot": sum(s.screenshot_path is not None for s in self._sites)}
