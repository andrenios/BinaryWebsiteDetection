"""Configuration loading. Every script takes --config <file>; paths come from
the YAML, never from code (WORKORDER.md 1.3)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def _root(self, key: str) -> Path:
        p = Path(self.raw[key])
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def data_root(self) -> Path:
        return self._root("data_root")

    @property
    def results_root(self) -> Path:
        return self._root("results_root")

    @property
    def figures_root(self) -> Path:
        return self._root("figures_root")

    @property
    def reports_root(self) -> Path:
        return self._root("reports_root")

    @property
    def timing_root(self) -> Path:
        return self.results_root / "timing"

    @property
    def raw_responses_root(self) -> Path:
        return self.data_root / "raw_responses"

    @property
    def derived_root(self) -> Path:
        return self.data_root / "derived"

    @property
    def fixture_root(self) -> Path:
        return self.data_root / "fixture"

    @property
    def bank_path(self) -> Path:
        return self._root("bank_path")

    @property
    def api_key(self) -> str | None:
        return os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")


def load_config(path: str | Path = "config.yaml") -> Config:
    load_dotenv(REPO_ROOT / ".env")
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=p)


def add_common_args(ap) -> None:
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="process at most N sites")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip sites already present in the output")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached Jev responses")
