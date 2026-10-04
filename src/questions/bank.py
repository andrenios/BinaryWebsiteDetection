"""Load the question bank and select question subsets per state variant."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_bank(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _strip_local_keys(q: dict[str, Any]) -> dict[str, Any]:
    """Return the question as the API expects it (type, instructions, criteria)."""
    return {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}


def bank_questions(bank: dict[str, Any], variant: str = "S1",
                   only: list[str] | None = None) -> dict[str, dict]:
    """Indicator questions for a state variant. S2-only questions are dropped
    unless variant == 'S2'."""
    out = {}
    for qid, q in bank["questions"].items():
        if q.get("s2_only") and variant != "S2":
            continue
        if only is not None and qid not in only:
            continue
        out[qid] = _strip_local_keys(q)
    return out


def direct_question(bank: dict[str, Any]) -> dict[str, dict]:
    return {"q_direct": _strip_local_keys(bank["questions"]["q_direct"])}


def structural_questions(bank: dict[str, Any]) -> dict[str, dict]:
    return {qid: _strip_local_keys(q) for qid, q in bank["structural"].items()}
