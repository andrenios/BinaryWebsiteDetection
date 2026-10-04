"""Deterministic pre-filter (T10): declare a site benign without any model call
when the summariser's own fields show no data-collection surface at all."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

GATES = {
    # name -> function(state) -> True if the site is passed as benign without a model call
    "no_forms_no_phishy_links": lambda st: (
        st.get("signals", {}).get("num_forms", 0) == 0
        and st.get("signals", {}).get("num_password_fields", 0) == 0
        and st.get("signals", {}).get("num_sensitive_fields", 0) == 0
        and (st.get("hosts", {}) or {}).get("num_phishy_keyword_links", 0) == 0),
    "no_password_no_sensitive": lambda st: (
        st.get("signals", {}).get("num_password_fields", 0) == 0
        and st.get("signals", {}).get("num_sensitive_fields", 0) == 0),
    "no_forms": lambda st: st.get("signals", {}).get("num_forms", 0) == 0,
}


def gate_table(cfg, dataset: str, variant: str, ids: list[str]) -> pd.DataFrame:
    rows = []
    for sid in ids:
        p = cfg.derived_root / "states" / dataset / variant / f"{sid}.json"
        if not p.exists():
            continue
        st = json.load(open(p, encoding="utf-8"))
        rows.append({"site_id": sid, **{f"gate_{g}": bool(fn(st)) for g, fn in GATES.items()}})
    return pd.DataFrame(rows)


def gate_stats(df: pd.DataFrame, gate: str) -> dict:
    """df has label and gate_<gate>. Fraction gated and phishing missed by the gate."""
    g = df[f"gate_{gate}"]
    pos = df.label == 1
    return {"gate": gate, "n": len(df), "fraction_gated": float(g.mean()),
            "benign_gated": float(g[~pos].mean()) if (~pos).any() else float("nan"),
            "phishing_missed": int((g & pos).sum()),
            "phishing_missed_rate": float(g[pos].mean()) if pos.any() else float("nan"),
            "prevalence_before": float(pos.mean()),
            "prevalence_after_gate": float(pos[~g].mean()) if (~g).any() else float("nan")}


def apply_gate(p_model: pd.Series, gated: pd.Series) -> pd.Series:
    """Gated sites get probability 0 (benign); others keep the model's probability."""
    return p_model.where(~gated, 0.0)
