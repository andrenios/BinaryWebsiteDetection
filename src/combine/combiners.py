"""Combiners over the question-bank probabilities (T07) and calibration (T13).

All fitting happens on the training split (or by cross-validation within it);
`apply` is called once on the test split. Features are the noul columns of a
decision table; optional deterministic `signals` columns can be appended.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIGNAL_COLS = ["num_forms", "num_password_fields", "num_hidden_fields", "num_sensitive_fields",
               "form_posts_offsite", "num_iframes", "num_external_hosts", "num_phishy_keyword_links"]


def question_columns(df: pd.DataFrame, exclude: tuple[str, ...] = ("q_direct",)) -> list[str]:
    """Indicator-question columns of a decision table (noul columns named q_*
    without '__' suffix), minus the excluded ones."""
    return [c for c in df.columns if c.startswith("q_") and "__" not in c and c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])]


def signals_from_states(cfg, dataset: str, variant: str, ids: list[str]) -> pd.DataFrame:
    """Deterministic summariser fields per site (for the signals-only and
    signals+Jev models)."""
    import json
    rows = []
    for sid in ids:
        p = cfg.derived_root / "states" / dataset / variant / f"{sid}.json"
        if not p.exists():
            continue
        st = json.load(open(p, encoding="utf-8"))
        sig = dict(st.get("signals", {}))
        hosts = st.get("hosts", {}) or {}
        sig["num_external_hosts"] = hosts.get("num_external_hosts", 0)
        sig["num_phishy_keyword_links"] = hosts.get("num_phishy_keyword_links", 0)
        sig["form_posts_offsite"] = int(bool(sig.get("form_posts_offsite", False)))
        rows.append({"site_id": sid, **{c: sig.get(c, 0) for c in SIGNAL_COLS}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- models
def make_vote(weights: dict[str, float] | None = None):
    return VoteCombiner(weights)


class VoteCombiner:
    """Fixed weighted vote: weighted mean of the nouls (weights default 1)."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {}
        self.cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.cols = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame):
        w = np.array([self.weights.get(c, 1.0) for c in self.cols])
        p = (X[self.cols].values * w).sum(axis=1) / w.sum()
        return np.column_stack([1 - p, p])

    def get_params(self, deep=True):
        return {"weights": self.weights}


def make_lr(C: float = 1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))   # L2 is the default penalty


def make_catboost(seed: int = 2107, iterations: int = 500):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_seed=seed, verbose=False, iterations=iterations, thread_count=4)


COMBINERS = {"vote": make_vote, "lr": make_lr, "catboost": make_catboost}


def _proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def cv_scores(model_factory, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, seed: int = 2107) -> np.ndarray:
    """Out-of-fold probabilities on the training split."""
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=min(n_splits, int(np.bincount(y).min())), shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = model_factory().fit(X.iloc[tr], y[tr])
        oof[te] = _proba(m, X.iloc[te])
    return oof


def cv_auroc(model_factory, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, seed: int = 2107) -> float:
    oof = cv_scores(model_factory, X, y, n_splits, seed)
    return float(roc_auc_score(y, oof))


def fit_apply(model_factory, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame):
    m = model_factory().fit(X_tr, y_tr)
    return m, _proba(m, X_te)


def lr_odds_ratios(model, cols: list[str]) -> pd.DataFrame:
    """Odds ratios per standardised feature from the LR pipeline."""
    lr = model.named_steps["logisticregression"]
    coef = lr.coef_.ravel()
    return pd.DataFrame({"question": cols, "coef_std": coef, "odds_ratio_per_sd": np.exp(coef)}) \
        .sort_values("odds_ratio_per_sd", ascending=False)


def catboost_importances(model, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"question": cols, "importance": model.get_feature_importance()}) \
        .sort_values("importance", ascending=False)


def leave_one_out(model_factory, X: pd.DataFrame, y: np.ndarray, seed: int = 2107) -> pd.DataFrame:
    base = cv_auroc(model_factory, X, y, seed=seed)
    rows = [{"dropped": "(none)", "cv_auroc": base, "delta": 0.0}]
    for c in X.columns:
        a = cv_auroc(model_factory, X.drop(columns=[c]), y, seed=seed)
        rows.append({"dropped": c, "cv_auroc": a, "delta": a - base})
    return pd.DataFrame(rows).sort_values("delta")


def forward_selection(model_factory, X: pd.DataFrame, y: np.ndarray, seed: int = 2107) -> pd.DataFrame:
    """Greedy forward selection by CV AUROC; one row per k with the chosen set."""
    chosen: list[str] = []
    remaining = list(X.columns)
    rows = []
    while remaining:
        best = None
        for c in remaining:
            a = cv_auroc(model_factory, X[chosen + [c]], y, seed=seed)
            if best is None or a > best[1]:
                best = (c, a)
        chosen.append(best[0])
        remaining.remove(best[0])
        rows.append({"k": len(chosen), "added": best[0], "cv_auroc": best[1], "questions": "|".join(chosen)})
    return pd.DataFrame(rows)


def learning_curve(model_factory, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, y_te: np.ndarray,
                   sizes: list[int], seed: int = 2107, repeats: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        n = min(n, len(y_tr))
        for r in range(repeats):
            idx = rng.choice(len(y_tr), n, replace=False)
            if len(set(y_tr[idx])) < 2:
                continue
            _, p = fit_apply(model_factory, X_tr.iloc[idx], y_tr[idx], X_te)
            rows.append({"n_train": n, "repeat": r, "test_auroc": roc_auc_score(y_te, p)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- calibration
class Platt:
    def fit(self, p, y):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return self

    def predict(self, p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return self.lr.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


class Isotonic:
    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(p, float), y)
        return self

    def predict(self, p):
        return self.iso.predict(np.asarray(p, float))
