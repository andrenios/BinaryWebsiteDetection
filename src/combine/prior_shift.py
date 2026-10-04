"""Label-free threshold transfer under prevalence shift (T21, method M3).

A combiner fitted on a balanced corpus outputs posteriors calibrated to a
source prior pi_s. For a target prior pi the calibrated posterior is obtained
by shifting the logit by  log(pi/(1-pi)) - log(pi_s/(1-pi_s)).  The target
prior is either known or estimated from unlabelled target scores with the EM
procedure of Saerens, Latinne and Decaestecker (2002). The operating threshold
for a target false-positive rate is then derived from the shifted posteriors
without labels: for calibrated posteriors the expected number of benign pages
scoring above t is sum_{p_i > t} (1 - p_i).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def prior_shift(p, pi_target: float, pi_source: float = 0.5) -> np.ndarray:
    """Posterior under the target prior from a posterior calibrated under pi_source."""
    pi_target = min(max(float(pi_target), EPS), 1 - EPS); pi_source = min(max(float(pi_source), EPS), 1 - EPS)
    delta = np.log(pi_target / (1 - pi_target)) - np.log(pi_source / (1 - pi_source))
    return expit(logit(p) + delta)


def em_prior(p, pi_source: float = 0.5, pi_init: float | None = None, max_iter: int = 1000, tol: float = 1e-8) -> dict:
    """Saerens-Latinne-Decaestecker EM: alternate between re-weighting the
    source posteriors by the current prior estimate and re-estimating the prior
    as the mean re-weighted posterior. Returns the estimate, iterations, and
    the trajectory's last change."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    pi = float(pi_init) if pi_init is not None else float(p.mean())
    pi = min(max(pi, EPS), 1 - EPS)
    for it in range(1, max_iter + 1):
        w1 = p * (pi / pi_source)
        w0 = (1 - p) * ((1 - pi) / (1 - pi_source))
        post = w1 / (w1 + w0)
        new = float(np.clip(post.mean(), EPS, 1 - EPS))
        change = abs(new - pi)
        pi = new
        if change < tol:
            break
    return {"pi_hat": pi, "iterations": it, "last_change": change}


def expected_fpr_curve(p_post) -> pd.DataFrame:
    """For calibrated posteriors, the expected FPR of threshold t is
    sum_{p_i >= t} (1 - p_i) / sum_i (1 - p_i). Returns the curve over the
    unique posterior values (descending)."""
    p = np.sort(np.asarray(p_post, dtype=float))[::-1]
    q = 1 - p
    cum = np.cumsum(q) / q.sum()
    return pd.DataFrame({"threshold": p, "expected_fpr": cum})


def threshold_for_expected_fpr(p_post, target_fpr: float) -> float:
    """Smallest threshold whose expected FPR (from calibrated posteriors) does
    not exceed target_fpr; label-free."""
    c = expected_fpr_curve(p_post)
    ok = c[c.expected_fpr <= target_fpr]
    if not len(ok):
        return float(c.threshold.iloc[0]) + EPS       # nothing admissible: flag nobody
    return float(ok.threshold.iloc[-1])


def source_fpr_at(p_source, y_source, threshold: float) -> float:
    p = np.asarray(p_source, dtype=float); y = np.asarray(y_source, dtype=int)
    neg = p[y == 0]
    return float((neg >= threshold).mean()) if len(neg) else float("nan")


class PerIndicatorIsotonic:
    """Per-indicator isotonic maps indicator -> P(phish) fitted on the source
    training split; applied to a target table before the combiner (T21's
    per-indicator calibration transfer)."""

    def __init__(self):
        self.maps = {}

    def fit(self, X: pd.DataFrame, y):
        from sklearn.isotonic import IsotonicRegression
        y = np.asarray(y, dtype=int)
        for c in X.columns:
            self.maps[c] = IsotonicRegression(out_of_bounds="clip").fit(X[c].astype(float).values, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy().astype(float)
        for c, m in self.maps.items():
            if c in out:
                out[c] = m.predict(out[c].values)
        return out
