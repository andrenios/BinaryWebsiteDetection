"""Detection, calibration and operational metrics (WORKORDER.md Section 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                             recall_score, accuracy_score, brier_score_loss)


def _np(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    m = ~np.isnan(p)
    return y[m], p[m]


def detection(y, p, threshold: float = 0.5) -> dict:
    y, p = _np(y, p)
    pred = (p >= threshold).astype(int)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "threshold": threshold,
           "f1": f1_score(y, pred, zero_division=0), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "accuracy": accuracy_score(y, pred)}
    if len(set(y)) == 2:
        out["auroc"] = roc_auc_score(y, p)
        out["auprc"] = average_precision_score(y, p)
    else:
        out["auroc"] = out["auprc"] = float("nan")
    return out


def best_f1_threshold(y, p) -> tuple[float, float]:
    """Threshold maximising F1 (fit on TRAIN only). Returns (threshold, f1)."""
    y, p = _np(y, p)
    best = (0.5, -1.0)
    for t in np.unique(p):
        f = f1_score(y, p >= t, zero_division=0)
        if f > best[1]:
            best = (float(t), float(f))
    return best


def ece_equal_mass(y, p, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """Expected calibration error with equal-mass bins; also returns the
    reliability table (bin, n, mean predicted, observed rate)."""
    y, p = _np(y, p)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    rows, ece = [], 0.0
    for b, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        conf, acc = p[idx].mean(), y[idx].mean()
        ece += len(idx) / len(p) * abs(acc - conf)
        rows.append({"bin": b, "n": len(idx), "p_mean": conf, "p_min": p[idx].min(), "p_max": p[idx].max(),
                     "observed": acc})
    return float(ece), pd.DataFrame(rows)


def brier(y, p) -> float:
    y, p = _np(y, p)
    return float(brier_score_loss(y, np.clip(p, 0, 1)))


def ppv_at_prevalence(y, p, threshold: float, prevalence: float, n_resamples: int = 1000,
                      n_total: int = 10000, seed: int = 2107) -> dict:
    """Paper-2 protocol: resample a population with the given prevalence from
    the positive and negative pools (with replacement), apply the threshold,
    report PPV with a 95 % bootstrap CI."""
    y, p = _np(y, p)
    rng = np.random.default_rng(seed)
    pos, neg = p[y == 1], p[y == 0]
    n_pos = max(1, int(round(n_total * prevalence)))
    n_neg = n_total - n_pos
    ppvs = []
    for _ in range(n_resamples):
        sp = rng.choice(pos, n_pos, replace=True)
        sn = rng.choice(neg, n_neg, replace=True)
        tp = (sp >= threshold).sum()
        fp = (sn >= threshold).sum()
        ppvs.append(tp / (tp + fp) if tp + fp > 0 else np.nan)
    ppvs = np.array(ppvs, dtype=float)
    return {"prevalence": prevalence, "threshold": threshold, "ppv": float(np.nanmean(ppvs)),
            "ppv_ci_low": float(np.nanpercentile(ppvs, 2.5)), "ppv_ci_high": float(np.nanpercentile(ppvs, 97.5)),
            "recall_at_threshold": float((pos >= threshold).mean()),
            "fpr_at_threshold": float((neg >= threshold).mean())}


def fp_per_1000_benign_at_recall(y, p, target_recall: float = 0.95) -> dict:
    """Threshold giving at least target_recall on positives; false positives per
    1,000 benign sites at that threshold."""
    y, p = _np(y, p)
    pos, neg = np.sort(p[y == 1]), p[y == 0]
    if len(pos) == 0:
        return {"target_recall": target_recall, "threshold": np.nan, "fp_per_1000_benign": np.nan}
    k = int(np.floor((1 - target_recall) * len(pos)))
    t = pos[min(k, len(pos) - 1)]
    return {"target_recall": target_recall, "threshold": float(t),
            "recall": float((pos >= t).mean()), "fp_per_1000_benign": float(1000 * (neg >= t).mean())}


def bootstrap_ci(y, p, fn, n: int = 1000, seed: int = 2107) -> tuple[float, float]:
    y, p = _np(y, p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def latency_summary(ms: pd.Series) -> dict:
    ms = ms.dropna()
    if not len(ms):
        return {"latency_p50_ms": np.nan, "latency_p95_ms": np.nan}
    return {"latency_p50_ms": float(ms.quantile(0.5)), "latency_p95_ms": float(ms.quantile(0.95)),
            "latency_mean_ms": float(ms.mean())}


def usd_per_1000(cost_usd: pd.Series) -> float:
    c = cost_usd.dropna()
    return float(1000 * c.mean()) if len(c) else float("nan")
