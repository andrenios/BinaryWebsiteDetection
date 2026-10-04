"""Conformal stopping thresholds for the evidence cascade (T20).

Split-conformal with per-class nonconformity:
  lower stop (declare benign, do not acquire further): nonconformity 1 - p_k on
    the phishing pages of the calibration half; threshold at the
    finite-sample-corrected (1 - alpha) quantile q, so that a new phishing
    page has 1 - p_k > q (i.e. p_k < 1 - q, is stopped as benign) with
    probability at most alpha, marginally over pages.
  upper stop (declare phishing): nonconformity p_k on benign pages, tolerance
    beta, symmetric.
Pages with t_low <= p_k <= t_high acquire the next level; the last level
decides at the operating threshold. Thresholds are calibrated per level on the
same calibration half, so the per-level miss guarantee holds marginally per
level; the reported realised rates are cumulative over levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile: the ceil((n+1)(1-alpha))-th
    smallest score (inf if that exceeds n)."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def conformal_thresholds(p_cal, y_cal, alpha: float, beta: float) -> tuple[float, float]:
    """(t_low, t_high): stop as benign when p < t_low, stop as phishing when p > t_high."""
    p = np.asarray(p_cal, dtype=float); y = np.asarray(y_cal, dtype=int)
    q_low = conformal_quantile(1 - p[y == 1], alpha)      # nonconformity 1 - p on phishing pages
    q_high = conformal_quantile(p[y == 0], beta)           # nonconformity p on benign pages
    t_low = 1 - q_low if np.isfinite(q_low) else -np.inf  # inf quantile -> never stop as benign
    t_high = q_high if np.isfinite(q_high) else np.inf
    return float(t_low), float(t_high)


def run_conformal(levels: list[str], probs: dict[str, pd.Series], thresholds: dict[str, tuple[float, float]],
                  threshold_final: float = 0.5) -> pd.DataFrame:
    """Per site: final p, stop level, decision (1 phishing / 0 benign)."""
    idx = probs[levels[0]].index
    final = pd.Series(np.nan, index=idx); stop = pd.Series("", index=idx); dec = pd.Series(np.nan, index=idx)
    active = pd.Series(True, index=idx)
    for k, lv in enumerate(levels):
        p = probs[lv].reindex(idx)
        here = active & p.notna()
        last = k == len(levels) - 1
        if last:
            s_ben = here & (p < threshold_final); s_phi = here & (p >= threshold_final)
        else:
            lo, hi = thresholds[lv]
            nxt_avail = probs[levels[k + 1]].reindex(idx).notna()
            s_ben = here & ((p < lo) | ~nxt_avail & (p < threshold_final))
            s_phi = here & ((p > hi) | ~nxt_avail & (p >= threshold_final))
        for mask, d in ((s_ben, 0), (s_phi, 1)):
            final[mask] = p[mask]; stop[mask] = lv; dec[mask] = d
        active &= ~(s_ben | s_phi)
    return pd.DataFrame({"p": final, "stop_level": stop, "decision": dec})


def conformal_summary(res: pd.DataFrame, y: pd.Series, levels: list[str], thresholds: dict[str, tuple[float, float]],
                      alpha: float, beta: float) -> list[dict]:
    """Per level: fraction stopped, realised miss rate (phishing pages declared
    benign at or before this level / all phishing pages) and realised
    false-positive rate (benign declared phishing at or before this level / all
    benign pages), cumulative over levels."""
    y = y.reindex(res.index)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    rows = []
    order = {lv: i for i, lv in enumerate(levels)}
    stopped_rank = res.stop_level.map(order)
    for i, lv in enumerate(levels):
        upto = stopped_rank <= i
        lo, hi = thresholds.get(lv, (np.nan, np.nan))
        rows.append({"alpha": alpha, "beta": beta, "level": lv, "threshold_low": lo, "threshold_high": hi,
                     "fraction_stopped": float((res.stop_level == lv).mean()),
                     "realised_miss_rate": float(((res.decision == 0) & upto & (y == 1)).sum() / n_pos) if n_pos else np.nan,
                     "realised_fp_rate": float(((res.decision == 1) & upto & (y == 0)).sum() / n_neg) if n_neg else np.nan,
                     "n": int(len(res))})
    return rows
