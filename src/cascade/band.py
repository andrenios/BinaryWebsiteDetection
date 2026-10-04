"""Escalation cascades (T11 model cascade, T11b evidence cascade).

A cascade is a sequence of stages. Each stage yields a probability for every
site; a site stops at the first stage whose probability falls outside the
uncertainty band [t_low, t_high] (or at the last stage). Costs are attached
per stage (USD per site, seconds per site, incl. evidence acquisition) and
accumulated for the sites that reach the stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from eval.metrics import detection, ppv_at_prevalence


@dataclass
class Stage:
    name: str
    prob: pd.Series            # index site_id -> probability (NaN = unavailable)
    usd_per_site: float = 0.0
    seconds_per_site: float = 0.0
    is_decision: bool = False  # stored hard decision (0/1); treated as final


def run_cascade(stages: list[Stage], t_low: float, t_high: float) -> pd.DataFrame:
    idx = stages[0].prob.index
    final = pd.Series(np.nan, index=idx)
    stop = pd.Series("", index=idx)
    usd = pd.Series(0.0, index=idx)
    sec = pd.Series(0.0, index=idx)
    active = pd.Series(True, index=idx)
    for k, st in enumerate(stages):
        p = st.prob.reindex(idx)
        reach = active & p.notna()
        usd[reach] += st.usd_per_site
        sec[reach] += st.seconds_per_site
        last = k == len(stages) - 1
        if st.is_decision or last:
            decide = reach
        else:
            decide = reach & ((p < t_low) | (p > t_high))
        final[decide] = p[decide]
        stop[decide] = st.name
        active = active & ~decide
        # sites with no probability at this stage keep the previous one
        if not last:
            continue
    # anything still active (no stage had a probability) keeps NaN
    return pd.DataFrame({"p": final, "stop_stage": stop, "usd": usd, "seconds": sec})


def evaluate_cascade(res: pd.DataFrame, y: pd.Series, threshold: float = 0.5,
                     prevalence_for_ppv: float = 0.01) -> dict:
    m = res.p.notna()
    det = detection(y[m], res.p[m], threshold)
    out = {**det, "mean_usd_per_site": float(res.usd.mean()), "usd_per_1000": float(1000 * res.usd.mean()),
           "mean_seconds_per_site": float(res.seconds.mean())}
    for stg, frac in res.stop_stage.value_counts(normalize=True).items():
        out[f"stop_{stg}"] = float(frac)
    if det["n_pos"] > 0 and det["n_pos"] < det["n"]:
        out["ppv_at_1pct"] = ppv_at_prevalence(y[m], res.p[m], threshold, prevalence_for_ppv, n_resamples=200)["ppv"]
    return out


def sweep_bands(stages: list[Stage], y: pd.Series, lows=None, highs=None, threshold: float = 0.5) -> pd.DataFrame:
    lows = lows if lows is not None else np.round(np.arange(0.05, 0.55, 0.05), 2)
    highs = highs if highs is not None else np.round(np.arange(0.5, 1.0, 0.05), 2)
    rows = []
    for lo, hi in product(lows, highs):
        if lo >= hi:
            continue
        r = run_cascade(stages, lo, hi)
        ev = evaluate_cascade(r, y, threshold)
        rows.append({"t_low": lo, "t_high": hi, **ev})
    return pd.DataFrame(rows)


def pick_band(sweep: pd.DataFrame, objective: str = "f1", max_escalation: float | None = None,
              first_stage: str | None = None) -> pd.Series:
    """Best band on the training sweep; optionally limit the escalation rate
    (1 - fraction stopping at the first stage)."""
    df = sweep.copy()
    if max_escalation is not None and first_stage is not None and f"stop_{first_stage}" in df:
        df = df[(1 - df[f"stop_{first_stage}"].fillna(0)) <= max_escalation]
    if not len(df):
        df = sweep
    return df.sort_values([objective, "mean_usd_per_site"], ascending=[False, True]).iloc[0]
