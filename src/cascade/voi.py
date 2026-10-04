"""Value-of-information evidence acquisition (T20, method component M2).

Levels are ordered by acquisition cost (S0 -> S1 -> S2; S0 -> S5 -> S1 -> S2 on
the fresh crawl). At level k the detector holds a probability p_k. With c_FN
and c_FP the costs of a missed phishing page and of a wrongly flagged benign
page, the expected cost of deciding now is

    decide(p) = min(c_FN * p, c_FP * (1 - p)).

The rule acquires level k+1 iff

    a_{k+1} + E[ decide(p_{k+1}) | p_k ] < decide(p_k)

where the conditional expectation comes from a transition model estimated on
the training split: p_k is binned into 20 equal-mass bins and the empirical
distribution of p_{k+1} in each bin is stored (as a 20 x 20 table of bin
probabilities with the mean next-level score per cell; the expectation uses
the per-cell mean, which is exact for the piecewise-linear decide() inside a
cell and otherwise a 20-bin approximation). The rule is myopic (one-step
lookahead), as specified in the work order. Acquisition costs a_k are given in
the same units as c_FP via a stated exchange rate (seconds per unit of c_FP).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def decide_cost(p: np.ndarray | float, c_fn: float, c_fp: float):
    p = np.asarray(p, dtype=float)
    return np.minimum(c_fn * p, c_fp * (1 - p))


def equal_mass_edges(p: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Bin edges with (approximately) equal mass; duplicates (ties at two
    decimals) are removed, so the number of bins can be smaller than n_bins."""
    p = np.asarray(p, dtype=float)
    cuts = np.unique(np.quantile(p, np.arange(1, n_bins) / n_bins))
    cuts = cuts[(cuts > p.min()) & (cuts <= p.max())]      # a cut at the minimum would leave an empty first bin
    return np.concatenate([[-np.inf], cuts, [np.inf]])


def bin_index(p, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, np.asarray(p, dtype=float), side="right") - 1, 0, len(edges) - 2)


@dataclass
class Transition:
    """Empirical p_k -> p_{k+1} model for one pair of levels."""
    level_from: str
    level_to: str
    edges_from: np.ndarray           # bins of p_k (equal mass on train)
    edges_to: np.ndarray             # bins of p_{k+1}
    prob: np.ndarray                 # [n_from, n_to] conditional probabilities
    mean_to: np.ndarray              # [n_from, n_to] mean p_{k+1} within the cell (NaN if empty)
    n_from: np.ndarray               # training sites per from-bin

    def expected_decide_cost(self, p_k, c_fn: float, c_fp: float) -> np.ndarray:
        """E[ decide(p_{k+1}) | p_k ] for every p_k."""
        cell_cost = np.where(np.isnan(self.mean_to), 0.0, decide_cost(np.nan_to_num(self.mean_to), c_fn, c_fp))
        per_bin = (self.prob * cell_cost).sum(axis=1)
        return per_bin[bin_index(p_k, self.edges_from)]

    def table(self) -> pd.DataFrame:
        rows = []
        for i in range(self.prob.shape[0]):
            for j in range(self.prob.shape[1]):
                rows.append({"level_from": self.level_from, "level_to": self.level_to, "from_bin": i,
                             "from_low": self.edges_from[i], "from_high": self.edges_from[i + 1], "n_from": int(self.n_from[i]),
                             "to_bin": j, "to_low": self.edges_to[j], "to_high": self.edges_to[j + 1],
                             "prob": float(self.prob[i, j]), "to_mean": float(self.mean_to[i, j])})
        return pd.DataFrame(rows)


def fit_transition(p_from: pd.Series, p_to: pd.Series, level_from: str, level_to: str, n_bins: int = 20) -> Transition:
    """Both series indexed by site id; only sites with both levels are used."""
    df = pd.concat([p_from.rename("a"), p_to.rename("b")], axis=1).dropna()
    ef, et = equal_mass_edges(df.a.values, n_bins), equal_mass_edges(df.b.values, n_bins)
    bi, bj = bin_index(df.a.values, ef), bin_index(df.b.values, et)
    nf, nt = len(ef) - 1, len(et) - 1
    counts = np.zeros((nf, nt)); sums = np.zeros((nf, nt))
    np.add.at(counts, (bi, bj), 1); np.add.at(sums, (bi, bj), df.b.values)
    n_from = counts.sum(axis=1)
    prob = np.divide(counts, n_from[:, None], out=np.zeros_like(counts), where=n_from[:, None] > 0)
    # an empty from-bin (cannot happen with equal-mass edges fitted on the same data, but be safe) acts as "no information":
    empty = n_from == 0
    if empty.any():
        marg = counts.sum(axis=0) / max(1, counts.sum())
        prob[empty] = marg
    mean_to = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return Transition(level_from, level_to, ef, et, prob, mean_to, n_from)


def transitions_from_table(df: pd.DataFrame) -> dict[tuple[str, str], Transition]:
    """Inverse of Transition.table() (to reuse a stored transition model)."""
    out = {}
    for (lf, lt), g in df.groupby(["level_from", "level_to"]):
        nf, nt = g.from_bin.max() + 1, g.to_bin.max() + 1
        prob = np.zeros((nf, nt)); mean_to = np.full((nf, nt), np.nan); n_from = np.zeros(nf)
        ef = np.full(nf + 1, np.nan); et = np.full(nt + 1, np.nan)
        for r in g.itertuples():
            prob[r.from_bin, r.to_bin] = r.prob; mean_to[r.from_bin, r.to_bin] = r.to_mean; n_from[r.from_bin] = r.n_from
            ef[r.from_bin], ef[r.from_bin + 1] = r.from_low, r.from_high
            et[r.to_bin], et[r.to_bin + 1] = r.to_low, r.to_high
        out[(lf, lt)] = Transition(lf, lt, ef, et, prob, mean_to, n_from)
    return out


def run_voi(levels: list[str], probs: dict[str, pd.Series], transitions: dict[tuple[str, str], Transition],
            acq_cost: dict[str, float], c_fn: float, c_fp: float, threshold: float = 0.5) -> pd.DataFrame:
    """Apply the myopic VoI rule per site. `probs[level]` are indexed by site id
    (NaN = level unavailable for that site; the rule then decides at the last
    available level). `acq_cost[level]` is the cost of acquiring that level, in
    units of c_FP. Returns per site: final p, stop level, acquisition cost paid,
    decision at `threshold`, expected cost (acq + decide at the final p)."""
    idx = probs[levels[0]].index
    final = pd.Series(np.nan, index=idx); stop = pd.Series("", index=idx); paid = pd.Series(0.0, index=idx)
    active = pd.Series(True, index=idx)
    for k, lv in enumerate(levels):
        p = probs[lv].reindex(idx)
        here = active & p.notna()
        if k == len(levels) - 1:
            decide = here
        else:
            nxt = levels[k + 1]
            tr = transitions[(lv, nxt)]
            e_next = pd.Series(tr.expected_decide_cost(p[here].values, c_fn, c_fp), index=p[here].index)
            now = pd.Series(decide_cost(p[here].values, c_fn, c_fp), index=p[here].index)
            has_next = probs[nxt].reindex(p[here].index).notna()
            acquire = (acq_cost.get(nxt, 0.0) + e_next < now) & has_next
            decide = pd.Series(False, index=idx); decide[acquire.index] = ~acquire
            paid[acquire.index[acquire]] += acq_cost.get(nxt, 0.0)
        final[decide] = p[decide]; stop[decide] = lv
        active &= ~decide
    exp_cost = paid + pd.Series(decide_cost(final.fillna(0.5).values, c_fn, c_fp), index=idx)
    return pd.DataFrame({"p": final, "stop_level": stop, "acq_cost": paid, "expected_cost": exp_cost,
                         "decision": (final >= threshold).astype(int)})
