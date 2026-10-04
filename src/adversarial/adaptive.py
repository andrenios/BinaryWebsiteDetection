"""A5: adaptive black-box attack on `visible_text` (T15 v2.3).

The attacker holds a phishing page's state, a pool of sentences taken from
benign pages of the same language, and black-box access to a scoring oracle
that returns the detector's probability for a state (q_direct, the lr bank
score, or the open-weight twin). Each query proposes one edit of
`visible_text` (insert a benign sentence at a random position, delete a
sentence, or replace a sentence by a benign one), keeps the edit when the
returned probability falls (greedy random local search), and stops when the
query budget is exhausted or the probability is below the operating threshold
with a margin. The run records the probability after every query, so the
results at budgets 50, 200 and 1,000 are prefixes of the same search.

The oracle is any callable state -> (probability, info dict); `info` may
carry the per-question probabilities (bank target), the cost in USD and the
cache flag. SpendCapReached from the Jev client ends the attack cleanly.
"""
from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str | None) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if s and s.strip()]


def benign_sentence_pool(states: dict[str, dict], languages: dict[str, str], min_len: int = 20, max_len: int = 240) -> dict[str, list[str]]:
    """Sentences of benign pages, grouped by language ('' pools everything)."""
    pool: dict[str, list[str]] = {}
    for sid, st in states.items():
        lang = (languages.get(sid) or "unk").lower()
        for s in sentences(st.get("visible_text")):
            if min_len <= len(s) <= max_len:
                pool.setdefault(lang, []).append(s)
                pool.setdefault("", []).append(s)
    return pool


def propose(text_sents: list[str], pool: list[str], rnd: random.Random) -> tuple[list[str], str]:
    ops = ["insert", "replace"] + (["delete"] if len(text_sents) > 1 else [])
    op = rnd.choice(ops)
    out = list(text_sents)
    if op == "insert" or not out:
        out.insert(rnd.randint(0, len(out)), rnd.choice(pool)); op = "insert"
    elif op == "delete":
        out.pop(rnd.randrange(len(out)))
    else:
        out[rnd.randrange(len(out))] = rnd.choice(pool)
    return out, op


@dataclass
class AttackResult:
    site_id: str
    p0: float
    trajectory: list[float] = field(default_factory=list)        # best p after each query
    queries: int = 0
    first_flip_query: int | None = None
    cost_usd: float = 0.0
    final_state: dict | None = None
    info0: dict = field(default_factory=dict)
    info_final: dict = field(default_factory=dict)
    stopped_reason: str = ""

    def p_at(self, budget: int) -> float:
        if not self.trajectory:
            return self.p0
        return self.trajectory[min(budget, len(self.trajectory)) - 1]

    def flipped_at(self, budget: int, threshold: float) -> bool:
        return self.first_flip_query is not None and self.first_flip_query <= budget


def attack_page(site_id: str, state: dict, oracle: Callable[[dict], tuple[float, dict]], pool: list[str],
                budget: int, threshold: float, seed: int = 2107, margin: float = 0.0,
                stop_cap_exc: type[BaseException] | tuple = ()) -> AttackResult:
    rnd = random.Random(f"{seed}|{site_id}")
    try:
        p0, info0 = oracle(state)
    except stop_cap_exc as e:  # type: ignore[misc]
        r = AttackResult(site_id, float("nan")); r.stopped_reason = f"spend cap: {e}"; return r
    res = AttackResult(site_id, p0, info0=info0, final_state=copy.deepcopy(state))
    res.cost_usd += float(info0.get("cost_usd", 0.0) or 0.0)
    res.queries = 1   # the clean query counts (the attacker needs it)
    best_p, best_sents, best_info = p0, sentences(state.get("visible_text")), info0
    res.trajectory.append(best_p)
    if best_p < threshold - margin:
        res.first_flip_query = 1; res.stopped_reason = "already below threshold"; return res
    if not pool:
        res.stopped_reason = "empty sentence pool"; return res
    while res.queries < budget:
        cand_sents, op = propose(best_sents, pool, rnd)
        cand = copy.deepcopy(state); cand["visible_text"] = " ".join(cand_sents)
        try:
            p, info = oracle(cand)
        except stop_cap_exc as e:  # type: ignore[misc]
            res.stopped_reason = f"spend cap: {e}"; break
        res.queries += 1
        res.cost_usd += float(info.get("cost_usd", 0.0) or 0.0)
        if p < best_p:
            best_p, best_sents, best_info = p, cand_sents, info
            res.final_state = cand
        res.trajectory.append(best_p)
        if res.first_flip_query is None and best_p < threshold - margin:
            res.first_flip_query = res.queries
            res.stopped_reason = "flipped"; break
    else:
        res.stopped_reason = res.stopped_reason or "budget exhausted"
    res.info_final = best_info
    return res


def summarise(results: list[AttackResult], budgets: list[int], threshold: float, score_name: str,
              indicator_drop: float = 0.2, indicator_keys: list[str] | None = None) -> pd.DataFrame:
    """Columns of paper/tables.md Table 12b: score, budget, n_pages, flip_rate,
    mean_prob_reduction, mean_queries_to_flip, usd_per_flipped_page,
    mean_indicators_dropped (bank target only: indicators whose probability fell
    by more than `indicator_drop` in successful attacks)."""
    rows = []
    valid = [r for r in results if not np.isnan(r.p0)]
    for b in budgets:
        flips = [r for r in valid if r.flipped_at(b, threshold)]
        red = [r.p0 - r.p_at(b) for r in valid]
        # cost attributable to the first b queries (per-query mean cost x queries used up to b)
        usd = sum((r.cost_usd / max(1, r.queries)) * min(r.queries, b) for r in valid)
        dropped = []
        if indicator_keys:
            for r in flips:
                a, c = r.info0.get("nouls", {}), r.info_final.get("nouls", {})
                dropped.append(sum(1 for k in indicator_keys if k in a and k in c and (a[k] - c[k]) > indicator_drop))
        rows.append({"score": score_name, "budget": b, "n_pages": len(valid),
                     "flip_rate": len(flips) / len(valid) if valid else np.nan,
                     "mean_prob_reduction": float(np.mean(red)) if red else np.nan,
                     "mean_queries_to_flip": float(np.mean([r.first_flip_query for r in flips])) if flips else np.nan,
                     "usd_per_flipped_page": usd / len(flips) if flips else np.nan,
                     "mean_indicators_dropped": float(np.mean(dropped)) if dropped else np.nan})
    return pd.DataFrame(rows)
