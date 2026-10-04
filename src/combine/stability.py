"""Indicator stability and transfer-aware selection (T19, method M1), and the
semantic-versus-deterministic partition of the bank (T07 v2.3 / T19).

Per indicator and corpus: AUROC against the label (label 1 = phishing) and the
sign of the association (+ if AUROC >= 0.5). Classification over two corpora:
  stable         same sign on both and oriented AUROC > 0.60 on both
  artefact       oriented AUROC > 0.65 on one corpus and (< 0.55 or sign
                 reversed) on the other
  uninformative  otherwise
where the oriented AUROC is max(a, 1 - a).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# indicators that duplicate deterministic summary fields (kept so that T07 can
# report what the model adds over the field) versus the semantic ones that no
# regular expression over the summary computes (WORKORDER v2.3, T07).
SEMANTIC_QUESTIONS = ["q_brand_domain_mismatch", "q_brand_in_url_not_domain", "q_urgency", "q_reward_bait",
                      "q_language_brand_mismatch", "q_shell_page", "q_missing_legitimacy", "q_cloned_assets",
                      "q_suspicious_url_shape", "q_ip_or_freehost", "q_external_favicon"]
DETERMINISTIC_QUESTIONS = ["q_collects_credentials", "q_form_offsite", "q_obfuscated_scripts"]

STABLE_FLOOR = 0.60
ARTEFACT_HIGH = 0.65
ARTEFACT_LOW = 0.55


def indicator_auroc(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = df.label.astype(int).values
    rows = []
    for c in cols:
        p = df[c].astype(float).values
        m = ~np.isnan(p)
        a = roc_auc_score(y[m], p[m]) if len(set(y[m])) == 2 else np.nan
        rows.append({"question": c, "auroc": a, "sign": "+" if (np.isnan(a) or a >= 0.5) else "-",
                     "oriented_auroc": max(a, 1 - a) if not np.isnan(a) else np.nan, "n": int(m.sum())})
    return pd.DataFrame(rows)


def classify(a_src: float, s_src: str, a_tgt: float, s_tgt: str) -> str:
    if np.isnan(a_src) or np.isnan(a_tgt):
        return "uninformative"
    o_src, o_tgt = max(a_src, 1 - a_src), max(a_tgt, 1 - a_tgt)
    same = s_src == s_tgt
    if same and o_src > STABLE_FLOOR and o_tgt > STABLE_FLOOR:
        return "stable"
    if (o_src > ARTEFACT_HIGH and (o_tgt < ARTEFACT_LOW or not same)) or \
       (o_tgt > ARTEFACT_HIGH and (o_src < ARTEFACT_LOW or not same)):
        return "artefact"
    return "uninformative"


def _short(name: str) -> str:
    for suf in ("_train", "_stab", "_test", "_all"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def stability_table(src: pd.DataFrame, tgt: pd.DataFrame, cols: list[str],
                    src_name: str = "putra_train", tgt_name: str = "phresh_stab") -> pd.DataFrame:
    """Columns follow paper/tables.md Table 13: question, auroc_<src>, sign_<src'>,
    auroc_<tgt>, sign_<tgt'>, class, rank_<src'>, rank_<tgt'> where <src'> / <tgt'>
    drop the _train / _stab suffix (putra_train -> putra, phresh_stab -> phresh)."""
    a = indicator_auroc(src, cols).set_index("question")
    b = indicator_auroc(tgt, cols).set_index("question")
    ss, ts = _short(src_name), _short(tgt_name)
    if ss == ts:
        ss, ts = ss + "_src", ts + "_tgt"
    out = pd.DataFrame({"question": cols})
    out[f"auroc_{src_name}"] = a.loc[cols, "auroc"].values
    out[f"sign_{ss}"] = a.loc[cols, "sign"].values
    out[f"auroc_{tgt_name}"] = b.loc[cols, "auroc"].values
    out[f"sign_{ts}"] = b.loc[cols, "sign"].values
    out["class"] = [classify(x, sx, y, sy) for x, sx, y, sy in zip(out[f"auroc_{src_name}"], out[f"sign_{ss}"], out[f"auroc_{tgt_name}"], out[f"sign_{ts}"])]
    out[f"rank_{ss}"] = a.loc[cols, "oriented_auroc"].rank(ascending=False, method="min").astype(int).values
    out[f"rank_{ts}"] = b.loc[cols, "oriented_auroc"].rank(ascending=False, method="min").astype(int).values
    return out


def stable_set(table: pd.DataFrame) -> list[str]:
    return list(table[table["class"] == "stable"].question)
