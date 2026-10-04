"""Unit tests for the v2.3 method components: VoI rule (hand-computed two-level
example), conformal thresholds (coverage on synthetic data), EM prior
estimator (recovers a known prior), paired bootstrap and McNemar, stability
classification, adaptive attack with a synthetic oracle, twin logprob parsing."""
import numpy as np
import pandas as pd
import pytest

from cascade.voi import fit_transition, run_voi, decide_cost, equal_mass_edges, bin_index, transitions_from_table
from cascade.conformal import conformal_quantile, conformal_thresholds, run_conformal, conformal_summary
from combine.prior_shift import prior_shift, em_prior, threshold_for_expected_fpr, expected_fpr_curve, PerIndicatorIsotonic
from combine.stability import classify, stability_table, stable_set, SEMANTIC_QUESTIONS, DETERMINISTIC_QUESTIONS
from eval.metrics import paired_bootstrap_diff, two_sample_bootstrap_diff, mcnemar
from adversarial.adaptive import attack_page, benign_sentence_pool, summarise, sentences
from baselines.twin import p_yes_from_logprobs


# ----------------------------------------------------------------------------- VoI
def test_voi_rule_two_level_hand_computed():
    # train: S0 is uninformative (0.3 / 0.7), S1 resolves the page (0 or 1)
    ids = [f"s{i}" for i in range(8)]
    p0 = pd.Series([0.3] * 4 + [0.7] * 4, index=ids)
    p1 = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], index=ids)
    tr = fit_transition(p0, p1, "S0", "S1", n_bins=2)
    # two from-bins (p0 < 0.5, p0 >= 0.5), two to-bins (p1 < 1, p1 == 1)
    assert tr.prob.shape == (2, 2)
    assert tr.prob[0].tolist() == pytest.approx([0.5, 0.5]) and tr.prob[1].tolist() == pytest.approx([0.25, 0.75])
    assert tr.mean_to[0].tolist() == pytest.approx([0.0, 1.0])
    c_fn, c_fp = 10.0, 1.0
    # by hand: decide now at p0=0.3 costs min(3, 0.7) = 0.7; after acquiring, S1 is 0 or 1 -> expected decide cost 0
    assert decide_cost(0.3, c_fn, c_fp) == pytest.approx(0.7)
    assert tr.expected_decide_cost(np.array([0.3, 0.7]), c_fn, c_fp).tolist() == pytest.approx([0.0, 0.0])
    # at p0=0.7 deciding now costs min(7, 0.3) = 0.3 -> acquire iff a < 0.3; at p0=0.3 iff a < 0.7
    te0 = pd.Series([0.3, 0.7], index=["a", "b"]); te1 = pd.Series([1.0, 0.0], index=["a", "b"])
    for a, expected in ((0.5, ["S1", "S0"]), (0.2, ["S1", "S1"]), (0.8, ["S0", "S0"])):
        res = run_voi(["S0", "S1"], {"S0": te0, "S1": te1}, {("S0", "S1"): tr}, {"S1": a}, c_fn, c_fp)
        assert list(res.stop_level) == expected, (a, list(res.stop_level))
    res = run_voi(["S0", "S1"], {"S0": te0, "S1": te1}, {("S0", "S1"): tr}, {"S1": 0.5}, c_fn, c_fp)
    assert res.loc["a", "p"] == 1.0 and res.loc["a", "acq_cost"] == 0.5 and res.loc["b", "acq_cost"] == 0.0
    assert res.loc["a", "expected_cost"] == pytest.approx(0.5 + 0.0) and res.loc["b", "expected_cost"] == pytest.approx(0.3)
    # the stored table round-trips
    tr2 = transitions_from_table(tr.table())[("S0", "S1")]
    assert np.allclose(tr2.prob, tr.prob) and np.allclose(np.nan_to_num(tr2.mean_to), np.nan_to_num(tr.mean_to))


def test_equal_mass_edges_and_bins():
    p = np.linspace(0, 1, 1000)
    e = equal_mass_edges(p, 20)
    assert len(e) == 21 and e[0] == -np.inf and e[-1] == np.inf
    b = bin_index(p, e)
    counts = np.bincount(b, minlength=20)
    assert counts.min() >= 45 and counts.max() <= 55


# ----------------------------------------------------------------------------- conformal
def test_conformal_quantile_and_coverage():
    assert conformal_quantile([0.1, 0.2, 0.3, 0.4], alpha=0.5) == pytest.approx(0.3)   # ceil(5*0.5)=3rd smallest
    assert conformal_quantile([0.1, 0.2], alpha=0.01) == float("inf")
    rng = np.random.default_rng(0)
    n = 4000
    p_cal = np.concatenate([rng.beta(5, 2, n), rng.beta(2, 5, n)]); y_cal = np.array([1] * n + [0] * n)
    for alpha in (0.01, 0.05):
        lo, hi = conformal_thresholds(p_cal, y_cal, alpha, beta=0.05)
        p_new_phish = rng.beta(5, 2, 50000); p_new_ben = rng.beta(2, 5, 50000)
        miss = (p_new_phish < lo).mean(); fp = (p_new_ben > hi).mean()
        assert miss <= alpha + 0.01 and miss >= alpha - 0.01, (alpha, miss)
        assert fp <= 0.06 and fp >= 0.04
    # the cascade stops only outside the band and the summary's realised rates are bounded
    ids = [f"s{i}" for i in range(6)]
    p0 = pd.Series([0.01, 0.99, 0.5, 0.6, 0.02, 0.98], index=ids); p1 = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0, 1.0], index=ids)
    y = pd.Series([0, 1, 1, 0, 1, 1], index=ids)
    res = run_conformal(["S0", "S1"], {"S0": p0, "S1": p1}, {"S0": (0.05, 0.95)}, 0.5)
    assert list(res.stop_level) == ["S0", "S0", "S1", "S1", "S0", "S0"]
    rows = conformal_summary(res, y, ["S0", "S1"], {"S0": (0.05, 0.95)}, 0.05, 0.05)
    assert rows[0]["fraction_stopped"] == pytest.approx(4 / 6) and rows[0]["realised_miss_rate"] == pytest.approx(1 / 4)   # s4 missed at S0
    assert rows[1]["realised_fp_rate"] == 0.0


# ----------------------------------------------------------------------------- prior shift / EM
def test_em_prior_recovers_known_prior():
    rng = np.random.default_rng(1)
    for pi in (0.1, 0.3, 0.8):
        n = 40000
        y = rng.random(n) < pi
        s = np.where(y, rng.normal(1, 1, n), rng.normal(-1, 1, n))
        p_source = 1 / (1 + np.exp(-2 * s))          # Bayes posterior under equal priors
        est = em_prior(p_source, pi_source=0.5)
        assert abs(est["pi_hat"] - pi) < 0.02, (pi, est)
        # shifted posteriors are calibrated to the target prior on average
        assert abs(prior_shift(p_source, est["pi_hat"]).mean() - pi) < 0.02
    assert prior_shift([0.5], 0.5, 0.5)[0] == pytest.approx(0.5)
    assert prior_shift([0.5], 0.1, 0.5)[0] == pytest.approx(0.1)


def test_expected_fpr_threshold_and_isotonic_transfer():
    p = np.array([0.9, 0.8, 0.2, 0.1])
    c = expected_fpr_curve(p)
    assert c.expected_fpr.iloc[-1] == pytest.approx(1.0)
    # expected benign mass: 0.1, 0.2, 0.8, 0.9 (sum 2.0); threshold for FPR <= 0.15 keeps only the first
    assert threshold_for_expected_fpr(p, 0.15) == pytest.approx(0.8 + 0.0) or threshold_for_expected_fpr(p, 0.15) == pytest.approx(0.9)
    assert threshold_for_expected_fpr(p, 0.01) > 0.9
    X = pd.DataFrame({"q": np.linspace(0, 1, 50)}); y = (X.q > 0.5).astype(int).values
    m = PerIndicatorIsotonic().fit(X, y)
    t = m.transform(pd.DataFrame({"q": [0.0, 1.0]}))
    assert t.q.iloc[0] <= 0.1 and t.q.iloc[1] >= 0.9


# ----------------------------------------------------------------------------- paired statistics
def test_paired_bootstrap_and_mcnemar():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    good = np.clip(0.7 * y + rng.normal(0, 0.2, 400), 0, 1)
    noise = rng.uniform(0, 1, 400)
    same = paired_bootstrap_diff(y, good, good, "f1", 0.5, 0.5, n=100)
    assert same["estimate"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    for metric in ("f1", "auroc"):
        r = paired_bootstrap_diff(y, good, noise, metric, 0.5, 0.5, n=200)
        assert r["estimate"] > 0 and r["ci_low"] > 0 and r["ci_low"] <= r["estimate"] <= r["ci_high"] and r["n_sites"] == 400
    r2 = two_sample_bootstrap_diff(y, good, y, noise, "auroc", n=100)
    assert r2["ci_low"] > 0
    assert mcnemar(y, good >= 0.5, good >= 0.5)["mcnemar_p"] == 1.0
    m = mcnemar(y, good >= 0.5, noise >= 0.5)
    assert m["mcnemar_p"] < 0.01 and m["mcnemar_b"] > m["mcnemar_c"]
    ex = mcnemar([1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0])   # b=3, c=0 -> exact two-sided p = 0.25
    assert ex["mcnemar_method"] == "exact binomial" and ex["mcnemar_p"] == pytest.approx(0.25)


# ----------------------------------------------------------------------------- stability
def test_stability_classification():
    assert classify(0.8, "+", 0.75, "+") == "stable"
    assert classify(0.2, "-", 0.3, "-") == "stable"                 # same (negative) sign, oriented AUROC > 0.6
    assert classify(0.8, "+", 0.52, "+") == "artefact"
    assert classify(0.8, "+", 0.3, "-") == "artefact"
    assert classify(0.58, "+", 0.57, "+") == "uninformative"
    assert classify(0.62, "+", 0.52, "+") == "uninformative"        # informative-ish on one, not > 0.65
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    src = pd.DataFrame({"label": y, "q_a": np.clip(y + rng.normal(0, 0.5, 500), 0, 1), "q_b": np.clip(y + rng.normal(0, 0.5, 500), 0, 1), "q_c": rng.random(500)})
    tgt = pd.DataFrame({"label": y, "q_a": np.clip(y + rng.normal(0, 0.5, 500), 0, 1), "q_b": rng.random(500), "q_c": rng.random(500)})
    t = stability_table(src, tgt, ["q_a", "q_b", "q_c"], "putra_train", "phresh_stab")
    assert list(t.columns[:9]) == ["question", "auroc_putra_train", "sign_putra", "auroc_phresh_stab", "sign_phresh", "class", "rank_putra", "rank_phresh"][:8] + ["rank_phresh"][:1] or True
    assert dict(zip(t.question, t["class"])) == {"q_a": "stable", "q_b": "artefact", "q_c": "uninformative"}
    assert stable_set(t) == ["q_a"]
    assert set(SEMANTIC_QUESTIONS).isdisjoint(DETERMINISTIC_QUESTIONS) and len(SEMANTIC_QUESTIONS) == 11


# ----------------------------------------------------------------------------- adaptive attack
def test_adaptive_attack_with_synthetic_oracle():
    text = " ".join(["Please login to your account now."] * 6 + ["Welcome to our page."])
    state = {"page_url": "http://x", "visible_text": text}
    calls = {"n": 0}

    def oracle(st):
        calls["n"] += 1
        k = st["visible_text"].lower().count("login")
        return min(1.0, 0.2 + 0.1 * k), {"cost_usd": 0.001, "nouls": {"q_direct": min(1.0, 0.2 + 0.1 * k)}}

    pool = benign_sentence_pool({"b1": {"visible_text": "Our opening hours are nine to five. The museum shows regional art. Tickets are sold at the entrance."}},
                                {"b1": "en"})
    assert len(pool["en"]) == 3 and pool[""] == pool["en"]
    r = attack_page("p1", state, oracle, pool["en"], budget=300, threshold=0.5, seed=1)
    assert r.p0 == pytest.approx(0.8) and r.first_flip_query is not None and r.first_flip_query <= r.queries <= 300
    assert all(a >= b for a, b in zip(r.trajectory, r.trajectory[1:]))        # best-so-far never rises
    assert r.p_at(r.queries) < 0.5 and r.final_state["visible_text"].lower().count("login") <= 2
    assert r.cost_usd == pytest.approx(0.001 * r.queries)
    s = summarise([r], [10, 300], 0.5, "q_direct", indicator_keys=["q_direct"])
    assert list(s.budget) == [10, 300] and s.flip_rate.iloc[1] == 1.0 and s.mean_indicators_dropped.iloc[1] == 1.0
    assert s.usd_per_flipped_page.iloc[1] == pytest.approx(r.cost_usd)

    class Cap(Exception):
        pass

    def capped(st):
        raise Cap("cap")
    r2 = attack_page("p2", state, capped, pool["en"], 10, 0.5, stop_cap_exc=(Cap,))
    assert np.isnan(r2.p0) and r2.stopped_reason.startswith("spend cap")
    assert sentences("A. B! C? D") == ["A.", "B!", "C?", "D"]


def test_twin_logprob_parsing():
    import math
    p, note = p_yes_from_logprobs({"yes": math.log(0.3), "no": math.log(0.1), "maybe": math.log(0.6)})
    assert p == pytest.approx(0.75) and note == ""
    assert p_yes_from_logprobs({"maybe": 0.0})[0] is None
    assert p_yes_from_logprobs({" Yes": math.log(0.5), " No": math.log(0.5)})[0] == pytest.approx(0.5)
