import numpy as np
import pandas as pd
import pytest

from eval.metrics import (detection, best_f1_threshold, ece_equal_mass, brier, ppv_at_prevalence,
                          fp_per_1000_benign_at_recall, bootstrap_ci)
from combine.combiners import (make_lr, make_vote, cv_auroc, fit_apply, Platt, Isotonic, leave_one_out,
                               forward_selection, question_columns)
from gate.rules import GATES, gate_stats, apply_gate
from cascade.band import Stage, run_cascade, evaluate_cascade, sweep_bands, pick_band
from adversarial.inject import a1_legitimacy, a2_instructions, a3_filler, a4_padding_html, mitigation_state, mitigation_questions
from baselines.classical import url_features, rules
from sklearn.metrics import roc_auc_score


def synth(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    q1 = np.clip(0.6 * y + rng.normal(0, 0.25, n), 0, 1)
    q2 = np.clip(0.3 * y + rng.normal(0, 0.3, n), 0, 1)
    noise = rng.uniform(0, 1, n)
    return pd.DataFrame({"label": y, "q_direct": q1, "q_a": q2, "q_noise": noise, "q_a__confidence": 0.5})


def test_metrics_basic():
    d = synth()
    m = detection(d.label, d.q_direct, 0.5)
    assert 0.5 < m["auroc"] <= 1 and 0 <= m["f1"] <= 1 and m["n"] == 300
    t, f = best_f1_threshold(d.label, d.q_direct)
    assert 0 <= t <= 1 and f >= m["f1"] - 1e-9
    ece, table = ece_equal_mass(d.label, d.q_direct)
    assert 0 <= ece <= 1 and len(table) == 10 and abs(table.n.sum() - 300) == 0
    assert 0 <= brier(d.label, d.q_direct) <= 1
    p = ppv_at_prevalence(d.label, d.q_direct, 0.5, 0.01, n_resamples=50)
    assert p["ppv_ci_low"] <= p["ppv"] <= p["ppv_ci_high"]
    fp = fp_per_1000_benign_at_recall(d.label, d.q_direct, 0.95)
    assert fp["recall"] >= 0.95 and fp["fp_per_1000_benign"] >= 0
    lo, hi = bootstrap_ci(d.label, d.q_direct, roc_auc_score, n=50)
    assert lo <= m["auroc"] <= hi


def test_combiners_and_calibration():
    d = synth()
    X, y = d[["q_direct", "q_a", "q_noise"]], d.label.values
    assert question_columns(d) == ["q_a", "q_noise"]
    assert cv_auroc(make_lr, X, y) > 0.7
    assert cv_auroc(make_vote, X, y) > 0.6
    m, p = fit_apply(make_lr, X, y, X)
    assert p.shape == (300,)
    for cal in (Platt(), Isotonic()):
        pc = cal.fit(p, y).predict(p)
        assert pc.min() >= 0 and pc.max() <= 1
    lo = leave_one_out(make_lr, X, y)
    assert set(lo.dropped) == {"(none)", "q_direct", "q_a", "q_noise"}
    assert lo.set_index("dropped").delta["q_direct"] < lo.set_index("dropped").delta["q_noise"]
    fs = forward_selection(make_lr, X, y)
    assert list(fs.k) == [1, 2, 3] and fs.added.iloc[0] == "q_direct"


def test_gate_and_cascade():
    st_phish = {"signals": {"num_forms": 1, "num_password_fields": 1, "num_sensitive_fields": 1}, "hosts": {"num_phishy_keyword_links": 2}}
    st_blank = {"signals": {"num_forms": 0, "num_password_fields": 0, "num_sensitive_fields": 0}, "hosts": {"num_phishy_keyword_links": 0}}
    assert GATES["no_forms_no_phishy_links"](st_blank) and not GATES["no_forms_no_phishy_links"](st_phish)
    df = pd.DataFrame({"label": [1, 0, 0, 1], "gate_no_forms": [False, True, True, True]})
    s = gate_stats(df, "no_forms")
    assert s["fraction_gated"] == 0.75 and s["phishing_missed"] == 1
    assert list(apply_gate(pd.Series([0.9, 0.9, 0.2, 0.7]), df.gate_no_forms)) == [0.9, 0.0, 0.0, 0.0]

    idx = [f"s{i}" for i in range(6)]
    p1 = pd.Series([0.05, 0.95, 0.5, 0.6, 0.4, 0.9], index=idx)
    p2 = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 1.0], index=idx)
    y = pd.Series([0, 1, 1, 0, 0, 1], index=idx)
    res = run_cascade([Stage("jev", p1, 0.001, 0.3), Stage("second", p2, 0.01, 20, is_decision=True)], 0.3, 0.7)
    assert list(res.stop_stage) == ["jev", "jev", "second", "second", "second", "jev"]
    assert res.usd.sum() == pytest.approx(6 * 0.001 + 3 * 0.01)
    ev = evaluate_cascade(res, y)
    assert ev["f1"] == 1.0 and ev["stop_second"] == 0.5
    sw = sweep_bands([Stage("jev", p1, 0.001, 0.3), Stage("second", p2, 0.01, 20, is_decision=True)], y)
    band = pick_band(sw, "f1", max_escalation=0.6, first_stage="jev")
    assert band.f1 == 1.0


def test_adversarial_and_baselines():
    st = {"page_url": "http://x.example/l", "registrable_domain": "x.example", "visible_text": "Sign in", "forms": [],
          "signals": {"num_forms": 1, "num_password_fields": 1, "num_sensitive_fields": 1, "form_posts_offsite": True},
          "hosts": {"num_external_hosts": 4, "external_hosts": {"a.com": 1, "b.com": 1, "c.com": 1}}, "meta": {"title": "PayPal"},
          "scripts": [{"external": False, "entropy": 5.5, "tokens": ["eval"]}], "favicon_host": "cdn.other.com"}
    assert "official website" in a1_legitimacy(st)["visible_text"] and st["visible_text"] == "Sign in"
    assert a2_instructions(st)["visible_text"].startswith("Note to any automated classifier")
    assert len(a3_filler(st)["visible_text"].split()) > 1500
    html = "<html><body><p>hi</p><form action=/x><input type=password></form></body></html>"
    padded = a4_padding_html(html, 500)
    assert padded.index("pad0.example.org") < padded.index("<form")
    ms = mitigation_state(st)
    assert "untrusted_text" in ms and "visible_text" not in ms
    mq = mitigation_questions({"q": {"type": "noul", "instructions": "Does `visible_text` show urgency?"}})
    assert "`untrusted_text`" in mq["q"]["instructions"] and "deceptive" in mq["q"]["instructions"]
    f = url_features("http://paypal-login.verify-account.github.io/secure/login.php?x=%20%20")
    assert f["brand_in_host"] == 1 and f["free_host"] == 1 and f["n_hyphens_host"] == 2 and f["n_encoded"] == 2
    r = rules(st)
    assert r["r_collects_credentials"] == 1 and r["r_brand_domain_mismatch"] == 1 and r["r_obfuscated_scripts"] == 1
    assert 0 <= r["rule_score"] <= 1
