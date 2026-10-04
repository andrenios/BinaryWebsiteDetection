import json

import httpx
import pytest

from jev.client import JevClient, ModelMismatch, SpendCapReached, AuthError


def make_client(tmp_path, handler, **kw):
    transport = httpx.MockTransport(handler)
    defaults = dict(api_key="k", model="jev-1.13.0", expected_response_model="jev-1.13.0",
                    raw_root=tmp_path / "raw", max_usd=1.0, experiment="test",
                    transport=transport, max_retries=3)
    defaults.update(kw)
    return JevClient(**defaults)


def ok_response(model="jev-1.13.0", tokens=1000):
    return httpx.Response(200, json={"model": model, "answers": {"q": {"type": "noul", "noul": 0.9}},
                                     "usage": {"input_tokens": tokens, "output_tokens": 20}},
                          headers={"x-typesafe-request-id": "req_test"})


def test_ask_persists_caches_and_costs(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        body = json.loads(request.content)
        assert body["model"] == "jev-1.13.0" and "q" in body["questions"]
        assert request.headers["authorization"] == "Bearer k"
        return ok_response(tokens=1_000_000)

    c = make_client(tmp_path, handler)
    q = {"q": {"type": "noul", "instructions": "x?"}}
    r1 = c.ask({"page_url": "u"}, q, {"site_id": "a"})
    assert r1.ok and not r1.cached and r1.noul("q") == 0.9
    assert r1.cost_usd == pytest.approx(0.042)
    assert r1.request_id == "req_test"
    r2 = c.ask({"page_url": "u"}, q, {"site_id": "a"})
    assert r2.cached and calls["n"] == 1
    assert r2.cache_key == r1.cache_key
    # different state -> different key
    r3 = c.ask({"page_url": "v"}, q)
    assert r3.cache_key != r1.cache_key and calls["n"] == 2
    lines = [json.loads(l) for l in open(c.log_path)]
    assert len(lines) == 3 and lines[0]["response"]["model"] == "jev-1.13.0"
    assert "request" in lines[0] and lines[0]["request"]["state"] == {"page_url": "u"}
    assert c.cumulative_usd == pytest.approx(0.084)
    assert c.session_stats()["api_calls"] == 2 and c.session_stats()["cache_hits"] == 1


def test_no_cache_flag(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return ok_response()

    c = make_client(tmp_path, handler, use_cache=False)
    q = {"q": {"type": "noul", "instructions": "x?"}}
    c.ask("s", q); c.ask("s", q)
    assert calls["n"] == 2


def test_model_mismatch_aborts(tmp_path):
    c = make_client(tmp_path, lambda r: ok_response(model="jev-1.14.0"))
    with pytest.raises(ModelMismatch):
        c.ask("s", {"q": {"type": "noul", "instructions": "x?"}})


def test_spend_cap(tmp_path):
    c = make_client(tmp_path, lambda r: ok_response(tokens=30_000_000), max_usd=1.0)
    q = {"q": {"type": "noul", "instructions": "x?"}}
    c.ask("a", q)            # 1.26 USD -> ledger over cap
    with pytest.raises(SpendCapReached):
        c.ask("b", q)
    # ledger survives a restart
    c2 = make_client(tmp_path, lambda r: ok_response(), max_usd=1.0)
    assert c2.cumulative_usd == pytest.approx(1.26)
    with pytest.raises(SpendCapReached):
        c2.ask("c", q)


def test_401_stops(tmp_path):
    c = make_client(tmp_path, lambda r: httpx.Response(401, json={"detail": "bad key"}))
    with pytest.raises(AuthError):
        c.ask("s", {"q": {"type": "noul", "instructions": "x?"}})


def test_retry_on_429_then_ok(tmp_path, monkeypatch):
    monkeypatch.setattr("jev.client.time.sleep", lambda s: None)
    seq = iter([httpx.Response(429, json={}, headers={"retry-after": "0"}),
                httpx.Response(529, json={}), ok_response()])
    c = make_client(tmp_path, lambda r: next(seq))
    r = c.ask("s", {"q": {"type": "noul", "instructions": "x?"}})
    assert r.ok and r.attempts == 3


def test_validation_error_is_recorded_not_raised(tmp_path):
    c = make_client(tmp_path, lambda r: httpx.Response(422, json={"detail": "bad"}))
    r = c.ask("s", {"q": {"type": "noul", "instructions": "x?"}}, {"site_id": "z"})
    assert not r.ok and r.http_status == 422 and r.cost_usd == 0
    rec = json.loads(open(c.log_path).readline())
    assert rec["ok"] is False and rec["site_id"] == "z"
