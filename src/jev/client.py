"""JevClient: pinned model, verbatim persistence, content-addressed cache,
retries, latency, cost accumulation and a hard spend cap (WORKORDER.md T03).

API shape (docs.typesafe.ai/api, checked 2026-10-04):
  POST {api_base}/v1/systemone  Authorization: Bearer <key>
  body: {"state": ..., "model": "jev-1.13.0", "questions": {id: {type, instructions[, criteria]}}}
  resp: {"model": "jev-1.13.0", "answers": {id: {"type": "noul", "noul": p} | ...},
         "usage": {"input_tokens": n, "output_tokens": m}}
  The response carries no cost field; cost is computed from input_tokens and
  the published price (reports/decisions.md, D2). Request id: header
  x-typesafe-request-id. Errors: 400/422 validation, 401 auth, 429/529 retry.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx


class JevError(Exception):
    pass


class AuthError(JevError):
    """401: stop and report; never try other credentials."""


class ModelMismatch(JevError):
    """response.model differs from the pinned model: abort and report."""


class SpendCapReached(JevError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class JevResult:
    ok: bool
    cache_key: str
    state_hash: str
    cached: bool
    model_returned: str | None
    request_id: str | None
    answers: dict[str, Any]
    usage: dict[str, int]
    cost_usd: float
    latency_ms: float | None
    http_status: int | None
    error: str | None = None
    attempts: int = 0

    def noul(self, qid: str) -> float | None:
        a = self.answers.get(qid)
        return None if a is None else a.get("noul")


@dataclass
class JevClient:
    api_key: str
    model: str
    expected_response_model: str
    raw_root: Path                      # data/raw_responses/
    api_base: str = "https://api.typesafe.ai"
    price_per_mtok_input: float = 0.042
    price_per_mtok_output: float = 0.0
    max_usd: float = 1.0
    bank_version: str = "bank_v1"
    timeout_s: float = 60.0
    max_retries: int = 6
    use_cache: bool = True
    experiment: str = "default"         # name of the JSONL log under raw_root
    transport: Any = None               # httpx transport override (tests)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _latencies: list[float] = field(default_factory=list, repr=False)
    _session_cost: float = field(default=0.0, repr=False)
    _session_calls: int = field(default=0, repr=False)
    _session_cached: int = field(default=0, repr=False)
    _ledger_total: float = field(default=0.0, repr=False)
    _http: httpx.Client | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        self.raw_root = Path(self.raw_root)
        (self.raw_root / "cache").mkdir(parents=True, exist_ok=True)
        self._ledger_total = self._read_ledger()
        self._http = httpx.Client(timeout=self.timeout_s, http2=False, transport=self.transport)

    @property
    def ledger_path(self) -> Path:
        return self.raw_root / "spend_ledger.jsonl"

    @property
    def log_path(self) -> Path:
        return self.raw_root / f"{self.experiment}.jsonl"

    def _read_ledger(self) -> float:
        total = 0.0
        if self.ledger_path.exists():
            with open(self.ledger_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        total += float(json.loads(line)["cost_usd"])
                    except Exception:  # noqa: BLE001
                        continue
        return total

    @property
    def cumulative_usd(self) -> float:
        """Spend over all runs (ledger) including this session's new calls."""
        return self._ledger_total

    @property
    def session_usd(self) -> float:
        return self._session_cost

    # ------------------------------------------------------------------ cache
    def cache_key(self, state: Any, questions: dict[str, dict]) -> str:
        payload = {"model": self.model, "bank_version": self.bank_version,
                   "state": state, "questions": questions}
        return sha256_hex(canonical_json(payload))

    def _cache_path(self, key: str) -> Path:
        return self.raw_root / "cache" / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if self.use_cache and p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _cache_put(self, key: str, record: dict) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        os.replace(tmp, p)

    def _append(self, path: Path, record: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ cost
    def cost_of(self, usage: dict[str, int]) -> float:
        return (usage.get("input_tokens", 0) * self.price_per_mtok_input
                + usage.get("output_tokens", 0) * self.price_per_mtok_output) / 1e6

    # ------------------------------------------------------------------ HTTP
    def _post(self, body: dict) -> tuple[int, dict | None, dict, float, int]:
        """Returns (status, json_body, headers, latency_ms, attempts). Retries
        429/529/5xx and transport errors with exponential backoff."""
        url = f"{self.api_base}/v1/systemone"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._http.post(url, content=data, headers=headers)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                time.sleep(delay + random.uniform(0, 0.25))
                delay = min(delay * 2, 16)
                continue
            latency_ms = (time.perf_counter() - t0) * 1000
            if r.status_code in (429, 529) or 500 <= r.status_code < 600:
                ra = r.headers.get("retry-after")
                try:
                    wait = float(ra) if ra else delay
                except ValueError:
                    wait = delay
                time.sleep(wait + random.uniform(0, 0.25))
                delay = min(delay * 2, 16)
                continue
            try:
                js = r.json()
            except ValueError:
                js = None
            return r.status_code, js, dict(r.headers), latency_ms, attempt
        raise JevError(f"gave up after {self.max_retries} attempts: {last_exc}")

    # ------------------------------------------------------------------ main
    def ask(self, state: Any, questions: dict[str, dict],
            meta: dict[str, Any] | None = None) -> JevResult:
        """Evaluate one state against a question map. Persists request and
        response verbatim to the experiment JSONL and the content-addressed
        cache. `meta` (site id, variant, repeat index ...) is logged only."""
        meta = dict(meta or {})
        state_hash = sha256_hex(canonical_json(state))
        key = self.cache_key(state, questions)
        body = {"state": state, "model": self.model, "questions": questions}

        cached = self._cache_get(key)
        if cached is not None and cached.get("ok"):
            res = JevResult(ok=True, cache_key=key, state_hash=state_hash, cached=True,
                            model_returned=cached["response"]["model"],
                            request_id=cached.get("request_id"),
                            answers=cached["response"]["answers"],
                            usage=cached["response"]["usage"],
                            cost_usd=cached["cost_usd"], latency_ms=cached.get("latency_ms"),
                            http_status=cached.get("http_status"), attempts=0)
            with self._lock:
                self._session_cached += 1
                self._append(self.log_path, {**meta, "ts": time.time(), "cached": True,
                                             "cache_key": key, "state_hash": state_hash,
                                             "model_requested": self.model,
                                             "model_returned": res.model_returned,
                                             "request_id": res.request_id,
                                             "usage": res.usage, "cost_usd": res.cost_usd,
                                             "latency_ms": res.latency_ms,
                                             "answers": res.answers})
            return res

        with self._lock:
            if self._ledger_total >= self.max_usd:
                raise SpendCapReached(f"cumulative spend {self._ledger_total:.4f} USD >= MAX_USD {self.max_usd}")

        status, js, headers, latency_ms, attempts = self._post(body)
        request_id = headers.get("x-typesafe-request-id")
        ts = time.time()

        if status == 401:
            raise AuthError(f"401 Unauthorized (request id {request_id}); stopping, no other credentials tried")

        if status != 200 or not js or "answers" not in js:
            err = json.dumps(js)[:500] if js is not None else "<non-JSON body>"
            rec = {**meta, "ts": ts, "cached": False, "ok": False, "cache_key": key,
                   "state_hash": state_hash, "model_requested": self.model,
                   "request_id": request_id, "http_status": status, "latency_ms": latency_ms,
                   "attempts": attempts, "error": err, "request": body}
            with self._lock:
                self._append(self.log_path, rec)
                self._session_calls += 1
                self._latencies.append(latency_ms)
            return JevResult(ok=False, cache_key=key, state_hash=state_hash, cached=False,
                             model_returned=None, request_id=request_id, answers={},
                             usage={}, cost_usd=0.0, latency_ms=latency_ms,
                             http_status=status, error=err, attempts=attempts)

        usage = js.get("usage", {})
        cost = self.cost_of(usage)
        model_returned = js.get("model")

        rec = {**meta, "ts": ts, "cached": False, "ok": True, "cache_key": key,
               "state_hash": state_hash, "model_requested": self.model,
               "model_returned": model_returned, "request_id": request_id,
               "http_status": status, "latency_ms": latency_ms, "attempts": attempts,
               "usage": usage, "cost_usd": cost, "request": body, "response": js}
        with self._lock:
            self._append(self.log_path, rec)
            self._append(self.ledger_path, {"ts": ts, "experiment": self.experiment,
                                            "request_id": request_id, "cache_key": key,
                                            "usage": usage, "cost_usd": cost})
            self._ledger_total += cost
            self._session_cost += cost
            self._session_calls += 1
            self._latencies.append(latency_ms)
            self._cache_put(key, {"ok": True, "cache_key": key, "state_hash": state_hash,
                                  "request_id": request_id, "http_status": status,
                                  "latency_ms": latency_ms, "cost_usd": cost,
                                  "request": body, "response": js})

        if model_returned != self.expected_response_model:
            raise ModelMismatch(f"response.model={model_returned!r} != pinned "
                                f"{self.expected_response_model!r} (request id {request_id})")

        return JevResult(ok=True, cache_key=key, state_hash=state_hash, cached=False,
                         model_returned=model_returned, request_id=request_id,
                         answers=js["answers"], usage=usage, cost_usd=cost,
                         latency_ms=latency_ms, http_status=status, attempts=attempts)

    def ask_many(self, items: Iterable[tuple[Any, dict[str, dict], dict[str, Any]]],
                 workers: int = 4,
                 on_result: Callable[[dict[str, Any], JevResult], None] | None = None
                 ) -> list[JevResult]:
        """Evaluate many (state, questions, meta) triples concurrently. Order of
        the returned list matches the input order."""
        items = list(items)
        results: list[JevResult | None] = [None] * len(items)

        def run(i: int) -> None:
            st, qs, meta = items[i]
            res = self.ask(st, qs, meta)
            results[i] = res
            if on_result:
                on_result(meta, res)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            list(ex.map(run, range(len(items))))
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------ stats
    def latency_stats(self) -> dict[str, float | int]:
        lat = sorted(self._latencies)
        if not lat:
            return {"n": 0}
        def q(p: float) -> float:
            k = max(0, min(len(lat) - 1, round(p * (len(lat) - 1))))
            return lat[k]
        return {"n": len(lat), "p50_ms": statistics.median(lat), "p95_ms": q(0.95),
                "mean_ms": statistics.fmean(lat), "max_ms": lat[-1]}

    def session_stats(self) -> dict[str, Any]:
        return {"api_calls": self._session_calls, "cache_hits": self._session_cached,
                "session_usd": round(self._session_cost, 6),
                "cumulative_usd": round(self._ledger_total, 6), "max_usd": self.max_usd,
                **self.latency_stats()}

    def close(self) -> None:
        if self._http is not None:
            self._http.close()


def client_from_config(cfg, experiment: str, use_cache: bool = True) -> JevClient:
    key = cfg.api_key
    if not key:
        raise AuthError("no TYPESAFE_API_KEY / JEV_API_KEY in .env")
    return JevClient(api_key=key, model=cfg["model"],
                     expected_response_model=cfg["expected_response_model"],
                     raw_root=cfg.raw_responses_root, api_base=cfg.get("api_base", "https://api.typesafe.ai"),
                     price_per_mtok_input=cfg["price_usd_per_mtok_input"],
                     price_per_mtok_output=cfg.get("price_usd_per_mtok_output", 0.0),
                     max_usd=float(cfg["MAX_USD"]), bank_version=cfg["bank_version"],
                     timeout_s=cfg.get("timeout_s", 60), max_retries=cfg.get("max_retries", 6),
                     use_cache=use_cache, experiment=experiment)
