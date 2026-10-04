"""Open-weight twin (T09): open models answer the bank questions by constrained
decoding to a single token from {yes, no}; P(yes) is read from the logprobs of
the two candidate tokens and renormalised over them.

Backends
  vllm    OpenAI-compatible /v1/completions (vLLM): max_tokens=1, logprobs=K,
          extra_body guided_choice=["yes","no"] (falls back to plain top-K
          logprobs when guided decoding is unavailable).
  ollama  /api/generate with `logprobs`/`top_logprobs` (Ollama >= 0.12) and a
          one-token limit; if the server does not return logprobs the answer
          text is used with P(yes) in {0, 1} and the row is flagged.
  fake    deterministic stand-in for exercising the code path without a GPU;
          allowed on bootstrap/fixture datasets only, writes under
          results/dryrun/. Never a result.

Every call is persisted verbatim under data/raw_responses/twin/<model>/ with a
content-addressed cache keyed by (model, backend, prompt), so re-runs cost no
GPU time. Cost per site is GPU seconds x the paper-2 rate (config
gpu_usd_per_hour).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPT = ("You are a strict classifier. Read the JSON state of a web page and answer the question with a single word, "
          "yes or no.\n\nSTATE:\n{state}\n\nQUESTION: {question}\n\nANSWER (yes or no):")
YES_TOKENS = ("yes", " yes", "Yes", " Yes", "YES", " YES")
NO_TOKENS = ("no", " no", "No", " No", "NO", " NO")


def _canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def p_yes_from_logprobs(top: dict[str, float]) -> tuple[float | None, str]:
    """Renormalise P(yes) over the yes/no token masses found in a top-logprob map."""
    ly = [lp for t, lp in top.items() if t in YES_TOKENS]
    ln = [lp for t, lp in top.items() if t in NO_TOKENS]
    if not ly and not ln:
        return None, "neither token in top logprobs"
    py = sum(math.exp(x) for x in ly); pn = sum(math.exp(x) for x in ln)
    if py + pn == 0:
        return None, "zero mass"
    return py / (py + pn), ""


@dataclass
class TwinResponse:
    ok: bool
    p_yes: float | None
    text: str
    seconds: float
    cached: bool
    note: str = ""
    raw: dict | None = None


class TwinClient:
    def __init__(self, model: str, backend: str, raw_root: Path, base_url: str = "http://localhost:11434",
                 top_logprobs: int = 20, timeout_s: float = 120.0, use_cache: bool = True):
        self.model, self.backend, self.base_url = model, backend, base_url.rstrip("/")
        self.top_logprobs, self.timeout_s, self.use_cache = top_logprobs, timeout_s, use_cache
        self.root = Path(raw_root) / "twin" / model.replace(":", "_").replace("/", "_")
        (self.root / "cache").mkdir(parents=True, exist_ok=True)
        self.log = self.root / f"{backend}.jsonl"
        self.gpu_seconds = 0.0
        self.calls = 0
        self._http = None

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(_canon({"model": self.model, "backend": self.backend, "prompt": prompt}).encode()).hexdigest()

    def ask(self, state: dict, question: str) -> TwinResponse:
        prompt = PROMPT.format(state=_canon(state), question=question)
        key = self._key(prompt)
        cp = self.root / "cache" / key[:2] / f"{key}.json"
        if self.use_cache and cp.exists():
            rec = json.loads(cp.read_text(encoding="utf-8"))
            return TwinResponse(True, rec["p_yes"], rec.get("text", ""), rec["seconds"], True, rec.get("note", ""), rec.get("response"))
        t0 = time.perf_counter()
        if self.backend == "fake":
            resp, text, top = self._fake(prompt)
        elif self.backend == "vllm":
            resp, text, top = self._vllm(prompt)
        elif self.backend == "ollama":
            resp, text, top = self._ollama(prompt)
        else:
            raise ValueError(self.backend)
        secs = time.perf_counter() - t0
        p, note = p_yes_from_logprobs(top) if top else (None, "no logprobs returned")
        if p is None and text:
            p = 1.0 if text.strip().lower().startswith("y") else 0.0
            note = (note + "; P(yes) from the answer text") if note else "P(yes) from the answer text"
        rec = {"ts": time.time(), "model": self.model, "backend": self.backend, "prompt": prompt, "response": resp,
               "text": text, "top_logprobs": top, "p_yes": p, "seconds": secs, "note": note, "cache_key": key}
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if p is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        self.gpu_seconds += secs; self.calls += 1
        return TwinResponse(p is not None, p, text, secs, False, note, resp)

    # ------------------------------------------------------------------ backends
    def _client(self):
        import httpx
        if self._http is None:
            self._http = httpx.Client(timeout=self.timeout_s)
        return self._http

    def _vllm(self, prompt: str):
        body = {"model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0, "logprobs": self.top_logprobs,
                "guided_choice": ["yes", "no"]}
        r = self._client().post(f"{self.base_url}/v1/completions", json=body)
        if r.status_code == 400 and "guided" in r.text.lower():
            body.pop("guided_choice"); r = self._client().post(f"{self.base_url}/v1/completions", json=body)
        r.raise_for_status()
        js = r.json()
        ch = js["choices"][0]
        lp = (ch.get("logprobs") or {}).get("top_logprobs") or [{}]
        return js, ch.get("text", ""), dict(lp[0]) if lp else {}

    def _ollama(self, prompt: str):
        body = {"model": self.model, "prompt": prompt, "stream": False, "logprobs": True, "top_logprobs": self.top_logprobs,
                "options": {"temperature": 0.0, "num_predict": 1}}
        r = self._client().post(f"{self.base_url}/api/generate", json=body)
        r.raise_for_status()
        js = r.json()
        top: dict[str, float] = {}
        lps = js.get("logprobs") or []
        if lps:
            first = lps[0]
            for alt in first.get("top_logprobs", []) or []:
                top[alt.get("token", "")] = float(alt.get("logprob", -1e9))
            if first.get("token") and first.get("token") not in top:
                top[first["token"]] = float(first.get("logprob", 0.0))
        return js, js.get("response", ""), top

    def _fake(self, prompt: str):
        """Deterministic pseudo-answer from the prompt hash (code-path checks only)."""
        h = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        lp_yes = math.log(max(1e-6, h)); lp_no = math.log(max(1e-6, 1 - h))
        return {"fake": True}, "yes" if h >= 0.5 else "no", {"yes": lp_yes, "no": lp_no}

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
