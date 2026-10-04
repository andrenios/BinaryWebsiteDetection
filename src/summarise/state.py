"""State builders (WORKORDER.md Section 4).

The summariser `reduce_html` is the SVLM paper's code, copied unchanged
(`html_truncation.py`). This module adds the two derived deterministic fields
(`registrable_domain`, `favicon_host`), strips `_truncated`, and assembles the
state variants:

  S0    page_url + registrable_domain
  S1    full summary at 1,000 tokens
  S1-b  summary at a given budget (100, 250, 500, 1000, 2000)
  S2    S1 + screenshot_text (OCR)
  S3    page_url + raw rendered HTML truncated to 30k tokens with the JCP
        repo's tag-based truncation
  S5    (fresh crawl only) summary of the plain GET body; same code path as S1
        with a different html input.

Every builder returns (state, timing) where timing holds wall seconds per stage.
"""
from __future__ import annotations

import time
from typing import Any
from urllib.parse import urljoin, urlparse

import tldextract

from .html_truncation import reduce_html, make_token_counter
from .jcp_truncation import truncate_html_to_tokens_merged

# Offline suffix list only (bundled snapshot): no network calls during state
# building. PSL private suffixes are included so that free hosting such as
# <x>.blob.core.windows.net or <x>.github.io is reported as the registrable
# domain instead of windows.net / github.io (reports/decisions.md, D5).
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True,
                                 include_psl_private_domains=True)


def _registered(ext) -> str:
    # tldextract >= 5.3 renamed registered_domain; keep both spellings working
    if hasattr(ext, "top_domain_under_public_suffix"):
        return ext.top_domain_under_public_suffix
    return ext.registered_domain

BUDGETS = (100, 250, 500, 1000, 2000)


def registrable_domain(url: str | None) -> str | None:
    """Public-suffix aware registrable domain; the bare host for IPs/localhost."""
    if not url:
        return None
    host = urlparse(url).hostname
    if not host:
        return None
    ext = _EXTRACT(host)
    reg = _registered(ext)
    if reg:
        return reg.lower()
    return host.lower()          # IP address or suffix-less host


def favicon_host(favicon: str | None, page_url: str | None) -> str | None:
    if not favicon:
        return None
    if favicon.startswith("data:"):
        return None
    resolved = urljoin(page_url, favicon) if page_url else favicon
    return urlparse(resolved).hostname


def _finish(summary: dict[str, Any], page_url: str) -> dict[str, Any]:
    """Add derived fields, strip `_truncated`, keep the summariser's key order."""
    out: dict[str, Any] = {}
    for k, v in summary.items():
        if k == "_truncated":
            continue
        out[k] = v
        if k == "page_url":
            out["registrable_domain"] = registrable_domain(page_url)
    if "registrable_domain" not in out:
        out = {"page_url": page_url, "registrable_domain": registrable_domain(page_url), **out}
    fav = (summary.get("meta") or {}).get("favicon") if isinstance(summary.get("meta"), dict) else None
    out["favicon_host"] = favicon_host(fav, page_url)
    return out


def build_s0(page_url: str) -> tuple[dict[str, Any], dict[str, float]]:
    t0 = time.perf_counter()
    st = {"page_url": page_url, "registrable_domain": registrable_domain(page_url)}
    return st, {"summarise_s": time.perf_counter() - t0}


def build_s1(html: str, page_url: str, budget: int = 1000,
             token_model: str = "gpt-4") -> tuple[dict[str, Any], dict[str, float]]:
    t0 = time.perf_counter()
    summary = reduce_html(html, page_url, budget, token_model)
    truncated = bool(summary.get("_truncated"))
    st = _finish(summary, page_url)
    return st, {"summarise_s": time.perf_counter() - t0, "truncated": float(truncated)}


def build_s1b(html: str, page_url: str, budgets=BUDGETS,
              token_model: str = "gpt-4") -> dict[int, tuple[dict[str, Any], dict[str, float]]]:
    return {b: build_s1(html, page_url, b, token_model) for b in budgets}


def build_s2(html: str, page_url: str, screenshot_text: str | None,
             ocr_seconds: float | None = None, budget: int = 1000,
             token_model: str = "gpt-4") -> tuple[dict[str, Any], dict[str, float]]:
    st, timing = build_s1(html, page_url, budget, token_model)
    st["screenshot_text"] = screenshot_text or ""
    if ocr_seconds is not None:
        timing["ocr_s"] = ocr_seconds
    return st, timing


def build_s3(html: str, page_url: str, max_tokens: int = 30000,
             token_model: str = "gpt-4") -> tuple[dict[str, Any], dict[str, float]]:
    t0 = time.perf_counter()
    truncated_html = truncate_html_to_tokens_merged(html, max_tokens, token_model)
    st = {"page_url": page_url, "registrable_domain": registrable_domain(page_url),
          "html": truncated_html}
    return st, {"summarise_s": time.perf_counter() - t0}


def state_tokens(state: dict[str, Any], token_model: str = "gpt-4") -> int:
    """tiktoken size of the serialised state (client-side estimate; the API's
    own `usage.input_tokens` is what cost is computed from)."""
    import json
    return make_token_counter(token_model)(json.dumps(state, ensure_ascii=False))
