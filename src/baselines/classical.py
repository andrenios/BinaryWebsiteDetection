"""Classical baselines (T08 a–c): URL-feature gradient boosting, TF-IDF + LR
over the serialised S1 state, and a hand-written rule set mirroring the bank."""
from __future__ import annotations

import json
import math
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

BRANDS = ("paypal", "apple", "microsoft", "office", "outlook", "google", "facebook", "instagram", "amazon",
          "netflix", "dhl", "usps", "fedex", "chase", "wellsfargo", "bankofamerica", "hsbc", "barclays",
          "santander", "sparkasse", "postbank", "ing", "bnp", "societegenerale", "creditagricole", "laposte",
          "orange", "free", "sfr", "docusign", "dropbox", "adobe", "linkedin", "whatsapp", "telegram",
          "binance", "coinbase", "metamask", "steam", "ebay", "att", "verizon", "tmobile", "vodafone", "ionos")
FREE_HOSTS = ("blogspot.", "weebly.", "wixsite.", "github.io", "netlify.app", "vercel.app", "web.app",
              "firebaseapp.com", "pages.dev", "glitch.me", "repl.co", "000webhostapp.com", "herokuapp.com",
              "azurewebsites.net", "blob.core.windows.net", "ipfs.", "dweb.link", "duckdns.org", "no-ip.",
              "webcindario", "godaddysites.com", "square.site", "surge.sh", "r2.dev", "workers.dev")
SHORTENERS = ("bit.ly", "t.co", "tinyurl", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "shorturl")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def url_features(url: str) -> dict:
    u = urlparse(url if "://" in url else "http://" + url)
    host = (u.hostname or "").lower()
    path = u.path or ""
    low = url.lower()
    labels = host.split(".") if host else []
    return {"url_len": len(url), "host_len": len(host), "path_len": len(path), "query_len": len(u.query or ""),
            "n_dots_host": host.count("."), "n_subdomains": max(0, len(labels) - 2), "n_hyphens_host": host.count("-"),
            "n_digits_host": sum(c.isdigit() for c in host), "n_digits_url": sum(c.isdigit() for c in url),
            "is_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))), "has_at": int("@" in url),
            "n_encoded": url.count("%"), "n_slashes": path.count("/"), "https": int(u.scheme == "https"),
            "port": int(u.port is not None and u.port not in (80, 443)),
            "brand_in_host": int(any(b in host for b in BRANDS)), "brand_in_path": int(any(b in path.lower() for b in BRANDS)),
            "free_host": int(any(f in host for f in FREE_HOSTS)), "shortener": int(any(s in host for s in SHORTENERS)),
            "phishy_words": sum(w in low for w in ("login", "secure", "account", "update", "verify", "signin", "confirm",
                                                    "password", "bank", "wallet", "invoice", "billing", "support")),
            "host_entropy": _entropy(host), "path_entropy": _entropy(path), "tld_len": len(labels[-1]) if labels else 0,
            "html_ext": int(low.endswith((".html", ".htm", ".php"))), "long_token": int(any(len(t) > 25 for t in re.split(r"[./\-_?=&]", low)))}


def make_url_gbm(seed: int = 2107):
    return GradientBoostingClassifier(random_state=seed)


def serialise_state(state: dict) -> str:
    return json.dumps(state, ensure_ascii=False)


def make_tfidf_lr(seed: int = 2107):
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True),
                         LogisticRegression(C=4.0, max_iter=2000))


# ----------------------------------------------------------------------------- rule set
def rules(state: dict) -> dict:
    sig = state.get("signals", {}) or {}
    forms = state.get("forms", []) or []
    hosts = state.get("hosts", {}) or {}
    meta = state.get("meta", {}) or {}
    text = (state.get("visible_text") or "").lower()
    reg = (state.get("registrable_domain") or "").lower()
    url = (state.get("page_url") or "").lower()
    title = ((meta.get("title") or "") + " " + " ".join(str(v) for v in (meta.get("meta") or {}).values())).lower()
    fav = (state.get("favicon_host") or "").lower()
    brand_text = any(b in title or b in text[:3000] for b in BRANDS)
    brand_url = any(b in url for b in BRANDS)
    r = {"r_collects_credentials": int(sig.get("num_password_fields", 0) > 0 or sig.get("num_sensitive_fields", 0) > 0),
         "r_form_offsite": int(bool(sig.get("form_posts_offsite"))),
         "r_brand_domain_mismatch": int(brand_text and not any(b in reg for b in BRANDS)),
         "r_brand_in_url_not_domain": int(brand_url and not any(b in reg for b in BRANDS)),
         "r_ip_or_freehost": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", reg)) or any(f in reg for f in FREE_HOSTS) or any(s in reg for s in SHORTENERS)),
         "r_urgency": int(any(w in text for w in ("urgent", "immediately", "suspend", "within 24", "expire", "limited", "verify now", "unusual activity"))),
         "r_reward_bait": int(any(w in text for w in ("congratulations", "winner", "prize", "gift card", "refund", "reward", "claim your"))),
         "r_obfuscated_scripts": int(any((not s.get("external")) and (s.get("entropy", 0) > 5 or any(t in (s.get("tokens") or []) for t in ("eval", "atob", "unescape", "fromcharcode"))) for s in (state.get("scripts") or []))),
         "r_external_favicon": int(bool(fav) and not fav.endswith(reg) if reg else 0),
         "r_shell_page": int(len(text) < 400 and (sig.get("num_password_fields", 0) > 0 or sig.get("num_sensitive_fields", 0) > 0)),
         "r_missing_legitimacy": int(not any(w in text for w in ("privacy", "terms", "contact", "©", "copyright", "impressum", "datenschutz"))),
         "r_suspicious_url_shape": int(url.count("-") > 3 or url.count("%") > 2 or len(url) > 120 or (reg and reg.count(".") + url.split("//")[-1].split("/")[0].count(".") > 4)),
         "r_cloned_assets": int(hosts.get("num_external_hosts", 0) >= 3 and sum(1 for h in (hosts.get("external_hosts") or {}) if reg and not h.endswith(reg)) >= 3)}
    r["rule_score"] = sum(v for k, v in r.items() if k.startswith("r_")) / 13
    return r
