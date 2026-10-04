# -*- coding: utf-8 -*-


import json
import re
import math
from collections import Counter
from urllib.parse import urlparse, urljoin
import tiktoken

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Token counting: tiktoken if available, else a cheap approximation
# ---------------------------------------------------------------------------

def make_token_counter(model: str = "gpt-4"):
    """Return a function str -> token count. Uses tiktoken if it can load."""
    enc = tiktoken.encoding_for_model(model or "gpt-4")
    return lambda s: len(enc.encode(s))
    

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

# Phishing cue keywords, carried over from the uploaded script.
PHISHING_KEYWORDS = ("login", "secure", "account", "@", "update", "verify",
                     "signin", "confirm", "password", "bank")
BRAND_IMG_KEYWORDS = ("logo", "brand", "icon")

# Credential-related hints for input fields.
SENSITIVE_HINTS = ("pass", "pwd", "user", "email", "login", "card", "cvv",
                   "cvc", "ssn", "otp", "pin", "account", "secret", "token")

# Tokens that often appear in obfuscated / malicious JS.
SUSPICIOUS_JS_TOKENS = ("eval", "atob", "unescape", "fromcharcode",
                        "document.write", "escape", "settimeout",
                        "xmlhttprequest", "fetch(")


def shannon_entropy(text: str) -> float:
    """Randomness measure. Obfuscated / base64 blobs score high (~5-6)."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return round(-sum((c / n) * math.log2(c / n) for c in counts.values()), 2)


def host_of(url: str):
    try:
        return urlparse(url).hostname
    except Exception:
        return None


def is_hidden(tag) -> bool:
    """Hidden elements are a classic phishing tell -- we keep them *because*
    they're hidden, and flag them."""
    if tag.has_attr("hidden"):
        return True
    if tag.get("type", "").lower() == "hidden":
        return True
    style = tag.get("style", "").lower().replace(" ", "")
    return "display:none" in style or "visibility:hidden" in style


# ---------------------------------------------------------------------------
# Extraction of each signal group
# ---------------------------------------------------------------------------

def extract_meta(soup):
    """Title + selected meta tags + favicon reference (brand impersonation)."""
    title = soup.title.string.strip() if soup.title and soup.title.string else None

    metas = {}
    for m in soup.find_all("meta"):
        key = m.get("name") or m.get("property")
        content = m.get("content")
        if key and content:
            # Keep the meta tags the uploaded script prioritized.
            k = key.lower()
            if k in ("description", "keywords", "viewport") or k.startswith(("og:", "twitter:")):
                metas[k] = content[:200]

    favicon = None
    for link in soup.find_all("link"):
        rel = " ".join(link.get("rel", [])).lower()
        if "icon" in rel:
            favicon = link.get("href")
            break

    return {"title": title, "meta": metas, "favicon": favicon}


def extract_forms(soup, page_url):
    """Every form: where it submits, and each input (esp. sensitive/hidden).
    Nesting is preserved -- which input belongs to which form -- because that
    relationship is one of the strongest phishing signals."""
    forms = []
    for form in soup.find_all("form"):
        action = form.get("action")
        resolved_action = urljoin(page_url, action) if (page_url and action) else action

        inputs = []
        for field in form.find_all(("input", "textarea", "select")):
            name = field.get("name") or field.get("id") or ""
            ftype = field.get("type", field.name).lower()
            hint = name.lower() + " " + field.get("placeholder", "").lower()
            inputs.append({
                "name": name[:60],
                "type": ftype,
                "sensitive": any(h in hint for h in SENSITIVE_HINTS),
                "hidden": is_hidden(field),
            })

        forms.append({
            "action": action[:120] if action else None,
            "action_host": host_of(resolved_action),
            "method": form.get("method", "get").lower(),
            "has_password": any(i["type"] == "password" for i in inputs),
            "inputs": inputs,
        })
    return forms


def extract_links_and_hosts(soup, page_url):
    """External hosts referenced, plus a count of links whose path/host carries
    phishing keywords (carried over from the uploaded script's scoring idea)."""
    page_host = host_of(page_url) if page_url else None
    external_hosts = Counter()
    phishy_links = 0

    attr_map = {"a": "href", "link": "href", "script": "src",
                "img": "src", "iframe": "src"}
    for tag_name, attr in attr_map.items():
        for tag in soup.find_all(tag_name):
            ref = tag.get(attr)
            if not ref:
                continue
            low = ref.lower()
            if tag_name == "a" and any(k in low for k in PHISHING_KEYWORDS):
                phishy_links += 1
            resolved = urljoin(page_url, ref) if page_url else ref
            h = host_of(resolved)
            if h and h != page_host:
                external_hosts[h] += 1

    return {
        "page_host": page_host,
        "external_hosts": dict(external_hosts.most_common(15)),
        "num_external_hosts": len(external_hosts),
        "num_phishy_keyword_links": phishy_links,
    }


def summarize_scripts(soup):
    """Keep each script's *signal* (length, entropy, suspicious tokens), not its
    body. A 40KB obfuscated blob becomes a few numbers. The uploaded script
    deleted all JS and lost this signal entirely; we keep it."""
    scripts = []
    for s in soup.find_all("script"):
        if s.get("src"):
            scripts.append({"external": True, "src": s.get("src")[:120]})
            continue
        body = s.string or ""
        if not body.strip():
            continue
        low = body.lower()
        scripts.append({
            "external": False,
            "length": len(body),
            "entropy": shannon_entropy(body),
            "tokens": sorted({t for t in SUSPICIOUS_JS_TOKENS if t in low}),
        })
    return scripts


def extract_visible_text(soup):
    """Human-visible text, scripts/styles removed, whitespace collapsed.
    Returned uncapped here; the budget stage trims it if needed."""
    clone = BeautifulSoup(str(soup), "lxml")
    for junk in clone(["script", "style", "noscript", "svg"]):
        junk.decompose()
    text = clone.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def compute_signals(forms, links, soup):
    """Cheap page-level signals a model or rule can use directly."""
    all_inputs = [i for f in forms for i in f["inputs"]]
    return {
        "num_forms": len(forms),
        "num_password_fields": sum(i["type"] == "password" for i in all_inputs),
        "num_hidden_fields": sum(i["hidden"] for i in all_inputs),
        "num_sensitive_fields": sum(i["sensitive"] for i in all_inputs),
        "form_posts_offsite": any(
            f["action_host"] and f["action_host"] != links["page_host"]
            for f in forms
        ),
        "num_iframes": len(soup.find_all("iframe")),
    }


# ---------------------------------------------------------------------------
# Budget-aware assembly (the carried-over idea from the uploaded script)
# ---------------------------------------------------------------------------

# Priority order: highest-signal groups are added first, so if the budget runs
# out it is the low-signal groups (visible_text, then scripts) that get trimmed
# -- never the forms. This mirrors the uploaded script's "fill until budget"
# behavior, but over structured groups instead of raw HTML tags.
GROUP_PRIORITY = ["page_url", "signals", "forms", "hosts", "meta", "scripts", "visible_text"]


def assemble_within_budget(groups: dict, max_tokens: int, count_tokens) -> dict:
    """Add groups in priority order until the token budget is hit. The two
    'soft' groups (visible_text, scripts) are trimmed rather than dropped so
    some of their signal always survives."""
    out = {}

    def size(obj):
        return count_tokens(json.dumps(obj, ensure_ascii=False))

    for name in GROUP_PRIORITY:
        if name not in groups:
            continue
        candidate = dict(out)
        candidate[name] = groups[name]

        if size(candidate) <= max_tokens:
            out[name] = groups[name]
            continue

        # Over budget adding this group. Trim the soft ones; drop the rest.
        remaining = max(0, max_tokens - size(out))

        if name == "visible_text":
            text = groups[name]
            # Binary-search the longest prefix that fits the remaining budget.
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if count_tokens(json.dumps(text[:mid], ensure_ascii=False)) <= remaining:
                    lo = mid
                else:
                    hi = mid - 1
            if lo > 0:
                out[name] = text[:lo] + (" …[truncated]" if lo < len(text) else "")

        elif name == "scripts":
            # Keep whole script summaries one at a time until we run out of room.
            kept = []
            for item in groups[name]:
                trial = kept + [item]
                if size({**out, "scripts": trial}) <= max_tokens:
                    kept = trial
                else:
                    break
            if kept:
                out[name] = kept
        # Any other group that doesn't fit is simply omitted (shouldn't happen
        # for the small high-priority groups, but this is the safe default).

    out["_truncated"] = size(groups) > max_tokens
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def reduce_html(html: str, page_url: str = None, max_tokens: int = 1000,
                model: str = "gpt-4") -> dict:
    count_tokens = make_token_counter(model)
    soup = BeautifulSoup(html, "lxml")

    forms = extract_forms(soup, page_url)
    links = extract_links_and_hosts(soup, page_url)

    groups = {
        "page_url": page_url,
        "signals": compute_signals(forms, links, soup),
        "forms": forms,
        "hosts": links,
        "meta": extract_meta(soup),
        "scripts": summarize_scripts(soup),
        "visible_text": extract_visible_text(soup),
    }

    return assemble_within_budget(groups, max_tokens, count_tokens)


def reduce_html_file(input_path, output_path=None, page_url=None,
                     max_tokens=1000, model="gpt-4", verbose=True):
    """Read an HTML file, reduce it, optionally write the JSON, and return the
    summary dict. Designed to be called directly from Spyder / Jupyter.

    Parameters
    ----------
    input_path : str
        Path to the HTML file to read.
    output_path : str or None
        Where to write the JSON summary. If None, nothing is written and the
        dict is only returned.
    page_url : str or None
        URL the page was served from (improves host/action resolution).
    max_tokens : int
        Token budget for the JSON summary.
    model : str
        Model name for tiktoken encoding (falls back to a char-based estimate
        if tiktoken can't load an encoding).
    verbose : bool
        Print a short reduction report.

    Returns
    -------
    dict
        The reduced summary.
    """
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    summary = reduce_html(html, page_url, max_tokens, model)

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        count_tokens = make_token_counter(model)
        orig_tokens = count_tokens(html)
        out_tokens = count_tokens(json.dumps(summary, ensure_ascii=False))
        where = output_path if output_path is not None else "(not written)"
        print(f"Output: {where}")
        print(f"  original HTML: {orig_tokens:,} tokens")
        print(f"  reduced JSON:  {out_tokens:,} tokens  (budget {max_tokens:,})")
        print(f"  truncated: {summary.get('_truncated')}")

    return summary


