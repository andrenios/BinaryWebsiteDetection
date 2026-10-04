# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup, Tag, NavigableString, Comment
from urllib.parse import urlparse
from copy import copy
import uuid
import tiktoken

# -------------------------
# Utility functions
# -------------------------
def extract_domain(url: str) -> str:
    if not url:
        return "None"
    parsed = urlparse(url)
    return parsed.netloc.lower() or "None"

def extract_url_from_tag(tag: Tag) -> str:
    return tag.get("href") or tag.get("src") or str(uuid.uuid1())

# -------------------------
# Importance scoring
# -------------------------
PHISHING_KEYWORDS = ["login", "secure", "account", "@", "update", "verify"]
BRAND_IMG_KEYWORDS = ["logo", "brand", "icon"]

CRITICAL_TAGS = {"title", "h1", "h2", "h3", "h4", "h5", "h6",
                 "p", "strong", "a", "img", "hr", "table", "tbody", "tr",
                 "th", "td", "ol", "ul", "li", "ruby", "label", "head", "body", "meta", "link"}

def get_importance_score(tag: Tag) -> int:
    name = tag.name or ""
    name_lower = name.lower()

    # Score 3: Paper's critical tags + phishing cues
    if name_lower in CRITICAL_TAGS:
        if name_lower == "a" and any(k in tag.get("href", "").lower() for k in PHISHING_KEYWORDS):
            return 3
        if name_lower == "img" and any(k in (tag.get("alt", "") + tag.get("src", "")).lower() for k in BRAND_IMG_KEYWORDS):
            return 3
        if name_lower == "meta" and (tag.get("name") or "").lower() in ["description", "keywords", "viewport"]:
            return 3
        if name_lower == "meta" and (tag.get("property") or "").lower().startswith(("og:", "twitter:")):
            return 3
        if name_lower == "link" and any(x in [r.lower() for r in (tag.get("rel") or [])] for x in ["icon", "stylesheet", "canonical"]):
            return 3
        return 3  # Paper's critical list default

    # Score 2: Layout/structural tags with potential context
    if name_lower in {"html", "head"}:
        return 2

    # Score 1: Keep content but unwrap to save space
    if name_lower in {"div", "span", "section", "article", "aside", "footer", "header"}:
        return 1

    # Score 0: Junk
    if not tag.get_text(strip=True) and not tag.attrs:
        return 0
    return 1

# -------------------------
# Pre-cleaning step
# -------------------------
def pre_clean_html(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")

    # Remove <style>, <script>, and comments
    for el in soup(["style", "script"]):
        el.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Unwrap/remove based on importance
    for tag in soup.find_all():
        score = get_importance_score(tag)
        if score == 0:
            tag.decompose()
        elif score == 1:
            tag.unwrap()

    # Remove empty elements
    for tag in soup.find_all():
        if not tag.get_text(strip=True) and not tag.attrs:
            tag.decompose()

    # Shorten href/src
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http"):
            domain = extract_domain(href)
            a["href"] = domain if len(href) > 50 else href
        elif len(href) > 50:
            a["href"] = href[:47] + "..."

    for img in soup.find_all("img", src=True):
        src = img["src"]
        if src.startswith("data:image") and len(src) > 100:
            img["src"] = src[:97] + "..."
        elif len(src) > 100:
            img["src"] = src[:97] + "..."

    return soup

# -------------------------
# Scored element extraction
# -------------------------
def extract_scored_elements(soup: BeautifulSoup, max_tokens: int, model: str = "") -> BeautifulSoup:
    reduced = BeautifulSoup("<html><head></head><body></body></html>", "html.parser")
    all_tags = list(soup.find_all())
    scored_tags = [(get_importance_score(tag), tag) for tag in all_tags]
    scored_tags.sort(key=lambda x: (-x[0], all_tags.index(x[1])))

    encoding = tiktoken.encoding_for_model(model or "gpt-4")

    for score, tag in scored_tags:
        cloned = copy(tag)
        cloned.clear()
        if tag.string:
            cloned.append(NavigableString(tag.string))
        target_parent = reduced.head if tag.name in ["title", "meta", "link"] else reduced.body
        target_parent.append(cloned)
        if len(encoding.encode(str(reduced))) >= max_tokens:
            break

    return reduced

# -------------------------
# Hybrid trimming
# -------------------------
def hybrid_trim(soup: BeautifulSoup, max_tokens: int, model: str = "") -> BeautifulSoup:
    encoding = tiktoken.encoding_for_model(model or "gpt-4")
    def token_len(): return len(encoding.encode(str(soup)))

    while token_len() > max_tokens:
        tags = [t for t in soup.find_all() if get_importance_score(t) == 1]
        if tags:
            name_counts = {}
            for t in tags:
                name_counts[t.name] = name_counts.get(t.name, 0) + 1
            most_freq_tag = max(name_counts, key=name_counts.get)
            tag_to_remove = soup.find(most_freq_tag)
            if tag_to_remove:
                tag_to_remove.decompose()
                continue
        all_tags = soup.find_all()
        if not all_tags:
            break
        all_tags[len(all_tags) // 2].decompose()
    return soup

# -------------------------
# Full truncation pipeline
# -------------------------
def truncate_html_to_tokens_merged(html: str, max_tokens: int = 1000, model: str = "") -> str:
    cleaned_soup = pre_clean_html(html)
    encoding = tiktoken.encoding_for_model(model or "gpt-4")

    if len(encoding.encode(str(cleaned_soup))) <= max_tokens:
        return str(cleaned_soup)

    reduced_soup = extract_scored_elements(cleaned_soup, max_tokens, model)

    if len(encoding.encode(str(reduced_soup))) > max_tokens:
        reduced_soup = hybrid_trim(reduced_soup, max_tokens, model)


    return str(reduced_soup)
