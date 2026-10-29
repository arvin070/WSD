"""
Offline link analysis.

The false-positive side matters as much as the detections: a tool that flags
amazon.in teaches people to ignore it.
"""

import pytest

from scamshield.core.links import (
    analyse_link,
    analyse_links_in,
    decode_punycode,
    extract_links,
    host_as_ip,
    registrable_domain,
)
from scamshield.resources import load_lexicons

MULTI = set(load_lexicons().links["multi_label_suffixes"])


# ---- structural parsing ---------------------------------------------------

@pytest.mark.parametrize("host,expected", [
    ("paypal.com.secure-login.xyz", "secure-login.xyz"),
    ("www.amazon.in", "amazon.in"),
    ("a.b.c.example.co.uk", "example.co.uk"),
    ("hdfcbank.com", "hdfcbank.com"),
    ("login.icicibank.com", "icicibank.com"),
])
def test_registrable_domain(host, expected):
    assert registrable_domain(host, MULTI) == expected


@pytest.mark.parametrize("host,ip", [
    ("127.0.0.1", "127.0.0.1"),
    ("2130706433", "127.0.0.1"),      # decimal
    ("0x7f000001", "127.0.0.1"),      # hex
    ("example.com", None),
])
def test_numeric_host_forms(host, ip):
    assert host_as_ip(host) == ip


def test_punycode_decoding():
    assert decode_punycode("xn--80ak6aa92e.com") == "аррӏе.com"
    assert decode_punycode("example.com") is None


# ---- deceptions that must be caught ---------------------------------------

@pytest.mark.parametrize("url,code", [
    ("https://paypal.com@evil-login.xyz/verify", "userinfo_in_url"),
    ("http://paypal.com.secure-login.xyz/signin", "brand_in_subdomain"),
    ("http://sbi-kyc-verify.xyz/l", "brand_in_domain"),
    ("http://arnazon.in/claim", "typosquat"),
    ("http://2130706433/admin", "ip_host"),
    ("https://xn--80ak6aa92e.com/login", "punycode_host"),
    ("https://cdn.example.com/update.apk", "dangerous_download"),
    ("javascript:alert(document.cookie)", "hostile_scheme"),
    ("https://x.example.com/go?url=http%3A%2F%2Fevil.xyz", "embedded_redirect"),
    ("http://bit.ly/3xK9zQa", "shortener"),
])
def test_deception_detected(url, code):
    report = analyse_link(url)
    assert code in {f.code for f in report.findings}, f"{code} not found in {url}"


@pytest.mark.parametrize("url", [
    "https://paypal.com@evil-login.xyz/verify",
    "http://paypal.com.secure-login.xyz/signin",
    "http://sbi-kyc-verify.xyz/l",
    "https://cdn.example.com/update.apk",
    "javascript:alert(1)",
    "https://xn--80ak6aa92e.com/login",
])
def test_deceptive_links_score_high(url):
    assert analyse_link(url).risk_score >= 0.75


# ---- legitimate links must stay quiet -------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.amazon.in/orders",
    "https://hdfcbank.com/netbanking",
    "https://onlinesbi.sbi/login",
    "https://pay.google.com/",
    "https://www.irctc.co.in/nget/train-search",
    "https://github.com/some/repo",
])
def test_legitimate_links_are_clean(url):
    report = analyse_link(url)
    assert report.risk_score < 0.20, f"false positive on {url}: {report.findings}"
    assert report.verdict == "clean"


def test_hostile_scheme_short_circuits():
    """A javascript: URL must never be parsed as a host with a port."""
    for url in ["javascript:alert(document.cookie)", "data:text/html;base64,PHN2Zz4=",
                "vbscript:msgbox(1)"]:
        assert analyse_link(url).risk_score == 1.0


def test_malformed_url_does_not_raise():
    for url in ["http://", "https://[::1", "http://a:b:c/", "", "???"]:
        analyse_link(url)  # must not raise


# ---- message-level -------------------------------------------------------

def test_extract_links_from_message():
    found = extract_links("Update KYC http://sbi-kyc.xyz/l now, or visit www.example.com ok?")
    assert any("sbi-kyc.xyz" in u for u in found)
    assert any("example.com" in u for u in found)


def test_embedded_redirect_is_followed_one_level():
    reports = analyse_links_in(
        "click https://go.example.com/r?redirect=http%3A%2F%2Fsbi-kyc.xyz%2Flogin")
    assert any("sbi-kyc.xyz" in r.url for r in reports), "nested destination not analysed"


def test_trailing_punctuation_stripped():
    assert extract_links("go to https://example.com/page.")[0] == "https://example.com/page"


def test_detector_reports_links():
    from scamshield.ml.detector import ModelNotTrained, get_detector
    try:
        d = get_detector()
    except ModelNotTrained:
        pytest.skip("run scamshield-train first")
    result = d.score("Your KYC expired, update at http://sbi-kyc-verify.xyz/login")
    assert result["links"] and result["links"][0]["verdict"] == "dangerous"
