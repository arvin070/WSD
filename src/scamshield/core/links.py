"""
Offline URL risk analysis — what can be known before anyone clicks.

Runs entirely on the text of the link. No DNS, no HTTP, no network of any
kind, so it is safe to run on every message automatically and cannot tell an
attacker that their target is live.

The deceptions this catches are the ones that work on people rather than on
software: a host that reads as a familiar brand until you look at which part
is the actual domain, a character that is not the letter it resembles, a
second URL smuggled through a query parameter, an address that is really a
number, or a link whose destination is an installable file.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlsplit

from ..resources import Lexicons, load_lexicons
from .normalizer import CYRILLIC_GREEK_RE, LATIN_RE, normalise

#: Schemes that should never appear in a chat link.
HOSTILE_SCHEMES = {"javascript", "data", "vbscript", "file", "blob"}
WEB_SCHEMES = {"http", "https"}

_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_DECIMAL_HOST_RE = re.compile(r"^\d{6,12}$")
_HEX_HOST_RE = re.compile(r"^0x[0-9a-f]+$", re.I)
_OCTAL_HOST_RE = re.compile(r"^0\d+(?:\.0\d+){3}$")
_URL_IN_TEXT = re.compile(r"(?i)\b(?:https?|ftp)://\S+"
                          r"|\b(?:javascript|data|vbscript|file):[^\s<>\"']+"
                          r"|\bwww\.[^\s<>\"']+")
#: A scheme is `name:`, with or without the `//`. `javascript:alert(1)` has no
#: `//`, so testing for `://` alone would treat it as a bare hostname.
_HAS_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*:", re.I)


@dataclass
class LinkFinding:
    """One reason a link looks wrong, in words a person can act on."""

    code: str
    label: str
    weight: float
    detail: str = ""

    def as_dict(self) -> dict:
        return {"code": self.code, "label": self.label,
                "weight": round(self.weight, 3), "detail": self.detail}


@dataclass
class LinkReport:
    url: str
    normalised_url: str = ""
    scheme: str = ""
    host: str = ""
    registrable_domain: str = ""
    risk_score: float = 0.0
    findings: list[LinkFinding] = field(default_factory=list)
    embedded_urls: list[str] = field(default_factory=list)
    is_shortener: bool = False

    @property
    def verdict(self) -> str:
        if self.risk_score >= 0.75:
            return "dangerous"
        if self.risk_score >= 0.45:
            return "suspicious"
        if self.risk_score >= 0.20:
            return "caution"
        return "clean"

    def as_dict(self) -> dict:
        return {
            "url": self.url, "scheme": self.scheme, "host": self.host,
            "registrable_domain": self.registrable_domain,
            "risk_score": round(self.risk_score, 4), "verdict": self.verdict,
            "is_shortener": self.is_shortener,
            "embedded_urls": self.embedded_urls,
            "findings": [f.as_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Host parsing
# ---------------------------------------------------------------------------

def registrable_domain(host: str, multi_label: set[str]) -> str:
    """
    The part a registrar actually sold, e.g. `evil.xyz` in
    `paypal.com.login.evil.xyz`. This is the only part that identifies who
    controls the site; everything to its left is chosen freely by them.
    """
    labels = host.strip(".").lower().split(".")
    if len(labels) <= 2:
        return ".".join(labels)
    if ".".join(labels[-2:]) in multi_label and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def host_as_ip(host: str) -> str | None:
    """
    Resolve the numeric forms of an IP literal.

    `http://2130706433/` and `http://0x7f000001/` are both 127.0.0.1 written to
    look like anything but an address.
    """
    h = host.strip("[]")
    try:
        return str(ipaddress.ip_address(h))
    except ValueError:
        pass
    if _DECIMAL_HOST_RE.match(h):
        try:
            return str(ipaddress.ip_address(int(h)))
        except (ValueError, ipaddress.AddressValueError):
            return None
    if _HEX_HOST_RE.match(h):
        try:
            return str(ipaddress.ip_address(int(h, 16)))
        except (ValueError, ipaddress.AddressValueError):
            return None
    if _OCTAL_HOST_RE.match(h):
        try:
            return str(ipaddress.ip_address(".".join(str(int(p, 8)) for p in h.split("."))))
        except (ValueError, ipaddress.AddressValueError):
            return None
    return None


def decode_punycode(host: str) -> str | None:
    """Return the Unicode form of an `xn--` host, if it is one."""
    if "xn--" not in host.lower():
        return None
    try:
        return host.encode("ascii").decode("idna")
    except (UnicodeError, UnicodeDecodeError):
        return None


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Levenshtein distance, short-circuited once it exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _maybe_base64_url(value: str) -> str | None:
    if len(value) < 16 or not re.fullmatch(r"[A-Za-z0-9+/=_-]+", value):
        return None
    try:
        pad = value.replace("-", "+").replace("_", "/")
        decoded = base64.b64decode(pad + "=" * (-len(pad) % 4), validate=False).decode(
            "utf-8", errors="strict")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    return decoded if decoded.lower().startswith(("http://", "https://")) else None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def extract_links(text: str) -> list[str]:
    """Pull candidate URLs out of free text, including bare `www.` hosts."""
    found = []
    for match in _URL_IN_TEXT.finditer(normalise(text or "").text):
        found.append(match.group(0).rstrip(".,;:!?)]}>\"'"))
    return found


def analyse_link(url: str, lex: Lexicons | None = None) -> LinkReport:  # noqa: C901
    """Score a single URL offline. Higher is worse."""
    lex = lex or load_lexicons()
    cfg = lex.links
    scoring = cfg.get("scoring", {})
    safe_domains = set(cfg.get("safe_domains", []))
    multi_label = set(cfg.get("multi_label_suffixes", []))

    raw = (url or "").strip()
    report = LinkReport(url=raw)
    if not raw:
        return report

    candidate = raw if _HAS_SCHEME_RE.match(raw) else f"http://{raw}"
    try:
        parts = urlsplit(candidate)
    except ValueError:
        report.findings.append(LinkFinding("malformed_url", "Link cannot be parsed", 0.60))
        report.risk_score = 0.60
        return report

    scheme = (parts.scheme or "").lower()
    report.scheme = scheme
    host = (parts.hostname or "").lower()
    report.host = host
    report.normalised_url = candidate

    def add(code: str, label: str, weight: float, detail: str = "") -> None:
        report.findings.append(LinkFinding(code, label, weight, detail))

    # -- scheme ------------------------------------------------------------
    if scheme in HOSTILE_SCHEMES:
        add("hostile_scheme", "Link runs code or embeds content instead of opening a page",
            1.00, f"{scheme}: scheme")
        report.risk_score = 1.0
        return report
    if scheme not in WEB_SCHEMES:
        add("unusual_scheme", "Unusual link type", 0.40, scheme)
    elif scheme == "http":
        add("no_tls", "Not encrypted (http, not https)", 0.20)

    if not host:
        add("no_host", "Link has no readable destination", 0.60)
        report.risk_score = min(sum(f.weight for f in report.findings), 1.0)
        return report

    # -- credentials embedded in the authority -----------------------------
    # https://paypal.com@evil.xyz reads as PayPal but goes to evil.xyz.
    if parts.username or "@" in (parts.netloc or "").split("/")[0]:
        add("userinfo_in_url", "Text before '@' disguises the real destination", 0.95,
            (parts.netloc or "").split("@")[0])

    # -- numeric and obfuscated hosts --------------------------------------
    ip = host_as_ip(host)
    if ip:
        add("ip_host", "Address is a raw IP, not a domain name", 0.85,
            f"{host} -> {ip}" if host != ip else ip)
        try:
            addr = ipaddress.ip_address(ip)
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                add("internal_host", "Points inside a private or local network", 0.90, ip)
        except ValueError:
            pass

    # -- punycode / homograph ----------------------------------------------
    decoded = decode_punycode(host)
    if decoded and decoded != host:
        mixed = LATIN_RE.search(decoded) and CYRILLIC_GREEK_RE.search(decoded)
        add("punycode_host",
            "Domain uses look-alike characters from another alphabet" if mixed
            else "Domain uses non-Latin characters",
            0.95 if mixed else 0.55, f"{host} -> {decoded}")
    if any(ord(c) > 127 for c in host):
        folded = normalise(host)
        if folded.homoglyphs_mapped or folded.mixed_script_words:
            add("homoglyph_host", "Domain mixes alphabets to imitate a real one", 0.95,
                f"{host} -> {folded.text}")

    # -- domain structure ---------------------------------------------------
    reg = registrable_domain(host, multi_label) if not ip else ""
    report.registrable_domain = reg
    labels = host.split(".")
    subdomain_depth = max(len(labels) - len(reg.split(".")), 0) if reg else 0
    is_safe = reg in safe_domains

    if reg in set(cfg.get("safe_domains", [])):
        pass  # trusted registrable domain; structural checks below still apply
    if subdomain_depth > int(scoring.get("max_subdomain_depth", 4)):
        add("deep_subdomains", "Unusually long chain of subdomains", 0.35,
            f"{subdomain_depth} levels")

    # A brand name that appears anywhere except the registrable domain is the
    # single most common phishing structure: paypal.com.secure-login.xyz
    if reg and not is_safe:
        prefix = host[: -len(reg)].rstrip(".") if host.endswith(reg) else host
        reg_name = reg.split(".")[0]
        for brand in lex.impersonated_brands:
            if brand in prefix and brand not in reg:
                add("brand_in_subdomain",
                    f"Looks like {brand} but the real domain is {reg}", 0.95,
                    f"{host} is controlled by whoever owns {reg}")
                break
            # The brand baked into the registrable domain itself:
            # sbi-kyc-verify.xyz is not SBI, it is whoever registered it.
            if brand in reg_name and reg_name != brand:
                official = lex.legit_domain_hints.get(brand)
                add("brand_in_domain",
                    f"Uses the name {brand} but is not "
                    f"{official or 'the official domain'}", 0.90,
                    f"registered domain is {reg}")
                break

    # -- typosquatting -------------------------------------------------------
    if reg and not is_safe and "." in reg:
        reg_name = reg.split(".")[0]
        limit = int(scoring.get("typosquat_max_distance", 2))
        for safe in safe_domains:
            safe_name = safe.split(".")[0]
            if len(safe_name) < 5 or reg_name == safe_name:
                continue
            distance = _edit_distance(reg_name, safe_name, limit)
            if 0 < distance <= limit:
                add("typosquat", f"Domain is a near-copy of {safe}", 0.90,
                    f"{reg} vs {safe} ({distance} character change"
                    f"{'s' if distance > 1 else ''})")
                break

    # -- reputation of the label itself -------------------------------------
    tld = reg.rsplit(".", 1)[-1] if reg else ""
    if tld and tld in lex.suspicious_tlds and not is_safe:
        add("suspicious_tld", "Domain ending is heavily abused", 0.55, f".{tld}")
    if reg in lex.shorteners:
        report.is_shortener = True
        add("shortener", "Shortened link hides where it actually goes", 0.60, reg)
    if reg and not is_safe and not reg.startswith("xn--") and reg.count("-") >= 2:
        add("hyphenated_domain", "Domain padded with hyphens to look official", 0.35, reg)

    # -- port ---------------------------------------------------------------
    try:
        port = parts.port
    except ValueError:
        port = None
        add("malformed_authority", "Destination is malformed", 0.55,
            (parts.netloc or "")[:60])
    if port and port not in (80, 443):
        add("unusual_port", "Connects on an unusual port", 0.40, str(port))

    # -- path and query ------------------------------------------------------
    path = unquote(parts.path or "")
    ext = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
    if ext in set(cfg.get("dangerous_extensions", [])):
        add("dangerous_download", "Link downloads an installable file", 0.95, f".{ext}")

    query = parse_qs(parts.query or "")
    for key, values in query.items():
        if key.lower() not in set(cfg.get("redirect_params", [])):
            continue
        for value in values:
            nested = unquote(value)
            if nested.lower().startswith(("http://", "https://")):
                report.embedded_urls.append(nested)
            else:
                decoded_b64 = _maybe_base64_url(value)
                if decoded_b64:
                    report.embedded_urls.append(decoded_b64)
    if report.embedded_urls:
        add("embedded_redirect", "Carries a second, hidden destination", 0.70,
            report.embedded_urls[0][:80])

    if len(raw) > int(scoring.get("long_url_chars", 120)):
        add("very_long_url", "Unusually long link", 0.25, f"{len(raw)} characters")
    if raw.count("%") >= 6:
        add("heavy_encoding", "Heavily encoded to obscure its contents", 0.40)

    # A trusted registrable domain cannot be undone by cosmetic findings, but
    # structural deceptions above still stand on their own.
    structural = {"userinfo_in_url", "hostile_scheme", "brand_in_subdomain",
                  "dangerous_download", "embedded_redirect", "punycode_host",
                  "homoglyph_host", "ip_host"}
    if is_safe:
        report.findings = [f for f in report.findings if f.code in structural]

    if report.findings:
        strongest = max(f.weight for f in report.findings)
        total = sum(f.weight for f in report.findings)
        # Saturating, like the message rule score: many weak hints must not
        # outrank one decisive structural deception.
        report.risk_score = min(max(strongest, 1 - 2.718 ** (-0.6 * total)), 1.0)
    report.findings.sort(key=lambda f: -f.weight)
    return report


def analyse_links_in(text: str, lex: Lexicons | None = None) -> list[LinkReport]:
    """Analyse every link in a message, following one level of embedding."""
    lex = lex or load_lexicons()
    reports: list[LinkReport] = []
    seen: set[str] = set()
    for url in extract_links(text):
        if url in seen:
            continue
        seen.add(url)
        report = analyse_link(url, lex)
        reports.append(report)
        for nested in report.embedded_urls[:2]:
            if nested not in seen:
                seen.add(nested)
                reports.append(analyse_link(nested, lex))
    return reports


def worst_link_score(text: str, lex: Lexicons | None = None) -> float:
    reports = analyse_links_in(text, lex)
    return max((r.risk_score for r in reports), default=0.0)


def _unicodedata_guard() -> None:  # pragma: no cover - import-time sanity
    unicodedata.normalize("NFKC", "")
