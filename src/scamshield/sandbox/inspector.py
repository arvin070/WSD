"""
Opt-in URL inspection.

Retrieves a link in isolation to answer the questions static analysis cannot:
where does this shortener actually land, and does the page at the end ask for
a password?

What it does **not** do is as important as what it does. No JavaScript is
executed, nothing is written to disk, no cookies or credentials are sent, and
redirects are followed manually so each hop can be revalidated. It is an
inspector, not a detonation chamber: analysing genuine malware needs a
disposable virtual machine, which is not something a library can honestly
claim to provide.

**Fetching is not free.** Requesting an attacker's URL tells them the target
is live and reveals the requesting IP. That is why it is off by default and
never runs automatically on inbound messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

import httpx

from ..config import Settings, get_settings
from ..core.links import analyse_link
from ..logging_setup import get_logger
from ..resources import load_lexicons
from .guard import BlockedTarget, resolve_and_check

log = get_logger(__name__)

SENSITIVE_INPUTS = {"password", "tel", "number"}
CREDENTIAL_WORDS = re.compile(
    r"(?i)\b(?:password|passcode|otp|one[- ]time|cvv|pin|card number|expiry|"
    r"aadhaar|aadhar|net ?banking|upi pin|mpin|verify your identity)\b")


class _PageParser(HTMLParser):
    """Minimal, non-executing scrape of the things that matter on a login page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self.input_types: list[str] = []
        self.form_actions: list[str] = []
        self.has_password_field = False
        self.text_parts: list[str] = []
        self.meta_refresh: str | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "input":
            itype = (a.get("type") or "text").lower()
            self.input_types.append(itype)
            if itype == "password":
                self.has_password_field = True
        elif tag == "form" and a.get("action"):
            self.form_actions.append(a["action"])
        elif tag == "meta" and (a.get("http-equiv") or "").lower() == "refresh":
            content = a.get("content") or ""
            match = re.search(r"url=([^;]+)", content, re.I)
            if match:
                self.meta_refresh = match.group(1).strip().strip("'\"")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data.strip()
        elif data.strip():
            self.text_parts.append(data.strip())


@dataclass
class Hop:
    url: str
    status: int
    location: str | None = None


@dataclass
class InspectionResult:
    url: str
    ok: bool
    final_url: str = ""
    chain: list[Hop] = field(default_factory=list)
    status: int | None = None
    content_type: str = ""
    title: str = ""
    server: str = ""
    has_password_field: bool = False
    asks_for_credentials: bool = False
    impersonated_brands: list[str] = field(default_factory=list)
    static_risk: float = 0.0
    final_static_risk: float = 0.0
    risk_score: float = 0.0
    findings: list[str] = field(default_factory=list)
    error: str = ""

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
            "url": self.url, "ok": self.ok, "error": self.error,
            "final_url": self.final_url, "status": self.status,
            "redirects": [{"url": h.url, "status": h.status, "location": h.location}
                          for h in self.chain],
            "content_type": self.content_type, "title": self.title, "server": self.server,
            "has_password_field": self.has_password_field,
            "asks_for_credentials": self.asks_for_credentials,
            "impersonated_brands": self.impersonated_brands,
            "static_risk": round(self.static_risk, 4),
            "final_static_risk": round(self.final_static_risk, 4),
            "risk_score": round(self.risk_score, 4), "verdict": self.verdict,
            "findings": self.findings,
        }


class SandboxDisabled(RuntimeError):
    """Inspection was requested but is not switched on."""


async def inspect_url(url: str, settings: Settings | None = None) -> InspectionResult:
    """Fetch and inspect one URL under the guards in `sandbox.guard`."""
    cfg = settings or get_settings()
    if not cfg.sandbox_enabled:
        raise SandboxDisabled(
            "URL inspection is off. Set SCAMSHIELD_SANDBOX_ENABLED=true to allow "
            "outbound fetches - note this reveals your IP to the link's owner.")

    lex = load_lexicons()
    result = InspectionResult(url=url, ok=False)
    result.static_risk = analyse_link(url, lex).risk_score

    headers = {
        "User-Agent": cfg.sandbox_user_agent,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
        "Accept-Language": "en",
        # No Referer, no Cookie, no Authorization - nothing that identifies a user.
    }

    current = url
    body = b""
    try:
        # follow_redirects stays off so each hop can be revalidated.
        async with httpx.AsyncClient(timeout=cfg.sandbox_timeout_seconds,
                                     follow_redirects=False, http2=False,
                                     trust_env=False) as client:
            for _ in range(cfg.sandbox_max_redirects + 1):
                resolve_and_check(current, cfg.sandbox_allow_private_hosts)
                response = await client.get(current, headers=headers)
                location = response.headers.get("location")
                result.chain.append(Hop(url=current, status=response.status_code,
                                        location=location))
                if response.is_redirect and location:
                    current = str(response.url.join(location))
                    continue

                result.status = response.status_code
                result.content_type = response.headers.get("content-type", "")[:120]
                result.server = response.headers.get("server", "")[:80]
                # Read at most max_bytes; never write anything to disk.
                chunks, total = [], 0
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= cfg.sandbox_max_bytes:
                        result.findings.append("Response truncated at the size limit.")
                        break
                body = b"".join(chunks)[: cfg.sandbox_max_bytes]
                break
            else:
                result.findings.append("Redirect limit reached before a final page.")
        result.ok = True
        result.final_url = current
    except BlockedTarget as exc:
        result.error = f"blocked: {exc}"
        result.risk_score = max(result.static_risk, 0.85)
        result.findings.append(f"Refused to fetch — {exc}")
        return result
    except (httpx.HTTPError, OSError) as exc:
        result.error = f"fetch failed: {type(exc).__name__}"
        result.risk_score = result.static_risk
        result.findings.append("The link could not be reached.")
        return result

    result.final_static_risk = analyse_link(result.final_url, lex).risk_score

    if "html" in result.content_type.lower() or body[:200].lstrip().lower().startswith(b"<"):
        parser = _PageParser()
        try:
            parser.feed(body.decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001 - malformed HTML must not break inspection
            log.debug("HTML parse failed for %s", result.final_url)
        result.title = parser.title[:160]
        result.has_password_field = parser.has_password_field
        page_text = " ".join(parser.text_parts)[:4000]
        result.asks_for_credentials = bool(CREDENTIAL_WORDS.search(page_text)) or \
            result.has_password_field
        haystack = f"{result.title} {page_text}".lower()
        final_host = (result.final_url or "").lower()
        result.impersonated_brands = sorted({
            b for b in lex.impersonated_brands if b in haystack and b not in final_host})
        if parser.meta_refresh:
            result.findings.append(f"Page silently redirects to {parser.meta_refresh[:80]}")

    # -- combine -----------------------------------------------------------
    score = max(result.static_risk, result.final_static_risk)
    if len(result.chain) > 1:
        result.findings.append(
            f"Redirects {len(result.chain) - 1} time(s), ending at {result.final_url[:80]}")
        score = max(score, 0.35)
    if result.has_password_field:
        result.findings.append("The page asks for a password.")
        score = max(score, 0.80)
    if result.asks_for_credentials and not result.has_password_field:
        result.findings.append("The page asks for codes or card details.")
        score = max(score, 0.70)
    if result.impersonated_brands:
        names = ", ".join(result.impersonated_brands[:3])
        result.findings.append(
            f"Page presents itself as {names} but is served from {result.final_url[:60]}")
        score = max(score, 0.90)
    result.risk_score = min(score, 1.0)
    return result
