"""
Business-account profiling.

Most WhatsApp Business accounts are exactly what they claim to be — a
restaurant confirming a booking, a courier with a delivery update. So
"is a business" is not by itself a risk signal, and treating it as one would
flag every legitimate merchant a user talks to.

What *is* a risk signal is a mismatch between the identity an account claims
and the standing it actually has:

  * an **unverified** account presenting itself as a bank or a major brand,
  * a support/executive role asking for credentials or off-platform payment,
  * a brand claim that no account-level marker corroborates.

WhatsApp exports record the account's standing in a system line, and Meta
distinguishes a plain *business account* from an **official** (verified) one.
That distinction carries most of the weight here: an official Amazon account
saying "your order shipped" is routine; an unverified number saying "Amazon
support here, confirm your card" is the scam this module exists to catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Account standing, ordered from least to most trustworthy.
UNKNOWN = "unknown"
PERSON = "person"
BUSINESS_UNVERIFIED = "business_unverified"
BUSINESS_OFFICIAL = "business_official"

ACCOUNT_TYPE_LABELS = {
    UNKNOWN: "Unknown account",
    PERSON: "Personal account",
    BUSINESS_UNVERIFIED: "Business account (not verified)",
    BUSINESS_OFFICIAL: "Official business account (verified by WhatsApp)",
}

#: Meta marks verified businesses as "official". Checked first - the
#: unverified pattern below would otherwise also match these lines.
OFFICIAL_BUSINESS_RE = re.compile(
    r"(?i)(?:official business account"
    r"|verified by whatsapp"
    r"|this business (?:has been |is )?verified)"
)
BUSINESS_ACCOUNT_RE = re.compile(
    r"(?i)(?:messaging with a business account"
    r"|chat is with a business account"
    r"|you're now messaging with a business"
    r"|this business uses a secure service from meta"
    r"|this business works with other companies to manage this chat"
    r"|business account\.?\s*(?:tap to learn more)?)"
)

#: Role claims. "I'm from support" is trivially cheap to say and is the
#: opening move of most impersonation scams.
SUPPORT_ROLE_RE = re.compile(
    r"(?i)\b(?:customer (?:care|support|service)|support (?:team|executive|agent|desk)"
    r"|helpline|help desk|helpdesk|executive|relationship manager|account manager"
    r"|fraud (?:department|team|prevention)|security team|verification (?:team|department)"
    r"|billing (?:team|department)|technical support|backend team|head office"
    r"|se bol raha hun|se bol rahi hun|se call kar raha)\b"
)

#: Claiming to *be* a named organisation.
_BRAND_CLAIM_TEMPLATES = (
    r"(?:i am|i'm|im|this is|we are|we're|calling|messaging|contacting)\s+"
    r"(?:from|with|at|on behalf of)?\s*{b}",
    r"{b}\s+(?:support|customer care|customer service|helpline|team|executive|"
    r"official|representative|agent|department|security|billing|verification)",
    # "on behalf of X" is an identity claim; a bare "from X" is not - "I
    # ordered from Flipkart" is a customer talking about their own order.
    r"on behalf of\s+(?:the\s+)?{b}",
    r"main\s+{b}\s+se",
)

#: Generic corporate-legitimacy claims, used when no specific brand is named.
LEGITIMACY_CLAIM_RE = re.compile(
    r"(?i)\b(?:registered company|govt (?:approved|registered)|government approved"
    r"|iso certified|gst registered|authorised (?:dealer|agent|partner|distributor)"
    r"|authorized (?:dealer|agent|partner|distributor)|official (?:partner|vendor|seller)"
    r"|licensed (?:agent|dealer)|trusted seller|verified seller)\b"
)


@dataclass(frozen=True)
class BusinessContext:
    """What the export tells us about the account standing of a chat."""

    account_type: str = UNKNOWN
    evidence: str = ""

    @property
    def is_business(self) -> bool:
        return self.account_type in (BUSINESS_UNVERIFIED, BUSINESS_OFFICIAL)

    @property
    def is_verified(self) -> bool:
        return self.account_type == BUSINESS_OFFICIAL

    @property
    def label(self) -> str:
        return ACCOUNT_TYPE_LABELS[self.account_type]


def detect_account_type(text: str) -> BusinessContext | None:
    """Read a WhatsApp system line for an account-standing marker."""
    if not text:
        return None
    if OFFICIAL_BUSINESS_RE.search(text):
        return BusinessContext(BUSINESS_OFFICIAL, text.strip()[:120])
    if BUSINESS_ACCOUNT_RE.search(text):
        return BusinessContext(BUSINESS_UNVERIFIED, text.strip()[:120])
    return None


def _brand_pattern(brand: str) -> re.Pattern[str]:
    b = re.escape(brand)
    return re.compile("(?i)(?:" + "|".join(t.format(b=b) for t in _BRAND_CLAIM_TEMPLATES) + ")")


_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def claimed_brands(text: str, brands: tuple[str, ...]) -> list[str]:
    """
    Brands the message claims to represent.

    Merely *mentioning* Amazon is not a claim ("my Amazon order is late").
    This looks for the identity assertion — "I am from Amazon", "Amazon
    support" — which is what impersonation actually looks like.
    """
    if not text:
        return []
    found = []
    for brand in brands:
        pattern = _PATTERN_CACHE.get(brand)
        if pattern is None:
            pattern = _PATTERN_CACHE[brand] = _brand_pattern(brand)
        if pattern.search(text):
            found.append(brand)
    return found


def claims_support_role(text: str) -> bool:
    return bool(text and SUPPORT_ROLE_RE.search(text))


def claims_legitimacy(text: str) -> bool:
    return bool(text and LEGITIMACY_CLAIM_RE.search(text))
