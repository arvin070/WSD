"""
Hand-engineered scam signals.

These complement the TF-IDF model: TF-IDF learns *vocabulary*, these capture
*structure and intent* — links, money asks, urgency, credential harvesting,
obfuscation — which generalises to wording the corpus has never seen.

All lexicons, labels, weights and scoring constants come from
`resources/lexicons.yaml`; this module holds only the logic that applies them.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import entropy as shannon_entropy
from sklearn.base import BaseEstimator, TransformerMixin

from ..resources import Lexicons, load_lexicons
from .business import claimed_brands, claims_legitimacy, claims_support_role
from .links import analyse_link
from .normalizer import normalise
from .signals import (
    ACCOUNT_RE,
    ASK_RE,
    DIRECTIVE_QUESTION_RE,
    EMAIL_RE,
    EMOJI_RE,
    IFSC_RE,
    IP_URL_RE,
    MONEY_RE,
    PHONE_RE,
    QUESTION_RE,
    REPEATED_PUNCT_RE,
    SELF_INQUIRY_RE,
    UPI_RE,
    URL_RE,
)


@dataclass
class Signal:
    """One human-readable red flag found in a message."""

    code: str
    label: str
    weight: float
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "label": self.label,
            "weight": round(self.weight, 3),
            "evidence": self.evidence[:4],
        }


def normalized_entropy(s: str) -> float:
    """
    Shannon entropy of the character distribution, scaled to [0, 1].

    Dividing by log(length) gives the "efficiency" of the string: ordinary
    prose sits well below 1, while a random-looking URL slug or a padded
    zero-width payload approaches it. That separates `bit.ly/aX9k2Qz` and
    filter-evading noise from `amazon.in/orders`.
    """
    if len(s) < 2:
        return 0.0
    counts = list(Counter(s).values())
    h = float(shannon_entropy(counts))  # natural log
    return min(h / math.log(len(s)), 1.0)


def extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for m in URL_RE.finditer(text):
        raw = m.group(1).rstrip(".,;:!?)")
        if re.fullmatch(r"[\d.]+", raw) or "." not in raw:
            continue  # version numbers and bare decimals are not links
        urls.append(raw)
    return urls


def _domain_of(url: str) -> str:
    d = re.sub(r"^https?://", "", url, flags=re.I).split("/")[0].split("?")[0].split(":")[0]
    d = d.lower()
    return d[4:] if d.startswith("www.") else d


def _max_amount(money: list[str]) -> float:
    biggest = 0.0
    for m in money:
        digits = re.sub(r"[^\d.]", "", m)
        try:
            val = float(digits) if digits else 0.0
        except ValueError:
            val = 0.0
        if re.search(r"lakh|lac|crore|cr\b|million|mn\b", m, re.I):
            val *= 100_000
        biggest = max(biggest, val)
    return biggest


def analyse_message(text: str, lex: Lexicons | None = None) -> tuple[dict[str, float], list[Signal]]:
    """Return (numeric feature dict, triggered human-readable signals)."""
    lex = lex or load_lexicons()
    sc = lex.scoring
    raw = text or ""
    # Step one, always: fold script spoofing so lexicon matching sees the
    # word the attacker meant to hide, and keep the edit counts as evidence.
    norm = normalise(raw)
    text = norm.text
    low = text.lower()
    words = re.findall(r"[a-z']+", low)
    n_words, n_chars = max(len(words), 1), max(len(text), 1)

    signals: list[Signal] = []

    def raise_signal(code: str, evidence: list[str]) -> None:
        if evidence:
            d = lex.derived[code]
            signals.append(Signal(d.code, d.label, d.weight, evidence))

    # ---- lexicon families -------------------------------------------------
    found: dict[str, list[str]] = {}
    features: dict[str, float] = {}
    for sig in lex.signals:
        hits = [t for t in sig.terms if t in low]
        found[sig.code] = hits
        if hits:
            signals.append(Signal(sig.code, sig.label, sig.weight, hits))
        features[sig.feature] = float(bool(hits)) if sig.count_mode == "bool" else math.log1p(len(hits))

    def has(code: str) -> bool:
        return bool(found.get(code))

    # ---- context: what pushes the score *down* ---------------------------
    narrative = [t for t in lex.narrative if t in low]
    safe_marker = [t for t in lex.safe_transactional if t in low]
    asks = ASK_RE.findall(text)
    # Business-identity claims. Cheap to assert, so they matter mainly in
    # combination with an ask - which is what the interactions below capture.
    brand_claims = claimed_brands(text, lex.impersonated_brands)
    support_role = claims_support_role(text)
    legitimacy = claims_legitimacy(text)
    is_question = bool(QUESTION_RE.search(text))
    self_inquiry = bool(SELF_INQUIRY_RE.search(text))
    directive_question = bool(DIRECTIVE_QUESTION_RE.search(text))
    features.update({
        "n_narrative": math.log1p(len(narrative)),
        "n_safe_marker": math.log1p(len(safe_marker)),
        "has_ask": float(bool(asks)),
        "x_narrative_noask": float(bool(narrative) and not asks),
        "x_safe_noask": float(bool(safe_marker) and not asks),
        # directionality: asking *about* an action vs being told to perform it
        "is_question": float(is_question),
        "is_self_inquiry": float(self_inquiry),
        "is_directive_question": float(directive_question),
        "x_inquiry_not_directive": float(self_inquiry and not directive_question),
        # claimed business identity
        "claims_brand_identity": float(bool(brand_claims)),
        "n_brand_claims": math.log1p(len(brand_claims)),
        "claims_support_role": float(support_role),
        "claims_legitimacy": float(legitimacy),
    })

    # ---- links ------------------------------------------------------------
    urls = extract_urls(text)
    domains = [_domain_of(u) for u in urls]
    shortened = [d for d in domains if d in lex.shorteners]
    sus_tld = [d for d in domains if d.rsplit(".", 1)[-1] in lex.suspicious_tlds]
    ip_urls = IP_URL_RE.findall(text)

    lookalike: list[str] = []
    for d in domains:
        for brand in lex.impersonated_brands:
            if brand in d:
                legit = lex.legit_domain_hints.get(brand)
                base = ".".join(d.split(".")[-2:])
                # Either the brand has a known real domain and this is not it,
                # or we have no reference and the host looks like a mashup.
                impostor = (base != legit and not d.endswith(legit)) if legit \
                    else d.count("-") >= 1
                if impostor:
                    lookalike.append(d)
                break

    # Full structural analysis of each link: brand-in-subdomain, punycode,
    # embedded redirects, installable downloads - none of which the coarse
    # flags below can see.
    link_reports = [analyse_link(u, lex) for u in urls[:5]]
    max_link_risk = max((r.risk_score for r in link_reports), default=0.0)
    risky_links = [r for r in link_reports if r.risk_score >= 0.45]
    if risky_links:
        worst = max(risky_links, key=lambda r: r.risk_score)
        top = worst.findings[0].label if worst.findings else "Deceptive link"
        code = "dangerous_download" if any(
            f.code == "dangerous_download" for f in worst.findings) else "risky_link"
        raise_signal(code, [f"{top}: {worst.host or worst.url[:40]}"])

    raise_signal("link_shortener", shortened)
    raise_signal("suspicious_tld", sus_tld)
    raise_signal("ip_link", ip_urls)
    raise_signal("lookalike_domain", sorted(set(lookalike)))

    # ---- financial identifiers -------------------------------------------
    upis = UPI_RE.findall(text)
    ifsc = IFSC_RE.findall(text)
    accounts = [a for a in ACCOUNT_RE.findall(text) if len(a) >= 11]
    money = MONEY_RE.findall(text)
    biggest = _max_amount(money)

    raise_signal("upi_id", upis)
    raise_signal("bank_details", ifsc + accounts)
    if biggest >= sc["large_sum_min"]:
        raise_signal("large_sum", money[:3])

    # ---- style / obfuscation ---------------------------------------------
    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
    emojis = EMOJI_RE.findall(text)
    repeated_punct = REPEATED_PUNCT_RE.findall(text)

    if caps_ratio > sc["caps_ratio_min"] and len(letters) > sc["caps_min_letters"]:
        raise_signal("shouting", [text[:40]])
    if norm.zero_width_removed:
        raise_signal("hidden_chars", [f"{norm.zero_width_removed} zero-width char(s)"])
    if norm.mixed_script_words or norm.homoglyphs_mapped:
        raise_signal("homoglyph_spoof",
                     [f"{norm.homoglyphs_mapped} look-alike char(s) in "
                      f"{norm.mixed_script_words} word(s)"])
    if len(emojis) >= sc["emoji_spam_min"]:
        raise_signal("emoji_spam", emojis[:5])

    # ---- contact pivots ---------------------------------------------------
    phones = [p.strip() for p in PHONE_RE.findall(text) if len(re.sub(r"\D", "", p)) >= 10]
    emails = EMAIL_RE.findall(text)
    if phones and (has("payment") or has("job_scam") or has("impersonation")):
        raise_signal("callback_number", phones[:3])

    # Claiming to be a company is only a red flag alongside a request; a real
    # business introducing itself is not suspicious on its own.
    if brand_claims and (has("credentials") or has("payment") or asks):
        raise_signal("brand_identity_claim", brand_claims)
    if support_role and asks:
        raise_signal("support_role_ask", ["claims a support role while requesting action"])
    if legitimacy and has("payment"):
        raise_signal("legitimacy_claim", ["unverifiable legitimacy claim with a payment ask"])

    # A friend sharing a link is ordinary; a friend-shaped message pushing a
    # structurally deceptive link is how compromised accounts spread. The
    # vouching only counts when the link itself does not hold up.
    if has("vouched_link") and max_link_risk >= 0.45:
        worst_url = max(link_reports, key=lambda r: r.risk_score)
        raise_signal("vouched_risky_link",
                     [f"{worst_url.host or worst_url.url[:40]} "
                      f"({int(worst_url.risk_score * 100)}% link risk)"])
    # Extortion is defined by the pairing: a threat on its own may be bluster,
    # a payment demand on its own may be legitimate. Together they are ransom.
    if has("extortion") and (has("payment") or has("investment") or upis):
        raise_signal("ransom_payment", found["extortion"][:2])
    if has("course_scam") and (has("payment") or has("job_scam") or upis):
        raise_signal("course_upsell", found["course_scam"][:2])

    features.update({
        "n_urls": math.log1p(len(urls)),
        "has_url": float(bool(urls)),
        "has_shortener": float(bool(shortened)),
        "has_suspicious_tld": float(bool(sus_tld)),
        "has_ip_url": float(bool(ip_urls)),
        "has_lookalike": float(bool(lookalike)),
        "max_link_risk": max_link_risk,
        "n_risky_links": math.log1p(len(risky_links)),
        "has_upi": float(bool(upis)),
        "has_bank_details": float(bool(ifsc or accounts)),
        "has_money": float(bool(money)),
        "log_max_amount": math.log1p(biggest),
        "has_phone": float(bool(phones)),
        "has_email": float(bool(emails)),
        "caps_ratio": caps_ratio,
        "n_emoji": math.log1p(len(emojis)),
        "n_exclam": math.log1p(text.count("!")),
        "n_repeated_punct": math.log1p(len(repeated_punct)),
        "len_words": math.log1p(n_words),
        "avg_word_len": sum(len(w) for w in words) / n_words,
        "digit_ratio": sum(c.isdigit() for c in text) / n_chars,
        "non_ascii_ratio": sum(ord(c) > 127 for c in text) / n_chars,
        # obfuscation: what had to be undone, and how random the text looks
        "char_entropy": normalized_entropy(text),
        "max_url_entropy": max((normalized_entropy(u) for u in urls), default=0.0),
        "n_homoglyphs": math.log1p(norm.homoglyphs_mapped),
        "n_mixed_script": math.log1p(norm.mixed_script_words),
        "has_zero_width": float(bool(norm.zero_width_removed)),
        "n_zero_width": math.log1p(norm.zero_width_removed),
        "n_digits_folded": math.log1p(norm.digits_folded),
        # interactions - the combinations that make a scam a scam
        "x_urgency_link": float(has("urgency") and bool(urls)),
        "x_urgency_money": float(has("urgency") and (has("payment") or has("credentials"))),
        "x_prize_fee": float(has("prize") and has("payment")),
        "x_threat_cred": float(has("threat") and has("credentials")),
        "x_job_payment": float(has("job_scam") and (has("payment") or has("investment"))),
        "x_brandlink_cred": float(has("impersonation") and (bool(urls) or has("credentials"))),
        "x_romance_money": float(has("romance") and (has("payment") or has("investment"))),
        # an identity claim paired with an ask is the impersonation pattern
        "x_brandclaim_credential": float(bool(brand_claims) and has("credentials")),
        "x_brandclaim_payment": float(bool(brand_claims) and has("payment")),
        "x_brandclaim_link": float(bool(brand_claims) and bool(urls)),
        "x_role_ask": float(support_role and bool(asks)),
        "x_legitimacy_payment": float(legitimacy and has("payment")),
        # the three playbooks added for dream-selling, ransom and shared links
        "x_course_payment": float(has("course_scam") and (has("payment") or has("job_scam"))),
        "x_course_earnings": float(has("course_scam") and has("investment")),
        "x_extortion_payment": float(has("extortion") and (has("payment") or has("investment"))),
        "x_extortion_secrecy": float(has("extortion") and has("secrecy")),
        "x_vouched_risky_link": float(has("vouched_link") and max_link_risk >= 0.45),
        "x_vouched_any_link": float(has("vouched_link") and bool(urls)),
    })
    return features, signals


def feature_names() -> list[str]:
    """Stable, deterministic ordering of the feature vector."""
    return list(analyse_message("seed text with no signals")[0].keys())


class HeuristicFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer exposing `analyse_message` as a dense matrix."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        self.feature_names_ = feature_names()
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        names = getattr(self, "feature_names_", None) or feature_names()
        lex = load_lexicons()
        # One analysis per message - not per feature.
        rows = []
        for text in X:
            feats = analyse_message(text, lex)[0]
            rows.append([feats[n] for n in names])
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(getattr(self, "feature_names_", None) or feature_names(), dtype=object)


def rule_score(text: str) -> tuple[float, list[Signal]]:
    """
    Transparent, model-free risk score in [0, 1].

    A safety net: an obvious credential-harvesting message stays flagged even
    if it is worded unlike anything in the training set.
    """
    lex = load_lexicons()
    sc = lex.scoring
    feats, signals = analyse_message(text, lex)
    if not signals:
        return 0.0, []

    # Saturating sum: many weak signals must not outrank one decisive signal.
    total = sum(s.weight for s in signals)
    score = 1.0 - math.exp(-sc["saturation_rate"] * total)
    score = max(score, max(s.weight for s in signals) * sc["strongest_floor"])

    # Context discount. A message that merely *mentions* OTPs — a friend
    # recounting a scam call, or a genuine bank alert saying "never share
    # this" — carries the same keywords as the attack it describes. The
    # discount applies only when the message asks nothing of the reader,
    # because real scams pair the reassuring wording with an ask.
    narrative, safe_marker = feats["n_narrative"] > 0, feats["n_safe_marker"] > 0

    # Directionality first. "Should I share my OTP?" and "why do you need my
    # account number?" contain a textbook credential ask, but the reader is
    # the one asking - they are the target, not the attacker. A second-person
    # directive dressed as a question ("Can you share the OTP?") is exempt,
    # otherwise adding a question mark would be a trivial bypass.
    if feats["x_inquiry_not_directive"]:
        return min(score * sc["discount_self_inquiry"], 1.0), sorted(signals, key=lambda s: -s.weight)

    if feats["has_ask"]:
        if narrative:
            score *= sc["discount_narrative_with_ask"]
    elif narrative:
        score *= sc["discount_narrative_no_ask"]
    elif safe_marker:
        score *= sc["discount_safe_marker_no_ask"]

    return min(score, 1.0), sorted(signals, key=lambda s: -s.weight)
