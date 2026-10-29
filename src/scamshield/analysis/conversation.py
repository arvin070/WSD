"""
Conversation-level threat analysis.

A single message is often ambiguous ("can you help me with something?").
Scams reveal themselves across a *thread*: an unknown number opens warmly,
builds rapport, then pivots to a request. This module scores each message with
the detector and then reasons about the shape of the whole conversation.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import timedelta

from ..config import Settings, get_settings
from ..core.business import BUSINESS_OFFICIAL, BUSINESS_UNVERIFIED, claimed_brands
from ..core.chat_parser import ChatMessage, ParsedChat
from ..ml.detector import CATEGORY_LABELS, ScamDetector
from ..resources import load_lexicons

#: A sender stored as a raw number was never in the user's contacts.
UNKNOWN_SENDER_RE = re.compile(r"^\+?[\d\s\-()]{7,}$")

CONVERSATION_ADVICE = {
    "dangerous": "Stop engaging. Block and report this contact, and do not send money or codes.",
    "likely_scam": "This conversation follows a known scam pattern. Verify independently before acting.",
    "suspicious": "Some warning signs. Confirm the sender's identity through a number you already trust.",
    "safe": "No meaningful scam indicators in this conversation.",
}

ESCALATION_NOTE = (
    "Risk rises sharply later in the thread - the classic build-trust-then-ask pattern."
)
LINK_SIGNAL_CODES = {"link_shortener", "suspicious_tld", "ip_link", "lookalike_domain"}
#: Signals severe enough that a verified badge should not soften them.
SEVERE_SIGNAL_CODES = {"credentials", "payment", "secrecy", "threat",
                       "lookalike_domain", "ip_link", "homoglyph_spoof"}


def is_unknown_sender(sender: str | None) -> bool:
    return bool(sender) and bool(UNKNOWN_SENDER_RE.match(sender.strip()))


def _band(score: float, detector: ScamDetector) -> tuple[str, str]:
    verdict = next(v for t, v, _ in detector.bands if score >= t)
    return verdict, CONVERSATION_ADVICE[verdict]


def detect_escalation(scores: list[float], cfg: Settings) -> tuple[bool, float]:
    """
    Compare the opening third of a sender's messages with the closing third.

    Peaks, not means: a grooming thread stays chatty throughout, so harmless
    filler around the actual ask ("I told you! 🎉") drags a mean down and hides
    the escalation. What matters is that the *worst* thing said early was
    benign and the worst thing said later was not.
    """
    if len(scores) < cfg.escalation_min_messages:
        return False, 0.0
    third = max(len(scores) // 3, 1)
    early_peak, late_peak = max(scores[:third]), max(scores[-third:])
    delta = late_peak - early_peak
    return bool(delta >= cfg.escalation_min_delta
                and late_peak >= cfg.escalation_min_late_peak), round(delta, 4)


def _burst_count(messages: list[ChatMessage], window: int) -> int:
    """Messages sent within `window` seconds of the previous one by the same sender."""
    burst = 0
    for prev, cur in zip(messages, messages[1:], strict=False):
        if (prev.timestamp and cur.timestamp
                and timedelta(0) <= cur.timestamp - prev.timestamp <= timedelta(seconds=window)):
            burst += 1
    return burst


def _empty_result(chat: ParsedChat) -> dict:
    return {
        "summary": {
            "risk_score": 0.0, "verdict": "safe",
            "advice": "No readable messages were found in this chat.",
            "message_count": 0, "flagged_count": 0,
            "participants": chat.participants, "unparsed_lines": chat.unparsed_lines,
            "account_type": chat.account_type, "account_label": chat.business.label,
            "primary_category": None, "category_breakdown": [],
        },
        "senders": [], "flagged_messages": [], "messages": [], "timeline": [],
    }


def _business_adjustment(items: list[dict], brands: list[str], chat: ParsedChat,
                         scoring: dict) -> tuple[float, float, list[str]]:
    """
    Adjust a sender's risk for the account standing recorded in the export.

    Being a business is not itself suspicious - most are exactly what they
    claim. What matters is the gap between claimed identity and actual
    standing: an *unverified* account presenting itself as a bank while asking
    for a code is the case worth flagging, and a Meta-verified account sending
    an ordinary order update is the case worth quietening.
    """
    boost, factor, reasons = 0.0, 1.0, []
    account = chat.account_type
    severe = any(sg["code"] in SEVERE_SIGNAL_CODES for i in items for sg in i["signals"])

    if account == BUSINESS_UNVERIFIED:
        if brands:
            boost += scoring.get("boost_unverified_business_claim", 0.15)
            reasons.append(
                f"Unverified business account presenting itself as "
                f"{', '.join(sorted(set(brands))[:3])}. WhatsApp has not confirmed this identity.")
        elif severe:
            boost += scoring.get("boost_unverified_business_claim", 0.15) / 2
            reasons.append("Unverified business account making a sensitive request.")
    elif account == BUSINESS_OFFICIAL:
        if severe:
            # A verified badge is not a licence to ask for an OTP. Report the
            # contradiction rather than discounting it away.
            reasons.append(
                "Marked an official business account, yet the messages request "
                "credentials or payment - verify through the company's own app before acting.")
        else:
            factor = scoring.get("discount_official_business", 0.55)
            reasons.append("Official business account, verified by WhatsApp.")
    elif brands:
        reasons.append(
            f"Claims to represent {', '.join(sorted(set(brands))[:3])}, but the chat carries "
            f"no business-account marker.")
    return boost, factor, reasons


def analyse_conversation(chat: ParsedChat, detector: ScamDetector,
                         focus_sender: str | None = None,
                         settings: Settings | None = None) -> dict:
    cfg = settings or get_settings()
    msgs = chat.user_messages
    if not msgs:
        return _empty_result(chat)
    lex = load_lexicons()

    scored = detector.score_batch([m.text for m in msgs])
    for m, s in zip(msgs, scored, strict=True):
        s["index"] = m.index
        s["sender"] = m.sender
        s["timestamp"] = m.timestamp.isoformat() if m.timestamp else None

    # ---- per-sender profiles ---------------------------------------------
    by_sender: dict[str, list[dict]] = {}
    for s in scored:
        by_sender.setdefault(s["sender"], []).append(s)

    senders: list[dict] = []
    for sender, items in by_sender.items():
        scores = [i["risk_score"] for i in items]
        flagged = [i for i in items if i["is_scam"]]
        escalated, delta = detect_escalation(scores, cfg)
        sender_msgs = [m for m in msgs if m.sender == sender]

        top = sorted(scores, reverse=True)[:3]
        # Peak risk dominates, but a consistently bad thread is worse than one
        # bad line in an otherwise normal chat.
        base = 0.7 * max(scores) + 0.3 * (sum(top) / len(top))
        if escalated:
            base = min(base + 0.10, 1.0)
        if is_unknown_sender(sender) and max(scores) > 0.3:
            base = min(base + 0.05, 1.0)

        reasons: list[str] = []
        if escalated:
            reasons.append(ESCALATION_NOTE)
        if is_unknown_sender(sender):
            reasons.append("Sender is an unsaved number, not a contact you have stored.")
        if len(flagged) > 1:
            reasons.append(f"{len(flagged)} separate messages from this sender were flagged.")
        link_msgs = sum(1 for i in items
                        if any(sg["code"] in LINK_SIGNAL_CODES for sg in i["signals"]))
        if link_msgs:
            reasons.append(f"{link_msgs} message(s) contained a suspicious link.")
        if _burst_count(sender_msgs, cfg.burst_window_seconds) >= 3 and len(sender_msgs) >= 5:
            reasons.append("Messages arrive in rapid bursts, typical of scripted outreach.")

        brands = [b for i in items for b in claimed_brands(i["text"], lex.impersonated_brands)]
        boost, factor, business_reasons = _business_adjustment(items, brands, chat, lex.scoring)
        base = min(base * factor + boost, 1.0)
        reasons.extend(business_reasons)

        verdict, advice = _band(base, detector)
        cats = Counter(i["category"] for i in flagged if i["category"])
        senders.append({
            "sender": sender,
            "message_count": len(items),
            "flagged_count": len(flagged),
            "risk_score": round(base, 4),
            "max_message_risk": round(max(scores), 4),
            "mean_message_risk": round(sum(scores) / len(scores), 4),
            "verdict": verdict, "advice": advice,
            "is_unknown_number": is_unknown_sender(sender),
            "claimed_brands": sorted(set(brands)),
            "escalation_detected": escalated, "escalation_delta": delta,
            "top_categories": [{"category": c, "label": CATEGORY_LABELS.get(c, c), "count": n}
                               for c, n in cats.most_common(3)],
            "reasons": reasons,
        })
    senders.sort(key=lambda s: -s["risk_score"])

    # ---- conversation summary --------------------------------------------
    pool = [s for s in senders if s["sender"] == focus_sender] if focus_sender else senders
    pool = pool or senders
    overall = max(s["risk_score"] for s in pool)
    verdict, advice = _band(overall, detector)

    flagged_messages = sorted(
        [s for s in scored if s["is_scam"] or s["risk_score"] >= detector.bands[2][0]],
        key=lambda s: -s["risk_score"])
    all_cats = Counter(s["category"] for s in scored if s["is_scam"] and s["category"])

    # ---- timeline: risk across the thread, bucketed for charting ---------
    bucket = max(len(scored) // 40, 1)
    timeline = [{
        "position": i,
        "timestamp": scored[i:i + bucket][0]["timestamp"],
        "risk": round(max(c["risk_score"] for c in scored[i:i + bucket]), 4),
        "sender": scored[i:i + bucket][0]["sender"],
    } for i in range(0, len(scored), bucket)]

    primary = all_cats.most_common(1)
    return {
        "summary": {
            "risk_score": round(overall, 4), "verdict": verdict, "advice": advice,
            "message_count": len(msgs),
            "flagged_count": sum(1 for s in scored if s["is_scam"]),
            "participants": chat.participants,
            "unparsed_lines": chat.unparsed_lines,
            "account_type": chat.account_type,
            "account_label": chat.business.label,
            "primary_category": (
                {"category": primary[0][0],
                 "label": CATEGORY_LABELS.get(primary[0][0], primary[0][0])} if primary else None),
            "category_breakdown": [{"category": c, "label": CATEGORY_LABELS.get(c, c), "count": n}
                                   for c, n in all_cats.most_common()],
        },
        "senders": senders,
        "flagged_messages": flagged_messages[:60],
        "messages": scored,
        "timeline": timeline,
    }
