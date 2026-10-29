"""Composes the advisory a flagged message triggers."""

from __future__ import annotations

HEADER = "⚠️ ScamShield warning"

FOOTER_BY_CATEGORY = {
    "otp_credential_theft":
        "Never share an OTP, PIN or card details — no genuine company will ask for them.",
    "phishing_bank_kyc":
        "Do not use the link. Open your bank's own app or type the address yourself.",
    "business_impersonation":
        "Contact the company through the number on their official website or app, not this chat.",
    "investment_crypto":
        "Guaranteed returns do not exist. Treat any 'no risk' offer as fraudulent.",
    "lottery_prize":
        "A genuine prize never requires a fee to release it.",
    "job_task_scam":
        "A real employer does not ask you to pay to start work.",
    "delivery_customs":
        "Check the parcel in the courier's own app. Do not pay fees requested over chat.",
    "family_impersonation":
        "Call the person on the number you already have saved before sending anything.",
    "digital_arrest":
        "Police and government agencies never demand payment or secrecy over chat.",
    "tech_support":
        "Do not install remote-access apps or share screen codes with anyone who contacts you.",
}
DEFAULT_FOOTER = (
    "Do not send money, codes or personal details. Verify through a channel you already trust.")


def compose_advisory(result: dict, max_reasons: int = 3) -> str:
    """
    Build the customer-facing warning.

    Deliberately plain: what was detected, why, and the one action that
    actually protects them. No jargon, no risk arithmetic beyond a percentage,
    and never an instruction to click anything.
    """
    risk = int(round(result.get("risk_score", 0) * 100))
    label = result.get("category_label")
    what = f" that matches a known pattern: {label}" if label else ""

    lines = [HEADER, "",
             f"The message you just received scores {risk}% on our scam check{what}.".strip()]

    reasons = [s.get("label") for s in (result.get("signals") or [])][:max_reasons]
    if reasons:
        lines += ["", "What stood out:"] + [f"• {r}" for r in reasons if r]

    lines += ["", FOOTER_BY_CATEGORY.get(result.get("category") or "", DEFAULT_FOOTER)]
    return "\n".join(lines)
