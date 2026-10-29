"""Rule-engine behaviour: the signals must fire on attacks, not on talk about attacks."""

import pytest

from scamshield.core.features import analyse_message, extract_urls, rule_score

SCAMS = [
    "Dear Customer, your SBI KYC expired. Account blocked today. Update: http://sbi-kyc-verify.xyz/l",
    "Congratulations! You won Rs 25,00,000. Pay processing fee 5500 to winner@paytm to claim.",
    "Please share the OTP sent to your phone so I can verify your account",
    "Sir main bank se bol raha hun, jo OTP aaya hai wo batayein",
]
HAM = [
    "hey are we still on for lunch at 1? i will book the table",
    "Sure, I sent the report to your email yesterday.",
    "beta ghar kab aa rahe ho?",
]


@pytest.mark.parametrize("text", SCAMS)
def test_scam_messages_trigger_signals(text):
    score, signals = rule_score(text)
    assert score > 0.5, f"expected a high rule score for: {text}"
    assert signals, "a flagged message must carry at least one explanation"


@pytest.mark.parametrize("text", HAM)
def test_ordinary_messages_are_quiet(text):
    score, _ = rule_score(text)
    assert score < 0.2


def test_discussing_a_scam_is_discounted():
    """A friend recounting a scam call contains the same keywords as the scam."""
    attack, _ = rule_score("Please share the OTP sent to your phone to verify")
    talk, _ = rule_score("someone called asking for my otp, I disconnected immediately")
    assert talk < attack / 2


def test_genuine_bank_alert_is_discounted():
    score, _ = rule_score(
        "Your OTP for HDFC net banking is 883021. Valid for 10 minutes. Never share this with anyone."
    )
    assert score < 0.5


def test_reassurance_plus_an_ask_is_not_discounted():
    """Scams borrow the reassuring wording, so the discount must not apply."""
    score, _ = rule_score(
        "Do not share this OTP with anyone... please forward it to this number for validation only."
    )
    assert score > 0.6


def test_lookalike_domain_detected():
    _, signals = rule_score("Claim your refund at http://amazon-reward-in.buzz/win")
    assert any(s.code in {"lookalike_domain", "suspicious_tld"} for s in signals)


def test_hidden_characters_detected():
    _, signals = rule_score("Sh​are your OT​P now please to verify account")
    assert any(s.code == "hidden_chars" for s in signals)


def test_url_extraction_ignores_numbers():
    assert extract_urls("the total was 3.50 and 12.4 percent") == []
    assert "bit.ly/abc123" in " ".join(extract_urls("click bit.ly/abc123 now"))


def test_feature_vector_is_stable():
    a, _ = analyse_message("hello")
    b, _ = analyse_message("Share your OTP at http://x.xyz")
    assert a.keys() == b.keys()
    assert all(isinstance(v, float) for v in b.values())
