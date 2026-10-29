"""
Business-account classification.

The failure mode to guard against is over-flagging: most WhatsApp Business
accounts are exactly what they claim to be, so "is a business" must never be
treated as evidence of a scam on its own.
"""

import pytest

from scamshield.analysis.conversation import analyse_conversation
from scamshield.core.business import (
    BUSINESS_OFFICIAL,
    BUSINESS_UNVERIFIED,
    UNKNOWN,
    claimed_brands,
    claims_legitimacy,
    claims_support_role,
    detect_account_type,
)
from scamshield.core.chat_parser import parse_any
from scamshield.core.features import analyse_message
from scamshield.ml.detector import ModelNotTrained, get_detector
from scamshield.resources import load_lexicons

BRANDS = load_lexicons().impersonated_brands

try:
    DET = get_detector()
except ModelNotTrained:  # pragma: no cover
    DET = None

needs_model = pytest.mark.skipif(DET is None, reason="run `scamshield-train` first")


# ---- account-standing markers --------------------------------------------

@pytest.mark.parametrize("line,expected", [
    ("You're messaging with a business account. Tap to learn more.", BUSINESS_UNVERIFIED),
    ("This chat is with a business account.", BUSINESS_UNVERIFIED),
    ("This business uses a secure service from Meta to manage this chat.", BUSINESS_UNVERIFIED),
    ("You're messaging with an official business account. Tap to learn more.", BUSINESS_OFFICIAL),
    ("Messages and calls are end-to-end encrypted.", None),
    ("hey are we still on for lunch?", None),
])
def test_account_marker_detection(line, expected):
    ctx = detect_account_type(line)
    assert (ctx.account_type if ctx else None) == expected


def test_official_outranks_plain_business():
    chat = parse_any(
        "12/03/2024, 10:44 pm - You're messaging with a business account.\n"
        "12/03/2024, 10:44 pm - You're messaging with an official business account.\n"
        "12/03/2024, 10:45 pm - Acme: hello")
    assert chat.account_type == BUSINESS_OFFICIAL


# ---- the marker must not be forgeable ------------------------------------

def test_participant_cannot_forge_a_verified_badge():
    """
    Genuine account notices are emitted by WhatsApp and carry no sender.
    Anything a participant types has one, so it must not confer trust.
    """
    chat = parse_any(
        "12/03/2024, 10:45 pm - +91 90000 11111: You're messaging with an official "
        "business account. Tap to learn more.\n"
        "12/03/2024, 10:46 pm - +91 90000 11111: Share the OTP to verify")
    assert chat.account_type == UNKNOWN


def test_forged_marker_message_is_still_scored():
    """The forgery attempt must not also excuse the message from analysis."""
    chat = parse_any(
        "12/03/2024, 10:45 pm - +91 90000 11111: You're messaging with an official "
        "business account.\n"
        "12/03/2024, 10:46 pm - +91 90000 11111: Share the OTP to verify")
    assert len(chat.user_messages) == 2


# ---- identity claims vs mentions -----------------------------------------

@pytest.mark.parametrize("text,brand", [
    ("Hi, I am from Amazon support", "amazon"),
    ("This is HDFC customer care calling", "hdfc"),
    ("main SBI se bol raha hun", "sbi"),
    ("We are contacting you on behalf of Paytm", "paytm"),
])
def test_identity_claims_detected(text, brand):
    assert brand in claimed_brands(text, BRANDS)


@pytest.mark.parametrize("text", [
    "my amazon order is late, so annoying",
    "I ordered from Flipkart yesterday",
    "did you get the Netflix subscription?",
    "I paid via Paytm",
])
def test_mentions_are_not_claims(text):
    """Talking about a company is not claiming to be it."""
    assert claimed_brands(text, BRANDS) == []


def test_support_role_and_legitimacy_claims():
    assert claims_support_role("This is the fraud prevention team")
    assert not claims_support_role("hey what time is dinner")
    assert claims_legitimacy("We are an authorised dealer")
    assert not claims_legitimacy("see you tomorrow")


def test_business_features_exposed():
    feats, _ = analyse_message("I am from Amazon support, please share the OTP")
    assert feats["claims_brand_identity"] == 1.0
    assert feats["x_brandclaim_credential"] == 1.0
    clean, _ = analyse_message("see you at 6")
    assert clean["claims_brand_identity"] == 0.0


# ---- conversation-level behaviour ----------------------------------------

HEADER_UNVERIFIED = "12/03/2024, 10:44 pm - You're messaging with a business account.\n"
HEADER_OFFICIAL = "12/03/2024, 10:44 pm - You're messaging with an official business account.\n"


def _analyse(chat_text):
    return analyse_conversation(parse_any(chat_text), DET)


@needs_model
def test_unverified_business_impersonating_a_bank_is_flagged():
    r = _analyse(HEADER_UNVERIFIED +
                 "12/03/2024, 10:45 pm - +91 90000 11111: I am from HDFC customer care\n"
                 "12/03/2024, 10:46 pm - +91 90000 11111: Share the OTP sent to your phone to verify")
    assert r["summary"]["account_type"] == BUSINESS_UNVERIFIED
    assert r["summary"]["risk_score"] > 0.7
    sender = r["senders"][0]
    assert "hdfc" in sender["claimed_brands"]
    assert any("not confirmed this identity" in x for x in sender["reasons"])


@needs_model
def test_legitimate_unverified_business_is_not_penalised():
    """Being an unverified business is not, by itself, evidence of anything."""
    r = _analyse(HEADER_UNVERIFIED +
                 "12/03/2024, 10:45 pm - Bloom Cafe: Hi! This is Riya from Bloom Cafe. "
                 "Your table for 4 is confirmed for 8:00 PM today.\n"
                 "12/03/2024, 10:46 pm - Bloom Cafe: See you soon! Call us to modify.")
    assert r["summary"]["risk_score"] < 0.15, "ordinary small business flagged"


@needs_model
def test_official_business_update_is_quietened():
    r = _analyse(HEADER_OFFICIAL +
                 "12/03/2024, 10:45 pm - Blue Dart: Your shipment is out for delivery today.\n"
                 "12/03/2024, 10:46 pm - Blue Dart: Delivered to the security desk.")
    assert r["summary"]["risk_score"] < 0.15
    assert any("verified by WhatsApp" in x for x in r["senders"][0]["reasons"])


@needs_model
def test_verified_badge_does_not_excuse_a_credential_request():
    """A verified badge is not a licence to ask for an OTP."""
    r = _analyse(HEADER_OFFICIAL +
                 "12/03/2024, 10:45 pm - SomeBrand: Your payment failed. Share the OTP we "
                 "sent to confirm your card details.")
    assert r["summary"]["risk_score"] > 0.5, "verified discount hid a credential request"
    assert any("yet the messages request" in x for x in r["senders"][0]["reasons"])


@needs_model
def test_brand_claim_without_any_business_marker_is_noted():
    r = _analyse("12/03/2024, 10:45 pm - +91 90000 11111: I am from Amazon support, "
                 "confirm your card number\n"
                 "12/03/2024, 10:46 pm - Asha: who is this?")
    assert any("no business-account marker" in x for x in r["senders"][0]["reasons"])


@needs_model
def test_account_type_reported_for_ordinary_chats():
    r = _analyse("12/03/2024, 10:45 pm - Rahul: hey, lunch tomorrow?\n"
                 "12/03/2024, 10:46 pm - Asha: sure!")
    assert r["summary"]["account_type"] == UNKNOWN
    assert r["summary"]["risk_score"] < 0.15
