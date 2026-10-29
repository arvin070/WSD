"""
Direction of the ask.

A scam instructs the reader; a worried user asks about the same thing using
the same words. Getting this wrong in either direction is costly: flagging the
victim's questions is noise, and discounting on a question mark alone would
hand attackers a one-character bypass.
"""

import pytest

from scamshield.core.features import analyse_message, rule_score

# The attacker telling the reader to act - must stay flagged.
DIRECTIVE = [
    "Please share the OTP sent to your phone",
    "Can you share the OTP that was just sent?",
    "Could you confirm your card number and CVV?",
    "Will you verify your account at this link http://sbi-kyc.xyz/l ?",
]

# The reader asking about it - should be discounted.
INQUIRY = [
    "Should I share my OTP with them?",
    "why do you need my OTP?",
    "is this a scam?",
    "did someone ask you for your OTP too?",
    "is it safe to pay this processing fee?",
    "how do I know this is genuine?",
]


@pytest.mark.parametrize("text", DIRECTIVE)
def test_directive_questions_are_not_discounted(text):
    """A question mark must not be a one-character bypass."""
    assert rule_score(text)[0] > 0.5, f"bypass via interrogative form: {text}"


@pytest.mark.parametrize("text", INQUIRY)
def test_reader_inquiries_are_discounted(text):
    assert rule_score(text)[0] < 0.35, f"victim's own question flagged: {text}"


def test_directive_question_flag_set():
    feats, _ = analyse_message("Can you share the OTP?")
    assert feats["is_question"] and feats["is_directive_question"]
    assert not feats["x_inquiry_not_directive"]


def test_self_inquiry_flag_set():
    feats, _ = analyse_message("Should I share my OTP with them?")
    assert feats["is_self_inquiry"] and feats["x_inquiry_not_directive"]


def test_statement_is_not_a_question():
    feats, _ = analyse_message("Share the OTP now.")
    assert not feats["is_question"]
