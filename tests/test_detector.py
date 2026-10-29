"""End-to-end model behaviour and conversation reasoning."""

import pytest

from scamshield.analysis.conversation import analyse_conversation
from scamshield.core.chat_parser import parse_any
from scamshield.ml.detector import ModelNotTrained, get_detector

try:
    DET = get_detector()
except ModelNotTrained:  # pragma: no cover - model must be trained first
    DET = None

pytestmark = pytest.mark.skipif(DET is None, reason="run `python -m ml.train` first")

CLEAR_SCAMS = [
    "Dear Customer, your SBI KYC has expired. Account will be blocked today. Update: http://sbi-kyc.xyz/v",
    "Congratulations! You won Rs 25,00,000 in the lucky draw. Pay 5,500 processing fee to claim your prize.",
    "Please share the OTP that was just sent to your phone so I can verify your account",
    "Work from home! Earn Rs 2000 daily, no investment needed. Join now on telegram",
]
CLEAR_HAM = [
    "hey are we still on for lunch tomorrow? I'll book a table",
    "Your OTP for HDFC net banking is 883021. Valid 10 minutes. Never share this with anyone.",
    "someone called claiming to be from the bank asking for my otp, I blocked them",
    "Can you send me 500 on gpay? I forgot my wallet, will return it friday",
    "I'll send the report by EOD today, the client meeting moved to 3pm",
]


@pytest.mark.parametrize("text", CLEAR_SCAMS)
def test_clear_scams_flagged(text):
    r = DET.score(text)
    assert r["is_scam"], f"missed: {text} (scored {r['risk_score']})"


@pytest.mark.parametrize("text", CLEAR_HAM)
def test_clear_ham_not_flagged(text):
    r = DET.score(text)
    assert not r["is_scam"], f"false alarm: {text} (scored {r['risk_score']})"


@pytest.mark.parametrize("text", ["Hi", "ok", "hello", "yes", "thanks"])
def test_short_messages_are_not_flagged(text):
    """A bare greeting carries no evidence and must never raise an alarm."""
    r = DET.score(text)
    assert r["low_information"] and r["risk_score"] <= 0.25


def test_verdict_bands_agree_with_the_decision():
    """A message above the threshold must never read as 'safe'."""
    for text in CLEAR_SCAMS + CLEAR_HAM:
        r = DET.score(text)
        if r["is_scam"]:
            assert r["verdict"] in {"likely_scam", "dangerous"}
        else:
            assert r["verdict"] in {"safe", "suspicious"}


def test_scores_are_bounded():
    for text in CLEAR_SCAMS + CLEAR_HAM:
        assert 0.0 <= DET.score(text)["risk_score"] <= 1.0


def test_flagged_messages_carry_an_explanation():
    r = DET.score(CLEAR_SCAMS[0])
    assert r["signals"] and r["category_label"]


def test_batch_matches_single():
    batch = DET.score_batch(CLEAR_SCAMS)
    assert [b["risk_score"] for b in batch] == [DET.score(t)["risk_score"] for t in CLEAR_SCAMS]


# ---- conversation level ---------------------------------------------------

CONV = """02/03/2024, 8:14 pm - +91 91234 55678: Hi
02/03/2024, 8:15 pm - +91 91234 55678: Hello, is this Meera? I got your number from the alumni group
02/03/2024, 8:31 pm - Meera: Hi, sorry who is this?
03/03/2024, 9:02 am - +91 91234 55678: Good morning! How is your day going?
04/03/2024, 8:12 am - Meera: it's going fine, busy with work
05/03/2024, 9:11 pm - +91 91234 55678: I have been following an investment desk, last month I withdrew 3,20,000 profit
06/03/2024, 7:16 pm - +91 91234 55678: The platform is regulated and your capital is insured, so there is genuinely no downside
08/03/2024, 10:12 am - +91 91234 55678: Transfer to this UPI and the desk credits your wallet: wealthdesk221@okaxis
10/03/2024, 2:40 pm - +91 91234 55678: Also the desk needs to verify you. Please share the OTP just sent to your phone
10/03/2024, 6:15 pm - Meera: why do you need my OTP?"""


def test_conversation_flags_the_scammer_not_the_victim():
    r = analyse_conversation(parse_any(CONV), DET)
    by = {s["sender"]: s for s in r["senders"]}
    assert by["+91 91234 55678"]["risk_score"] > 0.7
    assert by["Meera"]["risk_score"] < 0.3, "the victim must not be flagged"


def test_escalation_detected_across_the_thread():
    r = analyse_conversation(parse_any(CONV), DET)
    scammer = [s for s in r["senders"] if s["sender"].startswith("+91")][0]
    assert scammer["escalation_detected"]


def test_unknown_number_noted():
    r = analyse_conversation(parse_any(CONV), DET)
    scammer = [s for s in r["senders"] if s["sender"].startswith("+91")][0]
    assert scammer["is_unknown_number"]


def test_empty_conversation_is_safe():
    r = analyse_conversation(parse_any("   "), DET)
    assert r["summary"]["verdict"] == "safe"
    assert r["summary"]["message_count"] == 0
