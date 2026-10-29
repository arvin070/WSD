"""
Course-selling, extortion and trusted-contact-link playbooks.

Each has a legitimate twin that shares most of its vocabulary — a real course
advert, a genuine security notice, a friend sharing a link — so every
detection here is paired with the false positive it must not cause.
"""

import pytest

from scamshield.core.features import analyse_message, rule_score
from scamshield.ml.detector import ModelNotTrained, get_detector

try:
    DET = get_detector()
except ModelNotTrained:  # pragma: no cover
    DET = None

needs_model = pytest.mark.skipif(DET is None, reason="run scamshield-train first")

COURSE_SCAM = [
    "My mentorship batch closes tonight. 3 students crossed 1 lakh last month. Lifetime access 4999",
    "Free demo class on stock trading - learn to earn 5000 daily, no degree needed",
    "Fees increase to 25000 next week. Enroll now and get the recorded masterclass free",
    "I quit my 9 to 5 and now teach the roadmap. Only 5 seats left, pay to mentor12@okaxis",
]
COURSE_HAM = [
    "Your Coursera course 'Machine Learning' starts Monday. First assignment due on the 14th.",
    "NPTEL registration closes on the 15th. Certificates are optional.",
    "The photography workshop has 4 spots left, Rs 1,200 for the full day.",
    "Batch timings changed to 6 PM from next week. - Sunrise Tutorials",
]
EXTORTION = [
    "I recorded you through your webcam. Pay 50000 in bitcoin within 48 hours or I send it to your contacts",
    "All your files have been encrypted. Send 0.5 BTC to this wallet for the decryption key",
    "We have your son. Do not call the police. Arrange 5 lakh cash and wait for instructions",
    "I have your private photos. Do not tell anyone or I post them online. Pay to this wallet",
]
SECURITY_HAM = [
    "Your password was changed successfully. If this was not you, reset it from the app.",
    "We detected a new sign-in from Chrome on Windows. Review it in your settings.",
    "Antivirus scan finished. No threats found on your device.",
    "Two-factor authentication is now enabled on your account.",
]
VOUCHED_SCAM = [
    "Hey its me, is this you in this video? http://vidz-share.top/v?id=8821",
    "bro check this out, i made 40k with this app http://cash-app-free.buzz/get",
    "Please vote for my niece in this contest, takes 2 seconds http://contest-vote.top/p",
]
VOUCHED_HAM = [
    "here's the recipe I mentioned https://www.example-recipes.com/paneer-tikka",
    "check this out, the new season trailer just dropped https://www.youtube.com/watch",
    "the article is here if you want to read it https://www.thehindu.com/opinion",
    "look at this cat video 😂 it made my day",
]


# ---- rule engine ---------------------------------------------------------

@pytest.mark.parametrize("text", COURSE_SCAM + EXTORTION + VOUCHED_SCAM)
def test_new_playbooks_raise_signals(text):
    score, signals = rule_score(text)
    assert signals, f"no signal raised for: {text}"
    assert score > 0.4


@pytest.mark.parametrize("text", COURSE_HAM + SECURITY_HAM + VOUCHED_HAM)
def test_legitimate_twins_stay_quiet(text):
    assert rule_score(text)[0] < 0.35, f"false positive: {text}"


def test_extortion_needs_a_payment_demand():
    """A threat alone is not ransom; the pairing is what defines it."""
    _, with_demand = rule_score(
        "I recorded you through your webcam. Pay 50000 in bitcoin or I send it to your contacts")
    assert any(s.code == "ransom_payment" for s in with_demand)


def test_vouching_alone_is_not_enough():
    """Friends share links constantly; only a bad link makes it suspicious."""
    clean_feats, clean_signals = analyse_message(
        "check this out https://www.youtube.com/watch?v=abc")
    assert clean_feats["n_vouched_link"] > 0
    assert not any(s.code == "vouched_risky_link" for s in clean_signals)

    _, bad_signals = analyse_message("check this out http://cash-app-free.buzz/get")
    assert any(s.code == "vouched_risky_link" for s in bad_signals)


def test_course_upsell_requires_a_fee():
    _, signals = analyse_message("Free webinar on trading this Sunday, register now")
    assert not any(s.code == "course_upsell" for s in signals)
    _, paid = analyse_message("Join my mentorship batch, course fees Rs 4999, pay to me@okaxis")
    assert any(s.code == "course_upsell" for s in paid)


# ---- end to end ----------------------------------------------------------

@needs_model
@pytest.mark.parametrize("text", COURSE_SCAM + EXTORTION + VOUCHED_SCAM)
def test_model_flags_new_playbooks(text):
    result = DET.score(text)
    assert result["is_scam"], f"missed ({result['risk_score']}): {text}"


@needs_model
@pytest.mark.parametrize("text", COURSE_HAM + SECURITY_HAM + VOUCHED_HAM)
def test_model_does_not_flag_legitimate_twins(text):
    result = DET.score(text)
    assert not result["is_scam"], f"false alarm ({result['risk_score']}): {text}"


@needs_model
@pytest.mark.parametrize("text,expected", [
    (COURSE_SCAM[0], "course_mentorship_scam"),
    (EXTORTION[0], "extortion_ransom"),
    (VOUCHED_SCAM[0], "trusted_contact_link"),
])
def test_playbooks_are_categorised(text, expected):
    """Before these categories existed, extortion was labelled 'Romance'."""
    assert DET.score(text)["category"] == expected
