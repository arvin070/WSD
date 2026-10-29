"""Regex primitives shared by the feature extractor."""

from __future__ import annotations

import re

URL_RE = re.compile(
    r"""(?ix)
    \b(
        (?:https?://|www\.)[^\s<>"']+
        |
        (?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}
        (?:/[^\s<>"']*)?
    )\b"""
)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,13}\d(?!\d)")
UPI_RE = re.compile(r"\b[\w.\-]{2,256}@(?:ok\w+|paytm|ybl|axl|upi|apl|ibl|hdfcbank|sbi|icici)\b", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MONEY_RE = re.compile(
    r"(?ix)(?:(?:rs\.?|inr|₹|\$|usd|eur|£|gbp)\s?[\d,]+(?:\.\d+)?"
    r"(?:\s?(?:k|lakh|lakhs|lac|crore|cr|million|mn|thousand))?"
    r"|[\d,]+(?:\.\d+)?\s?(?:rupees|rs\.?|inr|dollars?|usd|lakh|lakhs|lac|crore|cr))"
)
ACCOUNT_RE = re.compile(r"\b\d{9,18}\b")
IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF" "\U00002190-\U000021FF" "]",
    flags=re.UNICODE,
)
ZERO_WIDTH_RE = re.compile(r"[​-‏‪-‮⁠-⁯﻿]")
REPEATED_PUNCT_RE = re.compile(r"([!?])\1{1,}")

#: An explicit request aimed at the reader. Its presence cancels the
#: "someone is just talking about a scam" discount.
ASK_RE = re.compile(
    r"(?i)\b(?:share|send|forward|tell|give|provide|confirm|verify|read out|enter|submit|"
    r"transfer|deposit|pay|install|download|click|call|register|apply)\b"
    r"[^.!?]{0,40}\b(?:otp|code|pin|cvv|password|card|account|ifsc|aadhaar|upi|"
    r"amount|fee|money|rs|link|app|apk|now|immediately|below|here)\b"
)


# ---------------------------------------------------------------------------
# Directionality: who is being asked to act?
#
# A scam *instructs* the reader ("share the OTP"). A worried user *asks about*
# the same thing ("should I share my OTP?"). Both contain the identical
# keywords, so interrogative form alone is not enough - a scammer writing
# "Can you share the OTP?" is still a scam. What separates them is direction:
# does the question ask the reader to do something, or ask about something?
# ---------------------------------------------------------------------------

#: Any interrogative: a question mark, or an interrogative opener.
QUESTION_RE = re.compile(
    r"(?i)(?:\?\s*$|\?)|^\s*(?:is|are|was|were|do|does|did|can|could|should|would|will|"
    r"what|why|how|who|when|where|which|whose|kya|kaun|kaise|kyun)\b"
)

#: The reader is asking *about* something, or pushing back. Discountable.
SELF_INQUIRY_RE = re.compile(
    r"(?i)\b(?:"
    r"should i|can i|do i|must i|why (?:do|would|should) (?:you|they|i)|"
    r"is (?:this|that|it) (?:a |real|genuine|safe|legit|fake|true)|"
    r"is (?:this|that|it) (?:a )?scam|"
    r"(?:did|has) (?:someone|anyone|somebody)|"
    r"who (?:is|are) (?:this|you)|who gave you|"
    r"what (?:should|do) i|how do i know|"
    r"is it safe|are you sure|really\?|"
    r"do i (?:need|have) to|"
    r"kya (?:yeh|ye) (?:sach|asli|fake)"
    r")\b"
)

#: Second-person directive framed as a question - still an instruction.
DIRECTIVE_QUESTION_RE = re.compile(
    r"(?i)\b(?:can|could|will|would|are)\s+(?:you|u|yu)\b[^.!?]{0,50}?\b(?:"
    r"share|send|forward|transfer|pay|deposit|confirm|verify|provide|give|tell|"
    r"install|download|click|register|complete|update"
    r")\b"
)
