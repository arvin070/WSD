"""Unicode folding, and the evasion evidence it produces."""

import json
from pathlib import Path

import pytest

from scamshield.core.features import normalized_entropy, rule_score
from scamshield.core.normalizer import normalise, normalise_text

# ---- folding --------------------------------------------------------------

@pytest.mark.parametrize("spoofed,expected", [
    ("pаypal", "paypal"),          # Cyrillic а
    ("аmazon", "amazon"),          # Cyrillic а
    ("ｐａｙｔｍ", "paytm"),           # fullwidth
    ("Café", "Cafe"),              # Latin accent
    ("ѕbi-verify", "sbi-verify"),  # Cyrillic ѕ
])
def test_homoglyphs_fold_to_latin(spoofed, expected):
    assert normalise_text(spoofed) == expected


def test_zero_width_stripped_and_counted():
    n = normalise("Sh​are your OT​P")
    assert n.text == "Share your OTP"
    assert n.zero_width_removed == 2


def test_unicode_digits_fold_to_ascii():
    assert normalise_text("१२३") == "123"          # Devanagari
    assert normalise_text("٤٥٦") == "456"          # Arabic-Indic


def test_devanagari_survives_folding():
    """NFKD strips combining marks; Indic vowels ARE combining marks."""
    hindi = "आपका खाता बंद हो जाएगा। तुरंत KYC अपडेट करें"
    out = normalise_text(hindi)
    assert "अपडेट" in out and "करें" in out and "जाएगा" in out


def test_clean_text_is_untouched():
    for t in ["hey are we still on for lunch?", "Your OTP is 481920", "beta ghar kab aaoge"]:
        n = normalise(t)
        assert n.text == t
        assert not n.was_obfuscated


def test_mixed_script_word_detected():
    assert normalise("pаypal").mixed_script_words == 1
    assert normalise("paypal").mixed_script_words == 0


def test_normalise_handles_empty_and_none():
    assert normalise_text("") == ""
    assert normalise_text(None) == ""


# ---- entropy --------------------------------------------------------------

def test_entropy_separates_random_from_prose():
    random_slug = normalized_entropy("bit.ly/aX9k2Qz")
    prose = normalized_entropy("hey are we still on for lunch tomorrow at one")
    assert random_slug > prose


def test_entropy_bounds():
    assert normalized_entropy("") == 0.0
    assert normalized_entropy("a") == 0.0
    assert normalized_entropy("aaaaaaaa") == 0.0     # no information
    assert 0.0 <= normalized_entropy("mixed text 123") <= 1.0


# ---- evasion raises risk rather than lowering it -------------------------

def test_spoofed_scam_scores_at_least_as_high_as_clean():
    clean = "Please share the OTP sent to your phone to verify your account"
    spoofed = "Please shаre the OTP sent to your phone to vеrify your аccount"
    assert rule_score(spoofed)[0] >= rule_score(clean)[0]


def test_homoglyph_signal_raised():
    _, signals = rule_score("Verify at pаypal-secure.com now")
    assert any(s.code == "homoglyph_spoof" for s in signals)


def test_zero_width_signal_still_raised():
    """Folding must not silently swallow the evasion it undoes."""
    _, signals = rule_score("Sh​are your OT​P now please to verify account")
    assert any(s.code == "hidden_chars" for s in signals)


def test_adversarial_suite_present():
    lines = Path("data/adversarial.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    assert {r["attack"] for r in rows} >= {"homoglyph", "zero_width", "fullwidth", "combined"}
