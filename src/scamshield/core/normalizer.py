"""
Unicode normalisation and homoglyph folding.

Runs first in the text pipeline. Two jobs, and the second matters as much as
the first:

1. **Fold** script-spoofed text back to a canonical form so that lexicon
   matching and TF-IDF see `paypal` whether the attacker wrote `pаypal`
   (Cyrillic а), `ｐａｙｐａｌ` (fullwidth) or `p​aypal` (zero-width).

2. **Report** what had to be folded. An ordinary message needs no folding at
   all, so Cyrillic letters inside a Latin word, or an invisible character in
   the middle of "share", are themselves strong evidence of evasion. Throwing
   that away to get clean text would lose signal, so `normalise()` returns
   both the text and the edit counts.

Deliberately dependency-free beyond the standard library.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Invisible characters used to break up keywords so filters miss them.
ZERO_WIDTH_CHARS = (
    "​‌‍‎‏⁠⁡⁢⁣⁤"
    "﻿­᠎‪‫‬‭‮"
)
ZERO_WIDTH_RE = re.compile(f"[{ZERO_WIDTH_CHARS}]")

#: Confusables that NFKD does not fold, because they are genuinely distinct
#: characters that merely *look* Latin. Source: the visually-confusable pairs
#: that actually appear in phishing domains and message bodies.
HOMOGLYPHS: dict[str, str] = {
    # Cyrillic -> Latin
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "і": "i", "ј": "j", "һ": "h",
    "ґ": "r", "ԁ": "d", "ԛ": "q", "ѕ": "s", "ӏ": "l",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "І": "I", "Ѕ": "S", "Ј": "J",
    # Greek -> Latin
    "α": "a", "ε": "e", "ο": "o", "ρ": "p", "υ": "u",
    "ν": "v", "τ": "t", "κ": "k", "ι": "i", "χ": "x",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Armenian / Georgian / Cherokee lookalikes seen in domain spoofing
    "օ": "o", "հ": "h", "ո": "n", "Ꭰ": "D", "Ꮐ": "G",
    # Latin letters with strokes used to dodge exact-string filters
    "ł": "l", "ı": "i", "ɩ": "i", "ǀ": "l",
}

#: Currency and punctuation variants that carry the same meaning.
PUNCT_FOLD = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "⁄": "/",
    " ": " ", " ": " ", " ": " ", "　": " ",
    "．": ".", "。": ".", "／": "/", "：": ":",
}

LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_GREEK_RE = re.compile(r"[Ͱ-ϿЀ-ӿԀ-ԯ]")
WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Normalised:
    """Folded text plus a record of the evasion that had to be undone."""

    text: str
    original: str
    zero_width_removed: int
    homoglyphs_mapped: int
    digits_folded: int
    mixed_script_words: int

    @property
    def was_obfuscated(self) -> bool:
        return bool(self.zero_width_removed or self.homoglyphs_mapped or self.mixed_script_words)


def _count_mixed_script_words(text: str) -> int:
    """
    Words mixing Latin with Cyrillic/Greek letters.

    Legitimate text does not do this. `pаypal` (one Cyrillic а) does, and it is
    close to a definitive spoofing indicator.
    """
    return sum(
        1 for w in WORD_RE.findall(text)
        if LATIN_RE.search(w) and CYRILLIC_GREEK_RE.search(w)
    )


def _fold_marks(text: str) -> str:
    """
    Drop combining marks left by NFKD — but only after Latin base letters.

    Indic scripts build vowels from combining marks (matras), so blanket-
    stripping category Mn would turn "अपडेट करें" into consonant soup and
    destroy the Hindi/Devanagari half of the corpus.
    """
    out: list[str] = []
    for ch in text:
        if unicodedata.combining(ch):
            prev = out[-1] if out else ""
            if prev and ord(prev) < 0x0250:  # Latin / ASCII base only
                continue
        out.append(ch)
    return "".join(out)


def normalise(text: str) -> Normalised:
    """Fold `text` to canonical form and report what needed folding."""
    original = text or ""

    stripped = ZERO_WIDTH_RE.sub("", original)
    zero_width = len(original) - len(stripped)

    mixed = _count_mixed_script_words(stripped)

    mapped_chars: list[str] = []
    homoglyphs = 0
    digits = 0
    for ch in stripped:
        if ch in HOMOGLYPHS:
            mapped_chars.append(HOMOGLYPHS[ch])
            homoglyphs += 1
            continue
        if ch in PUNCT_FOLD:
            mapped_chars.append(PUNCT_FOLD[ch])
            continue
        # Any Unicode decimal digit (Devanagari ०-९, Arabic-Indic, fullwidth)
        # folds to its ASCII value. NFKD does not do this for Devanagari.
        if not ch.isascii() and ch.isdigit():
            try:
                mapped_chars.append(str(unicodedata.digit(ch)))
                digits += 1
                continue
            except (TypeError, ValueError):
                pass
        mapped_chars.append(ch)

    folded = unicodedata.normalize("NFKD", "".join(mapped_chars))
    folded = _fold_marks(folded)
    # Recompose so downstream comparisons see a single canonical form.
    folded = unicodedata.normalize("NFKC", folded)

    return Normalised(
        text=folded,
        original=original,
        zero_width_removed=zero_width,
        homoglyphs_mapped=homoglyphs,
        digits_folded=digits,
        mixed_script_words=mixed,
    )


def normalise_text(text: str) -> str:
    """Convenience wrapper when only the folded text is needed."""
    return normalise(text).text
