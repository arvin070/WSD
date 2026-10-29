"""
Training corpus builder.

Templates live in `resources/templates.yaml`; this module only fills their
slots and applies realistic noise. Two design choices matter:

1. **Hard negatives.** A large share of the ham class is legitimate traffic
   that superficially looks scammy — real bank alerts, real delivery updates,
   a friend genuinely asking for a UPI transfer. Without these the model
   learns "mentions money => scam" and is useless in production.

2. **Template grouping.** Every row carries a `template_id` so training can
   split on it; otherwise near-duplicate augmentations leak across folds and
   inflate the reported scores.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

from ..config import get_settings
from ..logging_setup import get_logger
from ..resources import load_templates

log = get_logger(__name__)

TYPO_MAP = {"a": "@", "o": "0", "i": "1", "e": "3", "s": "$"}
ZERO_WIDTH = "​"
#: Latin -> confusable, the inverse of the normaliser's fold map. Scam rows
#: get script-spoofed variants so the classifier has something to learn from;
#: without these the homoglyph features are dead weight at training time.
HOMOGLYPH_SPOOF = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440",
                   "c": "\u0441", "y": "\u0443", "x": "\u0445", "i": "\u0456",
                   "s": "\u0455", "j": "\u0458"}


def _amount(rng: random.Random, small: bool = False) -> str:
    if small:
        return rng.choice(["99", "199", "250", "499", "550", "750", "999", "1,250", "1,500", "2,000"])
    style = rng.random()
    if style < 0.35:
        return f"{rng.randint(1, 50)},{rng.randint(10, 99)},000"
    if style < 0.60:
        return f"{rng.randint(1, 25)} Lakh"
    if style < 0.80:
        return f"{rng.randint(5, 99)},000"
    return f"{rng.randint(1, 9)} Crore"


def _url(rng: random.Random, slots: dict, brandish: str | None = None) -> str:
    if rng.random() < 0.3:
        tail = "".join(rng.choices("abcdefghijkmnpqrstuvwxyz0123456789", k=rng.randint(6, 8)))
        return f"https://{rng.choice(slots['shorteners'])}/{tail}"
    slug = rng.choice(["verify", "update-kyc", "secure-login", "claim", "reward", "offer",
                       "account", "confirm", "unlock", "refund", "portal", "care"])
    if brandish:
        base = re.sub(r"[^a-z]", "", brandish.lower())[:10]
        host = rng.choice([f"{base}-{slug}", f"{slug}-{base}", f"{base}{rng.randint(10, 99)}",
                           f"{base}-india", f"secure-{base}"])
    else:
        host = rng.choice(["quickreward", "winner-zone", "fastcash", "mega-offer", "task-hub",
                           "profit-pro", "gov-refund", "prime-deal"])
    return f"http://{host}.{rng.choice(slots['scam_tlds'])}/{slug}"


def _upi(rng: random.Random, slots: dict) -> str:
    name = rng.choice(["winner", "claim", "support", "agent", "helpdesk", "payments",
                       "reward", "verify", "office"]) + str(rng.randint(10, 999))
    return f"{name}@{rng.choice(slots['upi_suffixes'])}"


def _phone(rng: random.Random) -> str:
    return f"+91{rng.randint(70, 99)}{rng.randint(10000000, 99999999)}"


def fill_slots(rng: random.Random, tpl: str, slots: dict) -> str:
    brand = rng.choice(slots["ecommerce"] + slots["wallets"] + slots["streaming"])
    stream = rng.choice(slots["streaming"])
    return (
        tpl.replace("{bank}", rng.choice(slots["banks"]))
        .replace("{wallet}", rng.choice(slots["wallets"]))
        .replace("{brandlower}", re.sub(r"[^a-z]", "", brand.lower()))
        .replace("{brand}", brand)
        .replace("{streamlower}", re.sub(r"[^a-z]", "", stream.lower()))
        .replace("{stream}", stream)
        .replace("{courier}", rng.choice(slots["couriers"]))
        .replace("{telco}", rng.choice(slots["telcos"]))
        .replace("{name}", rng.choice(slots["first_names"]))
        .replace("{amt}", _amount(rng))
        .replace("{small}", _amount(rng, small=True))
        .replace("{url}", _url(rng, slots, brand if rng.random() < 0.5 else None))
        .replace("{upi}", _upi(rng, slots))
        .replace("{phone}", _phone(rng))
        .replace("{num}", str(rng.randint(1000, 9999)))
    )


def augment(rng: random.Random, text: str, scam: bool) -> str:
    """Realistic noise, so the model cannot lock onto exact template strings."""
    r = rng.random()
    if r < 0.12:
        text = text.upper()
    elif r < 0.22:
        text = text.lower()

    if scam and rng.random() < 0.18:  # leetspeak filter evasion
        idx = [i for i, c in enumerate(text.lower()) if c in TYPO_MAP]
        for i in rng.sample(idx, min(len(idx), rng.randint(1, 3))):
            text = text[:i] + TYPO_MAP[text[i].lower()] + text[i + 1:]
    if scam and rng.random() < 0.14:  # script spoofing (Cyrillic look-alikes)
        idx = [i for i, c in enumerate(text) if c.lower() in HOMOGLYPH_SPOOF]
        for i in rng.sample(idx, min(len(idx), rng.randint(1, 4))):
            text = text[:i] + HOMOGLYPH_SPOOF[text[i].lower()] + text[i + 1:]
    if scam and rng.random() < 0.15:  # zero-width evasion
        pos = rng.randint(1, max(len(text) - 1, 1))
        text = text[:pos] + ZERO_WIDTH + text[pos:]

    if rng.random() < 0.25:
        text = rng.choice(["🎉", "✅", "🔥", "⚠️", "💰", "🙏", "❗", ""]) + " " + text
    if rng.random() < 0.18:
        text += rng.choice([" Thanks.", " Regards.", " Pls reply.", " ok?", " 🙏", "!!", "..."])
    if rng.random() < 0.12 and len(text.split()) > 8:  # dropped word, sloppy typing
        w = text.split()
        del w[rng.randrange(len(w))]
        text = " ".join(w)
    return text.strip()


def _generate(rng, pool, target, category, prefix, scam, slots, seen, rows):
    made = attempts = 0
    while made < target and attempts < target * 25:
        attempts += 1
        idx = rng.randrange(len(pool))
        text = augment(rng, fill_slots(rng, pool[idx], slots), scam=scam)
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"text": text, "label": int(scam), "category": category,
                     "template_id": f"{prefix}:{category}:{idx}"})
        made += 1


def build_corpus(n_per_scam: int = 190, ham_multiplier: int = 19, seed: int = 42) -> list[dict]:
    """Generate the labelled corpus as a list of {text, label, category, template_id}."""
    tpl = load_templates()
    slots = tpl["slots"]
    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()

    # Per-category weights let conversational playbooks outnumber transactional
    # ones, which matters once large SMS corpora are folded in: without this
    # the grooming scams WhatsApp is actually used for get outvoted.
    weights = tpl.get("scam_weights", {})
    for category, templates in tpl["scam"].items():
        target = int(n_per_scam * float(weights.get(category, 1.0)))
        _generate(rng, templates, target, category, "scam", True, slots, seen, rows)

    for category, body in tpl["ham"].items():
        messages = body["messages"]
        target = int(len(messages) * ham_multiplier * float(body.get("weight", 1.0)))
        _generate(rng, messages, target, category, "ham", False, slots, seen, rows)

    rows.extend(load_external())
    rng.shuffle(rows)
    return rows


def load_external() -> list[dict]:
    """
    Fold in real-world CSVs dropped into `data/external/`.

    Expected columns: `text` and `label` (1 = scam/spam, 0 = ham). Anything
    else is ignored, so a Kaggle SMS-spam CSV can be renamed and used as-is.
    """
    ext_dir = get_settings().data_dir / "external"
    rows: list[dict] = []
    if not ext_dir.exists():
        return rows
    for path in sorted(ext_dir.glob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as fh:
                for rec in csv.DictReader(fh):
                    keys = {k.lower().strip(): v for k, v in rec.items() if k}
                    text = keys.get("text") or keys.get("message") or keys.get("v2") or ""
                    lab = str(keys.get("label") or keys.get("class") or keys.get("v1") or "").strip().lower()
                    if not text.strip():
                        continue
                    if lab in {"1", "spam", "scam", "fraud", "smishing"}:
                        label = 1
                    elif lab in {"0", "ham", "legit", "normal"}:
                        label = 0
                    else:
                        continue
                    rows.append({"text": text.strip(), "label": label,
                                 "category": f"external:{path.stem}",
                                 "template_id": f"external:{path.stem}:{len(rows)}"})
        except OSError as exc:
            log.warning("skipped %s: %s", path.name, exc)
    if rows:
        log.info("folded in %d external rows from %s", len(rows), ext_dir)
    return rows


def write_corpus(path: Path | None = None) -> Path:
    settings = get_settings()
    path = path or settings.data_dir / "corpus.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_corpus()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "label", "category", "template_id"])
        writer.writeheader()
        writer.writerows(rows)
    return path
