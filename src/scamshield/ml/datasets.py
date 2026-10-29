"""
Fetch third-party corpora into `data/external/`.

    scamshield-fetch-data

Provenance matters more than volume here, so each source is recorded with what
it actually is. Two are used and one deliberately is not:

* **CloveAI/india-spam-sms** (MIT) - 20k Indian SMS, on-domain scams: KYC
  phishing, fake refunds, prize fraud. It is *generated*, not intercepted -
  messages repeat with different digits and the ham carries slot-fill
  artifacts - so it adds vocabulary and brand diversity, not ground truth.

* **ucirvine/sms_spam** (public, UCI) - 5.5k genuinely real SMS. Its **ham** is
  the most valuable part: authentic human chatter, which a template generator
  cannot imitate. Its spam is 2005-era UK premium-rate marketing, which is
  spam but mostly not fraud, so it is imported under a separate tag and its
  effect is measured rather than assumed.

* **alusci/sms-otp-spam-dataset** - rejected. Its `label` column is OTP
  validity and its `status='spam'` is a *delivery* outcome, not a content
  label. Importing it would add noise wearing the costume of signal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings
from ..logging_setup import configure_logging, get_logger

log = get_logger(__name__)

HF = "https://huggingface.co/datasets"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    text_column: str
    label_column: str
    fmt: str
    licence: str
    note: str
    keep: str = "both"          # both | ham | spam
    #: Fraction reserved for evaluation, split by template signature.
    holdout: float = 0.0

    @property
    def filename(self) -> str:
        return f"{self.name}.csv"


SOURCES = [
    Source("india_sms", f"{HF}/CloveAI/india-spam-sms/resolve/main/data/train-00000-of-00001.parquet",
           "text", "label", "parquet", "MIT",
           "Indian SMS, on-domain scam vocabulary (generated, not intercepted)",
           holdout=0.2),
    Source("uci_sms_ham", f"{HF}/ucirvine/sms_spam/resolve/main/plain_text/train-00000-of-00001.parquet",
           "sms", "label", "parquet", "UCI / public",
           "Genuine human SMS - ham only; its spam is 2005 UK premium-rate marketing",
           keep="ham"),
    Source("uci_sms_spam", f"{HF}/ucirvine/sms_spam/resolve/main/plain_text/train-00000-of-00001.parquet",
           "sms", "label", "parquet", "UCI / public",
           "Genuine SMS spam, off-domain (premium-rate marketing rather than fraud)",
           keep="spam"),
]


_DIGITS = re.compile(r"\d+")
_URL = re.compile(r"(?i)\bhttps?://\S+|\bwww\.\S+")


def template_signature(text: str) -> str:
    """
    Collapse a message to the template it was generated from.

    These corpora are slot-filled, so "won iPhone 15! ...gift530.in" and
    "...gift139.in" are the same template. A random split would put both sides
    of that pair across the train/test boundary and inflate the score, exactly
    as an ungrouped split does on our own generator.
    """
    norm = _URL.sub(" URL ", text.lower())
    norm = _DIGITS.sub("#", norm)
    norm = re.sub(r"[^a-z#\s]", " ", norm)
    norm = " ".join(norm.split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def _split_by_template(rows: list[tuple[str, int]], fraction: float,
                       seed: int = 17) -> tuple[list, list]:
    """Assign whole template groups to train or holdout, never split one."""
    import random

    groups: dict[str, list[tuple[str, int]]] = {}
    for text, label in rows:
        groups.setdefault(template_signature(text), []).append((text, label))
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    cutoff = int(len(keys) * fraction)
    holdout_keys = set(keys[:cutoff])
    train, holdout = [], []
    for key, items in groups.items():
        (holdout if key in holdout_keys else train).extend(items)
    return train, holdout


def _download(url: str) -> bytes:
    log.info("fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "scamshield/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed https hosts
        return resp.read()


def _rows(source: Source, blob: bytes) -> list[tuple[str, int]]:
    if source.fmt == "parquet":
        import pandas as pd

        df = pd.read_parquet(io.BytesIO(blob))
    else:
        import pandas as pd

        df = pd.read_csv(io.BytesIO(blob))

    if source.text_column not in df.columns or source.label_column not in df.columns:
        raise ValueError(f"{source.name}: expected columns "
                         f"{source.text_column}/{source.label_column}, got {list(df.columns)}")

    out: list[tuple[str, int]] = []
    for text, label in zip(df[source.text_column], df[source.label_column], strict=True):
        text = str(text or "").strip()
        if not text:
            continue
        try:
            value = int(label)
        except (TypeError, ValueError):
            value = 1 if str(label).strip().lower() in {"spam", "scam", "1"} else 0
        if source.keep == "ham" and value != 0:
            continue
        if source.keep == "spam" and value != 1:
            continue
        out.append((text, value))
    return out


def fetch(out_dir: Path | None = None, sources: list[Source] | None = None) -> dict[str, int]:
    cfg = get_settings()
    out_dir = out_dir or (cfg.data_dir / "external")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for source in sources or SOURCES:
        try:
            rows = _rows(source, _download(source.url))
        except Exception as exc:  # noqa: BLE001 - one bad source must not stop the rest
            log.error("skipped %s: %s", source.name, exc)
            continue
        # Deduplicate within the source; these corpora repeat templates.
        seen: set[str] = set()
        unique = []
        for text, label in rows:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append((text, label))

        train_rows, holdout_rows = (
            _split_by_template(unique, source.holdout) if source.holdout else (unique, []))

        path = out_dir / source.filename
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["text", "label"])
            writer.writerows(train_rows)
        counts[source.name] = len(train_rows)
        spam = sum(v for _, v in train_rows)
        log.info("%-14s %6d rows (%d scam / %d ham)  [%s]",
                 source.name, len(train_rows), spam, len(train_rows) - spam, source.licence)

        if holdout_rows:
            # Kept outside data/external/ so the training loader cannot see it.
            hold_path = cfg.data_dir / "external_holdout.jsonl"
            with hold_path.open("w", encoding="utf-8") as fh:
                for text, label in holdout_rows:
                    fh.write(json.dumps({"text": text, "label": label,
                                         "source": source.name}, ensure_ascii=False) + "\n")
            hspam = sum(v for _, v in holdout_rows)
            log.info("%-14s %6d rows held out for evaluation (%d scam / %d ham) -> %s",
                     "", len(holdout_rows), hspam, len(holdout_rows) - hspam, hold_path.name)

    (out_dir / "PROVENANCE.md").write_text(
        "# External corpora\n\nFetched by `scamshield-fetch-data`. Not committed.\n\n"
        + "\n".join(f"- **{s.name}** ({s.licence}) — {s.note}\n  `{s.url}`"
                    for s in (sources or SOURCES)) + "\n",
        encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download third-party scam/spam corpora.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_json)
    counts = fetch(args.out_dir)
    log.info("total external rows: %d", sum(counts.values()))


if __name__ == "__main__":
    main()
