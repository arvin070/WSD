"""
Loaders for the YAML content files.

Lexicons and corpus templates are *content*, not code: they change whenever a
new scam pattern appears, and they should be editable by someone who does not
write Python. They ship with the package and are cached after first read.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .config import get_settings


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing resource file: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


@dataclass(frozen=True)
class SignalDef:
    """One named family of scam indicators."""

    code: str
    label: str
    weight: float
    feature: str
    terms: tuple[str, ...]
    #: "log1p" damps repeated hits; "bool" makes the feature a plain flag.
    count_mode: str = "log1p"


@dataclass(frozen=True)
class DerivedSignal:
    """A signal raised by a computed condition rather than a term list."""

    code: str
    label: str
    weight: float


@dataclass(frozen=True)
class Lexicons:
    signals: tuple[SignalDef, ...]
    derived: dict[str, DerivedSignal]
    scoring: dict[str, float]
    narrative: tuple[str, ...]
    safe_transactional: tuple[str, ...]
    shorteners: frozenset[str]
    suspicious_tlds: frozenset[str]
    impersonated_brands: tuple[str, ...]
    legit_domain_hints: dict[str, str]
    links: dict

    def by_code(self, code: str) -> SignalDef:
        for s in self.signals:
            if s.code == code:
                return s
        raise KeyError(code)


@lru_cache(maxsize=1)
def load_lexicons() -> Lexicons:
    raw = _load(get_settings().lexicons_path)
    signals = tuple(
        SignalDef(
            code=code,
            label=body["label"],
            weight=float(body["weight"]),
            feature=body["feature"],
            terms=tuple(body.get("terms", [])),
            count_mode=body.get("count_mode", "log1p"),
        )
        for code, body in raw["signals"].items()
    )
    derived = {
        code: DerivedSignal(code=code, label=b["label"], weight=float(b["weight"]))
        for code, b in raw.get("derived_signals", {}).items()
    }
    ctx, dom = raw.get("context", {}), raw.get("domains", {})
    return Lexicons(
        signals=signals,
        derived=derived,
        scoring={k: float(v) for k, v in raw.get("scoring", {}).items()},
        narrative=tuple(ctx.get("narrative", [])),
        safe_transactional=tuple(ctx.get("safe_transactional", [])),
        shorteners=frozenset(dom.get("shorteners", [])),
        suspicious_tlds=frozenset(dom.get("suspicious_tlds", [])),
        impersonated_brands=tuple(dom.get("impersonated_brands", [])),
        legit_domain_hints=dict(dom.get("legit_domain_hints", {})),
        links=raw.get("links", {}),
    )


@lru_cache(maxsize=1)
def load_templates() -> dict[str, Any]:
    raw = _load(get_settings().templates_path)
    for key in ("slots", "scam", "ham"):
        if key not in raw:
            raise ValueError(f"templates.yaml is missing the '{key}' section")
    return raw
