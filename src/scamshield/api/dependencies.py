"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException

from ..config import Settings, get_settings
from ..ml.detector import ModelNotTrained, ScamDetector, get_detector


def detector_dependency() -> ScamDetector:
    """Resolve the detector, turning a missing model into a clean 503."""
    try:
        return get_detector()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def settings_dependency() -> Settings:
    return get_settings()
