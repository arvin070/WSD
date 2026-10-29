"""API routes. Thin layer: parse, delegate to the analysis code, return."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..analysis.conversation import analyse_conversation
from ..config import Settings
from ..core.chat_parser import parse_any
from ..core.links import analyse_link, analyse_links_in
from ..logging_setup import get_logger
from ..ml.detector import ModelNotTrained, ScamDetector, get_detector
from .dependencies import detector_dependency, settings_dependency
from .schemas import (
    BatchRequest,
    BatchResponse,
    ChatRequest,
    HealthResponse,
    LinkRequest,
    LinkScanRequest,
    MessageRequest,
    MessageResult,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api")

Detector = Annotated[ScamDetector, Depends(detector_dependency)]
Config = Annotated[Settings, Depends(settings_dependency)]

NO_MESSAGES = "No messages could be read. Paste a WhatsApp export, or one message per line."


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    try:
        get_detector()
        return HealthResponse(status="ok", model_loaded=True)
    except ModelNotTrained as exc:
        return HealthResponse(status="degraded", model_loaded=False, detail=str(exc))


@router.get("/model", tags=["meta"])
def model_info(detector: Detector) -> dict:
    """Model card: thresholds, bands, corpus size, held-out metrics."""
    return detector.info()


@router.post("/analyze/message", response_model=MessageResult, tags=["analyze"])
def analyze_message(req: MessageRequest, detector: Detector) -> dict:
    return detector.score(req.text)


@router.post("/analyze/messages", response_model=BatchResponse, tags=["analyze"])
def analyze_messages(req: BatchRequest, detector: Detector) -> dict:
    return {"results": detector.score_batch(req.texts)}


@router.post("/analyze/chat", tags=["analyze"])
def analyze_chat(req: ChatRequest, detector: Detector) -> dict:
    chat = parse_any(req.content)
    if not chat.user_messages:
        raise HTTPException(status_code=422, detail=NO_MESSAGES)
    return analyse_conversation(chat, detector, req.focus_sender)


@router.post("/analyze/upload", tags=["analyze"])
async def analyze_upload(detector: Detector, cfg: Config,
                         file: Annotated[UploadFile, File()]) -> dict:
    raw = await file.read()
    if len(raw) > cfg.max_upload_bytes:
        raise HTTPException(status_code=413,
                            detail=f"File is larger than {cfg.max_upload_bytes // (1024 * 1024)} MB.")
    chat = parse_any(raw.decode("utf-8", errors="replace"))
    if not chat.user_messages:
        raise HTTPException(
            status_code=422,
            detail="No messages found. Export a chat from WhatsApp as a .txt file and upload that.")
    log.info("analysed upload %s (%d messages)", file.filename, len(chat.user_messages))
    result = analyse_conversation(chat, detector)
    result["summary"]["filename"] = file.filename
    return result


@router.post("/links/analyze", tags=["links"])
def analyze_link(req: LinkRequest) -> dict:
    """
    Offline analysis of one URL. No network request is made, so this is safe
    to call on anything and cannot signal to the link's owner that it was seen.
    """
    return analyse_link(req.url).as_dict()


@router.post("/links/scan", tags=["links"])
def scan_links(req: LinkScanRequest) -> dict:
    """Analyse every link in a message, following one level of embedding."""
    reports = [r.as_dict() for r in analyse_links_in(req.text)]
    return {
        "count": len(reports),
        "max_risk": max((r["risk_score"] for r in reports), default=0.0),
        "links": reports,
    }


@router.post("/links/inspect", tags=["links"])
async def inspect_link(req: LinkRequest, cfg: Config) -> dict:
    """
    Fetch the URL in isolation and report where it really lands.

    Opt-in: requires `SCAMSHIELD_SANDBOX_ENABLED`. Unlike the offline
    endpoints this makes a real outbound request, which reveals the caller's
    IP to whoever controls the link.
    """
    from ..sandbox.inspector import SandboxDisabled, inspect_url

    try:
        result = await inspect_url(req.url, cfg)
    except SandboxDisabled as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return result.as_dict()


@router.get("/sample", tags=["meta"])
def sample_chat(cfg: Config) -> dict:
    path = cfg.data_dir / "sample_chat.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample chat not available.")
    return {"content": path.read_text(encoding="utf-8")}
