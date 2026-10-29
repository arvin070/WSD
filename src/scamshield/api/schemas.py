"""Request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..config import get_settings

_cfg = get_settings()


class MessageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=_cfg.max_message_chars,
                      description="A single chat message to score.")


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=_cfg.max_batch_messages)


class ChatRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=_cfg.max_chat_chars,
                         description="Raw WhatsApp export text, or one message per line.")
    focus_sender: str | None = Field(
        None, description="Restrict the headline verdict to one participant.")


class LinkRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=2048,
                     description="A single URL to analyse.")


class LinkScanRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=_cfg.max_message_chars,
                      description="Message text; every link in it is analysed.")


class Signal(BaseModel):
    code: str
    label: str
    weight: float
    evidence: list[str]


class LinkFinding(BaseModel):
    code: str
    label: str
    weight: float
    detail: str = ""


class LinkReport(BaseModel):
    url: str
    scheme: str = ""
    host: str = ""
    registrable_domain: str = ""
    risk_score: float
    verdict: str
    is_shortener: bool = False
    embedded_urls: list[str] = []
    findings: list[LinkFinding] = []


class MessageResult(BaseModel):
    text: str
    risk_score: float
    ml_score: float
    rule_score: float
    is_scam: bool
    high_confidence: bool
    low_information: bool = False
    verdict: str
    advice: str
    category: str | None = None
    category_label: str | None = None
    category_confidence: float = 0.0
    signals: list[Signal] = []
    #: Every link analysed offline, so the user can judge it before clicking.
    links: list[LinkReport] = []


class BatchResponse(BaseModel):
    results: list[MessageResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: str | None = None
