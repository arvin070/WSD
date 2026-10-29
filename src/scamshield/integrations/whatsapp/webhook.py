"""
Webhook endpoints for the WhatsApp Cloud API.

Two rules govern the POST handler:

* **Authenticate before parsing.** The URL is public, so an unsigned body is
  rejected outright — otherwise anyone who learns the address can inject
  fabricated customer messages into the store.
* **Acknowledge quickly, and almost always with 200.** Meta retries on any
  non-2xx, so returning an error for a message we simply could not analyse
  would produce a retry loop. The message is durably stored before any
  analysis runs, so a 200 never means data was lost.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from ...config import Settings  # noqa: F401
from ...logging_setup import get_logger
from ...ml.detector import ModelNotTrained
from ..store import ConversationStore
from .models import parse_webhook
from .service import WhatsAppService, get_service
from .verify import SIGNATURE_HEADER, verify_challenge, verify_signature

log = get_logger(__name__)
router = APIRouter(prefix="/webhook/whatsapp", tags=["whatsapp"])


def service_dependency() -> WhatsAppService:
    try:
        return get_service()
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


Service = Annotated[WhatsAppService, Depends(service_dependency)]


@router.get("", response_class=PlainTextResponse)
def verify_subscription(
    service: Service,
    hub_mode: Annotated[str | None, Query(alias="hub.mode")] = None,
    hub_token: Annotated[str | None, Query(alias="hub.verify_token")] = None,
    hub_challenge: Annotated[str | None, Query(alias="hub.challenge")] = None,
) -> PlainTextResponse:
    """Meta's one-time subscription handshake."""
    challenge = verify_challenge(hub_mode, hub_token, hub_challenge,
                                 service.settings.wa_verify_token.get_secret_value())
    if challenge is None:
        log.warning("webhook verification rejected (mode=%s)", hub_mode)
        raise HTTPException(status_code=403, detail="Verification failed.")
    log.info("webhook subscription verified")
    return PlainTextResponse(challenge)


@router.post("")
async def receive(request: Request, service: Service,
                  signature: Annotated[str | None, Header(alias=SIGNATURE_HEADER)] = None):
    raw = await request.body()
    secret = service.settings.wa_app_secret.get_secret_value()

    if not secret:
        # Refuse to run unauthenticated rather than silently accepting anything.
        log.error("wa_app_secret is not configured; rejecting webhook delivery")
        raise HTTPException(status_code=503, detail="Webhook signing secret is not configured.")
    if not verify_signature(raw, signature, secret):
        log.warning("rejected webhook delivery with an invalid signature")
        raise HTTPException(status_code=403, detail="Invalid signature.")

    try:
        payload = await request.json()
    except ValueError:
        log.warning("webhook delivery had a malformed JSON body")
        return JSONResponse({"status": "ignored", "reason": "malformed json"}, status_code=200)

    event = parse_webhook(payload)
    if event.is_empty:
        return {"status": "ignored", "reason": "no messages"}

    try:
        result = await service.process(event)
    except Exception:
        # Already stored; a 200 stops Meta retrying a payload we cannot handle.
        log.exception("failed to process webhook delivery")
        return JSONResponse({"status": "error"}, status_code=200)
    return {"status": "ok", **result.as_dict()}


@router.get("/stats")
def stats(service: Service) -> dict:
    """Operational counters for the live integration."""
    store: ConversationStore = service.store
    return {
        "enabled": service.settings.wa_enabled,
        "auto_reply": service.settings.wa_auto_reply,
        "alert_threshold": round(service.alert_threshold, 4),
        "client_configured": service.client.configured,
        **store.stats(),
    }


@router.get("/contacts/{wa_id}")
def contact_profile(wa_id: str, service: Service) -> dict:
    """Risk history for one contact."""
    profile = service.store.contact(wa_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="No history for this contact.")
    history = service.store.history(wa_id, limit=25)
    return {
        "contact_id": profile.contact_id,
        "profile_name": profile.profile_name,
        "message_count": profile.message_count,
        "max_risk": round(profile.max_risk, 4),
        "alert_count": profile.alert_count,
        "last_alert": profile.last_alert_ts.isoformat() if profile.last_alert_ts else None,
        "recent": [{"direction": m.direction, "ts": m.ts.isoformat(),
                    "risk": m.risk, "verdict": m.verdict,
                    "text": m.text[:200]} for m in history],
    }
