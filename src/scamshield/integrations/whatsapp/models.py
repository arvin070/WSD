"""
Typed view over WhatsApp Cloud API webhook payloads.

Meta's envelope is deeply nested and every level is optional, so this parses
defensively: an unrecognised or partial payload yields no messages rather than
raising. The webhook must always return 200 quickly — Meta retries on any
non-2xx, and a parser crash would turn one bad payload into a retry storm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

#: Message types the detector can actually read. Anything else is recorded
#: but not scored - we do not analyse images, audio or documents.
TEXT_TYPES = {"text", "button", "interactive"}


@dataclass(frozen=True)
class InboundMessage:
    """One customer message delivered by the Cloud API."""

    message_id: str
    from_wa_id: str
    phone_number_id: str
    timestamp: datetime
    type: str
    text: str = ""
    profile_name: str = ""
    context_id: str | None = None

    @property
    def is_text(self) -> bool:
        return self.type in TEXT_TYPES and bool(self.text.strip())


@dataclass(frozen=True)
class StatusUpdate:
    """Delivery receipt. Recorded for observability, never scored."""

    message_id: str
    recipient_id: str
    status: str


@dataclass
class WebhookEvent:
    messages: list[InboundMessage] = field(default_factory=list)
    statuses: list[StatusUpdate] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.messages or self.statuses or self.errors)


def _extract_text(msg: dict) -> str:
    """Pull the body out of whichever text-bearing shape Meta used."""
    mtype = msg.get("type")
    if mtype == "text":
        return (msg.get("text") or {}).get("body", "")
    if mtype == "button":
        return (msg.get("button") or {}).get("text", "")
    if mtype == "interactive":
        inter = msg.get("interactive") or {}
        for key in ("button_reply", "list_reply"):
            if key in inter:
                node = inter[key] or {}
                return node.get("title") or node.get("description") or ""
    return ""


def _timestamp(raw: object) -> datetime:
    try:
        return datetime.fromtimestamp(int(str(raw)), tz=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(tz=timezone.utc)


def parse_webhook(payload: dict) -> WebhookEvent:
    """Flatten a Cloud API webhook body into messages and statuses."""
    event = WebhookEvent()
    if not isinstance(payload, dict):
        return event

    for entry in payload.get("entry") or []:
        for change in (entry or {}).get("changes") or []:
            value = (change or {}).get("value") or {}
            metadata = value.get("metadata") or {}
            phone_number_id = str(metadata.get("phone_number_id", ""))

            # profile names arrive in a sibling array keyed by wa_id
            names = {
                str(c.get("wa_id")): ((c.get("profile") or {}).get("name") or "")
                for c in value.get("contacts") or []
            }

            for msg in value.get("messages") or []:
                if not isinstance(msg, dict) or not msg.get("id"):
                    continue
                wa_id = str(msg.get("from", ""))
                event.messages.append(InboundMessage(
                    message_id=str(msg["id"]),
                    from_wa_id=wa_id,
                    phone_number_id=phone_number_id,
                    timestamp=_timestamp(msg.get("timestamp")),
                    type=str(msg.get("type", "unknown")),
                    text=_extract_text(msg),
                    profile_name=names.get(wa_id, ""),
                    context_id=(msg.get("context") or {}).get("id"),
                ))

            for st in value.get("statuses") or []:
                if not isinstance(st, dict):
                    continue
                event.statuses.append(StatusUpdate(
                    message_id=str(st.get("id", "")),
                    recipient_id=str(st.get("recipient_id", "")),
                    status=str(st.get("status", "")),
                ))
                event.errors.extend(st.get("errors") or [])

            event.errors.extend(value.get("errors") or [])
    return event
