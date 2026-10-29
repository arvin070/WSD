"""
Minimal WhatsApp Cloud API client.

Only the calls this integration needs: send a text reply, and optionally mark
an inbound message read. Everything else the Graph API offers is out of scope.

Sending is an outward-facing action performed with the business's own number,
so `WhatsAppClient.send_text` is never called unless `wa_auto_reply` is
explicitly enabled in configuration.
"""

from __future__ import annotations

import httpx

from ...config import Settings
from ...logging_setup import get_logger

log = get_logger(__name__)


class WhatsAppSendError(RuntimeError):
    """The Graph API rejected a send."""


class WhatsAppClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._token = settings.wa_access_token.get_secret_value()
        self._phone_number_id = settings.wa_phone_number_id

    @property
    def configured(self) -> bool:
        return bool(self._token and self._phone_number_id)

    @property
    def _endpoint(self) -> str:
        cfg = self.settings
        return f"{cfg.wa_api_base}/{cfg.wa_api_version}/{self._phone_number_id}/messages"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    async def _post(self, payload: dict) -> dict:
        if not self.configured:
            raise WhatsAppSendError("wa_access_token and wa_phone_number_id are required")
        async with httpx.AsyncClient(timeout=self.settings.wa_timeout_seconds) as client:
            resp = await client.post(self._endpoint, json=payload, headers=self._headers)
        if resp.status_code >= 400:
            # Never log the token; the body can carry the offending payload.
            raise WhatsAppSendError(f"{resp.status_code}: {resp.text[:300]}")
        return resp.json()

    async def send_text(self, to: str, body: str, reply_to: str | None = None) -> str:
        """
        Send a text message. Returns the new message id.

        `reply_to` threads the warning under the message it is about, which is
        what makes an inline warning legible in the customer's chat view.
        """
        payload: dict = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"body": body[:4096], "preview_url": False},
        }
        if reply_to:
            payload["context"] = {"message_id": reply_to}
        data = await self._post(payload)
        message_id = ((data.get("messages") or [{}])[0]).get("id", "")
        log.info("sent advisory to %s (message %s)", _mask(to), message_id)
        return message_id

    async def mark_read(self, message_id: str) -> None:
        try:
            await self._post({"messaging_product": "whatsapp", "status": "read",
                              "message_id": message_id})
        except WhatsAppSendError as exc:
            # Read receipts are cosmetic - never fail a webhook over one.
            log.debug("mark_read failed for %s: %s", message_id, exc)


def _mask(wa_id: str) -> str:
    """Phone numbers are personal data; keep only enough to correlate logs."""
    return f"***{wa_id[-4:]}" if len(wa_id) > 4 else "***"
