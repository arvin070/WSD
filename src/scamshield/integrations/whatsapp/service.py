"""
Turns inbound webhook messages into analysis, and optionally a reply.

Order matters here:

1. **Deduplicate first.** Meta retries a delivery until it receives a 2xx, so
   the same message arrives repeatedly. Storing on the message id makes the
   first write the only one that proceeds.
2. **Analyse the thread, not the message.** History is replayed from the store
   so escalation and per-sender profiling work on a live stream the same way
   they do on an exported chat.
3. **Reply last, and only if asked.** Sending uses the business's own number,
   so it stays behind `wa_auto_reply` and a per-contact cooldown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ...analysis.conversation import analyse_conversation
from ...config import Settings, get_settings
from ...core.chat_parser import ChatMessage, ParsedChat
from ...logging_setup import get_logger
from ...ml.detector import ScamDetector, get_detector
from ..store import ConversationStore
from .client import WhatsAppClient, WhatsAppSendError
from .models import InboundMessage, WebhookEvent
from .responder import compose_advisory

log = get_logger(__name__)

BUSINESS_SENDER = "Business"


@dataclass
class MessageOutcome:
    message_id: str
    contact: str
    status: str                       # analysed | duplicate | skipped_non_text
    risk_score: float | None = None
    verdict: str | None = None
    category: str | None = None
    alerted: bool = False
    alert_suppressed: str | None = None


@dataclass
class ProcessResult:
    outcomes: list[MessageOutcome] = field(default_factory=list)
    statuses: int = 0

    def as_dict(self) -> dict:
        return {
            "processed": len(self.outcomes),
            "analysed": sum(1 for o in self.outcomes if o.status == "analysed"),
            "duplicates": sum(1 for o in self.outcomes if o.status == "duplicate"),
            "alerts": sum(1 for o in self.outcomes if o.alerted),
            "statuses": self.statuses,
        }


def _mask(wa_id: str) -> str:
    return f"***{wa_id[-4:]}" if len(wa_id) > 4 else "***"


class WhatsAppService:
    def __init__(self, settings: Settings | None = None,
                 detector: ScamDetector | None = None,
                 store: ConversationStore | None = None,
                 client: WhatsAppClient | None = None):
        self.settings = settings or get_settings()
        self.detector = detector or get_detector()
        self.store = store or ConversationStore(self.settings.wa_store_path,
                                                self.settings.wa_history_limit)
        self.client = client or WhatsAppClient(self.settings)

    # -- thread reconstruction --------------------------------------------
    def _build_chat(self, contact_id: str) -> ParsedChat:
        """Replay stored history as a ParsedChat the analyser understands."""
        chat = ParsedChat()
        for i, m in enumerate(self.store.history(contact_id)):
            chat.messages.append(ChatMessage(
                index=i,
                sender=contact_id if m.direction == "inbound" else BUSINESS_SENDER,
                text=m.text,
                timestamp=m.ts,
            ))
        chat.participants = [contact_id]
        return chat

    @property
    def alert_threshold(self) -> float:
        configured = self.settings.wa_alert_threshold
        return configured if configured is not None else self.detector.strict_threshold

    def _cooldown_active(self, contact_id: str) -> bool:
        profile = self.store.contact(contact_id)
        if not profile or not profile.last_alert_ts:
            return False
        elapsed = (datetime.now(tz=timezone.utc) - profile.last_alert_ts).total_seconds()
        return elapsed < self.settings.wa_alert_cooldown_seconds

    # -- main entry point --------------------------------------------------
    async def process(self, event: WebhookEvent) -> ProcessResult:
        result = ProcessResult(statuses=len(event.statuses))
        for message in event.messages:
            result.outcomes.append(await self._process_one(message))
        for err in event.errors:
            log.warning("cloud api reported an error: %s", err)
        return result

    async def _process_one(self, message: InboundMessage) -> MessageOutcome:
        contact = message.from_wa_id
        outcome = MessageOutcome(message_id=message.message_id, contact=contact, status="analysed")

        if not message.is_text:
            # Images, audio and documents are outside what the model reads.
            outcome.status = "skipped_non_text"
            return outcome

        is_new = self.store.record_inbound(
            message.message_id, contact, message.text, message.timestamp,
            message.phone_number_id, message.profile_name)
        if not is_new:
            outcome.status = "duplicate"
            return outcome

        single = self.detector.score(message.text)
        chat = self._build_chat(contact)
        conversation = analyse_conversation(chat, self.detector, settings=self.settings)

        # The thread verdict can exceed the individual message - escalation and
        # repeated flags are properties of the conversation, not one line.
        risk = max(single["risk_score"], conversation["summary"]["risk_score"])
        outcome.risk_score = round(risk, 4)
        outcome.verdict = single["verdict"]
        outcome.category = single["category"]
        self.store.record_verdict(message.message_id, contact, risk,
                                  single["verdict"], single["category"])
        log.info("scored %s from %s: risk=%.3f verdict=%s",
                 message.message_id, _mask(contact), risk, single["verdict"])

        if self.settings.wa_mark_read and self.client.configured:
            await self.client.mark_read(message.message_id)

        await self._maybe_alert(message, single, risk, outcome)
        return outcome

    async def _maybe_alert(self, message: InboundMessage, single: dict,
                           risk: float, outcome: MessageOutcome) -> None:
        if risk < self.alert_threshold:
            return
        if not self.settings.wa_auto_reply:
            outcome.alert_suppressed = "auto_reply_disabled"
            return
        if not self.client.configured:
            outcome.alert_suppressed = "client_not_configured"
            return
        if self._cooldown_active(message.from_wa_id):
            outcome.alert_suppressed = "cooldown"
            return

        body = compose_advisory({**single, "risk_score": risk})
        try:
            sent_id = await self.client.send_text(message.from_wa_id, body,
                                                  reply_to=message.message_id)
        except WhatsAppSendError as exc:
            # A failed reply must not fail the webhook; the analysis is stored.
            log.error("advisory send failed for %s: %s", _mask(message.from_wa_id), exc)
            outcome.alert_suppressed = "send_failed"
            return
        if sent_id:
            self.store.record_outbound(sent_id, message.from_wa_id, body)
        self.store.record_alert(message.from_wa_id)
        outcome.alerted = True


_service: WhatsAppService | None = None


def get_service() -> WhatsAppService:
    global _service
    if _service is None:
        _service = WhatsAppService()
    return _service


def reset_service() -> None:
    global _service
    _service = None
