"""
WhatsApp Cloud API integration.

Covers the three things that break integrations in production: forged
deliveries, Meta's retries, and replies going out when nobody asked for them.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scamshield.config import Settings
from scamshield.integrations.store import ConversationStore
from scamshield.integrations.whatsapp.models import parse_webhook
from scamshield.integrations.whatsapp.responder import compose_advisory
from scamshield.integrations.whatsapp.service import WhatsAppService
from scamshield.integrations.whatsapp.verify import (
    sign_payload,
    verify_challenge,
    verify_signature,
)
from scamshield.integrations.whatsapp.webhook import router, service_dependency
from scamshield.ml.detector import ModelNotTrained, get_detector

APP_SECRET = "test-app-secret"
VERIFY_TOKEN = "test-verify-token"
CONTACT = "919876543210"

try:
    DET = get_detector()
except ModelNotTrained:  # pragma: no cover
    DET = None

needs_model = pytest.mark.skipif(DET is None, reason="run `scamshield-train` first")


def envelope(text, message_id="wamid.TEST1", mtype="text"):
    body = {"id": message_id, "from": CONTACT, "timestamp": "1710000000", "type": mtype}
    if mtype == "text":
        body["text"] = {"body": text}
    return {"object": "whatsapp_business_account", "entry": [{"id": "WABA", "changes": [{
        "field": "messages",
        "value": {
            "messaging_product": "whatsapp",
            "metadata": {"display_phone_number": "15550001", "phone_number_id": "PNID"},
            "contacts": [{"profile": {"name": "Asha"}, "wa_id": CONTACT}],
            "messages": [body],
        }}]}]}


# ---- payload parsing ------------------------------------------------------

def test_parse_text_message():
    ev = parse_webhook(envelope("hello there"))
    assert len(ev.messages) == 1
    m = ev.messages[0]
    assert m.text == "hello there" and m.from_wa_id == CONTACT
    assert m.profile_name == "Asha" and m.phone_number_id == "PNID" and m.is_text


def test_parse_interactive_and_button():
    payload = envelope(None, mtype="interactive")
    payload["entry"][0]["changes"][0]["value"]["messages"][0]["interactive"] = {
        "type": "button_reply", "button_reply": {"id": "1", "title": "Yes please"}}
    assert parse_webhook(payload).messages[0].text == "Yes please"


def test_parse_status_update():
    payload = {"entry": [{"changes": [{"value": {"statuses": [
        {"id": "wamid.X", "recipient_id": CONTACT, "status": "delivered"}]}}]}]}
    ev = parse_webhook(payload)
    assert not ev.messages and ev.statuses[0].status == "delivered"


@pytest.mark.parametrize("payload", [{}, {"entry": []}, {"entry": [{"changes": None}]},
                                     {"entry": [{"changes": [{"value": {}}]}]}, "not-a-dict"])
def test_parse_survives_garbage(payload):
    """A parser crash would turn one bad payload into a retry storm."""
    assert parse_webhook(payload).is_empty


def test_message_without_id_is_dropped():
    payload = envelope("hi")
    del payload["entry"][0]["changes"][0]["value"]["messages"][0]["id"]
    assert parse_webhook(payload).is_empty


# ---- authenticity ---------------------------------------------------------

def test_signature_roundtrip():
    body = b'{"a":1}'
    assert verify_signature(body, sign_payload(body, APP_SECRET), APP_SECRET)


@pytest.mark.parametrize("header", [None, "", "sha256=deadbeef", "md5=abc", "deadbeef"])
def test_bad_signatures_rejected(header):
    assert not verify_signature(b'{"a":1}', header, APP_SECRET)


def test_signature_is_body_specific():
    sig = sign_payload(b'{"a":1}', APP_SECRET)
    assert not verify_signature(b'{"a":2}', sig, APP_SECRET)


def test_signature_requires_the_right_secret():
    sig = sign_payload(b'{"a":1}', "other-secret")
    assert not verify_signature(b'{"a":1}', sig, APP_SECRET)


def test_challenge_verification():
    assert verify_challenge("subscribe", VERIFY_TOKEN, "12345", VERIFY_TOKEN) == "12345"
    assert verify_challenge("subscribe", "wrong", "12345", VERIFY_TOKEN) is None
    assert verify_challenge("unsubscribe", VERIFY_TOKEN, "12345", VERIFY_TOKEN) is None
    assert verify_challenge("subscribe", VERIFY_TOKEN, "12345", "") is None


# ---- service --------------------------------------------------------------

class FakeClient:
    """Records sends instead of calling Graph."""

    def __init__(self, configured=True):
        self.configured = configured
        self.sent: list[tuple[str, str, str | None]] = []

    async def send_text(self, to, body, reply_to=None):
        self.sent.append((to, body, reply_to))
        return f"wamid.OUT{len(self.sent)}"

    async def mark_read(self, message_id):
        return None


def build_service(tmp_path, **overrides):
    settings = Settings(data_dir=tmp_path, models_dir=Settings().models_dir, **overrides)
    store = ConversationStore(tmp_path / "conv.db")
    client = FakeClient()
    return WhatsAppService(settings=settings, detector=DET, store=store, client=client), client


SCAM = "Your account will be blocked today. Please share the OTP sent to your phone to verify."


@needs_model
@pytest.mark.anyio
async def test_scam_message_is_analysed_and_stored(tmp_path):
    svc, _ = build_service(tmp_path)
    result = await svc.process(parse_webhook(envelope(SCAM)))
    outcome = result.outcomes[0]
    assert outcome.status == "analysed"
    assert outcome.risk_score > 0.5
    assert svc.store.contact(CONTACT).message_count == 1


@needs_model
@pytest.mark.anyio
async def test_retried_delivery_is_deduplicated(tmp_path):
    """Meta redelivers until it gets a 2xx; the message must be scored once."""
    svc, _ = build_service(tmp_path)
    payload = parse_webhook(envelope(SCAM))
    first = await svc.process(payload)
    second = await svc.process(parse_webhook(envelope(SCAM)))
    assert first.outcomes[0].status == "analysed"
    assert second.outcomes[0].status == "duplicate"
    assert svc.store.stats()["messages"] == 1


@needs_model
@pytest.mark.anyio
async def test_non_text_is_skipped(tmp_path):
    svc, _ = build_service(tmp_path)
    payload = envelope(None, mtype="image")
    result = await svc.process(parse_webhook(payload))
    assert result.outcomes[0].status == "skipped_non_text"


@needs_model
@pytest.mark.anyio
async def test_no_reply_is_sent_by_default(tmp_path):
    """Replying uses the business's own number, so it must be opt-in."""
    svc, client = build_service(tmp_path)
    result = await svc.process(parse_webhook(envelope(SCAM)))
    assert client.sent == []
    assert result.outcomes[0].alert_suppressed == "auto_reply_disabled"


@needs_model
@pytest.mark.anyio
async def test_reply_sent_when_enabled(tmp_path):
    svc, client = build_service(tmp_path, wa_auto_reply=True, wa_alert_threshold=0.3)
    result = await svc.process(parse_webhook(envelope(SCAM)))
    assert result.outcomes[0].alerted
    to, body, reply_to = client.sent[0]
    assert to == CONTACT
    assert reply_to == "wamid.TEST1", "advisory should thread under the flagged message"
    assert "ScamShield" in body


@needs_model
@pytest.mark.anyio
async def test_cooldown_suppresses_a_second_alert(tmp_path):
    svc, client = build_service(tmp_path, wa_auto_reply=True, wa_alert_threshold=0.3,
                                wa_alert_cooldown_seconds=3600)
    await svc.process(parse_webhook(envelope(SCAM, "wamid.A")))
    result = await svc.process(parse_webhook(envelope(SCAM, "wamid.B")))
    assert len(client.sent) == 1
    assert result.outcomes[0].alert_suppressed == "cooldown"


@needs_model
@pytest.mark.anyio
async def test_benign_message_raises_no_alert(tmp_path):
    svc, client = build_service(tmp_path, wa_auto_reply=True)
    result = await svc.process(parse_webhook(envelope("hi, is my order shipped yet?")))
    assert not result.outcomes[0].alerted and client.sent == []


# ---- HTTP surface ---------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    svc, _ = build_service(tmp_path, wa_app_secret=APP_SECRET, wa_verify_token=VERIFY_TOKEN)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[service_dependency] = lambda: svc
    return TestClient(app)


@needs_model
def test_get_verification_handshake(client):
    r = client.get("/webhook/whatsapp", params={
        "hub.mode": "subscribe", "hub.verify_token": VERIFY_TOKEN, "hub.challenge": "9988"})
    assert r.status_code == 200 and r.text == "9988"


@needs_model
def test_get_verification_rejects_bad_token(client):
    r = client.get("/webhook/whatsapp", params={
        "hub.mode": "subscribe", "hub.verify_token": "nope", "hub.challenge": "9988"})
    assert r.status_code == 403


@needs_model
def test_post_rejects_unsigned_delivery(client):
    r = client.post("/webhook/whatsapp", json=envelope(SCAM))
    assert r.status_code == 403


@needs_model
def test_post_rejects_forged_signature(client):
    body = json.dumps(envelope(SCAM)).encode()
    r = client.post("/webhook/whatsapp", content=body,
                    headers={"X-Hub-Signature-256": sign_payload(body, "wrong-secret"),
                             "Content-Type": "application/json"})
    assert r.status_code == 403


@needs_model
def test_post_accepts_signed_delivery(client):
    body = json.dumps(envelope(SCAM)).encode()
    r = client.post("/webhook/whatsapp", content=body,
                    headers={"X-Hub-Signature-256": sign_payload(body, APP_SECRET),
                             "Content-Type": "application/json"})
    assert r.status_code == 200
    assert r.json()["analysed"] == 1


@needs_model
def test_stats_endpoint(client):
    body = json.dumps(envelope(SCAM)).encode()
    client.post("/webhook/whatsapp", content=body,
                headers={"X-Hub-Signature-256": sign_payload(body, APP_SECRET),
                         "Content-Type": "application/json"})
    stats = client.get("/webhook/whatsapp/stats").json()
    assert stats["messages"] == 1 and stats["auto_reply"] is False


@needs_model
def test_contact_profile_endpoint(client):
    body = json.dumps(envelope(SCAM)).encode()
    client.post("/webhook/whatsapp", content=body,
                headers={"X-Hub-Signature-256": sign_payload(body, APP_SECRET),
                         "Content-Type": "application/json"})
    r = client.get(f"/webhook/whatsapp/contacts/{CONTACT}")
    assert r.status_code == 200 and r.json()["message_count"] == 1
    assert client.get("/webhook/whatsapp/contacts/000").status_code == 404


# ---- advisory copy --------------------------------------------------------

def test_advisory_names_the_reason_and_the_action():
    body = compose_advisory({
        "risk_score": 0.93, "category": "otp_credential_theft",
        "category_label": "OTP or credential theft",
        "signals": [{"label": "Requests OTP / PIN / card credentials"}]})
    assert "93%" in body and "OTP or credential theft" in body
    assert "Never share an OTP" in body


def test_advisory_has_a_default_action():
    body = compose_advisory({"risk_score": 0.7, "signals": []})
    assert "Verify through a channel you already trust" in body


# ---- default posture ------------------------------------------------------

def test_webhook_not_mounted_unless_enabled():
    """A default install exposes no public callback endpoint."""
    from scamshield.api.app import create_app
    paths = set(create_app(Settings(wa_enabled=False)).openapi()["paths"])
    assert not any(p.startswith("/webhook") for p in paths)


def test_webhook_mounted_when_enabled():
    from scamshield.api.app import create_app
    paths = set(create_app(Settings(wa_enabled=True)).openapi()["paths"])
    assert "/webhook/whatsapp" in paths
