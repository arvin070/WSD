"""HTTP surface."""

import pytest
from fastapi.testclient import TestClient

from scamshield.api.app import create_app
from scamshield.ml.detector import ModelNotTrained, get_detector

try:
    get_detector()
    TRAINED = True
except ModelNotTrained:  # pragma: no cover
    TRAINED = False

client = TestClient(create_app())
pytestmark = pytest.mark.skipif(not TRAINED, reason="run `python -m ml.train` first")


def test_health():
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model_loaded"]


def test_model_info():
    body = client.get("/api/model").json()
    assert body["threshold"] > 0 and body["categories"]
    assert body["bands"][0]["verdict"] == "dangerous"


def test_analyze_message():
    r = client.post("/api/analyze/message",
                    json={"text": "Share the OTP sent to your phone to verify your account"})
    assert r.status_code == 200
    assert r.json()["is_scam"]


def test_analyze_message_rejects_empty():
    assert client.post("/api/analyze/message", json={"text": ""}).status_code == 422


def test_analyze_batch():
    r = client.post("/api/analyze/messages", json={"texts": ["hi there", "share your OTP now"]})
    assert len(r.json()["results"]) == 2


def test_analyze_chat():
    r = client.post("/api/analyze/chat", json={
        "content": "12/03/2024, 10:46 pm - +91 98765 43210: Your KYC expired, share OTP to verify\n"
                   "12/03/2024, 10:47 pm - Priya: who is this?"})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["message_count"] == 2 and body["senders"]


def test_analyze_chat_rejects_unreadable():
    assert client.post("/api/analyze/chat", json={"content": "   "}).status_code == 422


def test_upload():
    r = client.post("/api/analyze/upload",
                    files={"file": ("chat.txt", b"Scammer: share your OTP now to verify\n", "text/plain")})
    assert r.status_code == 200
    assert r.json()["summary"]["filename"] == "chat.txt"


def test_sample_available():
    assert "content" in client.get("/api/sample").json()


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "ScamShield" in r.text
