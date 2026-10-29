"""Pre-flight checks for the WhatsApp Business Platform connection."""

import pytest

from scamshield.config import Settings
from scamshield.integrations.whatsapp.preflight import (
    FAIL,
    OK,
    WARN,
    Report,
    check_config,
    dashboard_block,
)

CONFIGURED = {
    "wa_enabled": True,
    "wa_app_secret": "app-secret-value",
    "wa_verify_token": "verify-token-value",
    "wa_access_token": "access-token-value",
    "wa_phone_number_id": "123456789012345",
}


def _status(report: Report, name: str) -> str:
    return next(s for s, n, _ in report.rows if n == name)


def _detail(report: Report, name: str) -> str:
    return next(d for s, n, d in report.rows if n == name)


def test_unconfigured_install_fails_loudly():
    report = check_config(Settings(), None)
    assert report.failed >= 3
    assert _status(report, "integration enabled") == FAIL
    assert _status(report, "app secret") == FAIL
    assert _status(report, "verify token") == FAIL


def test_fully_configured_passes():
    report = check_config(Settings(**CONFIGURED), "https://scam.example.com")
    assert report.failed == 0


def test_missing_send_credentials_warn_but_do_not_fail():
    """Analysis-only deployments are valid; they just cannot reply."""
    cfg = Settings(wa_enabled=True, wa_app_secret="s", wa_verify_token="v")
    report = check_config(cfg, "https://x.example.com")
    assert _status(report, "access token") == WARN
    assert _status(report, "phone number id") == WARN
    assert report.failed == 0


def test_auto_reply_on_is_surfaced_as_a_warning():
    """Replying sends from the user's own number - never silent."""
    report = check_config(Settings(**CONFIGURED, wa_auto_reply=True), "https://x.example.com")
    assert _status(report, "auto-reply") == WARN
    off = check_config(Settings(**CONFIGURED), "https://x.example.com")
    assert _status(off, "auto-reply") == OK


def test_http_public_url_is_rejected():
    report = check_config(Settings(**CONFIGURED), "http://insecure.example.com")
    assert _status(report, "public URL scheme") == FAIL


def test_secrets_are_masked_in_output():
    report = check_config(Settings(**CONFIGURED), "https://x.example.com")
    rendered = report.render()
    assert "app-secret-value" not in rendered
    assert "access-token-value" not in rendered


def test_dashboard_block_has_the_exact_values_to_paste():
    block = dashboard_block(Settings(**CONFIGURED), "https://scam.example.com/")
    assert "https://scam.example.com/webhook/whatsapp" in block
    assert "verify-token-value" in block
    assert "messages" in block


@pytest.mark.parametrize("url,expected", [
    ("https://a.example.com", "https://a.example.com/webhook/whatsapp"),
    ("https://a.example.com/", "https://a.example.com/webhook/whatsapp"),
])
def test_trailing_slash_does_not_double(url, expected):
    assert expected in dashboard_block(Settings(**CONFIGURED), url)
