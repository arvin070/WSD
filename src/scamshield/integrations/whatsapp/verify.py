"""
Webhook authenticity checks.

The endpoint is public, so both checks matter:

* **GET** — Meta's one-time subscription handshake echoes `hub.challenge`
  only when `hub.verify_token` matches the value configured in the App
  dashboard.
* **POST** — every delivery carries `X-Hub-Signature-256`, an HMAC-SHA256 of
  the *raw* body keyed by the app secret. Without checking it, anyone who
  learns the URL can inject fabricated customer messages.

The signature is computed over the exact bytes received; re-serialising the
parsed JSON would change whitespace and break the comparison.
"""

from __future__ import annotations

import hashlib
import hmac

SIGNATURE_HEADER = "X-Hub-Signature-256"
_PREFIX = "sha256="


def verify_challenge(mode: str | None, token: str | None, challenge: str | None,
                     expected_token: str) -> str | None:
    """Return the challenge to echo, or None when the handshake fails."""
    if not expected_token or mode != "subscribe" or challenge is None:
        return None
    if not hmac.compare_digest(str(token or ""), expected_token):
        return None
    return challenge


def sign_payload(body: bytes, app_secret: str) -> str:
    """Produce the header value Meta would send. Used by tests and tooling."""
    digest = hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return _PREFIX + digest


def verify_signature(body: bytes, header: str | None, app_secret: str) -> bool:
    """Constant-time check of X-Hub-Signature-256 against the raw body."""
    if not app_secret or not header or not header.startswith(_PREFIX):
        return False
    return hmac.compare_digest(header, sign_payload(body, app_secret))
