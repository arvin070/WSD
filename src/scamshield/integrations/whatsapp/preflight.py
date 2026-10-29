"""
Pre-flight check for the WhatsApp Business Platform connection.

Code being correct and a number being connected are different things. This
reports which of the two you have: it validates configuration offline, and
with `--live` confirms the access token and phone number against the Graph
API. It prints the exact webhook URL and verify token to paste into the Meta
App Dashboard, so the values cannot drift between .env and the dashboard.

Run before going live, and again whenever deliveries stop arriving:

    scamshield-whatsapp-check --live --public-url https://your.domain
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from ...config import Settings, get_settings
from ...ml.detector import ModelNotTrained, get_detector
from ..store import ConversationStore

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
_MARK = {OK: "  [ok]  ", WARN: "  [warn]", FAIL: "  [FAIL]"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))

    def check(self, condition: bool, name: str, ok_detail: str = "",
              fail_detail: str = "", warn_only: bool = False) -> bool:
        if condition:
            self.add(OK, name, ok_detail)
        else:
            self.add(WARN if warn_only else FAIL, name, fail_detail)
        return condition

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == FAIL)

    def render(self) -> str:
        width = max(len(n) for _, n, _ in self.rows) + 2
        lines = [f"{_MARK[s]} {n:<{width}} {d}".rstrip() for s, n, d in self.rows]
        return "\n".join(lines)


def _masked(value: str) -> str:
    return f"set ({len(value)} chars, ends …{value[-4:]})" if len(value) > 6 else "set"


def check_config(cfg: Settings, public_url: str | None) -> Report:
    r = Report()

    r.check(cfg.wa_enabled, "integration enabled",
            "SCAMSHIELD_WA_ENABLED=true", "set SCAMSHIELD_WA_ENABLED=true or no webhook is mounted")

    secret = cfg.wa_app_secret.get_secret_value()
    r.check(bool(secret), "app secret", _masked(secret),
            "SCAMSHIELD_WA_APP_SECRET is empty - the webhook refuses every delivery without it")

    token = cfg.wa_verify_token.get_secret_value()
    r.check(bool(token), "verify token", _masked(token),
            "SCAMSHIELD_WA_VERIFY_TOKEN is empty - Meta's handshake cannot succeed")

    access = cfg.wa_access_token.get_secret_value()
    r.check(bool(access), "access token", _masked(access),
            "SCAMSHIELD_WA_ACCESS_TOKEN is empty - replies cannot be sent", warn_only=True)

    r.check(bool(cfg.wa_phone_number_id), "phone number id", cfg.wa_phone_number_id,
            "SCAMSHIELD_WA_PHONE_NUMBER_ID is empty - replies cannot be sent", warn_only=True)

    # Replying is a deliberate choice, so report the posture rather than judging it.
    r.add(OK if not cfg.wa_auto_reply else WARN, "auto-reply",
          "off - analysis only, nothing is sent from your number" if not cfg.wa_auto_reply
          else f"ON - warnings will be sent from {cfg.wa_phone_number_id or 'your number'}")

    try:
        detector = get_detector()
        r.add(OK, "model", f"{detector.kind}, threshold {detector.threshold:.3f}")
    except ModelNotTrained:
        r.add(FAIL, "model", "no trained model - run scamshield-train")

    try:
        store = ConversationStore(cfg.wa_store_path, cfg.wa_history_limit)
        stats = store.stats()
        r.add(OK, "conversation store",
              f"{cfg.wa_store_path} ({stats['messages']} messages, {stats['contacts']} contacts)")
    except Exception as exc:  # noqa: BLE001 - surfaced as a check, not a crash
        r.add(FAIL, "conversation store", f"{type(exc).__name__}: {exc}")

    if public_url:
        url = public_url.rstrip("/")
        r.check(url.startswith("https://"), "public URL scheme", url,
                f"Meta requires HTTPS; got {url}")
    else:
        r.add(WARN, "public URL", "not given - pass --public-url to print the dashboard values")
    return r


async def check_live(cfg: Settings) -> Report:
    """Confirm the token and number against the Graph API."""
    r = Report()
    access = cfg.wa_access_token.get_secret_value()
    if not (access and cfg.wa_phone_number_id):
        r.add(WARN, "live check", "skipped - access token and phone number id are required")
        return r

    base = f"{cfg.wa_api_base}/{cfg.wa_api_version}/{cfg.wa_phone_number_id}"
    try:
        async with httpx.AsyncClient(timeout=cfg.wa_timeout_seconds) as client:
            resp = await client.get(base, params={
                "fields": "display_phone_number,verified_name,quality_rating,platform_type"},
                headers={"Authorization": f"Bearer {access}"})
    except httpx.HTTPError as exc:
        r.add(FAIL, "graph api reachable", f"{type(exc).__name__}")
        return r

    if resp.status_code == 200:
        d = resp.json()
        r.add(OK, "access token", "accepted by Graph API")
        r.add(OK, "phone number", f"{d.get('display_phone_number', '?')} "
                                  f"({d.get('verified_name', 'unverified')})")
        rating = (d.get("quality_rating") or "unknown")
        r.add(OK if rating in {"GREEN", "UNKNOWN", "unknown"} else WARN,
              "quality rating", rating)
    elif resp.status_code in (400, 401, 403):
        detail = (resp.json().get("error", {}) or {}).get("message", resp.text[:120])
        r.add(FAIL, "access token", f"{resp.status_code}: {detail}")
    else:
        r.add(FAIL, "graph api", f"{resp.status_code}: {resp.text[:120]}")
    return r


def dashboard_block(cfg: Settings, public_url: str | None) -> str:
    url = (public_url or "https://<your-public-host>").rstrip("/")
    token = cfg.wa_verify_token.get_secret_value() or "<SCAMSHIELD_WA_VERIFY_TOKEN>"
    return (
        "\nPaste these into Meta App Dashboard -> WhatsApp -> Configuration:\n"
        f"  Callback URL   {url}/webhook/whatsapp\n"
        f"  Verify token   {token}\n"
        "  Webhook fields subscribe to 'messages'\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the WhatsApp Business Platform connection.")
    parser.add_argument("--live", action="store_true",
                        help="also verify the token and number against the Graph API")
    parser.add_argument("--public-url", default=None,
                        help="the HTTPS base URL Meta will call, e.g. https://scam.example.com")
    args = parser.parse_args()
    cfg = get_settings()

    print("ScamShield - WhatsApp Business Platform pre-flight\n")
    report = check_config(cfg, args.public_url)
    print(report.render())
    failures = report.failed

    if args.live:
        print()
        live = asyncio.run(check_live(cfg))
        print(live.render())
        failures += live.failed

    print(dashboard_block(cfg, args.public_url))
    if failures:
        print(f"{failures} check(s) failed - the integration will not work until they are fixed.")
        sys.exit(1)
    print("Configuration looks complete.")


if __name__ == "__main__":
    main()
