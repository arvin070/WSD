"""
Runtime configuration.

Every tunable lives here rather than being scattered as literals through the
code. Values come from the environment (prefix `SCAMSHIELD_`) or a local
`.env`, so the same image can run in development and behind a reverse proxy
without edits. See `.env.example`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
# src/scamshield -> src -> project root
PROJECT_ROOT = PACKAGE_DIR.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCAMSHIELD_", env_file=".env", extra="ignore",
        protected_namespaces=(),
    )

    # -- paths -------------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    models_dir: Path = Field(default=PROJECT_ROOT / "models")
    resources_dir: Path = Field(default=PACKAGE_DIR / "resources")

    # -- scoring -----------------------------------------------------------
    #: Messages shorter than this with no red flags cannot be judged; "Hi" is
    #: not evidence of anything.
    low_info_tokens: int = 4
    low_info_cap: float = 0.25
    #: Hide the predicted scam playbook when the posterior is this uncertain.
    min_category_confidence: float = 0.25
    #: Optional override of the threshold learned during training.
    threshold_override: float | None = None

    # -- conversation analysis --------------------------------------------
    escalation_min_messages: int = 6
    escalation_min_delta: float = 0.30
    escalation_min_late_peak: float = 0.50
    burst_window_seconds: int = 60

    # -- api ---------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    #: Locked to loopback by default: the app's premise is that nothing leaves
    #: the machine, so a wildcard origin would contradict it.
    #: NoDecode keeps pydantic-settings from JSON-decoding the env value, so
    #: the validator below receives the raw "a,b" string it expects.
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:8000", "http://127.0.0.1:8000",
    ]
    max_upload_bytes: int = 8 * 1024 * 1024
    max_batch_messages: int = 500
    max_message_chars: int = 10_000
    max_chat_chars: int = 4_000_000

    # -- URL inspection sandbox --------------------------------------------
    #: Off by default. Fetching a link tells its owner the target is live and
    #: reveals the requesting IP, so it is never automatic.
    sandbox_enabled: bool = False
    sandbox_timeout_seconds: float = 8.0
    sandbox_max_redirects: int = 5
    sandbox_max_bytes: int = 512_000
    sandbox_user_agent: str = "Mozilla/5.0 (compatible; ScamShield-Inspector/1.0)"
    #: Test-only. Never enable where untrusted URLs are accepted - it disables
    #: the SSRF address filter.
    sandbox_allow_private_hosts: bool = False

    # -- WhatsApp Cloud API integration ------------------------------------
    #: Master switch. The webhook routes are not mounted unless this is on.
    wa_enabled: bool = False
    #: Echoed back during Meta's subscription handshake (GET verification).
    wa_verify_token: SecretStr = SecretStr("")
    #: App secret used to check X-Hub-Signature-256 on every delivery.
    wa_app_secret: SecretStr = SecretStr("")
    #: Permanent access token for the system user that owns the number.
    wa_access_token: SecretStr = SecretStr("")
    wa_phone_number_id: str = ""
    wa_api_version: str = "v21.0"
    wa_api_base: str = "https://graph.facebook.com"
    wa_timeout_seconds: float = 10.0
    #: Replying sends a message from the business's own number, so it is
    #: off until someone deliberately turns it on.
    wa_auto_reply: bool = False
    #: Risk at or above which a warning is sent. None uses the model's own
    #: high-confidence threshold.
    wa_alert_threshold: float | None = None
    #: Minimum gap between warnings to one contact, so a burst of scam
    #: messages does not produce a burst of replies.
    wa_alert_cooldown_seconds: int = 3600
    wa_history_limit: int = 100
    wa_mark_read: bool = False

    # -- logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def wa_store_path(self) -> Path:
        return self.data_dir / "conversations.db"

    @property
    def model_path(self) -> Path:
        return self.models_dir / "scam_model.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / "metadata.json"

    @property
    def lexicons_path(self) -> Path:
        return self.resources_dir / "lexicons.yaml"

    @property
    def templates_path(self) -> Path:
        return self.resources_dir / "templates.yaml"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
