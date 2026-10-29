"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings, get_settings
from ..logging_setup import configure_logging, get_logger
from ..ml.detector import ModelNotTrained, get_detector
from .routes import router

log = get_logger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup so the first request is not slow."""
    try:
        detector = get_detector()
        log.info("model ready: %s (threshold %.3f)", detector.kind, detector.threshold)
    except ModelNotTrained as exc:
        log.warning("starting without a model: %s", exc)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    configure_logging(cfg.log_level, cfg.log_json)

    app = FastAPI(
        title="ScamShield",
        description="Scam detection for WhatsApp conversations.",
        version="1.0.0",
        lifespan=lifespan,
    )
    # Loopback only by default: the app's premise is that nothing leaves the
    # machine, so a wildcard origin would contradict it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)

    # The Cloud API webhook is mounted only when the integration is switched
    # on, so a default install exposes no public callback endpoint at all.
    if cfg.wa_enabled:
        from ..integrations.whatsapp.webhook import router as whatsapp_router

        app.include_router(whatsapp_router)
        log.info("WhatsApp Cloud API webhook mounted at /webhook/whatsapp "
                 "(auto-reply %s)", "on" if cfg.wa_auto_reply else "off")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
