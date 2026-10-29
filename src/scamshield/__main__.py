"""`python -m scamshield` / `scamshield-serve` - run the web app."""

from __future__ import annotations

import uvicorn

from .config import get_settings
from .logging_setup import configure_logging


def main() -> None:
    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_json)
    uvicorn.run("scamshield.api.app:app", host=cfg.host, port=cfg.port, log_config=None)


if __name__ == "__main__":
    main()
