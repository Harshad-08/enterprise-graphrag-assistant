"""Shared helpers for scripts: import path setup and container construction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import configure_logging, load_settings  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402


def make_container(**overrides) -> Container:
    settings = load_settings()
    if overrides:
        settings = settings.model_copy(update=overrides)
    configure_logging(settings.log_level)
    return build_container(settings)
