"""FastAPI dependencies and shared error handling."""

from __future__ import annotations

from fastapi import HTTPException

from app.graph.store import GraphStoreError
from app.services.container import Container, get_container


def container_dependency() -> Container:
    try:
        return get_container()
    except GraphStoreError as exc:
        raise HTTPException(status_code=503, detail=f"Graph database unavailable: {exc}") from exc
