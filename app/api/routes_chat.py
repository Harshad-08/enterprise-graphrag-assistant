"""Chat endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import container_dependency
from app.generation.llm_client import LLMError
from app.graph.store import GraphStoreError
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.services.container import Container

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, container: Container = Depends(container_dependency)) -> ChatResponse:
    try:
        return ChatService(container).answer(request)
    except GraphStoreError as exc:
        raise HTTPException(status_code=503, detail=f"Graph database error: {exc}") from exc
    except LLMError as exc:
        logger.error("LLM failure: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc
