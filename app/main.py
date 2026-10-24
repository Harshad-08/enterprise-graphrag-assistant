"""FastAPI application entry point: ``uvicorn app.main:app``."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.api import routes_chat, routes_documents, routes_evaluation
from app.api.dependencies import container_dependency
from app.config import configure_logging, get_settings
from app.models.schemas import HealthResponse
from app.services.container import Container, get_container
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    container = get_container()  # load models once at startup, not on the first request
    if settings.seed_on_startup and not container.store.list_documents():
        logger.info("SEED_ON_STARTUP: indexing synthetic corpus")
        DocumentService(container).seed_directory(settings.synthetic_docs_dir)
    yield


app = FastAPI(
    title="Enterprise Knowledge Assistant (Reliability-Focused GraphRAG)",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(routes_documents.router)
app.include_router(routes_chat.router)
app.include_router(routes_evaluation.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health(container: Container = Depends(container_dependency)) -> HealthResponse:
    connected = container.store.verify_connectivity()
    documents = len(container.store.list_documents()) if connected else 0
    return HealthResponse(
        status="ok" if connected else "degraded",
        graph_backend=container.store.backend_name,
        graph_connected=connected,
        llm_provider=container.llm.name,
        mock_mode=container.llm.is_mock,
        embedding_backend=container.embedder.name,
        reranker=container.reranker.name,
        verifier=container.checker.name,
        documents_indexed=documents,
        details=container.store.stats() if connected else {},
    )
