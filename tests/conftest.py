"""Shared deterministic fixtures. Unit tests never call external APIs or download models."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

# Keep a developer's .env (e.g. a real GEMINI_API_KEY or Neo4j settings) out of unit tests.
os.environ.setdefault("LLM_PROVIDER", "mock")

from app.config import PROJECT_ROOT, Settings  # noqa: E402
from app.generation.llm_client import MockLLMClient  # noqa: E402
from app.generation.prompts import AnswerSchema  # noqa: E402
from app.graph.memory_store import InMemoryGraphStore  # noqa: E402
from app.retrieval.reranking import LexicalReranker  # noqa: E402
from app.retrieval.vector_search import HashingEmbedder  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402


def make_settings(tmp_path: Path | None = None, **overrides) -> Settings:
    """Settings for deterministic tests: in-memory graph, hashing embeddings, lexical
    reranker and verifier, mock LLM, and fixed thresholds (no calibration file)."""
    base = dict(
        llm_provider="mock",
        gemini_api_key=None,
        graph_backend="memory",
        embedding_backend="hashing",
        reranker_enabled=False,
        verifier_backend="lexical",
        sufficiency_threshold=0.5,
        min_chunk_relevance=0.3,
        min_concept_coverage=0.5,
        strong_relevance=0.9,
        recovery_max_attempts=2,
        final_top_k=4,
        ragas_enabled=False,
    )
    if tmp_path is not None:
        base["upload_dir"] = tmp_path / "uploads"
        base["eval_results_dir"] = tmp_path / "results"
    base.update(overrides)
    return Settings(**base)


def make_container(tmp_path: Path | None = None, llm=None, **overrides) -> Container:
    settings = make_settings(tmp_path, **overrides)
    return build_container(
        settings,
        store=InMemoryGraphStore(),
        embedder=HashingEmbedder(settings.embedding_dimension),
        reranker=LexicalReranker(),
        llm=llm or MockLLMClient(),
        load_nli=False,
    )


def ingest_texts(container: Container, documents: dict[str, str]) -> None:
    for filename, text in documents.items():
        container.ingestion.ingest_bytes(filename, text.encode())
    container.ingestion.relink_documents()
    container.ingestion.refresh_conflict_resolutions()


class ScriptedLLM:
    """Returns pre-defined structured answers in order; records prompts and contexts."""

    name = "scripted-test-llm"
    is_mock = False

    def __init__(self, answers: list[dict] | Callable[[dict], dict]) -> None:
        self.answers = answers
        self.calls: list[dict] = []

    def generate_structured(self, prompt, schema, *, context=None):
        self.calls.append({"prompt": prompt, "context": context or {}})
        if schema is not AnswerSchema:
            raise AssertionError(f"Unexpected schema {schema}")
        if callable(self.answers):
            payload = self.answers(context or {})
        else:
            payload = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        return AnswerSchema.model_validate(payload)


@pytest.fixture
def container(tmp_path) -> Container:
    return make_container(tmp_path)


@pytest.fixture(scope="session")
def seeded_container(tmp_path_factory) -> Container:
    """The full synthetic corpus indexed with deterministic components (session-scoped)."""
    c = make_container(tmp_path_factory.mktemp("seeded"))
    DocumentService(c).seed_directory(PROJECT_ROOT / "data" / "synthetic_documents")
    return c
