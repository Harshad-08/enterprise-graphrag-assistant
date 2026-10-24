"""Builds and wires all components once per process (simple manual dependency injection)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.generation.answer_generator import AnswerGenerator
from app.generation.llm_client import LLMClient, build_llm
from app.graph.entity_extraction import Gazetteer
from app.graph.memory_store import InMemoryGraphStore
from app.graph.store import GraphStore
from app.ingestion.pipeline import IngestionPipeline
from app.reliability.answer_verification import SupportChecker, build_support_checker
from app.reliability.contradiction_detection import ContradictionDetector
from app.reliability.nli import NLIModel, build_nli
from app.retrieval.graph_retrieval import GraphRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.keyword_search import BM25Index
from app.retrieval.recovery import RetrievalOrchestrator
from app.retrieval.reranking import Reranker, build_reranker
from app.retrieval.vector_search import Embedder, VectorSearcher, build_embedder

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    store: GraphStore
    embedder: Embedder
    reranker: Reranker
    nli: NLIModel | None
    llm: LLMClient
    gazetteer: Gazetteer
    keyword_index: BM25Index
    hybrid: HybridRetriever
    graph: GraphRetriever
    orchestrator: RetrievalOrchestrator
    detector: ContradictionDetector
    checker: SupportChecker
    generator: AnswerGenerator
    ingestion: IngestionPipeline


def build_store(settings: Settings) -> GraphStore:
    if settings.graph_backend == "memory":
        return InMemoryGraphStore()
    if settings.graph_backend == "neo4j":
        from app.graph.neo4j_client import Neo4jGraphStore

        return Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password, settings.neo4j_database)
    raise ValueError(f"Unknown GRAPH_BACKEND '{settings.graph_backend}'")


def build_container(
    settings: Settings | None = None,
    *,
    store: GraphStore | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    nli: NLIModel | None = None,
    llm: LLMClient | None = None,
    load_nli: bool = True,
) -> Container:
    settings = settings or get_settings()
    store = store or build_store(settings)
    embedder = embedder or build_embedder(settings.embedding_backend, settings.embedding_model, settings.embedding_dimension)
    reranker = reranker or build_reranker(settings.reranker_enabled, settings.reranker_model)
    if nli is None and load_nli and settings.verifier_backend == "nli":
        nli = build_nli(settings.nli_model)
    llm = llm or build_llm(settings)

    if store.verify_connectivity():
        store.setup_schema(embedder.dimension)
    else:
        logger.error("Graph store %s is not reachable; requests will fail until it is.", store.backend_name)

    gazetteer = Gazetteer.from_ontology(settings.ontology_path)
    keyword_index = BM25Index()
    vector = VectorSearcher(store, embedder)
    hybrid = HybridRetriever(store, vector, keyword_index, reranker, settings)
    graph = GraphRetriever(store, gazetteer)
    orchestrator = RetrievalOrchestrator(store, hybrid, graph, gazetteer, settings)
    detector = ContradictionDetector(settings, embedder=embedder, nli=nli)
    checker = build_support_checker(settings, nli, llm)
    generator = AnswerGenerator(llm, min_relevance=settings.min_chunk_relevance)
    ingestion = IngestionPipeline(store, embedder, gazetteer, detector, keyword_index, settings, llm=llm)
    try:
        ingestion.refresh_indexes()
    except Exception as exc:  # store unreachable at startup
        logger.error("Could not build keyword index at startup: %s", exc)
    return Container(
        settings=settings, store=store, embedder=embedder, reranker=reranker, nli=nli, llm=llm,
        gazetteer=gazetteer, keyword_index=keyword_index, hybrid=hybrid, graph=graph,
        orchestrator=orchestrator, detector=detector, checker=checker, generator=generator, ingestion=ingestion,
    )


_container: Container | None = None
_lock = threading.Lock()


def get_container() -> Container:
    global _container
    with _lock:
        if _container is None:
            _container = build_container()
        return _container


def set_container(container: Container | None) -> None:
    """Used by tests to inject a container built from deterministic components."""
    global _container
    with _lock:
        _container = container
