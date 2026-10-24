"""End-to-end tests against a real Neo4j instance.

Opt-in because they DELETE ALL DATA in the target database:

    NEO4J_TEST_URI=bolt://localhost:7687 NEO4J_TEST_PASSWORD=... pytest tests/integration/test_neo4j_integration.py

Without NEO4J_TEST_URI the module is skipped with an explicit reason (visible with ``pytest -rs``).
"""

from __future__ import annotations

import os

import pytest

from app.config import PROJECT_ROOT
from app.generation.llm_client import MockLLMClient
from app.models.schemas import ChatRequest, RetrievalOutcome
from app.retrieval.reranking import LexicalReranker
from app.retrieval.vector_search import HashingEmbedder
from app.services.chat_service import ChatService
from app.services.container import build_container
from app.services.document_service import DocumentService
from tests.conftest import make_settings

URI = os.getenv("NEO4J_TEST_URI")
PASSWORD = os.getenv("NEO4J_TEST_PASSWORD")
pytestmark = pytest.mark.skipif(
    not (URI and PASSWORD),
    reason="NEO4J_TEST_URI/NEO4J_TEST_PASSWORD not set (Neo4j integration tests are opt-in; they wipe the database)",
)


@pytest.fixture(scope="module")
def neo4j_container(tmp_path_factory):
    from app.graph.neo4j_client import Neo4jGraphStore

    store = Neo4jGraphStore(URI, os.getenv("NEO4J_TEST_USERNAME", "neo4j"), PASSWORD)
    if not store.verify_connectivity():
        pytest.skip(f"Neo4j not reachable at {URI}")
    store.clear()
    settings = make_settings(tmp_path_factory.mktemp("neo4j"), graph_backend="neo4j")
    container = build_container(settings, store=store, embedder=HashingEmbedder(384), reranker=LexicalReranker(),
                                llm=MockLLMClient(), load_nli=False)
    DocumentService(container).seed_directory(PROJECT_ROOT / "data" / "synthetic_documents")
    yield container
    store.clear()
    store.close()


def test_schema_constraints_and_vector_index(neo4j_container):
    store = neo4j_container.store
    constraints = {r["name"] for r in store._run("SHOW CONSTRAINTS YIELD name RETURN name")}
    assert {"document_id", "chunk_id", "claim_id", "entity_id"} <= constraints
    indexes = {r["name"]: r["type"] for r in store._run("SHOW INDEXES YIELD name, type RETURN name, type")}
    assert indexes.get("chunk_embedding") == "VECTOR"


def test_graph_contents(neo4j_container):
    stats = neo4j_container.store.stats()
    assert stats["documents"] == 13 and stats["chunks"] > 30 and stats["claims"] > 30
    assert stats["supersessions"] == 1
    labels = {r["label"] for r in neo4j_container.store._run("MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label")}
    assert {"Document", "Chunk", "Claim", "Policy", "Department", "System", "Procedure"} <= labels
    rel_types = {r["t"] for r in neo4j_container.store._run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t")}
    assert {"CONTAINS", "MENTIONS", "GOVERNS", "DEPENDS_ON", "OWNED_BY", "RELATED_TO", "APPLIES_TO",
            "SUPERSEDES", "CONTRADICTS", "SUPPORTS", "DEFINES", "REFERS_TO", "EXTRACTED_FROM"} <= rel_types


def test_reingestion_does_not_duplicate(neo4j_container):
    before = neo4j_container.store.stats()
    path = PROJECT_ROOT / "data" / "synthetic_documents" / "SEC-POL-001_information_security_policy.txt"
    DocumentService(neo4j_container).ingest_path(path)
    assert neo4j_container.store.stats()["chunks"] == before["chunks"]
    assert neo4j_container.store.stats()["entities"] == before["entities"]


def test_vector_search_and_traversal(neo4j_container):
    store = neo4j_container.store
    hits = store.vector_search(neo4j_container.embedder.embed(["password characters"])[0], 3)
    assert hits and 0.0 <= hits[0][1] <= 1.0
    paths = store.traverse(["system:atlas-cluster"], max_hops=2)
    described = [p.describe() for p in paths]
    assert any("Atlas Cluster -[RELATED_TO]-> Production Database" in d for d in described)
    assert any("SEC-STD-004" in p.document_ids for p in paths)
    # Departments are not traversed through.
    assert not any("Department" in p.nodes[1:-1] for p in paths)


def test_challenges_end_to_end_on_neo4j(neo4j_container):
    chat = ChatService(neo4j_container)
    multi_hop = chat.answer(ChatRequest(question="Which procedure depends on the API authentication policy?"))
    assert "PROC-PART-003" in {c.document_id for c in multi_hop.citations}
    conflict = chat.answer(ChatRequest(question="What is the inactivity timeout for VPN sessions?"))
    assert conflict.conflict_detected and any(not c.resolved for c in conflict.conflicts)
    absent = chat.answer(ChatRequest(question="What is the company's maternity leave policy?"))
    assert absent.abstained and absent.retrieval_outcome == RetrievalOutcome.NOT_IN_CORPUS.value


def test_delete_removes_graph_vector_and_keyword_entries(neo4j_container):
    service = DocumentService(neo4j_container)
    assert service.delete("SEC-GDL-009")
    store = neo4j_container.store
    assert store.get_document("SEC-GDL-009") is None
    assert not store.chunks_for_documents(["SEC-GDL-009"], 10)
    assert not any(c.document_id == "SEC-GDL-009" for c in store.all_claims())
    leftover = store._run("MATCH ()-[r]->() WHERE 'SEC-GDL-009' IN coalesce(r.source_documents, []) RETURN count(r) AS n")
    assert leftover[0]["n"] == 0
    assert not any(c.document_id == "SEC-GDL-009" for c, _ in neo4j_container.keyword_index.search("VPN inactivity", 20))
    # The unresolved VPN conflict disappears with its source document.
    assert not any("SEC-GDL-009" in (r["document_a"], r["document_b"]) for r in store.claim_relations())
