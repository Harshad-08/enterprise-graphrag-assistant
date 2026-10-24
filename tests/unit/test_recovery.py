"""Challenge 1 — a relevant document exists but is not retrieved.

Small controlled corpora with deterministic components (hashing embeddings, lexical
reranker) make each retrieval outcome reproducible.
"""

from __future__ import annotations

from app.models.schemas import ChatRequest, RetrievalOutcome
from app.services.chat_service import ChatService
from tests.conftest import ingest_texts, make_container

ATLAS_CORPUS = {
    "overview.txt": """---
document_id: TECH-ATL
title: Atlas Overview
doc_type: technical
status: active
---
# 1. Overview
Atlas is a production database cluster for orders. Atlas rules for sharding are documented here.

# 2. Snapshots
Atlas rules for snapshots: Atlas snapshots are taken nightly.
""",
    "standard.txt": """---
document_id: SEC-DBA
title: Production Database Access Standard
doc_type: standard
status: active
authority: department
governs: Production Database
---
# 1. Requirements
Access to a production database requires an approved access request.
""",
    "passwords.txt": """---
document_id: SEC-PWD
title: Password Standard
doc_type: standard
status: active
---
# 1. Passwords
Passwords must contain at least 14 characters.
""",
    "remote.txt": """---
document_id: HR-RW
title: Remote Work Program
doc_type: policy
status: active
---
# 1. Schedule
Employees may work up to 3 days per week under the remote work program.
""",
}

ATLAS_QUESTION = "What are the access rules for Atlas?"


def _container(tmp_path, **overrides):
    defaults = {"final_top_k": 2, "min_concept_coverage": 0.6}
    defaults.update(overrides)
    c = make_container(tmp_path, **defaults)
    ingest_texts(c, ATLAS_CORPUS)
    return c


def test_initial_retrieval_succeeds_without_recovery(tmp_path):
    c = _container(tmp_path)
    result = c.orchestrator.retrieve("How many characters must passwords contain?")
    assert result.outcome == RetrievalOutcome.INITIAL_SUFFICIENT
    assert result.recovery_attempts == []
    assert result.evidence[0].chunk.document_id == "SEC-PWD"


def test_graph_recovery_finds_missed_document(tmp_path):
    c = _container(tmp_path)
    result = c.orchestrator.retrieve(ATLAS_QUESTION)
    initial_docs = {cid.split("::")[0] for cid in result.initial_evidence_chunk_ids}
    final_docs = {e.chunk.document_id for e in result.evidence}
    assert not result.initial_assessment.sufficient
    assert "SEC-DBA" not in initial_docs  # missed by initial retrieval
    assert result.outcome == RetrievalOutcome.RECOVERED
    assert "SEC-DBA" in final_docs
    attempt = result.recovery_attempts[0]
    assert any("Production Database" in q for q in attempt.reformulated_queries)
    assert result.final_assessment.sufficient
    # The graph path that justified the recovery is reported.
    assert any("Production Database" in p.describe() for p in result.graph_paths)


def test_query_expansion_recovers_document(tmp_path):
    c = _container(tmp_path)
    result = c.orchestrator.retrieve("How many days can staff telecommute?")
    assert not result.initial_assessment.sufficient
    assert result.outcome == RetrievalOutcome.RECOVERED
    assert result.evidence[0].chunk.document_id == "HR-RW"
    assert any("remote work" in q for q in result.recovery_attempts[0].reformulated_queries)


def test_document_absent_from_corpus(tmp_path):
    c = _container(tmp_path)
    result = c.orchestrator.retrieve("What is the maternity leave policy?")
    assert result.outcome == RetrievalOutcome.NOT_IN_CORPUS
    # "Absent" is only concluded after every recovery attempt failed.
    assert len(result.recovery_attempts) == c.settings.recovery_max_attempts


def test_irrelevant_documents_are_not_used(tmp_path):
    c = _container(tmp_path)
    response = ChatService(c).answer(ChatRequest(question="What is the inactivity timeout for VPN sessions?"))
    assert response.abstained
    assert response.citations == []
    assert response.retrieved_chunk_ids  # something was retrieved, but it was judged irrelevant


def test_documents_exist_but_contain_no_answer(tmp_path):
    c = _container(tmp_path)
    result = c.orchestrator.retrieve("What is the storage capacity of Atlas?")
    assert result.outcome == RetrievalOutcome.DOCUMENTS_WITHOUT_ANSWER


def test_recovery_respects_retry_limit(tmp_path):
    for limit in (1, 3):
        c = _container(tmp_path / str(limit), recovery_max_attempts=limit)
        result = c.orchestrator.retrieve("What is the maternity leave policy?")
        assert len(result.recovery_attempts) == limit
    disabled = _container(tmp_path / "off", recovery_enabled=False)
    result = disabled.orchestrator.retrieve(ATLAS_QUESTION)
    assert result.outcome == RetrievalOutcome.RECOVERY_DISABLED and result.recovery_attempts == []


def test_evidence_still_insufficient_after_recovery_leads_to_abstention(tmp_path):
    c = _container(tmp_path)
    response = ChatService(c).answer(ChatRequest(question="What is the storage capacity of Atlas?"))
    assert response.abstained and response.recovery_performed
    assert response.retrieval_outcome == "documents_without_answer"
    assert "couldn't verify" in response.answer.lower()
    assert not response.verified


def test_recovered_answer_is_cited_and_verified(tmp_path):
    c = _container(tmp_path)
    response = ChatService(c).answer(ChatRequest(question=ATLAS_QUESTION))
    assert not response.abstained and response.recovery_performed and response.verified
    assert "SEC-DBA" in {cit.document_id for cit in response.citations}
