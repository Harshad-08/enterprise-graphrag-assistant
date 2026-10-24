"""Evidence sufficiency assessment."""

from __future__ import annotations

from app.config import PROJECT_ROOT
from app.graph.entity_extraction import Gazetteer, analyze_query
from app.models.schemas import Chunk, Claim, DocumentMetadata, Polarity, RetrievedChunk
from app.retrieval.evidence_assessment import assess_evidence
from tests.conftest import make_settings

GAZ = Gazetteer.from_ontology(PROJECT_ROOT / "data" / "ontology.json")
META = DocumentMetadata(document_id="D-1", title="Doc")


def _evidence(text: str, relevance: float, cid: str = "D-1::c000") -> RetrievedChunk:
    chunk = Chunk(chunk_id=cid, document_id="D-1", text=text, chunk_index=0, content_hash="h")
    return RetrievedChunk(chunk=chunk, document=META, rerank_score=relevance)


def _claim(text: str, cid: str = "D-1::c000") -> Claim:
    return Claim(claim_id=f"{cid}::k00", document_id="D-1", chunk_id=cid, text=text, polarity=Polarity.REQUIRE)


SETTINGS = make_settings()


def test_sufficient_when_relevant_passage_covers_entities_concepts_and_claims():
    analysis = analyze_query("What is the minimum length for passwords?", GAZ)
    text = "Passwords must contain at least 14 characters."
    result = assess_evidence(analysis, [_evidence(text, 0.8)], [_claim(text)], SETTINGS, GAZ)
    assert result.sufficient
    assert result.entity_coverage == 1.0 and result.concept_coverage == 1.0 and result.claim_support == 1.0


def test_high_similarity_alone_is_not_sufficient():
    """A passage about the right entity that does not address the asked aspect is rejected."""
    analysis = analyze_query("What are the access requirements for the Atlas cluster?", GAZ)
    text = "Atlas is a PostgreSQL cluster with two replicas."
    result = assess_evidence(analysis, [_evidence(text, 0.85)], [], SETTINGS, GAZ)
    assert not result.sufficient
    assert "access" in result.missing_concepts


def test_no_relevant_passage():
    analysis = analyze_query("What is the maternity leave policy?", GAZ)
    result = assess_evidence(analysis, [_evidence("Backups are kept for 35 days.", 0.01)], [], SETTINGS, GAZ)
    assert not result.sufficient and result.relevant_chunk_count == 0


def test_graph_equivalents_count_for_entity_coverage():
    analysis = analyze_query("What are the access requirements for the Atlas cluster?", GAZ)
    text = "Access to a production database requires an approved access request."
    equivalents = {"system:atlas-cluster": {"atlas", "atlas cluster", "production database"}}
    without = assess_evidence(analysis, [_evidence(text, 0.7)], [_claim(text)], SETTINGS, GAZ)
    with_graph = assess_evidence(analysis, [_evidence(text, 0.7)], [_claim(text)], SETTINGS, GAZ, equivalents)
    assert without.missing_entities == ["Atlas Cluster"]
    assert with_graph.missing_entities == [] and with_graph.sufficient


def test_strong_relevance_waives_lexical_concept_gate_only_when_entities_covered():
    # "stop being usable" shares no words (or catalog synonyms) with "expire", so lexical coverage fails.
    analysis = analyze_query("When do access tokens stop being usable?", GAZ)
    text = "Access tokens expire after 60 minutes."
    strong = assess_evidence(analysis, [_evidence(text, 0.97)], [_claim(text)], SETTINGS, GAZ)
    weak = assess_evidence(analysis, [_evidence(text, 0.6)], [_claim(text)], SETTINGS, GAZ)
    assert strong.concept_coverage < SETTINGS.min_concept_coverage
    assert strong.sufficient
    assert not weak.sufficient


def test_threshold_is_configurable():
    analysis = analyze_query("What is the minimum length for passwords?", GAZ)
    text = "Passwords must contain at least 14 characters."
    strict = SETTINGS.model_copy(update={"sufficiency_threshold": 0.99})
    assert not assess_evidence(analysis, [_evidence(text, 0.8)], [_claim(text)], strict, GAZ).sufficient
