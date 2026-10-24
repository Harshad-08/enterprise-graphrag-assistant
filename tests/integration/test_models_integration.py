"""Integration with the real local models (sentence-transformers embeddings, ms-marco
cross-encoder, NLI cross-encoder). Skipped with a reason if the models cannot be loaded
(e.g. no network on first run)."""

from __future__ import annotations

import pytest

from app.config import PROJECT_ROOT
from app.models.schemas import ChatRequest, RetrievalOutcome
from app.reliability.answer_verification import NLISupportChecker
from app.retrieval.reranking import CrossEncoderReranker
from app.retrieval.vector_search import SentenceTransformerEmbedder
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from tests.conftest import make_settings


@pytest.fixture(scope="module")
def models():
    try:
        from app.reliability.nli import NLIModel

        settings = make_settings()
        return {
            "embedder": SentenceTransformerEmbedder(settings.embedding_model),
            "reranker": CrossEncoderReranker(settings.reranker_model),
            "nli": NLIModel(settings.nli_model),
        }
    except Exception as exc:  # download or load failure
        pytest.skip(f"Local models unavailable: {exc}")


def test_cross_encoder_ranks_relevant_passage_first(models):
    scores = models["reranker"].score("How long are application logs retained?", [
        "Application logs must be retained for 12 months.", "Passwords must contain at least 14 characters."])
    assert scores[0] > 0.5 > scores[1]


def test_nli_checker_supports_entailed_and_rejects_contradicted_claims(models):
    checker = NLISupportChecker(models["nli"], entailment_threshold=0.5, contradiction_threshold=0.7)
    passage = "VPN sessions must be disconnected after 15 minutes of inactivity. Split tunnelling is not permitted."
    assert checker.check("VPN sessions must be disconnected after 15 minutes of inactivity.", passage)[0] == "supported"
    assert checker.check("VPN sessions stay connected indefinitely when inactive.", passage)[0] in ("contradicted", "unsupported")
    assert checker.check("VPN sessions are disconnected after 60 minutes of inactivity.", passage)[0] != "supported"


def test_full_pipeline_with_real_models(models, tmp_path):
    from app.generation.llm_client import MockLLMClient
    from app.graph.memory_store import InMemoryGraphStore
    from app.services.container import build_container

    settings = make_settings(tmp_path, embedding_backend="sentence-transformers", reranker_enabled=True,
                             verifier_backend="nli", min_chunk_relevance=0.25, sufficiency_threshold=0.55,
                             final_top_k=6)
    container = build_container(settings, store=InMemoryGraphStore(), embedder=models["embedder"],
                                reranker=models["reranker"], nli=models["nli"], llm=MockLLMClient())
    DocumentService(container).seed_directory(PROJECT_ROOT / "data" / "synthetic_documents")
    chat = ChatService(container)

    recovered = chat.answer(ChatRequest(question="What are the access requirements for the Atlas cluster?"))
    assert recovered.retrieval_outcome == RetrievalOutcome.RECOVERED.value and recovered.recovery_performed
    assert "SEC-STD-004" in {c.document_id for c in recovered.citations}

    resolved = chat.answer(ChatRequest(question="How many days per week can employees work from home?"))
    assert any(c.resolved and c.applicable_document == "HR-POL-007" for c in resolved.conflicts)
    assert "2 days per week" not in resolved.answer

    absent = chat.answer(ChatRequest(question="What is the company's maternity leave policy?"))
    assert absent.abstained and absent.retrieval_outcome == RetrievalOutcome.NOT_IN_CORPUS.value


def test_self_referencing_passage_supports_claim_naming_the_document(models):
    """Regression: 'This procedure depends on X' must support '<Procedure title> depends on X',
    and a borderline score on an unrelated passage must not steal the citation."""
    from app.models.schemas import Chunk, DocumentMetadata, GeneratedAnswer, GeneratedClaim, RetrievedChunk
    from app.reliability.answer_verification import verify_answer

    def item(doc_id, title, text):
        return RetrievedChunk(
            chunk=Chunk(chunk_id=f"{doc_id}::c000", document_id=doc_id, text=text, chunk_index=0, content_hash="h"),
            document=DocumentMetadata(document_id=doc_id, title=title),
        )

    evidence = {
        "E1": item("PROC-PART-003", "Partner Integration Onboarding Procedure",
                   "This procedure describes how a new external partner is connected to the Public API Gateway. "
                   "This procedure depends on the API Authentication Standard."),
        "E2": item("TECH-API-002", "API Authentication Standard",
                   "Access tokens expire after 60 minutes. Static API keys are prohibited for new integrations."),
    }
    checker = NLISupportChecker(models["nli"], entailment_threshold=0.5, contradiction_threshold=0.7)
    claim = GeneratedClaim(text="Partner Integration Onboarding Procedure depends on the API Authentication Standard.",
                           citation_ids=["E1"])
    report = verify_answer(GeneratedAnswer(claims=[claim]), evidence, checker)
    assert report.verifications[0].status.value == "supported"
    assert report.verifications[0].supporting_ids == ["E1"]
    miscited = GeneratedClaim(text=claim.text, citation_ids=["E2"])
    report = verify_answer(GeneratedAnswer(claims=[miscited]), evidence, checker)
    assert "E2" not in report.verifications[0].supporting_ids
