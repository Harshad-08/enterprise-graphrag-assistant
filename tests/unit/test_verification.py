"""Challenge 3 — the LLM generates unsupported answers."""

from __future__ import annotations

from app.models.schemas import (
    ChatRequest,
    Chunk,
    Citation,
    DocumentMetadata,
    GeneratedAnswer,
    GeneratedClaim,
    RetrievedChunk,
    VerificationStatus,
)
from app.reliability.answer_verification import LexicalSupportChecker, repair_citations, verify_answer
from app.reliability.citations import best_excerpt, validate_citations
from app.services.chat_service import ChatService
from tests.conftest import ScriptedLLM, ingest_texts, make_container

CORPUS = {
    "pwd.txt": "---\ndocument_id: SEC-PWD\ntitle: Password Standard\ndoc_type: standard\nstatus: active\n---\n"
               "# 1. Passwords\nPasswords must contain at least 14 characters. Passwords must be stored hashed.",
    "bkp.txt": "---\ndocument_id: OPS-BKP\ntitle: Backup Guide\ndoc_type: technical\nstatus: active\n---\n"
               "# 1. Backups\nDatabase backups are kept for 35 days.",
    "vpn.txt": "---\ndocument_id: NET-VPN\ntitle: VPN Guide\ndoc_type: technical\nstatus: active\n---\n"
               "# 1. Sessions\nVPN sessions are disconnected after 15 minutes of inactivity.",
}
QUESTION = "How many characters must passwords contain?"
SUPPORTED = "Passwords must contain at least 14 characters."


def label_for(context: dict, needle: str) -> str:
    for label, item in context["evidence"].items():
        if needle in item.chunk.text:
            return label
    raise AssertionError(f"No evidence passage contains {needle!r}")


def chat_with(tmp_path, answers) -> tuple[ChatService, ScriptedLLM]:
    llm = ScriptedLLM(answers)
    c = make_container(tmp_path, llm=llm, final_top_k=3)
    ingest_texts(c, CORPUS)
    return ChatService(c), llm


def test_fully_supported_answer(tmp_path):
    chat, llm = chat_with(tmp_path, lambda ctx: {"claims": [{"text": SUPPORTED, "citation_ids": [label_for(ctx, "14 characters")]}]})
    response = chat.answer(ChatRequest(question=QUESTION))
    assert response.verified and not response.abstained
    assert len(llm.calls) == 1  # no correction needed
    citation = response.citations[0]
    assert citation.document_id == "SEC-PWD" and citation.chunk_id.startswith("SEC-PWD::")
    assert citation.page is None  # TXT has no pages; none is fabricated
    assert citation.excerpt in chat.c.store.get_chunks([citation.chunk_id])[0].text
    assert "[1]" in response.answer


def test_unsupported_claim_is_corrected_once(tmp_path):
    def answers(ctx):
        label = label_for(ctx, "14 characters")
        if "drop_claims" in ctx:  # correction round: the model drops the unsupported claim
            return {"claims": [{"text": SUPPORTED, "citation_ids": [label]}]}
        return {"claims": [{"text": SUPPORTED, "citation_ids": [label]},
                           {"text": "Passwords must be rotated every 30 days.", "citation_ids": [label]}]}

    chat, llm = chat_with(tmp_path, answers)
    response = chat.answer(ChatRequest(question=QUESTION))
    assert len(llm.calls) == 2
    assert "30 days" not in response.answer and SUPPORTED in response.answer
    assert response.verified
    assert any("correction attempt" in n for n in response.reliability_notes)


def test_unsupported_claim_removed_when_correction_does_not_fix_it(tmp_path):
    def answers(ctx):
        label = label_for(ctx, "14 characters")
        return {"claims": [{"text": SUPPORTED, "citation_ids": [label]},
                           {"text": "Passwords must be rotated every 30 days.", "citation_ids": [label]}]}

    chat, llm = chat_with(tmp_path, answers)
    response = chat.answer(ChatRequest(question=QUESTION))
    assert len(llm.calls) == 2  # bounded: one correction attempt only
    assert "30 days" not in response.answer
    assert any("removed" in n for n in response.reliability_notes)


def test_invalid_citation_id(tmp_path):
    chat, _ = chat_with(tmp_path, [{"claims": [{"text": "Passwords never expire.", "citation_ids": ["E99"]}]}])
    response = chat.answer(ChatRequest(question=QUESTION))
    statuses = [v.status for v in response.claim_verifications]
    assert VerificationStatus.INVALID_CITATION in statuses
    assert response.abstained  # nothing verifiable remains
    assert all(c.chunk_id for c in response.citations)


def test_supported_claim_with_invalid_label_is_reattributed_to_real_passage(tmp_path):
    chat, _ = chat_with(tmp_path, [{"claims": [{"text": SUPPORTED, "citation_ids": ["E99"]}]}])
    response = chat.answer(ChatRequest(question=QUESTION))
    assert not response.abstained
    assert [c.document_id for c in response.citations] == ["SEC-PWD"]
    assert any("re-attributed" in n for n in response.reliability_notes)


def test_citation_pointing_to_unrelated_passage_is_repaired(tmp_path):
    chat, _ = chat_with(tmp_path, lambda ctx: {"claims": [{"text": SUPPORTED, "citation_ids": [label_for(ctx, "backups")]}]})
    response = chat.answer(ChatRequest(question="How many characters must passwords contain? backups"))
    assert [c.document_id for c in response.citations] == ["SEC-PWD"]  # not the backup guide


def test_contradicted_claim_is_rejected(tmp_path):
    chat, _ = chat_with(tmp_path, lambda ctx: {"claims": [
        {"text": "Passwords must not contain at least 14 characters.", "citation_ids": [label_for(ctx, "14 characters")]}]})
    response = chat.answer(ChatRequest(question=QUESTION))
    assert response.claim_verifications[0].status == VerificationStatus.CONTRADICTED
    assert response.abstained


def test_unanswerable_question_abstains_without_calling_llm(tmp_path):
    chat, llm = chat_with(tmp_path, [{"claims": [{"text": "Maternity leave is 26 weeks.", "citation_ids": ["E1"]}]}])
    response = chat.answer(ChatRequest(question="What is the company's maternity leave policy?"))
    assert response.abstained and llm.calls == []
    assert "couldn't verify" in response.answer.lower()
    assert "maternity leave" in response.answer.lower()
    assert response.citations == []


def test_failed_correction_leads_to_abstention(tmp_path):
    bad = lambda ctx: {"claims": [{"text": "Passwords must be rotated every 30 days.", "citation_ids": [label_for(ctx, "14 characters")]}]}  # noqa: E731
    chat, llm = chat_with(tmp_path, bad)
    response = chat.answer(ChatRequest(question=QUESTION))
    assert len(llm.calls) == 2
    assert response.abstained and not response.verified
    assert "not answering" in response.answer


def test_baseline_mode_has_no_verification(tmp_path):
    chat, _ = chat_with(tmp_path, [{"claims": [{"text": "Passwords must be rotated every 30 days.", "citation_ids": ["E1"]}]}])
    response = chat.answer(ChatRequest(question=QUESTION, mode="baseline"))
    assert not response.verified and not response.abstained
    assert "30 days" in response.answer  # the unsupported claim reaches the user


class TestVerificationUnits:
    meta = DocumentMetadata(document_id="SEC-PWD", title="Password Standard")
    chunk = Chunk(chunk_id="SEC-PWD::c000", document_id="SEC-PWD", text="Passwords must contain at least 14 characters.",
                  chunk_index=0, content_hash="h", page=2)
    evidence = {"E1": RetrievedChunk(chunk=chunk, document=meta)}

    def test_number_mismatch_is_unsupported(self):
        report = verify_answer(GeneratedAnswer(claims=[GeneratedClaim(text="Passwords must contain at least 12 characters.", citation_ids=["E1"])]),
                               self.evidence, LexicalSupportChecker(0.6))
        assert report.verifications[0].status == VerificationStatus.UNSUPPORTED

    def test_repair_only_applies_to_verifiably_supported_claims(self):
        report = verify_answer(GeneratedAnswer(claims=[GeneratedClaim(text="Unrelated statement about cats.", citation_ids=[])]),
                               self.evidence, LexicalSupportChecker(0.6))
        repaired, count = repair_citations(report)
        assert count == 0 and repaired.verifications[0].status == VerificationStatus.UNSUPPORTED

    def test_citation_validation_against_store(self, tmp_path):
        c = make_container(tmp_path)
        ingest_texts(c, CORPUS)
        real = c.store.all_chunks()[0]
        good = Citation(citation_id=1, document_id=real.document_id, document_title="t", chunk_id=real.chunk_id,
                        excerpt=best_excerpt("x", real.text))
        missing = good.model_copy(update={"citation_id": 2, "chunk_id": "NOPE::c000"})
        fabricated_page = good.model_copy(update={"citation_id": 3, "page": 7})
        altered_excerpt = good.model_copy(update={"citation_id": 4, "excerpt": "Text that is not in the source."})
        valid, problems = validate_citations([good, missing, fabricated_page, altered_excerpt], c.store)
        assert [v.citation_id for v in valid] == [1]
        assert len(problems) == 3
