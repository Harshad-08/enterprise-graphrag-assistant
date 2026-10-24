"""Evidence-constrained answer generation and one bounded correction attempt."""

from __future__ import annotations

from app.generation.llm_client import LLMClient
from app.generation.prompts import (
    ANSWER_PROMPT,
    BASELINE_PROMPT,
    CORRECTION_PROMPT,
    SYSTEM_RULES,
    AnswerSchema,
    format_evidence,
)
from app.models.schemas import (
    GeneratedAnswer,
    GeneratedClaim,
    RetrievedChunk,
    VerificationReport,
    VerificationStatus,
)


def _to_answer(schema: AnswerSchema) -> GeneratedAnswer:
    return GeneratedAnswer(
        claims=[GeneratedClaim(text=c.text.strip(), citation_ids=[i.strip() for i in c.citation_ids]) for c in schema.claims if c.text.strip()],
        insufficient_evidence=schema.insufficient_evidence,
        notes=schema.notes,
    )


class AnswerGenerator:
    def __init__(self, llm: LLMClient, min_relevance: float | None = None) -> None:
        self.llm = llm
        self.min_relevance = min_relevance

    def generate(
        self,
        question: str,
        evidence: dict[str, RetrievedChunk],
        conflict_notes: list[str],
        extra_terms: list[str] | None = None,
    ) -> GeneratedAnswer:
        prompt = ANSWER_PROMPT.format(
            rules=SYSTEM_RULES,
            question=question,
            evidence=format_evidence(evidence),
            conflicts="\n".join(conflict_notes) or "None.",
        )
        context = {"question": question, "evidence": evidence, "min_relevance": self.min_relevance, "extra_terms": extra_terms or []}
        return _to_answer(self.llm.generate_structured(prompt, AnswerSchema, context=context))

    def correct(
        self,
        question: str,
        evidence: dict[str, RetrievedChunk],
        conflict_notes: list[str],
        report: VerificationReport,
        extra_terms: list[str] | None = None,
    ) -> GeneratedAnswer:
        feedback = "\n".join(
            f"- [{v.status.value}] {v.claim} (cited {', '.join(v.cited_ids) or 'nothing'}){' — ' + v.note if v.note else ''}"
            for v in report.verifications
        )
        prompt = CORRECTION_PROMPT.format(
            rules=SYSTEM_RULES,
            question=question,
            evidence=format_evidence(evidence),
            conflicts="\n".join(conflict_notes) or "None.",
            feedback=feedback,
        )
        context = {
            "question": question,
            "evidence": evidence,
            "min_relevance": self.min_relevance,
            "extra_terms": extra_terms or [],
            "drop_claims": [v.claim for v in report.verifications if v.status != VerificationStatus.SUPPORTED],
        }
        return _to_answer(self.llm.generate_structured(prompt, AnswerSchema, context=context))

    def baseline(self, question: str, evidence: dict[str, RetrievedChunk]) -> GeneratedAnswer:
        prompt = BASELINE_PROMPT.format(question=question, evidence=format_evidence(evidence))
        context = {"question": question, "evidence": evidence, "min_relevance": None}
        return _to_answer(self.llm.generate_structured(prompt, AnswerSchema, context=context))
