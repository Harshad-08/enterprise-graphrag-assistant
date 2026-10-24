"""Chat orchestration for the improved GraphRAG pipeline and the baseline RAG."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    ClaimVerification,
    ConflictSummary,
    GeneratedClaim,
    RetrievalOutcome,
    RetrievalResult,
    RetrievedChunk,
    VerificationReport,
    VerificationStatus,
)
from app.reliability.abstention import abstention_message, should_abstain
from app.reliability.answer_verification import repair_citations, verify_answer
from app.reliability.citations import best_excerpt, build_citations, citation_label_map, validate_citations
from app.reliability.conflict_analysis import ConflictAnalysis, analyze_conflicts
from app.retrieval.reranking import rerank
from app.services.container import Container
from app.reliability.claim_extraction import detect_polarity
from app.utils.text import content_terms, numbers_in

logger = logging.getLogger(__name__)

MOCK_PREFIX = "[MOCK MODE: extractive answer assembled from evidence sentences; no LLM reasoning] "

_OUTCOME_NOTES = {
    RetrievalOutcome.INITIAL_SUFFICIENT: "Initial hybrid retrieval found sufficient evidence.",
    RetrievalOutcome.RECOVERED: "Initial retrieval was insufficient; graph-based recovery found the missing evidence.",
    RetrievalOutcome.DOCUMENTS_WITHOUT_ANSWER: "Documents about this topic exist, but none contains the requested answer.",
    RetrievalOutcome.NOT_IN_CORPUS: "No indexed document appears to cover this topic (checked after recovery).",
    RetrievalOutcome.INSUFFICIENT_AFTER_RECOVERY: "Evidence remained insufficient after the recovery limit was reached.",
    RetrievalOutcome.RECOVERY_DISABLED: "Evidence was insufficient and recovery is disabled.",
}


@dataclass
class ChatTrace:
    """Internal record used by evaluation (never returned by the API)."""

    evidence: dict[str, RetrievedChunk] = field(default_factory=dict)
    final_claims: list[GeneratedClaim] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    conflicts: ConflictAnalysis | None = None
    initial_report: VerificationReport | None = None
    correction_attempted: bool = False


def _restates(answer_claim: str, rule: str, threshold: float = 0.8) -> bool:
    """True when an answer claim restates a given rule: same numbers, same deontic polarity and
    high term overlap. ("3 days per week" does not restate "2 days per week".)"""
    if numbers_in(answer_claim) != numbers_in(rule) or detect_polarity(answer_claim) != detect_polarity(rule):
        return False
    ta, tb = content_terms(answer_claim), content_terms(rule)
    return bool(ta) and len(ta & tb) / len(ta) >= threshold


class ChatService:
    def __init__(self, container: Container) -> None:
        self.c = container

    def answer(self, request: ChatRequest) -> ChatResponse:
        return self.answer_with_trace(request)[0]

    def answer_with_trace(self, request: ChatRequest) -> tuple[ChatResponse, ChatTrace]:
        start = time.perf_counter()
        if request.mode == "baseline":
            response, trace = self._baseline(request)
        else:
            response, trace = self._improved(request)
        response.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return response, trace

    # ------------------------------------------------------------------------------
    def _filters(self, request: ChatRequest) -> dict:
        return {"department": request.department, "doc_type": request.doc_type}

    def _base_response(self, request: ChatRequest, evidence: list[RetrievedChunk]) -> dict:
        return {
            "llm_provider": self.c.llm.name,
            "mock_mode": self.c.llm.is_mock,
            "mode": request.mode,
            "retrieved_chunk_ids": [e.chunk_id for e in evidence],
            "retrieved_contexts": [e.chunk.text for e in evidence],
        }

    def _baseline(self, request: ChatRequest) -> tuple[ChatResponse, ChatTrace]:
        """Plain vector RAG: top-k similarity search + one generation call, no checks."""
        evidence = self.c.hybrid.vector_only(request.question, request.top_k or self.c.settings.final_top_k, self._filters(request))
        evidence_map = citation_label_map(evidence)
        generated = self.c.generator.baseline(request.question, evidence_map)
        citations: list[Citation] = []
        numbering: dict[str, int] = {}
        parts = []
        for claim in generated.claims:
            markers = []
            for label in claim.citation_ids:
                item = evidence_map.get(label)
                if item is None:
                    continue  # baseline simply drops unknown labels from the display
                if label not in numbering:
                    numbering[label] = len(citations) + 1
                    citations.append(self._citation(numbering[label], item, claim.text))
                markers.append(f"[{numbering[label]}]")
            parts.append(f"{claim.text} {''.join(markers)}".strip())
        abstained = not generated.claims
        text = " ".join(parts) if parts else "The context does not contain an answer."
        response = ChatResponse(
            answer=(MOCK_PREFIX if self.c.llm.is_mock else "") + text,
            citations=citations,
            supporting_chunk_ids=[c.chunk_id for c in citations],
            abstained=abstained,
            reliability_notes=["Baseline mode: vector search only; no recovery, conflict handling or verification."],
            **self._base_response(request, evidence),
        )
        return response, ChatTrace(evidence=evidence_map, final_claims=generated.claims)

    @staticmethod
    def _citation(number: int, item: RetrievedChunk, claim: str) -> Citation:
        return Citation(
            citation_id=number,
            document_id=item.document.document_id,
            document_title=item.document.title,
            chunk_id=item.chunk.chunk_id,
            page=item.chunk.page,
            section=item.chunk.section,
            excerpt=best_excerpt(claim, item.chunk.text),
            document_status=item.document.status.value,
            version=item.document.version,
        )

    # ------------------------------------------------------------------------------
    def _improved(self, request: ChatRequest) -> tuple[ChatResponse, ChatTrace]:
        c = self.c
        retrieval = c.orchestrator.retrieve(request.question, self._filters(request), request.top_k)
        trace = ChatTrace(retrieval=retrieval)
        notes = [_OUTCOME_NOTES[retrieval.outcome]]
        if retrieval.recovery_triggered:
            for attempt in retrieval.recovery_attempts:
                docs = ", ".join(attempt.graph_documents[:5]) or "none"
                notes.append(
                    f"Recovery attempt {attempt.attempt}: graph traversal reached documents [{docs}]; "
                    f"evidence score {attempt.assessment.score:.2f} ({'sufficient' if attempt.assessment.sufficient else 'insufficient'})."
                )
        graph_paths = [p.describe() for p in retrieval.graph_paths[:5]]
        base = self._base_response(request, retrieval.evidence)
        common = {
            "recovery_performed": retrieval.recovery_triggered,
            "retrieval_outcome": retrieval.outcome.value,
            "graph_paths": graph_paths,
            **base,
        }

        if not retrieval.final_assessment.sufficient:
            notes += retrieval.final_assessment.reasons[:2]
            message = abstention_message(retrieval.outcome, retrieval.analysis, "evidence")
            return ChatResponse(answer=message, abstained=True, reliability_notes=notes, **common), trace

        # Conflicts ------------------------------------------------------------------
        seeds = c.graph.link_entities(retrieval.analysis)
        equivalents = c.graph.equivalents(seeds)
        conflicts = analyze_conflicts(retrieval.analysis, retrieval.evidence, c.store, c.detector, c.settings, equivalents)
        trace.conflicts = conflicts
        evidence = list(retrieval.evidence)
        if conflicts.extra_chunk_ids:
            extra = c.hybrid.to_retrieved(c.store.get_chunks(conflicts.extra_chunk_ids), "conflict")
            evidence += rerank(c.reranker, request.question, extra)
        # Multi-hop grounding: when evidence was reached through a RELATED_TO hop (Atlas ->
        # Production Database), include the passage that states that relationship so the
        # answer can cite each hop separately instead of fusing them into one unverifiable claim.
        neighbours = c.graph.neighbours(seeds)
        hop_chunks = self._relationship_passages(seeds, neighbours, {e.chunk_id for e in evidence})
        if hop_chunks:
            evidence += rerank(c.reranker, request.question, c.hybrid.to_retrieved(hop_chunks, "graph"))
        evidence_map = citation_label_map(evidence)
        trace.evidence = evidence_map
        label_of_chunk = {item.chunk_id: label for label, item in evidence_map.items()}
        conflict_notes = self._conflict_notes(conflicts) + self._graph_notes(seeds, neighbours, evidence_map)

        # Generation, verification and one bounded correction ------------------------------
        extra_terms = sorted({t for e in retrieval.analysis.entities for t in content_terms(e.name)})
        generated = c.generator.generate(request.question, evidence_map, conflict_notes, extra_terms)
        report, repaired = repair_citations(verify_answer(generated, evidence_map, c.checker, c.settings.citation_repair_threshold))
        trace.initial_report = report
        if repaired:
            notes.append(f"{repaired} citation(s) were re-attributed to the passage that actually supports the claim.")
        if not report.all_supported and generated.claims and c.settings.max_correction_attempts > 0:
            trace.correction_attempted = True
            unsupported_before = len(report.verifications) - len(report.supported)
            corrected = c.generator.correct(request.question, evidence_map, conflict_notes, report, extra_terms)
            corrected_report, _ = repair_citations(verify_answer(corrected, evidence_map, c.checker, c.settings.citation_repair_threshold))
            notes.append(
                f"{unsupported_before} unsupported claim(s) detected; one correction attempt made "
                f"({len(corrected_report.verifications) - len(corrected_report.supported)} still unsupported after correction)."
            )
            # A correction must not lower quality: keep whichever attempt has more verified claims.
            if len(corrected_report.supported) >= len(report.supported):
                report = corrected_report
            else:
                notes.append("The correction verified fewer claims than the original answer, so the original verified claims were kept.")

        # Drop claims that restate a rule overridden by conflict resolution.
        final: list[ClaimVerification] = []
        dropped_superseded = 0
        for verification in report.verifications:
            if verification.status != VerificationStatus.SUPPORTED:
                continue
            if any(_restates(verification.claim, loser.text) for loser in conflicts.losing_claims):
                dropped_superseded += 1
                continue
            final.append(verification)
        removed = len(report.verifications) - len(report.supported)
        if removed:
            notes.append(f"{removed} claim(s) failed verification and were removed from the answer.")
        if dropped_superseded:
            notes.append(f"{dropped_superseded} claim(s) restating a superseded/overridden rule were removed.")

        # Deterministic disclosure for unresolved conflicts: both claims, verbatim, cited.
        disclosure: list[ClaimVerification] = []
        for report_item in conflicts.unresolved:
            for claim in (report_item.comparison.claim_a, report_item.comparison.claim_b):
                label = label_of_chunk.get(claim.chunk_id)
                if label:
                    disclosure.append(
                        ClaimVerification(
                            claim=claim.text, status=VerificationStatus.SUPPORTED, cited_ids=[label],
                            valid_citation_ids=[label], supporting_ids=[label], support_score=1.0, method="provenance",
                        )
                    )
        final_report = VerificationReport(verifications=final + disclosure, method=report.method)
        abstain, reason = should_abstain(True, final_report)
        trace.final_claims = [GeneratedClaim(text=v.claim, citation_ids=v.supporting_ids) for v in final_report.verifications]

        conflict_summaries = self._conflict_summaries(conflicts)
        for comp in conflicts.compatible[:3]:
            notes.append(f"Rules differ by condition (not a contradiction): {comp.explanation}")
        if abstain:
            message = abstention_message(retrieval.outcome, retrieval.analysis, reason)
            return ChatResponse(
                answer=message, abstained=True, conflict_detected=bool(conflicts.conflicts), conflicts=conflict_summaries,
                reliability_notes=notes, claim_verifications=report.verifications, **common,
            ), trace

        citations, numbering = build_citations(final_report.verifications, evidence_map)
        citations, citation_problems = validate_citations(citations, c.store)
        notes += citation_problems
        valid_numbers = {cit.citation_id for cit in citations}
        text = self._compose(final, disclosure, conflicts, numbering, valid_numbers)
        if c.llm.is_mock:
            notes.append("Mock LLM mode: the answer is extractive and verification used the "
                         f"'{report.method}' checker.")
        response = ChatResponse(
            answer=(MOCK_PREFIX if c.llm.is_mock else "") + text,
            citations=citations,
            supporting_chunk_ids=[cit.chunk_id for cit in citations],
            conflict_detected=bool(conflicts.conflicts),
            conflicts=conflict_summaries,
            verified=bool(final_report.verifications) and not citation_problems,
            abstained=False,
            reliability_notes=notes,
            claim_verifications=report.verifications,
            **common,
        )
        return response, trace

    # ------------------------------------------------------------------------------
    def _relationship_passages(self, seeds, neighbours, present: set[str]) -> list:
        """Stored passages that state a seed's RELATED_TO link (provenance of the graph edge)."""
        chunks = []
        for seed in seeds:
            linked = neighbours.get(seed.entity_id, [])
            if not linked:
                continue
            seed_names = {seed.name.lower(), *[a.lower() for a in seed.aliases]}
            for chunk in self.c.store.relation_chunks([seed.entity_id], 20):
                text = chunk.text.lower()
                mentions_seed = any(n in text for n in seed_names)
                mentions_neighbour = any(
                    n in text for e in linked for n in {e.name.lower(), *[a.lower() for a in e.aliases]}
                )
                if mentions_seed and mentions_neighbour and chunk.chunk_id not in present:
                    chunks.append(chunk)
                    present.add(chunk.chunk_id)
        return chunks[:3]

    @staticmethod
    def _graph_notes(seeds, neighbours, evidence_map: dict[str, RetrievedChunk]) -> list[str]:
        notes = []
        for seed in seeds:
            for neighbour in neighbours.get(seed.entity_id, []):
                names = {neighbour.name.lower(), *[a.lower() for a in neighbour.aliases]}
                seed_names = {seed.name.lower(), *[a.lower() for a in seed.aliases]}
                labels = [
                    label for label, item in evidence_map.items()
                    if any(n in item.chunk.text.lower() for n in names) and any(n in item.chunk.text.lower() for n in seed_names)
                ]
                if labels:
                    notes.append(
                        f"GRAPH: {seed.name} RELATED_TO {neighbour.name}; the relationship is stated in {', '.join(labels[:2])}. "
                        f"Rules written for {neighbour.name} may apply to {seed.name}; cite each separately."
                    )
        return notes

    @staticmethod
    def _conflict_notes(conflicts: ConflictAnalysis) -> list[str]:
        notes = []
        for report in conflicts.conflicts:
            a, b = report.comparison.claim_a, report.comparison.claim_b
            if report.resolution.resolved:
                winner = a if report.resolution.winner_claim_id == a.claim_id else b
                loser = b if winner is a else a
                notes.append(
                    f"RESOLVED: '{loser.text}' ({loser.document_id}) conflicts with '{winner.text}' ({winner.document_id}). "
                    f"Applicable rule: {winner.document_id} ({report.resolution.explanation}). Do not present the other rule as current."
                )
            else:
                notes.append(
                    f"UNRESOLVED: '{a.text}' ({a.document_id}) vs '{b.text}' ({b.document_id}). "
                    "Present both rules with citations; do not choose one."
                )
        return notes

    @staticmethod
    def _conflict_summaries(conflicts: ConflictAnalysis) -> list[ConflictSummary]:
        summaries = []
        for report in conflicts.conflicts:
            a, b = report.comparison.claim_a, report.comparison.claim_b
            summaries.append(
                ConflictSummary(
                    subject=a.subject, property=a.property,
                    claim_a=a.text, claim_a_document=f"{a.document_title} ({a.document_id})",
                    claim_b=b.text, claim_b_document=f"{b.document_title} ({b.document_id})",
                    resolved=report.resolution.resolved,
                    applicable_document=report.resolution.winner_document_id,
                    rule=report.resolution.rule,
                    explanation=report.resolution.explanation,
                )
            )
        return summaries

    @staticmethod
    def _compose(final, disclosure, conflicts: ConflictAnalysis, numbering: dict[str, int], valid_numbers: set[int]) -> str:
        def markers(v: ClaimVerification) -> str:
            nums = sorted({numbering[l] for l in v.supporting_ids if l in numbering and numbering[l] in valid_numbers})
            return "".join(f"[{n}]" for n in nums)

        lines = []
        disclosed = {d.claim for d in disclosure}
        body = [f"{v.claim} {markers(v)}".strip() for v in final if v.claim not in disclosed]
        if body:
            lines.append(" ".join(body))
        if disclosure:
            lines.append("⚠ Unresolved conflict between sources — the documents disagree and precedence could not be established:")
            for v in disclosure:
                lines.append(f"• {v.claim} {markers(v)}".rstrip())
        for report in conflicts.conflicts:
            if report.resolution.resolved:
                topic = report.comparison.claim_a.subject or "this topic"
                prop = (report.comparison.claim_a.property or "rule").replace("_", " ")
                lines.append(
                    f"Note: documents disagree on {topic} ({prop}); {report.resolution.explanation} "
                    f"The applicable document is {report.resolution.winner_document_id}."
                )
        return "\n".join(lines)

