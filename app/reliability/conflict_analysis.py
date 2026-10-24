"""Query-time conflict analysis over the retrieved evidence.

Conflicts are looked up not only among retrieved passages: for every claim in
the evidence, claims about the same subject/property elsewhere in the graph are
pulled in too, so a contradicting document that retrieval did not return is
still detected and disclosed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.graph.store import GraphStore
from app.models.schemas import (
    Claim,
    ClaimComparison,
    ClaimRelationType,
    ConflictReport,
    QueryAnalysis,
    RetrievedChunk,
)
from app.reliability.conflict_resolution import ConflictResolver
from app.reliability.contradiction_detection import ContradictionDetector
from app.utils.text import content_terms


@dataclass
class ConflictAnalysis:
    conflicts: list[ConflictReport] = field(default_factory=list)
    compatible: list[ClaimComparison] = field(default_factory=list)
    extra_chunk_ids: list[str] = field(default_factory=list)  # counterpart passages to add to evidence

    @property
    def unresolved(self) -> list[ConflictReport]:
        return [c for c in self.conflicts if not c.resolution.resolved]

    @property
    def losing_claims(self) -> list[Claim]:
        losers = []
        for report in self.conflicts:
            if report.resolution.resolved:
                a, b = report.comparison.claim_a, report.comparison.claim_b
                losers.append(b if report.resolution.winner_claim_id == a.claim_id else a)
        return losers


def _relevant_to_query(claim: Claim, analysis: QueryAnalysis, entity_names: set[str]) -> bool:
    if not analysis.entities and not analysis.concepts:
        return True
    if analysis.entities and claim.subject and claim.subject.lower() in entity_names:
        return True
    # A question about a policy concerns every claim made by the documents defining it.
    if claim.document_title and any(
        e.label in ("Policy", "Procedure") and e.name.lower() == claim.document_title.lower() for e in analysis.entities
    ):
        return True
    terms = content_terms(claim.text)
    return bool(set(analysis.concepts) & terms) and (not analysis.entities or any(n in claim.text.lower() for n in entity_names))


def analyze_conflicts(
    analysis: QueryAnalysis,
    evidence: list[RetrievedChunk],
    store: GraphStore,
    detector: ContradictionDetector,
    settings: Settings,
    entity_equivalents: dict[str, set[str]] | None = None,
) -> ConflictAnalysis:
    relevant_chunks = [e.chunk_id for e in evidence if e.relevance >= settings.min_chunk_relevance]
    evidence_claims = store.claims_for_chunks(relevant_chunks)
    entity_names: set[str] = set()
    for names in (entity_equivalents or {}).values():
        entity_names |= names
    for entity in analysis.entities:
        entity_names |= {entity.name.lower(), *[a.lower() for a in entity.aliases]}
    evidence_claims = [c for c in evidence_claims if _relevant_to_query(c, analysis, entity_names)]
    if not evidence_claims:
        return ConflictAnalysis()

    pairs = sorted({(c.subject_id, c.property) for c in evidence_claims if c.subject_id and c.property})
    related = store.claims_for_subjects(pairs) if pairs else []
    evidence_ids = {c.claim_id for c in evidence_claims}
    others = [c for c in related if c.claim_id not in evidence_ids]
    comparisons = detector.detect(evidence_claims) + detector.detect(evidence_claims, against=others)

    resolver = ConflictResolver(settings, store.supersession_edges())
    result = ConflictAnalysis()
    in_evidence = {e.chunk_id for e in evidence}
    for comparison in comparisons:
        if comparison.relation == ClaimRelationType.CONTRADICTS:
            resolution = resolver.resolve(comparison.claim_a, comparison.claim_b)
            result.conflicts.append(ConflictReport(comparison=comparison, resolution=resolution))
            for claim in (comparison.claim_a, comparison.claim_b):
                if claim.chunk_id not in in_evidence and claim.chunk_id not in result.extra_chunk_ids:
                    result.extra_chunk_ids.append(claim.chunk_id)
        elif comparison.relation == ClaimRelationType.COMPATIBLE_CONDITIONAL:
            result.compatible.append(comparison)
    return result
