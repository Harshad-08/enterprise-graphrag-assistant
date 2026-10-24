"""Citation construction and validation against stored source data.

Citations are built by the application from evidence labels — the LLM only ever
sees and emits labels like ``E2``, so it cannot invent document IDs, chunk IDs
or page numbers. Every citation is re-validated against the store before it is
returned, and pages are included only when the source format has them.
"""

from __future__ import annotations

from app.graph.store import GraphStore
from app.models.schemas import Citation, ClaimVerification, RetrievedChunk, VerificationStatus
from app.utils.text import content_terms, split_sentences


def best_excerpt(claim: str, passage: str, max_chars: int = 320) -> str:
    """The passage sentence(s) that best match the claim — verbatim source text."""
    sentences = split_sentences(passage) or [passage]
    terms = content_terms(claim)
    best = max(sentences, key=lambda s: len(terms & content_terms(s)))
    return best if len(best) <= max_chars else best[: max_chars - 1].rstrip() + "…"


def build_citations(
    verifications: list[ClaimVerification], evidence: dict[str, RetrievedChunk]
) -> tuple[list[Citation], dict[str, int]]:
    """Number citations in order of first use. Returns (citations, label -> number)."""
    citations: list[Citation] = []
    numbering: dict[str, int] = {}
    for verification in verifications:
        if verification.status != VerificationStatus.SUPPORTED:
            continue
        for label in verification.supporting_ids:
            if label in numbering or label not in evidence:
                continue
            item = evidence[label]
            numbering[label] = len(citations) + 1
            citations.append(
                Citation(
                    citation_id=numbering[label],
                    document_id=item.document.document_id,
                    document_title=item.document.title,
                    chunk_id=item.chunk.chunk_id,
                    page=item.chunk.page,
                    section=item.chunk.section,
                    excerpt=best_excerpt(verification.claim, item.chunk.text),
                    document_status=item.document.status.value,
                    version=item.document.version,
                )
            )
    return citations, numbering


def validate_citations(citations: list[Citation], store: GraphStore) -> tuple[list[Citation], list[str]]:
    """Drop any citation whose chunk is not stored or whose excerpt is not in the chunk."""
    stored = {c.chunk_id: c for c in store.get_chunks([c.chunk_id for c in citations])}
    valid, problems = [], []
    for citation in citations:
        chunk = stored.get(citation.chunk_id)
        if chunk is None:
            problems.append(f"Citation {citation.citation_id} references missing chunk {citation.chunk_id}.")
            continue
        if chunk.document_id != citation.document_id:
            problems.append(f"Citation {citation.citation_id} has a mismatched document ID.")
            continue
        excerpt = citation.excerpt.rstrip("…")
        if excerpt not in chunk.text:
            problems.append(f"Citation {citation.citation_id} excerpt not found in stored chunk.")
            continue
        if citation.page is not None and chunk.page != citation.page:
            problems.append(f"Citation {citation.citation_id} page does not match stored chunk.")
            continue
        valid.append(citation)
    return valid, problems


def citation_label_map(evidence: list[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    return {f"E{i}": item for i, item in enumerate(evidence, start=1)}
