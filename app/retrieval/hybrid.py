"""Hybrid retrieval: vector + BM25, fused with RRF, metadata-filtered, cross-encoder reranked."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.graph.store import GraphStore
from app.models.schemas import Chunk, DocumentMetadata, RetrievedChunk
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keyword_search import BM25Index
from app.retrieval.reranking import Reranker, rerank
from app.retrieval.vector_search import VectorSearcher


class HybridRetriever:
    def __init__(
        self,
        store: GraphStore,
        vector: VectorSearcher,
        keyword: BM25Index,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self.store = store
        self.vector = vector
        self.keyword = keyword
        self.reranker = reranker
        self.settings = settings

    # -- helpers -----------------------------------------------------------------
    def store_filters(self, filters: dict[str, Any] | None) -> dict[str, Any]:
        result = {k: v for k, v in (filters or {}).items() if v}
        if not self.settings.include_superseded_in_retrieval:
            result.setdefault("statuses", ["active", "unknown", "draft"])
        return result

    def _allowed_documents(self, filters: dict[str, Any]) -> set[str] | None:
        if not filters:
            return None
        allowed = set()
        for record in self.store.list_documents():
            meta = record.metadata
            if filters.get("department") and meta.department != filters["department"]:
                continue
            if filters.get("doc_type") and meta.doc_type != filters["doc_type"]:
                continue
            if filters.get("statuses") and meta.status.value not in filters["statuses"]:
                continue
            allowed.add(meta.document_id)
        return allowed

    def to_retrieved(self, chunks: list[Chunk], source: str) -> list[RetrievedChunk]:
        metadata = self.store.get_documents_metadata(sorted({c.document_id for c in chunks}))
        return [
            RetrievedChunk(chunk=c, document=metadata[c.document_id], sources=[source])
            for c in chunks
            if c.document_id in metadata
        ]

    # -- retrieval stages ----------------------------------------------------------
    def vector_only(self, query: str, k: int, filters: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        """Baseline retrieval: plain top-k vector similarity, no fusion or reranking."""
        hits = self.vector.search(query, k, self.store_filters(filters))
        retrieved = self.to_retrieved([c for c, _ in hits], "vector")
        scores = {c.chunk_id: s for c, s in hits}
        for item in retrieved:
            item.vector_score = scores[item.chunk_id]
        return retrieved

    def candidates(self, query: str, filters: dict[str, Any] | None = None, source_tag: str | None = None) -> list[RetrievedChunk]:
        """Vector + BM25 candidates merged with Reciprocal Rank Fusion (not yet reranked)."""
        store_filters = self.store_filters(filters)
        vector_hits = self.vector.search(query, self.settings.vector_top_k, store_filters)
        keyword_hits = self.keyword.search(query, self.settings.keyword_top_k, self._allowed_documents(store_filters))
        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c, _ in vector_hits], [c.chunk_id for c, _ in keyword_hits]], k=self.settings.rrf_k
        )
        by_id: dict[str, Chunk] = {c.chunk_id: c for c, _ in vector_hits + keyword_hits}
        vector_scores = {c.chunk_id: s for c, s in vector_hits}
        keyword_scores = {c.chunk_id: s for c, s in keyword_hits}
        fused_chunks = [by_id[cid] for cid, _ in fused]
        retrieved = self.to_retrieved(fused_chunks, "hybrid")
        rrf = dict(fused)
        for item in retrieved:
            item.rrf_score = rrf[item.chunk_id]
            item.vector_score = vector_scores.get(item.chunk_id)
            item.keyword_score = keyword_scores.get(item.chunk_id)
            item.sources = [s for s, scores in (("vector", vector_scores), ("keyword", keyword_scores)) if item.chunk_id in scores]
            if source_tag:
                item.sources.append(source_tag)
        return retrieved

    def rerank_pool(
        self, query: str, pool: list[RetrievedChunk], top_k: int, alt_queries: list[str] | None = None
    ) -> list[RetrievedChunk]:
        merged = merge_candidates(pool)
        ranked = rerank(self.reranker, query, merged, alt_queries)
        return ranked[:top_k]

    def retrieve(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        pool = self.candidates(query, filters)[: self.settings.rerank_candidates]
        return self.rerank_pool(query, pool, top_k or self.settings.final_top_k)


def merge_candidates(items: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Deduplicate by chunk ID, keeping the best scores and the union of sources."""
    merged: dict[str, RetrievedChunk] = {}
    for item in items:
        existing = merged.get(item.chunk_id)
        if existing is None:
            merged[item.chunk_id] = item.model_copy(deep=True)
            continue
        for field in ("vector_score", "keyword_score", "rrf_score"):
            new, old = getattr(item, field), getattr(existing, field)
            if new is not None and (old is None or new > old):
                setattr(existing, field, new)
        existing.sources = sorted(set(existing.sources) | set(item.sources))
    return list(merged.values())


def document_ids(items: list[RetrievedChunk]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item.chunk.document_id not in seen:
            seen.append(item.chunk.document_id)
    return seen


def metadata_of(items: list[RetrievedChunk]) -> dict[str, DocumentMetadata]:
    return {i.document.document_id: i.document for i in items}
