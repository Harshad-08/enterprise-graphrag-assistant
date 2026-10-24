"""BM25 keyword index (rank-bm25) rebuilt from the store's chunks.

The corpus for this project is small, so the index is rebuilt in memory after
every ingestion or deletion. That keeps it consistent with the graph without a
second persistent store. A large deployment would use Neo4j's full-text index
(created in schema.py) or a dedicated search engine instead.
"""

from __future__ import annotations

import threading

from rank_bm25 import BM25Okapi

from app.models.schemas import Chunk
from app.utils.text import tokenize


class BM25Index:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._doc_freq: dict[str, int] = {}

    def build(self, chunks: list[Chunk]) -> None:
        tokenized = [tokenize(f"{c.section or ''} {c.text}") for c in chunks]
        with self._lock:
            self._chunks = list(chunks)
            self._bm25 = BM25Okapi(tokenized) if chunks else None
            doc_freq: dict[str, int] = {}
            for tokens in tokenized:
                for token in set(tokens):
                    doc_freq[token] = doc_freq.get(token, 0) + 1
            self._doc_freq = doc_freq

    def __len__(self) -> int:
        return len(self._chunks)

    def document_frequency(self, term: str) -> int:
        """Number of chunks containing the (stemmed) term; used to judge topic absence."""
        stemmed = tokenize(term)
        return min((self._doc_freq.get(t, 0) for t in stemmed), default=0)

    def search(self, query: str, k: int, allowed_document_ids: set[str] | None = None) -> list[tuple[Chunk, float]]:
        with self._lock:
            if self._bm25 is None:
                return []
            tokens = tokenize(query)
            if not tokens:
                return []
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(zip(self._chunks, scores), key=lambda item: item[1], reverse=True)
        results = []
        for chunk, score in ranked:
            if score <= 0:
                break
            if allowed_document_ids is not None and chunk.document_id not in allowed_document_ids:
                continue
            results.append((chunk, float(score)))
            if len(results) >= k:
                break
        return results
