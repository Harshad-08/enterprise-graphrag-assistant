"""Embedding backends and vector search over the graph store's vector index."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol

import numpy as np

from app.graph.store import GraphStore
from app.models.schemas import Chunk
from app.utils.text import tokenize

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (default all-MiniLM-L6-v2, 384-d)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dimension = int(dim())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder for tests and fully offline runs.

    It captures lexical overlap only — no semantics — which is exactly what makes
    it useful for constructing deterministic "vector search misses" in tests.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.name = f"hashing-{dimension}"
        self.dimension = dimension

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % self.dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                index, sign = self._bucket(token)
                matrix[row, index] += sign
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix


def build_embedder(backend: str, model_name: str, dimension: int) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder(dimension)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}'")


class VectorSearcher:
    def __init__(self, store: GraphStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, query: str, k: int, filters: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]:
        vector = self.embedder.embed([query])[0]
        return self.store.vector_search(vector, k, filters)
