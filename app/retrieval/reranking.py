"""Cross-encoder reranking with a transparent lexical fallback."""

from __future__ import annotations

import logging
import math
from typing import Protocol

from app.models.schemas import RetrievedChunk
from app.utils.text import content_terms

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    """ms-marco cross-encoders output relevance logits; a sigmoid maps them to [0, 1]
    so the evidence-sufficiency thresholds are interpretable."""

    def __init__(self, model_name: str, model=None) -> None:
        self.name = model_name
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)
        self._model = model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        logits = self._model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [_sigmoid(float(x)) for x in logits]


class LexicalReranker:
    """Fallback when the cross-encoder is disabled/unavailable: fraction of query terms
    present in the passage. Purely lexical — documented as such in the UI and README."""

    name = "lexical-overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        terms = content_terms(query)
        if not terms:
            return [0.0 for _ in passages]
        return [len(terms & content_terms(p)) / len(terms) for p in passages]


def build_reranker(enabled: bool, model_name: str) -> Reranker:
    if not enabled:
        return LexicalReranker()
    try:
        return CrossEncoderReranker(model_name)
    except Exception as exc:  # model download / load failure
        logger.warning("Cross-encoder %s unavailable (%s); using lexical reranker.", model_name, exc)
        return LexicalReranker()


def rerank(
    reranker: Reranker, query: str, candidates: list[RetrievedChunk], alt_queries: list[str] | None = None
) -> list[RetrievedChunk]:
    """Score candidates against the query; with ``alt_queries`` (graph reformulations)
    a candidate keeps its best score across the original and reformulated queries."""
    if not candidates:
        return []
    passages = [f"{c.document.title}. {c.chunk.section or ''}. {c.chunk.text}" for c in candidates]
    scores = reranker.score(query, passages)
    for alt in alt_queries or []:
        scores = [max(a, b) for a, b in zip(scores, reranker.score(alt, passages))]
    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)
    return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
