"""Adaptive retrieval with bounded graph-based recovery.

Flow (improved configuration):

1. Hybrid retrieval (vector + BM25 + RRF + rerank). For relational questions
   ("which procedure depends on ...") graph traversal is a planned step and its
   passages join the initial pool.
2. Evidence sufficiency assessment.
3. If insufficient, up to ``recovery_max_attempts`` recovery rounds. Each round
   traverses the graph from the query's entities (hop radius grows per round),
   collects connected passages, expands the query with synonyms and related
   entity names, reruns hybrid retrieval, merges, reranks against the *original*
   question and reassesses.
4. If still insufficient, classify why (topic absent vs. documents without an
   answer vs. unresolved) so the abstention message is accurate.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.graph.entity_extraction import Gazetteer, analyze_query, expand_query, reformulate_query
from app.graph.store import GraphStore
from app.models.schemas import (
    EvidenceAssessment,
    QueryAnalysis,
    RecoveryAttempt,
    RetrievalOutcome,
    RetrievalResult,
    RetrievedChunk,
)
from app.retrieval.evidence_assessment import assess_evidence
from app.retrieval.graph_retrieval import GraphRetriever
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class RetrievalOrchestrator:
    def __init__(
        self,
        store: GraphStore,
        hybrid: HybridRetriever,
        graph: GraphRetriever,
        gazetteer: Gazetteer,
        settings: Settings,
    ) -> None:
        self.store = store
        self.hybrid = hybrid
        self.graph = graph
        self.gazetteer = gazetteer
        self.settings = settings

    def _assess(self, analysis: QueryAnalysis, evidence: list[RetrievedChunk], equivalents) -> EvidenceAssessment:
        claims = self.store.claims_for_chunks([e.chunk_id for e in evidence])
        return assess_evidence(analysis, evidence, claims, self.settings, self.gazetteer, equivalents)

    def retrieve(self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or self.settings.final_top_k
        analysis = analyze_query(query, self.gazetteer)
        seeds = self.graph.link_entities(analysis)
        equivalents = self.graph.equivalents(seeds)

        # -- initial retrieval -----------------------------------------------------
        pool = self.hybrid.candidates(query, filters)[: self.settings.rerank_candidates]
        graph_paths = []
        graph_used = False
        if analysis.relational and seeds:
            expansion = self.graph.expand(seeds, max_hops=1, max_chunks=self.settings.graph_max_chunks)
            pool += self.hybrid.to_retrieved(expansion.chunks, "graph")
            graph_paths = expansion.paths
            graph_used = bool(expansion.chunks)
        evidence = self.hybrid.rerank_pool(query, pool, top_k)
        initial_ids = [e.chunk_id for e in evidence]
        initial_assessment = self._assess(analysis, evidence, equivalents)
        final_assessment = initial_assessment
        attempts: list[RecoveryAttempt] = []

        if initial_assessment.sufficient:
            outcome = RetrievalOutcome.INITIAL_SUFFICIENT
        elif not self.settings.recovery_enabled:
            outcome = RetrievalOutcome.RECOVERY_DISABLED
        else:
            neighbours = self.graph.neighbours(seeds)
            reformulations = reformulate_query(
                query, self.gazetteer, {k: [e.name for e in v] for k, v in neighbours.items()}
            )
            for attempt in range(1, self.settings.recovery_max_attempts + 1):
                hops = min(attempt, self.settings.graph_max_hops)
                expansion = self.graph.expand(seeds, max_hops=hops, max_chunks=self.settings.graph_max_chunks)
                related_names = list(expansion.related.values())[:5] if attempt > 1 else list(expansion.related.values())[:2]
                expanded = expand_query(query, self.gazetteer, related_names)
                recovery_pool = list(evidence)
                recovery_pool += self.hybrid.to_retrieved(expansion.chunks, "graph")
                recovery_pool += self.hybrid.candidates(expanded, filters, source_tag="expansion")[: self.settings.rerank_candidates]
                for variant in reformulations:
                    recovery_pool += self.hybrid.candidates(variant, filters, source_tag="reformulation")[: self.settings.rerank_candidates]
                previous = {e.chunk_id for e in evidence}
                evidence = self.hybrid.rerank_pool(query, recovery_pool, top_k, alt_queries=reformulations)
                final_assessment = self._assess(analysis, evidence, equivalents)
                graph_paths = expansion.paths or graph_paths
                graph_used = graph_used or bool(expansion.chunks)
                attempts.append(
                    RecoveryAttempt(
                        attempt=attempt,
                        expanded_query=expanded,
                        reformulated_queries=reformulations,
                        graph_entities=list(expansion.related.values())[:15],
                        graph_documents=expansion.document_ids[:15],
                        new_chunk_ids=[e.chunk_id for e in evidence if e.chunk_id not in previous],
                        assessment=final_assessment,
                    )
                )
                logger.info("Recovery attempt %d for %r: sufficient=%s", attempt, query, final_assessment.sufficient)
                if final_assessment.sufficient:
                    break
            outcome = (
                RetrievalOutcome.RECOVERED if final_assessment.sufficient else self._classify_failure(analysis, seeds)
            )

        return RetrievalResult(
            analysis=analysis,
            evidence=evidence,
            initial_evidence_chunk_ids=initial_ids,
            initial_assessment=initial_assessment,
            final_assessment=final_assessment,
            outcome=outcome,
            recovery_attempts=attempts,
            graph_paths=graph_paths[:10],
            graph_used=graph_used,
        )

    def _classify_failure(self, analysis: QueryAnalysis, seeds) -> RetrievalOutcome:
        """Distinguish a topic absent from the corpus from documents that lack the answer.

        This is only called after every recovery round failed, so a single empty
        search never leads to an "absent" verdict.
        """
        if seeds:
            return RetrievalOutcome.DOCUMENTS_WITHOUT_ANSWER
        if analysis.concepts:
            present = [c for c in analysis.concepts if self.hybrid.keyword.document_frequency(c) > 0]
            if len(present) / len(analysis.concepts) < 0.5:
                return RetrievalOutcome.NOT_IN_CORPUS
        return RetrievalOutcome.INSUFFICIENT_AFTER_RECOVERY
