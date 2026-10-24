"""End-to-end ingestion: parse -> metadata -> chunk -> extract -> graph -> vectors -> BM25 -> conflicts."""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.graph.entity_extraction import Gazetteer, document_entity, extract_mentions
from app.graph.relationship_extraction import (
    cross_document_relations,
    relations_from_metadata,
    relations_from_text,
)
from app.graph.store import GraphStore
from app.ingestion.chunking import chunk_document
from app.ingestion.metadata import build_metadata, strip_front_matter
from app.ingestion.parsers import parse_document
from app.models.schemas import ClaimComparison, ClaimRelationType, IngestionResult
from app.reliability.claim_extraction import extract_claims, extract_claims_with_llm, validate_claim_provenance
from app.reliability.conflict_resolution import ConflictResolver
from app.reliability.contradiction_detection import ContradictionDetector
from app.retrieval.keyword_search import BM25Index
from app.retrieval.vector_search import Embedder

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        gazetteer: Gazetteer,
        detector: ContradictionDetector,
        keyword_index: BM25Index,
        settings: Settings,
        llm=None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.gazetteer = gazetteer
        self.detector = detector
        self.keyword_index = keyword_index
        self.settings = settings
        self.llm = llm

    def ingest_bytes(self, filename: str, content: bytes, overrides: dict[str, Any] | None = None) -> IngestionResult:
        fmt, pages = parse_document(filename, content)
        fields, pages = strip_front_matter(pages)
        body = "\n".join(p.text for p in pages)
        if not body.strip():
            raise ValueError("Document has no content after removing metadata.")
        metadata = build_metadata(
            fields,
            filename=filename,
            source_format=fmt,
            body_text=body,
            overrides=overrides,
            known_authorities=set(self.settings.authority_levels),
        )
        chunks = chunk_document(
            metadata.document_id,
            pages,
            max_chars=self.settings.chunk_max_chars,
            overlap_sentences=self.settings.chunk_overlap_sentences,
        )
        if not chunks:
            raise ValueError("Document produced no chunks.")

        # Entities and relationships ----------------------------------------------
        defines = document_entity(metadata)
        if defines is not None:
            defines = self.gazetteer.add(defines)
        meta_entities, meta_relations = relations_from_metadata(metadata, self.gazetteer, chunks[0])
        mention_entities, mentions = extract_mentions(chunks, self.gazetteer)
        text_relations = relations_from_text(metadata, chunks, self.gazetteer)
        entities = {e.entity_id: e for e in meta_entities + mention_entities}
        for rel in text_relations:
            for entity_id in (rel.source_id, rel.target_id):
                entity = self.gazetteer.get(entity_id)
                if entity:
                    entities.setdefault(entity_id, entity)

        # Claims -----------------------------------------------------------------
        claims = extract_claims(metadata, chunks, self.gazetteer)
        if self.settings.extraction_backend == "llm" and self.llm is not None and not getattr(self.llm, "is_mock", True):
            claims += extract_claims_with_llm(self.llm, metadata, chunks, self.gazetteer)
        chunk_by_id = {c.chunk_id: c for c in chunks}
        claims = [c for c in claims if validate_claim_provenance(c, chunk_by_id[c.chunk_id])]
        for claim in claims:
            if claim.subject_id and claim.subject_id not in entities:
                subject = self.gazetteer.get(claim.subject_id)
                if subject:
                    entities[subject.entity_id] = subject

        # Embeddings + persistence ---------------------------------------------------
        embeddings = self.embedder.embed([f"{metadata.title}. {c.section or ''}. {c.text}" for c in chunks])
        self.store.upsert_document(
            metadata, chunks, embeddings, list(entities.values()),
            meta_relations + text_relations, mentions, claims, defines,
        )
        link_warnings = self.relink_documents()
        conflicts = self.detect_conflicts_for(claims)
        self.refresh_indexes()
        logger.info(
            "Ingested %s: %d chunks, %d entities, %d relations, %d claims, %d conflicts",
            metadata.document_id, len(chunks), len(entities), len(meta_relations) + len(text_relations), len(claims), conflicts,
        )
        return IngestionResult(
            document_id=metadata.document_id,
            title=metadata.title,
            chunk_count=len(chunks),
            entity_count=len(entities),
            relation_count=len(meta_relations) + len(text_relations),
            claim_count=len(claims),
            conflicts_detected=conflicts,
            warnings=metadata.metadata_warnings + [w for w in link_warnings if w.startswith(metadata.document_id)],
        )

    def relink_documents(self) -> list[str]:
        """Re-resolve depends_on/supersedes across all indexed documents.

        Needed because a document may reference one that is ingested later.
        """
        documents = [r.metadata for r in self.store.list_documents()]
        relations, supersessions, warnings = cross_document_relations(documents, self.gazetteer)
        entities = {}
        for rel in relations:
            for entity_id in (rel.source_id, rel.target_id):
                entity = self.gazetteer.get(entity_id)
                if entity:
                    entities[entity_id] = entity
        self.store.upsert_relations(list(entities.values()), relations)
        self.store.set_supersessions(supersessions)
        return warnings

    def detect_conflicts_for(self, claims) -> int:
        existing = [c for c in self.store.all_claims() if c.claim_id not in {n.claim_id for n in claims}]
        comparisons = self.detector.detect(claims, against=existing)
        self._store_comparisons(comparisons)
        self.refresh_conflict_resolutions()
        return sum(1 for c in comparisons if c.relation == ClaimRelationType.CONTRADICTS)

    def _store_comparisons(self, comparisons: list[ClaimComparison]) -> None:
        rows = []
        for comp in comparisons:
            rel_type = {
                ClaimRelationType.CONTRADICTS: "CONTRADICTS",
                ClaimRelationType.SUPPORTS: "SUPPORTS",
                ClaimRelationType.COMPATIBLE_CONDITIONAL: "RELATED_TO",
            }.get(comp.relation)
            if rel_type is None:
                continue
            rows.append(
                {
                    "claim_a_id": comp.claim_a.claim_id,
                    "claim_b_id": comp.claim_b.claim_id,
                    "type": rel_type,
                    "kind": comp.relation.value,
                    "method": comp.method,
                    "confidence": comp.confidence,
                    "explanation": comp.explanation,
                    "subject": comp.claim_a.subject,
                    "property": comp.claim_a.property,
                    "document_a": comp.claim_a.document_id,
                    "document_b": comp.claim_b.document_id,
                }
            )
        if rows:
            self.store.upsert_claim_relations(rows)

    def refresh_conflict_resolutions(self) -> None:
        """Resolutions depend on metadata of *other* documents (e.g. a new superseding
        document), so they are recomputed for every stored contradiction."""
        contradictions = [r for r in self.store.claim_relations() if r.get("type") == "CONTRADICTS"]
        if not contradictions:
            return
        claims = {c.claim_id: c for c in self.store.all_claims()}
        resolver = ConflictResolver(self.settings, self.store.supersession_edges())
        updates = []
        for rel in contradictions:
            a, b = claims.get(rel["claim_a_id"]), claims.get(rel["claim_b_id"])
            if a is None or b is None:
                continue
            resolution = resolver.resolve(a, b)
            updates.append(
                {
                    **rel,
                    "resolved": resolution.resolved,
                    "winner_claim_id": resolution.winner_claim_id,
                    "winner_document_id": resolution.winner_document_id,
                    "rule": resolution.rule,
                    "resolution_explanation": resolution.explanation,
                }
            )
        self.store.upsert_claim_relations(updates)

    def refresh_indexes(self) -> None:
        self.keyword_index.build(self.store.all_chunks())
        for entity in self.store.all_entities():
            self.gazetteer.add(entity)

    def delete(self, document_id: str) -> bool:
        deleted = self.store.delete_document(document_id)
        if deleted:
            self.relink_documents()
            self.refresh_conflict_resolutions()
            self.refresh_indexes()
        return deleted
