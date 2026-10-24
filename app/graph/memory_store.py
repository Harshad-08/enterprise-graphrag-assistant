"""In-process implementation of :class:`GraphStore` with the same semantics as Neo4j.

Used by unit tests and by ``GRAPH_BACKEND=memory`` for offline demos. It is not
persistent and not intended for multi-process deployments.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any

import numpy as np

from app.graph.store import TRAVERSABLE_RELATIONSHIPS
from app.models.schemas import (
    Chunk,
    Claim,
    DocumentMetadata,
    DocumentRecord,
    Entity,
    GraphPath,
    Mention,
    Relation,
)


class InMemoryGraphStore:
    backend_name = "memory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.clear()

    # -- lifecycle -----------------------------------------------------------
    def verify_connectivity(self) -> bool:
        return True

    def setup_schema(self, embedding_dimension: int) -> None:
        self.embedding_dimension = embedding_dimension

    def clear(self) -> None:
        with self._lock:
            self.documents: dict[str, DocumentMetadata] = {}
            self.chunks: dict[str, Chunk] = {}
            self.embeddings: dict[str, np.ndarray] = {}
            self.entities: dict[str, Entity] = {}
            # (source, type, target) -> {"documents": set, "chunks": set}
            self.relations: dict[tuple[str, str, str], dict[str, set]] = {}
            self.mentions: dict[str, set[str]] = defaultdict(set)
            self.defines: dict[str, str] = {}
            self.claims: dict[str, Claim] = {}
            self.claim_rel: dict[tuple[str, str, str], dict[str, Any]] = {}
            self.supersedes: set[tuple[str, str]] = set()

    # -- writes ----------------------------------------------------------------
    def upsert_document(self, metadata, chunks, embeddings, entities, relations, mentions, claims, defines) -> None:
        with self._lock:
            self.delete_document(metadata.document_id)
            self.documents[metadata.document_id] = metadata
            for chunk, vector in zip(chunks, embeddings):
                self.chunks[chunk.chunk_id] = chunk
                self.embeddings[chunk.chunk_id] = np.asarray(vector, dtype=np.float32)
            self.upsert_relations(entities + ([defines] if defines else []), relations)
            for mention in mentions:
                self.mentions[mention.chunk_id].add(mention.entity_id)
            if defines:
                self.defines[metadata.document_id] = defines.entity_id
            for claim in claims:
                self.claims[claim.claim_id] = claim

    def upsert_relations(self, entities: list[Entity], relations: list[Relation]) -> None:
        with self._lock:
            for entity in entities:
                existing = self.entities.get(entity.entity_id)
                if existing:
                    entity = existing.model_copy(update={"aliases": sorted(set(existing.aliases) | set(entity.aliases))})
                self.entities[entity.entity_id] = entity
            for rel in relations:
                if rel.source_id not in self.entities or rel.target_id not in self.entities:
                    continue
                slot = self.relations.setdefault((rel.source_id, rel.type, rel.target_id), {"documents": set(), "chunks": set()})
                slot["documents"].add(rel.document_id)
                if rel.chunk_id:
                    slot["chunks"].add(rel.chunk_id)

    def set_supersessions(self, edges: list[tuple[str, str]]) -> None:
        with self._lock:
            self.supersedes = {(a, b) for a, b in edges if a in self.documents and b in self.documents}

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self.documents:
                return False
            del self.documents[document_id]
            prefix = f"{document_id}::"
            for chunk_id in [c for c in self.chunks if c.startswith(prefix)]:
                self.chunks.pop(chunk_id, None)
                self.embeddings.pop(chunk_id, None)
                self.mentions.pop(chunk_id, None)
            removed_claims = {cid for cid, c in self.claims.items() if c.document_id == document_id}
            for cid in removed_claims:
                del self.claims[cid]
            self.claim_rel = {k: v for k, v in self.claim_rel.items() if k[0] not in removed_claims and k[2] not in removed_claims}
            for key in list(self.relations):
                slot = self.relations[key]
                slot["documents"].discard(document_id)
                slot["chunks"] = {c for c in slot["chunks"] if not c.startswith(prefix)}
                if not slot["documents"]:
                    del self.relations[key]
            self.defines.pop(document_id, None)
            self.supersedes = {(a, b) for a, b in self.supersedes if document_id not in (a, b)}
            self._drop_orphan_entities()
            return True

    def _drop_orphan_entities(self) -> None:
        used = set(self.defines.values())
        for s, _, t in self.relations:
            used.update((s, t))
        for ids in self.mentions.values():
            used.update(ids)
        used.update(c.subject_id for c in self.claims.values() if c.subject_id)
        self.entities = {k: v for k, v in self.entities.items() if k in used}

    # -- document reads -------------------------------------------------------
    def _record(self, doc: DocumentMetadata) -> DocumentRecord:
        prefix = f"{doc.document_id}::"
        return DocumentRecord(
            metadata=doc,
            chunk_count=sum(1 for c in self.chunks if c.startswith(prefix)),
            claim_count=sum(1 for c in self.claims.values() if c.document_id == doc.document_id),
        )

    def list_documents(self) -> list[DocumentRecord]:
        return [self._record(d) for d in sorted(self.documents.values(), key=lambda d: d.document_id)]

    def get_document(self, document_id: str) -> DocumentRecord | None:
        doc = self.documents.get(document_id)
        return self._record(doc) if doc else None

    def get_documents_metadata(self, document_ids: list[str]) -> dict[str, DocumentMetadata]:
        return {d: self.documents[d] for d in document_ids if d in self.documents}

    def all_chunks(self) -> list[Chunk]:
        return sorted(self.chunks.values(), key=lambda c: c.chunk_id)

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self.chunks[c] for c in chunk_ids if c in self.chunks]

    def chunks_for_documents(self, document_ids: list[str], limit: int) -> list[Chunk]:
        wanted = set(document_ids)
        return [c for c in self.all_chunks() if c.document_id in wanted][:limit]

    def chunks_mentioning(self, entity_ids: list[str], limit: int) -> list[Chunk]:
        wanted = set(entity_ids)
        hits = [self.chunks[cid] for cid, ents in self.mentions.items() if ents & wanted and cid in self.chunks]
        return sorted(hits, key=lambda c: c.chunk_id)[:limit]

    def relation_chunks(self, entity_ids: list[str], limit: int) -> list[Chunk]:
        """Chunks that provide provenance for relationships touching the given entities."""
        wanted = set(entity_ids)
        chunk_ids: set[str] = set()
        for (s, _, t), slot in self.relations.items():
            if s in wanted or t in wanted:
                chunk_ids |= slot["chunks"]
        return self.get_chunks(sorted(chunk_ids))[:limit]

    def vector_search(self, embedding, k, filters=None):
        filters = filters or {}
        query = np.asarray(embedding, dtype=np.float32)
        qn = np.linalg.norm(query) or 1.0
        scored = []
        for chunk_id, vector in self.embeddings.items():
            chunk = self.chunks[chunk_id]
            if not _passes_filters(self.documents.get(chunk.document_id), filters):
                continue
            cosine = float(np.dot(query, vector) / (qn * (np.linalg.norm(vector) or 1.0)))
            # Same normalisation Neo4j uses for cosine vector indexes: (1 + cos) / 2.
            scored.append((chunk, (1.0 + cosine) / 2.0))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    # -- entity reads -----------------------------------------------------------
    def all_entities(self) -> list[Entity]:
        return list(self.entities.values())

    def get_entities(self, entity_ids: list[str]) -> list[Entity]:
        return [self.entities[e] for e in entity_ids if e in self.entities]

    def traverse(self, seed_ids: list[str], max_hops: int, limit: int = 200) -> list[GraphPath]:
        adjacency: dict[str, list[tuple[str, str, frozenset]]] = defaultdict(list)
        for (s, rel_type, t), slot in self.relations.items():
            if rel_type not in TRAVERSABLE_RELATIONSHIPS:
                continue
            docs = frozenset(slot["documents"])
            adjacency[s].append((t, rel_type, docs))
            adjacency[t].append((s, "<" + rel_type, docs))
        defined_by: dict[str, set[str]] = defaultdict(set)
        for doc_id, entity_id in self.defines.items():
            defined_by[entity_id].add(doc_id)

        paths: list[GraphPath] = []
        queue = deque([(seed, [seed], [], set()) for seed in seed_ids if seed in self.entities])
        while queue and len(paths) < limit:
            node, nodes, rels, docs = queue.popleft()
            if len(rels) >= max_hops:
                continue
            # Departments own many artifacts; walking *through* them would connect
            # unrelated policies, so they may only be a seed or an end node.
            if len(nodes) > 1 and self.entities[node].label == "Department":
                continue
            for neighbour, rel_type, rel_docs in sorted(adjacency[node]):
                if neighbour in nodes:
                    continue
                new_nodes, new_rels = nodes + [neighbour], rels + [rel_type]
                new_docs = docs | rel_docs
                path_docs = set(new_docs)
                for n in new_nodes:
                    path_docs |= defined_by.get(n, set())
                paths.append(
                    GraphPath(
                        nodes=[self.entities[n].name for n in new_nodes],
                        node_ids=new_nodes,
                        relationships=new_rels,
                        document_ids=sorted(path_docs),
                    )
                )
                queue.append((neighbour, new_nodes, new_rels, new_docs))
        return paths[:limit]

    # -- claims -----------------------------------------------------------------
    def all_claims(self) -> list[Claim]:
        return list(self.claims.values())

    def claims_for_chunks(self, chunk_ids: list[str]) -> list[Claim]:
        wanted = set(chunk_ids)
        return [c for c in self.claims.values() if c.chunk_id in wanted]

    def claims_for_subjects(self, subject_property_pairs: list[tuple[str, str]]) -> list[Claim]:
        wanted = set(subject_property_pairs)
        return [c for c in self.claims.values() if (c.subject_id, c.property) in wanted]

    def upsert_claim_relations(self, relations: list[dict[str, Any]]) -> None:
        with self._lock:
            for rel in relations:
                if rel["claim_a_id"] in self.claims and rel["claim_b_id"] in self.claims:
                    key = (rel["claim_a_id"], rel["type"], rel["claim_b_id"])
                    self.claim_rel[key] = dict(rel)

    def claim_relations(self, claim_ids: list[str] | None = None) -> list[dict[str, Any]]:
        wanted = set(claim_ids) if claim_ids is not None else None
        return [
            dict(v) for (a, _, b), v in self.claim_rel.items()
            if wanted is None or a in wanted or b in wanted
        ]

    def supersession_edges(self) -> list[tuple[str, str]]:
        return sorted(self.supersedes)

    def stats(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "relationships": len(self.relations),
            "claims": len(self.claims),
            "claim_relationships": len(self.claim_rel),
            "supersessions": len(self.supersedes),
        }


def _passes_filters(doc: DocumentMetadata | None, filters: dict[str, Any]) -> bool:
    if doc is None:
        return False
    if filters.get("department") and doc.department != filters["department"]:
        return False
    if filters.get("doc_type") and doc.doc_type != filters["doc_type"]:
        return False
    statuses = filters.get("statuses")
    if statuses and doc.status.value not in statuses:
        return False
    return True
