"""Neo4j implementation of :class:`app.graph.store.GraphStore`."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.graph import graph_queries as q
from app.graph.schema import ENTITY_LABELS, VECTOR_INDEX_NAME, schema_statements
from app.graph.store import CLAIM_RELATIONSHIPS, TRAVERSABLE_RELATIONSHIPS, GraphStoreError
from app.models.schemas import (
    Chunk,
    Claim,
    DocumentMetadata,
    DocumentRecord,
    Entity,
    GraphPath,
    Relation,
)

logger = logging.getLogger(__name__)

_DOC_LIST_FIELDS = ("supersedes", "governs", "applies_to", "depends_on", "related", "metadata_warnings")


def _native(value: Any) -> Any:
    """Convert neo4j.time types back to Python natives."""
    return value.to_native() if hasattr(value, "to_native") else value


class Neo4jGraphStore:
    backend_name = "neo4j"

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j") -> None:
        self.database = database
        self._driver: Driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self._driver.close()

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        try:
            records, _, _ = self._driver.execute_query(query, params, database_=self.database)
        except ServiceUnavailable as exc:
            raise GraphStoreError(f"Neo4j is unavailable: {exc}") from exc
        except Neo4jError as exc:
            raise GraphStoreError(f"Neo4j query failed: {exc.message}") from exc
        return [r.data() for r in records]

    def _write_tx(self, work) -> None:
        try:
            with self._driver.session(database=self.database) as session:
                session.execute_write(work)
        except ServiceUnavailable as exc:
            raise GraphStoreError(f"Neo4j is unavailable: {exc}") from exc
        except Neo4jError as exc:
            raise GraphStoreError(f"Neo4j write failed: {exc.message}") from exc

    # -- lifecycle ---------------------------------------------------------------
    def verify_connectivity(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as exc:  # driver raises a variety of auth/network errors
            logger.warning("Neo4j connectivity check failed: %s", exc)
            return False

    def setup_schema(self, embedding_dimension: int) -> None:
        for statement in schema_statements(embedding_dimension):
            self._run(statement)

    def clear(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")

    # -- writes -------------------------------------------------------------------
    def upsert_document(self, metadata, chunks, embeddings, entities, relations, mentions, claims, defines) -> None:
        self.delete_document(metadata.document_id)
        props = metadata.model_dump(mode="python", exclude={"document_id"})
        props["status"] = metadata.status.value
        chunk_rows = [
            {**c.model_dump(), "embedding": [float(x) for x in vec]} for c, vec in zip(chunks, embeddings)
        ]
        claim_rows = [{"claim_id": c.claim_id, "props": self._claim_props(c)} for c in claims]
        all_entities = entities + ([defines] if defines else [])

        def work(tx) -> None:
            tx.run(q.UPSERT_DOCUMENT, document_id=metadata.document_id, props=props)
            tx.run(q.UPSERT_CHUNKS, document_id=metadata.document_id, chunks=chunk_rows)
            self._tx_upsert_entities(tx, all_entities)
            tx.run(q.UPSERT_MENTIONS, mentions=[m.model_dump() for m in mentions])
            if defines:
                tx.run(q.UPSERT_DEFINES, document_id=metadata.document_id, entity_id=defines.entity_id)
            self._tx_upsert_relations(tx, relations)
            tx.run(q.UPSERT_CLAIMS, document_id=metadata.document_id, claims=claim_rows)

        self._write_tx(work)

    @staticmethod
    def _claim_props(claim: Claim) -> dict[str, Any]:
        props = claim.model_dump(mode="python", exclude={"claim_id"})
        props["polarity"] = claim.polarity.value
        props["status"] = claim.status.value
        return props

    @staticmethod
    def _tx_upsert_entities(tx, entities: list[Entity]) -> None:
        by_label: dict[str, list[dict]] = {}
        for entity in entities:
            if entity.label not in ENTITY_LABELS:
                raise GraphStoreError(f"Unsupported entity label {entity.label}")
            by_label.setdefault(entity.label, []).append(entity.model_dump())
        for label, rows in by_label.items():
            tx.run(q.UPSERT_ENTITIES.format(label=label), entities=rows)

    @staticmethod
    def _tx_upsert_relations(tx, relations: list[Relation]) -> None:
        by_type: dict[str, list[dict]] = {}
        for rel in relations:
            if rel.type not in TRAVERSABLE_RELATIONSHIPS:
                raise GraphStoreError(f"Unsupported relationship type {rel.type}")
            by_type.setdefault(rel.type, []).append(rel.model_dump())
        for rel_type, rows in by_type.items():
            tx.run(q.UPSERT_RELATIONS.format(rel_type=rel_type), relations=rows)

    def upsert_relations(self, entities: list[Entity], relations: list[Relation]) -> None:
        def work(tx) -> None:
            self._tx_upsert_entities(tx, entities)
            self._tx_upsert_relations(tx, relations)

        self._write_tx(work)

    def set_supersessions(self, edges: list[tuple[str, str]]) -> None:
        def work(tx) -> None:
            tx.run(q.CLEAR_SUPERSESSIONS)
            tx.run(q.SET_SUPERSESSIONS, edges=[list(e) for e in edges])

        self._write_tx(work)

    def delete_document(self, document_id: str) -> bool:
        deleted = self._run(q.DELETE_DOCUMENT_NODES, document_id=document_id)
        self._run(q.RETRACT_RELATION_PROVENANCE, document_id=document_id, chunk_prefix=f"{document_id}::")
        self._run(q.DELETE_ORPHAN_ENTITIES)
        return bool(deleted and deleted[0]["deleted"])

    # -- document reads -----------------------------------------------------------
    @staticmethod
    def _to_metadata(document_id: str, props: dict[str, Any]) -> DocumentMetadata:
        data = {k: _native(v) for k, v in props.items() if k in DocumentMetadata.model_fields}
        for field in _DOC_LIST_FIELDS:
            data[field] = list(data.get(field) or [])
        data["document_id"] = document_id
        return DocumentMetadata(**data)

    def list_documents(self) -> list[DocumentRecord]:
        rows = self._run(
            """
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
            WITH d, count(c) AS chunks
            OPTIONAL MATCH (d)-[:CONTAINS]->(cl:Claim)
            RETURN d.document_id AS id, properties(d) AS props, chunks, count(cl) AS claims
            ORDER BY id
            """
        )
        return [
            DocumentRecord(metadata=self._to_metadata(r["id"], r["props"]), chunk_count=r["chunks"], claim_count=r["claims"])
            for r in rows
        ]

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return next((d for d in self.list_documents() if d.metadata.document_id == document_id), None)

    def get_documents_metadata(self, document_ids: list[str]) -> dict[str, DocumentMetadata]:
        rows = self._run(
            "MATCH (d:Document) WHERE d.document_id IN $ids RETURN d.document_id AS id, properties(d) AS props",
            ids=list(document_ids),
        )
        return {r["id"]: self._to_metadata(r["id"], r["props"]) for r in rows}

    @staticmethod
    def _chunks(rows: list[dict[str, Any]]) -> list[Chunk]:
        return [Chunk(**r["chunk"]) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        return self._chunks(
            self._run(
                "MATCH (ch:Chunk) RETURN ch {.chunk_id, .document_id, .text, .chunk_index, .page, .section, "
                ".content_hash} AS chunk ORDER BY ch.chunk_id"
            )
        )

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        rows = self._run(
            "MATCH (ch:Chunk) WHERE ch.chunk_id IN $ids RETURN ch {.chunk_id, .document_id, .text, .chunk_index, "
            ".page, .section, .content_hash} AS chunk",
            ids=list(chunk_ids),
        )
        by_id = {c.chunk_id: c for c in self._chunks(rows)}
        return [by_id[c] for c in chunk_ids if c in by_id]

    def chunks_for_documents(self, document_ids: list[str], limit: int) -> list[Chunk]:
        return self._chunks(self._run(q.CHUNKS_FOR_DOCUMENTS, document_ids=list(document_ids), limit=limit))

    def chunks_mentioning(self, entity_ids: list[str], limit: int) -> list[Chunk]:
        return self._chunks(self._run(q.CHUNKS_MENTIONING, entity_ids=list(entity_ids), limit=limit))

    def relation_chunks(self, entity_ids: list[str], limit: int) -> list[Chunk]:
        return self._chunks(self._run(q.RELATION_CHUNKS, entity_ids=list(entity_ids), limit=limit))

    def vector_search(self, embedding, k, filters=None):
        filters = filters or {}
        rows = self._run(
            q.VECTOR_SEARCH,
            index_name=VECTOR_INDEX_NAME,
            # Over-fetch so metadata post-filtering still leaves k results.
            k=k * 3 if filters else k,
            embedding=[float(x) for x in np.asarray(embedding)],
            department=filters.get("department"),
            doc_type=filters.get("doc_type"),
            statuses=filters.get("statuses"),
        )
        return [(Chunk(**r["chunk"]), float(r["score"])) for r in rows][:k]

    # -- entities -----------------------------------------------------------------
    @staticmethod
    def _entity(row: dict[str, Any]) -> Entity:
        label = next((lbl for lbl in row["labels"] if lbl in ENTITY_LABELS), "Concept")
        return Entity(entity_id=row["id"], name=row["name"], label=label, aliases=list(row["aliases"] or []))

    def all_entities(self) -> list[Entity]:
        rows = self._run("MATCH (e:Entity) RETURN e.entity_id AS id, e.name AS name, labels(e) AS labels, e.aliases AS aliases")
        return [self._entity(r) for r in rows]

    def get_entities(self, entity_ids: list[str]) -> list[Entity]:
        rows = self._run(
            "MATCH (e:Entity) WHERE e.entity_id IN $ids "
            "RETURN e.entity_id AS id, e.name AS name, labels(e) AS labels, e.aliases AS aliases",
            ids=list(entity_ids),
        )
        return [self._entity(r) for r in rows]

    def traverse(self, seed_ids: list[str], max_hops: int, limit: int = 200) -> list[GraphPath]:
        hops = max(1, min(int(max_hops), 4))  # bounded: variable-length expansion is exponential
        query = q.TRAVERSE.format(rel_types="|".join(TRAVERSABLE_RELATIONSHIPS), max_hops=hops)
        rows = self._run(query, seed_ids=list(seed_ids), limit=limit)
        return [
            GraphPath(nodes=r["names"], node_ids=r["ids"], relationships=r["rels"], document_ids=sorted(set(r["docs"])))
            for r in rows
        ]

    # -- claims -------------------------------------------------------------------
    @staticmethod
    def _claim(row: dict[str, Any]) -> Claim:
        props = {k: _native(v) for k, v in row["props"].items() if k in Claim.model_fields and k != "claim_id"}
        props["conditions"] = list(props.get("conditions") or [])
        return Claim(claim_id=row["id"], **props)

    def all_claims(self) -> list[Claim]:
        return [self._claim(r) for r in self._run("MATCH (c:Claim) RETURN c.claim_id AS id, properties(c) AS props")]

    def claims_for_chunks(self, chunk_ids: list[str]) -> list[Claim]:
        rows = self._run(
            "MATCH (c:Claim) WHERE c.chunk_id IN $ids RETURN c.claim_id AS id, properties(c) AS props",
            ids=list(chunk_ids),
        )
        return [self._claim(r) for r in rows]

    def claims_for_subjects(self, subject_property_pairs: list[tuple[str, str]]) -> list[Claim]:
        rows = self._run(
            "UNWIND $pairs AS p MATCH (c:Claim {subject_id: p[0], property: p[1]}) "
            "RETURN DISTINCT c.claim_id AS id, properties(c) AS props",
            pairs=[list(p) for p in subject_property_pairs],
        )
        return [self._claim(r) for r in rows]

    def upsert_claim_relations(self, relations: list[dict[str, Any]]) -> None:
        by_type: dict[str, list[dict]] = {}
        for rel in relations:
            if rel["type"] not in CLAIM_RELATIONSHIPS:
                raise GraphStoreError(f"Unsupported claim relationship {rel['type']}")
            by_type.setdefault(rel["type"], []).append(rel)

        def work(tx) -> None:
            for rel_type, rows in by_type.items():
                tx.run(q.UPSERT_CLAIM_RELATIONS.format(rel_type=rel_type), relations=rows)

        self._write_tx(work)

    def claim_relations(self, claim_ids: list[str] | None = None) -> list[dict[str, Any]]:
        rows = self._run(
            "MATCH (a:Claim)-[r:CONTRADICTS|SUPPORTS|RELATED_TO]->(b:Claim) "
            "WHERE $ids IS NULL OR a.claim_id IN $ids OR b.claim_id IN $ids RETURN properties(r) AS props",
            ids=list(claim_ids) if claim_ids is not None else None,
        )
        return [r["props"] for r in rows]

    def supersession_edges(self) -> list[tuple[str, str]]:
        rows = self._run("MATCH (a:Document)-[:SUPERSEDES]->(b:Document) RETURN a.document_id AS a, b.document_id AS b")
        return sorted((r["a"], r["b"]) for r in rows)

    def stats(self) -> dict[str, int]:
        row = self._run(
            """
            CALL () { MATCH (d:Document) RETURN count(d) AS documents }
            CALL () { MATCH (c:Chunk) RETURN count(c) AS chunks }
            CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
            CALL () { MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS relationships }
            CALL () { MATCH (c:Claim) RETURN count(c) AS claims }
            CALL () { MATCH (:Claim)-[r]->(:Claim) RETURN count(r) AS claim_relationships }
            CALL () { MATCH (:Document)-[r:SUPERSEDES]->() RETURN count(r) AS supersessions }
            RETURN documents, chunks, entities, relationships, claims, claim_relationships, supersessions
            """
        )
        return row[0] if row else {}

