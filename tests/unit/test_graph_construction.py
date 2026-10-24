"""Entity linking, relationship extraction and graph store behaviour."""

from __future__ import annotations

import numpy as np

from app.config import PROJECT_ROOT
from app.graph.entity_extraction import Gazetteer, analyze_query, document_entity, reformulate_query
from app.graph.memory_store import InMemoryGraphStore
from app.graph.relationship_extraction import (
    cross_document_relations,
    relations_from_metadata,
    relations_from_text,
)
from app.models.schemas import Chunk, DocumentMetadata, DocumentStatus, Entity, Mention, Relation

GAZ = Gazetteer.from_ontology(PROJECT_ROOT / "data" / "ontology.json")


def _meta(**kw) -> DocumentMetadata:
    base = dict(document_id="DOC-A", title="Production Database Access Standard", doc_type="standard",
                status=DocumentStatus.ACTIVE)
    base.update(kw)
    return DocumentMetadata(**base)


def _chunk(text: str, doc: str = "DOC-A", idx: int = 0) -> Chunk:
    return Chunk(chunk_id=f"{doc}::c{idx:03d}", document_id=doc, text=text, chunk_index=idx, content_hash="h")


class TestEntityLinking:
    def test_longest_alias_wins(self):
        names = [e.name for e in GAZ.copy().entities_in("VPN sessions expire; the Corporate VPN is required.")]
        assert names == ["VPN Sessions", "Corporate VPN"]

    def test_document_entity_has_policy_aliases(self):
        entity = document_entity(DocumentMetadata(document_id="T-1", title="API Authentication Standard", doc_type="standard"))
        assert entity.label == "Policy"
        assert "api authentication policy" in entity.aliases
        assert document_entity(DocumentMetadata(document_id="T-2", title="Overview", doc_type="technical")) is None

    def test_query_analysis_separates_entities_and_concepts(self):
        analysis = analyze_query("What are the access requirements for the Atlas cluster?", GAZ)
        assert [e.name for e in analysis.entities] == ["Atlas Cluster"]
        assert "access" in analysis.concepts and "requir" in analysis.concepts
        assert analyze_query("Which procedure depends on the API policy?", GAZ).relational

    def test_reformulation_substitutes_related_entities_and_synonyms(self):
        variants = reformulate_query(
            "access requirements for the Atlas cluster", GAZ, {"system:atlas-cluster": ["Production Database"]}
        )
        assert "access requirements for the Production Database" in variants
        assert reformulate_query("Can staff telecommute?", GAZ, {}) == ["Can staff remote work?"]


class TestRelationshipExtraction:
    def test_metadata_relations_have_provenance(self):
        gaz = GAZ.copy()
        meta = _meta(department="Information Security", governs=["Production Database"],
                     related=["Atlas Cluster | RELATED_TO | Production Database", "bad entry"])
        entities, relations = relations_from_metadata(meta, gaz, _chunk("x"))
        triples = {(r.source_id, r.type, r.target_id) for r in relations}
        assert ("policy:production-database-access-standard", "GOVERNS", "system:production-database") in triples
        assert ("policy:production-database-access-standard", "OWNED_BY", "department:information-security") in triples
        assert ("system:atlas-cluster", "RELATED_TO", "system:production-database") in triples
        assert all(r.document_id == "DOC-A" and r.chunk_id == "DOC-A::c000" for r in relations)
        assert any("malformed" in w for w in meta.metadata_warnings)

    def test_text_patterns(self):
        gaz = GAZ.copy()
        chunk = _chunk(
            "This standard governs the Production Database. This standard implements the Information Security Policy. "
            "Platform Engineering owns the Atlas Cluster. Atlas is a production database cluster."
        )
        triples = {(r.source_id, r.type, r.target_id) for r in relations_from_text(_meta(), [chunk], gaz)}
        assert ("policy:production-database-access-standard", "GOVERNS", "system:production-database") in triples
        assert ("policy:production-database-access-standard", "DEPENDS_ON", "policy:information-security-policy") in triples
        assert ("system:atlas-cluster", "OWNED_BY", "department:platform-engineering") in triples
        assert ("system:atlas-cluster", "RELATED_TO", "system:production-database") in triples

    def test_cross_document_references_are_resolved_or_reported(self):
        a = DocumentMetadata(document_id="P-1", title="Partner Onboarding Procedure", doc_type="procedure", depends_on=["S-1", "MISSING-9"])
        b = DocumentMetadata(document_id="S-1", title="API Authentication Standard", doc_type="standard", supersedes=["S-0"])
        relations, supersessions, warnings = cross_document_relations([a, b], GAZ.copy())
        assert [(r.source_id, r.type, r.target_id) for r in relations] == [
            ("procedure:partner-onboarding-procedure", "DEPENDS_ON", "policy:api-authentication-standard")
        ]
        assert supersessions == []  # S-0 is not indexed, so no edge is invented
        assert any("MISSING-9" in w for w in warnings) and any("S-0" in w for w in warnings)


class TestMemoryStore:
    def _populate(self, store: InMemoryGraphStore) -> None:
        e_sys = Entity(entity_id="system:db", name="DB", label="System")
        e_pol = Entity(entity_id="policy:p", name="P", label="Policy")
        e_dept = Entity(entity_id="department:d", name="D", label="Department")
        e_other = Entity(entity_id="policy:other", name="Other", label="Policy")
        for doc_id, entities, relations in (
            ("A", [e_sys, e_pol, e_dept], [Relation(source_id="policy:p", target_id="system:db", type="GOVERNS", document_id="A", chunk_id="A::c000"),
                                           Relation(source_id="policy:p", target_id="department:d", type="OWNED_BY", document_id="A")]),
            ("B", [e_other, e_dept], [Relation(source_id="policy:other", target_id="department:d", type="OWNED_BY", document_id="B")]),
        ):
            chunk = _chunk(f"text of {doc_id}", doc=doc_id)
            store.upsert_document(
                DocumentMetadata(document_id=doc_id, title=doc_id), [chunk], np.ones((1, 4)), entities, relations,
                [Mention(chunk_id=chunk.chunk_id, entity_id=entities[0].entity_id)], [], None,
            )

    def test_upsert_is_idempotent(self):
        store = InMemoryGraphStore()
        self._populate(store)
        self._populate(store)
        assert store.stats()["documents"] == 2 and store.stats()["chunks"] == 2

    def test_traversal_does_not_pass_through_departments(self):
        store = InMemoryGraphStore()
        self._populate(store)
        reached = {p.node_ids[-1] for p in store.traverse(["system:db"], max_hops=3)}
        assert "policy:p" in reached and "department:d" in reached
        assert "policy:other" not in reached  # only reachable via the department hub

    def test_delete_removes_chunks_relations_and_orphans(self):
        store = InMemoryGraphStore()
        self._populate(store)
        assert store.delete_document("A")
        assert store.get_document("A") is None
        assert all(not c.startswith("A::") for c in store.chunks)
        assert ("policy:p", "GOVERNS", "system:db") not in store.relations
        assert "system:db" not in store.entities  # orphaned
        assert "department:d" in store.entities  # still referenced by B
        assert store.delete_document("A") is False
