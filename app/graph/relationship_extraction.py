"""Relationship extraction from document metadata and explicit sentence patterns.

Two sources, both with provenance:

* Front matter (``governs``, ``applies_to``, ``depends_on``, ``related``) — the
  most reliable signal, authored with the document.
* Sentences with explicit trigger phrases ("This standard governs ...",
  "... depends on ...", "X owns Y", "X is a production database").
"""

from __future__ import annotations

import logging
import re

from app.graph.entity_extraction import Gazetteer, document_entity, entity_id_for
from app.models.schemas import Chunk, DocumentMetadata, Entity, Relation
from app.utils.text import split_sentences

logger = logging.getLogger(__name__)

ENTITY_RELATIONSHIP_TYPES = ("GOVERNS", "APPLIES_TO", "DEPENDS_ON", "OWNED_BY", "RELATED_TO")

_SELF_REFERENCE_RE = re.compile(r"\bthis (policy|standard|procedure|document|guideline|guidelines)\b", re.I)

# (trigger regex, relation type, direction). "forward": left -> right entities;
# "owner": left is the owner, right entities are OWNED_BY it; "owned_by": left
# entities are OWNED_BY the right-hand department.
_TRIGGERS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bgoverns?\b|\bis governed by\b", re.I), "GOVERNS", "forward"),
    (re.compile(r"\bdepends? on\b|\bis based on\b|\bin accordance with\b|\bimplements\b|\bas defined (in|by)\b", re.I), "DEPENDS_ON", "forward"),
    (re.compile(r"\bappl(?:y|ies) to\b", re.I), "APPLIES_TO", "forward"),
    (re.compile(r"\bis owned by\b|\bare owned by\b|\bowned by\b", re.I), "OWNED_BY", "owned_by"),
    (re.compile(r"\bowns?\b|\bis responsible for\b|\bmaintains\b", re.I), "OWNED_BY", "owner"),
    (re.compile(r"\bis (?:a|an|one of the|part of the|part of)\b", re.I), "RELATED_TO", "forward"),
]


def _make(source: Entity, target: Entity, rel_type: str, metadata: DocumentMetadata, chunk: Chunk | None, evidence: str) -> Relation | None:
    if source.entity_id == target.entity_id:
        return None
    return Relation(
        source_id=source.entity_id,
        target_id=target.entity_id,
        type=rel_type,
        document_id=metadata.document_id,
        chunk_id=chunk.chunk_id if chunk else None,
        evidence=evidence[:300],
    )


def _resolve_entity(name: str, gazetteer: Gazetteer, default_label: str) -> Entity:
    found = gazetteer.lookup(name)
    if found:
        return found
    # Front matter may introduce an entity that the catalog does not know yet.
    return gazetteer.add(Entity(entity_id=entity_id_for(default_label, name), name=name, label=default_label, aliases=[name.lower()]))


def relations_from_metadata(
    metadata: DocumentMetadata, gazetteer: Gazetteer, first_chunk: Chunk | None
) -> tuple[list[Entity], list[Relation]]:
    """GOVERNS/APPLIES_TO/related relations authored in front matter.

    ``depends_on`` and ``supersedes`` reference other *documents* and are linked
    by :func:`cross_document_relations` once both documents are indexed.
    """
    self_entity = document_entity(metadata)
    entities: dict[str, Entity] = {}
    relations: list[Relation] = []
    evidence = f"Document metadata of {metadata.document_id}"
    if self_entity is not None:
        self_entity = gazetteer.add(self_entity)
        entities[self_entity.entity_id] = self_entity
        if metadata.department:
            dept = _resolve_entity(metadata.department, gazetteer, "Department")
            entities[dept.entity_id] = dept
            rel = _make(self_entity, dept, "OWNED_BY", metadata, first_chunk, evidence)
            relations += [rel] if rel else []
        for field_name, rel_type, label in (("governs", "GOVERNS", "System"), ("applies_to", "APPLIES_TO", "Concept")):
            for name in getattr(metadata, field_name):
                target = _resolve_entity(name, gazetteer, label)
                entities[target.entity_id] = target
                rel = _make(self_entity, target, rel_type, metadata, first_chunk, evidence)
                relations += [rel] if rel else []
    for spec in metadata.related:
        parts = [p.strip() for p in spec.split("|")]
        if len(parts) != 3 or parts[1].upper() not in ENTITY_RELATIONSHIP_TYPES:
            metadata.metadata_warnings.append(f"Ignored malformed related entry '{spec}'.")
            continue
        source = _resolve_entity(parts[0], gazetteer, "System")
        target = _resolve_entity(parts[2], gazetteer, "System")
        entities[source.entity_id] = source
        entities[target.entity_id] = target
        rel = _make(source, target, parts[1].upper(), metadata, first_chunk, evidence)
        relations += [rel] if rel else []
    return list(entities.values()), relations


def relations_from_text(metadata: DocumentMetadata, chunks: list[Chunk], gazetteer: Gazetteer) -> list[Relation]:
    self_entity = document_entity(metadata)
    if self_entity is not None:
        self_entity = gazetteer.add(self_entity)
    relations: list[Relation] = []
    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            for pattern, rel_type, direction in _TRIGGERS:
                match = pattern.search(sentence)
                if not match:
                    continue
                left_text, right_text = sentence[: match.start()], sentence[match.end():]
                left = [e for e, _, _ in gazetteer.find(left_text)]
                right = [e for e, _, _ in gazetteer.find(right_text)]
                if _SELF_REFERENCE_RE.search(left_text) and self_entity is not None:
                    left = [self_entity]
                if not left or not right:
                    continue
                if direction == "forward":
                    pairs = [(left[-1], target) for target in right]
                elif direction == "owner":
                    owners = [e for e in left if e.label == "Department"]
                    pairs = [(owned, owners[-1]) for owned in right if owned.label != "Department"] if owners else []
                else:  # owned_by
                    depts = [e for e in right if e.label == "Department"]
                    pairs = [(owned, depts[0]) for owned in left if owned.label != "Department"] if depts else []
                if rel_type == "RELATED_TO":
                    pairs = pairs[:1]  # "X is a Y ..." links only to the first object
                for source, target in pairs:
                    rel = _make(source, target, rel_type, metadata, chunk, sentence)
                    if rel:
                        relations.append(rel)
                break  # one trigger per sentence keeps extraction predictable
    return relations


def cross_document_relations(documents: list[DocumentMetadata], gazetteer: Gazetteer) -> tuple[list[Relation], list[tuple[str, str]], list[str]]:
    """Resolve ``depends_on`` and ``supersedes`` references between indexed documents.

    Returns (entity relations, document supersession edges, warnings). References
    to documents that are not indexed are reported, never invented.
    """
    by_id = {d.document_id: d for d in documents}
    relations: list[Relation] = []
    supersessions: list[tuple[str, str]] = []
    warnings: list[str] = []
    for doc in documents:
        source = document_entity(doc)
        for ref in doc.depends_on:
            target_entity: Entity | None = None
            if ref in by_id:
                target_entity = document_entity(by_id[ref])
            else:
                target_entity = gazetteer.lookup(ref)
            if source is None or target_entity is None:
                warnings.append(
                    f"{doc.document_id}: depends_on '{ref}' is not indexed yet; the link is created when it is ingested."
                )
                continue
            gazetteer.add(target_entity)
            rel = _make(source, target_entity, "DEPENDS_ON", doc, None, f"Document metadata of {doc.document_id}")
            if rel:
                relations.append(rel)
        for ref in doc.supersedes:
            if ref in by_id:
                supersessions.append((doc.document_id, ref))
            else:
                warnings.append(f"{doc.document_id}: supersedes '{ref}', which is not indexed.")
    return relations, supersessions, warnings
