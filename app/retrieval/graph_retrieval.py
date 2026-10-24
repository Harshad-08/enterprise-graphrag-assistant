"""Graph-aware retrieval: link query entities to Neo4j and traverse relationships.

This is what makes the system GraphRAG rather than "vectors stored in Neo4j":
traversal over GOVERNS / DEPENDS_ON / APPLIES_TO / OWNED_BY / RELATED_TO edges
finds documents connected to the question's entities even when they share no
vocabulary with the question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.graph.entity_extraction import Gazetteer
from app.graph.store import GraphStore
from app.models.schemas import Chunk, Entity, GraphPath, QueryAnalysis


@dataclass
class GraphExpansion:
    seeds: list[Entity]
    paths: list[GraphPath] = field(default_factory=list)
    related: dict[str, str] = field(default_factory=dict)  # entity_id -> name, ordered by hop distance
    document_ids: list[str] = field(default_factory=list)  # ordered by hop distance
    chunks: list[Chunk] = field(default_factory=list)


class GraphRetriever:
    def __init__(self, store: GraphStore, gazetteer: Gazetteer) -> None:
        self.store = store
        self.gazetteer = gazetteer

    def link_entities(self, analysis: QueryAnalysis) -> list[Entity]:
        """Query entities that actually exist in the indexed graph."""
        if not analysis.entities:
            return []
        present = {e.entity_id: e for e in self.store.get_entities([e.entity_id for e in analysis.entities])}
        return [present[e.entity_id] for e in analysis.entities if e.entity_id in present]

    def neighbours(self, seeds: list[Entity]) -> dict[str, list[Entity]]:
        """Entities one RELATED_TO hop away (e.g. Atlas Cluster -> Production Database).

        Only RELATED_TO is used: it encodes "is a / part of", so evidence about the
        neighbour is evidence about the seed. GOVERNS/DEPENDS_ON do not imply that.
        """
        result: dict[str, list[Entity]] = {}
        for seed in seeds:
            found: list[Entity] = []
            for path in self.store.traverse([seed.entity_id], max_hops=1, limit=50):
                if len(path.node_ids) == 2 and path.relationships[0] == "RELATED_TO":
                    neighbour = self.gazetteer.get(path.node_ids[1])
                    if neighbour and neighbour not in found:
                        found.append(neighbour)
            result[seed.entity_id] = found
        return result

    def equivalents(self, seeds: list[Entity]) -> dict[str, set[str]]:
        """Names/aliases of each seed and its RELATED_TO neighbours, used for entity coverage.

        Evidence about "Production Database" covers a question about "Atlas" when the
        graph says Atlas RELATED_TO Production Database.
        """
        neighbours = self.neighbours(seeds)
        result: dict[str, set[str]] = {}
        for seed in seeds:
            names = {seed.name.lower(), *[a.lower() for a in seed.aliases]}
            for neighbour in neighbours.get(seed.entity_id, []):
                names |= {neighbour.name.lower(), *[a.lower() for a in neighbour.aliases]}
            result[seed.entity_id] = names
        return result

    def expand(self, seeds: list[Entity], max_hops: int, max_chunks: int) -> GraphExpansion:
        expansion = GraphExpansion(seeds=seeds)
        if not seeds:
            return expansion
        seed_ids = [s.entity_id for s in seeds]
        paths = self.store.traverse(seed_ids, max_hops=max_hops, limit=200)
        expansion.paths = paths
        for path in sorted(paths, key=lambda p: len(p.relationships)):
            for node_id, name in zip(path.node_ids[1:], path.nodes[1:]):
                if node_id not in seed_ids:
                    expansion.related.setdefault(node_id, name)
            for doc_id in path.document_ids:
                if doc_id not in expansion.document_ids:
                    expansion.document_ids.append(doc_id)

        # Candidate passages, most specific first: passages that state a relationship
        # touching the seeds, passages that mention related entities, then passages
        # of documents reached through the traversal.
        related_ids = [eid for eid, _ in expansion.related.items()
                       if (e := self.gazetteer.get(eid)) is None or e.label != "Department"]
        chunks: dict[str, Chunk] = {}
        for batch in (
            self.store.relation_chunks(seed_ids, max_chunks),
            self.store.chunks_mentioning(related_ids, max_chunks),
            self.store.chunks_for_documents(expansion.document_ids, max_chunks),
        ):
            for chunk in batch:
                if len(chunks) >= max_chunks:
                    break
                chunks.setdefault(chunk.chunk_id, chunk)
        expansion.chunks = list(chunks.values())
        return expansion
