"""Neo4j constraints and indexes. All statements are idempotent (IF NOT EXISTS)."""

from __future__ import annotations

NODE_LABELS = ("Document", "Chunk", "Claim", "Policy", "Department", "System", "Procedure", "Concept")
ENTITY_LABELS = ("Policy", "Department", "System", "Procedure", "Concept")

VECTOR_INDEX_NAME = "chunk_embedding"
FULLTEXT_INDEX_NAME = "chunk_text"


def schema_statements(embedding_dimension: int) -> list[str]:
    if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be a positive integer")
    return [
        # Uniqueness constraints give stable IDs and prevent duplicate nodes on re-ingestion.
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE",
        # Every typed entity also carries the generic :Entity label keyed by entity_id.
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX chunk_document IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
        "CREATE INDEX claim_subject_property IF NOT EXISTS FOR (c:Claim) ON (c.subject_id, c.property)",
        "CREATE INDEX claim_chunk IF NOT EXISTS FOR (c:Claim) ON (c.chunk_id)",
        "CREATE INDEX document_status IF NOT EXISTS FOR (d:Document) ON (d.status)",
        (
            f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dimension}, "
            "`vector.similarity_function`: 'cosine'}}"
        ),
        f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
    ]
