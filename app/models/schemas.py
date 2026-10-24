"""Pydantic models shared across ingestion, retrieval, reliability and the API."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Documents and chunks
# ---------------------------------------------------------------------------


class DocumentStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DRAFT = "draft"
    ARCHIVED = "archived"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    document_id: str
    title: str
    doc_type: str = "document"  # policy | standard | procedure | guideline | technical | register | document
    department: str | None = None
    version: str | None = None
    publication_date: date | None = None
    effective_date: date | None = None
    status: DocumentStatus = DocumentStatus.UNKNOWN
    authority: str | None = None  # key into Settings.authority_levels
    supersedes: list[str] = Field(default_factory=list)
    governs: list[str] = Field(default_factory=list)
    applies_to: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)  # "A | RELATION | B" entity relations
    filename: str | None = None
    source_format: str | None = None
    metadata_warnings: list[str] = Field(default_factory=list)


class ParsedPage(BaseModel):
    text: str
    page_number: int | None = None  # None when the format has no pages (TXT/DOCX)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    page: int | None = None
    section: str | None = None
    content_hash: str


class Entity(BaseModel):
    entity_id: str
    name: str
    label: str  # Policy | System | Department | Procedure | Concept
    aliases: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    source_id: str
    target_id: str
    type: str  # GOVERNS | APPLIES_TO | DEPENDS_ON | OWNED_BY | RELATED_TO | ...
    document_id: str
    chunk_id: str | None = None
    evidence: str | None = None


class Mention(BaseModel):
    chunk_id: str
    entity_id: str


class Polarity(str, Enum):
    REQUIRE = "require"
    PERMIT = "permit"
    PROHIBIT = "prohibit"
    STATE = "state"  # descriptive statement without deontic force


class Claim(BaseModel):
    claim_id: str
    document_id: str
    chunk_id: str
    subject_id: str | None = None
    subject: str | None = None
    property: str | None = None
    text: str
    polarity: Polarity = Polarity.STATE
    quantity: float | None = None
    unit: str | None = None  # normalised unit dimension, e.g. "minutes", "characters"
    conditions: list[str] = Field(default_factory=list)
    page: int | None = None
    section: str | None = None
    # Provenance metadata copied from the parent document so that conflict
    # resolution never has to guess.
    publication_date: date | None = None
    effective_date: date | None = None
    version: str | None = None
    status: DocumentStatus = DocumentStatus.UNKNOWN
    authority: str | None = None
    document_title: str | None = None
    extraction_method: str = "rule"


class DocumentRecord(BaseModel):
    """What the store returns about an indexed document."""

    metadata: DocumentMetadata
    chunk_count: int = 0
    claim_count: int = 0
    indexing_status: str = "indexed"


class IngestionResult(BaseModel):
    document_id: str
    title: str
    chunk_count: int
    entity_count: int
    relation_count: int
    claim_count: int
    conflicts_detected: int
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    chunk: Chunk
    document: DocumentMetadata
    vector_score: float | None = None
    keyword_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None  # probability-like relevance in [0, 1]
    sources: list[str] = Field(default_factory=list)  # vector | keyword | graph | expansion

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def relevance(self) -> float:
        return self.rerank_score if self.rerank_score is not None else (self.rrf_score or 0.0)


class QueryAnalysis(BaseModel):
    query: str
    entities: list[Entity] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)  # important non-entity terms
    relational: bool = False  # asks about relationships (depends on / governs / owns ...)


class EvidenceAssessment(BaseModel):
    sufficient: bool
    score: float
    relevance: float
    entity_coverage: float
    concept_coverage: float
    claim_support: float
    relevant_chunk_count: int
    missing_entities: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class RetrievalOutcome(str, Enum):
    INITIAL_SUFFICIENT = "initial_sufficient"
    RECOVERED = "recovered"  # document existed but was missed by initial retrieval
    DOCUMENTS_WITHOUT_ANSWER = "documents_without_answer"  # topic present, answer absent
    NOT_IN_CORPUS = "not_in_corpus"  # no indexed document covers the topic
    INSUFFICIENT_AFTER_RECOVERY = "insufficient_after_recovery"
    RECOVERY_DISABLED = "recovery_disabled"


class GraphPath(BaseModel):
    nodes: list[str]  # entity names along the path
    node_ids: list[str] = Field(default_factory=list)
    # Relationship types in path order; a leading "<" means the edge points backwards.
    relationships: list[str]
    document_ids: list[str] = Field(default_factory=list)

    def describe(self) -> str:
        parts = [self.nodes[0]]
        for rel, node in zip(self.relationships, self.nodes[1:]):
            parts.append(f"<-[{rel[1:]}]- {node}" if rel.startswith("<") else f"-[{rel}]-> {node}")
        return " ".join(parts)


class RecoveryAttempt(BaseModel):
    attempt: int
    expanded_query: str
    reformulated_queries: list[str] = Field(default_factory=list)
    graph_entities: list[str] = Field(default_factory=list)
    graph_documents: list[str] = Field(default_factory=list)
    new_chunk_ids: list[str] = Field(default_factory=list)
    assessment: EvidenceAssessment


class RetrievalResult(BaseModel):
    analysis: QueryAnalysis
    evidence: list[RetrievedChunk]
    initial_evidence_chunk_ids: list[str] = Field(default_factory=list)
    initial_assessment: EvidenceAssessment
    final_assessment: EvidenceAssessment
    outcome: RetrievalOutcome
    recovery_attempts: list[RecoveryAttempt] = Field(default_factory=list)
    graph_paths: list[GraphPath] = Field(default_factory=list)
    graph_used: bool = False

    @property
    def recovery_triggered(self) -> bool:
        return bool(self.recovery_attempts)


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------


class ClaimRelationType(str, Enum):
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    COMPATIBLE_CONDITIONAL = "COMPATIBLE_CONDITIONAL"
    UNRELATED = "UNRELATED"


class ClaimComparison(BaseModel):
    claim_a: Claim
    claim_b: Claim
    relation: ClaimRelationType
    method: str  # structured | nli | llm
    confidence: float = 1.0
    explanation: str


class ConflictResolution(BaseModel):
    claim_a_id: str
    claim_b_id: str
    resolved: bool
    winner_claim_id: str | None = None
    winner_document_id: str | None = None
    rule: str | None = None
    explanation: str
    rules_considered: list[str] = Field(default_factory=list)


class ConflictReport(BaseModel):
    comparison: ClaimComparison
    resolution: ConflictResolution


# ---------------------------------------------------------------------------
# Generation and verification
# ---------------------------------------------------------------------------


class GeneratedClaim(BaseModel):
    text: str
    citation_ids: list[str] = Field(default_factory=list)  # evidence labels such as "E1"


class GeneratedAnswer(BaseModel):
    claims: list[GeneratedClaim] = Field(default_factory=list)
    insufficient_evidence: bool = False
    notes: str | None = None


class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    INVALID_CITATION = "invalid_citation"


class ClaimVerification(BaseModel):
    claim: str
    status: VerificationStatus
    cited_ids: list[str] = Field(default_factory=list)
    valid_citation_ids: list[str] = Field(default_factory=list)
    supporting_ids: list[str] = Field(default_factory=list)
    support_score: float = 0.0
    method: str = "lexical"
    note: str | None = None


class VerificationReport(BaseModel):
    verifications: list[ClaimVerification] = Field(default_factory=list)
    method: str

    @property
    def supported(self) -> list[ClaimVerification]:
        return [v for v in self.verifications if v.status == VerificationStatus.SUPPORTED]

    @property
    def all_supported(self) -> bool:
        return bool(self.verifications) and len(self.supported) == len(self.verifications)


class Citation(BaseModel):
    citation_id: int
    document_id: str
    document_title: str
    chunk_id: str
    page: int | None = None
    section: str | None = None
    excerpt: str
    document_status: str | None = None
    version: str | None = None


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    mode: str = Field("improved", pattern="^(improved|baseline)$")
    top_k: int | None = Field(None, ge=1, le=20)
    department: str | None = None
    doc_type: str | None = None


class ConflictSummary(BaseModel):
    subject: str | None
    property: str | None
    claim_a: str
    claim_a_document: str
    claim_b: str
    claim_b_document: str
    resolved: bool
    applicable_document: str | None = None
    rule: str | None = None
    explanation: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    recovery_performed: bool = False
    conflict_detected: bool = False
    conflicts: list[ConflictSummary] = Field(default_factory=list)
    verified: bool = False
    abstained: bool = False
    retrieval_outcome: str | None = None
    reliability_notes: list[str] = Field(default_factory=list)
    graph_paths: list[str] = Field(default_factory=list)
    llm_provider: str
    mock_mode: bool = False
    mode: str = "improved"
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_contexts: list[str] = Field(default_factory=list)
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: str
    graph_backend: str
    graph_connected: bool
    llm_provider: str
    mock_mode: bool
    embedding_backend: str
    reranker: str
    verifier: str
    documents_indexed: int
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    configurations: list[str] = Field(default_factory=lambda: ["baseline", "improved"])
    include_ragas: bool = False
    limit: int | None = Field(None, ge=1)
    categories: list[str] | None = None
