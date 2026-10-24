"""Prompt templates and the structured output schemas the LLM must return."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.schemas import RetrievedChunk

SYSTEM_RULES = """You are an enterprise knowledge assistant. Follow these rules strictly:
- Answer ONLY with facts stated in the evidence passages below. Do not use outside knowledge.
- Every claim must cite one or more evidence labels exactly as given (e.g. "E2"). Never invent labels,
  document IDs, page numbers or titles.
- Keep each claim to a single verifiable statement, close to the wording of the evidence.
- If the evidence does not answer the question, return no claims and set insufficient_evidence to true.
- If the notes report an UNRESOLVED conflict, present both rules with their citations and do not pick one.
- If the notes report a RESOLVED conflict, answer with the applicable rule only.
- If the notes report a GRAPH relationship (e.g. "X is a Y") and the passages state rules for Y,
  state the relationship as its own claim citing the passage that establishes it, then state each rule
  using the passage's own wording for Y, citing the rule's passage. Never merge both into one sentence about X.
  Right: "X is a Y." [relationship passage] + "Access to a Y requires approval." [rule passage]
  Wrong: "Access to X requires approval." (no single passage says this)
"""

ANSWER_PROMPT = """{rules}
Question: {question}

Evidence passages:
{evidence}

Notes (conflicts and graph relationships):
{conflicts}

Return JSON with "claims" (list of {{"text", "citation_ids"}}), "insufficient_evidence" (bool) and "notes" (short string or null).
"""

CORRECTION_PROMPT = """{rules}
A previous answer contained statements that the evidence does not support.

Question: {question}

Evidence passages:
{evidence}

Notes (conflicts and graph relationships):
{conflicts}

Previous claims and verification results:
{feedback}

Rewrite the answer: keep every claim marked [supported] exactly as it is, fix citations, and remove or
rewrite unsupported claims using only the evidence (split claims that combine two passages). If the evidence cannot answer the question, return no claims and set insufficient_evidence to true.
Return the same JSON format.
"""

BASELINE_PROMPT = """Answer the question using the context passages. Cite passage labels (e.g. "E1") for each claim.

Question: {question}

Context:
{evidence}

Return JSON with "claims" (list of {{"text", "citation_ids"}}), "insufficient_evidence" (bool) and "notes".
"""

VERIFY_PROMPT = """Does the passage fully support the claim? Answer "supported" only if every part of the claim,
including numbers and conditions, is stated in the passage; "contradicted" if the passage states the opposite;
otherwise "unsupported".

Claim: {claim}

Passage: {passage}
"""

CLAIM_EXTRACTION_PROMPT = """Extract normative or factual claims (rules, requirements, permissions, prohibitions,
durations, limits) from the passage. Copy each claim's sentence verbatim from the passage. Name the subject
(system, data type or group it applies to) and list any applicability conditions.

Passage:
{passage}
"""


class GeneratedClaimSchema(BaseModel):
    text: str
    citation_ids: list[str] = Field(default_factory=list)


class AnswerSchema(BaseModel):
    claims: list[GeneratedClaimSchema] = Field(default_factory=list)
    insufficient_evidence: bool = False
    notes: str | None = None


class SupportJudgement(BaseModel):
    label: str
    explanation: str | None = None


class ExtractedClaimItem(BaseModel):
    text: str
    subject: str = ""
    conditions: list[str] = Field(default_factory=list)


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaimItem] = Field(default_factory=list)


def format_evidence_item(label: str, item: RetrievedChunk, *, for_judge: bool = False) -> str:
    """One passage with its document metadata header, as shown to the generator.

    ``for_judge=True`` (evaluation contexts for Ragas) also includes the document ID and
    supersession metadata, which the system's deterministic conflict notes are built from, so
    a passage-only judge can check those notes instead of counting them as unsupported.
    """
    doc = item.document
    meta = [f"{doc.title} ({doc.document_id})" if for_judge else doc.title, f"status: {doc.status.value}"]
    if doc.version:
        meta.append(f"version {doc.version}")
    if doc.effective_date:
        meta.append(f"effective {doc.effective_date.isoformat()}")
    if for_judge and doc.supersedes:
        meta.append(f"supersedes {', '.join(doc.supersedes)}")
    if item.chunk.section:
        meta.append(f"section: {item.chunk.section}")
    return f"[{label}] ({'; '.join(meta)})\n{item.chunk.text}"


def format_evidence(evidence: dict[str, RetrievedChunk]) -> str:
    blocks = [format_evidence_item(label, item) for label, item in evidence.items()]
    return "\n\n".join(blocks) if blocks else "(no passages)"
