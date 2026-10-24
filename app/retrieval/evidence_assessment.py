"""Evidence sufficiency assessment.

A high similarity score only says a passage is *about* the topic, not that it
answers the question. Sufficiency therefore combines four signals:

* relevance        – best cross-encoder relevance among retrieved passages
* entity coverage  – query entities (or graph-equivalent entities) present in relevant passages
* concept coverage – the question's non-entity concepts ("retention", "access") present
* claim support    – an extracted normative claim in a relevant passage that is about the
                     query's entities and concepts

and two hard gates (at least one passage above ``min_chunk_relevance`` and concept
coverage above ``min_concept_coverage``). The concept gate is lexical, so it is waived
when the cross-encoder relevance reaches ``strong_relevance`` and every query entity is
covered — the cross-encoder then acts as the evidence-relevance verifier. Thresholds are calibrated on the
evaluation set by ``scripts/calibrate_thresholds.py``.
"""

from __future__ import annotations

import re

from app.config import Settings
from app.graph.entity_extraction import Gazetteer
from app.models.schemas import Claim, EvidenceAssessment, QueryAnalysis, RetrievedChunk
from app.utils.text import content_terms, tokenize


def _concept_variants(concept: str, gazetteer: Gazetteer) -> set[str]:
    """A concept is covered by itself or by any catalog synonym of it."""
    variants = {concept}
    for phrase, synonyms in gazetteer.synonyms.items():
        if concept in tokenize(phrase):
            for synonym in synonyms:
                variants.update(tokenize(synonym))
    return variants


def _display_terms(stems: list[str], query: str) -> list[str]:
    """Map internal stems back to the words the user typed ("leav" -> "leave") for messages."""
    words = {stem: word for word in tokenize(query, stemmed=False) for stem in tokenize(word)}
    return [words.get(s, s) for s in stems]


def _mentions(text: str, names: set[str]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", lowered) for n in names if n)


def assess_evidence(
    analysis: QueryAnalysis,
    evidence: list[RetrievedChunk],
    claims: list[Claim],
    settings: Settings,
    gazetteer: Gazetteer,
    entity_equivalents: dict[str, set[str]] | None = None,
) -> EvidenceAssessment:
    reasons: list[str] = []
    relevant = [e for e in evidence if e.relevance >= settings.min_chunk_relevance]
    relevance = max((e.relevance for e in evidence), default=0.0)
    relevant_text = " ".join(e.chunk.text for e in relevant)
    relevant_terms = content_terms(relevant_text)

    # Entity coverage ---------------------------------------------------------
    equivalents = entity_equivalents or {}
    missing_entities: list[str] = []
    for entity in analysis.entities:
        names = equivalents.get(entity.entity_id) or {entity.name.lower(), *[a.lower() for a in entity.aliases]}
        if not _mentions(relevant_text, names):
            missing_entities.append(entity.name)
    entity_coverage = (
        1.0 - len(missing_entities) / len(analysis.entities) if analysis.entities else None
    )

    # Concept coverage ----------------------------------------------------------
    missing_concepts = [
        c for c in analysis.concepts if not (_concept_variants(c, gazetteer) & relevant_terms)
    ]
    concept_coverage = 1.0 - len(missing_concepts) / len(analysis.concepts) if analysis.concepts else None

    # Claim support ---------------------------------------------------------------
    relevant_ids = {e.chunk_id for e in relevant}
    entity_names: set[str] = set()
    for entity in analysis.entities:
        entity_names |= equivalents.get(entity.entity_id) or {entity.name.lower(), *[a.lower() for a in entity.aliases]}
    concept_variants: set[str] = set()
    for concept in analysis.concepts:
        concept_variants |= _concept_variants(concept, gazetteer)
    supporting_claims = [
        c for c in claims
        if c.chunk_id in relevant_ids
        and (not entity_names or _mentions(c.text, entity_names))
        and (not concept_variants or concept_variants & content_terms(c.text))
    ]
    claim_support = 1.0 if supporting_claims else 0.0

    # Weighted score (weights of signals that do not apply are redistributed) ---
    signals = [
        (settings.weight_relevance, relevance),
        (settings.weight_entity_coverage, entity_coverage),
        (settings.weight_concept_coverage, concept_coverage),
        (settings.weight_claim_support, claim_support),
    ]
    applicable = [(w, v) for w, v in signals if v is not None]
    total_weight = sum(w for w, _ in applicable) or 1.0
    score = sum(w * v for w, v in applicable) / total_weight

    sufficient = True
    if not relevant:
        sufficient = False
        reasons.append(f"No passage reached the relevance threshold ({settings.min_chunk_relevance:.2f}).")
    strong_match = relevance >= settings.strong_relevance and not missing_entities
    if concept_coverage is not None and concept_coverage < settings.min_concept_coverage:
        if strong_match:
            reasons.append(
                f"Concept terms ({', '.join(_display_terms(missing_concepts, analysis.query))}) not found lexically, but cross-encoder "
                f"relevance {relevance:.2f} confirms the passage addresses the question."
            )
        else:
            sufficient = False
            reasons.append("Relevant passages do not address: " + ", ".join(_display_terms(missing_concepts, analysis.query)) + ".")
    if score < settings.sufficiency_threshold:
        sufficient = False
        reasons.append(f"Combined evidence score {score:.2f} is below threshold {settings.sufficiency_threshold:.2f}.")
    if missing_entities:
        reasons.append("Entities not covered by evidence: " + ", ".join(missing_entities) + ".")
    if sufficient:
        reasons.append(f"Evidence sufficient (score {score:.2f}, {len(relevant)} relevant passage(s)).")

    return EvidenceAssessment(
        sufficient=sufficient,
        score=round(score, 4),
        relevance=round(relevance, 4),
        entity_coverage=round(entity_coverage if entity_coverage is not None else 1.0, 4),
        concept_coverage=round(concept_coverage if concept_coverage is not None else 1.0, 4),
        claim_support=claim_support,
        relevant_chunk_count=len(relevant),
        missing_entities=missing_entities,
        missing_concepts=missing_concepts,
        reasons=reasons,
    )
