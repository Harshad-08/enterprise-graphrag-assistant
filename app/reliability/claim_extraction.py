"""Claim extraction: turn normative sentences into structured, provenance-carrying claims.

The rule-based extractor recognises deontic markers (must / must not / may /
prohibited / required ...), quantities with units, applicability conditions
("only when ...", "if ...", "unless ..."), the subject entity (via the entity
catalog) and a normalised property key (via ``ontology.json`` "properties").

Limitations (documented): sentences without a catalog entity fall back to the
document's own policy entity as subject; properties not in the lexicon are left
empty, so those claims are compared only semantically (NLI), not structurally.
An optional LLM extractor can add claims, but every LLM claim must be found
verbatim in its chunk (provenance check) or it is discarded.
"""

from __future__ import annotations

import logging
import re

from app.graph.entity_extraction import Gazetteer, document_entity
from app.models.schemas import Chunk, Claim, DocumentMetadata, Polarity
from app.utils.text import split_sentences

logger = logging.getLogger(__name__)

_PROHIBIT_RE = re.compile(
    r"\b(must not|shall not|may not|cannot|can ?not|is not permitted|are not permitted|not allowed|"
    r"is prohibited|are prohibited|prohibited|forbidden|never|must never)\b",
    re.I,
)
_REQUIRE_RE = re.compile(r"\b(must|shall|is required|are required|required|requires|mandatory|needs? to)\b", re.I)
_PERMIT_RE = re.compile(r"\b(may|is permitted|are permitted|permitted|is allowed|are allowed|allowed|can|eligible)\b", re.I)
_CONDITION_RE = re.compile(
    r"\b(only if|only when|if|when|unless|except|provided that|as long as|while|during)\b([^.;]*)", re.I
)

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirty": 30, "ninety": 90,
}
_QUANTITY_RE = re.compile(
    r"\b(\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")\s*(?:\(\d+\)\s*)?"
    r"(business days?|working days?|minutes?|mins?|hours?|days?|weeks?|months?|years?|characters?|"
    r"requests per minute)(\s+per\s+week|\s+a\s+week)?",
    re.I,
)
# Time is normalised to minutes so "12 months" and "90 days" are comparable.
_TO_MINUTES = {"minute": 1, "min": 1, "hour": 60, "day": 1440, "week": 10080, "month": 43200, "year": 525600}


def parse_quantity(sentence: str) -> tuple[float | None, str | None]:
    match = _QUANTITY_RE.search(sentence)
    if not match:
        return None, None
    raw, unit, per_week = match.group(1).lower(), match.group(2).lower(), match.group(3)
    value = float(_NUMBER_WORDS.get(raw, raw))
    unit = unit.rstrip("s") if not unit.endswith("days") else unit[:-1]
    if per_week:
        return value, f"{unit}s_per_week"
    if unit in ("business day", "working day"):
        return value, "business_days"
    if unit == "character":
        return value, "characters"
    if unit.startswith("requests per minute"):
        return value, "requests_per_minute"
    if unit in _TO_MINUTES:
        return value * _TO_MINUTES[unit], "minutes"
    return value, unit


def detect_polarity(sentence: str) -> Polarity:
    if _PROHIBIT_RE.search(sentence):
        return Polarity.PROHIBIT
    if _REQUIRE_RE.search(sentence):
        return Polarity.REQUIRE
    if _PERMIT_RE.search(sentence):
        return Polarity.PERMIT
    return Polarity.STATE


def extract_conditions(sentence: str) -> list[str]:
    conditions = []
    for match in _CONDITION_RE.finditer(sentence):
        clause = f"{match.group(1).lower()}{match.group(2)}".strip(" ,")
        # Ignore "when" used for timing in a requirement ("... within 1 hour when discovered").
        if len(clause.split()) >= 3:
            conditions.append(clause)
    return conditions


def normalize_condition(condition: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", condition.lower()))


def match_property(sentence: str, properties: list[dict]) -> str | None:
    lowered = sentence.lower()
    for prop in properties:
        if all(keyword in lowered for keyword in prop["all"]):
            return prop["key"]
    return None


def _subject(sentence: str, gazetteer: Gazetteer, fallback):
    entities = [e for e, _, _ in gazetteer.find(sentence)]
    for preferred in (("System", "Concept", "Procedure"), ("Policy",)):
        for entity in entities:
            if entity.label in preferred:
                return entity
    return fallback


def is_claim_sentence(sentence: str, polarity: Polarity, quantity: float | None, prop: str | None) -> bool:
    if len(sentence) < 15 or sentence.endswith("?"):
        return False
    return polarity != Polarity.STATE or (quantity is not None and prop is not None)


def extract_claims(metadata: DocumentMetadata, chunks: list[Chunk], gazetteer: Gazetteer) -> list[Claim]:
    fallback = document_entity(metadata)
    claims: list[Claim] = []
    for chunk in chunks:
        index = 0
        for sentence in split_sentences(chunk.text):
            polarity = detect_polarity(sentence)
            quantity, unit = parse_quantity(sentence)
            prop = match_property(sentence, gazetteer.properties)
            if not is_claim_sentence(sentence, polarity, quantity, prop):
                continue
            subject = _subject(sentence, gazetteer, fallback)
            claims.append(
                Claim(
                    claim_id=f"{chunk.chunk_id}::k{index:02d}",
                    document_id=metadata.document_id,
                    chunk_id=chunk.chunk_id,
                    subject_id=subject.entity_id if subject else None,
                    subject=subject.name if subject else None,
                    property=prop,
                    text=sentence,
                    polarity=polarity,
                    quantity=quantity,
                    unit=unit,
                    conditions=extract_conditions(sentence),
                    page=chunk.page,
                    section=chunk.section,
                    publication_date=metadata.publication_date,
                    effective_date=metadata.effective_date,
                    version=metadata.version,
                    status=metadata.status,
                    authority=metadata.authority,
                    document_title=metadata.title,
                    extraction_method="rule",
                )
            )
            index += 1
    return claims


def validate_claim_provenance(claim: Claim, chunk: Chunk) -> bool:
    """A claim is only trusted if its text actually occurs in its source chunk."""
    def norm(s: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", s.lower()))

    return claim.chunk_id == chunk.chunk_id and norm(claim.text) in norm(chunk.text)


def extract_claims_with_llm(llm, metadata: DocumentMetadata, chunks: list[Chunk], gazetteer: Gazetteer) -> list[Claim]:
    """Optional LLM extraction (EXTRACTION_BACKEND=llm). Claims failing provenance are dropped."""
    from app.generation.prompts import CLAIM_EXTRACTION_PROMPT, ExtractedClaims

    fallback = document_entity(metadata)
    claims: list[Claim] = []
    for chunk in chunks:
        try:
            result: ExtractedClaims = llm.generate_structured(
                CLAIM_EXTRACTION_PROMPT.format(passage=chunk.text), ExtractedClaims
            )
        except Exception as exc:
            logger.warning("LLM claim extraction failed for %s: %s", chunk.chunk_id, exc)
            continue
        for index, item in enumerate(result.claims):
            subject = gazetteer.lookup(item.subject) or _subject(item.text, gazetteer, fallback)
            quantity, unit = parse_quantity(item.text)
            claim = Claim(
                claim_id=f"{chunk.chunk_id}::l{index:02d}",
                document_id=metadata.document_id,
                chunk_id=chunk.chunk_id,
                subject_id=subject.entity_id if subject else None,
                subject=subject.name if subject else None,
                property=match_property(item.text, gazetteer.properties),
                text=item.text,
                polarity=detect_polarity(item.text),
                quantity=quantity,
                unit=unit,
                conditions=[c for c in item.conditions if c],
                page=chunk.page,
                section=chunk.section,
                publication_date=metadata.publication_date,
                effective_date=metadata.effective_date,
                version=metadata.version,
                status=metadata.status,
                authority=metadata.authority,
                document_title=metadata.title,
                extraction_method="llm",
            )
            if validate_claim_provenance(claim, chunk):
                claims.append(claim)
            else:
                logger.info("Discarded LLM claim without verbatim provenance: %r", item.text[:80])
    return claims
