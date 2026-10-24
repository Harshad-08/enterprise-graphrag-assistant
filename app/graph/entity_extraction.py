"""Entity linking against the enterprise catalog (``data/ontology.json``) and the graph.

Rule-based linking is deliberately transparent: every mention is an exact
alias match on word boundaries (longest match wins). It will miss entities that
are not in the catalog or in document front matter; that limitation is
documented in the README.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.models.schemas import Chunk, DocumentMetadata, Entity, Mention, QueryAnalysis
from app.utils.text import STOPWORDS, slugify, stem, tokenize

DOC_TYPE_LABELS = {
    "policy": "Policy",
    "standard": "Policy",
    "guideline": "Policy",
    "guidelines": "Policy",
    "procedure": "Procedure",
    "sop": "Procedure",
}

_RELATIONAL_RE = re.compile(
    r"\b(depend\w*|govern\w*|own(s|ed|er|ership)?|responsible|establish\w*|relat\w*|"
    r"appl(y|ies)\s+to|which\s+(documents?|policies|procedures|standards|systems))\b",
    re.IGNORECASE,
)

# Words that carry little topical meaning in a question and should not count as
# "concepts" that the evidence must cover.
QUESTION_STOPWORDS = frozenset(
    {stem(w) for w in """what which who whom how does do is are can could would should tell explain
    describe give list find show say says said rule rules current currently exist exists existing
    there need needs know two different our we us use used using read get make allow allowed able
    happen happens mean means anyone someone people staff employee employees remain remains""".split()}
)


def entity_id_for(label: str, name: str) -> str:
    return f"{label.lower()}:{slugify(name)}"


def document_entity(metadata: DocumentMetadata) -> Entity | None:
    """Policies/standards/procedures are first-class graph entities defined by documents."""
    label = DOC_TYPE_LABELS.get(metadata.doc_type)
    if label is None:
        return None
    aliases = {metadata.title.lower(), metadata.document_id.lower()}
    # Users say "API authentication policy" for a document titled "... Standard".
    base = re.sub(r"\s+(policy|standard|guidelines?|procedure)$", "", metadata.title, flags=re.IGNORECASE)
    if base != metadata.title:
        for suffix in ("policy", "standard", "procedure", "guidelines"):
            aliases.add(f"{base.lower()} {suffix}")
    return Entity(
        entity_id=entity_id_for(label, metadata.title),
        name=metadata.title,
        label=label,
        aliases=sorted(aliases),
    )


@dataclass
class Gazetteer:
    entities: dict[str, Entity] = field(default_factory=dict)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    properties: list[dict] = field(default_factory=list)
    _pattern: re.Pattern | None = None
    _alias_index: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_ontology(cls, path: Path) -> "Gazetteer":
        data = json.loads(Path(path).read_text())
        gazetteer = cls(synonyms=data.get("synonyms", {}), properties=data.get("properties", []))
        for item in data.get("entities", []):
            gazetteer.add(
                Entity(
                    entity_id=entity_id_for(item["label"], item["name"]),
                    name=item["name"],
                    label=item["label"],
                    aliases=item.get("aliases", []),
                )
            )
        return gazetteer

    def copy(self) -> "Gazetteer":
        clone = Gazetteer(synonyms=dict(self.synonyms), properties=list(self.properties))
        for entity in self.entities.values():
            clone.add(entity)
        return clone

    def add(self, entity: Entity) -> Entity:
        existing = self.entities.get(entity.entity_id)
        if existing:
            merged = sorted(set(existing.aliases) | set(entity.aliases))
            entity = existing.model_copy(update={"aliases": merged})
        self.entities[entity.entity_id] = entity
        for alias in {entity.name.lower(), *[a.lower() for a in entity.aliases]}:
            # First registration wins so catalog entries are not hijacked by uploads.
            self._alias_index.setdefault(alias, entity.entity_id)
        self._pattern = None
        return entity

    def get(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def lookup(self, name: str) -> Entity | None:
        entity_id = self._alias_index.get(name.strip().lower())
        return self.entities.get(entity_id) if entity_id else None

    def _compiled(self) -> re.Pattern:
        if self._pattern is None:
            aliases = sorted(self._alias_index, key=len, reverse=True)
            alternation = "|".join(re.escape(a) for a in aliases) or r"(?!x)x"
            self._pattern = re.compile(rf"(?<![\w-])({alternation})(?![\w-])", re.IGNORECASE)
        return self._pattern

    def find(self, text: str) -> list[tuple[Entity, int, int]]:
        """Non-overlapping, longest-first alias matches in reading order."""
        matches = []
        for m in self._compiled().finditer(text):
            entity = self.entities[self._alias_index[m.group(1).lower()]]
            matches.append((entity, m.start(), m.end()))
        return matches

    def entities_in(self, text: str) -> list[Entity]:
        seen: dict[str, Entity] = {}
        for entity, _, _ in self.find(text):
            seen.setdefault(entity.entity_id, entity)
        return list(seen.values())


def extract_mentions(chunks: list[Chunk], gazetteer: Gazetteer) -> tuple[list[Entity], list[Mention]]:
    entities: dict[str, Entity] = {}
    mentions: list[Mention] = []
    for chunk in chunks:
        for entity in gazetteer.entities_in(chunk.text):
            entities[entity.entity_id] = entity
            mentions.append(Mention(chunk_id=chunk.chunk_id, entity_id=entity.entity_id))
    return list(entities.values()), mentions


def analyze_query(query: str, gazetteer: Gazetteer) -> QueryAnalysis:
    matches = gazetteer.find(query)
    entities: dict[str, Entity] = {}
    remainder = query
    for entity, start, end in sorted(matches, key=lambda m: m[1], reverse=True):
        entities.setdefault(entity.entity_id, entity)
        remainder = remainder[:start] + " " + remainder[end:]
    concepts: list[str] = []
    for token in tokenize(remainder):
        if token in QUESTION_STOPWORDS or token in STOPWORDS or len(token) < 3 or token.isdigit():
            continue
        if token not in concepts:
            concepts.append(token)
    return QueryAnalysis(
        query=query,
        entities=list(reversed(list(entities.values()))),
        concepts=concepts,
        relational=bool(_RELATIONAL_RE.search(query)),
    )


def expand_query(query: str, gazetteer: Gazetteer, related_names: list[str] | None = None) -> str:
    """Add catalog synonyms and graph-related entity names to the query text."""
    lowered = query.lower()
    additions: list[str] = []
    for phrase, synonyms in gazetteer.synonyms.items():
        if re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", lowered):
            additions.extend(synonyms)
    for entity in gazetteer.entities_in(query):
        additions.extend(a for a in entity.aliases[:3] if a.lower() not in lowered)
    additions.extend(related_names or [])
    unique: list[str] = []
    for term in additions:
        if term.lower() not in lowered and term.lower() not in (u.lower() for u in unique):
            unique.append(term)
    return f"{query} {' '.join(unique)}".strip()


def reformulate_query(query: str, gazetteer: Gazetteer, replacements: dict[str, list[str]], limit: int = 4) -> list[str]:
    """Rewrite the query by substitution (not appending), so rerankers see a fluent question.

    * graph: "access requirements for the Atlas cluster" + (Atlas RELATED_TO Production Database)
      -> "access requirements for the Production Database"
    * catalog synonyms: "days can staff telecommute" -> "days can staff remote work"
    """
    variants: list[str] = []
    for entity, start, end in gazetteer.find(query):
        for name in replacements.get(entity.entity_id, []):
            variants.append(f"{query[:start]}{name}{query[end:]}")
    for phrase, synonyms in gazetteer.synonyms.items():
        match = re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", query, re.IGNORECASE)
        if match and synonyms:
            variants.append(f"{query[:match.start()]}{synonyms[0]}{query[match.end():]}")
    unique = [v for i, v in enumerate(variants) if v != query and v not in variants[:i]]
    return unique[:limit]
