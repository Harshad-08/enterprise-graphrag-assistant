"""Evaluation dataset models and loaders."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalItem(BaseModel):
    id: str
    category: str
    question: str
    reference_answer: str | None = None
    relevant_doc_ids: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    stale_facts: list[str] = Field(default_factory=list)  # facts from overridden/superseded rules
    should_abstain: bool = False
    expected_conflict: str | None = None  # None | "unresolved" | "resolved:<doc id>"
    expects_recovery: bool = False


class GoldConflict(BaseModel):
    documents: list[str]
    subject: str
    property: str | None = None
    expected_resolution: str
    expected_rule: str | None = None


class CompatiblePair(BaseModel):
    documents: list[str]
    subject: str
    reason: str = ""


class GoldConflicts(BaseModel):
    contradictions: list[GoldConflict]
    compatible_pairs: list[CompatiblePair] = Field(default_factory=list)


def load_eval_items(path: Path, categories: list[str] | None = None, limit: int | None = None) -> list[EvalItem]:
    data = json.loads(Path(path).read_text())
    items = [EvalItem(**item) for item in data["items"]]
    if categories:
        items = [i for i in items if i.category in set(categories)]
    return items[:limit] if limit else items


def load_gold_conflicts(path: Path) -> GoldConflicts:
    return GoldConflicts(**json.loads(Path(path).read_text()))
