"""Claim-level contradiction detection.

Two claims are compared only if they come from different documents and refer to
the same subject entity. Then:

* **Structured** (same normalised property): opposing deontic polarity
  (permit/require vs prohibit) or different quantities in the same unit are a
  contradiction *if the applicability conditions are the same*. Different
  conditions make the rules compatible (``COMPATIBLE_CONDITIONAL``) — e.g. a
  general prohibition and a narrowly-conditioned exception.
* **Semantic** (property unknown): embedding similarity selects related claims
  and an NLI model classifies entailment / contradiction. Without an NLI model
  such pairs are left unclassified rather than guessed.

Different wording alone never produces a contradiction.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations

import numpy as np

from app.config import Settings
from app.models.schemas import Claim, ClaimComparison, ClaimRelationType, Polarity
from app.reliability.claim_extraction import normalize_condition
from app.reliability.nli import NLIModel
from app.retrieval.vector_search import Embedder

logger = logging.getLogger(__name__)

_OPPOSING = {
    frozenset({Polarity.PERMIT, Polarity.PROHIBIT}),
    frozenset({Polarity.REQUIRE, Polarity.PROHIBIT}),
}


def _format_quantity(claim: Claim) -> str:
    if claim.quantity is None:
        return "no value"
    if claim.unit == "minutes":
        for unit, size in (("years", 525600), ("months", 43200), ("days", 1440), ("hours", 60)):
            if claim.quantity >= size and claim.quantity % size == 0:
                return f"{claim.quantity / size:g} {unit}"
        return f"{claim.quantity:g} minutes"
    return f"{claim.quantity:g} {claim.unit}"


class ContradictionDetector:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, nli: NLIModel | None = None) -> None:
        self.settings = settings
        self.embedder = embedder
        self.nli = nli

    def compare(self, a: Claim, b: Claim) -> ClaimComparison:
        def result(relation: ClaimRelationType, method: str, explanation: str, confidence: float = 1.0) -> ClaimComparison:
            return ClaimComparison(claim_a=a, claim_b=b, relation=relation, method=method, confidence=confidence, explanation=explanation)

        if a.document_id == b.document_id:
            return result(ClaimRelationType.UNRELATED, "structured", "Claims come from the same document.")
        if not a.subject_id or a.subject_id != b.subject_id:
            return result(ClaimRelationType.UNRELATED, "structured", "Claims concern different subjects.")

        if a.property and b.property:
            if a.property != b.property:
                return result(ClaimRelationType.UNRELATED, "structured", "Claims concern different properties.")
            return self._structured(a, b, result)
        return self._semantic(a, b, result)

    def _structured(self, a: Claim, b: Claim, result) -> ClaimComparison:
        conditions_a = {normalize_condition(c) for c in a.conditions}
        conditions_b = {normalize_condition(c) for c in b.conditions}
        same_conditions = conditions_a == conditions_b
        opposing = frozenset({a.polarity, b.polarity}) in _OPPOSING
        numeric_conflict = (
            a.quantity is not None and b.quantity is not None and a.unit == b.unit and abs(a.quantity - b.quantity) > 1e-9
        )
        topic = f"{a.subject} / {a.property}"
        if opposing or numeric_conflict:
            what = (
                f"{a.polarity.value} vs {b.polarity.value}" if opposing
                else f"{_format_quantity(a)} vs {_format_quantity(b)}"
            )
            if same_conditions:
                return result(ClaimRelationType.CONTRADICTS, "structured", f"Conflicting rules for {topic}: {what}.")
            return result(
                ClaimRelationType.COMPATIBLE_CONDITIONAL,
                "structured",
                f"Rules for {topic} differ ({what}) but apply under different conditions: "
                f"{sorted(conditions_a) or ['(general rule)']} vs {sorted(conditions_b) or ['(general rule)']}.",
            )
        same_value = a.quantity == b.quantity and a.unit == b.unit
        if same_value and a.polarity == b.polarity and same_conditions and (a.polarity != Polarity.STATE or a.quantity is not None):
            return result(ClaimRelationType.SUPPORTS, "structured", f"Both documents state the same rule for {topic}.")
        return result(ClaimRelationType.UNRELATED, "structured", f"No conflict detected for {topic}.")

    def _semantic(self, a: Claim, b: Claim, result) -> ClaimComparison:
        if self.embedder is None or self.nli is None:
            return result(ClaimRelationType.UNRELATED, "none", "Unstructured claims not compared (no NLI model configured).", 0.0)
        vectors = self.embedder.embed([a.text, b.text])
        similarity = float(np.dot(vectors[0], vectors[1]) / ((np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])) or 1.0))
        if similarity < self.settings.contradiction_similarity_threshold:
            return result(ClaimRelationType.UNRELATED, "semantic", f"Low semantic similarity ({similarity:.2f}).", 1 - similarity)
        forward, backward = self.nli.predict([(a.text, b.text), (b.text, a.text)])
        contradiction = max(forward.contradiction, backward.contradiction)
        entailment = min(forward.entailment, backward.entailment)
        if contradiction >= self.settings.nli_contradiction_threshold:
            return result(ClaimRelationType.CONTRADICTS, "nli", f"NLI contradiction probability {contradiction:.2f}.", contradiction)
        if entailment >= self.settings.nli_entailment_threshold:
            return result(ClaimRelationType.SUPPORTS, "nli", f"NLI mutual entailment {entailment:.2f}.", entailment)
        return result(ClaimRelationType.UNRELATED, "nli", "NLI found no contradiction or entailment.", max(forward.neutral, backward.neutral))

    def detect(self, claims: list[Claim], against: list[Claim] | None = None) -> list[ClaimComparison]:
        """Compare candidate pairs. With ``against``, only pairs (new, existing) are checked."""
        groups: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims + (against or []):
            if claim.subject_id:
                groups[claim.subject_id].append(claim)
        new_ids = {c.claim_id for c in claims}
        seen: set[tuple[str, str]] = set()
        comparisons: list[ClaimComparison] = []
        for group in groups.values():
            for a, b in combinations(group, 2):
                if a.document_id == b.document_id:
                    continue
                if against is not None and not ((a.claim_id in new_ids) ^ (b.claim_id in new_ids)):
                    continue
                key = tuple(sorted((a.claim_id, b.claim_id)))
                if key in seen:
                    continue
                seen.add(key)
                if a.property and b.property and a.property != b.property:
                    continue
                comparison = self.compare(a, b)
                if comparison.relation != ClaimRelationType.UNRELATED:
                    comparisons.append(comparison)
        return comparisons
