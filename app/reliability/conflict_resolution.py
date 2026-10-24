"""Explicit, configurable precedence rules for contradictory claims.

Rules are evaluated in ``Settings.conflict_precedence`` order; the first rule
that can decide *with the available metadata* wins. A rule that lacks the
metadata it needs is skipped (and the reason recorded), never guessed. If no
rule decides, the conflict is reported as unresolved and both claims are shown.

Notably, "newest wins" is only applied when both documents have known, equal
authority and are both active — a newer document from an unknown or lower
authority does not silently override an established policy.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date

from app.config import Settings
from app.models.schemas import Claim, ConflictResolution, DocumentStatus

_INACTIVE = {DocumentStatus.SUPERSEDED, DocumentStatus.ARCHIVED}

RuleResult = tuple[Claim | None, str]  # (winner or None, explanation)


def _parse_version(version: str | None) -> tuple[int, ...] | None:
    if not version:
        return None
    numbers = re.findall(r"\d+", version)
    return tuple(int(n) for n in numbers) if numbers else None


class ConflictResolver:
    def __init__(self, settings: Settings, supersessions: list[tuple[str, str]], today: date | None = None) -> None:
        self.settings = settings
        self.today = today or date.today()
        self._supersedes = self._closure(supersessions)
        self._rules: dict[str, Callable[[Claim, Claim], RuleResult]] = {
            "supersession": self._rule_supersession,
            "status": self._rule_status,
            "authority": self._rule_authority,
            "effective_date": self._rule_effective_date,
            "version": self._rule_version,
        }

    @staticmethod
    def _closure(edges: list[tuple[str, str]]) -> set[tuple[str, str]]:
        closure = set(edges)
        changed = True
        while changed:
            changed = False
            for a, b in list(closure):
                for c, d in list(closure):
                    if b == c and (a, d) not in closure:
                        closure.add((a, d))
                        changed = True
        return closure

    # -- rules ---------------------------------------------------------------------
    def _rule_supersession(self, a: Claim, b: Claim) -> RuleResult:
        a_over_b = (a.document_id, b.document_id) in self._supersedes
        b_over_a = (b.document_id, a.document_id) in self._supersedes
        if a_over_b and b_over_a:
            return None, "Supersession metadata is circular; ignored."
        if a_over_b:
            return a, f"{a.document_id} explicitly supersedes {b.document_id}."
        if b_over_a:
            return b, f"{b.document_id} explicitly supersedes {a.document_id}."
        return None, "No supersession relationship between the documents."

    def _rule_status(self, a: Claim, b: Claim) -> RuleResult:
        if DocumentStatus.UNKNOWN in (a.status, b.status):
            return None, "Document status unknown for at least one document."
        ranks = {DocumentStatus.ACTIVE: 2, DocumentStatus.DRAFT: 1, DocumentStatus.SUPERSEDED: 0, DocumentStatus.ARCHIVED: 0}
        if ranks[a.status] == ranks[b.status]:
            return None, f"Both documents have status '{a.status.value}'/'{b.status.value}'."
        winner, loser = (a, b) if ranks[a.status] > ranks[b.status] else (b, a)
        return winner, f"{winner.document_id} is {winner.status.value}; {loser.document_id} is {loser.status.value}."

    def _rule_authority(self, a: Claim, b: Claim) -> RuleResult:
        levels = self.settings.authority_levels
        rank_a, rank_b = levels.get(a.authority or ""), levels.get(b.authority or "")
        if rank_a is None or rank_b is None:
            return None, "Source authority unknown for at least one document."
        if rank_a == rank_b:
            return None, f"Both documents have the same authority level ('{a.authority}')."
        winner, loser = (a, b) if rank_a > rank_b else (b, a)
        return winner, f"{winner.document_id} has higher authority ('{winner.authority}') than {loser.document_id} ('{loser.authority}')."

    def _rule_effective_date(self, a: Claim, b: Claim) -> RuleResult:
        if a.effective_date is None or b.effective_date is None:
            return None, "Effective date missing for at least one document."
        a_future, b_future = a.effective_date > self.today, b.effective_date > self.today
        if a_future != b_future:
            winner, loser = (b, a) if a_future else (a, b)
            return winner, f"{loser.document_id} is not yet effective ({loser.effective_date}); {winner.document_id} currently applies."
        levels = self.settings.authority_levels
        rank_a, rank_b = levels.get(a.authority or ""), levels.get(b.authority or "")
        if rank_a is None or rank_b is None or rank_a != rank_b:
            return None, "Newer-date precedence requires both documents to have known, equal authority."
        if a.status != DocumentStatus.ACTIVE or b.status != DocumentStatus.ACTIVE:
            return None, "Newer-date precedence requires both documents to be active."
        if a.effective_date == b.effective_date:
            return None, "Both documents have the same effective date."
        winner, loser = (a, b) if a.effective_date > b.effective_date else (b, a)
        return winner, f"Same authority; {winner.document_id} has the later effective date ({winner.effective_date} vs {loser.effective_date})."

    def _rule_version(self, a: Claim, b: Claim) -> RuleResult:
        if not a.document_title or a.document_title != b.document_title:
            return None, "Version comparison only applies to versions of the same document."
        va, vb = _parse_version(a.version), _parse_version(b.version)
        if va is None or vb is None or va == vb:
            return None, "Versions missing or equal."
        winner = a if va > vb else b
        return winner, f"{winner.document_id} is a later version ({winner.version}) of '{a.document_title}'."

    # -- public --------------------------------------------------------------------
    def resolve(self, a: Claim, b: Claim) -> ConflictResolution:
        considered: list[str] = []
        for rule_name in self.settings.conflict_precedence:
            rule = self._rules.get(rule_name)
            if rule is None:
                considered.append(f"{rule_name}: unknown rule (ignored)")
                continue
            winner, explanation = rule(a, b)
            considered.append(f"{rule_name}: {explanation}")
            if winner is not None:
                return ConflictResolution(
                    claim_a_id=a.claim_id,
                    claim_b_id=b.claim_id,
                    resolved=True,
                    winner_claim_id=winner.claim_id,
                    winner_document_id=winner.document_id,
                    rule=rule_name,
                    explanation=explanation,
                    rules_considered=considered,
                )
        return ConflictResolution(
            claim_a_id=a.claim_id,
            claim_b_id=b.claim_id,
            resolved=False,
            explanation=(
                f"Conflict between {a.document_id} and {b.document_id} could not be resolved from "
                "available metadata (no supersession, same status and authority, and no usable effective-date or version precedence)."
            ),
            rules_considered=considered,
        )
