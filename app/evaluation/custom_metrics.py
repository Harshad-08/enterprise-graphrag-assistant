"""Custom reliability metrics. Each function documents its exact definition and denominator.

All metrics return ``None`` when their denominator is zero, so an empty category
is reported as "n/a" rather than as a misleading 0 or 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ItemRecord:
    """Everything measured for one (question, configuration) run."""

    item_id: str
    category: str
    configuration: str
    question: str
    answer: str
    abstained: bool
    should_abstain: bool
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]  # ordered, deduplicated, final evidence documents (top-K)
    contexts: list[str]
    reference: str | None
    expected_facts: list[str]
    stale_facts: list[str]
    expected_conflict: str | None
    expects_recovery: bool
    recovery_triggered: bool = False
    retrieval_outcome: str | None = None
    conflicts: list[dict[str, Any]] = field(default_factory=list)  # [{"resolved": bool, "winner": str|None}]
    citations_total: int = 0
    citations_valid: int = 0
    claims_total: int = 0
    claims_supported: int = 0
    latency_ms: float | None = None
    error: str | None = None


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def retrieval_recall_at_k(records: list[ItemRecord]) -> float | None:
    """Mean over items with ≥1 relevant document of |relevant ∩ retrieved top-K docs| / |relevant|."""
    scored = [r for r in records if r.relevant_doc_ids]
    if not scored:
        return None
    values = [len(set(r.relevant_doc_ids) & set(r.retrieved_doc_ids)) / len(r.relevant_doc_ids) for r in scored]
    return round(sum(values) / len(values), 4)


def recovery_success_rate(records: list[ItemRecord]) -> float | None:
    """Among answerable items (should_abstain = False) where recovery was triggered:
    fraction whose final evidence was judged sufficient (outcome 'recovered') AND contains
    ≥1 relevant document. Denominator: answerable items with recovery triggered."""
    triggered = [r for r in records if r.recovery_triggered and not r.should_abstain]
    hits = [r for r in triggered if r.retrieval_outcome == "recovered" and set(r.relevant_doc_ids) & set(r.retrieved_doc_ids)]
    return _ratio(len(hits), len(triggered))


def expected_recovery_hit_rate(records: list[ItemRecord]) -> float | None:
    """Among items labelled expects_recovery: fraction whose final evidence contains a relevant
    document (regardless of whether recovery or initial retrieval found it)."""
    labelled = [r for r in records if r.expects_recovery]
    hits = [r for r in labelled if set(r.relevant_doc_ids) & set(r.retrieved_doc_ids)]
    return _ratio(len(hits), len(labelled))


def citation_validity_rate(records: list[ItemRecord]) -> float | None:
    """Valid citations / all citations emitted in answers. A citation is valid when it refers to a
    passage that was actually supplied to the generator and exists in the store."""
    return _ratio(sum(r.citations_valid for r in records), sum(r.citations_total for r in records))


def claim_support_rate(records: list[ItemRecord]) -> float | None:
    """Supported claims / all claims in non-abstained answers, judged by the configured verifier."""
    answered = [r for r in records if not r.abstained]
    return _ratio(sum(r.claims_supported for r in answered), sum(r.claims_total for r in answered))


def unsupported_answer_rate(records: list[ItemRecord]) -> float | None:
    """Non-abstained answers containing ≥1 unsupported claim / non-abstained answers."""
    answered = [r for r in records if not r.abstained]
    return _ratio(sum(1 for r in answered if r.claims_supported < r.claims_total), len(answered))


def abstention_precision(records: list[ItemRecord]) -> float | None:
    """Correct abstentions (should_abstain and abstained) / all abstentions."""
    abstained = [r for r in records if r.abstained]
    return _ratio(sum(1 for r in abstained if r.should_abstain), len(abstained))


def abstention_recall(records: list[ItemRecord]) -> float | None:
    """Correct abstentions / items labelled should_abstain."""
    labelled = [r for r in records if r.should_abstain]
    return _ratio(sum(1 for r in labelled if r.abstained), len(labelled))


def false_abstention_rate(records: list[ItemRecord]) -> float | None:
    """Abstentions on answerable items / answerable items."""
    answerable = [r for r in records if not r.should_abstain]
    return _ratio(sum(1 for r in answerable if r.abstained), len(answerable))


def expected_fact_recall(records: list[ItemRecord]) -> float | None:
    """Over answerable items with expected_facts: fraction of expected fact strings that appear
    (case-insensitive substring) in the answer. A deterministic, LLM-free correctness proxy."""
    total = found = 0
    for r in records:
        if r.should_abstain or not r.expected_facts:
            continue
        for fact in r.expected_facts:
            total += 1
            found += fact.lower() in r.answer.lower()
    return _ratio(found, total)


def stale_answer_rate(records: list[ItemRecord]) -> float | None:
    """Over items with stale_facts (rules overridden by a resolved conflict): fraction of answers
    that state a stale fact *without* any conflict disclosure."""
    labelled = [r for r in records if r.stale_facts]
    stale = [
        r for r in labelled
        if any(f.lower() in r.answer.lower() for f in r.stale_facts) and not r.conflicts
    ]
    return _ratio(len(stale), len(labelled))


def query_conflict_handling_accuracy(records: list[ItemRecord]) -> float | None:
    """Over items with expected_conflict: fraction where the response reports the expected outcome —
    an unresolved conflict for 'unresolved', or a resolved conflict won by the expected document for
    'resolved:<doc>'. Configurations without conflict handling score 0 on these items."""
    labelled = [r for r in records if r.expected_conflict]
    correct = 0
    for r in labelled:
        if r.expected_conflict == "unresolved":
            correct += any(not c["resolved"] for c in r.conflicts)
        else:
            winner = r.expected_conflict.split(":", 1)[1]
            correct += any(c["resolved"] and c["winner"] == winner for c in r.conflicts)
    return _ratio(correct, len(labelled))


def contradiction_detection_scores(
    detected: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> dict[str, float | int | None]:
    """Corpus-level precision/recall of contradiction detection.

    detected / gold entries: {"documents": [a, b], "subject": str, ...}. A detected pair is a true
    positive when the unordered document pair and the subject match a gold contradiction.
    precision = TP / detected pairs; recall = TP / gold pairs. Duplicate detections of the same
    (document pair, subject) count once.
    """
    def key(entry: dict[str, Any]) -> tuple[frozenset, str]:
        return frozenset(entry["documents"]), (entry.get("subject") or "").lower()

    detected_keys = {key(d) for d in detected}
    gold_keys = {key(g) for g in gold}
    tp = len(detected_keys & gold_keys)
    return {
        "true_positives": tp,
        "detected_pairs": len(detected_keys),
        "gold_pairs": len(gold_keys),
        "precision": _ratio(tp, len(detected_keys)),
        "recall": _ratio(tp, len(gold_keys)),
    }


def conflict_resolution_accuracy(detected: list[dict[str, Any]], gold: list[dict[str, Any]]) -> float | None:
    """Over all gold contradictions: fraction where a matching detected conflict has the expected
    outcome (winning document, or unresolved). Undetected gold conflicts count as incorrect."""
    by_key = {}
    for d in detected:
        by_key[(frozenset(d["documents"]), (d.get("subject") or "").lower())] = d
    correct = 0
    for g in gold:
        d = by_key.get((frozenset(g["documents"]), g["subject"].lower()))
        if d is None:
            continue
        outcome = d.get("winner_document_id") if d.get("resolved") else "unresolved"
        correct += outcome == g["expected_resolution"]
    return _ratio(correct, len(gold))


def compatible_false_positive_count(detected: list[dict[str, Any]], compatible: list[dict[str, Any]]) -> int:
    """Number of known-compatible pairs wrongly reported as contradictions."""
    detected_keys = {(frozenset(d["documents"]), (d.get("subject") or "").lower()) for d in detected}
    return sum(1 for c in compatible if (frozenset(c["documents"]), c["subject"].lower()) in detected_keys)


def summarize(records: list[ItemRecord]) -> dict[str, Any]:
    ok = [r for r in records if r.error is None]
    latencies = [r.latency_ms for r in ok if r.latency_ms is not None]
    return {
        "items": len(records),
        "errors": len(records) - len(ok),
        "retrieval_recall_at_k": retrieval_recall_at_k(ok),
        "recovery_trigger_count": sum(1 for r in ok if r.recovery_triggered),
        "recovery_success_rate": recovery_success_rate(ok),
        "expected_recovery_hit_rate": expected_recovery_hit_rate(ok),
        "citation_validity_rate": citation_validity_rate(ok),
        "claim_support_rate": claim_support_rate(ok),
        "unsupported_answer_rate": unsupported_answer_rate(ok),
        "abstention_precision": abstention_precision(ok),
        "abstention_recall": abstention_recall(ok),
        "false_abstention_rate": false_abstention_rate(ok),
        "expected_fact_recall": expected_fact_recall(ok),
        "stale_answer_rate": stale_answer_rate(ok),
        "query_conflict_handling_accuracy": query_conflict_handling_accuracy(ok),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
    }
