"""Run baseline and improved configurations on the same labelled questions."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.evaluation.custom_metrics import (
    ItemRecord,
    compatible_false_positive_count,
    conflict_resolution_accuracy,
    contradiction_detection_scores,
    summarize,
)
from app.evaluation.datasets import EvalItem, load_eval_items, load_gold_conflicts
from app.evaluation.ragas_evaluation import run_ragas
from app.evaluation.reporting import write_report
from app.generation.prompts import format_evidence_item
from app.models.schemas import ChatRequest, GeneratedAnswer, VerificationStatus
from app.reliability.answer_verification import verify_answer
from app.services.chat_service import ChatService
from app.services.container import Container

logger = logging.getLogger(__name__)

CONFIGURATIONS = ("baseline", "improved")


def _ordered_docs(chunk_ids: list[str]) -> list[str]:
    docs: list[str] = []
    for chunk_id in chunk_ids:
        doc = chunk_id.split("::")[0]
        if doc not in docs:
            docs.append(doc)
    return docs


def evaluate_item(container: Container, chat: ChatService, item: EvalItem, configuration: str) -> ItemRecord:
    record = ItemRecord(
        item_id=item.id, category=item.category, configuration=configuration, question=item.question,
        answer="", abstained=False, should_abstain=item.should_abstain, relevant_doc_ids=item.relevant_doc_ids,
        retrieved_doc_ids=[], contexts=[], reference=item.reference_answer, expected_facts=item.expected_facts,
        stale_facts=item.stale_facts, expected_conflict=item.expected_conflict, expects_recovery=item.expects_recovery,
    )
    try:
        response, trace = chat.answer_with_trace(ChatRequest(question=item.question, mode=configuration))
    except Exception as exc:  # a failing item is recorded, never silently dropped
        logger.exception("Evaluation item %s/%s failed", configuration, item.id)
        record.error = str(exc)
        return record

    record.answer = response.answer
    record.abstained = response.abstained
    record.latency_ms = response.latency_ms
    # Retrieval is measured on the top-K evidence only (conflict counterparts added later are excluded).
    record.retrieved_doc_ids = _ordered_docs(response.retrieved_chunk_ids)
    # Judges see what the generator saw: every supplied passage (including conflict and graph-hop
    # passages added after retrieval) with its metadata header, plus document IDs and supersession
    # metadata used by the deterministic conflict notes. Abstentions made before generation fall
    # back to the retrieved passages.
    record.contexts = (
        [format_evidence_item(label, item, for_judge=True) for label, item in trace.evidence.items()]
        if trace.evidence else response.retrieved_contexts
    )
    record.recovery_triggered = response.recovery_performed
    record.retrieval_outcome = response.retrieval_outcome
    record.conflicts = [{"resolved": c.resolved, "winner": c.applicable_document} for c in response.conflicts]

    if not response.abstained and trace.final_claims:
        stored = {c.chunk_id for c in container.store.get_chunks([e.chunk_id for e in trace.evidence.values()])}
        for claim in trace.final_claims:
            for label in claim.citation_ids:
                record.citations_total += 1
                item_ = trace.evidence.get(label)
                record.citations_valid += bool(item_ is not None and item_.chunk_id in stored)
        # Post-hoc grounding check with the same verifier for both configurations.
        report = verify_answer(
            GeneratedAnswer(claims=trace.final_claims), trace.evidence, container.checker,
            container.settings.citation_repair_threshold,
        )
        record.claims_total = len(report.verifications)
        record.claims_supported = sum(1 for v in report.verifications if v.status == VerificationStatus.SUPPORTED)
    return record


def corpus_conflict_metrics(container: Container) -> dict[str, Any]:
    gold = load_gold_conflicts(container.settings.gold_conflicts_path)
    detected = [
        {
            "documents": [r["document_a"], r["document_b"]],
            "subject": r.get("subject"),
            "property": r.get("property"),
            "resolved": r.get("resolved"),
            "winner_document_id": r.get("winner_document_id"),
            "rule": r.get("rule"),
            "method": r.get("method"),
        }
        for r in container.store.claim_relations()
        if r.get("type") == "CONTRADICTS"
    ]
    gold_list = [g.model_dump() for g in gold.contradictions]
    return {
        "detection": contradiction_detection_scores(detected, gold_list),
        "resolution_accuracy": conflict_resolution_accuracy(detected, gold_list),
        "compatible_pairs_flagged_as_contradiction": compatible_false_positive_count(
            detected, [c.model_dump() for c in gold.compatible_pairs]
        ),
        "detected_conflicts": detected,
    }


def run_evaluation(
    container: Container,
    configurations: list[str] | None = None,
    include_ragas: bool = False,
    limit: int | None = None,
    categories: list[str] | None = None,
    write_files: bool = True,
) -> dict[str, Any]:
    configurations = [c for c in (configurations or list(CONFIGURATIONS)) if c in CONFIGURATIONS]
    items = load_eval_items(container.settings.eval_dataset_path, categories, limit)
    if not container.store.list_documents():
        raise RuntimeError("No documents are indexed. Seed the corpus before running evaluation.")
    chat = ChatService(container)
    records: list[ItemRecord] = []
    for configuration in configurations:
        for index, item in enumerate(items, start=1):
            logger.info("Evaluating %s %d/%d (%s)", configuration, index, len(items), item.id)
            records.append(evaluate_item(container, chat, item, configuration))

    s = container.settings
    s.eval_results_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    run_id = generated_at
    report: dict[str, Any] = {
        "generated_at": generated_at,
        "dataset": str(s.eval_dataset_path.name),
        "item_count": len(items),
        "configurations": configurations,
        "environment": {
            "llm_provider": container.llm.name,
            "mock_mode": container.llm.is_mock,
            "graph_backend": container.store.backend_name,
            "embedding_model": container.embedder.name,
            "reranker": container.reranker.name,
            "verifier": container.checker.name,
            "sufficiency_threshold": s.sufficiency_threshold,
            "min_chunk_relevance": s.min_chunk_relevance,
            "min_concept_coverage": s.min_concept_coverage,
            "recovery_max_attempts": s.recovery_max_attempts,
            "final_top_k": s.final_top_k,
        },
        "custom_metrics": {
            c: summarize([r for r in records if r.configuration == c]) for c in configurations
        },
        "custom_metrics_by_category": {
            c: {
                cat: summarize([r for r in records if r.configuration == c and r.category == cat])
                for cat in sorted({i.category for i in items})
            }
            for c in configurations
        },
        "corpus_conflict_metrics": corpus_conflict_metrics(container),
        "ragas": (
            run_ragas(records, s, checkpoint=s.eval_results_dir / "ragas_checkpoint.jsonl", run_id=run_id)
            if include_ragas else {"status": "not_requested", "reason": "include_ragas=false"}
        ),
        "notes": [
            "Claim support and unsupported-answer rates are judged by the configured verifier "
            f"('{container.checker.name}'), which the improved pipeline also uses internally; this favours "
            "the improved configuration. Ragas Faithfulness (when available) is an independent LLM judge.",
            "Corpus-level contradiction metrics describe the ingestion-time detector; the baseline has no "
            "conflict mechanism, so query-level conflict handling scores 0 for it by construction.",
        ],
    }
    if container.llm.is_mock:
        report["notes"].append(
            "MOCK LLM: answers are extractive (copied evidence sentences). Results characterise retrieval, "
            "recovery, conflict handling and abstention logic, not LLM generation quality."
        )
    if write_files:
        report["files"] = write_report(report, records, s.eval_results_dir)
    return report
