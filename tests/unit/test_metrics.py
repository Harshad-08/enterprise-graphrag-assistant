"""Custom reliability metrics have exact, documented definitions."""

from __future__ import annotations

import pytest

from app.evaluation.custom_metrics import (
    ItemRecord,
    abstention_precision,
    abstention_recall,
    citation_validity_rate,
    claim_support_rate,
    compatible_false_positive_count,
    conflict_resolution_accuracy,
    contradiction_detection_scores,
    expected_fact_recall,
    query_conflict_handling_accuracy,
    recovery_success_rate,
    retrieval_recall_at_k,
    stale_answer_rate,
    summarize,
    unsupported_answer_rate,
)
from app.evaluation.ragas_evaluation import ragas_availability
from tests.conftest import make_settings


def rec(**kw) -> ItemRecord:
    base = dict(item_id="Q", category="c", configuration="improved", question="q", answer="", abstained=False,
                should_abstain=False, relevant_doc_ids=[], retrieved_doc_ids=[], contexts=[], reference=None,
                expected_facts=[], stale_facts=[], expected_conflict=None, expects_recovery=False)
    base.update(kw)
    return ItemRecord(**base)


def test_recall_at_k():
    records = [rec(relevant_doc_ids=["A", "B"], retrieved_doc_ids=["A", "X"]), rec(relevant_doc_ids=["C"], retrieved_doc_ids=["C"]),
               rec(relevant_doc_ids=[], retrieved_doc_ids=["Z"])]
    assert retrieval_recall_at_k(records) == pytest.approx((0.5 + 1.0) / 2)


def test_recovery_success_rate_denominator_is_answerable_triggered_items():
    records = [
        rec(recovery_triggered=True, retrieval_outcome="recovered", relevant_doc_ids=["A"], retrieved_doc_ids=["A"]),
        rec(recovery_triggered=True, retrieval_outcome="documents_without_answer", relevant_doc_ids=["A"], retrieved_doc_ids=["B"]),
        rec(recovery_triggered=True, retrieval_outcome="not_in_corpus", should_abstain=True),  # excluded
        rec(recovery_triggered=False),  # excluded
    ]
    assert recovery_success_rate(records) == 0.5
    assert recovery_success_rate([rec()]) is None


def test_citation_and_claim_rates():
    records = [rec(citations_total=4, citations_valid=3, claims_total=3, claims_supported=3),
               rec(citations_total=1, citations_valid=1, claims_total=2, claims_supported=1),
               rec(abstained=True, claims_total=5, claims_supported=0)]
    assert citation_validity_rate(records) == 0.8
    assert claim_support_rate(records) == 0.8  # abstained answers excluded
    assert unsupported_answer_rate(records) == 0.5


def test_abstention_precision_recall():
    records = [rec(abstained=True, should_abstain=True), rec(abstained=True, should_abstain=False),
               rec(abstained=False, should_abstain=True), rec(abstained=False, should_abstain=False)]
    assert abstention_precision(records) == 0.5
    assert abstention_recall(records) == 0.5


def test_fact_recall_stale_and_conflict_handling():
    records = [
        rec(answer="Up to 3 days per week.", expected_facts=["3 days per week"], stale_facts=["2 days per week"],
            expected_conflict="resolved:HR", conflicts=[{"resolved": True, "winner": "HR"}]),
        rec(answer="Up to 2 days per week.", expected_facts=["3 days per week"], stale_facts=["2 days per week"],
            expected_conflict="resolved:HR", conflicts=[]),
        rec(answer="60 or 15", expected_conflict="unresolved", conflicts=[{"resolved": False, "winner": None}]),
    ]
    assert expected_fact_recall(records) == 0.5
    assert stale_answer_rate(records) == 0.5
    assert query_conflict_handling_accuracy(records) == pytest.approx(2 / 3, abs=1e-4)


def test_corpus_contradiction_metrics():
    gold = [{"documents": ["A", "B"], "subject": "VPN", "expected_resolution": "unresolved"},
            {"documents": ["C", "D"], "subject": "Logs", "expected_resolution": "C"}]
    detected = [{"documents": ["B", "A"], "subject": "vpn", "resolved": False},
                {"documents": ["C", "D"], "subject": "Logs", "resolved": True, "winner_document_id": "D"},
                {"documents": ["E", "F"], "subject": "Other", "resolved": False}]
    scores = contradiction_detection_scores(detected, gold)
    assert (scores["true_positives"], scores["precision"], scores["recall"]) == (2, pytest.approx(0.6667, abs=1e-4), 1.0)
    assert conflict_resolution_accuracy(detected, gold) == 0.5  # C/D resolved to the wrong winner
    assert compatible_false_positive_count(detected, [{"documents": ["E", "F"], "subject": "other"}]) == 1


def test_summary_excludes_errored_items_and_handles_empty_denominators():
    summary = summarize([rec(error="boom"), rec(abstained=True, should_abstain=True)])
    assert summary["errors"] == 1 and summary["abstention_precision"] == 1.0
    assert summary["retrieval_recall_at_k"] is None


def test_ragas_reports_unavailability_instead_of_scores():
    available, reason = ragas_availability(make_settings(ragas_enabled=True, gemini_api_key=None))
    assert not available and "GROQ_API_KEY" in reason and "GEMINI_API_KEY" in reason


def test_ragas_judge_models_per_metric():
    from app.evaluation.ragas_evaluation import judge_models

    groq = make_settings(llm_provider="groq", groq_api_key="k")
    models = judge_models(groq)
    assert models["context_precision"] == "openai/gpt-oss-20b" and models["faithfulness"] == "openai/gpt-oss-120b"
    single = make_settings(llm_provider="groq", groq_api_key="k", ragas_judge_model="m1")
    assert set(judge_models(single).values()) == {"m1"}
    override = make_settings(llm_provider="groq", groq_api_key="k", ragas_judge_models={"faithfulness": "m2"})
    assert judge_models(override)["faithfulness"] == "m2"
    assert judge_models(make_settings()) == {}


def test_ragas_checkpoint_aggregation_uses_matched_items(tmp_path):
    import json

    from app.evaluation.ragas_evaluation import judge_models, run_ragas

    settings = make_settings(llm_provider="groq", groq_api_key="k", ragas_judge_model="judge-x", ragas_enabled=True)
    run_id = "R1|" + json.dumps(judge_models(settings), sort_keys=True)
    checkpoint = tmp_path / "ckpt.jsonl"
    scores = lambda f: {"context_recall": None, "context_precision": None, "faithfulness": f, "answer_correctness": None}  # noqa: E731
    rows = [
        {"run_id": run_id, "configuration": "baseline", "item_id": "Q1", "scores": scores(0.5), "errors": {}},
        {"run_id": run_id, "configuration": "improved", "item_id": "Q1", "scores": scores(1.0), "errors": {}},
        {"run_id": run_id, "configuration": "baseline", "item_id": "Q2", "scores": scores(0.0), "errors": {}},  # unmatched
        {"run_id": "R1|other-judge", "configuration": "improved", "item_id": "Q2", "scores": scores(1.0), "errors": {}},
    ]
    checkpoint.write_text("\n".join(json.dumps(r) for r in rows))
    records = [rec(item_id=i, configuration=c) for i in ("Q1", "Q2") for c in ("baseline", "improved")]
    result = run_ragas(records, settings, checkpoint=checkpoint, run_id="R1", score_missing=False)
    assert result["status"] == "partial" and result["records_scored"] == 3
    assert result["matched"]["faithfulness"] == {"items": 1, "baseline": 0.5, "improved": 1.0}
    assert result["aggregate"]["baseline"]["faithfulness"]["mean"] == 0.25  # unmatched means differ


def test_evaluation_contexts_include_metadata_headers(tmp_path):
    """Ragas judges must see the metadata (e.g. supersession) that conflict notes rely on."""
    from app.config import PROJECT_ROOT
    from app.evaluation.datasets import EvalItem
    from app.evaluation.runner import evaluate_item
    from app.services.chat_service import ChatService
    from app.services.document_service import DocumentService
    from tests.conftest import make_container

    c = make_container(tmp_path)
    DocumentService(c).seed_directory(PROJECT_ROOT / "data" / "synthetic_documents")
    item = EvalItem(id="Q02", category="conflict_resolved", question="How many days per week can employees work from home?")
    record = evaluate_item(c, ChatService(c), item, "improved")
    assert any("supersedes HR-POL-007-V1" in ctx for ctx in record.contexts)
    assert all(ctx.startswith("[E") for ctx in record.contexts)
