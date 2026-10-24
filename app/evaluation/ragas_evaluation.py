"""Ragas evaluation (ragas 0.4.x ``ragas.metrics.collections`` API).

Metrics (LLM-judged):

* ContextRecall                 – is the reference answer's information present in the retrieved contexts?
* ContextPrecisionWithReference – are relevant contexts ranked above irrelevant ones?
* Faithfulness                  – are the response's claims supported by the retrieved contexts?
* AnswerCorrectness             – agreement with the reference answer (factual overlap + embedding similarity)

Judge provider follows the configured LLM:

* Groq   – an ``openai.AsyncOpenAI`` client pointed at Groq's OpenAI-compatible endpoint with
           ``llm_factory(provider="openai")``. (ragas 0.4.3's native "groq" path calls an
           Anthropic-style ``client.messages.create`` that the Groq SDK does not have.)
           Groq rate limits are per model, so metrics can be spread over several judge models
           (``RAGAS_JUDGE_MODELS``) and scored concurrently. Each metric always uses the same judge
           for every configuration, so baseline and improved stay comparable.
* Gemini – ``llm_factory(provider="google")`` with a google-genai client.

Reasoning models (gpt-oss) need ``max_tokens`` well above ragas' default of 1024, otherwise
structured judgments are truncated; ``RAGAS_MAX_TOKENS`` defaults to 4096.

Ragas *measures* the pipeline; it does not implement recovery, conflict handling or
verification. Without GROQ_API_KEY or GEMINI_API_KEY the run is reported as skipped — no
scores are invented. Scoring is checkpointed per item so an interrupted run can resume.

Scoping: context metrics and AnswerCorrectness need a reference answer, so they run only on
answerable items. Faithfulness is skipped for abstentions (no claims to ground).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any

from app.config import Settings
from app.evaluation.custom_metrics import ItemRecord

logger = logging.getLogger(__name__)

METRIC_NAMES = ("context_recall", "context_precision", "faithfulness", "answer_correctness")

# Context precision makes one judge call per retrieved passage, so it gets its own model bucket.
DEFAULT_GROQ_JUDGES = {
    "context_precision": "openai/gpt-oss-20b",
    "context_recall": "openai/gpt-oss-120b",
    "faithfulness": "openai/gpt-oss-120b",
    "answer_correctness": "openai/gpt-oss-120b",
}


def judge_provider(settings: Settings) -> str | None:
    provider = settings.resolved_llm_provider()
    if provider == "gemini" and settings.gemini_api_key:
        return "gemini"
    if provider == "groq" and settings.groq_api_key:
        return "groq"
    # A mock-mode run can still be judged if a key is configured.
    if settings.groq_api_key:
        return "groq"
    if settings.gemini_api_key:
        return "gemini"
    return None


def judge_models(settings: Settings) -> dict[str, str]:
    provider = judge_provider(settings)
    if provider is None:
        return {}
    if provider == "groq":
        models = dict(DEFAULT_GROQ_JUDGES)
    else:
        models = {name: settings.gemini_model for name in METRIC_NAMES}
    if settings.ragas_judge_model:  # one model for every metric
        models = {name: settings.ragas_judge_model for name in METRIC_NAMES}
    models.update({k: v for k, v in settings.ragas_judge_models.items() if k in METRIC_NAMES})
    return models


def ragas_availability(settings: Settings) -> tuple[bool, str]:
    if not settings.ragas_enabled:
        return False, "RAGAS_ENABLED=false"
    if judge_provider(settings) is None:
        return False, "Neither GROQ_API_KEY nor GEMINI_API_KEY is set; Ragas metrics require an LLM judge."
    try:
        import ragas  # noqa: F401
        from ragas.metrics.collections import Faithfulness  # noqa: F401
    except Exception as exc:  # import errors from incompatible dependency versions
        return False, f"Ragas could not be imported: {exc}"
    return True, "available"


def _build_metrics(settings: Settings) -> dict[str, Any]:
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerCorrectness,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    models = judge_models(settings)
    llms: dict[str, Any] = {}

    def llm_for(model: str):
        if model in llms:
            return llms[model]
        if judge_provider(settings) == "groq":
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url,
                                 max_retries=settings.llm_max_retries, timeout=settings.llm_timeout_seconds * 3)
            if model.startswith("openai/gpt-oss"):
                extra = {"reasoning_effort": settings.groq_reasoning_effort}
            elif model.startswith("qwen/"):
                extra = {"reasoning_effort": "none"}  # judging needs no thinking tokens; saves quota
            else:
                extra = {}
            llms[model] = llm_factory(model, provider="openai", client=client, max_tokens=settings.ragas_max_tokens, **extra)
        else:
            from google import genai

            client = genai.Client(api_key=settings.gemini_api_key)
            llms[model] = llm_factory(model, provider="google", client=client, max_tokens=settings.ragas_max_tokens)
        return llms[model]

    embeddings = HuggingFaceEmbeddings(model=settings.embedding_model)
    return {
        "context_recall": ContextRecall(llm=llm_for(models["context_recall"])),
        "context_precision": ContextPrecisionWithReference(llm=llm_for(models["context_precision"])),
        "faithfulness": Faithfulness(llm=llm_for(models["faithfulness"])),
        "answer_correctness": AnswerCorrectness(llm=llm_for(models["answer_correctness"]), embeddings=embeddings),
    }


async def _score_record(metrics: dict[str, Any], record: ItemRecord) -> tuple[dict[str, float | None], dict[str, str]]:
    contexts = record.contexts or [""]
    calls: dict[str, Any] = {}
    if record.reference:
        calls["context_recall"] = metrics["context_recall"].ascore(
            user_input=record.question, retrieved_contexts=contexts, reference=record.reference
        )
        calls["context_precision"] = metrics["context_precision"].ascore(
            user_input=record.question, reference=record.reference, retrieved_contexts=contexts
        )
        calls["answer_correctness"] = metrics["answer_correctness"].ascore(
            user_input=record.question, response=record.answer, reference=record.reference
        )
    if not record.abstained:
        calls["faithfulness"] = metrics["faithfulness"].ascore(
            user_input=record.question, response=record.answer, retrieved_contexts=contexts
        )
    scores: dict[str, float | None] = {name: None for name in METRIC_NAMES}
    errors: dict[str, str] = {}
    # Metrics use different judge models (separate rate-limit buckets), so run them concurrently.
    results = await asyncio.gather(*calls.values(), return_exceptions=True)
    for name, result in zip(calls, results):
        if isinstance(result, Exception):  # judge failures are recorded, not hidden
            errors[name] = str(result)[:300]
            logger.warning("Ragas %s failed for %s/%s: %s", name, record.configuration, record.item_id, result)
            continue
        value = float(getattr(result, "value", result))
        scores[name] = None if math.isnan(value) else value
    return scores, errors


def _load_checkpoint(path: Path | None, run_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    done: dict[tuple[str, str], dict[str, Any]] = {}
    if path is None or not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("run_id") == run_id and not row.get("errors"):
            done[(row["configuration"], row["item_id"])] = row
    return done


def run_ragas(
    records: list[ItemRecord], settings: Settings, checkpoint: Path | None = None, run_id: str = "adhoc",
    score_missing: bool = True,
) -> dict[str, Any]:
    """Score records (resuming from ``checkpoint``). With ``score_missing=False`` only already
    checkpointed scores are aggregated and no API calls are made."""
    available, reason = ragas_availability(settings)
    if not available:
        return {"status": "skipped", "reason": reason, "per_item": {}, "aggregate": {}}
    metrics = _build_metrics(settings) if score_missing else {}
    # The judge is part of the checkpoint key: scores from different judges are never mixed.
    run_id = f"{run_id}|{json.dumps(judge_models(settings), sort_keys=True)}"
    done = _load_checkpoint(checkpoint, run_id)
    # Interleave configurations per question so that, if a quota runs out mid-run, every
    # configuration has been scored on the same questions (see "matched" aggregates).
    order = {cfg: i for i, cfg in enumerate(dict.fromkeys(r.configuration for r in records))}
    eligible = sorted((r for r in records if r.error is None), key=lambda r: (r.item_id, order[r.configuration]))

    async def run_all() -> list[dict[str, Any]]:
        rows = []
        for index, record in enumerate(eligible, start=1):  # sequential items: stays within rate limits
            key = (record.configuration, record.item_id)
            if key in done:
                rows.append(done[key])
                continue
            if not score_missing:
                continue
            logger.info("Ragas %d/%d: %s/%s", index, len(eligible), *key)
            scores, errors = await _score_record(metrics, record)
            row = {"run_id": run_id, "configuration": key[0], "item_id": key[1], "scores": scores, "errors": errors}
            if checkpoint is not None:
                with checkpoint.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
            rows.append(row)
        return rows

    rows = asyncio.run(run_all())
    per_item: dict[str, dict[str, dict[str, float | None]]] = {}
    errors: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        per_item.setdefault(row["configuration"], {})[row["item_id"]] = row["scores"]
        if row.get("errors"):
            errors.setdefault(row["configuration"], {})[row["item_id"]] = row["errors"]
    aggregate: dict[str, dict[str, Any]] = {}
    for configuration, items in per_item.items():
        aggregate[configuration] = {}
        for name in METRIC_NAMES:
            values = [s[name] for s in items.values() if s[name] is not None]
            failed = sum(1 for e in errors.get(configuration, {}).values() if name in e)
            aggregate[configuration][name] = {
                "mean": round(sum(values) / len(values), 4) if values else None,
                "scored_items": len(values),
                "failed_items": failed,
            }
    # Matched comparison: per metric, only questions scored successfully in every configuration.
    configs = list(per_item)
    matched: dict[str, dict[str, Any]] = {}
    for name in METRIC_NAMES:
        common = set.intersection(*[
            {item for item, s in per_item.get(c, {}).items() if s[name] is not None} for c in configs
        ]) if configs else set()
        matched[name] = {
            "items": len(common),
            **{c: (round(sum(per_item[c][i][name] for i in common) / len(common), 4) if common else None) for c in configs},
        }
    total_failed = sum(len(e) for cfg in errors.values() for e in cfg.values())
    unscored = len(eligible) - len(rows)
    incomplete = total_failed or unscored
    return {
        "status": "partial" if incomplete else "completed",
        "reason": (
            f"{unscored} record(s) not yet scored and {total_failed} metric evaluation(s) failed; compare "
            "configurations with the 'matched' aggregates and re-run scripts/run_ragas.py to resume."
        ) if incomplete else None,
        "records_total": len(eligible), "records_scored": len(rows),
        "judge_provider": judge_provider(settings), "judge_models": judge_models(settings),
        "per_item": per_item, "errors": errors, "aggregate": aggregate, "matched": matched,
    }
