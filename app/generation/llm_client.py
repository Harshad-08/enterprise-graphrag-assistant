"""LLM clients: Google Gemini (google-genai SDK), Groq (official groq SDK) and a deterministic mock.

The mock exists so tests and offline demos run without an API key. It performs
*extractive* answering (it copies evidence sentences that overlap with the
question) — it does no semantic reasoning, and every response produced through
it is labelled as mock output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from app.config import Settings
from app.utils.text import content_terms, split_sentences

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_NORMATIVE_RE = re.compile(r"\b(must|may|required|requires|prohibited|permitted|retained|expire|within)\b", re.I)


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str
    is_mock: bool

    def generate_structured(self, prompt: str, schema: type[T], *, context: dict[str, Any] | None = None) -> T: ...


class GeminiClient:
    is_mock = False

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, timeout_seconds: int = 60) -> None:
        from google import genai

        if not api_key:
            raise LLMError("GEMINI_API_KEY is required for the Gemini provider.")
        self.name = f"gemini:{model}"
        self.model = model
        self.temperature = temperature
        self._client = genai.Client(
            api_key=api_key, http_options=genai.types.HttpOptions(timeout=timeout_seconds * 1000)
        )

    def generate_structured(self, prompt: str, schema: type[T], *, context: dict[str, Any] | None = None) -> T:
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        except Exception as exc:  # network, quota, invalid model name ...
            raise LLMError(f"Gemini request failed: {exc}") from exc
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        try:
            return schema.model_validate_json(response.text or "{}")
        except Exception as exc:
            raise LLMError(f"Gemini returned invalid structured output: {exc}") from exc


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema -> the strict subset accepted by OpenAI-style structured outputs:
    $refs inlined, every property required, no additional properties, no defaults/titles."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def convert(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return convert(defs[node["$ref"].split("/")[-1]])
            node = {k: convert(v) for k, v in node.items() if k not in ("default", "title")}
            if node.get("type") == "object":
                node["required"] = list(node.get("properties", {}))
                node["additionalProperties"] = False
            return node
        if isinstance(node, list):
            return [convert(item) for item in node]
        return node

    return convert(schema)


class GroqClient:
    """Groq chat completions with strict JSON-schema structured output.

    Falls back to JSON mode + Pydantic validation if a model rejects strict schemas, and
    re-asks once on invalid output. Rate limits (HTTP 429) are retried by the SDK with
    backoff that honours Retry-After. Reasoning text of reasoning models is never used
    or exposed; only the final JSON content is parsed.
    """

    is_mock = False

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, timeout_seconds: int = 60,
                 max_retries: int = 8, reasoning_effort: str | None = "low") -> None:
        from groq import Groq

        if not api_key:
            raise LLMError("GROQ_API_KEY is required for the Groq provider.")
        self.name = f"groq:{model}"
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort if model.startswith("openai/gpt-oss") else None
        self._client = Groq(api_key=api_key, timeout=timeout_seconds, max_retries=max_retries)
        self._strict_supported = True

    def _complete(self, prompt: str, response_format: dict[str, Any]) -> str:
        from groq import GroqError

        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "Respond only with a JSON object matching the requested schema."},
                {"role": "user", "content": prompt},
            ],
            "response_format": response_format,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            response = self._client.chat.completions.create(**kwargs)
        except GroqError as exc:
            raise LLMError(f"Groq request failed: {exc}") from exc
        return response.choices[0].message.content or "{}"

    def generate_structured(self, prompt: str, schema: type[T], *, context: dict[str, Any] | None = None) -> T:
        from groq import BadRequestError

        last_error: Exception | None = None
        for attempt in range(2):
            if self._strict_supported:
                fmt = {"type": "json_schema",
                       "json_schema": {"name": schema.__name__, "schema": strict_json_schema(schema), "strict": True}}
            else:
                fmt = {"type": "json_object"}
            text_prompt = prompt if self._strict_supported else (
                f"{prompt}\n\nJSON schema:\n{json.dumps(schema.model_json_schema())}"
            )
            try:
                content = self._complete(text_prompt, fmt)
            except LLMError as exc:
                if self._strict_supported and isinstance(exc.__cause__, BadRequestError):
                    logger.warning("%s rejected strict JSON schema; falling back to JSON mode.", self.model)
                    self._strict_supported = False
                    continue
                raise
            try:
                return schema.model_validate_json(content)
            except Exception as exc:  # pydantic.ValidationError or malformed JSON
                last_error = exc
                logger.warning("Groq returned invalid structured output (attempt %d): %s", attempt + 1, exc)
        raise LLMError(f"Groq returned invalid structured output: {last_error}")


class MockLLMClient:
    """Deterministic extractive stand-in for an LLM (clearly labelled MOCK)."""

    name = "mock-extractive"
    is_mock = True

    def __init__(self, max_claims: int = 3, min_overlap: int = 1) -> None:
        self.max_claims = max_claims
        self.min_overlap = min_overlap

    def generate_structured(self, prompt: str, schema: type[T], *, context: dict[str, Any] | None = None) -> T:
        from app.generation.prompts import AnswerSchema, ExtractedClaims, SupportJudgement

        context = context or {}
        if schema is AnswerSchema:
            return schema.model_validate(self._answer(context))  # type: ignore[return-value]
        if schema is ExtractedClaims:
            return schema.model_validate({"claims": []})  # type: ignore[return-value]
        if schema is SupportJudgement:
            return schema.model_validate({"label": "unsupported", "explanation": "mock"})  # type: ignore[return-value]
        raise LLMError(f"Mock LLM does not support schema {schema.__name__}")

    def _answer(self, context: dict[str, Any]) -> dict[str, Any]:
        question: str = context.get("question", "")
        evidence = context.get("evidence", {})  # label -> RetrievedChunk
        drop = set(context.get("drop_claims", []))
        min_relevance = context.get("min_relevance")
        terms = content_terms(question) | set(context.get("extra_terms", []))
        candidates = []
        for rank, (label, item) in enumerate(evidence.items()):
            if min_relevance is not None and item.relevance < min_relevance:
                continue
            for sentence in split_sentences(item.chunk.text):
                overlap = len(terms & content_terms(sentence))
                if overlap >= self.min_overlap and sentence not in drop:
                    # Prefer rule-like sentences; still purely lexical, no reasoning.
                    overlap += 1 if _NORMATIVE_RE.search(sentence) else 0
                    candidates.append((overlap, -rank, sentence, label))
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        claims, seen = [], set()
        for _, _, sentence, label in candidates:
            if sentence in seen:
                continue
            seen.add(sentence)
            claims.append({"text": sentence, "citation_ids": [label]})
            if len(claims) >= self.max_claims:
                break
        for forced in context.get("force_claims", []):  # used by correction flows
            claims.append(forced)
        return {"claims": claims, "insufficient_evidence": not claims, "notes": "MOCK extractive answer"}


def build_llm(settings: Settings) -> LLMClient:
    provider = settings.resolved_llm_provider()
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set.")
        return GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.llm_temperature, settings.llm_timeout_seconds)
    if provider == "groq":
        if not settings.groq_api_key:
            raise LLMError("LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
        return GroqClient(
            settings.groq_api_key, settings.groq_model, settings.llm_temperature, settings.llm_timeout_seconds,
            settings.llm_max_retries, settings.groq_reasoning_effort,
        )
    if provider == "mock":
        logger.warning("Using the MOCK extractive LLM. Answers are not produced by a real language model.")
        return MockLLMClient()
    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")
