"""LLM client behaviour without network access (the Groq SDK client is replaced by a fake)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.generation.llm_client import GroqClient, LLMError, MockLLMClient, build_llm, strict_json_schema
from app.generation.prompts import AnswerSchema
from tests.conftest import make_settings


class FakeCompletions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out))])


def groq_with(outputs, model="openai/gpt-oss-120b") -> tuple[GroqClient, FakeCompletions]:
    client = GroqClient("test-key", model)
    fake = FakeCompletions(outputs)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


VALID = '{"claims": [{"text": "Passwords need 14 characters.", "citation_ids": ["E1"]}], "insufficient_evidence": false, "notes": null}'


def test_strict_schema_is_self_contained_and_closed():
    schema = strict_json_schema(AnswerSchema)
    assert "$defs" not in str(schema) and "$ref" not in str(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"claims", "insufficient_evidence", "notes"}
    item = schema["properties"]["claims"]["items"]
    assert item["additionalProperties"] is False and set(item["required"]) == {"text", "citation_ids"}


def test_groq_uses_strict_json_schema_and_low_reasoning_effort():
    client, fake = groq_with([VALID])
    answer = client.generate_structured("prompt", AnswerSchema)
    assert answer.claims[0].citation_ids == ["E1"]
    call = fake.calls[0]
    assert call["response_format"]["type"] == "json_schema" and call["response_format"]["json_schema"]["strict"]
    assert call["reasoning_effort"] == "low" and call["temperature"] == 0.0


def test_groq_reasks_once_on_invalid_output_then_fails_loudly():
    client, fake = groq_with(['{"claims": "oops"}', VALID])
    assert client.generate_structured("p", AnswerSchema).claims
    assert len(fake.calls) == 2
    client, _ = groq_with(['not json', '{"claims": 1}'])
    with pytest.raises(LLMError):
        client.generate_structured("p", AnswerSchema)


def test_groq_falls_back_to_json_mode_when_strict_schema_rejected():
    import httpx
    from groq import BadRequestError

    rejected = BadRequestError("strict not supported", response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None)
    client, fake = groq_with([rejected, VALID], model="qwen/some-model")
    assert client.generate_structured("p", AnswerSchema).claims
    assert fake.calls[1]["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in fake.calls[1]  # only sent to gpt-oss models


def test_provider_selection():
    assert make_settings(llm_provider="auto", groq_api_key="k").resolved_llm_provider() == "groq"
    assert make_settings(llm_provider="auto", groq_api_key="k", gemini_api_key="g").resolved_llm_provider() == "gemini"
    assert make_settings(llm_provider="auto").resolved_llm_provider() == "mock"
    assert isinstance(build_llm(make_settings(llm_provider="groq", groq_api_key="k")), GroqClient)
    assert isinstance(build_llm(make_settings(llm_provider="mock")), MockLLMClient)
    with pytest.raises(LLMError):
        build_llm(make_settings(llm_provider="groq", groq_api_key=None))
