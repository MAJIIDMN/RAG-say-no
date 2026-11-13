from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from ragkit.config import LLMSettings
from ragkit.core.schemas import GeneratedAnswer
from ragkit.llm.base import StructuredOutputError, create_llm_client
from ragkit.llm.structured import extract_json, strict_json_schema

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx2")


class Verdict(BaseModel):
    ok: bool
    reason: str = ""


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"ok": true}\n```') == '{"ok": true}'
    assert extract_json('prefix {"a": {"b": "}"}} suffix') == '{"a": {"b": "}"}}'
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_strict_schema_closes_objects_and_requires_all_fields():
    schema = strict_json_schema(GeneratedAnswer)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    claim = schema["$defs"]["Claim"]
    assert claim["additionalProperties"] is False


class StubCompletions:
    """Rejects json_schema like providers without schema support, then answers in text."""

    def __init__(self, content):
        self.content, self.calls = content, []

    def create(self, **kw):
        mode = (kw.get("response_format") or {}).get("type", "text")
        self.calls.append(mode)
        if mode == "json_schema":
            req = httpx.Request("POST", "http://stub")
            raise openai.BadRequestError(
                "json_schema unsupported", response=httpx.Response(400, request=req), body=None
            )
        msg = SimpleNamespace(content=self.content)
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def client_with(content, modes="json_schema,json_object,text"):
    settings = LLMSettings(
        provider="openai_compatible", model="m", api_key="k", structured_modes=modes
    )
    client = create_llm_client(settings)
    stub = StubCompletions(content)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return client, stub


def test_falls_back_from_json_schema_to_json_object():
    client, stub = client_with('{"ok": true, "reason": "fine"}')
    out, usage = client.generate_structured("sys", "user", Verdict)
    assert out == Verdict(ok=True, reason="fine")
    assert stub.calls == ["json_schema", "json_object"]
    assert usage == {"input_tokens": 11, "output_tokens": 7}


def test_invalid_output_in_every_mode_raises_structured_error():
    client, stub = client_with("I cannot answer that.")
    with pytest.raises(StructuredOutputError) as err:
        client.generate_structured("sys", "user", Verdict)
    assert stub.calls == ["json_schema", "json_object", "text"]
    assert "json_object" in str(err.value)


def test_missing_model_is_a_clear_configuration_error():
    with pytest.raises(ValueError, match="LLM_MODEL"):
        create_llm_client(LLMSettings(provider="openai_compatible", model=""))


def test_fake_client_records_calls():
    client = create_llm_client(LLMSettings(provider="fake"))
    out, _ = client.generate_structured("s", "u", Verdict)
    assert isinstance(out, Verdict) and client.calls[0][2] == "Verdict"
