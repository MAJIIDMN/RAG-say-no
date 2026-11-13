"""Any endpoint that speaks the OpenAI chat-completions protocol (OpenRouter, DeepSeek, Groq, ...)."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from ragkit.config import LLMSettings
from ragkit.core.registry import LLM_CLIENTS
from ragkit.llm.base import StructuredOutputError
from ragkit.llm.structured import extract_json, schema_instructions, strict_json_schema

T = TypeVar("T", bound=BaseModel)


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenAICompatibleClient:
    def __init__(self, settings: LLMSettings):
        try:
            import openai
        except ImportError as e:
            raise ImportError("Needs the `llm` extra: uv sync --extra llm") from e
        self._openai = openai
        self.settings = settings
        self.model = settings.model
        # Local servers (Ollama, vLLM) accept any key; the SDK refuses an empty one.
        self.client = openai.OpenAI(
            base_url=settings.base_url, api_key=settings.api_key or "unused"
        )

    def _request(self, mode: str, system: str, user: str, schema: type[BaseModel]) -> Any:
        kwargs: dict[str, Any] = {}
        if mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": strict_json_schema(schema),
                    "strict": True,
                },
            }
        else:
            system += schema_instructions(schema)
            if mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=self.settings.max_tokens,
            **kwargs,
        )

    def generate_structured(
        self, system: str, user: str, schema: type[T]
    ) -> tuple[T, dict[str, int]]:
        oa, errors = self._openai, []
        for mode in self.settings.structured_modes:
            try:
                resp = self._request(mode, system, user, schema)
            except (oa.BadRequestError, oa.UnprocessableEntityError) as e:
                # Usually "response_format not supported by this model": try the next mode.
                errors.append(f"{mode}: {type(e).__name__}: {e}")
                continue
            content = resp.choices[0].message.content or ""
            try:
                parsed = schema.model_validate_json(extract_json(content))
            except (ValueError, ValidationError) as e:
                errors.append(f"{mode}: invalid output: {str(e)[:200]}")
                continue
            u = resp.usage
            usage = (
                {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens} if u else {}
            )
            return parsed, usage
        raise StructuredOutputError("; ".join(errors) or "no structured_modes configured")


@LLM_CLIENTS.register("openai_compatible", _NoParams)
def _openai_compatible(params: _NoParams, settings: LLMSettings) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(settings)
