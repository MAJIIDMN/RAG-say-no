"""Native Anthropic API with schema-constrained structured output (`messages.parse`)."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from ragkit.config import LLMSettings
from ragkit.core.registry import LLM_CLIENTS, NoParams
from ragkit.llm.base import StructuredOutputError

T = TypeVar("T", bound=BaseModel)


class AnthropicClient:
    def __init__(self, settings: LLMSettings):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("Needs the `llm` extra: uv sync --extra llm") from e
        self.settings = settings
        self.model = settings.model
        self.client = anthropic.Anthropic(api_key=settings.api_key)

    def generate_structured(
        self, system: str, user: str, schema: type[T]
    ) -> tuple[T, dict[str, int]]:
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=self.settings.max_tokens,
            # Identical for every query, so it is a cacheable prefix.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if resp.stop_reason == "refusal":
            raise StructuredOutputError(f"model declined: {getattr(resp, 'stop_details', None)}")
        if resp.parsed_output is None:
            raise StructuredOutputError(f"no parsed output (stop_reason={resp.stop_reason})")
        u = resp.usage
        return resp.parsed_output, {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_input_tokens": u.cache_read_input_tokens or 0,
            "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
        }


@LLM_CLIENTS.register("anthropic")
def _anthropic(params: NoParams, settings: LLMSettings) -> AnthropicClient:
    return AnthropicClient(settings)
