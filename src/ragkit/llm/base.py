from __future__ import annotations

from ragkit.config import LLMSettings, llm_settings
from ragkit.core.registry import LLM_CLIENTS
from ragkit.core.stages import LLMClient


class StructuredOutputError(RuntimeError):
    """The model returned nothing that validates against the schema in any configured mode."""


def create_llm_client(settings: LLMSettings) -> LLMClient:
    if settings.provider != "fake" and not settings.model:
        raise ValueError(
            "No model configured: set LLM_MODEL (and JUDGE_MODEL for the judge) in .env"
        )
    return LLM_CLIENTS.create({"impl": settings.provider}, settings=settings)


def client_for(role: str = "generator") -> LLMClient:
    return create_llm_client(llm_settings(role))  # type: ignore[arg-type]
