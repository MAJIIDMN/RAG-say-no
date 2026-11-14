"""Scripted LLM for tests and offline smoke runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, get_args, get_origin

from pydantic import BaseModel

from ragkit.config import LLMSettings
from ragkit.core.registry import LLM_CLIENTS, NoParams

T = TypeVar("T", bound=BaseModel)
Responder = Callable[[str, str, type[BaseModel]], BaseModel]


def default_value(annotation: Any) -> Any:
    if annotation is bool:
        return True
    if annotation is str:
        return "fake"
    if get_origin(annotation) is list:
        return []
    args = get_args(annotation)
    return args[0] if args and isinstance(args[0], str) else None


def agreeable(system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    """Says yes to every boolean and picks the first option of every Literal."""
    return schema(
        **{
            k: default_value(f.annotation)
            for k, f in schema.model_fields.items()
            if f.is_required()
        }
    )


class FakeLLMClient:
    def __init__(self, responder: Responder | None = None):
        self.model = "fake"
        self.responder = responder or agreeable
        self.calls: list[tuple[str, str, str]] = []

    def generate_structured(
        self, system: str, user: str, schema: type[T]
    ) -> tuple[T, dict[str, int]]:
        self.calls.append((system, user, schema.__name__))
        return self.responder(system, user, schema), {"input_tokens": 0, "output_tokens": 0}  # type: ignore[return-value]


@LLM_CLIENTS.register("fake")
def _fake(params: NoParams, settings: LLMSettings | None = None) -> FakeLLMClient:
    return FakeLLMClient()
