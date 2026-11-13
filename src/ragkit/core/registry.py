"""Name → factory registries. Configs pick implementations by name; the pipeline never branches."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class StageSpec(BaseModel):
    """`{impl: <name>, ...params}` from YAML. Params are validated by the implementation."""

    model_config = ConfigDict(extra="allow")
    impl: str

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, tuple[Callable[..., T], type[BaseModel]]] = {}

    def register(self, name: str, params: type[BaseModel] | None = None):
        if name in self._entries:
            raise ValueError(f"{self.kind} '{name}' is already registered")

        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            self._entries[name] = (factory, params or _NoParams)
            return factory

        return deco

    def names(self) -> list[str]:
        return sorted(self._entries)

    def params_model(self, name: str) -> type[BaseModel]:
        return self._lookup(name)[1]

    def create(self, spec: StageSpec | dict[str, Any], **deps: Any) -> T:
        spec = spec if isinstance(spec, StageSpec) else StageSpec.model_validate(spec)
        factory, params_model = self._lookup(spec.impl)
        unknown = set(spec.params) - set(params_model.model_fields)
        if unknown:
            raise ValueError(f"{self.kind} '{spec.impl}' got unknown params: {sorted(unknown)}")
        return factory(params_model.model_validate(spec.params), **deps)

    def _lookup(self, name: str) -> tuple[Callable[..., T], type[BaseModel]]:
        if name not in self._entries:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names())}")
        return self._entries[name]


CHUNKERS: Registry = Registry("chunker")
EMBEDDERS: Registry = Registry("embedder")
RETRIEVERS: Registry = Registry("retriever")
RERANKERS: Registry = Registry("reranker")
LLM_CLIENTS: Registry = Registry("llm client")
GENERATORS: Registry = Registry("generator")
VERIFIERS: Registry = Registry("verifier")
JUDGES: Registry = Registry("judge")
