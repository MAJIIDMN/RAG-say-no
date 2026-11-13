"""Experiment configuration.

Pipeline structure lives in YAML; a file may `extends:` another, and keys are deep-merged so an
ablation variant lists only what it changes. Secrets and model choice live in the environment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from ragkit.core.registry import StageSpec


class CorpusConfig(BaseModel):
    path: str = "data/toy"
    glob: str = "**/*"
    extensions: list[str] = [".md", ".txt", ".html", ".pdf"]


class RetrievalConfig(BaseModel):
    candidates: int = Field(50, description="Top-k from each retriever and after fusion.")
    rerank_candidates: int = Field(30, description="How many fused candidates the reranker scores.")
    context_size: int = Field(5, description="Chunks passed to the generator.")


class AbstentionConfig(BaseModel):
    enabled: bool = True
    min_score: float = Field(0.0, description="Gate 1 threshold on the top rerank score.")
    sufficiency_check: bool = Field(False, description="Gate 2: separate LLM sufficiency call.")
    respect_status: bool = Field(True, description="Gate 3: trust status=insufficient_context.")
    message: str = "The documentation does not contain an answer to this question."


class RAGConfig(BaseModel):
    name: str = "default"
    description: str = ""
    corpus: CorpusConfig = CorpusConfig()
    chunker: StageSpec = StageSpec(impl="markdown_heading")
    embedder: StageSpec = StageSpec(impl="hashing")
    retriever: StageSpec = StageSpec(impl="bm25")
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: StageSpec = StageSpec(impl="none")
    generator: StageSpec = StageSpec(impl="extractive")
    abstention: AbstentionConfig = AbstentionConfig()
    verifier: StageSpec = StageSpec(impl="none")
    judge: StageSpec = StageSpec(impl="lexical")
    artifacts_dir: str = "artifacts"

    def index_key(self) -> str:
        """Hash of everything that changes the stored index. Variants sharing it share the index."""
        payload = {
            "corpus": self.corpus.model_dump(),
            "chunker": self.chunker.model_dump(),
            "embedder": self.embedder.model_dump(),
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.chunker.impl}-{self.embedder.impl}-{digest}"

    def index_dir(self) -> Path:
        return Path(self.artifacts_dir) / "index" / self.index_key()


class LLMSettings(BaseSettings):
    """One LLM endpoint. Read from LLM_* (generator) or JUDGE_* (judge) environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider: Literal["openai_compatible", "anthropic"] = "openai_compatible"
    model: str = ""
    base_url: str | None = None
    api_key: str | None = None
    structured_modes: Annotated[list[str], NoDecode] = ["json_schema", "json_object", "text"]
    max_tokens: int = 4096
    input_price_per_mtok: float = 0.0
    output_price_per_mtok: float = 0.0

    @field_validator("structured_modes", mode="before")
    @classmethod
    def _split(cls, v: Any) -> Any:
        return [m.strip() for m in v.split(",") if m.strip()] if isinstance(v, str) else v

    @field_validator("base_url", "api_key", mode="before")
    @classmethod
    def _empty_is_none(cls, v: Any) -> Any:
        return v or None


def llm_settings(role: Literal["generator", "judge"] = "generator") -> LLMSettings:
    generator = LLMSettings(_env_prefix="LLM_")
    if role == "generator":
        return generator
    judge = LLMSettings(_env_prefix="JUDGE_")
    # Unset JUDGE_* values inherit from LLM_*.
    explicit = {k: v for k, v in judge.model_dump().items() if k in judge.model_fields_set and v}
    return generator.model_copy(update=explicit)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_raw(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular `extends` at {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    return deep_merge(_load_raw(path.parent / parent, seen), raw) if parent else raw


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RAGConfig:
    raw = _load_raw(Path(path), set())
    return RAGConfig.model_validate(deep_merge(raw, overrides or {}))


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """["retrieval.context_size=8"] -> {"retrieval": {"context_size": 8}}; values parsed as YAML."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        node = out
        *parents, leaf = key.strip().split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return out
