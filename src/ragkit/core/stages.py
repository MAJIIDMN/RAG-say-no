"""The seams of the pipeline. Each Protocol has several adapters, chosen by `impl:` in YAML."""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np
from pydantic import BaseModel

from ragkit.core.schemas import Chunk, Claim, Document, Generation, ScoredChunk

T = TypeVar("T", bound=BaseModel)


class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


class Embedder(Protocol):
    """Returns L2-normalised float32 vectors, so a dot product is cosine similarity."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[ScoredChunk]: ...


class Reranker(Protocol):
    """Returns at most `top_n` chunks, best first; `score` is the value abstention gates read."""

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...


class LLMClient(Protocol):
    model: str

    def generate_structured(
        self, system: str, user: str, schema: type[T]
    ) -> tuple[T, dict[str, int]]: ...


class Generator(Protocol):
    """Never raises on model failure: returns `Generation(answer=None, error=...)` instead."""

    def generate(self, query: str, context: list[ScoredChunk]) -> Generation: ...


class Verifier(Protocol):
    """One verdict per claim: is it supported by the passages it cites?"""

    def verify(self, claims: list[Claim], context: list[ScoredChunk]) -> list[bool]: ...


class Judge(Protocol):
    """Eval only: does `answer` state the key facts of `reference`?"""

    def correct(self, question: str, reference: str, answer: str) -> bool: ...
