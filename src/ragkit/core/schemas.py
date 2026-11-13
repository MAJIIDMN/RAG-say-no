"""Typed data passed between pipeline stages."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    title: str = ""
    section: str = Field(default="", description="Heading path, e.g. 'Page > H2 > H3'.")
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def indexed_text(self) -> str:
        """Text given to embedders and BM25: the breadcrumb supplies context a bare chunk lacks."""
        return f"{self.section}\n{self.text}" if self.section else self.text


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    scores: dict[str, float] = Field(
        default_factory=dict, description="Score from every stage that touched the chunk."
    )


# LLM structured output. Kept flat so strict JSON-schema modes (Anthropic, Groq) accept it.


class ClaimCitation(BaseModel):
    source_id: str = Field(description="ID of a context passage, e.g. 'S2'.")
    quote: str = Field(description="Short quote copied character for character from that passage.")


class Claim(BaseModel):
    text: str = Field(description="One sentence of the answer.")
    citations: list[ClaimCitation] = Field(description="Passages that support this sentence.")


class GeneratedAnswer(BaseModel):
    status: Literal["answered", "insufficient_context"] = Field(
        description="'insufficient_context' unless the passages fully answer the question."
    )
    claims: list[Claim] = Field(default_factory=list, description="Empty when not answered.")
    missing_information: str = Field(
        default="", description="When not answered: what the passages lack."
    )


class Generation(BaseModel):
    answer: GeneratedAnswer | None
    usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None


class Verification(BaseModel):
    verdicts: list[bool] = Field(description="One per claim: supported by the passages it cites.")
    usage: dict[str, int] = Field(default_factory=dict)


# Public response.


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str = ""
    url: str = ""
    quote: str
    verified: bool = Field(
        description="Quote found verbatim (whitespace/case-normalised) in chunk."
    )


class RAGResponse(BaseModel):
    query: str
    answered: bool
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(description="Top reranker score; in [0, 1] for cross-encoders.")
    abstain_reason: str | None = None
    retrieved: list[ScoredChunk] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(
        default_factory=dict, description="Raw signals kept so eval can re-simulate gates offline."
    )
