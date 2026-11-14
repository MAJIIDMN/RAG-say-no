"""Claim verification: drop answer sentences their citations do not support."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ragkit.core.registry import VERIFIERS, NoParams
from ragkit.core.schemas import Chunk, Citation, Claim, ScoredChunk, Verification, add_usage
from ragkit.core.stages import LLMClient
from ragkit.generation import prompts

_EDGE = "\"'“”‘’.,;:… "


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in(quote: str, text: str) -> bool:
    q = _norm(quote).strip(_EDGE)
    return bool(q) and q in _norm(text)


def _labels(context: list[ScoredChunk]) -> dict[str, Chunk]:
    return {f"S{i}": sc.chunk for i, sc in enumerate(context, start=1)}


def _resolve(label: str, by_label: dict[str, Chunk]) -> Chunk | None:
    return by_label.get(label.strip().strip("[]").upper())


def resolve_citations(claims: list[Claim], context: list[ScoredChunk]) -> list[Citation]:
    by_label, seen, out = _labels(context), set(), []
    for cl in claims:
        for cit in cl.citations:
            chunk = _resolve(cit.source_id, by_label)
            if chunk is None or (chunk.chunk_id, cit.quote) in seen:
                continue
            seen.add((chunk.chunk_id, cit.quote))
            out.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    url=str(chunk.metadata.get("url", "")),
                    quote=cit.quote,
                    verified=quote_in(cit.quote, chunk.text),
                )
            )
    return out


class Entailment(BaseModel):
    supported: bool
    reason: str = Field(description="One sentence explaining the decision.")


class NoVerifier:
    def verify(self, claims: list[Claim], context: list[ScoredChunk]) -> Verification:
        return Verification(verdicts=[True] * len(claims))


class QuoteVerifier:
    """Supported if at least one citation's quote appears verbatim in the passage it names."""

    def verify(self, claims: list[Claim], context: list[ScoredChunk]) -> Verification:
        by_label = _labels(context)
        verdicts = [
            any(
                (chunk := _resolve(c.source_id, by_label)) is not None
                and quote_in(c.quote, chunk.text)
                for c in cl.citations
            )
            for cl in claims
        ]
        return Verification(verdicts=verdicts)


class LLMVerifier:
    """Asks an LLM whether each claim is entailed by the passages it cites. One call per claim."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def verify(self, claims: list[Claim], context: list[ScoredChunk]) -> Verification:
        by_label, out, total = _labels(context), [], {}
        for cl in claims:
            evidence = [ch.text for c in cl.citations if (ch := _resolve(c.source_id, by_label))]
            if not evidence:
                out.append(False)
                continue
            try:
                verdict, usage = self.llm.generate_structured(
                    prompts.VERIFY_CLAIM, prompts.verify_user(cl.text, evidence), Entailment
                )
            except Exception as e:
                return Verification(verdicts=[], usage=total, error=f"{type(e).__name__}: {e}")
            add_usage(total, usage)
            out.append(verdict.supported)
        return Verification(verdicts=out, usage=total)


@VERIFIERS.register("none")
def _none(params: NoParams, **_) -> NoVerifier:
    return NoVerifier()


@VERIFIERS.register("quote")
def _quote(params: NoParams, **_) -> QuoteVerifier:
    return QuoteVerifier()


@VERIFIERS.register("llm")
def _llm(params: NoParams, llm: LLMClient | None = None, **_) -> LLMVerifier:
    if llm is None:
        from ragkit.llm.base import client_for

        llm = client_for("generator")
    return LLMVerifier(llm)
