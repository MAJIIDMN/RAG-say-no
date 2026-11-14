from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from ragkit.core.registry import GENERATORS
from ragkit.core.schemas import Claim, ClaimCitation, GeneratedAnswer, Generation, ScoredChunk
from ragkit.core.stages import LLMClient
from ragkit.core.text import words
from ragkit.generation import prompts

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _prose_sentences(text: str) -> list[str]:
    """Sentences from prose lines only: headings and fenced code are not answers."""
    out, in_code = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or line.lstrip().startswith("#"):
            continue
        out += [s.strip() for s in _SENTENCE.split(line) if s.strip()]
    return out


class ExtractiveParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_overlap: float = Field(0.0, description="Answer only above this query-token overlap.")


class ExtractiveGenerator:
    """No-LLM baseline: the sentence from the top passages that shares the most query tokens."""

    def __init__(self, params: ExtractiveParams):
        self.p = params

    def generate(self, query: str, context: list[ScoredChunk]) -> Generation:
        q = set(words(query))
        best: tuple[float, int, str] = (0.0, 0, "")
        for i, sc in enumerate(context, start=1):
            for sentence in _prose_sentences(sc.chunk.text):
                overlap = len(q & set(words(sentence))) / (len(q) or 1)
                if overlap > best[0]:
                    best = (overlap, i, sentence)
        overlap, label, sentence = best
        if not sentence or overlap <= self.p.min_overlap:
            return Generation(
                answer=GeneratedAnswer(
                    status="insufficient_context", missing_information="no overlap"
                )
            )
        claim = Claim(
            text=sentence, citations=[ClaimCitation(source_id=f"S{label}", quote=sentence)]
        )
        return Generation(answer=GeneratedAnswer(status="answered", claims=[claim]))


@GENERATORS.register("extractive", ExtractiveParams)
def _extractive(params: ExtractiveParams, **_) -> ExtractiveGenerator:
    return ExtractiveGenerator(params)


class LLMGeneratorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allow_abstain: bool = Field(
        True, description="False uses the naive prompt for the no-abstention ablation variants."
    )


class LLMGenerator:
    def __init__(self, params: LLMGeneratorParams, llm: LLMClient):
        self.llm = llm
        self.system = prompts.ANSWER_GROUNDED if params.allow_abstain else prompts.ANSWER_NAIVE

    def generate(self, query: str, context: list[ScoredChunk]) -> Generation:
        try:
            answer, usage = self.llm.generate_structured(
                self.system, prompts.answer_user(query, context), GeneratedAnswer
            )
        except Exception as e:  # noqa: BLE001 - a failed call is an abstention, never a crash
            return Generation(answer=None, error=f"{type(e).__name__}: {e}"[:500])
        return Generation(answer=answer, usage=usage)


@GENERATORS.register("llm", LLMGeneratorParams)
def _llm(params: LLMGeneratorParams, llm: LLMClient | None = None, **_) -> LLMGenerator:
    if llm is None:
        from ragkit.llm.base import client_for

        llm = client_for("generator")
    return LLMGenerator(params, llm)
