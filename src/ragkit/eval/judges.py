"""Answer-correctness judges. Audit an LLM judge on hand labels before reporting its numbers."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from ragkit.core.registry import JUDGES
from ragkit.core.schemas import Judgement
from ragkit.core.stages import LLMClient
from ragkit.generation import prompts

_STOP = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "then",
        "this",
        "to",
        "up",
        "was",
        "what",
        "when",
        "which",
        "with",
        "you",
        "your",
        "yes",
        "no",
    ]
)


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if w not in _STOP}


class LexicalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_recall: float = Field(0.6, description="Share of reference content words the answer needs.")


class LexicalJudge:
    """Offline proxy: content-word recall of the reference, and every reference number must match."""

    def __init__(self, params: LexicalParams):
        self.p = params

    def judge(self, question: str, reference: str, answer: str) -> Judgement:
        ref, ans = _content(reference), _content(answer)
        numbers = {w for w in ref if w.isdigit()}
        if numbers - ans:
            return Judgement(correct=False, reason=f"missing numbers {sorted(numbers - ans)}")
        recall = len(ref & ans) / len(ref) if ref else 0.0
        return Judgement(correct=recall >= self.p.min_recall, reason=f"recall={recall:.2f}")


class Correctness(BaseModel):
    correct: bool
    reason: str = Field(description="One sentence explaining the grade.")


class LLMJudgeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMJudge:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def judge(self, question: str, reference: str, answer: str) -> Judgement:
        out, usage = self.llm.generate_structured(
            prompts.JUDGE_CORRECTNESS, prompts.judge_user(question, reference, answer), Correctness
        )
        return Judgement(correct=out.correct, reason=out.reason, usage=usage)


@JUDGES.register("lexical", LexicalParams)
def _lexical(params: LexicalParams, **_) -> LexicalJudge:
    return LexicalJudge(params)


@JUDGES.register("llm", LLMJudgeParams)
def _llm(params: LLMJudgeParams, llm: LLMClient, **_) -> LLMJudge:
    return LLMJudge(llm)
