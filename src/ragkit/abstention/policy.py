"""Abstention gates. Every refusal carries a reason so reports can split them by gate."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ragkit.config import AbstentionConfig
from ragkit.core.schemas import Generation, ScoredChunk
from ragkit.core.stages import LLMClient
from ragkit.generation import prompts


class Sufficiency(BaseModel):
    sufficient: bool = Field(description="True only if the passages fully answer the question.")
    reason: str = Field(description="One sentence explaining the decision.")


@dataclass
class Decision:
    reason: str | None = None
    detail: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def abstain(self) -> bool:
        return self.reason is not None


class AbstentionPolicy:
    def __init__(self, cfg: AbstentionConfig, llm: LLMClient | None = None):
        self.cfg, self.llm = cfg, llm

    def before_generation(self, query: str, context: list[ScoredChunk]) -> Decision:
        if not context:
            return Decision("no_retrieval_results")
        if not self.cfg.enabled:
            return Decision()
        if context[0].score < self.cfg.min_score:
            return Decision("low_retrieval_score")
        if self.cfg.sufficiency_check:
            if self.llm is None:
                raise ValueError("abstention.sufficiency_check needs an LLM client")
            verdict, usage = self.llm.generate_structured(
                prompts.SUFFICIENCY, prompts.answer_user(query, context), Sufficiency
            )
            if not verdict.sufficient:
                return Decision("insufficient_context", verdict.reason, usage)
            return Decision(usage=usage)
        return Decision()

    def after_generation(self, generation: Generation) -> Decision:
        if generation.answer is None:
            return Decision("generation_failed", generation.error or "")
        answer = generation.answer
        if self.cfg.enabled and self.cfg.respect_status and answer.status == "insufficient_context":
            return Decision("llm_insufficient_context", answer.missing_information)
        if not answer.claims:
            return Decision("empty_answer")
        return Decision()
