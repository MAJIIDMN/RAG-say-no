"""Rerankers. The top reranked score is the confidence that abstention gate 1 reads."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ragkit.core.registry import RERANKERS, NoParams
from ragkit.core.schemas import ScoredChunk
from ragkit.core.text import words


def _rescored(candidates: list[ScoredChunk], scores: list[float], top_n: int) -> list[ScoredChunk]:
    out = [
        ScoredChunk(chunk=c.chunk, score=s, scores={**c.scores, "rerank": s})
        for c, s in zip(candidates, scores, strict=True)
    ]
    return sorted(out, key=lambda c: c.score, reverse=True)[:top_n]


class NoReranker:
    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        return candidates[:top_n]


class LexicalReranker:
    """Fraction of query tokens present in the passage. Offline stand-in for tests and CI."""

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        q = set(words(query))
        scores = [
            len(q & set(words(c.chunk.indexed_text))) / len(q) if q else 0.0 for c in candidates
        ]
        return _rescored(candidates, scores, top_n)


class CrossEncoderParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    device: str = "cpu"


class CrossEncoderReranker:
    def __init__(self, params: CrossEncoderParams):
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError("Needs the `ml` extra: uv sync --extra ml") from e
        self.p = params
        self.model = CrossEncoder(params.model, max_length=params.max_length, device=params.device)
        self._sigmoid = torch.nn.Sigmoid()

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self.model.predict(
            [(query, c.chunk.indexed_text) for c in candidates],
            batch_size=self.p.batch_size,
            activation_fn=self._sigmoid,  # explicit, so thresholds stay in [0, 1] across models
            show_progress_bar=False,
        )
        return _rescored(candidates, [float(s) for s in scores], top_n)


@RERANKERS.register("none")
def _none(params: NoParams) -> NoReranker:
    return NoReranker()


@RERANKERS.register("lexical")
def _lexical(params: NoParams) -> LexicalReranker:
    return LexicalReranker()


@RERANKERS.register("cross_encoder", CrossEncoderParams)
def _cross_encoder(params: CrossEncoderParams) -> CrossEncoderReranker:
    return CrossEncoderReranker(params)
