from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragkit.core.registry import RETRIEVERS, StageSpec
from ragkit.core.schemas import ScoredChunk
from ragkit.core.stages import Retriever


def rrf_fuse(
    ranked_lists: list[list[ScoredChunk]], k: int = 60, top_k: int = 50
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank). Insensitive to score scales."""
    fused: dict[str, ScoredChunk] = {}
    for results in ranked_lists:
        for rank, sc in enumerate(results, start=1):
            entry = fused.setdefault(
                sc.chunk.chunk_id, ScoredChunk(chunk=sc.chunk, score=0.0, scores={})
            )
            entry.score += 1.0 / (k + rank)
            entry.scores.update(sc.scores)
    out = sorted(fused.values(), key=lambda s: s.score, reverse=True)[:top_k]
    for sc in out:
        sc.scores["rrf"] = sc.score
    return out


class HybridParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retrievers: list[StageSpec] = Field(
        default_factory=lambda: [StageSpec(impl="bm25"), StageSpec(impl="dense")]
    )
    rrf_k: int = 60


class HybridRetriever:
    def __init__(self, retrievers: list[Retriever], rrf_k: int):
        self.retrievers, self.rrf_k = retrievers, rrf_k

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        return rrf_fuse([r.search(query, k) for r in self.retrievers], k=self.rrf_k, top_k=k)


@RETRIEVERS.register("hybrid", HybridParams)
def _hybrid(params: HybridParams, **deps: Any) -> HybridRetriever:
    return HybridRetriever([RETRIEVERS.create(s, **deps) for s in params.retrievers], params.rrf_k)
