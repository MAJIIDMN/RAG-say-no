from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from ragkit.core.registry import RETRIEVERS
from ragkit.core.schemas import ScoredChunk
from ragkit.core.stages import Embedder
from ragkit.index.store import Index


class DenseParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DenseRetriever:
    """Exact cosine search; vectors are L2-normalised so a dot product suffices."""

    def __init__(self, index: Index, embedder: Embedder):
        self.index, self.embedder = index, embedder

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if not self.index.size:
            return []
        sims = self.index.vectors @ self.embedder.embed_query(query)
        top = np.argsort(-sims)[:k]
        return [
            ScoredChunk(
                chunk=self.index.chunks[i], score=float(sims[i]), scores={"dense": float(sims[i])}
            )
            for i in top
        ]


@RETRIEVERS.register("dense", DenseParams)
def _dense(params: DenseParams, index: Index, embedder: Embedder, **_) -> DenseRetriever:
    return DenseRetriever(index, embedder)
