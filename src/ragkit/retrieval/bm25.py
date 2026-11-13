from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ragkit.core.registry import RETRIEVERS
from ragkit.core.schemas import ScoredChunk
from ragkit.index.store import Index
from ragkit.ingest.chunkers import _TOKEN


class BM25Params(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = Field(
        "lucene", description="bm25s variant: lucene, robertson, atire, bm25l, bm25+"
    )
    stopwords: list[str] = Field(default_factory=list)


class BM25Retriever:
    def __init__(self, params: BM25Params, index: Index):
        import bm25s

        self.index = index
        self.stopwords = frozenset(w.lower() for w in params.stopwords)
        self.bm25 = bm25s.BM25(method=params.method)
        corpus = [self._tokens(c.indexed_text) or ["_"] for c in index.chunks]
        self.bm25.index(corpus, show_progress=False)

    def _tokens(self, text: str) -> list[str]:
        return [t for t in _TOKEN.findall(text.lower()) if t.isalnum() and t not in self.stopwords]

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        tokens = self._tokens(query)
        if not tokens or not self.index.size:
            return []
        ids, scores = self.bm25.retrieve([tokens], k=min(k, self.index.size), show_progress=False)
        return [
            ScoredChunk(chunk=self.index.chunks[int(i)], score=float(s), scores={"bm25": float(s)})
            for i, s in zip(ids[0], scores[0], strict=True)
            if s > 0
        ]


@RETRIEVERS.register("bm25", BM25Params)
def _bm25(params: BM25Params, index: Index, **_) -> BM25Retriever:
    return BM25Retriever(params, index)
