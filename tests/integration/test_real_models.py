"""Real embedding and reranker models. Opt-in: RAGKIT_SLOW=1 and `uv sync --extra ml`."""

import os

import pytest

from ragkit.core.registry import EMBEDDERS, RERANKERS
from ragkit.core.schemas import Chunk, ScoredChunk

pytestmark = pytest.mark.skipif(os.environ.get("RAGKIT_SLOW") != "1", reason="set RAGKIT_SLOW=1")


def test_default_embedder_ranks_relevant_passage_first():
    emb = EMBEDDERS.create({"impl": "sentence_transformers"})
    docs = emb.embed_documents(["Run code after returning a response.", "Configure CORS origins."])
    q = emb.embed_query("How do I run a task after the response is sent?")
    assert docs.shape[0] == 2 and (docs @ q)[0] > (docs @ q)[1]


def test_default_cross_encoder_scores_are_probabilities():
    rr = RERANKERS.create({"impl": "cross_encoder"})
    cands = [
        ScoredChunk(chunk=Chunk(chunk_id=str(i), doc_id=str(i), text=t), score=0.0)
        for i, t in enumerate(["Paris is the capital of France.", "FastAPI uses Starlette."])
    ]
    out = rr.rerank("What does FastAPI build on?", cands, top_n=2)
    assert out[0].chunk.chunk_id == "1"
    assert all(0.0 <= c.score <= 1.0 for c in out)
