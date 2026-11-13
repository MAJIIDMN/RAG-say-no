from ragkit.core.registry import RERANKERS
from ragkit.core.schemas import Chunk, ScoredChunk


def cands(*texts):
    return [
        ScoredChunk(
            chunk=Chunk(chunk_id=str(i), doc_id=str(i), text=t),
            score=0.1 * i,
            scores={"rrf": 0.1 * i},
        )
        for i, t in enumerate(texts)
    ]


def test_none_keeps_order_and_truncates():
    out = RERANKERS.create({"impl": "none"}).rerank("q", cands("a", "b", "c"), top_n=2)
    assert [c.chunk.chunk_id for c in out] == ["0", "1"]


def test_lexical_scores_query_coverage_in_unit_interval():
    out = RERANKERS.create({"impl": "lexical"}).rerank(
        "background tasks response",
        cands("unrelated text", "background tasks run after the response"),
        top_n=2,
    )
    assert out[0].chunk.chunk_id == "1" and out[0].score == 1.0
    assert 0.0 <= out[1].score < 1.0
    assert "rrf" in out[0].scores and "rerank" in out[0].scores


def test_empty_candidates():
    assert RERANKERS.create({"impl": "lexical"}).rerank("q", [], top_n=5) == []
