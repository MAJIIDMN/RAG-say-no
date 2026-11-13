from ragkit.core.registry import RETRIEVERS
from ragkit.core.schemas import Chunk, ScoredChunk
from ragkit.retrieval.hybrid import rrf_fuse


def make(spec, tiny_index):
    index, emb = tiny_index
    return RETRIEVERS.create(spec, index=index, embedder=emb)


def test_bm25_finds_exact_api_name(tiny_index):
    hits = make({"impl": "bm25"}, tiny_index).search("How do I use BackgroundTasks?", k=2)
    assert hits[0].chunk.doc_id == "bg"
    assert "bm25" in hits[0].scores


def test_bm25_returns_nothing_for_stopword_only_query(tiny_index):
    assert make({"impl": "bm25", "stopwords": ["the"]}, tiny_index).search("the", k=3) == []


def test_dense_ranks_by_cosine(tiny_index):
    hits = make({"impl": "dense"}, tiny_index).search("CORSMiddleware origins browser", k=4)
    assert hits[0].chunk.doc_id == "cors"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_rrf_prefers_items_ranked_by_both_lists():
    mk = lambda i: ScoredChunk(chunk=Chunk(chunk_id=i, doc_id=i, text=i), score=1.0)  # noqa: E731
    fused = rrf_fuse([[mk("a"), mk("b")], [mk("b"), mk("c")]], k=60, top_k=3)
    assert [f.chunk.chunk_id for f in fused][0] == "b"
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-12


def test_hybrid_keeps_scores_from_each_retriever(tiny_index):
    hits = make({"impl": "hybrid"}, tiny_index).search("Depends dependencies", k=3)
    assert hits[0].chunk.doc_id == "deps"
    assert {"bm25", "dense", "rrf"} <= set(hits[0].scores)
