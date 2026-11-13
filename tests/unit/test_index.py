import numpy as np
import pytest

from ragkit.config import RAGConfig
from ragkit.core.registry import EMBEDDERS
from ragkit.index.store import build_index, load_index


@pytest.fixture
def cfg(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# Alpha\n\nAlpha talks about routing and path parameters.")
    (corpus / "b.md").write_text("# Beta\n\nBeta covers background tasks after a response.")
    return RAGConfig.model_validate(
        {
            "corpus": {"path": str(corpus)},
            "embedder": {"impl": "hashing", "dim": 128},
            "artifacts_dir": str(tmp_path / "artifacts"),
        }
    )


def test_hashing_embedder_is_normalised_and_deterministic():
    emb = EMBEDDERS.create({"impl": "hashing", "dim": 64})
    vecs = emb.embed_documents(["path parameters", "path parameters", "background tasks"])
    assert vecs.shape == (3, 64) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert np.allclose(vecs[0], vecs[1])
    assert vecs[0] @ emb.embed_query("path parameters") > vecs[2] @ emb.embed_query(
        "path parameters"
    )


def test_build_index_persists_and_reuses(cfg):
    idx = build_index(cfg)
    assert idx.size == 2 and idx.vectors.shape == (2, 128)
    assert (idx.path / "manifest.json").exists()
    mtime = (idx.path / "vectors.npy").stat().st_mtime_ns
    again = build_index(cfg)
    assert (again.path / "vectors.npy").stat().st_mtime_ns == mtime
    assert [c.chunk_id for c in load_index(cfg).chunks] == [c.chunk_id for c in idx.chunks]


def test_load_index_without_build_explains_what_to_run(cfg):
    with pytest.raises(FileNotFoundError, match="ragkit ingest"):
        load_index(cfg)
