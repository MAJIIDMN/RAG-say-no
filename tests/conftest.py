import numpy as np
import pytest

from ragkit.core.schemas import Chunk
from ragkit.index.store import Index

TEXTS = {
    "deps": "Use Depends to declare dependencies that run before the path operation.",
    "bg": "BackgroundTasks run a function after the response has been sent.",
    "cors": "CORSMiddleware configures which origins may call the API from a browser.",
    "path": "Path parameters are declared with braces in the route path.",
}


@pytest.fixture
def tiny_index(tmp_path):
    from ragkit.core.registry import EMBEDDERS

    chunks = [
        Chunk(chunk_id=f"{k}::c0000", doc_id=k, text=t, title=k.title(), section=k.title())
        for k, t in TEXTS.items()
    ]
    emb = EMBEDDERS.create({"impl": "hashing", "dim": 256})
    vectors = emb.embed_documents([c.indexed_text for c in chunks]).astype(np.float32)
    return Index(chunks=chunks, vectors=vectors, path=tmp_path), emb
