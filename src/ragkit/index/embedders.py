"""Embedders. `hashing` is an offline stand-in for tests and CI, not for real evaluation."""

from __future__ import annotations

import hashlib

import numpy as np
from pydantic import BaseModel, ConfigDict

from ragkit.core.registry import EMBEDDERS
from ragkit.core.text import TOKEN


class HashingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim: int = 384


class HashingEmbedder:
    """Signed feature hashing of lower-cased tokens."""

    def __init__(self, params: HashingParams):
        self.dim = params.dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@EMBEDDERS.register("hashing", HashingParams)
def _hashing(params: HashingParams) -> HashingEmbedder:
    return HashingEmbedder(params)


class SentenceTransformerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    doc_prefix: str = ""
    batch_size: int = 32
    max_seq_length: int | None = 512
    device: str = "cpu"


class SentenceTransformerEmbedder:
    def __init__(self, params: SentenceTransformerParams):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("Needs the `ml` extra: uv sync --extra ml") from e
        self.p = params
        self.model = SentenceTransformer(params.model, device=params.device)
        if params.max_seq_length:
            self.model.max_seq_length = params.max_seq_length

    def _encode(self, texts: list[str], progress: bool) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.p.batch_size,
            normalize_embeddings=True,
            show_progress_bar=progress,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.p.doc_prefix + t for t in texts], progress=len(texts) > 100)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.p.query_prefix + text], progress=False)[0]


@EMBEDDERS.register("sentence_transformers", SentenceTransformerParams)
def _sentence_transformers(params: SentenceTransformerParams) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(params)
