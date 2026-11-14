"""Persisted index: chunks.jsonl + vectors.npy + manifest.json, keyed by `RAGConfig.index_key()`.
A cached index is rebuilt when the corpus content digest changes.

Dense search is exact brute force over a numpy matrix.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ragkit.config import RAGConfig
from ragkit.core.registry import CHUNKERS, EMBEDDERS, StageSpec
from ragkit.core.schemas import Chunk
from ragkit.core.stages import Embedder
from ragkit.ingest.chunkers import count_tokens
from ragkit.ingest.loaders import corpus_fingerprint, load_documents


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray
    path: Path

    @property
    def size(self) -> int:
        return len(self.chunks)

    def by_id(self) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}


@lru_cache(maxsize=4)
def _cached_embedder(spec_json: str) -> Embedder:
    return EMBEDDERS.create(StageSpec.model_validate_json(spec_json))


def get_embedder(spec: StageSpec) -> Embedder:
    """Embedding models are large; reuse one instance per spec within a process."""
    return _cached_embedder(spec.model_dump_json())


def build_index(cfg: RAGConfig, force: bool = False) -> Index:
    out = cfg.index_dir()
    docs = load_documents(cfg.corpus)
    fingerprint = corpus_fingerprint(docs)
    if not force and read_manifest(cfg).get("corpus_fingerprint") == fingerprint:
        return load_index(cfg)
    t0 = time.perf_counter()
    chunker = CHUNKERS.create(cfg.chunker)
    chunks = [c for d in docs for c in chunker.chunk(d)]
    vectors = get_embedder(cfg.embedder).embed_documents([c.indexed_text for c in chunks])
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in chunks))
    np.save(out / "vectors.npy", vectors)
    tokens = [count_tokens(c.text) for c in chunks]
    manifest = {
        "index_key": cfg.index_key(),
        "corpus": cfg.corpus.model_dump(),
        "corpus_fingerprint": fingerprint,
        "chunker": cfg.chunker.model_dump(),
        "embedder": cfg.embedder.model_dump(),
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "chunk_tokens_mean": round(float(np.mean(tokens)), 1),
        "chunk_tokens_p95": round(float(np.percentile(tokens, 95)), 1),
        "chunk_tokens_max": int(max(tokens)),
        "build_seconds": round(time.perf_counter() - t0, 2),
    }
    # Manifest last: its presence marks a complete index.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return Index(chunks=chunks, vectors=vectors, path=out)


def read_manifest(cfg: RAGConfig) -> dict:
    """The index manifest, or {} when no index is built."""
    path = cfg.index_dir() / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_index(cfg: RAGConfig) -> Index:
    out = cfg.index_dir()
    if not (out / "manifest.json").exists():
        raise FileNotFoundError(f"No index at {out}. Run `ragkit ingest -c <config>` first.")
    chunks = [Chunk.model_validate_json(line) for line in (out / "chunks.jsonl").open()]
    return Index(chunks=chunks, vectors=np.load(out / "vectors.npy"), path=out)
