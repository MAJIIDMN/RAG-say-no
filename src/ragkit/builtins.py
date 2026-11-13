"""Importing this module registers every built-in stage implementation."""

from ragkit.index import embedders  # noqa: F401
from ragkit.ingest import chunkers  # noqa: F401
from ragkit.rerank import rerankers  # noqa: F401
from ragkit.retrieval import bm25, dense, hybrid  # noqa: F401
