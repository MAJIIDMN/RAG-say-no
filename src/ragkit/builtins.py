"""Importing this module registers every built-in stage implementation."""

from ragkit.eval import judges  # noqa: F401
from ragkit.generation import generators  # noqa: F401
from ragkit.index import embedders  # noqa: F401
from ragkit.ingest import chunkers  # noqa: F401
from ragkit.llm import anthropic_client, fake, openai_compatible  # noqa: F401
from ragkit.rerank import rerankers  # noqa: F401
from ragkit.retrieval import bm25, dense, hybrid  # noqa: F401
from ragkit.verification import verifiers  # noqa: F401
