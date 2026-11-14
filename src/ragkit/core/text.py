"""Regex tokenizer shared by chunking, BM25, rerankers and the extractive generator."""

from __future__ import annotations

import re

TOKEN = re.compile(r"\w+|[^\w\s]")


def words(text: str) -> list[str]:
    """Lowercased alphanumeric tokens; punctuation tokens are dropped."""
    return [t for t in TOKEN.findall(text.lower()) if t.isalnum()]
