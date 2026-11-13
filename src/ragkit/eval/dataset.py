"""Eval set: JSONL, one item per line (protocol in docs/architecture.md).

Relevance comes from `evidence`, verbatim spans of the source documents, so the set stays valid when
chunking changes. `group` (usually the source page) keeps related questions in one split.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ragkit.core.schemas import Document
from ragkit.verification.verifiers import quote_in

Split = Literal["dev", "calib", "test"]


class EvalItem(BaseModel):
    id: str
    question: str
    answerable: bool
    split: Split
    group: str = Field(description="Questions sharing a group must share a split.")
    category: str = ""
    source: Literal["manual", "synthetic-reviewed"] = "manual"
    reference_answer: str = ""
    gold_doc_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


def load_eval_set(path: str | Path, split: str | None = None) -> list[EvalItem]:
    items = []
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if line.strip():
            try:
                items.append(EvalItem.model_validate_json(line))
            except ValueError as e:
                raise ValueError(f"{path}:{n}: {e}") from e
    return [i for i in items if i.split == split] if split and split != "all" else items


def write_eval_set(items: list[EvalItem], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(i.model_dump()) + "\n" for i in items))


def check_eval_set(items: list[EvalItem], documents: list[Document]) -> list[str]:
    """Every problem found, one line each. An empty list means the set is usable."""
    problems = [f"duplicate id '{k}'" for k, v in Counter(i.id for i in items).items() if v > 1]

    splits: dict[str, set[str]] = defaultdict(set)
    for i in items:
        splits[i.group].add(i.split)
    problems += [
        f"group '{g}' appears in several splits: {', '.join(sorted(s))}"
        for g, s in splits.items()
        if len(s) > 1
    ]

    docs = {d.doc_id: d.text for d in documents}
    for i in items:
        if not i.answerable:
            if i.evidence:
                problems.append(f"{i.id}: unanswerable item has evidence")
            continue
        if not i.evidence:
            problems.append(f"{i.id}: answerable item has no evidence")
        unknown = [d for d in i.gold_doc_ids if d not in docs]
        problems += [f"{i.id}: unknown gold doc '{d}'" for d in unknown]
        known = [docs[d] for d in i.gold_doc_ids if d in docs] or list(docs.values())
        problems += [
            f"{i.id}: evidence not found in its documents: {e!r}"
            for e in i.evidence
            if not any(quote_in(e, text) for text in known)
        ]
    return problems


def describe(items: list[EvalItem]) -> dict:
    return {
        "n": len(items),
        "answerable": sum(i.answerable for i in items),
        "unanswerable": sum(not i.answerable for i in items),
        "by_split": dict(Counter(i.split for i in items)),
        "by_category": dict(Counter(i.category or "-" for i in items)),
    }
