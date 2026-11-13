import json

import pytest

from ragkit.core.schemas import Document
from ragkit.eval.dataset import EvalItem, check_eval_set, describe, load_eval_set


def item(id, split="dev", group="g1", answerable=True, evidence=("Tasks run later",), docs=("bg",)):
    return EvalItem(
        id=id,
        split=split,
        group=group,
        question="q?",
        answerable=answerable,
        evidence=list(evidence) if answerable else [],
        gold_doc_ids=list(docs) if answerable else [],
    )


DOCS = [Document(doc_id="bg", title="BG", text="Tasks run   later, after the response.")]


def test_load_filters_by_split_and_reports_bad_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(
        "".join(json.dumps(i.model_dump()) + "\n" for i in [item("a"), item("b", "test", "g2")])
    )
    assert [i.id for i in load_eval_set(p, split="test")] == ["b"]
    p.write_text('{"id": "x"}\n')
    with pytest.raises(ValueError, match=":1:"):
        load_eval_set(p)


def test_check_passes_a_clean_set():
    assert check_eval_set([item("a"), item("b", answerable=False)], DOCS) == []


def test_check_flags_duplicates_leakage_and_missing_evidence():
    problems = check_eval_set(
        [
            item("a"),
            item("a"),
            item("c", split="test"),
            item("d", evidence=["not in docs"]),
            item("e", docs=["nope"]),
            item("f", evidence=[]),
        ],
        DOCS,
    )
    text = "\n".join(problems)
    assert "duplicate id 'a'" in text
    assert "group 'g1'" in text and "dev" in text and "test" in text
    assert "d: evidence not found" in text
    assert "e: unknown gold doc 'nope'" in text
    assert "f: answerable item has no evidence" in text


def test_describe_counts_splits_and_unanswerable_share():
    out = describe([item("a"), item("b", answerable=False)])
    assert out["n"] == 2 and out["unanswerable"] == 1 and out["by_split"] == {"dev": 2}
