import pytest

from ragkit.eval.runner import aggregate


def record(answerable, answered, correct):
    return {
        "answerable": answerable,
        "answered": answered,
        "correct": correct,
        "retrieval": {},
        "context_recall": 1.0,
        "citations": [],
        "abstain_reason": None if answered else "low_retrieval_score",
        "latency_ms": 10.0,
        "cost_usd": 0.0,
    }


def test_correctness_counts_refusals_on_answerable_questions_as_misses():
    m = aggregate(
        [
            record(True, True, True),
            record(True, True, False),
            record(True, False, None),
            record(False, False, None),
        ]
    )
    assert m["correctness"] == pytest.approx(1 / 3)
    assert m["accuracy_when_answered"] == pytest.approx(1 / 2)
    assert m["coverage"] == pytest.approx(1 / 2)
