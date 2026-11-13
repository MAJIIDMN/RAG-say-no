import pytest

from ragkit.eval import metrics as M


def test_retrieval_metrics_count_evidence_spans_not_chunk_ids():
    ranked = [("unrelated text", "a"), ("retried up to 3 times by default", "b"), ("x", "c")]
    out = M.retrieval_metrics(ranked, ["retried up to 3 times", "starting at 2 seconds"], ks=(1, 3))
    assert out["recall@1"] == 0.0 and out["recall@3"] == 0.5
    assert out["hit@1"] == 0.0 and out["hit@3"] == 1.0
    assert out["mrr"] == 0.5


def test_retrieval_metrics_fall_back_to_gold_docs_without_evidence():
    out = M.retrieval_metrics([("a", "x"), ("b", "gold")], [], ks=(2,), gold_doc_ids=["gold"])
    assert out["recall@2"] == 1.0 and out["mrr"] == 0.5


def test_abstention_metrics_treat_should_abstain_as_positive():
    should = [True, True, False, False]
    did = [True, False, True, False]
    out = M.abstention_metrics(should, did)
    assert out["abstain_precision"] == 0.5 and out["abstain_recall"] == 0.5
    assert out["over_refusal_rate"] == 0.5 and out["unanswerable_answered_rate"] == 0.5
    assert out["coverage"] == 0.5


def test_crag_score_rewards_correct_answers_and_right_refusals():
    # answerable+correct=1, answerable+wrong=-1, answerable+abstain=0,
    # unanswerable+abstain=1, unanswerable+answered=-1
    gold = [True, True, True, False, False]
    answered = [True, True, False, False, True]
    correct = [True, False, None, None, None]
    assert M.crag_scores(gold, answered, correct) == [1, -1, 0, 1, -1]


def test_aurc_is_zero_when_errors_have_the_lowest_confidence():
    assert M.aurc([0.9, 0.8, 0.1], [False, False, True]) == pytest.approx((0 + 0 + 1 / 3) / 3)
    assert M.aurc([0.1, 0.8, 0.9], [False, False, True]) == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)


def recs():
    # confidence, answered with the gate off, gold answerable, correct when answered
    rows = [
        (0.9, True, True, True),
        (0.6, True, True, True),
        (0.4, True, False, False),
        (0.2, True, False, False),
        (0.3, False, False, False),
    ]
    return [dict(confidence=c, answered=a, answerable=g, correct=k) for c, a, g, k in rows]


def test_selective_curve_simulates_the_score_gate_offline():
    curve = {round(p["threshold"], 2): p for p in M.selective_curve(recs())}
    assert curve[0.0]["coverage"] == pytest.approx(4 / 5)
    assert curve[0.0]["unanswerable_answered_rate"] == pytest.approx(2 / 3)
    best = curve[0.6]
    assert best["coverage"] == pytest.approx(2 / 5) and best["selective_accuracy"] == 1.0
    assert best["unanswerable_answered_rate"] == 0.0 and best["over_refusal_rate"] == 0.0


def test_pick_threshold_maximises_crag_and_respects_over_refusal_ceiling():
    curve = M.selective_curve(recs())
    assert M.pick_threshold(curve)["threshold"] == pytest.approx(0.6)
    capped = M.pick_threshold(curve, max_over_refusal=0.0)
    assert capped["over_refusal_rate"] == 0.0
