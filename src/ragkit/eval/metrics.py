"""Pure metric functions over prediction records. No I/O and no models, so all are unit-tested."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ragkit.verification.verifiers import quote_in


def retrieval_metrics(
    ranked: list[tuple[str, str]],
    evidence: list[str],
    ks: tuple[int, ...] = (1, 3, 5, 10, 20),
    gold_doc_ids: list[str] | None = None,
) -> dict[str, float]:
    """`ranked` is (chunk_text, doc_id) in rank order. Relevance = contains an evidence span.

    Without evidence, a chunk from a gold document counts as relevant.
    """
    gold = gold_doc_ids or []

    def hits(text: str, doc_id: str) -> set[int]:
        if evidence:
            return {i for i, e in enumerate(evidence) if quote_in(e, text)}
        return {0} if doc_id in gold else set()

    n_targets = len(evidence) or 1
    covered: list[set[int]] = []
    seen: set[int] = set()
    first_hit = None
    for rank, (text, doc_id) in enumerate(ranked, start=1):
        found = hits(text, doc_id)
        if found and first_hit is None:
            first_hit = rank
        seen |= found
        covered.append(set(seen))
    out = {}
    for k in ks:
        got = covered[min(k, len(covered)) - 1] if covered else set()
        out[f"recall@{k}"] = len(got) / n_targets
        out[f"hit@{k}"] = float(first_hit is not None and first_hit <= k)
    out["mrr"] = 1.0 / first_hit if first_hit else 0.0
    return out


def abstention_metrics(should_abstain: list[bool], did_abstain: list[bool]) -> dict[str, float]:
    """Positive class = "should abstain" (the documents do not answer the question)."""
    pairs = list(zip(should_abstain, did_abstain, strict=True))
    tp = sum(s and d for s, d in pairs)
    fp = sum(d and not s for s, d in pairs)
    fn = sum(s and not d for s, d in pairs)
    n_pos = sum(should_abstain)
    n_neg = len(pairs) - n_pos
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "abstain_precision": p,
        "abstain_recall": r,
        "abstain_f1": 2 * p * r / (p + r) if p + r else 0.0,
        "over_refusal_rate": fp / n_neg if n_neg else 0.0,
        "unanswerable_answered_rate": fn / n_pos if n_pos else 0.0,
        "coverage": sum(not d for d in did_abstain) / len(pairs) if pairs else 0.0,
    }


def crag_scores(
    answerable: list[bool], answered: list[bool], correct: list[bool | None]
) -> list[int]:
    """CRAG-style: right answer or right refusal = 1, abstaining on an answerable item = 0, and any
    wrong answer = -1, so a hallucination costs more than a refusal."""
    out = []
    for gold, ans, ok in zip(answerable, answered, correct, strict=True):
        if not ans:
            out.append(0 if gold else 1)
        else:
            out.append(1 if gold and ok else -1)
    return out


def aurc(confidences: list[float], errors: list[bool]) -> float:
    """Area under the risk-coverage curve: mean error rate over every top-k by confidence."""
    if not confidences:
        return math.nan
    order = np.argsort(-np.asarray(confidences, dtype=float), kind="stable")
    err = np.asarray(errors, dtype=float)[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def selective_curve(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Replay the score gate at every observed confidence.

    Records come from a run with the score gate off and need `confidence`, `answered` (the pipeline
    answered), `answerable` (gold) and `correct` (the answer was right; False when unanswerable).
    """
    n = len(records)
    out = []
    for t in sorted({0.0, *(r["confidence"] for r in records)}):
        answered = [r["answered"] and r["confidence"] >= t for r in records]
        gold = [r["answerable"] for r in records]
        correct = [r["correct"] for r in records]
        n_ans = sum(answered)
        n_right = sum(a and bool(c) for a, c in zip(answered, correct, strict=True))
        ab = abstention_metrics([not g for g in gold], [not a for a in answered])
        out.append(
            {
                "threshold": t,
                "coverage": n_ans / n if n else 0.0,
                "selective_accuracy": n_right / n_ans if n_ans else 1.0,
                "crag_score": mean(crag_scores(gold, answered, correct)),
                "abstain_f1": ab["abstain_f1"],
                "over_refusal_rate": ab["over_refusal_rate"],
                "unanswerable_answered_rate": ab["unanswerable_answered_rate"],
            }
        )
    return out


def pick_threshold(
    curve: list[dict[str, float]], max_over_refusal: float | None = None
) -> dict[str, float]:
    """Best CRAG score, optionally under an over-refusal ceiling. Ties go to the lower threshold."""
    ok = [
        c for c in curve if max_over_refusal is None or c["over_refusal_rate"] <= max_over_refusal
    ]
    return max(ok or curve, key=lambda c: (c["crag_score"], -c["threshold"]))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if len(xs) else math.nan


def percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else math.nan
