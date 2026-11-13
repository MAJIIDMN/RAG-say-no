"""Offline end to end: ingest the toy corpus, run the toy eval set, read the run directory."""

import itertools
import json
from pathlib import Path

import pytest

from ragkit.config import RAGConfig
from ragkit.eval.dataset import check_eval_set, load_eval_set
from ragkit.eval.runner import run_eval
from ragkit.index.store import build_index
from ragkit.ingest.loaders import load_documents
from ragkit.pipeline import RAGPipeline

ROOT = Path(__file__).parents[2]


@pytest.fixture
def cfg(tmp_path):
    return RAGConfig.model_validate(
        {
            "name": "toy-e2e",
            "corpus": {"path": str(ROOT / "data/toy")},
            "retriever": {"impl": "hybrid"},
            "reranker": {"impl": "lexical"},
            "verifier": {"impl": "quote"},
            "abstention": {"min_score": 0.0},
            "artifacts_dir": str(tmp_path / "artifacts"),
        }
    )


def test_toy_eval_set_is_valid_against_the_toy_corpus(cfg):
    items = load_eval_set(ROOT / "evals/datasets/toy.jsonl")
    assert check_eval_set(items, load_documents(cfg.corpus)) == []


def test_eval_run_writes_config_predictions_metrics_and_curve(cfg, tmp_path):
    build_index(cfg)
    items = load_eval_set(ROOT / "evals/datasets/toy.jsonl", split="dev")
    out = run_eval(cfg, items, runs_dir=tmp_path / "runs", split="dev", workers=2)

    assert {p.name for p in out.iterdir()} == {
        "config.yaml",
        "predictions.jsonl",
        "metrics.json",
        "curve.json",
    }
    preds = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert [p["id"] for p in preds] == [i.id for i in items]
    assert all("error" not in p for p in preds)

    m = json.loads((out / "metrics.json").read_text())
    assert m["n"] == len(items) and m["n_errors"] == 0
    for key in (
        "recall@5",
        "mrr",
        "context_recall",
        "abstain_f1",
        "over_refusal_rate",
        "unanswerable_answered_rate",
        "crag_score",
        "citation_precision",
        "aurc",
    ):
        assert key in m, key
    assert m["meta"]["split"] == "dev" and m["meta"]["judge"] == "lexical"


def test_runner_survives_a_failing_item(cfg, tmp_path):
    build_index(cfg)
    pipe = RAGPipeline(cfg)
    items = load_eval_set(ROOT / "evals/datasets/toy.jsonl", split="dev")[:2]
    pipe.answer = lambda q: (_ for _ in ()).throw(RuntimeError("boom"))
    out = run_eval(cfg, items, runs_dir=tmp_path / "runs", pipeline=pipe, workers=1)
    m = json.loads((out / "metrics.json").read_text())
    assert m["n_errors"] == 2


def test_faithfulness_is_reported_when_a_claim_checker_is_given(cfg, tmp_path):
    from ragkit.core.registry import VERIFIERS
    from ragkit.llm.fake import FakeLLMClient
    from ragkit.verification.verifiers import Entailment

    build_index(cfg)
    verdicts = itertools.cycle([True, False])
    llm = FakeLLMClient(lambda s, u, schema: Entailment(supported=next(verdicts), reason=""))
    checker = VERIFIERS.create({"impl": "llm"}, llm=llm)
    items = [i for i in load_eval_set(ROOT / "evals/datasets/toy.jsonl") if i.id in {"d01", "d02"}]
    out = run_eval(cfg, items, runs_dir=tmp_path / "runs", faithfulness=checker, workers=1)
    m = json.loads((out / "metrics.json").read_text())
    assert m["faithfulness"] == 0.5


def test_a_failing_judge_marks_the_item_but_keeps_the_run(cfg, tmp_path):
    class BrokenJudge:
        def judge(self, question, reference, answer):
            raise TimeoutError("judge timed out")

    build_index(cfg)
    items = load_eval_set(ROOT / "evals/datasets/toy.jsonl", split="dev")
    out = run_eval(cfg, items, runs_dir=tmp_path / "runs", judge=BrokenJudge(), workers=2)
    preds = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    judged = [p for p in preds if p["answerable"] and p["answered"]]
    assert judged and all("TimeoutError" in p["judge_error"] for p in judged)
    m = json.loads((out / "metrics.json").read_text())
    assert m["n_errors"] == 0 and m["n_judge_errors"] == len(judged)
