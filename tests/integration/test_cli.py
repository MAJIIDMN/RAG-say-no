"""The CLI on the offline toy config: ingest, ask, eval, tune, report, evalset check."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ragkit.cli import app

ROOT = Path(__file__).parents[2]
runner = CliRunner()


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    art = ["-o", f"artifacts_dir={tmp_path / 'artifacts'}"]

    def run(*args, overrides=True):
        result = runner.invoke(app, [*args, *(art if overrides else [])])
        return result

    return run


def test_ingest_then_ask_returns_a_cited_answer(cli):
    assert cli("ingest", "-c", "configs/toy.yaml").exit_code == 0
    out = cli(
        "ask",
        "How many times is a failed job retried by default?",
        "-c",
        "configs/toy.yaml",
        "--json",
    )
    assert out.exit_code == 0, out.output
    resp = json.loads(out.output)
    assert resp["answered"] and "3 times" in resp["answer"]
    assert resp["citations"][0]["doc_id"] == "background-jobs"


def test_ask_without_an_index_explains_how_to_build_it(cli):
    out = cli("ask", "anything", "-c", "configs/toy.yaml")
    assert out.exit_code == 1 and "ragkit ingest" in out.output


def test_eval_tune_and_report(cli, tmp_path):
    runs = str(tmp_path / "runs")
    cli("ingest", "-c", "configs/toy.yaml")
    out = cli(
        "eval",
        "-c",
        "configs/toy.yaml",
        "-e",
        "evals/datasets/toy.jsonl",
        "--split",
        "test",
        "--runs-dir",
        runs,
    )
    assert out.exit_code == 0, out.output
    assert "crag_score" in out.output

    tuned = tmp_path / "tuned.yaml"
    out = cli(
        "tune",
        "-c",
        "configs/toy.yaml",
        "-e",
        "evals/datasets/toy.jsonl",
        "--runs-dir",
        runs,
        "--write",
        str(tuned),
    )
    assert out.exit_code == 0, out.output
    assert yaml.safe_load(tuned.read_text())["abstention"]["enabled"] is True

    out = cli("report", "--runs-dir", runs, "--split", "test", overrides=False)
    assert out.exit_code == 0 and "| toy |" in out.output


def test_evalset_check_passes_toy_and_fails_a_broken_set(cli, tmp_path):
    ok = cli(
        "evalset",
        "check",
        "-e",
        "evals/datasets/toy.jsonl",
        "-c",
        "configs/toy.yaml",
        overrides=False,
    )
    assert ok.exit_code == 0 and "0 problems" in ok.output
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "id": "x",
                "question": "q",
                "answerable": True,
                "split": "dev",
                "group": "g",
                "evidence": ["not in the corpus"],
            }
        )
        + "\n"
    )
    out = cli("evalset", "check", "-e", str(bad), "-c", "configs/toy.yaml", overrides=False)
    assert out.exit_code == 1 and "evidence not found" in out.output
