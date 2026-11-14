import json

import pytest
import yaml

from ragkit.eval.report import ablation_table, latest_runs, plot_curve


def run(tmp_path, name, stamp, split="test", tag="", **metrics):
    d = tmp_path / f"{stamp}_{name}_{split}"
    d.mkdir()
    meta = {
        "config_name": name,
        "split": split,
        "tag": tag,
        "eval_set": {"n": 10, "unanswerable": 3},
    }
    (d / "metrics.json").write_text(json.dumps({**metrics, "meta": meta}))
    (d / "config.yaml").write_text(yaml.safe_dump({"abstention": {"enabled": True}}))
    return d


def test_latest_runs_keeps_newest_untagged_run_per_config(tmp_path):
    run(tmp_path, "E001", "20260101-000000")
    new = run(tmp_path, "E001", "20260102-000000")
    run(tmp_path, "E002", "20260103-000000", tag="debug")
    run(tmp_path, "E003", "20260103-000000", split="calib")
    assert latest_runs(tmp_path, "test") == [new]


def test_ablation_table_formats_metrics_and_marks_missing_ones(tmp_path):
    d = run(
        tmp_path, "E001", "20260101-000000", mrr=0.5, latency_p95_ms=1500.0, crag_score=float("nan")
    )
    table = ablation_table([d])
    row = table.splitlines()[2]
    assert row.startswith("| E001 |") and "0.500" in row and "1.5s" in row
    assert "–" in row
    assert "10 questions, 3 unanswerable" in table


def test_plot_curve_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    curve = tmp_path / "curve.json"
    curve.write_text(
        json.dumps(
            [
                {"threshold": 0.0, "coverage": 1.0, "selective_accuracy": 0.6},
                {"threshold": 0.5, "coverage": 0.5, "selective_accuracy": 0.9},
            ]
        )
    )
    out = plot_curve(curve, tmp_path / "plot.png", chosen=0.5)
    assert out.stat().st_size > 0


def test_ablation_table_shows_correctness_coverage_and_abstention_precision_recall(tmp_path):
    header = ablation_table([run(tmp_path, "E001", "20260101-000000")]).splitlines()[0]
    for title in ("Correctness", "Coverage", "Abstain precision", "Abstain recall"):
        assert f"| {title} |" in header
