import pytest
import yaml

from ragkit.config import RAGConfig, load_config
from ragkit.eval.dataset import EvalItem
from ragkit.eval.tuning import tune_threshold, write_tuned_config


def item(id, q, answerable, split="calib", evidence=()):
    return EvalItem(
        id=id,
        question=q,
        answerable=answerable,
        split=split,
        group="g",
        reference_answer="after the response has been sent" if answerable else "",
        evidence=list(evidence),
    )


ITEMS = [
    item(
        "a",
        "When do BackgroundTasks run after the response?",
        True,
        evidence=["after the response has been sent"],
    ),
    item("b", "Do BackgroundTasks retry failed functions?", False),
]


CFG = RAGConfig.model_validate(
    {
        "reranker": {"impl": "lexical"},
        "verifier": {"impl": "quote"},
        "abstention": {"min_score": 0.9},
    }
)


@pytest.fixture
def deps(tiny_index):
    index, emb = tiny_index
    return {"index": index, "embedder": emb}


def test_tuning_refuses_items_outside_calib(deps, tmp_path):
    with pytest.raises(ValueError, match="calib"):
        tune_threshold(CFG, [*ITEMS, item("c", "q", False, split="test")], tmp_path, **deps)


def test_tuning_runs_with_the_gate_off_and_picks_a_separating_threshold(deps, tmp_path):
    result = tune_threshold(CFG, ITEMS, tmp_path, **deps)
    assert CFG.abstention.min_score == 0.9
    assert (result.run_dir / "curve.json").exists()
    assert result.point["unanswerable_answered_rate"] == 0.0
    assert result.point["over_refusal_rate"] == 0.0
    assert 0.0 < result.threshold <= 1.0


def test_write_tuned_config_extends_the_base(tmp_path):
    base = tmp_path / "configs" / "E002.yaml"
    base.parent.mkdir()
    base.write_text(yaml.safe_dump({"name": "E002", "abstention": {"min_score": 0.0}}))
    out = write_tuned_config(base, 0.42, tmp_path / "configs" / "tuned" / "E002.yaml")
    cfg = load_config(out)
    assert cfg.name == "E002-tuned" and cfg.abstention.min_score == 0.42
