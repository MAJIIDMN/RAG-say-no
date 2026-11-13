"""Pick the score-gate threshold on the calib split only; test is for reporting only."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from ragkit.config import RAGConfig, deep_merge
from ragkit.core.stages import Embedder, Judge
from ragkit.eval import metrics as M
from ragkit.eval.dataset import EvalItem
from ragkit.eval.runner import run_eval
from ragkit.index.store import Index
from ragkit.pipeline import RAGPipeline


@dataclass
class TuneResult:
    threshold: float
    point: dict[str, float]
    run_dir: Path


def tune_threshold(
    cfg: RAGConfig,
    items: list[EvalItem],
    runs_dir: str | Path = "runs",
    max_over_refusal: float | None = None,
    judge: Judge | None = None,
    workers: int = 4,
    index: Index | None = None,
    embedder: Embedder | None = None,
) -> TuneResult:
    wrong = sorted({i.split for i in items} - {"calib"})
    if wrong:
        raise ValueError(f"Tune on the calib split only; got items from {wrong}")
    gate_off = cfg.model_copy(
        update={"abstention": cfg.abstention.model_copy(update={"min_score": 0.0})}
    )
    pipe = RAGPipeline(gate_off, index=index, embedder=embedder)
    run_dir = run_eval(
        gate_off,
        items,
        runs_dir,
        split="calib",
        pipeline=pipe,
        judge=judge,
        workers=workers,
        tag="tune",
    )
    curve = json.loads((run_dir / "curve.json").read_text())
    point = M.pick_threshold(curve, max_over_refusal)
    return TuneResult(threshold=point["threshold"], point=point, run_dir=run_dir)


def write_tuned_config(
    base: str | Path, threshold: float, out: str | Path, overrides: dict | None = None
) -> Path:
    """A config that extends `base` with the overrides used while tuning, plus the threshold."""
    base, out = Path(base), Path(out)
    name = (yaml.safe_load(base.read_text()) or {}).get("name", base.stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    gate = {"abstention": {"enabled": True, "min_score": round(float(threshold), 4)}}
    doc = {
        "extends": os.path.relpath(base, out.parent),
        "name": f"{name}-tuned",
        **deep_merge(overrides or {}, gate),
    }
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out
