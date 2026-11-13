"""Run one config over an eval set and write a self-contained run directory.

runs/<timestamp>_<config>_<split>[_<tag>]/
    config.yaml        resolved config
    predictions.jsonl  one record per item
    metrics.json       aggregate metrics plus run metadata
    curve.json         coverage/accuracy per score threshold (only when the score gate is off)
"""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ragkit.config import LLMSettings, RAGConfig, llm_settings
from ragkit.core.registry import JUDGES, VERIFIERS
from ragkit.core.schemas import Chunk
from ragkit.core.stages import Judge, Verifier
from ragkit.eval import metrics as M
from ragkit.eval.dataset import EvalItem, describe
from ragkit.generation.prompts import PROMPT_VERSION
from ragkit.pipeline import LazyLLM, RAGPipeline


def cost_usd(usage: dict[str, int], settings: LLMSettings) -> float:
    return (
        usage.get("input_tokens", 0) * settings.input_price_per_mtok
        + usage.get("output_tokens", 0) * settings.output_price_per_mtok
    ) / 1e6


def evaluate_item(
    item: EvalItem,
    pipe: RAGPipeline,
    judge: Judge,
    faithfulness: Verifier | None = None,
    chunks: dict[str, Chunk] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "id": item.id,
        "split": item.split,
        "category": item.category,
        "question": item.question,
        "answerable": item.answerable,
        "reference_answer": item.reference_answer,
    }
    try:
        resp = pipe.answer(item.question)
    except Exception as e:  # noqa: BLE001 - one bad item must not end a long run
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    chunks = chunks if chunks is not None else pipe.index.by_id()
    candidates = [chunks[c] for c in resp.diagnostics.get("candidate_ids", []) if c in chunks]
    context = [sc.chunk for sc in resp.retrieved]
    rec.update(
        answered=resp.answered,
        answer=resp.answer,
        abstain_reason=resp.abstain_reason,
        confidence=resp.confidence,
        claims=[c.model_dump() for c in resp.claims],
        citations=[c.model_dump() for c in resp.citations],
        context_ids=[c.chunk_id for c in context],
        latency_ms=resp.timings_ms.get("total"),
        timings_ms=resp.timings_ms,
        usage=resp.usage,
        diagnostics={k: v for k, v in resp.diagnostics.items() if k != "candidate_ids"},
    )
    if item.answerable:
        rec["retrieval"] = M.retrieval_metrics(
            [(c.text, c.doc_id) for c in candidates], item.evidence, gold_doc_ids=item.gold_doc_ids
        )
        ctx = M.retrieval_metrics(
            [(c.text, c.doc_id) for c in context],
            item.evidence,
            ks=(max(len(context), 1),),
            gold_doc_ids=item.gold_doc_ids,
        )
        rec["context_recall"] = ctx[f"recall@{max(len(context), 1)}"]

    rec["correct"], judge_usage = False, {}
    try:
        if resp.answered and item.answerable:
            verdict = judge.judge(item.question, item.reference_answer, resp.answer)
            rec.update(correct=verdict.correct, judge_reason=verdict.reason)
            judge_usage = dict(verdict.usage)
        if resp.answered and faithfulness is not None:
            check = faithfulness.verify(resp.claims, resp.retrieved)
            rec["faithful"] = check.verdicts
            for k, v in check.usage.items():
                judge_usage[k] = judge_usage.get(k, 0) + v
    except Exception as e:  # noqa: BLE001 - keep the answer; drop the item from judged metrics
        rec["judge_error"] = f"{type(e).__name__}: {e}"
    rec["judge_usage"] = judge_usage
    return rec


def _scored(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records whose correctness is known: a judge failure leaves it unknown."""
    return [r for r in records if "error" not in r and "judge_error" not in r]


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if "error" not in r]
    out: dict[str, Any] = {"n": len(records), "n_errors": len(records) - len(ok)}
    if not ok:
        return out
    answerable = [r for r in ok if r["answerable"]]
    for name in answerable[0].get("retrieval", {}) if answerable else []:
        out[name] = M.mean([r["retrieval"][name] for r in answerable])
    out["context_recall"] = M.mean([r["context_recall"] for r in answerable])

    gold = [r["answerable"] for r in ok]
    answered = [r["answered"] for r in ok]
    out.update(M.abstention_metrics([not g for g in gold], [not a for a in answered]))
    scored = _scored(ok)
    out["n_judge_errors"] = len(ok) - len(scored)
    out["crag_score"] = M.mean(
        M.crag_scores(
            [r["answerable"] for r in scored],
            [r["answered"] for r in scored],
            [r["correct"] for r in scored],
        )
    )
    judged = [r for r in scored if r["answerable"] and r["answered"]]
    out["accuracy_when_answered"] = M.mean([float(r["correct"]) for r in judged])

    faithful = [v for r in ok for v in r.get("faithful", [])]
    if faithful:
        out["faithfulness"] = M.mean([float(v) for v in faithful])
    cites = [c["verified"] for r in ok if r["answered"] for c in r["citations"]]
    out["citation_precision"] = M.mean([float(v) for v in cites])
    reasons: dict[str, int] = {}
    for r in ok:
        if r["abstain_reason"]:
            reasons[r["abstain_reason"]] = reasons.get(r["abstain_reason"], 0) + 1
    out["abstain_reasons"] = reasons

    latency = [r["latency_ms"] for r in ok if r.get("latency_ms") is not None]
    out["latency_p50_ms"], out["latency_p95_ms"] = (
        M.percentile(latency, 50),
        M.percentile(latency, 95),
    )
    out["cost_per_query_usd"] = M.mean([r["cost_usd"] for r in ok])
    out["judge_cost_usd_total"] = sum(r.get("judge_cost_usd", 0.0) for r in ok)
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_eval(
    cfg: RAGConfig,
    items: list[EvalItem],
    runs_dir: str | Path = "runs",
    split: str = "test",
    pipeline: RAGPipeline | None = None,
    judge: Judge | None = None,
    workers: int = 4,
    tag: str = "",
    faithfulness: Verifier | None = None,
) -> Path:
    """`faithfulness` checks each answered claim against its cited passages. By default it is the
    judge LLM when the judge is `llm`, and skipped otherwise."""
    pipe = pipeline or RAGPipeline(cfg)
    judge_llm = LazyLLM(role="judge")
    judge = judge or JUDGES.create(cfg.judge, llm=judge_llm)
    if faithfulness is None and cfg.judge.impl == "llm":
        faithfulness = VERIFIERS.create({"impl": "llm"}, llm=judge_llm)
    gen_prices, judge_prices = llm_settings("generator"), llm_settings("judge")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(runs_dir) / f"{stamp}_{cfg.name}_{split}{'_' + tag if tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))

    chunks = pipe.index.by_id()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(
            pool.map(lambda it: evaluate_item(it, pipe, judge, faithfulness, chunks), items)
        )
    for r in records:
        if "error" not in r:
            r["cost_usd"] = cost_usd(r["usage"], gen_prices)
            r["judge_cost_usd"] = cost_usd(r["judge_usage"], judge_prices)
    with (out / "predictions.jsonl").open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)

    summary = aggregate(records)
    ok = [r for r in records if "error" not in r]
    if (not cfg.abstention.enabled or cfg.abstention.min_score <= 0) and ok:
        (out / "curve.json").write_text(json.dumps(M.selective_curve(_scored(ok)), indent=2))
        answered = [r for r in _scored(ok) if r["answered"]]
        summary["aurc"] = M.aurc(
            [r["confidence"] for r in answered], [not r["correct"] for r in answered]
        )
    summary["meta"] = {
        "config_name": cfg.name,
        "split": split,
        "tag": tag,
        "eval_set": describe(items),
        "judge": cfg.judge.impl,
        "generator": cfg.generator.impl,
        "llm_model": gen_prices.model or None,
        "prompt_version": PROMPT_VERSION,
        "index_key": cfg.index_key(),
        "git_commit": _git_commit(),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out
