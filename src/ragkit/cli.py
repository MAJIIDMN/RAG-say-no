"""`ragkit` command line. Every command takes a config (-c) and dotted overrides (-o key=value)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ragkit.config import RAGConfig, load_config, parse_overrides

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
evalset_app = typer.Typer(no_args_is_help=True, help="Eval-set utilities.")
app.add_typer(evalset_app, name="evalset")

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="YAML config.")]
OverrideOpt = Annotated[
    list[str] | None, typer.Option("--override", "-o", help="key.path=value, repeatable.")
]
EvalSetOpt = Annotated[Path, typer.Option("--eval-set", "-e", help="JSONL eval set.")]
RunsOpt = Annotated[Path, typer.Option("--runs-dir", help="Where run directories go.")]


def _cfg(config: Path, override: list[str] | None) -> RAGConfig:
    return load_config(config, parse_overrides(override or []))


def _fail(message: str) -> None:
    typer.secho(message, fg="red", err=True)
    raise typer.Exit(1)


def _require_index(cfg: RAGConfig) -> None:
    if not (cfg.index_dir() / "manifest.json").exists():
        _fail(f"No index at {cfg.index_dir()}. Run `ragkit ingest -c <config>` first.")


def _pipeline(cfg: RAGConfig):
    from ragkit.pipeline import RAGPipeline

    _require_index(cfg)
    return RAGPipeline(cfg)


@app.command()
def ingest(
    config: ConfigOpt,
    override: OverrideOpt = None,
    force: Annotated[bool, typer.Option(help="Rebuild even if the index exists.")] = False,
) -> None:
    """Chunk and embed the corpus into artifacts/index/<key>/."""
    from ragkit.index.store import build_index

    cfg = _cfg(config, override)
    build_index(cfg, force=force)
    typer.echo((cfg.index_dir() / "manifest.json").read_text())


@app.command()
def ask(
    question: str,
    config: ConfigOpt,
    override: OverrideOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full response.")] = False,
) -> None:
    """Answer one question with citations, or abstain."""
    resp = _pipeline(_cfg(config, override)).answer(question)
    if as_json:
        typer.echo(resp.model_dump_json(indent=2, exclude={"retrieved"}))
        return
    typer.echo(resp.answer)
    for i, c in enumerate(resp.citations, start=1):
        typer.echo(f'  [{i}] {c.title} {c.url or c.doc_id}\n      "{c.quote}"')
    if not resp.answered:
        typer.echo(f"  (abstained: {resp.abstain_reason})")


@app.command("eval")
def eval_(
    config: ConfigOpt,
    eval_set: EvalSetOpt,
    split: Annotated[str, typer.Option(help="dev, calib, test or all.")] = "test",
    runs_dir: RunsOpt = Path("runs"),
    workers: int = 4,
    tag: str = "",
    override: OverrideOpt = None,
) -> None:
    """Run the eval set and write runs/<stamp>_<name>_<split>/."""
    from ragkit.eval.dataset import load_eval_set
    from ragkit.eval.runner import run_eval

    cfg = _cfg(config, override)
    items = load_eval_set(eval_set, split)
    if not items:
        _fail(f"No items with split={split} in {eval_set}")
    out = run_eval(cfg, items, runs_dir, split, pipeline=_pipeline(cfg), workers=workers, tag=tag)
    metrics = json.loads((out / "metrics.json").read_text())
    metrics.pop("meta")
    typer.echo(json.dumps(metrics, indent=2))
    typer.echo(f"Run: {out}")


@app.command()
def tune(
    config: ConfigOpt,
    eval_set: EvalSetOpt,
    max_over_refusal: Annotated[float | None, typer.Option(help="Ceiling, e.g. 0.15.")] = None,
    runs_dir: RunsOpt = Path("runs"),
    write: Annotated[Path | None, typer.Option(help="Tuned config path.")] = None,
    workers: int = 4,
    override: OverrideOpt = None,
) -> None:
    """Pick the score-gate threshold on the calib split and write a tuned config."""
    from ragkit.eval.dataset import load_eval_set
    from ragkit.eval.tuning import tune_threshold, write_tuned_config

    cfg = _cfg(config, override)
    items = load_eval_set(eval_set, "calib")
    if not items:
        _fail(f"No calib items in {eval_set}")
    _require_index(cfg)
    result = tune_threshold(cfg, items, runs_dir, max_over_refusal, workers=workers)
    target = write or Path("configs/tuned") / config.name
    out = write_tuned_config(config, result.threshold, target, parse_overrides(override or []))
    typer.echo(json.dumps(result.point, indent=2))
    typer.echo(f"Tuned config: {out}  (curve: {result.run_dir / 'curve.json'})")


@app.command()
def report(
    runs_dir: RunsOpt = Path("runs"),
    split: str = "test",
    prefix: Annotated[str, typer.Option(help="Only configs whose name starts with this.")] = "",
    out: Annotated[Path | None, typer.Option(help="Also write the table here.")] = None,
) -> None:
    """Ablation table from the newest run per config."""
    from ragkit.eval.report import ablation_table, latest_runs

    runs = latest_runs(runs_dir, split, prefix)
    if not runs:
        _fail(f"No untagged runs for split={split} in {runs_dir}")
    table = ablation_table(runs)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table)
    typer.echo(table)


@app.command()
def plot(
    run_dir: Path,
    out: Annotated[Path, typer.Option(help="PNG path.")] = Path(
        "docs/images/coverage_accuracy.png"
    ),
    threshold: Annotated[float | None, typer.Option(help="Mark this threshold.")] = None,
) -> None:
    """Coverage vs accuracy from a run's curve.json (needs the `evals` extra)."""
    from ragkit.eval.report import plot_curve

    if not (run_dir / "curve.json").exists():
        _fail(f"{run_dir} has no curve.json; run eval with the score gate off.")
    typer.echo(plot_curve(run_dir / "curve.json", out, threshold))


@app.command()
def serve(
    config: ConfigOpt,
    override: OverrideOpt = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve POST /ask and GET /health (needs the `api` extra)."""
    import uvicorn

    from ragkit.api import create_app

    uvicorn.run(create_app(_pipeline(_cfg(config, override))), host=host, port=port)


@evalset_app.command("check")
def evalset_check(eval_set: EvalSetOpt, config: ConfigOpt, override: OverrideOpt = None) -> None:
    """Schema, duplicate ids, group leakage across splits, evidence present in the corpus."""
    from ragkit.eval.dataset import check_eval_set, describe, load_eval_set
    from ragkit.ingest.loaders import load_documents

    cfg = _cfg(config, override)
    try:
        items = load_eval_set(eval_set)
    except ValueError as e:
        _fail(str(e))
    problems = check_eval_set(items, load_documents(cfg.corpus))
    typer.echo(json.dumps(describe(items), indent=2))
    for p in problems:
        typer.secho(f"- {p}", fg="yellow")
    typer.echo(f"{len(problems)} problems")
    if problems:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
