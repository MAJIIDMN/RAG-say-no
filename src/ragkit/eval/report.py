"""Turn run directories into the ablation table and the coverage/accuracy plot."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

COLUMNS: list[tuple[str, str | None]] = [
    ("Varian", None),
    ("Recall@20", "recall@20"),
    ("Context recall", "context_recall"),
    ("MRR", "mrr"),
    ("Faithfulness", "faithfulness"),
    ("Citation precision", "citation_precision"),
    ("Accuracy (answered)", "accuracy_when_answered"),
    ("CRAG score", "crag_score"),
    ("Abstention F1", "abstain_f1"),
    ("Over-refusal", "over_refusal_rate"),
    ("Unanswerable answered", "unanswerable_answered_rate"),
    ("p95 latency", "latency_p95_ms"),
    ("$/query", "cost_per_query_usd"),
]


def _fmt(key: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if key == "latency_p95_ms":
        return f"{value / 1000:.1f}s"
    if key == "cost_per_query_usd":
        return f"{value:.4f}"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def latest_runs(runs_dir: str | Path, split: str, prefix: str = "") -> list[Path]:
    """Newest untagged run per config name for `split`, sorted by name."""
    latest: dict[str, tuple[str, Path]] = {}
    for path in Path(runs_dir).glob("*/metrics.json"):
        meta = json.loads(path.read_text()).get("meta", {})
        name = meta.get("config_name", "")
        if meta.get("split") != split or meta.get("tag") or not name.startswith(prefix):
            continue
        stamp = path.parent.name.split("_", 1)[0]
        if name not in latest or stamp > latest[name][0]:
            latest[name] = (stamp, path.parent)
    return [latest[k][1] for k in sorted(latest)]


def ablation_table(run_dirs: list[Path]) -> str:
    lines = [
        "| " + " | ".join(title for title, _ in COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for d in run_dirs:
        m = json.loads((d / "metrics.json").read_text())
        cells = [m["meta"]["config_name"]]
        cells += [_fmt(key, m.get(key)) for _, key in COLUMNS[1:] if key]
        lines.append("| " + " | ".join(cells) + " |")
    if run_dirs:
        meta = json.loads((run_dirs[0] / "metrics.json").read_text())["meta"]
        ev = meta["eval_set"]
        lines.append("")
        lines.append(
            f"Eval set: {ev['n']} pertanyaan, {ev['unanswerable']} tak terjawab "
            f"(split: {meta['split']})."
        )
    return "\n".join(lines) + "\n"


def plot_curve(curve_path: Path, out_png: Path, chosen: float | None = None) -> Path:
    """Needs the `evals` extra (matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = json.loads(Path(curve_path).read_text())
    cov = [c["coverage"] for c in curve]
    acc = [c["selective_accuracy"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    ax.plot(cov, acc, color="#2f6f9f", linewidth=2, marker="o", markersize=3)
    if chosen is not None:
        pt = min(curve, key=lambda c: abs(c["threshold"] - chosen))
        ax.scatter([pt["coverage"]], [pt["selective_accuracy"]], s=60, color="#c0392b", zorder=3)
        ax.annotate(
            f"threshold={pt['threshold']:.2f}",
            (pt["coverage"], pt["selective_accuracy"]),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=9,
        )
    ax.set_xlabel("Coverage (fraksi pertanyaan yang dijawab)")
    ax.set_ylabel("Akurasi pada yang dijawab")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return Path(out_png)
