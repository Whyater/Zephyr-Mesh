"""Create the simulated consensus figures from a runner output directory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path.name} contains no simulated rows")
    return rows


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def make_figures(directory: Path) -> tuple[Path, Path]:
    """Write both figures using only data in ``directory`` and return paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.horizon_mpl import apply_style, ramp, sequential_ramp

    directory = Path(directory)
    loss_rows = _read_csv(directory / "loss_delay.csv")
    loss_summary = json.loads((directory / "loss_delay_summary.json").read_text(encoding="utf-8"))
    acting_summary = json.loads((directory / "acting_summary.json").read_text(encoding="utf-8"))
    if not isinstance(loss_summary, list) or not isinstance(acting_summary, list):
        raise ValueError("consensus summaries must be JSON arrays")
    pal = apply_style("light")
    prediction = {
        (str(row["graph"]), float(row["p"]), int(row["delay_steps"]), _as_bool(row["symmetric_loss"])):
        row.get("mean_field_predicted_steps_zero_delay")
        for row in loss_summary
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    graphs = ["ring", "line", "star", "complete", "random_geometric"]
    graph_colors = dict(zip(graphs, ramp([pal["accent"], pal["target"], pal["ok"], pal["keepOut"]], len(graphs))))
    graph_labels = {"ring": "ring of 6", "line": "line of 5", "star": "star of 5", "complete": "complete 4", "random_geometric": "random geometric 6"}
    for graph_index, graph in enumerate(graphs):
        color = graph_colors[graph]
        for symmetric, linestyle, marker in ((True, "-", "o"), (False, "--", "s")):
            selected = [
                row for row in loss_rows
                if row["graph"] == graph
                and int(row["delay_steps"]) == 0
                and _as_bool(row["symmetric_loss"]) == symmetric
            ]
            xs = sorted({float(row["p"]) for row in selected})
            if not xs:
                continue
            ys, lows, highs, predicted = [], [], [], []
            for probability in xs:
                summary = next(row for row in loss_summary if row["graph"] == graph and float(row["p"]) == probability and int(row["delay_steps"]) == 0 and _as_bool(row["symmetric_loss"]) == symmetric)
                ys.append(float(summary["median_steps"]) if summary["median_steps"] is not None else np.nan)
                lows.append(float(summary["median_steps_low"]) if summary["median_steps_low"] is not None else np.nan)
                highs.append(float(summary["median_steps_high"]) if summary["median_steps_high"] is not None else np.nan)
                value = prediction.get((graph, probability, 0, symmetric))
                predicted.append(float(value) if value is not None else np.nan)
            ax.plot(
                xs,
                ys,
                linestyle=linestyle,
                marker=marker,
                color=color,
                label=f"{graph_labels[graph]}, {'symmetric' if symmetric else 'one-way'}",
            )
            ax.fill_between(xs, lows, highs, color=color, alpha=0.12, linewidth=0)
            ax.plot(xs, predicted, linestyle="none", marker=".", color=color, ms=5,
                    label="mean-field prediction (loss only, no delay)" if graph_index == 0 and symmetric else None)
    ax.set_xlabel("Loss probability p")
    ax.set_ylabel("Steps to 1 % agreement (updates)")
    ax.set_title("Consensus convergence under loss (zero delay)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    loss_path = directory / "consensus_loss.png"
    fig.savefig(loss_path, dpi=150, metadata={"Software": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    acting_graph = "complete" if any(row.get("graph") == "complete" for row in acting_summary) else None
    acting_rows = [row for row in acting_summary if row.get("graph") == acting_graph and int(row["delay_steps"]) == 0]
    probs = sorted({float(row["p"]) for row in acting_rows})
    acting_colors = dict(zip(probs, sequential_ramp(["info", "accent", "target", "ok", "keepOut"], len(probs))))
    for probability in sorted({float(row["p"]) for row in acting_rows}):
        selected = sorted(
            (row for row in acting_rows if float(row["p"]) == probability),
            key=lambda row: int(row["updates"]),
        )
        ax.plot(
            [int(row["updates"]) for row in selected],
            [float(row["fraction"]) for row in selected],
            marker="o",
            color=acting_colors[probability],
            label=f"p={probability:g}",
        )
    ax.set_xlabel("Updates before acting (0.1 s each)")
    ax.set_ylabel("Fraction > 0.10 m apart")
    ax.set_title("Disagreement at action time (complete graph)")
    if acting_rows:
        ax.legend()
    fig.tight_layout()
    acting_path = directory / "consensus_acting.png"
    fig.savefig(acting_path, dpi=150, metadata={"Software": None})
    plt.close(fig)
    return loss_path, acting_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="runner output directory")
    args = parser.parse_args(argv)
    make_figures(args.directory)


if __name__ == "__main__":
    main()
