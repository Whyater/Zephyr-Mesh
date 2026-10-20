from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def make_figures(output: Path, rows: list[dict], summary: list[dict]) -> None:
    """Create deterministic Horizon-style figures from simulated dropout data."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools.horizon_mpl import apply_style

    palette = apply_style("light")
    series_colors = [
        palette["accent"], palette["ok"], palette["keepOut"], palette["target"], palette["textSecondary"],
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    cells = sorted({(row["n"], row["k"]) for row in rows if row["mode"] == "silent"})
    for index, (n, k) in enumerate(cells):
        color = series_colors[index % len(series_colors)]
        for metric, marker, label in (
            ("detection_time_s", "o", "detection"),
            ("recoverage_time_s", "s", "re-coverage"),
        ):
            points = []
            for item in summary:
                if item["mode"] != "silent" or item["n"] != n or item["k"] != k:
                    continue
                interval = item[metric]
                if interval:
                    points.append(
                        (
                            item["link_loss"],
                            interval["median"],
                            interval["median"] - interval["lo"],
                            interval["hi"] - interval["median"],
                        )
                    )
            points.sort()
            if not points:
                continue
            x_values, y_values, lows, highs = zip(*points)
            low_error = np.maximum(np.asarray(lows, dtype=float), 0.0)
            high_error = np.maximum(np.asarray(highs, dtype=float), 0.0)
            axis.errorbar(
                x_values,
                y_values,
                yerr=[low_error, high_error],
                marker=marker,
                color=color,
                capsize=3,
                linestyle="-",
                label=f"N={n}, k={k} {label}",
            )
        hand = []
        for loss in sorted(
            {
                row["link_loss"]
                for row in rows
                if row["mode"] == "silent" and row["n"] == n and row["k"] == k
            }
        ):
            values = [
                row["hand_detection_registered_s"]
                for row in rows
                if row["mode"] == "silent"
                and row["n"] == n
                and row["k"] == k
                and row["link_loss"] == loss
                and row["hand_detection_registered_s"] is not None
            ]
            if values:
                hand.append((loss, values[0]))
        if hand:
            x_values, y_values = zip(*hand)
            axis.plot(
                x_values,
                y_values,
                marker="x",
                linestyle="",
                color=color,
                label=f"N={n}, k={k} registered hand",
            )
    axis.set_xlabel("neighbor link loss probability")
    axis.set_ylabel("time after drop (s)")
    axis.set_title("Simulated silent mode dropout detection and re-coverage times")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "dropout_times.png", dpi=150, metadata={"Software": None})
    plt.close(fig)

    requested = [
        row
        for row in rows
        if row["n"] == 4
        and row["k"] == 1
        and row["mode"] == "silent"
        and abs(row["link_loss"] - 0.1) < 1e-12
    ]
    if not requested:
        raise ValueError("requested timeline cell N=4, k=1, silent, loss=0.1 is missing")
    representative = min(requested, key=lambda row: row["seed"])
    timeline = representative.get("timeline", [])
    if not timeline:
        raise ValueError("requested timeline cell has no timeline data")
    times, gaps = zip(*timeline)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(times, gaps, color=palette["accent"], label="survivor_gap_deg")
    axis.axvline(
        10.0,
        color=palette["keepOut"],
        linewidth=1.2,
        linestyle="--",
        label="drop",
    )
    detection = representative.get("detection_time_s")
    recovery = representative.get("recoverage_time_s")
    if detection is not None:
        axis.axvline(
            10.0 + detection,
            color=palette["lineStrong"],
            linewidth=1.0,
            linestyle=":",
            label="detection",
        )
    if recovery is not None:
        axis.axvline(
            10.0 + recovery,
            color=palette["ok"],
            linewidth=1.0,
            label="re-coverage",
        )
    axis.axhline(
        120.0,
        color=palette["lineStrong"],
        linewidth=0.8,
        label="120 degree reference",
    )
    axis.set_xlim(left=5.0)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("survivor_gap_deg (degrees)")
    axis.set_title(f"Simulated silent mode timeline, N=4, k=1, loss=0.1, seed={representative['seed']}")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "dropout_timeline.png", dpi=150, metadata={"Software": None})
    plt.close(fig)
