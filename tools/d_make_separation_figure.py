"""Create the simulated separation-buffer figures from runner outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def make_figures(directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.horizon_mpl import apply_style

    directory = Path(directory)
    palette = apply_style("light")
    data = json.loads((directory / "buffer.json").read_text(encoding="utf-8"))
    envelope = data.get("envelope", [])
    speeds = sorted({float(e["speed_mps"]) for e in envelope})
    delays = sorted({float(e["delay_s"]) for e in envelope})
    colors = {speed: c for speed, c in zip(speeds, (palette["accent"], palette["target"], palette["ok"]))}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, dr in zip(axes, (False, True)):
        for speed in speeds:
            vals = [next((float(e["buffer_m"]) for e in envelope if bool(e["dead_reckoning"]) == dr and float(e["speed_mps"]) == speed and float(e["delay_s"]) == delay), np.nan) for delay in delays]
            ax.plot(delays, vals, marker="o", color=colors[speed], label=f"{speed:g} m/s")
            # This is a hand model reference, plotted as a dashed line.
            hand = (2.0 * speed) * (np.asarray(delays) + 0.05)
            ax.plot(delays, hand, linestyle="--", linewidth=0.9, color=colors[speed], alpha=0.75)
        ax.set_title("Dead reckoning on" if dr else "Dead reckoning off")
        ax.set_xlabel("one-way delay (s)")
        ax.set_xticks(delays)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True)
    axes[0].set_ylabel("required buffer (m)")
    if speeds:
        axes[1].legend(title="speed", loc="best")
    fig.suptitle("Simulated separation buffer versus stale-neighbor delay")
    fig.tight_layout()
    fig.savefig(directory / "separation_buffer.png", dpi=150, metadata={"Software": None})
    plt.close(fig)

    rows = list(csv.DictReader((directory / "encounters.csv").open(encoding="utf-8", newline="")))
    kinds = sorted({row["kind"] for row in rows})
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    kind_colors = {kind: c for kind, c in zip(kinds, (palette["accent"], palette["target"], palette["ok"]))}
    for kind in kinds:
        selected = [r for r in rows if r["kind"] == kind and float(r["speed_mps"]) == 1.0 and float(r["loss_probability"]) == 0.3 and float(r["noise_std_m"]) == 0.05 and r["dead_reckoning"].lower() == "false" and r["pass"] == "baseline"]
        if not selected:
            selected = [r for r in rows if r["kind"] == kind and float(r["speed_mps"]) == 1.0 and r["dead_reckoning"].lower() == "false" and r["pass"] == "baseline"]
        ys = []
        for delay in sorted({float(r["delay_s"]) for r in selected}):
            vals = [float(r["violated"]) for r in selected if float(r["delay_s"]) == delay]
            ys.append((delay, float(np.mean(vals))) if vals else (delay, np.nan))
        if ys:
            ax.plot([x for x, _ in ys], [y for _, y in ys], marker="o", color=kind_colors[kind], label=kind.replace("_", " "))
    ax.axhline(0.05, color=palette["alert"], linestyle="--", linewidth=1.0, label="5% threshold")
    ax.set_xlabel("one-way delay (s)")
    ax.set_ylabel("violation rate")
    ax.set_ylim(bottom=0)
    ax.set_title("Simulated baseline violation rate, speed 1.0 m/s")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(directory / "separation_baseline.png", dpi=150, metadata={"Software": None})
    plt.close(fig)


if __name__ == "__main__":
    import sys
    make_figures(Path(sys.argv[1]))
