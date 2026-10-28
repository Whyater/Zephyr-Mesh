"""Figure for EXP-040 (proposal): recovery tracker error vs motion noise q.

Reads a summary.json written by tools/run_s6_q_sweep.py and writes a PNG:

    PYTHONPATH=. ./.venv/bin/python tools/make_s6_q_sweep_figure.py \
        --summary sim/runs/s6-q-sweep/summary.json --output sim/runs/s6-q-sweep/s6_q_sweep.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

from tools import horizon_mpl

Q_TICKS = (0.01, 0.03, 0.1, 0.3, 1, 3, 10)
DELAY_MARKERS = {0.0: "o", 0.2: "s", 0.3: "^"}
TRUTH_NOISES = (0.02, 0.10)  # at 0.05 m told 5 cm and told the truth are the same run
LOSS_TITLES = {"none": "no loss", "burst30": "30 % loss in bursts"}
MOTION_TITLES = {"constant_velocity": "straight line", "constant_turn": "1 m turn"}


def make_figure(summary_path: Path, output: Path) -> Path:
    summary = json.loads(summary_path.read_text())
    design = summary["design"]
    q_grid = [float(q) for q in design["q_grid"]]
    registered_q = float(design["registered_q"])
    n_seeds = design["n_seeds"]
    colors = horizon_mpl.apply_style("light")
    noise_colors = {0.02: colors["accent"], 0.05: colors["lineStrong"], 0.10: colors["keepOut"]}
    cells = {(c["motion"], c["loss"], c["noise_std_m"], c["delay_s"]): c for c in summary["cells"]}
    research_line_m = float(summary["config"]["failure_p95_m"])
    noises = sorted({c["noise_std_m"] for c in summary["cells"]})
    delays = sorted({c["delay_s"] for c in summary["cells"]})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row")
    for row, loss in enumerate(LOSS_TITLES):
        for col, motion in enumerate(MOTION_TITLES):
            ax = axes[row][col]
            for noise in noises:
                for k, delay in enumerate(delays):
                    by_q = cells[(motion, loss, noise, delay)]["by_q"]
                    xs = [q * 1.04 ** (k - 1) for q in q_grid]  # small horizontal offset so bars do not overlap
                    marker = DELAY_MARKERS.get(delay, "D")
                    arms = [by_q[repr(q)]["told_5cm"] for q in q_grid]
                    ys = [a["p95_median_m"] for a in arms]
                    yerr = [[a["p95_median_m"] - a["p95_ci_low_m"] for a in arms],
                            [a["p95_ci_high_m"] - a["p95_median_m"] for a in arms]]
                    ax.errorbar(xs, ys, yerr=yerr, color=noise_colors[noise], marker=marker, ms=4.5, lw=1.3,
                                elinewidth=0.8, capsize=1.5, zorder=3)
                    if noise in TRUTH_NOISES:
                        truth = [by_q[repr(q)]["told_truth"]["p95_median_m"] for q in q_grid]
                        ax.plot(xs, truth, color=noise_colors[noise], marker=marker, ms=3, mfc="none", mew=0.7,
                                lw=0.7, ls=(0, (3, 2)), zorder=2)
            ax.axhline(research_line_m, color=colors["alert"], lw=1, ls="--", zorder=1)
            ax.axvline(registered_q, color=colors["lineStrong"], lw=1.2, alpha=0.5, zorder=1)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(Q_TICKS))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if loss == "burst30":
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_ylim(bottom=0)
            ax.set_title(f"{MOTION_TITLES[motion]}, {LOSS_TITLES[loss]}")
            if col == 0:
                ax.set_ylabel("median p95 error (m)")
            if row == 1:
                ax.set_xlabel("motion noise q (m²/s³)")
    ymax_none = axes[0][0].get_ylim()[1]
    # the bottom-right panel has empty space just below the line, so the label sits there
    axes[1][1].text(1.02 * q_grid[0], research_line_m * 0.93, f"{research_line_m:g} m research line (not a safety limit)",
                    fontsize=8, color=colors["alert"], va="top", ha="left")
    axes[0][0].text(registered_q * 1.06, 0.97 * ymax_none, "registered q", fontsize=8,
                    color=colors["textSecondary"], va="top", ha="left")

    handles = [Line2D([], [], color=noise_colors[n], lw=2, label=f"true noise {n * 100:.0f} cm") for n in noises]
    handles += [Line2D([], [], color=colors["textSecondary"], marker=DELAY_MARKERS.get(d, "D"), ms=5, lw=0,
                       label=f"delay {d:g} s") for d in delays]
    handles += [Line2D([], [], color=colors["textSecondary"], lw=1.3, label="told 5 cm (solid, bars: 95 % interval)"),
                Line2D([], [], color=colors["textSecondary"], lw=0.8, ls=(0, (3, 2)),
                       label="told the true noise (dashed, 2 and 10 cm only)"),
                Line2D([], [], color=colors["alert"], lw=1, ls="--", label="0.25 m research line"),
                Line2D([], [], color=colors["lineStrong"], lw=1.2, alpha=0.5, label="registered q = 0.1")]
    fig.suptitle("EXP-040 (proposal): recovery tracker error vs motion noise q", fontsize=11)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.012, f"Simulated: synthetic link and Gaussian sensor noise. "
                         f"Median of per-flight p95 error over {n_seeds} seeds.",
             ha="center", va="bottom", fontsize=8.5, color=colors["textSecondary"])
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, metadata={"Software": None})
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(make_figure(args.summary, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
