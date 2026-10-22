"""Draw the S6 recovery figures from sim/runs/s6-recovery/summary.json.

Usage (from the repo root):

    PYTHONPATH=. ./.venv/bin/python tools/make_s6_figure.py sim/runs/s6-recovery/summary.json \
        --output-dir sim/runs/s6-recovery

Writes two PNGs into the output directory:

    s6_recovery.png            the main figure (noise 5 cm, two motions)
    s6_recovery_all_noise.png  supplementary: every noise level, same encoding

Every plotted number is read from summary.json. The script only converts
seconds to milliseconds, meters to centimeters, and fractions to percent for
labels. Colors come from tools/design/horizon_tokens.json through tools/horizon_mpl.py.
The same JSON always produces the same PNG bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from tools import horizon_mpl  # noqa: E402

SCHEMA = "zephyr-s6-recovery-1"
SLICE = "main"
MAIN_NOISE_M = 0.05
MOTIONS = ("constant_velocity", "constant_turn")
LOSSES = (0.0, 0.10, 0.30, 0.50)
DPI = 200
TOL = 1e-9
# Loss is ordered, so the four colors are a sequential ramp from the Horizon
# accent through keep-out to a dark violet. Red and amber are never used for data.


class FigureError(Exception):
    """Raised for unusable input; main() turns it into a clear exit message."""


# ---------------------------------------------------------------- input


def load_summary(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            doc = json.load(handle)
    except OSError as err:
        raise FigureError(f"cannot read {path}: {err}") from err
    except json.JSONDecodeError as err:
        raise FigureError(f"{path} is not valid JSON: {err}") from err
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        found = doc.get("schema") if isinstance(doc, dict) else type(doc).__name__
        raise FigureError(f"unexpected schema {found!r}; this script reads {SCHEMA!r}")
    for key in ("config", "design", "cells"):
        if key not in doc:
            raise FigureError(f"summary.json is missing the {key!r} field")
    for key in ("failure_p95_m", "target_speed_mps", "turn_radius_m", "observation_period_s",
                "tracker_measurement_std_m"):
        if key not in doc["config"]:
            raise FigureError(f"summary.json config is missing {key!r}")
    if "n_seeds" not in doc["design"]:
        raise FigureError("summary.json design is missing 'n_seeds'")
    return doc


def index_cells(doc: dict) -> dict:
    """Map (motion, noise, loss, delay) to the cell for the main slice."""
    table = {}
    for cell in doc["cells"]:
        if cell.get("slice") != SLICE:
            continue
        key = (
            cell["motion"],
            round(cell["noise_std_m"], 6),
            round(cell["loss_probability"], 6),
            round(cell["delay_s"], 6),
        )
        table[key] = cell
    return table


def series(table: dict, motion: str, noise: float, loss: float, arm: str) -> dict:
    """Delay-sorted values for one line: x in ms and median / low / high p95 in m."""
    rows = sorted(
        (key[3], cell)
        for key, cell in table.items()
        if key[0] == motion and key[1] == round(noise, 6) and key[2] == round(loss, 6)
    )
    if not rows:
        raise FigureError(
            f"no {SLICE!r} cells for motion={motion}, noise={noise}, loss={loss}"
        )
    out = {"x_ms": [], "med": [], "low": [], "high": []}
    for delay_s, cell in rows:
        summary = cell["arms"][arm]
        out["x_ms"].append(delay_s * 1000.0)
        out["med"].append(summary["p95_median_m"])
        out["low"].append(summary["p95_ci_low_m"])
        out["high"].append(summary["p95_ci_high_m"])
    return out


def loss_colors(pal: dict) -> list[str]:
    """One color per loss level: light accent, accent, keep-out, dark violet."""
    mix = horizon_mpl.mix
    stops = [
        mix(pal["accent"], pal["surface"], 0.45),
        pal["accent"],
        pal["keepOut"],
        mix(pal["keepOut"], pal["text"], 0.7),
    ]
    return stops


def noises_present(table: dict) -> list[float]:
    return sorted({key[1] for key in table})


def fmt_g(value: float) -> str:
    return f"{value:g}"


def motion_title(doc: dict, motion: str) -> str:
    speed = fmt_g(doc["config"]["target_speed_mps"])
    if motion == "constant_velocity":
        return f"Straight line, {speed} m/s"
    return f"Turning, {speed} m/s, {fmt_g(doc['config']['turn_radius_m'])} m radius"


# ---------------------------------------------------------------- drawing


def loss_label(loss: float) -> str:
    return f"{fmt_g(loss * 100)} %"


def draw_panel(ax, table, motion, noise, colors, pal, failure_m, label_failure=True,
               xticks=(0, 100, 200, 300)):
    """Draw the eight lines, bands and the failure line for one motion and noise level."""
    top = failure_m
    for loss, color in zip(LOSSES, colors):
        naive = series(table, motion, noise, loss, "naive")
        rec = series(table, motion, noise, loss, "recovery")
        top = max(top, max(naive["high"]), max(rec["high"]))
        # Recovery: solid line with a light band for the 95 % interval.
        ax.fill_between(rec["x_ms"], rec["low"], rec["high"], color=color, alpha=0.16,
                        linewidth=0, zorder=2)
        ax.plot(rec["x_ms"], rec["med"], color=color, linestyle="-", linewidth=1.5,
                marker="o", markersize=3.2, markerfacecolor=color, markeredgewidth=0,
                clip_on=False, zorder=4)
        # Naive: dashed line with thin caps for the 95 % interval.
        lower = [m - lo for m, lo in zip(naive["med"], naive["low"])]
        upper = [hi - m for m, hi in zip(naive["med"], naive["high"])]
        ax.errorbar(naive["x_ms"], naive["med"], yerr=[lower, upper], fmt="none",
                    ecolor=color, elinewidth=0.8, capsize=2, capthick=0.8,
                    clip_on=False, zorder=3)
        ax.plot(naive["x_ms"], naive["med"], color=color, linestyle=(0, (4, 2.2)),
                linewidth=1.3, marker="o", markersize=3.2, markerfacecolor=pal["surface"],
                markeredgecolor=color, markeredgewidth=0.8, clip_on=False, zorder=3)
    ax.axhline(failure_m, color=pal["alert"], linewidth=1.0, zorder=1.5)
    if label_failure:
        ax.annotate(f"failure line\n(p95 {failure_m:.2f} m)", xy=(0.012, failure_m),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8,
                    linespacing=1.15, color=pal["textSecondary"])
    ax.set_xlim(0, 300)
    ax.set_xticks(list(xticks))
    ax.grid(axis="x", visible=False)
    return top


def legend_handles(pal, colors, n_seeds):
    arm_handles = [
        Line2D([0], [0], color=pal["textSecondary"], linestyle=(0, (4, 2.2)), linewidth=1.3,
               marker="o", markersize=3.2, markerfacecolor=pal["surface"],
               markeredgecolor=pal["textSecondary"], markeredgewidth=0.8),
        Line2D([0], [0], color=pal["textSecondary"], linestyle="-", linewidth=1.5,
               marker="o", markersize=3.2, markerfacecolor=pal["textSecondary"],
               markeredgewidth=0),
    ]
    arm_labels = ["Naive (arrival-time stamps)", "Capture-time + predict-forward"]
    loss_handles = [Line2D([0], [0], color=c, linewidth=2.2) for c in colors]
    loss_labels = [loss_label(p) for p in LOSSES]
    interval_handles = [
        Patch(facecolor=pal["textSecondary"], alpha=0.16, linewidth=0),
        Line2D([0], [0], color=pal["textSecondary"], marker="_", markersize=7,
               linestyle="none", markeredgewidth=0.8),
    ]
    interval_labels = ["band: solid lines", "caps: dashed lines"]
    return (
        (arm_handles, arm_labels, "Line style: tracker arm"),
        (loss_handles, loss_labels, "Color: random packet loss"),
        (interval_handles, interval_labels, f"95 % interval, {n_seeds} seeds"),
    )


def add_legends(fig, groups, pal, y, x_positions):
    columns = (1, 2, 1)
    for (handles, labels, title), x, ncol in zip(groups, x_positions, columns):
        leg = fig.legend(handles, labels, title=title, loc="upper left",
                         bbox_to_anchor=(x, y), alignment="left", handlelength=2.6,
                         labelspacing=0.45, borderaxespad=0, handletextpad=0.7,
                         columnspacing=1.6, ncol=ncol)
        leg.get_title().set_color(pal["textSecondary"])
        leg.get_title().set_fontsize(8.5)


def footnote_text(doc):
    hz = 1.0 / doc["config"]["observation_period_s"]
    return (
        f"Simulation: simulated link and sensor, {fmt_g(hz)} Hz measurements, "
        f"noise {fmt_g(MAIN_NOISE_M * 100)} cm, {doc['design']['n_seeds']} seeds per point. "
        "Not flight data."
    )


def make_main_figure(doc, table, pal, colors, path):
    fig = plt.figure(figsize=(9.0, 5.0))
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.775, bottom=0.30,
                          wspace=0.09)
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    failure_m = doc["config"]["failure_p95_m"]
    top = failure_m
    for i, (ax, motion) in enumerate(zip(axes, MOTIONS)):
        top = max(top, draw_panel(ax, table, motion, MAIN_NOISE_M, colors, pal, failure_m,
                                  label_failure=(i == 0)))
        ax.set_title(motion_title(doc, motion), loc="left")
        ax.set_xlabel("One-way delay (ms)")
    axes[0].set_ylim(0, top * 1.08)
    axes[0].set_ylabel("p95 tracking error (m)")
    axes[1].tick_params(labelleft=False)
    fig.text(0.075, 0.955, "Tracking error versus link delay, naive and capture-time trackers",
             fontsize=12, color=pal["text"], ha="left", va="top")
    fig.text(0.075, 0.895, "p95 error of the target estimate, median over seeds, "
             "by packet loss and target motion",
             fontsize=9, color=pal["textSecondary"], ha="left", va="top")
    add_legends(fig, legend_handles(pal, colors, doc["design"]["n_seeds"]), pal,
                y=0.185, x_positions=(0.075, 0.41, 0.70))
    fig.text(0.075, 0.03, footnote_text(doc), fontsize=8, color=pal["textSecondary"],
             ha="left", va="bottom")
    save(fig, path)


def make_all_noise_figure(doc, table, pal, colors, path):
    noises = noises_present(table)
    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(2, len(noises), left=0.075, right=0.985, top=0.84, bottom=0.30,
                          wspace=0.17, hspace=0.32)
    failure_m = doc["config"]["failure_p95_m"]
    axes = [[None] * len(noises) for _ in MOTIONS]
    top = failure_m
    for r, motion in enumerate(MOTIONS):
        for c, noise in enumerate(noises):
            ax = fig.add_subplot(gs[r, c], sharex=axes[0][0] if (r or c) else None,
                                 sharey=axes[0][0] if (r or c) else None)
            axes[r][c] = ax
            top = max(top, draw_panel(ax, table, motion, noise, colors, pal, failure_m,
                                      label_failure=(r == 0 and c == 0),
                                      xticks=(0, 150, 300)))
            if r == 0:
                ax.set_title(f"Sensor noise {fmt_g(round(noise * 100, 3))} cm", loc="left",
                             fontsize=9.5)
            if r == len(MOTIONS) - 1:
                ax.set_xlabel("One-way delay (ms)")
            else:
                ax.tick_params(labelbottom=False)
            if c > 0:
                ax.tick_params(labelleft=False)
    axes[0][0].set_ylim(0, top * 1.08)
    for r, motion in enumerate(MOTIONS):
        axes[r][0].set_ylabel(f"{motion_title(doc, motion)}\np95 error (m)", fontsize=8.5)
    fig.text(0.075, 0.965, "Tracking error across sensor noise levels, same encoding as the "
             "main figure", fontsize=12, color=pal["text"], ha="left", va="top")
    add_legends(fig, legend_handles(pal, colors, doc["design"]["n_seeds"]), pal,
                y=0.185, x_positions=(0.075, 0.31, 0.50))
    hz = 1.0 / doc["config"]["observation_period_s"]
    fig.text(0.075, 0.03, f"Simulation: simulated link and sensor, {fmt_g(hz)} Hz measurements, "
             f"{doc['design']['n_seeds']} seeds per point. Not flight data.\nThe tracker assumes "
             f"{fmt_g(round(doc['config']['tracker_measurement_std_m'] * 100, 3))} cm noise in every panel, so the "
             "other columns are mis-tuned on purpose.",
             fontsize=8, color=pal["textSecondary"], ha="left", va="bottom")
    save(fig, path)


def save(fig, path: Path):
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------- main


def build(summary_path: Path, output_dir: Path) -> list[Path]:
    doc = load_summary(summary_path)
    table = index_cells(doc)
    if not table:
        raise FigureError(f"summary.json has no cells in slice {SLICE!r}")
    pal = horizon_mpl.apply_style("light")
    colors = loss_colors(pal)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "s6_recovery.png"
    noise_path = output_dir / "s6_recovery_all_noise.png"
    make_main_figure(doc, table, pal, colors, main_path)
    make_all_noise_figure(doc, table, pal, colors, noise_path)
    return [main_path, noise_path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summary", type=Path, help="path to summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        written = build(args.summary, args.output_dir)
    except FigureError as err:
        print(f"make_s6_figure: error: {err}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
