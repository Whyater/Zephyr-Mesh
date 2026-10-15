"""Draw the turn-aware recovery figures from a run directory (summary.json).

Questions answered: which recovery arm closes the turn-under-loss gap, if any, and what does
it cost on a straight line?

Usage (from the repo root):

    PYTHONPATH=. ./.venv/bin/python tools/make_s6_turn_figure.py runs/s6-turn \
        [--output-dir DIR]

Writes two PNGs into the output directory (default: the run directory):

    s6_turn.png         median p95 versus one-way delay for all seven arms at 5 cm noise,
                        columns are the target motion (straight, turn, mixed), rows are
                        packet loss (0 and 30 %)
    s6_turn_phases.png  the mixed target only: p95 in each phase (straight, turn, straight
                        again) per arm, at one representative delay

Every plotted number is read from summary.json. The script only converts seconds to
milliseconds and fractions to percent for labels. Colors come from
design/horizon_tokens.json through tools/horizon_mpl.py. If the output directory has a
manifest.json, the PNG hashes are added to its "outputs" map. The same input always
produces the same PNG bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from tools import horizon_mpl  # noqa: E402

EXPERIMENT = "s6-turn"
SLICE = "main"
LOSS_MODEL = "independent"
NOISE_M = 0.05
TARGETS = ("cv", "ct", "mixed")
LOSSES = (0.0, 0.30)
ARMS = ("naive", "recovery", "ca", "ca_q10", "ct", "ct_q10", "imm")
PHASES = ("straight1", "turn", "straight2")
FAIL_RATE_THRESHOLD = 0.5
DPI = 200
SUMMARY_NAME = "summary.json"
MAIN_PNG = "s6_turn.png"
PHASE_PNG = "s6_turn_phases.png"

ARM_LABELS = {
    "naive": "Naive (arrival-time stamps)",
    "recovery": "Capture-time, straight-line model",
    "ca": "Acceleration model",
    "ca_q10": "Acceleration model, 10x process noise",
    "ct": "Turn-rate model",
    "ct_q10": "Turn-rate model, 10x process noise",
    "imm": "Two-model mix (straight and turning)",
}
TARGET_TITLES = {
    "cv": "Straight line",
    "ct": "Steady turn",
    "mixed": "Straight, half turn, straight",
}
PHASE_LABELS = {
    "straight1": "Straight (before the turn)",
    "turn": "During the turn",
    "straight2": "Straight (after the turn)",
}


class FigureError(Exception):
    """Raised for unusable input; main() turns it into a clear exit message."""


# ---------------------------------------------------------------- input


def load_summary(run_dir: Path) -> dict:
    path = run_dir / SUMMARY_NAME
    try:
        with path.open(encoding="utf-8") as handle:
            doc = json.load(handle)
    except OSError as err:
        raise FigureError(f"cannot read {path}: {err}") from err
    except json.JSONDecodeError as err:
        raise FigureError(f"{path} is not valid JSON: {err}") from err
    if not isinstance(doc, dict) or doc.get("experiment") != EXPERIMENT:
        found = doc.get("experiment") if isinstance(doc, dict) else type(doc).__name__
        raise FigureError(f"unexpected experiment {found!r}; this script reads {EXPERIMENT!r}")
    for key in ("config", "cells", "seeds", "failure_rule"):
        if key not in doc:
            raise FigureError(f"{SUMMARY_NAME} is missing the {key!r} field")
    for key in ("failure_p95_m", "observation_period_s"):
        if key not in doc["config"]:
            raise FigureError(f"{SUMMARY_NAME} config is missing {key!r}")
    return doc


def index_cells(doc: dict) -> dict:
    """Map (target, loss, delay) to the main-slice cell at the main noise level."""
    table = {}
    for cell in doc["cells"]:
        if cell.get("slice") != SLICE:
            continue
        spec = cell["cell"]
        if spec.get("loss_model") != LOSS_MODEL:
            continue
        if abs(spec["noise_std_m"] - NOISE_M) > 1e-9:
            continue
        key = (spec["target"], round(spec["loss_probability"], 6), round(spec["delay_s"], 6))
        table[key] = cell
    return table


def delays_for(table: dict, target: str, loss: float) -> list[float]:
    return sorted(k[2] for k in table if k[0] == target and k[1] == round(loss, 6))


def series(table: dict, target: str, loss: float, arm: str) -> dict:
    """Delay-sorted values for one line: x in ms, median / low / high p95 in m, fail flags."""
    out = {"x_ms": [], "med": [], "low": [], "high": [], "fails": []}
    for delay in delays_for(table, target, loss):
        summary = table[(target, round(loss, 6), delay)]["arms"][arm]
        out["x_ms"].append(delay * 1000.0)
        out["med"].append(summary["p95_median_m"])
        out["low"].append(summary["p95_ci_low_m"])
        out["high"].append(summary["p95_ci_high_m"])
        out["fails"].append(summary["fail_rate"] >= FAIL_RATE_THRESHOLD)
    if not out["x_ms"]:
        raise FigureError(f"no {SLICE!r} cells for target={target}, loss={loss}, noise={NOISE_M}")
    return out


def representative_delay(table: dict) -> float:
    """The median of the delays present for the mixed target (lower middle if even)."""
    present = sorted({k[2] for k in table if k[0] == "mixed"})
    if not present:
        raise FigureError("no mixed-target cells in the main slice")
    return present[(len(present) - 1) // 2]


def fmt_g(value: float) -> str:
    return f"{value:g}"


def loss_title(loss: float) -> str:
    if loss == 0:
        return "No packet loss"
    return f"{fmt_g(round(loss * 100, 3))} % packet loss"


# ---------------------------------------------------------------- style


def arm_styles(pal: dict) -> dict:
    """Per-arm color, line style and marker from horizon_mpl.arm_style, plus width and z-order.

    The same arm has the same color, line style and marker in every S6 figure. Red and
    amber stay reserved for the failure line.
    """
    extra = {
        "naive": {"lw": 1.3, "z": 3},
        "recovery": {"lw": 1.5, "z": 3.2},
        "ca": {"lw": 1.7, "z": 4},
        "ca_q10": {"lw": 1.7, "z": 3.8},
        "ct": {"lw": 1.9, "z": 5},
        "ct_q10": {"lw": 1.7, "z": 4.8},
        "imm": {"lw": 1.9, "z": 5},
    }
    out = {}
    for arm in ARMS:
        base = horizon_mpl.arm_style(arm, pal)
        out[arm] = {"color": base["color"], "ls": base["linestyle"], "marker": base["marker"],
                    **extra[arm]}
    return out


# ---------------------------------------------------------------- drawing


def draw_panel(ax, table, target, loss, styles, pal, failure_m, label_failure):
    top = failure_m
    xs_all = []
    for arm in ARMS:
        st = styles[arm]
        data = series(table, target, loss, arm)
        xs_all.extend(data["x_ms"])
        top = max(top, max(data["high"]))
        ax.fill_between(data["x_ms"], data["low"], data["high"], color=st["color"],
                        alpha=0.10, linewidth=0, zorder=2)
        ax.plot(data["x_ms"], data["med"], color=st["color"], linestyle=st["ls"],
                linewidth=st["lw"], zorder=st["z"], clip_on=False)
        # Filled marker: the cell fails (at least half of the seeds fail). Open: it passes.
        for x, y, fails in zip(data["x_ms"], data["med"], data["fails"]):
            ax.plot([x], [y], marker=st["marker"], markersize=3.8,
                    markerfacecolor=st["color"] if fails else pal["surface"],
                    markeredgecolor=st["color"], markeredgewidth=0.9, linestyle="none",
                    zorder=st["z"] + 0.1, clip_on=False)
    ax.axhline(failure_m, color=pal["alert"], linewidth=1.0, zorder=1.5)
    if label_failure:
        ax.annotate(f"failure line (p95 {fmt_g(round(failure_m * 100, 3))} cm)", xy=(0.02, failure_m),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8,
                    color=pal["textSecondary"])
    ax.grid(axis="x", visible=False)
    return top, sorted(set(xs_all))


def legend_groups(pal, styles):
    def handle(arm):
        st = styles[arm]
        return Line2D([0], [0], color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                      marker=st["marker"], markersize=3.8, markerfacecolor=pal["surface"],
                      markeredgecolor=st["color"], markeredgewidth=0.9)

    baselines = ("naive", "recovery")
    models = ("ca", "ca_q10", "ct", "ct_q10", "imm")
    marker_handles = [
        Line2D([0], [0], color=pal["textSecondary"], marker="o", markersize=3.8,
               markerfacecolor=pal["surface"], markeredgewidth=0.9, linestyle="none"),
        Line2D([0], [0], color=pal["textSecondary"], marker="o", markersize=3.8,
               markerfacecolor=pal["textSecondary"], markeredgewidth=0.9, linestyle="none"),
        Patch(facecolor=pal["textSecondary"], alpha=0.16, linewidth=0),
    ]
    marker_labels = ["open marker: cell passes", "filled marker: cell fails", "95 % interval"]
    return (
        ([handle(a) for a in baselines], [ARM_LABELS[a] for a in baselines], "Baselines"),
        ([handle(a) for a in models], [ARM_LABELS[a] for a in models], "Model-based arms"),
        (marker_handles, marker_labels, "Markers"),
    )


def add_legends(fig, groups, pal, y, x_positions):
    for (handles, labels, title), x in zip(groups, x_positions):
        leg = fig.legend(handles, labels, title=title, loc="upper left",
                         bbox_to_anchor=(x, y), alignment="left", handlelength=2.6,
                         labelspacing=0.4, borderaxespad=0, handletextpad=0.7)
        leg.get_title().set_color(pal["textSecondary"])
        leg.get_title().set_fontsize(8.5)


def footnote(doc, extra: str | None = None) -> str:
    n_seeds = len(doc["seeds"])
    hz = 1.0 / doc["config"]["observation_period_s"]
    first = (f"Simulated link and sensor, {n_seeds} seeds per cell, {fmt_g(hz)} Hz measurements, "
             f"noise {fmt_g(NOISE_M * 100)} cm. Median p95 with 95 % interval.")
    if extra:
        first = (f"Simulated link and sensor, {n_seeds} seeds per cell, {fmt_g(hz)} Hz "
                 f"measurements, noise {fmt_g(NOISE_M * 100)} cm. {extra}")
    rule = ("A cell fails when at least 50 % of seeds fail (rule proposed, not yet adopted). "
            "Not flight data.")
    return first + "\n" + rule


def make_main_figure(doc, table, pal, styles, path):
    fig = plt.figure(figsize=(11.0, 8.3))
    gs = fig.add_gridspec(len(LOSSES), len(TARGETS), left=0.07, right=0.985, top=0.835,
                          bottom=0.33, wspace=0.08, hspace=0.30)
    failure_m = doc["config"]["failure_p95_m"]
    axes = [[None] * len(TARGETS) for _ in LOSSES]
    top = failure_m
    xs = []
    for r, loss in enumerate(LOSSES):
        for c, target in enumerate(TARGETS):
            first = axes[0][0]
            ax = fig.add_subplot(gs[r, c], sharex=first, sharey=first)
            axes[r][c] = ax
            t, x = draw_panel(ax, table, target, loss, styles, pal, failure_m,
                              label_failure=(r == 0 and c == 0))
            top = max(top, t)
            xs.extend(x)
            if r == 0:
                ax.set_title(TARGET_TITLES[target], loc="left")
            if r == len(LOSSES) - 1:
                ax.set_xlabel("One-way delay (ms)")
            else:
                ax.tick_params(labelbottom=False)
            if c > 0:
                ax.tick_params(labelleft=False)
    ticks = sorted(set(xs))
    axes[0][0].set_xlim(min(ticks), max(ticks))
    axes[0][0].set_xticks(ticks)
    axes[0][0].set_ylim(0, top * 1.08)
    for r, loss in enumerate(LOSSES):
        axes[r][0].set_ylabel(f"{loss_title(loss)}\np95 tracking error (cm)")
    # Axis labels stay in centimeters: scale the tick labels, not the data.
    for row in axes:
        for ax in row:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{v * 100:g}"))
    fig.text(0.07, 0.965, "Does any tracker close the turn gap, and what does it cost on a straight line?",
             fontsize=12.5, color=pal["text"], ha="left", va="top")
    fig.text(0.07, 0.918, "p95 error of the target estimate versus link delay, median over seeds, "
             "seven tracker arms at 5 cm sensor noise",
             fontsize=9, color=pal["textSecondary"], ha="left", va="top")
    add_legends(fig, legend_groups(pal, styles), pal, y=0.235, x_positions=(0.07, 0.40, 0.745))
    fig.text(0.07, 0.025, footnote(doc), fontsize=8, color=pal["textSecondary"],
             ha="left", va="bottom", linespacing=1.4)
    save(fig, path)


def make_phase_figure(doc, table, pal, styles, path):
    delay = representative_delay(table)
    failure_m = doc["config"]["failure_p95_m"]
    fig = plt.figure(figsize=(11.0, 7.2))
    gs = fig.add_gridspec(1, len(LOSSES), left=0.07, right=0.985, top=0.795, bottom=0.40,
                          wspace=0.07)
    n_arms = len(ARMS)
    width = 0.8 / n_arms
    top = failure_m
    values = {}
    for loss in LOSSES:
        cell = table.get(("mixed", round(loss, 6), delay))
        if cell is None or "phase_p95" not in cell:
            raise FigureError(f"mixed cell at loss={loss}, delay={delay} has no phase_p95")
        values[loss] = cell["phase_p95"]
        for arm in ARMS:
            for phase in PHASES:
                top = max(top, values[loss][arm][phase])
    axes = []
    for i, loss in enumerate(LOSSES):
        ax = fig.add_subplot(gs[0, i], sharey=axes[0] if axes else None)
        axes.append(ax)
        for j, arm in enumerate(ARMS):
            st = styles[arm]
            xs = [p + (j - (n_arms - 1) / 2) * width for p in range(len(PHASES))]
            ys = [values[loss][arm][phase] for phase in PHASES]
            ax.bar(xs, ys, width=width * 0.88, color=st["color"], linewidth=0, zorder=3)
        ax.axhline(failure_m, color=pal["alert"], linewidth=1.0, zorder=4)
        if i == 0:
            ax.annotate(f"failure line (p95 {fmt_g(round(failure_m * 100, 3))} cm)", xy=(0.02, failure_m),
                        xycoords=("axes fraction", "data"), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom", fontsize=8,
                        color=pal["textSecondary"])
        ax.set_xticks(range(len(PHASES)))
        ax.set_xticklabels([PHASE_LABELS[p] for p in PHASES])
        ax.set_xlim(-0.5, len(PHASES) - 0.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(loss_title(loss), loc="left")
        if i > 0:
            ax.tick_params(labelleft=False)
    axes[0].set_ylim(0, top * 1.12)
    axes[0].set_ylabel("p95 tracking error (cm)")
    for ax in axes:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{v * 100:g}"))
    fig.text(0.07, 0.965, "Mixed target: in which phase of the path does each tracker lose accuracy?",
             fontsize=12.5, color=pal["text"], ha="left", va="top")
    fig.text(0.07, 0.92, f"p95 error in each phase of the path, median over seeds, "
             f"{fmt_g(round(delay * 1000, 3))} ms one-way delay, 5 cm sensor noise",
             fontsize=9, color=pal["textSecondary"], ha="left", va="top")
    baselines = ("naive", "recovery")
    models = ("ca", "ca_q10", "ct", "ct_q10", "imm")

    def swatch(arm):
        return Patch(facecolor=styles[arm]["color"], linewidth=0)

    groups = (
        ([swatch(a) for a in baselines], [ARM_LABELS[a] for a in baselines], "Baselines"),
        ([swatch(a) for a in models], [ARM_LABELS[a] for a in models], "Model-based arms"),
    )
    add_legends(fig, groups, pal, y=0.275, x_positions=(0.07, 0.40))
    extra = ("Median over seeds of each seed's p95 within the phase (no interval is drawn).")
    fig.text(0.07, 0.025, footnote(doc, extra), fontsize=8, color=pal["textSecondary"],
             ha="left", va="bottom", linespacing=1.4)
    save(fig, path)


def save(fig, path: Path):
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------- manifest


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_manifest(output_dir: Path, pngs: list[Path]) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    outputs = manifest.setdefault("outputs", {})
    for png in pngs:
        outputs[png.name] = sha256_file(png)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


# ---------------------------------------------------------------- main


def build(run_dir: Path, output_dir: Path) -> list[Path]:
    doc = load_summary(run_dir)
    table = index_cells(doc)
    if not table:
        raise FigureError(f"{SUMMARY_NAME} has no {SLICE!r} cells at {NOISE_M} m noise")
    present = {a["name"] for a in doc.get("arms", [])}
    if present and not set(ARMS) <= present:
        raise FigureError(f"summary is missing arms: {sorted(set(ARMS) - present)}")
    pal = horizon_mpl.apply_style("light")
    styles = arm_styles(pal)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / MAIN_PNG
    phase_path = output_dir / PHASE_PNG
    make_main_figure(doc, table, pal, styles, main_path)
    make_phase_figure(doc, table, pal, styles, phase_path)
    update_manifest(output_dir, [main_path, phase_path])
    return [main_path, phase_path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="run directory that holds summary.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where to write the PNGs (default: the run directory)")
    args = parser.parse_args(argv)
    try:
        written = build(args.run_dir, args.output_dir or args.run_dir)
    except FigureError as err:
        print(f"make_s6_turn_figure: error: {err}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
