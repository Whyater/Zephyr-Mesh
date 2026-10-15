"""Draw the late-packet re-filtering figures from a runs/s6-retro/ output folder.

Usage (from the repo root):

    PYTHONPATH=. ./.venv/bin/python tools/make_s6_retro_figure.py runs/s6-retro \
        [--output-dir DIR]

Writes two PNGs (default: into the run folder):

    s6_retro.png          main figure
    s6_retro_reorder.png  supporting figure

Main figure. Question: when link delay jitters, packets arrive out of order. Does
putting the late ones back into the filter (retro) beat dropping them (recovery),
and by how much? The figure shows p95 tracking error against delay for the naive
tracker, the recovery tracker and the retro tracker. Columns are the straight line
and the turn, rows are two jitter sizes, line style is packet loss, and the red line
is the failure line.

Supporting figure. Questions: how often do packets arrive after a newer one (the
reorder census), and how many centimeters does retro save compared with recovery,
paired seed by seed?

Arm colors and markers come from horizon_mpl.arm_style and are the same in every S6
figure. Red is used only for the failure line and for rings around cells that fail.

Every plotted number is read from the run files (summary.json). The script only
converts seconds to milliseconds, meters to centimeters and fractions to
percent for labels. Colors come from design/horizon_tokens.json through
tools/horizon_mpl.py. The same input files always give the same PNG bytes.
When the output folder has a manifest.json, the PNG hashes are added to its
"outputs" map.
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
from matplotlib.lines import Line2D  # noqa: E402

from tools import horizon_mpl  # noqa: E402

MAIN_NAME = "s6_retro.png"
REORDER_NAME = "s6_retro_reorder.png"
SLICES = ("J1", "J2")
TARGETS = ("cv", "ct")
LOSSES = (0.0, 0.30)
ARMS = ("naive", "recovery", "retro")
ARM_LABELS = {
    "naive": "Naive (arrival-time stamps)",
    "recovery": "Recovery (drops late packets)",
    "retro": "Retro (re-filters late packets)",
}
DPI = 200
LOSS_STYLES = {0.0: "-", 0.30: (0, (4, 2.2))}


class FigureError(Exception):
    """Raised for unusable input; main() turns it into a clear exit message."""


# ---------------------------------------------------------------- input


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    try:
        with path.open(encoding="utf-8") as handle:
            doc = json.load(handle)
    except OSError as err:
        raise FigureError(f"cannot read {path}: {err}") from err
    except json.JSONDecodeError as err:
        raise FigureError(f"{path} is not valid JSON: {err}") from err
    if not isinstance(doc, dict):
        raise FigureError(f"{path} does not hold a JSON object")
    for key in ("config", "cells"):
        if key not in doc:
            raise FigureError(f"summary.json is missing the {key!r} field")
    for key in ("failure_p95_m", "target_speed_mps", "turn_radius_m"):
        if key not in doc["config"]:
            raise FigureError(f"summary.json config is missing {key!r}")
    if not doc["cells"]:
        raise FigureError("summary.json has no cells")
    return doc


def index_cells(doc: dict) -> dict:
    """Map (slice, target, loss, delay) to the cell."""
    table = {}
    for cell in doc["cells"]:
        spec = cell["cell"]
        key = (
            cell["slice"],
            spec["target"],
            round(spec["loss_probability"], 6),
            round(spec["delay_s"], 6),
        )
        table[key] = cell
    return table


def seeds_per_cell(doc: dict) -> int:
    counts = {
        arm["n_seeds"] for cell in doc["cells"] for arm in cell["arms"].values()
    }
    if not counts:
        raise FigureError("summary.json cells hold no arm results")
    return max(counts)


def rows(table: dict, slice_name: str, target: str, loss: float) -> list:
    """Delay-sorted (delay_s, cell) pairs for one slice, target and loss."""
    return sorted(
        (key[3], cell)
        for key, cell in table.items()
        if key[0] == slice_name and key[1] == target and key[2] == round(loss, 6)
    )


def arm_series(table, slice_name, target, loss, arm) -> dict:
    out = {"x_ms": [], "med": [], "low": [], "high": [], "label": []}
    for delay_s, cell in rows(table, slice_name, target, loss):
        summary = cell["arms"][arm]
        out["x_ms"].append(delay_s * 1000.0)
        out["med"].append(summary["p95_median_m"] * 100.0)
        out["low"].append(summary["p95_ci_low_m"] * 100.0)
        out["high"].append(summary["p95_ci_high_m"] * 100.0)
        out["label"].append(summary.get("label"))
    return out


def census_series(table, slice_name, loss) -> dict:
    """Reorder census; it depends only on the link draws, so the first target with cells is used."""
    for target in TARGETS:
        found = rows(table, slice_name, target, loss)
        if found:
            out = {"x_ms": [], "mean": [], "low": [], "high": []}
            for delay_s, cell in found:
                census = cell["reorder_census"]
                out["x_ms"].append(delay_s * 1000.0)
                out["mean"].append(census["mean"] * 100.0)
                out["low"].append(census["min"] * 100.0)
                out["high"].append(census["max"] * 100.0)
            return out
    return {"x_ms": [], "mean": [], "low": [], "high": []}


def gain_series(table, slice_name, target, loss) -> dict:
    """Paired gain of retro over recovery in cm (positive means retro is better)."""
    out = {"x_ms": [], "med": [], "low": [], "high": []}
    for delay_s, cell in rows(table, slice_name, target, loss):
        diff = cell["paired_vs_recovery"]["retro"]
        out["x_ms"].append(delay_s * 1000.0)
        out["med"].append(diff["p95_diff_median_m"] * 100.0)
        out["low"].append(diff["p95_diff_ci_low_m"] * 100.0)
        out["high"].append(diff["p95_diff_ci_high_m"] * 100.0)
    return out


# ---------------------------------------------------------------- labels


def fmt_g(value: float) -> str:
    return f"{value:g}"


def target_title(doc: dict, target: str) -> str:
    speed = fmt_g(doc["config"]["target_speed_mps"])
    if target == "cv":
        return f"Straight line, {speed} m/s"
    return f"Turning, {speed} m/s, {fmt_g(doc['config']['turn_radius_m'])} m radius"


def jitter_label(table: dict, slice_name: str) -> str:
    """Describe the jitter size of a slice from the cells themselves."""
    ratios = set()
    for key, cell in table.items():
        if key[0] == slice_name and key[3] > 0:
            ratios.add(round(cell["cell"]["jitter_s"] / cell["cell"]["delay_s"], 4))
    if len(ratios) != 1:
        return f"Jitter slice {slice_name}"
    ratio = ratios.pop()
    if abs(ratio - 0.5) < 1e-9:
        return "Jitter = delay / 2"
    if abs(ratio - 1.0) < 1e-9:
        return "Jitter = delay"
    return f"Jitter = {fmt_g(ratio)} x delay"


def loss_label(loss: float) -> str:
    return f"{fmt_g(round(loss * 100, 6))} % packet loss"


def arm_styles(pal: dict) -> dict:
    """Color, line style and marker per arm, shared with every other S6 figure."""
    return {arm: horizon_mpl.arm_style(arm, pal) for arm in ARMS}


def footnote(n_seeds: int, failure_shown: bool) -> str:
    text = f"Simulated link and sensor, {n_seeds} seeds per cell"
    if failure_shown:
        text += ", median p95 with 95 % interval."
    else:
        text += ", median p95 differences with 95 % interval."
    if failure_shown:
        text += ("\nCell fails when at least 50 % of seeds fail (rule proposed, not yet "
                 "adopted).")
    return text


def x_ticks(delays_ms: list[float]) -> list[float]:
    top = max(delays_ms) if delays_ms else 300.0
    step = 100.0 if top > 150 else 50.0
    ticks, value = [], 0.0
    while value <= top + 1e-9:
        ticks.append(value)
        value += step
    return ticks


def empty_note(ax, pal):
    ax.text(0.5, 0.5, "no cells in this run", transform=ax.transAxes, ha="center",
            va="center", fontsize=9, color=pal["textSecondary"])


# ---------------------------------------------------------------- main figure


def draw_arm_line(ax, series, style, loss, arm, pal, z):
    if not series["x_ms"]:
        return
    color, marker = style["color"], style["marker"]
    hollow = loss > 0
    face = pal["surface"] if hollow else color
    if arm == "retro":
        width, size, alpha_band = 2.6, 4.4, 0.16
    elif arm == "recovery":
        width, size, alpha_band = 1.3, 3.0, 0.0
    else:
        width, size, alpha_band = 1.3, 3.4, 0.0
    if arm == "retro":
        ax.fill_between(series["x_ms"], series["low"], series["high"], color=color,
                        alpha=alpha_band, linewidth=0, zorder=2)
    elif arm == "naive":
        lower = [m - lo for m, lo in zip(series["med"], series["low"])]
        upper = [hi - m for m, hi in zip(series["med"], series["high"])]
        ax.errorbar(series["x_ms"], series["med"], yerr=[lower, upper], fmt="none",
                    ecolor=color, elinewidth=0.7, capsize=1.8, capthick=0.7,
                    clip_on=False, zorder=z - 0.1)
    ax.plot(series["x_ms"], series["med"], color=color, linestyle=LOSS_STYLES[loss],
            linewidth=width, marker=marker, markersize=size, markerfacecolor=face,
            markeredgecolor=color, markeredgewidth=0.8, clip_on=False, zorder=z)


def draw_main_panel(ax, table, slice_name, target, styles, pal, failure_cm, label_failure):
    top = failure_cm
    delays = []
    z = {"naive": 3.0, "retro": 3.5, "recovery": 4.0}
    found = False
    for loss in LOSSES:
        for arm in ARMS:
            series = arm_series(table, slice_name, target, loss, arm)
            if series["x_ms"]:
                found = True
                top = max(top, max(series["high"]))
                delays.extend(series["x_ms"])
            draw_arm_line(ax, series, styles[arm], loss, arm, pal, z[arm])
        retro = arm_series(table, slice_name, target, loss, "retro")
        fails = [(x, m) for x, m, lab in zip(retro["x_ms"], retro["med"], retro["label"])
                 if lab == "fails"]
        if fails:
            ax.plot([f[0] for f in fails], [f[1] for f in fails], linestyle="none",
                    marker="o", markersize=9, markerfacecolor="none",
                    markeredgecolor=pal["alert"], markeredgewidth=1.0, zorder=5)
    ax.axhline(failure_cm, color=pal["alert"], linewidth=1.0, zorder=1.5)
    if label_failure:
        ax.annotate(f"failure line (p95 {fmt_g(failure_cm)} cm)", xy=(0.012, failure_cm),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8,
                    color=pal["textSecondary"])
    ax.grid(axis="x", visible=False)
    if not found:
        empty_note(ax, pal)
    return top, delays


def main_legends(fig, styles, pal, y):
    arm_handles = [Line2D([0], [0], color=styles[a]["color"], linewidth=2.2,
                          marker=styles[a]["marker"], markersize=4.6,
                          markeredgecolor=styles[a]["color"]) for a in ARMS]
    arm_labels = [ARM_LABELS[a] for a in ARMS]
    loss_handles = [
        Line2D([0], [0], color=pal["textSecondary"], linestyle=LOSS_STYLES[p], linewidth=1.5,
               marker="o", markersize=3.6,
               markerfacecolor=pal["textSecondary"] if p == 0 else pal["surface"],
               markeredgecolor=pal["textSecondary"], markeredgewidth=0.8)
        for p in LOSSES
    ]
    loss_labels = [loss_label(p) for p in LOSSES]
    fail_handle = [Line2D([0], [0], linestyle="none", marker="o", markersize=8,
                          markerfacecolor="none", markeredgecolor=pal["alert"],
                          markeredgewidth=1.0)]
    groups = (
        (arm_handles, arm_labels, "Color and marker: tracker", 0.075),
        (loss_handles, loss_labels, "Line style: link loss", 0.40),
        (fail_handle, ["retro cell fails"], "Marker", 0.62),
    )
    for handles, labels, title, x in groups:
        leg = fig.legend(handles, labels, title=title, loc="upper left",
                         bbox_to_anchor=(x, y), alignment="left", handlelength=2.6,
                         labelspacing=0.45, borderaxespad=0, handletextpad=0.7)
        leg.get_title().set_color(pal["textSecondary"])
        leg.get_title().set_fontsize(8.5)
    fig.text(0.78, y - 0.012, "Bands: 95 % interval, retro.\nCaps: 95 % interval, naive.",
             fontsize=8, color=pal["textSecondary"], ha="left", va="top", linespacing=1.3)


def make_main_figure(doc, table, pal, path):
    styles = arm_styles(pal)
    fig = plt.figure(figsize=(9.5, 8.0))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, top=0.835, bottom=0.27,
                          wspace=0.08, hspace=0.20)
    failure_cm = doc["config"]["failure_p95_m"] * 100.0
    axes = [[None, None], [None, None]]
    top, delays = failure_cm, []
    for r, slice_name in enumerate(SLICES):
        for c, target in enumerate(TARGETS):
            ax = fig.add_subplot(gs[r, c], sharex=axes[0][0] if (r or c) else None,
                                 sharey=axes[0][0] if (r or c) else None)
            axes[r][c] = ax
            t, d = draw_main_panel(ax, table, slice_name, target, styles, pal, failure_cm,
                                   label_failure=(r == 0 and c == 0))
            top = max(top, t)
            delays.extend(d)
            if r == 0:
                ax.set_title(target_title(doc, target), loc="left")
            if r == 1:
                ax.set_xlabel("Mean one-way delay (ms)")
            else:
                ax.tick_params(labelbottom=False)
            if c == 1:
                ax.tick_params(labelleft=False)
    axes[0][0].set_ylim(0, top * 1.08)
    axes[0][0].set_xlim(0, max(delays) if delays else 300.0)
    axes[0][0].set_xticks(x_ticks(delays))
    for r, slice_name in enumerate(SLICES):
        axes[r][0].set_ylabel(f"{jitter_label(table, slice_name)}\np95 tracking error (cm)",
                              fontsize=9)
    fig.text(0.085, 0.965, "Does re-filtering late packets beat dropping them, and by how much?",
             fontsize=12, color=pal["text"], ha="left", va="top")
    fig.text(0.085, 0.925, "p95 error of the target estimate against link delay, median over seeds, "
             "by jitter size and target motion", fontsize=9, color=pal["textSecondary"],
             ha="left", va="top")
    main_legends(fig, styles, pal, y=0.165)
    fig.text(0.085, 0.025, footnote(seeds_per_cell(doc), True), fontsize=8,
             color=pal["textSecondary"], ha="left", va="bottom", linespacing=1.4)
    save(fig, path)


# ---------------------------------------------------------------- supporting figure


def slice_colors(pal: dict) -> dict:
    return {"J1": pal["textSecondary"], "J2": pal["text"]}


def draw_band_line(ax, series, key, color, loss, pal, low="low", high="high"):
    if not series["x_ms"]:
        return
    ax.fill_between(series["x_ms"], series[low], series[high], color=color, alpha=0.12,
                    linewidth=0, zorder=2)
    ax.plot(series["x_ms"], series[key], color=color, linestyle=LOSS_STYLES[loss],
            linewidth=1.6, marker="o", markersize=3.6,
            markerfacecolor=color if loss == 0 else pal["surface"], markeredgecolor=color,
            markeredgewidth=0.8, clip_on=False, zorder=4)


def make_reorder_figure(doc, table, pal, path):
    scolors = slice_colors(pal)
    fig = plt.figure(figsize=(11.0, 5.2))
    gs = fig.add_gridspec(1, 3, left=0.065, right=0.985, top=0.775, bottom=0.30, wspace=0.34)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_cv = fig.add_subplot(gs[0, 1])
    ax_ct = fig.add_subplot(gs[0, 2], sharey=ax_cv)
    delays = []
    found_census = False
    for slice_name in SLICES:
        for loss in LOSSES:
            s = census_series(table, slice_name, loss)
            if s["x_ms"]:
                found_census = True
                delays.extend(s["x_ms"])
            draw_band_line(ax_c, s, "mean", scolors[slice_name], loss, pal)
    ax_c.set_title("Packets arriving after a newer one", loc="left", fontsize=10)
    ax_c.set_ylabel("Late packets (% of delivered)")
    ax_c.set_ylim(bottom=0)
    if not found_census:
        empty_note(ax_c, pal)
    for ax, target in ((ax_cv, "cv"), (ax_ct, "ct")):
        found = False
        for slice_name in SLICES:
            for loss in LOSSES:
                g = gain_series(table, slice_name, target, loss)
                if g["x_ms"]:
                    found = True
                    delays.extend(g["x_ms"])
                draw_band_line(ax, g, "med", scolors[slice_name], loss, pal)
        ax.axhline(0, color=pal["lineStrong"], linewidth=0.8, zorder=1.5)
        ax.set_title(f"Retro gain over recovery: {target_title(doc, target).split(',')[0].lower()}",
                     loc="left", fontsize=10)
        if not found:
            empty_note(ax, pal)
    ax_cv.set_ylabel("p95 error saved (cm)")
    ax_ct.tick_params(labelleft=False)
    ticks = x_ticks(delays)
    for ax in (ax_c, ax_cv, ax_ct):
        ax.set_xlim(0, max(delays) if delays else 300.0)
        ax.set_xticks(ticks)
        ax.set_xlabel("Mean one-way delay (ms)")
        ax.grid(axis="x", visible=False)
    fig.text(0.065, 0.965, "How often do packets arrive late, and how much does re-filtering them save?",
             fontsize=12, color=pal["text"], ha="left", va="top")
    fig.text(0.065, 0.925, "Left: share of delivered packets that arrive after a newer one.\n"
             "Right: p95 error saved by retro against recovery, same seeds (above 0, retro had the lower error).",
             fontsize=9, color=pal["textSecondary"], ha="left", va="top")
    slice_handles = [Line2D([0], [0], color=scolors[s], linewidth=2.2) for s in SLICES]
    slice_labels = [jitter_label(table, s) for s in SLICES]
    loss_handles = [
        Line2D([0], [0], color=pal["textSecondary"], linestyle=LOSS_STYLES[p], linewidth=1.5,
               marker="o", markersize=3.6,
               markerfacecolor=pal["textSecondary"] if p == 0 else pal["surface"],
               markeredgecolor=pal["textSecondary"], markeredgewidth=0.8)
        for p in LOSSES
    ]
    for handles, labels, title, x in (
        (slice_handles, slice_labels, "Color: jitter", 0.065),
        (loss_handles, [loss_label(p) for p in LOSSES], "Line style: link loss", 0.27),
    ):
        leg = fig.legend(handles, labels, title=title, loc="upper left",
                         bbox_to_anchor=(x, 0.185), alignment="left", handlelength=2.6,
                         labelspacing=0.45, borderaxespad=0, handletextpad=0.7)
        leg.get_title().set_color(pal["textSecondary"])
        leg.get_title().set_fontsize(8.5)
    fig.text(0.48, 0.173, "Bands: left, lowest to highest seed; right, 95 % interval of the "
             "paired median.", fontsize=8, color=pal["textSecondary"], ha="left", va="top")
    fig.text(0.065, 0.025, footnote(seeds_per_cell(doc), False), fontsize=8,
             color=pal["textSecondary"], ha="left", va="bottom")
    save(fig, path)


def save(fig, path: Path):
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------- manifest and CLI


def update_manifest(output_dir: Path, paths: list[Path]) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    outputs = manifest.setdefault("outputs", {})
    for path in paths:
        outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return True


def build(run_dir: Path, output_dir: Path) -> list[Path]:
    doc = load_summary(run_dir)
    table = index_cells(doc)
    pal = horizon_mpl.apply_style("light")
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / MAIN_NAME
    reorder_path = output_dir / REORDER_NAME
    make_main_figure(doc, table, pal, main_path)
    make_reorder_figure(doc, table, pal, reorder_path)
    written = [main_path, reorder_path]
    update_manifest(output_dir, written)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="run folder holding summary.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where to write the PNGs (default: the run folder)")
    args = parser.parse_args(argv)
    try:
        written = build(args.run_dir, args.output_dir or args.run_dir)
    except FigureError as err:
        print(f"make_s6_retro_figure: error: {err}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
