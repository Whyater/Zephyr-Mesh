"""Draw the clock-sync figures from a run folder written by tools/run_s6_clocks.py.

Usage (from the repo root):

    PYTHONPATH=. ./.venv/bin/python tools/make_s6_clocks_figure.py runs/s6-clocks \
        [--output-dir DIR]

Writes two PNGs (into the run folder unless --output-dir is given):

    s6_clocks.png       Question: how wrong can the sensor clock be before the
                        capture-time tracker stops beating the naive tracker?
                        Median p95 versus fixed clock offset, naive, recovery and
                        IMM, by target motion and link delay. The hatched band
                        marks offsets where recovery is not clearly better than
                        naive.
    s6_clocks_sync.png  Question: does a simple NTP-style sync over the radio keep
                        the clock error small enough that it does not matter?
                        Extra recovery p95 (cm) compared with the same cell on
                        perfect clocks, by sync interval, path asymmetry and
                        clock drift, against a 1 cm budget.

Every plotted number is read from summary.json (and per_seed.csv.gz for the paired
differences in the second figure). The script only converts seconds to
milliseconds and meters to centimeters for labels. Colors come from
design/horizon_tokens.json through tools/horizon_mpl.py. The same files always
produce the same PNG bytes. When the output folder has a manifest.json, the PNG
hashes are added to its "outputs" map.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import binom  # noqa: E402

from tools import horizon_mpl  # noqa: E402

MAIN_NAME = "s6_clocks.png"
SYNC_NAME = "s6_clocks_sync.png"
DPI = 200
TOL = 1e-9
ARMS = ("naive", "recovery", "imm")
ARM_LABELS = {"naive": "Naive (arrival-time stamps)", "recovery": "Capture-time + predict-forward",
              "imm": "IMM (capture-time)"}
TARGET_NAMES = {"cv": "Straight line", "ct": "Turning"}
TARGET_ORDER = ("cv", "ct")
PREFERRED_NOISE_M = 0.05
BUDGET_CM = 1.0
FIG_WIDTH_IN = 10.5


class FigureError(Exception):
    """Raised for unusable input; main() turns it into a clear exit message."""


# ---------------------------------------------------------------- input


def read_json(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as err:
        raise FigureError(f"cannot read {path}: {err}") from err
    except json.JSONDecodeError as err:
        raise FigureError(f"{path} is not valid JSON: {err}") from err


def load_summary(run_dir: Path) -> dict:
    doc = read_json(run_dir / "summary.json")
    if not isinstance(doc, dict) or doc.get("experiment_id") != "s6-clocks":
        found = doc.get("experiment_id") if isinstance(doc, dict) else type(doc).__name__
        raise FigureError(f"unexpected experiment {found!r}; this script reads 's6-clocks' runs")
    for key in ("cells", "failure_rule"):
        if key not in doc:
            raise FigureError(f"summary.json is missing the {key!r} field")
    if "p95_m" not in doc["failure_rule"]:
        raise FigureError("summary.json failure_rule is missing 'p95_m'")
    return doc


def load_per_seed(run_dir: Path) -> dict:
    """Map (cell key, seed index) to recovery's p95 error in meters."""
    path = run_dir / "per_seed.csv.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as err:
        raise FigureError(f"cannot read {path}: {err}") from err
    out = {}
    for row in rows:
        if row.get("arm") != "recovery":
            continue
        out[(row["cell_key"], int(row["seed_index"]))] = float(row["p95_error_m"])
    if not out:
        raise FigureError("per_seed.csv.gz has no rows for the recovery arm")
    return out


def pick_noise(cells: list[dict]) -> float:
    noises = sorted({round(c["cell"]["noise_std_m"], 6) for c in cells})
    if not noises:
        raise FigureError("summary.json has no cells")
    return min(noises, key=lambda v: (abs(v - PREFERRED_NOISE_M), v))


def split_slices(doc: dict) -> tuple[list[dict], list[dict]]:
    u_cells = [c for c in doc["cells"] if c.get("slice") == "U"]
    n_cells = [c for c in doc["cells"] if c.get("slice") == "N"]
    return u_cells, n_cells


def n_seeds_of(doc: dict) -> int:
    counts = {arm["n_seeds"] for cell in doc["cells"] for arm in cell["arms"].values()}
    if len(counts) == 1:
        return counts.pop()
    return len(doc.get("seeds", [])) or max(counts)


def fmt_g(value: float) -> str:
    return f"{value:g}"


def ms(value_s: float) -> float:
    return round(value_s * 1000.0, 6)


def footnote(n_seeds: int, failure_shown: bool, extra: str = "") -> str:
    text = (f"Simulated link and sensor, {n_seeds} seeds per cell, median p95 with 95 % "
            "interval.")
    if extra:
        text += " " + extra
    if failure_shown:
        text += ("\nCell fails when at least 50 % of seeds fail (rule proposed, not yet "
                 "adopted).")
    return text


def arm_styles(pal: dict) -> dict:
    """Color, line style and marker per arm, shared with every other S6 figure."""
    return {arm: horizon_mpl.arm_style(arm, pal) for arm in ARMS}


def median_interval(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Median and a distribution-free interval from order statistics (binomial)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    median = float(np.median(x))
    tail = (1.0 - confidence) / 2.0
    candidates = [k for k in range(1, n // 2 + 1) if binom.cdf(k - 1, n, 0.5) <= tail]
    if not candidates:
        return median, float(x[0]), float(x[-1])
    k = max(candidates)
    return median, float(x[k - 1]), float(x[n - k])


# ---------------------------------------------------------------- figure 1: slice U


def index_u(u_cells: list[dict], noise: float) -> dict:
    """Map (target, delay ms, loss, offset ms) to the cell."""
    table = {}
    for cell in u_cells:
        spec = cell["cell"]
        if abs(spec["noise_std_m"] - noise) > TOL:
            continue
        clock = spec["clock"]
        key = (spec["target"], ms(spec["delay_s"]), round(spec["loss_probability"], 6),
               ms(clock["offset_s"]))
        table[key] = cell
    return table


def u_series(table: dict, target: str, delay: float, loss: float) -> list[tuple]:
    return sorted((key[3], cell) for key, cell in table.items()
                  if key[0] == target and key[1] == delay and key[2] == loss)


def not_better_spans(rows: list[tuple], lo: float, hi: float) -> list[tuple]:
    """x spans around offsets where recovery is not clearly better than naive.

    Clearly better means the 95 % interval of the paired difference (naive minus
    recovery) lies above zero. Each offset claims the stretch halfway to its
    neighbours; the outermost ones run to the panel edge.
    """
    xs = [x for x, _ in rows]
    flags = [cell["paired_vs_naive"]["recovery"]["p95_diff_ci_low_m"] <= 0.0
             for _, cell in rows]
    spans = []
    for i, (x, bad) in enumerate(zip(xs, flags)):
        if not bad:
            continue
        left = lo if i == 0 else (xs[i - 1] + x) / 2.0
        right = hi if i == len(xs) - 1 else (x + xs[i + 1]) / 2.0
        if spans and abs(spans[-1][1] - left) < TOL:
            spans[-1] = (spans[-1][0], right)
        else:
            spans.append((left, right))
    return spans


def draw_u_panel(ax, table, target, delay, losses, styles, pal, failure_cm, xlim):
    top = failure_cm
    bottom = None
    base_rows = u_series(table, target, delay, losses[0])
    for lo, hi in not_better_spans(base_rows, *xlim):
        ax.axvspan(lo, hi, facecolor="none", edgecolor=pal["line"], hatch="////", linewidth=0,
                   zorder=0.6)
    if xlim[0] < delay < xlim[1]:
        ax.axvline(delay, color=pal["lineStrong"], linewidth=0.8, linestyle=(0, (1, 2)),
                   zorder=1.2)
    for loss in losses:
        rows = u_series(table, target, delay, loss)
        solid = loss == losses[0]
        for arm in ARMS:
            x = [r[0] for r in rows]
            med = [r[1]["arms"][arm]["p95_median_m"] * 100.0 for r in rows]
            low = [r[1]["arms"][arm]["p95_ci_low_m"] * 100.0 for r in rows]
            high = [r[1]["arms"][arm]["p95_ci_high_m"] * 100.0 for r in rows]
            top = max(top, max(high))
            bottom = min(low) if bottom is None else min(bottom, min(low))
            color = styles[arm]["color"]
            # Line style is the link-loss code in this figure; color and marker name the arm.
            if solid:
                ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(x, med, color=color, linestyle="-" if solid else (0, (4, 2.2)),
                    linewidth=1.5 if solid else 1.2, marker=styles[arm]["marker"],
                    markersize=3.4,
                    markerfacecolor=color if solid else pal["surface"], markeredgecolor=color,
                    markeredgewidth=0.8, clip_on=False, zorder=4 if arm == "recovery" else 3)
            fails = [(xx, m) for xx, m, (_, cell) in zip(x, med, rows)
                     if cell["arms"][arm]["label"] == "fails"]
            if fails:
                ax.plot([f[0] for f in fails], [f[1] for f in fails], linestyle="none",
                        marker="o", markersize=8.5, markerfacecolor="none",
                        markeredgecolor=pal["alert"], markeredgewidth=1.0, zorder=5)
    ax.axhline(failure_cm, color=pal["alert"], linewidth=1.0, zorder=1.5)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", visible=False)
    return top, bottom


def offset_ticks(xlim: tuple) -> list[float]:
    ticks = [t for t in (-100, -50, 0, 50, 100) if xlim[0] <= t <= xlim[1]]
    return ticks or [0]


def main_legends(fig, styles, pal, losses, y):
    arm_handles = [Line2D([0], [0], color=styles[a]["color"], linewidth=2.2,
                          marker=styles[a]["marker"], markersize=4.6,
                          markeredgecolor=styles[a]["color"]) for a in ARMS]
    arm_labels = [ARM_LABELS[a] for a in ARMS]
    loss_handles = [
        Line2D([0], [0], color=pal["textSecondary"], linewidth=1.5,
               linestyle="-" if i == 0 else (0, (4, 2.2)), marker="o", markersize=3.6,
               markerfacecolor=pal["textSecondary"] if i == 0 else pal["surface"],
               markeredgecolor=pal["textSecondary"], markeredgewidth=0.8)
        for i in range(len(losses))
    ]
    loss_labels = [f"{fmt_g(round(p * 100, 6))} % packet loss" for p in losses]
    mark_handles = [
        Patch(facecolor="none", edgecolor=pal["lineStrong"], hatch="////", linewidth=0),
        Line2D([0], [0], linestyle="none", marker="o", markersize=8, markerfacecolor="none",
               markeredgecolor=pal["alert"], markeredgewidth=1.0),
        Line2D([0], [0], color=pal["lineStrong"], linewidth=0.8, linestyle=(0, (1, 2))),
    ]
    mark_labels = ["recovery not clearly better\nthan naive (no loss)", "cell fails",
                   "offset equals link delay"]
    groups = (
        (arm_handles, arm_labels, "Color and marker: tracker", 0.075),
        (loss_handles, loss_labels, "Line style: link loss", 0.37),
        (mark_handles, mark_labels, "Marks", 0.58),
    )
    for handles, labels, title, x in groups:
        leg = fig.legend(handles, labels, title=title, loc="upper left",
                         bbox_to_anchor=(x, y), alignment="left", handlelength=2.6,
                         labelspacing=0.5, borderaxespad=0, handletextpad=0.7)
        leg.get_title().set_color(pal["textSecondary"])
        leg.get_title().set_fontsize(8.5)


def make_main_figure(doc, u_table, pal, path):
    styles = arm_styles(pal)
    targets = [t for t in TARGET_ORDER if any(k[0] == t for k in u_table)]
    delays = sorted({k[1] for k in u_table})
    losses = sorted({k[2] for k in u_table})
    offsets = sorted({k[3] for k in u_table})
    if not (targets and delays and losses and offsets):
        raise FigureError("summary.json has no usable slice U cells")
    pad = max(6.0, 0.05 * (offsets[-1] - offsets[0]))
    xlim = (offsets[0] - pad, offsets[-1] + pad)
    failure_cm = doc["failure_rule"]["p95_m"] * 100.0
    nrows, ncols = len(targets), len(delays)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 2.7 * nrows + 3.7))
    height = fig.get_figheight()
    top_frac = 1.0 - 1.35 / height
    bottom_frac = 2.3 / height
    gs = fig.add_gridspec(nrows, ncols, left=0.075, right=0.985, top=top_frac,
                          bottom=bottom_frac, wspace=0.07, hspace=0.30)
    axes = [[None] * ncols for _ in range(nrows)]
    top, bottom = failure_cm, 0.0
    for r, target in enumerate(targets):
        for c, delay in enumerate(delays):
            ax = fig.add_subplot(gs[r, c], sharex=axes[0][0] if (r or c) else None,
                                 sharey=axes[0][0] if (r or c) else None)
            axes[r][c] = ax
            if not u_series(u_table, target, delay, losses[0]):
                ax.text(0.5, 0.5, "no cells in this run", transform=ax.transAxes, ha="center",
                        va="center", fontsize=9, color=pal["textSecondary"])
                continue
            t, b = draw_u_panel(ax, u_table, target, delay, losses, styles, pal, failure_cm,
                                xlim)
            top, bottom = max(top, t), min(bottom, b)
            ax.set_xticks(offset_ticks(xlim))
            if r == 0:
                ax.set_title(f"Link delay {fmt_g(delay)} ms", loc="left")
            if r == nrows - 1:
                ax.set_xlabel("Sensor clock offset (ms)")
            else:
                ax.tick_params(labelbottom=False)
            if c > 0:
                ax.tick_params(labelleft=False)
    axes[0][0].set_ylim(min(0.0, bottom * 1.05), top * 1.08)
    for r, target in enumerate(targets):
        axes[r][0].set_ylabel(f"{TARGET_NAMES[target]}\np95 tracking error (cm)", fontsize=9)
    axes[0][0].annotate(f"failure line ({fmt_g(failure_cm)} cm)", xy=(0.02, failure_cm),
                        xycoords=("axes fraction", "data"), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom", fontsize=8,
                        color=pal["textSecondary"])
    fig.text(0.075, 1.0 - 0.25 / height, "How much sensor clock offset can the capture-time tracker tolerate?",
             fontsize=12, color=pal["text"], ha="left", va="top")
    fig.text(0.075, 1.0 - 0.72 / height,
             "Negative offset: the sensor clock is behind, so stamps are early. Fixed offset, "
             "no drift, no sync.", fontsize=9, color=pal["textSecondary"], ha="left", va="top")
    main_legends(fig, styles, pal, losses, y=1.6 / height)
    fig.text(0.075, 0.28 / height, footnote(n_seeds_of(doc), True), fontsize=8,
             color=pal["textSecondary"], ha="left", va="bottom", linespacing=1.4)
    save(fig, path)


# ---------------------------------------------------------------- figure 2: slice N


def paired_diff(per_seed: dict, key: str, base_key: str, n_seeds: int) -> tuple:
    """Median of per-seed recovery p95 minus the same seed's perfect-clock p95, in cm."""
    diffs = []
    for seed in range(n_seeds):
        a = per_seed.get((key, seed))
        b = per_seed.get((base_key, seed))
        if a is None or b is None:
            raise FigureError(f"per_seed.csv.gz is missing seed {seed} for a cell used in "
                              "the sync figure")
        diffs.append((a - b) * 100.0)
    return median_interval(diffs)


def perfect_clock_keys(u_cells: list[dict], noise: float) -> dict:
    """Map (target, delay ms, loss) to the cell key of the zero-offset unsynced cell."""
    out = {}
    for cell in u_cells:
        spec = cell["cell"]
        if abs(spec["noise_std_m"] - noise) > TOL or abs(spec["clock"]["offset_s"]) > TOL:
            continue
        out[(spec["target"], ms(spec["delay_s"]), round(spec["loss_probability"], 6))] = cell["key"]
    return out


def unsynced_keys(u_cells: list[dict], noise: float) -> dict:
    """Map (target, delay ms, loss, offset ms) to the unsynced cell key."""
    out = {}
    for cell in u_cells:
        spec = cell["cell"]
        if abs(spec["noise_std_m"] - noise) > TOL:
            continue
        out[(spec["target"], ms(spec["delay_s"]), round(spec["loss_probability"], 6),
             ms(spec["clock"]["offset_s"]))] = cell["key"]
    return out


def sync_points(n_cells, u_cells, per_seed, noise, n_seeds) -> dict:
    """Map (target, raw offset ms, drift, interval, asymmetry ms, sync loss) to a paired value."""
    perfect = perfect_clock_keys(u_cells, noise)
    points = {}
    for cell in n_cells:
        spec = cell["cell"]
        if abs(spec["noise_std_m"] - noise) > TOL:
            continue
        clock = spec["clock"]
        base = perfect.get((spec["target"], ms(spec["delay_s"]), round(spec["loss_probability"], 6)))
        if base is None:
            raise FigureError("a slice N cell has no matching zero-offset slice U cell "
                              f"(target {spec['target']}, delay {ms(spec['delay_s'])} ms)")
        sync_loss = spec.get("sync_loss_per_exchange") or 0.0
        key = (spec["target"], ms(clock["offset_s"]), clock["drift_ppm"],
               clock["sync_interval_s"], ms(clock["sync_asymmetry_s"]), round(sync_loss, 6))
        points[key] = paired_diff(per_seed, cell["key"], base, n_seeds)
    return points


def asym_colors(pal: dict, n: int) -> list[str]:
    """Light to dark ramp of the recovery arm color, most negative asymmetry to most positive."""
    mix = horizon_mpl.mix
    base = horizon_mpl.arm_style("recovery", pal)["color"]
    if n == 1:
        return [base]
    stops = [mix(base, pal["surface"], 0.55), base, mix(base, pal["text"], 0.6)]
    return horizon_mpl.ramp(stops, n)


def interval_label(value_s: float) -> str:
    return f"{fmt_g(value_s)} s"


def asym_label(value_ms: float) -> str:
    sign = "+" if value_ms > 0 else ""
    return f"{sign}{fmt_g(value_ms)} ms"


def row_title(target: str, offset_ms: float) -> str:
    sign = "+" if offset_ms > 0 else ""
    return f"{TARGET_NAMES[target]}\nraw clock {sign}{fmt_g(offset_ms)} ms"


def make_sync_figure(doc, u_cells, n_cells, per_seed, noise, pal, path):
    if not n_cells:
        raise FigureError("summary.json has no slice N cells")
    n_seeds = n_seeds_of(doc)
    points = sync_points(n_cells, u_cells, per_seed, noise, n_seeds)
    targets = [t for t in TARGET_ORDER if any(k[0] == t for k in points)]
    offsets = sorted({k[1] for k in points}, reverse=True)
    drifts = sorted({k[2] for k in points})
    intervals = sorted({k[3] for k in points})
    asyms = sorted({k[4] for k in points})
    syncloss = sorted({k[5] for k in points})
    rows = [(t, o) for t in targets for o in offsets if any(k[0] == t and k[1] == o for k in points)]
    colors = asym_colors(pal, len(asyms))
    rec_marker = horizon_mpl.arm_style("recovery", pal)["marker"]
    unsynced = unsynced_keys(u_cells, noise)
    delay_for = {c["cell"]["target"]: ms(c["cell"]["delay_s"]) for c in n_cells}
    loss_for = {c["cell"]["target"]: round(c["cell"]["loss_probability"], 6) for c in n_cells}
    perfect = perfect_clock_keys(u_cells, noise)
    sync_delay = fmt_g(ms(n_cells[0]["cell"]["clock"]["sync_delay_s"]))

    nrows, ncols = len(rows), len(drifts)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 1.95 * nrows + 4.0))
    height = fig.get_figheight()
    gs = fig.add_gridspec(nrows, ncols, left=0.12, right=0.985, top=1.0 - 1.7 / height,
                          bottom=2.3 / height, wspace=0.07, hspace=0.34)
    axes = [[None] * ncols for _ in range(nrows)]
    slots = len(asyms) * len(syncloss)
    step = 0.62 / max(slots, 1) * 1.0
    lows, highs = [0.0, BUDGET_CM], [0.0, BUDGET_CM]
    for r, (target, offset) in enumerate(rows):
        ref = None
        ref_key = unsynced.get((target, delay_for[target], loss_for[target], offset))
        base_key = perfect.get((target, delay_for[target], loss_for[target]))
        if ref_key is not None and base_key is not None:
            ref = paired_diff(per_seed, ref_key, base_key, n_seeds)[0]
            highs.append(ref)
            lows.append(ref)
        for c, drift in enumerate(drifts):
            ax = fig.add_subplot(gs[r, c], sharex=axes[0][0] if (r or c) else None,
                                 sharey=axes[0][0] if (r or c) else None)
            axes[r][c] = ax
            for gi, interval in enumerate(intervals):
                for ai, asym in enumerate(asyms):
                    for li, sl in enumerate(syncloss):
                        value = points.get((target, offset, drift, interval, asym, sl))
                        if value is None:
                            continue
                        med, lo, hi = value
                        lows.append(lo)
                        highs.append(hi)
                        slot = ai * len(syncloss) + li
                        x = gi + (slot - (slots - 1) / 2.0) * step
                        color = colors[ai]
                        ax.errorbar([x], [med], yerr=[[med - lo], [hi - med]], fmt="none",
                                    ecolor=color, elinewidth=0.9, capsize=1.8, capthick=0.9,
                                    zorder=3)
                        ax.plot([x], [med], linestyle="none", marker=rec_marker, markersize=4.6,
                                markerfacecolor=color if li == 0 else pal["surface"],
                                markeredgecolor=color, markeredgewidth=1.0, clip_on=False,
                                zorder=4)
            ax.axhline(0.0, color=pal["lineStrong"], linewidth=0.8, zorder=1.4)
            ax.axhline(BUDGET_CM, color=pal["textSecondary"], linewidth=1.0,
                       linestyle=(0, (4, 2.2)), zorder=1.5)
            if ref is not None:
                ax.axhline(ref, color=pal["lineStrong"], linewidth=1.0, linestyle=(0, (1, 2)),
                           zorder=1.6)
            ax.set_xlim(-0.55, len(intervals) - 0.45)
            ax.set_xticks(range(len(intervals)))
            ax.set_xticklabels([interval_label(v) for v in intervals])
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(f"Clock drift {fmt_g(drift)} ppm", loc="left")
            if r == nrows - 1:
                ax.set_xlabel("Sync interval")
            else:
                ax.tick_params(labelbottom=False)
            if c > 0:
                ax.tick_params(labelleft=False)
    span = max(highs) - min(lows)
    axes[0][0].set_ylim(min(lows) - 0.06 * span, max(highs) + 0.10 * span)
    for r, (target, offset) in enumerate(rows):
        axes[r][0].set_ylabel(f"{row_title(target, offset)}\nextra p95 (cm)", fontsize=8.5)
    fig.text(0.12, 1.0 - 0.25 / height, "Does radio clock sync keep the extra tracking error small? "
             "Recovery tracker", fontsize=12, color=pal["text"], ha="left", va="top")
    fig.text(0.12, 1.0 - 0.72 / height,
             "Recovery p95 minus its own p95 on perfect clocks (same seeds), grouped by sync "
             f"interval.\nNTP-style sync over the simulated radio link, {sync_delay} ms one-way "
             "sync delay.",
             fontsize=9, color=pal["textSecondary"], ha="left", va="top", linespacing=1.35)
    sync_legends(fig, pal, colors, asyms, syncloss, ref_shown=True, y=1.6 / height)
    fig.text(0.12, 0.28 / height, footnote(
        n_seeds, False, "Points are the median of paired per-seed differences."),
        fontsize=8, color=pal["textSecondary"], ha="left", va="bottom", linespacing=1.4)
    save(fig, path)


def sync_legends(fig, pal, colors, asyms, syncloss, ref_shown, y):
    asym_handles = [Line2D([0], [0], linestyle="none", marker="o", markersize=6,
                           markerfacecolor=c, markeredgecolor=c) for c in colors]
    asym_labels = [asym_label(a) for a in asyms]
    loss_handles = [
        Line2D([0], [0], linestyle="none", marker="o", markersize=6,
               markerfacecolor=pal["textSecondary"] if i == 0 else pal["surface"],
               markeredgecolor=pal["textSecondary"], markeredgewidth=1.0)
        for i in range(len(syncloss))
    ]
    loss_labels = ["no sync loss" if p == 0 else f"{fmt_g(round(p * 100, 6))} % of exchanges lost"
                   for p in syncloss]
    line_handles = [Line2D([0], [0], color=pal["textSecondary"], linewidth=1.0,
                           linestyle=(0, (4, 2.2)))]
    line_labels = [f"{fmt_g(BUDGET_CM)} cm budget"]
    if ref_shown:
        line_handles.append(Line2D([0], [0], color=pal["lineStrong"], linewidth=1.0,
                                   linestyle=(0, (1, 2))))
        line_labels.append("no sync at all (raw offset)")
    groups = (
        (asym_handles, asym_labels, "Color: path asymmetry (up minus down)", 0.12),
        (loss_handles, loss_labels, "Marker: sync loss", 0.46),
        (line_handles, line_labels, "Reference lines", 0.74),
    )
    for handles, labels, title, x in groups:
        leg = fig.legend(handles, labels, title=title, loc="upper left",
                         bbox_to_anchor=(x, y), alignment="left", handlelength=1.8,
                         labelspacing=0.5, borderaxespad=0, handletextpad=0.7)
        leg.get_title().set_color(pal["textSecondary"])
        leg.get_title().set_fontsize(8.5)


# ---------------------------------------------------------------- output


def save(fig, path: Path):
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


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
    u_cells, n_cells = split_slices(doc)
    if not u_cells:
        raise FigureError("summary.json has no slice U cells")
    noise = pick_noise(u_cells)
    per_seed = load_per_seed(run_dir)
    pal = horizon_mpl.apply_style("light")
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / MAIN_NAME
    sync_path = output_dir / SYNC_NAME
    make_main_figure(doc, index_u(u_cells, noise), pal, main_path)
    make_sync_figure(doc, u_cells, n_cells, per_seed, noise, pal, sync_path)
    written = [main_path, sync_path]
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
        print(f"make_s6_clocks_figure: error: {err}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
