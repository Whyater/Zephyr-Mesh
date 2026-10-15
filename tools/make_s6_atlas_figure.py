"""Draw the recovery atlas figures from an S6 atlas run directory.

Usage (from the repo root):

    PYTHONPATH=. ./.venv/bin/python tools/make_s6_atlas_figure.py runs/s6-atlas \
        [--output-dir DIR]

The run directory must hold atlas.json and summary.json. The output directory
defaults to the run directory. Files written:

    atlas_overview.png            figure 1, central stratum
    atlas_which_method.png        figure 2, central stratum
    atlas_overview_<stratum>.png  figure 1 again for every other stratum in the run

Question of figure 1: for each target motion, how much one-way delay does each tracker
arm tolerate at each packet-loss level? One shared color scale; red cells mark arms that
fail even at 0 ms and the darkest cells mark arms that never break within the tested
range.

Question of figure 2: for each target motion and loss level, which arm tolerates the
most delay, and is there any cell where no arm tolerates even 0 ms?

Every plotted number is read from atlas.json (and summary.json for the tie-break by
median p95 error). The script only converts seconds to milliseconds and fractions
to percent for labels. Colors come from design/horizon_tokens.json through
tools/horizon_mpl.py. The same input files always produce the same PNG bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from tools import horizon_mpl  # noqa: E402

ATLAS_SCHEMA = "1.0"
ATLAS_EXPERIMENT = "s6-atlas"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sim" / "estimation" / "atlas_schema.json"
CENTRAL = "central"
DPI = 200
PANEL_COLUMNS = 3
GROUP_GAP = 0.35  # extra horizontal space between the random and burst loss groups

VARIANT_ORDER = ("cv_0.5", "cv_1", "cv_2", "ct_r0.5", "ct_r1", "ct_r3", "jink", "figure8",
                 "start_stop")
ARM_ORDER = ("naive", "recovery", "imm", "ct", "adaptive", "retro", "recovery_gated", "imm_gated", "ct_gated")
ARM_LABELS = {
    "naive": "Naive",
    "recovery": "Recovery",
    "imm": "IMM",
    "ct": "Turn-rate (CT)",
    "adaptive": "Adaptive",
    "retro": "Retro",
    "recovery_gated": "Recovery gated",
    "imm_gated": "IMM gated",
    "ct_gated": "CT gated",
}


class FigureError(Exception):
    """Raised for unusable input; main() turns it into a clear exit message."""


# ---------------------------------------------------------------- input


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            doc = json.load(handle)
    except OSError as err:
        raise FigureError(f"cannot read {path}: {err}") from err
    except json.JSONDecodeError as err:
        raise FigureError(f"{path} is not valid JSON: {err}") from err
    if not isinstance(doc, dict):
        raise FigureError(f"{path} does not hold a JSON object")
    return doc


def load_atlas(path: Path) -> dict:
    doc = _read_json(path)
    if doc.get("schema") != ATLAS_SCHEMA or doc.get("experiment") != ATLAS_EXPERIMENT:
        raise FigureError(
            f"unexpected schema {doc.get('schema')!r} / experiment {doc.get('experiment')!r}; "
            f"this script reads schema {ATLAS_SCHEMA!r} of {ATLAS_EXPERIMENT!r}"
        )
    for key in ("entries", "seeds", "strata", "failure_rule"):
        if key not in doc:
            raise FigureError(f"atlas.json is missing the {key!r} field")
    if SCHEMA_PATH.exists():
        try:
            import jsonschema

            with SCHEMA_PATH.open(encoding="utf-8") as handle:
                jsonschema.validate(doc, json.load(handle))
        except ImportError:
            pass
        except jsonschema.ValidationError as err:
            raise FigureError(f"atlas.json does not match its schema: {err.message}") from err
    if not doc["entries"]:
        raise FigureError("atlas.json has no entries")
    return doc


def load_summary(path: Path) -> dict | None:
    """summary.json is only used for the tie-break; a missing file is not fatal."""
    if not path.exists():
        return None
    return _read_json(path)


def index_p95(summary: dict | None) -> dict:
    """Map (stratum, variant, loss model, loss, delay, arm) to the median p95 in meters."""
    table: dict = {}
    if summary is None:
        return table
    for cell in summary.get("cells", []):
        info = cell.get("cell", {})
        for arm, stats in cell.get("arms", {}).items():
            table[(cell.get("stratum"), cell.get("variant"), info.get("loss_model"),
                   round(float(info.get("loss_probability", 0.0)), 6),
                   round(float(info.get("delay_s", 0.0)), 6), arm)] = stats.get("p95_median_m")
    return table


def tested_delays(summary: dict | None) -> list[float]:
    if summary is None:
        return []
    return sorted({round(float(c["cell"]["delay_s"]), 6) for c in summary.get("cells", [])
                   if "cell" in c and "delay_s" in c["cell"]})


# ---------------------------------------------------------------- reading entries


def censor_threshold(text: str) -> float:
    """'>0.3' -> 0.3 (seconds)."""
    return float(text.lstrip(">"))


def tolerance_rank(entry: dict) -> tuple[float, float]:
    """(rank, delay) for one atlas entry; larger rank means more delay tolerated.

    rank -1: fails at zero. Right-censored entries rank just above their threshold, so
    they beat every finite value. The delay is the last delay the arm tolerated.
    """
    tol = entry["max_tolerated_delay_s"]
    if tol["fails_at_zero"]:
        return (-1.0, 0.0)
    if tol["censored"] is not None:
        threshold = censor_threshold(tol["censored"])
        return (threshold + 1e-9, threshold)
    return (float(tol["value"]), float(tol["value"]))


def loss_key(entry: dict) -> tuple[int, float]:
    """Random loss columns first, then burst; each sorted by probability."""
    return (1 if entry["loss_model"] == "burst" else 0, round(entry["loss_probability"], 6))


def variant_sort_key(name: str) -> tuple[int, str]:
    return (VARIANT_ORDER.index(name) if name in VARIANT_ORDER else len(VARIANT_ORDER), name)


def arm_sort_key(name: str) -> tuple[int, str]:
    return (ARM_ORDER.index(name) if name in ARM_ORDER else len(ARM_ORDER), name)


def group_entries(atlas: dict) -> dict:
    """stratum -> {(variant, loss_key): {arm: entry}}."""
    out: dict = {}
    for entry in atlas["entries"]:
        slot = out.setdefault(entry["stratum"], {})
        slot.setdefault((entry["variant"], loss_key(entry)), {})[entry["arm"]] = entry
    return out


def stratum_axes(groups: dict) -> tuple[list[str], list[tuple[int, float]], list[str]]:
    """Sorted variants, loss keys and arms present in one stratum's groups."""
    variants = sorted({v for v, _ in groups}, key=variant_sort_key)
    losses = sorted({lk for _, lk in groups})
    arms = sorted({a for arms in groups.values() for a in arms}, key=arm_sort_key)
    return variants, losses, arms


# ---------------------------------------------------------------- labels and colors


def fmt_g(value: float) -> str:
    return f"{value:g}"


def ms(seconds: float) -> float:
    return round(seconds * 1000.0, 6)


def variant_title(name: str) -> str:
    if name.startswith("cv_"):
        return f"Straight line, {name[3:]} m/s"
    if name.startswith("ct_r"):
        return f"Circle, {name[4:]} m radius"
    return {"jink": "Random jinking", "figure8": "Figure eight",
            "start_stop": "Start and stop"}.get(name, name.replace("_", " "))


def arm_label(name: str) -> str:
    return ARM_LABELS.get(name, name.replace("_", " ").capitalize())


def arm_colors(pal: dict) -> dict[str, str]:
    """One fixed color per arm from horizon_mpl.arm_style. Red and amber are never arm colors."""
    return {arm: horizon_mpl.arm_style(arm, pal)["color"] for arm in ARM_ORDER}


def luminance(hex_color: str) -> float:
    r, g, b = (c for c in horizon_mpl._to_rgb(hex_color))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def text_on(hex_color: str, pal: dict) -> str:
    return pal["onAlert"] if luminance(hex_color) < 0.30 else pal["text"]


def loss_columns(losses: list[tuple[int, float]]) -> list[float]:
    """x position of each loss column, with a gap between the random and burst groups."""
    xs, offset, seen_burst = [], 0.0, False
    for i, (is_burst, _) in enumerate(losses):
        if is_burst and not seen_burst:
            seen_burst = True
            offset = GROUP_GAP if i > 0 else 0.0
        xs.append(i + offset)
    return xs


def delay_bins(atlas: dict, summary: dict | None) -> tuple[list[float], str]:
    """Finite tolerated-delay bins (ms) over the whole run, and the label of the beyond bin."""
    finite: set[float] = set()
    beyond = None
    for entry in atlas["entries"]:
        tol = entry["max_tolerated_delay_s"]
        if tol["value"] is not None:
            finite.add(ms(tol["value"]))
        if tol["censored"] is not None:
            beyond = tol["censored"] if beyond is None else max(
                beyond, tol["censored"], key=censor_threshold)
    delays = tested_delays(summary)
    if len(delays) > 1:
        finite.update(ms(d) for d in delays[:-1])
    label = "beyond tested range"
    if beyond is not None:
        label = f">{fmt_g(ms(censor_threshold(beyond)))} ms"
    elif delays:
        label = f">{fmt_g(ms(delays[-1]))} ms"
    return sorted(finite), label


def bin_colors(pal: dict, n: int) -> list[str]:
    """Sequential Horizon ramp, light to dark; the last color is the beyond bin."""
    mix = horizon_mpl.mix
    stops = [mix(pal["accent"], pal["surface"], 0.82), pal["accent"],
             mix(pal["accent"], pal["text"], 0.72)]
    return horizon_mpl.ramp(stops, n)


def stratum_description(atlas: dict, stratum: str) -> str:
    info = atlas["strata"].get(stratum, {})
    parts = []
    if "rate_hz" in info:
        parts.append(f"{fmt_g(info['rate_hz'])} Hz measurements")
    if info.get("noise_model") == "student_t":
        parts.append(f"heavy-tailed noise, {fmt_g(round(info['noise_std_m'] * 100, 3))} cm std")
    elif "noise_std_m" in info:
        parts.append(f"Gaussian noise {fmt_g(round(info['noise_std_m'] * 100, 3))} cm")
    if info.get("jitter_rule") not in (None, "none"):
        parts.append("delay jitter")
    return ", ".join(parts)


def footnote(atlas: dict, stratum: str, with_tiebreak: bool, width: int = 125) -> str:
    rule = atlas["failure_rule"]["per_seed"]
    n_seeds = len(atlas["seeds"])
    lines = [
        f"Simulated link and sensor, {n_seeds} seeds per cell, median p95 with 95 % interval. "
        f"A seed fails above {fmt_g(round(rule['p95_error_m'] * 100, 3))} cm p95 or "
        f"{fmt_g(round(rule['max_error_m'] * 100, 3))} cm peak error; "
        "a cell fails when at least 50 % of seeds fail (rule proposed, not yet adopted).",
        f"Stratum: {stratum_description(atlas, stratum)}. Maximum tolerated delay is the "
        "largest tested delay before the first failing delay. Not flight data.",
    ]
    if with_tiebreak:
        lines.append("Ties in tolerated delay are broken by the smaller median p95 error "
                     "at that delay.")
    return "\n".join(textwrap.fill(line, width=width) for line in lines)


# ---------------------------------------------------------------- figure 1


def draw_overview_panel(ax, groups, variant, losses, arms, xs, colors, bins, pal, show_x):
    """One heatmap panel: rows are arms, columns are loss conditions."""
    fail_color = pal["alert"]
    n_bins = len(bins)
    for r, arm in enumerate(arms):
        for (lk, x) in zip(losses, xs):
            entry = groups.get((variant, lk), {}).get(arm)
            if entry is None:
                ax.add_patch(Rectangle((x - 0.5, r - 0.5), 1, 1, facecolor=pal["background"],
                                       edgecolor=pal["surface"], linewidth=1.4))
                continue
            tol = entry["max_tolerated_delay_s"]
            marker, label = None, None
            if tol["fails_at_zero"]:
                face, marker = fail_color, "×"
            elif tol["censored"] is not None:
                face, marker = colors[n_bins], ">"
            else:
                value_ms = ms(tol["value"])
                idx = bins.index(value_ms) if value_ms in bins else n_bins - 1
                face, label = colors[idx], fmt_g(value_ms)
            ax.add_patch(Rectangle((x - 0.5, r - 0.5), 1, 1, facecolor=face,
                                   edgecolor=pal["surface"], linewidth=1.4))
            text = marker or label
            ax.text(x, r + 0.02, text, ha="center", va="center",
                    fontsize=9 if marker else 7, color=text_on(face, pal))
    ax.set_xlim(-0.5, xs[-1] + 0.5)
    ax.set_ylim(len(arms) - 0.5, -0.5)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([arm_label(a) for a in arms], fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([fmt_g(round(p * 100, 3)) for _, p in losses], fontsize=8)
    ax.tick_params(length=0, pad=3)
    ax.tick_params(axis="x", labelbottom=show_x)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(variant_title(variant), loc="left", fontsize=10)


def group_labels(ax, losses, xs):
    """Text under the loss ticks naming the random and burst groups."""
    spans: dict[int, list[float]] = {}
    for (is_burst, _), x in zip(losses, xs):
        spans.setdefault(is_burst, []).append(x)
    names = {0: "random loss (%)", 1: "burst loss (%)"}
    for is_burst, positions in spans.items():
        lo, hi = min(positions) - 0.4, max(positions) + 0.4
        ax.annotate("", xy=(lo, -0.22), xytext=(hi, -0.22),
                    xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                    arrowprops={"arrowstyle": "-", "color": ax.spines["left"].get_edgecolor(),
                                "linewidth": 0.8, "shrinkA": 0, "shrinkB": 0})
        ax.text((lo + hi) / 2, -0.30, names[is_burst], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=ax.xaxis.label.get_color())


def draw_colorbar(ax, bins, beyond_label, colors, pal, has_fail):
    """Shared discrete colorbar: one swatch per tolerated-delay bin, plus the failure swatch."""
    n = len(bins) + 1
    for i in range(n):
        ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=colors[i], edgecolor=pal["surface"],
                               linewidth=1.4))
        text = beyond_label if i == n - 1 else fmt_g(bins[i])
        ax.text(i + 0.5, -0.18, text, ha="center", va="top", fontsize=8.5,
                color=pal["textSecondary"])
    ax.text(n - 0.5, 0.5, ">", ha="center", va="center", fontsize=11,
            color=text_on(colors[n - 1], pal))
    right = n
    if has_fail:
        x0 = n + 0.8
        ax.add_patch(Rectangle((x0, 0), 1, 1, facecolor=pal["alert"], edgecolor=pal["surface"],
                               linewidth=1.4))
        ax.text(x0 + 0.5, 0.5, "×", ha="center", va="center", fontsize=11,
                color=pal["onAlert"])
        ax.text(x0 + 0.5, -0.18, "fails at 0 ms", ha="center", va="top", fontsize=8.5,
                color=pal["alert"])
        right = x0 + 1
    ax.set_xlim(0, right)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return right


def make_overview(atlas, summary, stratum, groups, pal, path):
    variants, losses, arms = stratum_axes(groups)
    xs = loss_columns(losses)
    bins, beyond_label = delay_bins(atlas, summary)
    colors = bin_colors(pal, len(bins) + 1)
    has_fail = any(e["max_tolerated_delay_s"]["fails_at_zero"] for e in atlas["entries"])
    n_rows = math.ceil(len(variants) / PANEL_COLUMNS)
    width = 12.0
    height = 1.1 + 2.1 * n_rows + 2.7
    fig = plt.figure(figsize=(width, height))
    top, bottom = 1 - 1.1 / height, 2.7 / height
    gs = fig.add_gridspec(n_rows, PANEL_COLUMNS, left=0.085, right=0.985, top=top,
                          bottom=bottom, wspace=0.14, hspace=0.38)
    for i, variant in enumerate(variants):
        ax = fig.add_subplot(gs[i // PANEL_COLUMNS, i % PANEL_COLUMNS])
        show_x = i + PANEL_COLUMNS >= len(variants)
        draw_overview_panel(ax, groups, variant, losses, arms, xs, colors, bins, pal, show_x)
        if show_x:
            group_labels(ax, losses, xs)
        if i % PANEL_COLUMNS:
            ax.tick_params(labelleft=False)
    desc = stratum_description(atlas, stratum)
    fig.text(0.085, 1 - 0.22 / height, "How much delay does each tracker tolerate, by target "
             "motion and packet loss?", fontsize=12.5, color=pal["text"], ha="left", va="top")
    fig.text(0.085, 1 - 0.62 / height, f"Maximum tolerated one-way delay (ms), {desc}"
             + ("" if stratum == CENTRAL else f" (stratum: {stratum})"),
             fontsize=9, color=pal["textSecondary"], ha="left", va="top")
    cbar = fig.add_axes([0.085, 1.2 / height, 0.45, 0.3 / height])
    draw_colorbar(cbar, bins, beyond_label, colors, pal, has_fail)
    fig.text(0.085, 1.65 / height, "Maximum tolerated delay (ms)",
             fontsize=8.5, color=pal["textSecondary"], ha="left", va="bottom")
    fig.text(0.085, 0.15 / height, footnote(atlas, stratum, with_tiebreak=False), fontsize=8,
             color=pal["textSecondary"], ha="left", va="bottom", linespacing=1.35)
    save(fig, path)


# ---------------------------------------------------------------- figure 2


def pick_winner(atlas_arms: dict, p95_table: dict, stratum, variant, loss_entry_key):
    """Arm with the largest tolerated delay; ties go to the smaller median p95.

    Returns (arm, entry) or (None, None) when no arm tolerates 0 ms.
    """
    best, best_key = None, None
    for arm in sorted(atlas_arms, key=arm_sort_key):
        entry = atlas_arms[arm]
        rank, delay = tolerance_rank(entry)
        if rank < 0:
            continue
        p95 = p95_table.get((stratum, variant, entry["loss_model"],
                             round(entry["loss_probability"], 6), round(delay, 6), arm))
        key = (rank, -(p95 if p95 is not None else math.inf))
        if best_key is None or key > best_key:
            best, best_key = arm, key
    if best is None:
        return None, None
    return best, atlas_arms[best]


def delay_text(entry: dict) -> str:
    tol = entry["max_tolerated_delay_s"]
    if tol["censored"] is not None:
        return f">{fmt_g(ms(censor_threshold(tol['censored'])))} ms"
    return f"{fmt_g(ms(tol['value']))} ms"


def make_which_method(atlas, summary, stratum, groups, pal, path):
    p95_table = index_p95(summary)
    variants, losses, arms = stratum_axes(groups)
    xs = loss_columns(losses)
    colors = arm_colors(pal)
    fig_w = 11.6
    cell_h = 0.62
    height = cell_h * len(variants) + 3.5
    fig = plt.figure(figsize=(fig_w, height))
    ax = fig.add_axes([0.14, 1.95 / height, 0.845, cell_h * len(variants) / height])
    winners_used: list[str] = []
    for r, variant in enumerate(variants):
        for lk, x in zip(losses, xs):
            arms_here = groups.get((variant, lk), {})
            arm, entry = pick_winner(arms_here, p95_table, stratum, variant, lk)
            if not arms_here:
                face = pal["background"]
                ax.add_patch(Rectangle((x - 0.5, r - 0.5), 1, 1, facecolor=face,
                                       edgecolor=pal["surface"], linewidth=2))
                continue
            if arm is None:
                ax.add_patch(Rectangle((x - 0.5, r - 0.5), 1, 1, facecolor=pal["surface"],
                                       edgecolor=pal["alert"], linewidth=1.2))
                ax.text(x, r - 0.12, "none", ha="center", va="center", fontsize=9,
                        color=pal["alert"])
                ax.text(x, r + 0.17, "fails at 0 ms", ha="center", va="center", fontsize=7,
                        color=pal["alert"])
                continue
            winners_used.append(arm)
            face = colors.get(arm, pal["textSecondary"])
            ax.add_patch(Rectangle((x - 0.5, r - 0.5), 1, 1, facecolor=face,
                                   edgecolor=pal["surface"], linewidth=2))
            ax.text(x, r - 0.12, arm_label(arm), ha="center", va="center", fontsize=9,
                    color=text_on(face, pal))
            ax.text(x, r + 0.17, delay_text(entry), ha="center", va="center", fontsize=7.5,
                    color=text_on(face, pal))
    ax.set_xlim(-0.5, xs[-1] + 0.5)
    ax.set_ylim(len(variants) - 0.5, -0.5)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([variant_title(v) for v in variants], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_xticks(xs)
    ax.set_xticklabels([fmt_g(round(p * 100, 3)) for _, p in losses], fontsize=9)
    ax.tick_params(length=0, pad=4)
    ax.grid(False)
    for side in ("left", "bottom", "top"):
        ax.spines[side].set_visible(False)
    spans: dict[int, list[float]] = {}
    for (is_burst, _), x in zip(losses, xs):
        spans.setdefault(is_burst, []).append(x)
    for is_burst, positions in spans.items():
        ax.annotate("random loss (%)" if not is_burst else "burst loss (%)",
                    xy=((min(positions) + max(positions)) / 2, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, 24),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5,
                    color=pal["textSecondary"])
    ax.xaxis.set_label_position("top")
    fig.text(0.14, 1 - 0.22 / height, "Which method tolerates the most delay, and where does none?",
             fontsize=12.5, color=pal["text"], ha="left", va="top")
    desc = stratum_description(atlas, stratum)
    fig.text(0.14, 1 - 0.62 / height, f"Arm with the largest maximum tolerated delay (ms), {desc}",
             fontsize=9, color=pal["textSecondary"], ha="left", va="top")
    # Legend: one swatch per arm that wins somewhere, plus the "no arm" marker.
    present = [a for a in arms if a in winners_used]
    leg = fig.add_axes([0.14, 1.5 / height, 0.845, 0.3 / height])
    x_cursor = 0.0
    for arm in present:
        leg.add_patch(Rectangle((x_cursor, 0), 0.5, 1, facecolor=colors.get(arm, pal["textSecondary"]),
                                edgecolor="none"))
        leg.text(x_cursor + 0.62, 0.5, arm_label(arm), va="center", ha="left", fontsize=9,
                 color=pal["text"])
        x_cursor += 0.62 + 0.115 * len(arm_label(arm)) + 0.55
    leg.add_patch(Rectangle((x_cursor, 0), 0.5, 1, facecolor=pal["surface"],
                            edgecolor=pal["alert"], linewidth=1.2))
    leg.text(x_cursor + 0.62, 0.5, "no arm tolerates 0 ms", va="center", ha="left", fontsize=9,
             color=pal["alert"])
    leg.set_xlim(0, max(x_cursor + 3.2, 12))
    leg.set_ylim(0, 1)
    leg.axis("off")
    fig.text(0.14, 0.15 / height, footnote(atlas, stratum, with_tiebreak=True, width=112)
             + "\nCell text: arm, then its maximum tolerated delay.", fontsize=8,
             color=pal["textSecondary"], ha="left", va="bottom", linespacing=1.35)
    save(fig, path)


# ---------------------------------------------------------------- output


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_manifest(output_dir: Path, written: list[Path]) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = _read_json(manifest_path)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    for path in written:
        outputs[path.name] = sha256_of(path)
    manifest["outputs"] = outputs
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")


def build(run_dir: Path, output_dir: Path) -> list[Path]:
    atlas = load_atlas(run_dir / "atlas.json")
    summary = load_summary(run_dir / "summary.json")
    by_stratum = group_entries(atlas)
    if CENTRAL not in by_stratum:
        raise FigureError(f"atlas.json has no entries in the {CENTRAL!r} stratum")
    pal = horizon_mpl.apply_style("light")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    path = output_dir / "atlas_overview.png"
    make_overview(atlas, summary, CENTRAL, by_stratum[CENTRAL], pal, path)
    written.append(path)
    path = output_dir / "atlas_which_method.png"
    make_which_method(atlas, summary, CENTRAL, by_stratum[CENTRAL], pal, path)
    written.append(path)
    for stratum in sorted(s for s in by_stratum if s != CENTRAL):
        path = output_dir / f"atlas_overview_{stratum}.png"
        make_overview(atlas, summary, stratum, by_stratum[stratum], pal, path)
        written.append(path)
    update_manifest(output_dir, written)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="run directory holding atlas.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where to write the PNGs (default: the run directory)")
    args = parser.parse_args(argv)
    try:
        written = build(args.run_dir, args.output_dir or args.run_dir)
    except FigureError as err:
        print(f"make_s6_atlas_figure: error: {err}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
