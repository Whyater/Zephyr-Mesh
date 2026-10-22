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
most delay? A cell names an arm only when that arm's tolerated delay is strictly larger
than every other arm's. When two or more arms share the best tolerated delay the cell
says "Tie" in a neutral color and lists them. Cells where no arm tolerates even 0 ms keep
a red outline.

Every plotted number is read from atlas.json (summary.json supplies the tested delays and
the tracker settings quoted in the footnote). The script only converts seconds to
milliseconds and fractions to percent for labels. Colors come from tools/design/horizon_tokens.json through
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
    """summary.json supplies tested delays and tracker settings; a missing file is not fatal."""
    if not path.exists():
        return None
    return _read_json(path)


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
        parts.append(f"delay jitter ({info['jitter_rule']})")
    return ", ".join(parts)


def context_lines(atlas: dict, summary: dict | None, stratum: str) -> list[str]:
    """Plain-language notes on what differs in this stratum from the central one.

    Everything comes from the strata definitions, the run config and burst_mean_s.
    """
    info = atlas["strata"].get(stratum, {})
    base = atlas["strata"].get(CENTRAL, {})
    config = (summary or {}).get("config", {})
    lines = []
    if stratum != CENTRAL and "noise_std_m" in info and info.get("noise_std_m") != base.get(
            "noise_std_m"):
        assumed = config.get("tracker_measurement_std_m", base.get("noise_std_m"))
        if assumed is not None:
            lines.append(f"The tracker still assumes {fmt_g(round(assumed * 100, 3))} cm "
                         "measurement noise in this stratum, whatever the sensor noise is.")
    if stratum != CENTRAL and "rate_hz" in info and info.get("rate_hz") != base.get("rate_hz"):
        burst = (summary or {}).get("burst_mean_s")
        if burst is not None:
            lines.append(f"Burst loss uses {fmt_g(burst)} s mean bursts at every measurement rate.")
    if info.get("jitter_rule") not in (None, "none"):
        lines.append(f"Delay jitter: {info['jitter_rule'].replace('_s', '')}, "
                     "so the jitter size grows with the delay.")
    return lines


def footnote(atlas: dict, stratum: str, summary: dict | None = None, width: int = 125,
             extra: list[str] | None = None) -> str:
    rule = atlas["failure_rule"]["per_seed"]
    n_seeds = len(atlas["seeds"])
    lines = [
        f"Simulated link and sensor, {n_seeds} seeds per cell. "
        f"A seed fails above {fmt_g(round(rule['p95_error_m'] * 100, 3))} cm p95 or "
        f"{fmt_g(round(rule['max_error_m'] * 100, 3))} cm peak error; "
        "a cell fails when at least 50\u00a0% of seeds fail (rule proposed, not yet adopted).",
        f"Stratum: {stratum_description(atlas, stratum)}. Cells show the maximum tolerated "
        "delay, the largest tested delay before the first failing delay; no p95 error is "
        "plotted (p95 values are in summary.json). Not flight data.",
    ]
    lines.extend(context_lines(atlas, summary, stratum))
    lines.extend(extra or [])
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
    foot = footnote(atlas, stratum, summary)
    foot_h = 0.15 + 0.155 * (foot.count("\n") + 1)
    cbar_y = max(1.2, foot_h + 0.3)  # inches from the bottom
    bottom_in = cbar_y + 1.5
    height = 1.1 + 2.1 * n_rows + bottom_in
    fig = plt.figure(figsize=(width, height))
    top, bottom = 1 - 1.1 / height, bottom_in / height
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
    cbar = fig.add_axes([0.085, cbar_y / height, 0.45, 0.3 / height])
    draw_colorbar(cbar, bins, beyond_label, colors, pal, has_fail)
    fig.text(0.085, (cbar_y + 0.45) / height, "Maximum tolerated delay (ms)",
             fontsize=8.5, color=pal["textSecondary"], ha="left", va="bottom")
    fig.text(0.085, 0.15 / height, foot, fontsize=8,
             color=pal["textSecondary"], ha="left", va="bottom", linespacing=1.35)
    save(fig, path)


# ---------------------------------------------------------------- figure 2


NONE, WINNER, TIE = "none", "winner", "tie"
SHORT_LABELS = {
    "naive": "Naive", "recovery": "Recov.", "imm": "IMM", "ct": "CT", "adaptive": "Adapt.",
    "retro": "Retro", "recovery_gated": "Rec-g", "imm_gated": "IMM-g", "ct_gated": "CT-g",
}
TIE_LISTED_MAX = 3  # more tied arms than this are summarised as a count


def short_label(name: str) -> str:
    return SHORT_LABELS.get(name, arm_label(name))


def classify_cell(atlas_arms: dict) -> tuple[str, list[str], dict | None]:
    """Decide what one cell shows: (kind, arms, entry).

    NONE: no arm tolerates 0 ms. WINNER: exactly one arm has the largest tolerated delay.
    TIE: two or more arms share the largest tolerated delay; arms lists all of them in arm
    order and entry is one of their entries (the delay is the same for all).
    """
    ranked = {}
    for arm in sorted(atlas_arms, key=arm_sort_key):
        rank, _ = tolerance_rank(atlas_arms[arm])
        if rank >= 0:
            ranked[arm] = rank
    if not ranked:
        return NONE, [], None
    best = max(ranked.values())
    top = [arm for arm, rank in ranked.items() if rank == best]
    kind = WINNER if len(top) == 1 else TIE
    return kind, top, atlas_arms[top[0]]


def tie_text(arms: list[str]) -> str:
    if len(arms) > TIE_LISTED_MAX:
        return f"Tie ({len(arms)} arms)"
    return "Tie: " + ", ".join(short_label(a) for a in arms)


def delay_text(entry: dict) -> str:
    tol = entry["max_tolerated_delay_s"]
    if tol["censored"] is not None:
        return f">{fmt_g(ms(censor_threshold(tol['censored'])))} ms"
    return f"{fmt_g(ms(tol['value']))} ms"


def cell_counts(classified: list[tuple[str, list[str], dict | None]]) -> dict[str, int]:
    kinds = [k for k, _, _ in classified]
    return {"cells": len(kinds), "none": kinds.count(NONE), "winner": kinds.count(WINNER),
            "tie": kinds.count(TIE), "survive": kinds.count(WINNER) + kinds.count(TIE)}


def count_line(counts: dict[str, int]) -> str:
    return (f"Ties: {counts['tie']} of {counts['survive']} cells where some arm survives; "
            f"clear winner: {counts['winner']}; no arm tolerates 0 ms: {counts['none']}.")


def make_which_method(atlas, summary, stratum, groups, pal, path):
    variants, losses, arms = stratum_axes(groups)
    xs = loss_columns(losses)
    colors = arm_colors(pal)
    tie_face = pal["line"]  # neutral Horizon token, never an arm color, red or amber
    fig_w = 11.6
    cell_h = 0.72
    classified = {(v, lk): classify_cell(groups.get((v, lk), {}))
                  for v in variants for lk in losses if groups.get((v, lk))}
    counts = cell_counts(list(classified.values()))
    foot = footnote(atlas, stratum, summary, width=112, extra=[
        count_line(counts),
        "Cell text: the arm and its maximum tolerated delay. A cell names an arm only when "
        "it tolerates strictly more delay than every other arm; otherwise it reads Tie."])
    foot_h = 0.15 + 0.155 * (foot.count("\n") + 1)
    legend_y = foot_h + 0.2
    bottom_in = legend_y + 0.3 + 0.35  # legend, then a small gap under the grid
    height = cell_h * len(variants) + bottom_in + 1.55  # title, subtitle, loss labels
    fig = plt.figure(figsize=(fig_w, height))
    ax = fig.add_axes([0.14, bottom_in / height, 0.845, cell_h * len(variants) / height])
    winners_used: list[str] = []
    any_tie = False
    for r, variant in enumerate(variants):
        for lk, x in zip(losses, xs):
            if (variant, lk) not in classified:
                ax.add_patch(Rectangle((x - 0.5, r - 0.5), 1, 1, facecolor=pal["background"],
                                       edgecolor=pal["surface"], linewidth=2))
                continue
            kind, tied, entry = classified[(variant, lk)]
            if kind == NONE:
                ax.add_patch(Rectangle((x - 0.5, r - 0.5), 1, 1, facecolor=pal["surface"],
                                       edgecolor=pal["alert"], linewidth=1.2))
                ax.text(x, r - 0.12, "none", ha="center", va="center", fontsize=9,
                        color=pal["alert"])
                ax.text(x, r + 0.17, "fails at 0 ms", ha="center", va="center", fontsize=7,
                        color=pal["alert"])
            elif kind == TIE:
                any_tie = True
                ax.add_patch(Rectangle((x - 0.5, r - 0.5), 1, 1, facecolor=tie_face,
                                       edgecolor=pal["surface"], linewidth=2))
                if len(tied) > TIE_LISTED_MAX:
                    ax.text(x, r - 0.12, tie_text(tied), ha="center", va="center", fontsize=8,
                            color=pal["text"])
                else:
                    ax.text(x, r - 0.27, "Tie", ha="center", va="center", fontsize=9,
                            color=pal["text"])
                    ax.text(x, r - 0.03, ", ".join(short_label(a) for a in tied), ha="center",
                            va="center", fontsize=6.5, color=pal["textSecondary"])
                ax.text(x, r + 0.25, delay_text(entry), ha="center", va="center",
                        fontsize=7.5, color=pal["text"])
            else:
                arm = tied[0]
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
    fig.text(0.14, 1 - 0.22 / height, "Which method tolerates the most delay, where is it a "
             "tie, and where does none?", fontsize=12.5, color=pal["text"], ha="left", va="top")
    desc = stratum_description(atlas, stratum)
    fig.text(0.14, 1 - 0.62 / height, "An arm is named only when it tolerates strictly more "
             f"delay than every other arm (maximum tolerated delay, ms), {desc}",
             fontsize=9, color=pal["textSecondary"], ha="left", va="top")
    # Legend: one swatch per arm that wins somewhere, the tie swatch, and the "no arm" marker.
    present = [a for a in arms if a in winners_used]
    leg = fig.add_axes([0.14, legend_y / height, 0.845, 0.3 / height])
    x_cursor = 0.0

    def swatch(face, edge, label, label_color):
        nonlocal x_cursor
        leg.add_patch(Rectangle((x_cursor, 0), 0.5, 1, facecolor=face, edgecolor=edge,
                                linewidth=1.2 if edge != "none" else 0))
        leg.text(x_cursor + 0.62, 0.5, label, va="center", ha="left", fontsize=9,
                 color=label_color)
        x_cursor += 0.62 + 0.115 * len(label) + 0.55

    for arm in present:
        swatch(colors.get(arm, pal["textSecondary"]), "none", arm_label(arm), pal["text"])
    if any_tie:
        swatch(tie_face, "none", "tie: two or more arms share the best delay", pal["text"])
    if counts["none"]:
        swatch(pal["surface"], pal["alert"], "no arm tolerates 0 ms", pal["alert"])
    leg.set_xlim(0, max(x_cursor, 12))
    leg.set_ylim(0, 1)
    leg.axis("off")
    fig.text(0.14, 0.15 / height, foot, fontsize=8, color=pal["textSecondary"], ha="left",
             va="bottom", linespacing=1.35)
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
