"""Draw the mis-tuned filter figures from a tuning run (summary.json, consistency_check.json).

Usage (from the repo root):

    PYTHONPATH=. ./.venv/bin/python tools/make_s6_tuning_figure.py runs/s6-tuning \
        [--output-dir DIR]

Questions answered: how wrong can the tuning be before the tracker fails, and does the NIS
monitor warn when it is?

Writes two PNGs into the output directory (default: the run directory):

    s6_tuning.png          main figure: median p95 error against the Q-to-R ratio, one line
                           per filter model, panels by target motion and link delay
    s6_tuning_monitor.png  second figure: the monitor's median warn fraction against the same
                           axis, with the 0.5 "warned" level and the failing cells ringed

The x axis is the Q-to-R ratio relative to nominal on a log scale. Scaling the measurement
standard deviation by f moves the ratio by 1 / f^2, and scaling Q by g moves it by g, so both
kinds of mis-tuning share one axis and the twins line up. If the output directory has a
manifest.json, the sha256 of each PNG is added to its "outputs" map.

Every plotted number is read from the run files. The script only converts meters to
centimeters and fractions to percent for labels. Colors come from design/horizon_tokens.json
through tools/horizon_mpl.py. The same input files always produce the same PNG bytes.
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
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from tools import horizon_mpl  # noqa: E402

SUMMARY_NAME = "summary.json"
CONSISTENCY_NAME = "consistency_check.json"
MANIFEST_NAME = "manifest.json"
MAIN_PNG = "s6_tuning.png"
MONITOR_PNG = "s6_tuning_monitor.png"
SLICE = "main"
MODELS = ("cv", "ca", "ct", "imm")
TARGETS = ("cv", "ct")
WARNED_LEVEL = 0.5
FAIL_RATE_RULE = 0.5
XTICKS = (0.04, 0.2, 1.0, 5.0, 25.0)
DPI = 200
RATIO_DIGITS = 6


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
    doc = read_json(run_dir / SUMMARY_NAME)
    if not isinstance(doc, dict):
        raise FigureError(f"{SUMMARY_NAME} is not a JSON object")
    for key in ("config", "arms", "cells"):
        if key not in doc:
            raise FigureError(f"{SUMMARY_NAME} is missing the {key!r} field")
    config = doc["config"].get("recovery", {})
    for key in ("failure_p95_m", "target_speed_mps", "turn_radius_m", "tracker_measurement_std_m"):
        if key not in config:
            raise FigureError(f"{SUMMARY_NAME} config.recovery is missing {key!r}")
    return doc


def load_consistency(run_dir: Path) -> list[dict]:
    path = run_dir / CONSISTENCY_NAME
    if not path.exists():
        return []
    doc = read_json(path)
    entries = doc if isinstance(doc, list) else [doc]
    return [entry for entry in entries if isinstance(entry, dict)]


def ratio_of(spec: dict, base_std_m: float) -> float:
    """Q-to-R ratio relative to nominal for one arm spec."""
    q_factor = float(spec.get("q_factor") or 1.0)
    r_std = spec.get("r_std_m")
    r_factor = 1.0 if r_std is None else float(r_std) / base_std_m
    return round(q_factor / (r_factor * r_factor), RATIO_DIGITS)


def classify_arms(doc: dict) -> dict:
    """Per model: nominal arm name, and sorted (ratio, name) lists for the R and Q families."""
    base = float(doc["config"]["recovery"]["tracker_measurement_std_m"])
    out = {m: {"nominal": None, "r": [], "q": []} for m in MODELS}
    for spec in doc["arms"]:
        model = spec.get("model")
        if model not in out or spec.get("adaptive") or spec.get("stamping") != "capture":
            continue
        has_r = spec.get("r_std_m") is not None
        has_q = abs(float(spec.get("q_factor") or 1.0) - 1.0) > 1e-12
        if has_r and has_q:
            continue
        name = spec["name"]
        if has_r:
            out[model]["r"].append((ratio_of(spec, base), name))
        elif has_q:
            out[model]["q"].append((ratio_of(spec, base), name))
        elif out[model]["nominal"] is None:
            out[model]["nominal"] = name
    for family in out.values():
        family["r"].sort()
        family["q"].sort()
    present = {m: f for m, f in out.items() if f["nominal"] and (f["r"] or f["q"])}
    if not present:
        raise FigureError("no mis-tuned arms found for the cv, ca, ct or imm models")
    return present


def index_cells(doc: dict) -> dict:
    """Map (target, delay_s, loss) to the arms table of the main-slice cell."""
    table = {}
    for cell in doc["cells"]:
        if cell.get("slice") != SLICE:
            continue
        spec = cell["cell"]
        key = (spec["target"], round(spec["delay_s"], 6), round(spec["loss_probability"], 6))
        table[key] = cell["arms"]
    return table


def choose_loss(table: dict) -> float:
    losses = sorted({key[2] for key in table})
    if not losses:
        raise FigureError(f"{SUMMARY_NAME} has no cells in slice {SLICE!r}")
    return 0.0 if 0.0 in losses else losses[0]


def n_seeds_of(doc: dict, table: dict) -> int:
    for arms in table.values():
        for summary in arms.values():
            if "n_seeds" in summary:
                return int(summary["n_seeds"])
    return len(doc.get("seeds", []))


# ---------------------------------------------------------------- labels


def fmt_g(value: float) -> str:
    return f"{value:g}"


def model_colors(pal: dict) -> dict:
    """One color per filter model, from horizon_mpl.arm_style (cv shares the recovery color)."""
    return {m: horizon_mpl.arm_style(m, pal)["color"] for m in MODELS}


def model_label(model: str) -> str:
    return {"cv": "CV (recovery arm)", "ca": "CA", "ct": "CT", "imm": "IMM"}[model]


def target_title(config: dict, target: str) -> str:
    speed = fmt_g(config["target_speed_mps"])
    if target == "cv":
        return f"Straight line, {speed} m/s"
    return f"Turning, {speed} m/s, {fmt_g(config['turn_radius_m'])} m radius"


def delay_title(delay_s: float) -> str:
    return f"One-way delay {fmt_g(round(delay_s * 1000.0, 3))} ms"


def loss_text(loss: float) -> str:
    return f"{fmt_g(round(loss * 100.0, 3))} %"


def ratio_tick_label(value: float) -> str:
    return "1\n(nominal)" if value == 1.0 else f"{fmt_g(value)}x"


# ---------------------------------------------------------------- drawing


def style_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(0.03, 33.0)
    ax.xaxis.set_major_locator(FixedLocator(list(XTICKS)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([ratio_tick_label(v) for v in XTICKS])
    ax.grid(axis="x", visible=True)


def p95_rows(arms: dict, family: list, nominal: str) -> dict:
    """Ratio-sorted x, median, low, high in cm for the given family (with the nominal arm)."""
    names = ([(1.0, nominal)] if nominal in arms else []) + [(r, n) for r, n in family if n in arms]
    names.sort()
    return {
        "x": [r for r, _ in names],
        "med": [arms[n]["p95_median_m"] * 100.0 for _, n in names],
        "low": [arms[n]["p95_ci_low_m"] * 100.0 for _, n in names],
        "high": [arms[n]["p95_ci_high_m"] * 100.0 for _, n in names],
    }


def draw_error_panel(ax, arms, classes, colors, pal, failure_cm, label_failure):
    top = failure_cm
    naive = arms.get("naive")
    if naive is not None:
        naive_style = horizon_mpl.arm_style("naive", pal)
        ax.axhline(naive["p95_median_m"] * 100.0, color=naive_style["color"],
                   linestyle=naive_style["linestyle"], linewidth=1.1, zorder=2)
        top = max(top, naive["p95_median_m"] * 100.0)
    ax.axhline(failure_cm, color=pal["alert"], linewidth=1.0, zorder=1.5)
    if label_failure:
        ax.annotate(f"failure line (p95 {failure_cm:g} cm)", xy=(0.02, failure_cm),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8,
                    color=pal["textSecondary"], zorder=8,
                    bbox={"facecolor": pal["surface"], "edgecolor": "none", "alpha": 0.9,
                          "pad": 1.2})
    for model, family in classes.items():
        color = colors[model]
        q_rows = p95_rows(arms, family["q"], family["nominal"])
        r_rows = p95_rows(arms, family["r"], family["nominal"])
        if q_rows["x"]:
            ax.fill_between(q_rows["x"], q_rows["low"], q_rows["high"], color=color, alpha=0.13,
                            linewidth=0, zorder=2)
            ax.plot(q_rows["x"], q_rows["med"], color=color, linewidth=1.4, marker="s",
                    markersize=3.8, markeredgewidth=0, zorder=4, clip_on=False)
            top = max(top, max(q_rows["high"]))
        r_only = [i for i, x in enumerate(r_rows["x"]) if x != 1.0]
        if r_only:
            xs = [r_rows["x"][i] for i in r_only]
            med = [r_rows["med"][i] for i in r_only]
            lower = [r_rows["med"][i] - r_rows["low"][i] for i in r_only]
            upper = [r_rows["high"][i] - r_rows["med"][i] for i in r_only]
            ax.errorbar(xs, med, yerr=[lower, upper], fmt="none", ecolor=color, elinewidth=0.8,
                        capsize=2, capthick=0.8, zorder=3, clip_on=False)
            ax.plot(xs, med, linestyle="none", marker="o", markersize=5.0,
                    markerfacecolor=pal["surface"], markeredgecolor=color, markeredgewidth=1.1,
                    zorder=5, clip_on=False)
            top = max(top, max(r_rows["high"][i] for i in r_only))
        nominal = arms.get(family["nominal"])
        if nominal is not None:
            ax.plot([1.0], [nominal["p95_median_m"] * 100.0], linestyle="none", marker="D",
                    markersize=5.2, markerfacecolor=color, markeredgecolor=pal["surface"],
                    markeredgewidth=0.8, zorder=6, clip_on=False)
    return top


def draw_monitor_panel(ax, arms, classes, colors, pal):
    ax.axhline(WARNED_LEVEL * 100.0, color=pal["textSecondary"], linestyle=(0, (1.2, 2.4)),
               linewidth=1.1, zorder=1.5)
    for model, family in classes.items():
        color = colors[model]
        for rows_family, marker in ((family["q"], "s"), (family["r"], "o")):
            names = ([(1.0, family["nominal"])] if family["nominal"] in arms else [])
            names += [(r, n) for r, n in rows_family if n in arms]
            names.sort()
            if not names:
                continue
            xs = [r for r, _ in names]
            warn = [arms[n]["x_monitor_warn_fraction_median"] * 100.0 for _, n in names]
            failed = [arms[n]["fail_rate"] >= FAIL_RATE_RULE for _, n in names]
            if marker == "s":
                ax.plot(xs, warn, color=color, linewidth=1.4, marker="s", markersize=3.8,
                        markeredgewidth=0, zorder=4, clip_on=False)
            else:
                keep = [i for i, x in enumerate(xs) if x != 1.0]
                ax.plot([xs[i] for i in keep], [warn[i] for i in keep], linestyle="none",
                        marker="o", markersize=5.0, markerfacecolor=pal["surface"],
                        markeredgecolor=color, markeredgewidth=1.1, zorder=5, clip_on=False)
            ring = [i for i, f in enumerate(failed) if f and (marker == "s" or xs[i] != 1.0)]
            if ring:
                ax.plot([xs[i] for i in ring], [warn[i] for i in ring], linestyle="none",
                        marker="o", markersize=10.5, markerfacecolor="none",
                        markeredgecolor=pal["alert"], markeredgewidth=1.0, zorder=6,
                        clip_on=False)
        nominal = arms.get(family["nominal"])
        if nominal is not None:
            ax.plot([1.0], [nominal["x_monitor_warn_fraction_median"] * 100.0], linestyle="none",
                    marker="D", markersize=5.2, markerfacecolor=color,
                    markeredgecolor=pal["surface"], markeredgewidth=0.8, zorder=6, clip_on=False)
            if nominal["fail_rate"] >= FAIL_RATE_RULE:
                ax.plot([1.0], [nominal["x_monitor_warn_fraction_median"] * 100.0],
                        linestyle="none", marker="o", markersize=10.5, markerfacecolor="none",
                        markeredgecolor=pal["alert"], markeredgewidth=1.0, zorder=6,
                        clip_on=False)


FIG_SIZE = (11.0, 7.8)
LEGEND_TOP = 0.235


def make_grid(run: dict, ylabel_for_row, draw, title, subtitle, legend_groups, footnote):
    """Shared layout: rows are targets, columns are delays; absent cells stay empty."""
    pal = run["pal"]
    targets, delays = run["targets"], run["delays"]
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(len(targets), len(delays), left=0.085, right=0.985, top=0.83,
                          bottom=0.34, wspace=0.07, hspace=0.36)
    axes = {}
    first_drawn = {}
    for r, target in enumerate(targets):
        for c, delay in enumerate(delays):
            ax = fig.add_subplot(gs[r, c])
            axes[(r, c)] = ax
            style_axis(ax)
            arms = run["table"].get((target, delay, run["loss"]))
            if r == 0:
                ax.set_title(delay_title(delay), loc="left", fontsize=10)
            if r == len(targets) - 1:
                ax.set_xlabel("Q-to-R ratio vs nominal (log scale)")
            else:
                ax.tick_params(labelbottom=False)
            if arms is None:
                ax.text(0.5, 0.5, "not in this run", transform=ax.transAxes, ha="center",
                        va="center", fontsize=9, color=pal["textSecondary"])
                ax.tick_params(labelleft=False)
                continue
            first_drawn.setdefault(r, c)
            draw(ax, arms, label=(r == 0 and c == first_drawn[0]))
            if c != first_drawn[r]:
                ax.tick_params(labelleft=False)
    for r, target in enumerate(targets):
        box = axes[(r, 0)].get_position()
        fig.text(0.022, (box.y0 + box.y1) / 2.0, ylabel_for_row(target), rotation=90,
                 ha="center", va="center", fontsize=9.5, color=pal["textSecondary"],
                 linespacing=1.3)
    fig.text(0.085, 0.965, title, fontsize=12, color=pal["text"], ha="left", va="top")
    fig.text(0.085, 0.915, subtitle, fontsize=9, color=pal["textSecondary"], ha="left",
             va="top")
    for (handles, labels, group_title), x in zip(legend_groups, (0.085, 0.30, 0.60)):
        leg = fig.legend(handles, labels, title=group_title, loc="upper left",
                         bbox_to_anchor=(x, LEGEND_TOP), alignment="left", handlelength=2.4,
                         labelspacing=0.45, borderaxespad=0, handletextpad=0.7)
        leg.get_title().set_color(pal["textSecondary"])
        leg.get_title().set_fontsize(8.5)
    fig.text(0.085, 0.02, footnote, fontsize=8, color=pal["textSecondary"], ha="left",
             va="bottom", linespacing=1.4)
    return fig, axes


def model_group(run):
    colors = run["colors"]
    handles = [Line2D([0], [0], color=colors[m], linewidth=2.2) for m in run["classes"]]
    labels = [model_label(m) for m in run["classes"]]
    return handles, labels, "Color: filter model"


def marker_group(run):
    pal = run["pal"]
    gray = pal["textSecondary"]
    handles = [
        Line2D([0], [0], color=gray, linewidth=1.4, marker="s", markersize=3.8,
               markeredgewidth=0),
        Line2D([0], [0], linestyle="none", marker="o", markersize=5.0,
               markerfacecolor=pal["surface"], markeredgecolor=gray, markeredgewidth=1.1),
        Line2D([0], [0], linestyle="none", marker="D", markersize=5.2, markerfacecolor=gray,
               markeredgecolor=pal["surface"], markeredgewidth=0.8),
    ]
    labels = ["Q scaled, R correct", "R scaled, Q correct", "Nominal tuning"]
    return handles, labels, "Marker: which number was scaled"


def seeds_clause(n_seeds: int) -> str:
    return f"{n_seeds} seeds per cell"


def make_error_figure(run: dict, path: Path) -> None:
    pal, colors, classes = run["pal"], run["colors"], run["classes"]
    failure_cm = run["failure_cm"]
    tops = []

    def draw(ax, arms, label):
        tops.append(draw_error_panel(ax, arms, classes, colors, pal, failure_cm, label))

    gray = pal["textSecondary"]
    line_group = (
        [Line2D([0], [0], color=horizon_mpl.arm_style("naive", pal)["color"],
                linestyle=horizon_mpl.arm_style("naive", pal)["linestyle"], linewidth=1.1),
         Line2D([0], [0], color=pal["alert"], linewidth=1.0),
         Patch(facecolor=gray, alpha=0.16, linewidth=0),
         Line2D([0], [0], color=gray, marker="_", markersize=7, linestyle="none",
                markeredgewidth=0.8)],
        ["Naive tracker (arrival-time stamps)", "Failure line",
         "95 % interval, Q line (band)", "95 % interval, R markers (caps)"],
        "Reference lines and intervals")
    footnote = (
        f"Simulated link and sensor, {seeds_clause(run['n_seeds'])}, median p95 with 95 % "
        f"interval. Packet loss {loss_text(run['loss'])}, sensor noise "
        f"{fmt_g(run['noise_cm'])} cm. Scaling the R standard deviation by f moves the "
        "ratio by 1/f^2.\n"
        "Left of 1: lazy filter. Right of 1: eager filter. "
        "Cell fails when at least 50 % of seeds fail (rule proposed, not yet adopted)."
    )
    fig, axes = make_grid(
        run, lambda t: f"{target_title(run['config'], t)}\np95 tracking error (cm)", draw,
        "How wrong can the filter tuning be before the tracker fails?",
        "Median p95 error of the capture-time tracker against the Q-to-R ratio, "
        "relative to the nominal tuning, by filter model",
        [model_group(run), marker_group(run), line_group], footnote)
    ylim_top = max(tops) * 1.06
    for ax in axes.values():
        ax.set_ylim(0, ylim_top)
    save(fig, path)


def consistency_text(entries: list[dict]) -> str:
    if not entries:
        return "Honest-filter check: not in this run."
    parts = []
    for entry in entries:
        inside = entry.get("fraction_windows_inside_band")
        mean_nis = entry.get("mean_of_seed_mean_nis")
        low, high = entry.get("band_low"), entry.get("band_high")
        if inside is None or mean_nis is None or low is None or high is None:
            continue
        parts.append(
            "Honest-filter check (filter tuned right, target moving exactly as the filter "
            f"assumes, no delay, no loss, {entry.get('n_seeds', '?')} seeds):\n"
            f"mean NIS {mean_nis:.2f} (3 expected), {inside * 100.0:.0f} % of windows inside "
            f"the {low:.2f} to {high:.2f} band (about 95 % expected).")
    return "\n".join(parts) if parts else "Honest-filter check: not in this run."


def make_monitor_figure(run: dict, path: Path) -> None:
    pal, colors, classes = run["pal"], run["colors"], run["classes"]

    def draw(ax, arms, label):
        draw_monitor_panel(ax, arms, classes, colors, pal)
        if label:
            ax.annotate("warned (at least 50 % of windows)", xy=(0.02, WARNED_LEVEL * 100.0),
                        xycoords=("axes fraction", "data"), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom", fontsize=8,
                        color=pal["textSecondary"])

    gray = pal["textSecondary"]
    line_group = (
        [Line2D([0], [0], linestyle="none", marker="o", markersize=10.5,
                markerfacecolor="none", markeredgecolor=pal["alert"], markeredgewidth=1.0),
         Line2D([0], [0], color=gray, linestyle=(0, (1.2, 2.4)), linewidth=1.1)],
        ["Red ring: cell fails", "Warned level (0.5)"],
        "Failure and warning")
    footnote = (
        f"Simulated link and sensor, {seeds_clause(run['n_seeds'])}, median over seeds. "
        f"Packet loss {loss_text(run['loss'])}.\n"
        "Cell fails when at least 50 % of seeds fail (rule proposed, not yet adopted).\n"
        + consistency_text(run["consistency"])
    )
    fig, axes = make_grid(
        run, lambda t: f"{target_title(run['config'], t)}\nwindows warned (%)", draw,
        "Does the monitor warn when the tuning is wrong?",
        "Median share of NIS windows flagged as outside the honest band, "
        "against the same Q-to-R ratio axis",
        [model_group(run), marker_group(run), line_group], footnote)
    for ax in axes.values():
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
    save(fig, path)


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------- main


def update_manifest(output_dir: Path, written: list[Path]) -> None:
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise FigureError(f"{manifest_path} is not a JSON object")
    outputs = manifest.setdefault("outputs", {})
    for path in written:
        outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")


def build(run_dir: Path, output_dir: Path) -> list[Path]:
    doc = load_summary(run_dir)
    classes = classify_arms(doc)
    table = index_cells(doc)
    loss = choose_loss(table)
    targets = [t for t in TARGETS if any(k[0] == t and k[2] == loss for k in table)]
    delays = sorted({k[1] for k in table if k[2] == loss and k[0] in targets})
    if not targets or not delays:
        raise FigureError("no usable cells for the tuning figures")
    config = doc["config"]["recovery"]
    pal = horizon_mpl.apply_style("light")
    run = {
        "pal": pal,
        "colors": model_colors(pal),
        "classes": classes,
        "table": table,
        "loss": loss,
        "targets": targets,
        "delays": delays,
        "config": config,
        "failure_cm": config["failure_p95_m"] * 100.0,
        "noise_cm": config["tracker_measurement_std_m"] * 100.0,
        "n_seeds": n_seeds_of(doc, table),
        "consistency": load_consistency(run_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / MAIN_PNG
    monitor_path = output_dir / MONITOR_PNG
    make_error_figure(run, main_path)
    make_monitor_figure(run, monitor_path)
    written = [main_path, monitor_path]
    update_manifest(output_dir, written)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="run directory with summary.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where to write the PNGs (default: the run directory)")
    args = parser.parse_args(argv)
    try:
        written = build(args.run_dir, args.output_dir or args.run_dir)
    except FigureError as err:
        print(f"make_s6_tuning_figure: error: {err}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
