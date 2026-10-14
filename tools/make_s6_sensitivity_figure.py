"""Draw the S6 sensitivity figures from a sensitivity run folder.

Usage (from the repo root):

    PYTHONPATH=. ./.venv/bin/python tools/make_s6_sensitivity_figure.py runs/s6-sensitivity \
        [--output-dir DIR]

Reads morris.json, oat.json and ranking.json from the run folder and writes two PNGs
into the output directory (default: the run folder):

    s6_sensitivity_morris.png  Morris screening: mu* against sigma for every input
    s6_sensitivity_oat.png     one-at-a-time response ranges, ranked, two central cases

One question per figure.
  Morris: which inputs move the tracking error the most on average, and are they the
  same for both trackers? Points above the diagonal (sigma larger than mu*) have an
  effect that is nonlinear or depends on other inputs.
  One-at-a-time: with every other input held at the central case, how far does the p95
  error move when one input is swept across its whole range?

Every plotted number is read from the run files. The script only converts seconds to
milliseconds, meters to centimeters and fractions to percent for labels. Arm colors and
markers come from horizon_mpl.arm_style and are the same in every S6 figure. Red and amber
are not used here because no failure line is drawn.
Colors come from design/horizon_tokens.json through tools/horizon_mpl.py. The same input
files always give the same PNG bytes. When the output folder has a manifest.json, the PNG
hashes are added to its "outputs" map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from tools import horizon_mpl  # noqa: E402

MORRIS_NAME = "s6_sensitivity_morris.png"
OAT_NAME = "s6_sensitivity_oat.png"
# Class key in morris.json and ranking.json, and the matching case key in the OAT files.
CLASSES = (("straight", "cv"), ("turn", "ct"))
CLASS_TITLES = {"straight": "Straight line", "turn": "Turning target"}
ARMS = ("recovery", "imm")
ARM_LABELS = {
    "recovery": "Recovery tracker",
    "imm": "Maneuver-aware tracker (imm)",
}
INPUT_LABELS = {
    "delay": "Link delay",
    "loss": "Packet loss",
    "burst_mean_length": "Burst length",
    "noise": "Sensor noise",
    "speed": "Target speed",
    "turn_rate": "Turn rate",
    "sensor_rate": "Sensor rate",
    "clock_error": "Clock error",
    "tuning_error": "Tracker noise setting",
}
DPI = 200


class FigureError(Exception):
    """Raised for unusable input; main() turns it into a clear exit message."""


# ---------------------------------------------------------------- input


def _load_json(path: Path) -> dict:
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


def load_run(run_dir: Path) -> dict:
    morris = _load_json(run_dir / "morris.json")
    oat = _load_json(run_dir / "oat.json")
    ranking = _load_json(run_dir / "ranking.json")
    for key in ("classes", "design", "inputs"):
        if key not in morris:
            raise FigureError(f"morris.json is missing the {key!r} field")
    for key in ("names", "units"):
        if key not in morris["inputs"]:
            raise FigureError(f"morris.json inputs is missing {key!r}")
    for key in ("design",):
        for sub in ("seeds", "r"):
            if sub not in morris[key]:
                raise FigureError(f"morris.json {key} is missing {sub!r}")
    for cls, _case in CLASSES:
        if cls not in morris["classes"]:
            raise FigureError(f"morris.json has no class {cls!r}")
        for arm in ARMS:
            if arm not in morris["classes"][cls]["effects"]:
                raise FigureError(f"morris.json class {cls!r} has no arm {arm!r}")
    for key in ("oat_ranges", "oat_rank"):
        if key not in ranking:
            raise FigureError(f"ranking.json is missing the {key!r} field")
    if "cells" not in oat:
        raise FigureError("oat.json is missing the 'cells' field")
    return {"morris": morris, "oat": oat, "ranking": ranking}


def n_seeds(run: dict) -> int:
    return len(run["morris"]["design"]["seeds"])


def fmt_g(value: float) -> str:
    return f"{round(value, 6):g}"


# ---------------------------------------------------------------- labels


def input_label(name: str) -> str:
    return INPUT_LABELS.get(name, name.replace("_", " ").capitalize())


def range_text(spec: dict) -> str:
    """A short 'low to high unit' string for an input range, converted for display."""
    low, high, unit = spec["low"], spec["high"], spec["unit"]
    if unit == "s":
        return f"{fmt_g(low * 1000)} to {fmt_g(high * 1000)} ms"
    if unit == "probability":
        return f"{fmt_g(low * 100)} to {fmt_g(high * 100)} %"
    if unit == "m":
        return f"{fmt_g(low * 100)} to {fmt_g(high * 100)} cm"
    if unit == "factor on std":
        return f"{fmt_g(low)}x to {fmt_g(high)}x"
    return f"{fmt_g(low)} to {fmt_g(high)} {unit}"


def arm_colors(pal: dict) -> dict:
    """Arm name -> arm_style dict (color, linestyle, marker), shared with the other S6 figures."""
    return {arm: horizon_mpl.arm_style(arm, pal) for arm in ARMS}


# ---------------------------------------------------------------- Morris figure


def morris_points(morris: dict, cls: str, arm: str) -> list[dict]:
    """One dict per input: name, mu* and sigma in cm."""
    names = morris["inputs"]["names"]
    effects = morris["classes"][cls]["effects"][arm]
    out = []
    for i, name in enumerate(names):
        out.append(
            {
                "name": name,
                "x": effects["mu_star"][i] * 100.0,
                "y": effects["sigma"][i] * 100.0,
            }
        )
    return out


def place_labels(fig, ax, points, pal) -> None:
    """Name every point, close to it, without any label covering another label or a marker.

    Greedy placement, most crowded points first: each label tries positions on rings of
    growing distance around its marker and keeps the nearest free one. Every label gets a
    thin leader line to its own marker. Positions are penalized when the label sits closer
    to another marker than to its own, or when its leader runs through another marker, so
    a label cannot be read as belonging to a neighbour.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    pad = 2.0
    pixels = [ax.transData.transform((p["x"], p["y"])) for p in points]
    markers = [Bbox.from_extents(x - 7, y - 7, x + 7, y + 7) for x, y in pixels]
    rings = (7, 9, 12, 17, 24, 33, 45, 60, 80, 100, 125, 150)
    angles = (20, -20, 160, -160, 90, -90, 0, 180, 50, -50, 130, -130) + tuple(
        a for a in range(-165, 181, 15) if a not in (20, -20, 160, -160, 90, -90, 0, 180, 50, -50, 130, -130))
    placed: list[Bbox] = []
    leaders: list[tuple] = []  # (marker x, marker y, nearest label x, nearest label y) in pixels
    px_per_pt = fig.dpi / 72.0

    def label_size(label):
        probe = ax.text(0, 0, label, fontsize=8.5)
        box = probe.get_window_extent(renderer)
        probe.remove()
        return box.width, box.height

    def leader_to(i, box):
        """Segment from marker i to the nearest point of a label box (pixels)."""
        mx, my = pixels[i]
        return (mx, my, min(max(mx, box.x0), box.x1), min(max(my, box.y0), box.y1))

    def segment_hits(seg, box, steps=24):
        x0, y0, x1, y1 = seg
        if abs(x1 - x0) < 1.0 and abs(y1 - y0) < 1.0:
            return False
        for k in range(1, steps):
            t = k / steps
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            if box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1:
                return True
        return False

    def gap_to(box, i):
        """Pixel distance from marker i to the nearest point of a label box."""
        mx, my = pixels[i]
        return math.hypot(mx - min(max(mx, box.x0), box.x1), my - min(max(my, box.y0), box.y1))

    def twins(i, j):
        """Two markers so close that they read as one point; nothing can tell them apart."""
        return math.hypot(pixels[i][0] - pixels[j][0], pixels[i][1] - pixels[j][1]) < 9.0

    def seg_point_distance(seg, point):
        x0, y0, x1, y1 = seg
        dx, dy = x1 - x0, y1 - y0
        length2 = dx * dx + dy * dy
        t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((point[0] - x0) * dx + (point[1] - y0) * dy) / length2))
        return math.hypot(point[0] - (x0 + t * dx), point[1] - (y0 + t * dy))

    def crowd(i):
        xi, yi = pixels[i]
        return -sum(1 for j, (x, y) in enumerate(pixels)
                    if j != i and abs(x - xi) < 220 and abs(y - yi) < 60)

    for i in sorted(range(len(points)), key=lambda k: (crowd(k), k)):
        p = points[i]
        width, height = label_size(p["label"])
        best = None
        for ring in rings:
            for ang in angles:
                dx = ring * math.cos(math.radians(ang))
                dy = ring * math.sin(math.radians(ang))
                ha = "left" if dx > 3 else ("right" if dx < -3 else "center")
                va = "bottom" if dy > 3 else ("top" if dy < -3 else "center")
                ax_px = pixels[i][0] + dx * px_per_pt
                ay_px = pixels[i][1] + dy * px_per_pt
                x0 = {"left": ax_px, "right": ax_px - width, "center": ax_px - width / 2}[ha]
                y0 = {"bottom": ay_px, "top": ay_px - height, "center": ay_px - height / 2}[va]
                padded = Bbox.from_extents(x0 - pad, y0 - pad, x0 + width + pad,
                                           y0 + height + pad)
                penalty = 0.0
                if (x0 < axes_box.x0 or x0 + width > axes_box.x1 or y0 < axes_box.y0
                        or y0 + height > axes_box.y1):
                    penalty += 1000.0
                penalty += 300.0 * sum(1 for o in placed if padded.overlaps(o))
                penalty += 300.0 * sum(1 for mb in markers if padded.overlaps(mb))
                own = leader_to(i, padded)
                penalty += 40.0 * sum(1 for o in placed if segment_hits(own, o))
                penalty += 40.0 * sum(1 for seg in leaders if segment_hits(seg, padded))
                own_gap = gap_to(padded, i)
                penalty += 45.0 * sum(1 for j in range(len(points))
                                      if j != i and gap_to(padded, j) <= own_gap + 4.0
                                      and not twins(i, j))
                penalty += 60.0 * sum(1 for j in range(len(points))
                                      if j != i and seg_point_distance(own, pixels[j]) < 9.0
                                      and not twins(i, j))
                score = penalty + ring * 0.02
                if best is None or score < best[0]:
                    best = (score, padded, penalty, dx, dy, ha, va, ring)
            if best[2] == 0.0:
                break
        _score, padded, _pen, dx, dy, ha, va, ring = best
        relpos = ({"left": 0.0, "right": 1.0, "center": 0.5}[ha],
                  {"bottom": 0.0, "top": 1.0, "center": 0.5}[va])
        arrow = {"arrowstyle": "-", "color": pal["textSecondary"], "linewidth": 0.7,
                 "shrinkA": 1, "shrinkB": 3, "relpos": relpos}
        ax.annotate(p["label"], xy=(p["x"], p["y"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=8.5,
                    color=pal["text"], zorder=6, arrowprops=arrow)
        placed.append(padded)
        leaders.append(leader_to(i, padded))


def draw_morris_panel(fig, ax, points, limit, style, pal) -> None:
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax.plot([0, limit], [0, limit], color=pal["lineStrong"], linewidth=0.9,
            linestyle=(0, (4, 3)), zorder=1.5)
    ax.scatter([p["x"] for p in points], [p["y"] for p in points], s=34, color=style["color"],
               marker=style["marker"],
               edgecolor=pal["surface"], linewidth=0.8, zorder=4, clip_on=False)
    ticks = [t for t in (0, 1, 2, 5, 10, 20, 50, 100, 200) if t <= limit]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_box_aspect(1)
    place_labels(fig, ax, points, pal)


def build_morris_figure(run: dict, pal: dict, colors: dict):
    """Draw the Morris figure and return it (the caller saves or inspects it)."""
    morris = run["morris"]
    names = morris["inputs"]["names"]
    data = {}
    for cls, _case in CLASSES:
        for arm in ARMS:
            pts = morris_points(morris, cls, arm)
            for p in pts:
                p["label"] = input_label(p["name"])
                if p["x"] == 0.0 and p["y"] == 0.0:
                    p["label"] += " (no effect)"
            data[(cls, arm)] = pts
    fig = plt.figure(figsize=(10.5, 10.6), dpi=DPI)
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.975, top=0.835, bottom=0.145,
                          wspace=0.16, hspace=0.24)
    for r, (cls, _case) in enumerate(CLASSES):
        # One scale per row so both trackers can be compared; square axes so the
        # diagonal is exactly sigma = mu*. The extra room on the right is for labels.
        top = max(max(p["x"], p["y"]) for arm in ARMS for p in data[(cls, arm)])
        limit = top * 2.4 if top > 0 else 1.0
        for c, arm in enumerate(ARMS):
            ax = fig.add_subplot(gs[r, c])
            draw_morris_panel(fig, ax, data[(cls, arm)], limit, colors[arm], pal)
            ax.set_title(f"{CLASS_TITLES[cls]}: {ARM_LABELS[arm]}", loc="left", fontsize=10)
            if r == len(CLASSES) - 1:
                ax.set_xlabel("μ*: average effect size (cm of p95 per full input range)")
            if c == 0:
                ax.set_ylabel("σ: spread of the effect (cm per full range)")
            if r == 0 and c == 0:
                ax.annotate("σ = μ*", xy=(1.7, 1.7), ha="right",
                            va="bottom", rotation=45, rotation_mode="anchor", fontsize=8,
                            color=pal["textSecondary"], zorder=3)
    fig.text(0.085, 0.965, "Which inputs move the tracking error most, and are they the same for both trackers?",
             fontsize=12, color=pal["text"], ha="left", va="top")
    fig.text(0.085, 0.93,
             "Each point is one input. Further right: it changes the p95 error more, on average, per full\n"
             "sweep of its range. Above the dashed line: the effect is not steady (nonlinear, depends on other\n"
             "inputs, or partly seed noise with 10 seeds per point). Axes are linear up to 1 cm, then logarithmic.",
             fontsize=9, color=pal["textSecondary"], ha="left", va="top", linespacing=1.35)
    seeds = n_seeds(run)
    traj = morris["design"]["r"]
    fig.text(
        0.085, 0.025,
        "Simulated link and sensor, "
        f"{seeds} seeds per design point, {traj} Morris trajectories per class, "
        f"{len(names)} inputs. Response is the median p95 error over seeds.\n"
        "This is screening, not variance decomposition: it ranks inputs but does not split the "
        "error into shares.\n"
        "The ranking depends on the input ranges chosen (a wider range makes an input look more "
        "important).\n"
        "The turn rate cannot change a straight line, so its point at zero is a built-in check.",
        fontsize=8, color=pal["textSecondary"], ha="left", va="bottom", linespacing=1.4)
    return fig


def make_morris_figure(run: dict, pal: dict, colors: dict, path: Path) -> None:
    save(build_morris_figure(run, pal, colors), path)


# ---------------------------------------------------------------- one-at-a-time figure


def central_description(run: dict, case: str) -> str:
    """Describe the central case from the OAT cells, using a cell where the field is not swept."""
    cells = [c for c in run["oat"]["cells"] if c.get("case") == case]

    def pick(field: str, swept: str):
        for cell in cells:
            if cell.get("input") == swept:
                return cell["cell"].get(field)
        return None

    parts = []
    rate = pick("rate_hz", "delay")
    if rate is not None:
        parts.append(f"{fmt_g(rate)} Hz")
    noise = pick("noise_std_m", "delay")
    if noise is not None:
        parts.append(f"{fmt_g(noise * 100)} cm noise")
    delay = pick("delay_s", "loss")
    if delay is not None:
        parts.append(f"{fmt_g(delay * 1000)} ms delay")
    loss = pick("loss_probability", "delay")
    model = pick("loss_model", "delay")
    if loss is not None:
        parts.append(f"{fmt_g(loss * 100)} % {model + ' ' if model else ''}loss")
    return ", ".join(parts)


def oat_rows(run: dict, cls: str, case: str) -> list[dict]:
    """Inputs of one case, ranked by the recovery arm's range (rank read from ranking.json)."""
    ranking = run["ranking"]
    ranges = ranking["oat_ranges"][case]
    ranks = ranking["oat_rank"][case]["recovery"]
    specs = run["morris"]["classes"][cls]["input_ranges"]
    names = sorted(ranks, key=lambda n: (ranks[n], n))
    rows = []
    for name in names:
        rows.append(
            {
                "name": name,
                "label": f"{input_label(name)}\n{range_text(specs[name])}",
                "values": {arm: ranges[arm][name] * 100.0 for arm in ARMS},
            }
        )
    return rows


def make_oat_figure(run: dict, pal: dict, colors: dict, path: Path) -> None:
    rows_by = {cls: oat_rows(run, cls, case) for cls, case in CLASSES}
    top = max(v for rows in rows_by.values() for r in rows for v in r["values"].values())
    xmax = top * 1.16 if top > 0 else 1.0
    fig = plt.figure(figsize=(11.0, 7.4), dpi=DPI)
    gs = fig.add_gridspec(1, 2, left=0.145, right=0.985, top=0.80, bottom=0.215, wspace=0.52)
    bar_h = 0.36
    axes = []
    for k, (cls, case) in enumerate(CLASSES):
        ax = fig.add_subplot(gs[0, k], sharex=axes[0] if axes else None)
        axes.append(ax)
        rows = rows_by[cls]
        n = len(rows)
        for i, row in enumerate(rows):
            y = n - 1 - i
            for j, arm in enumerate(ARMS):
                yy = y + (0.5 - j) * (bar_h + 0.03)
                value = row["values"][arm]
                ax.barh(yy, value, height=bar_h, color=colors[arm]["color"], zorder=3)
                ax.annotate(f"{value:.1f}", xy=(value, yy), xytext=(3, 0),
                            textcoords="offset points", ha="left", va="center", fontsize=7.5,
                            color=pal["textSecondary"], zorder=4)
        ax.set_yticks([n - 1 - i for i in range(n)])
        ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5, linespacing=1.25)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(0, xmax)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{CLASS_TITLES[cls]}", loc="left", fontsize=10.5)
        ax.set_xlabel("Change in p95 error across the input's range (cm)")
    fig.text(0.015, 0.965, "How far does each input move the tracking error when swept alone?",
             fontsize=12, color=pal["text"], ha="left", va="top")
    fig.text(0.015, 0.925,
             "Each bar is the highest minus the lowest median p95 error while one input is swept "
             "across its range and the rest stay at the central case.\n"
             "Inputs are ranked by the recovery tracker, largest at the top.",
             fontsize=9, color=pal["textSecondary"], ha="left", va="top", linespacing=1.35)
    # Legend as colored swatches below the panels.
    handles = [Patch(facecolor=colors[a]["color"], linewidth=0) for a in ARMS]
    leg = fig.legend(handles, [ARM_LABELS[a] for a in ARMS], loc="upper left",
                     bbox_to_anchor=(0.145, 0.145), ncol=2, handlelength=1.4,
                     columnspacing=2.2, borderaxespad=0, frameon=False)
    for t in leg.get_texts():
        t.set_color(pal["text"])
    seeds = n_seeds(run)
    central = []
    for _cls, case in CLASSES:
        desc = central_description(run, case)
        if desc:
            central.append(desc)
    central_line = ""
    if central and len(set(central)) == 1:
        central_line = f"Central case: {central[0]}.  "
    fig.text(
        0.015, 0.03,
        f"{central_line}Simulated link and sensor, {seeds} seeds per point.\n"
        "This is screening, not variance decomposition. The bars depend on the ranges chosen "
        "(a wider range makes an input look more important)\n"
        "and each point carries seed noise, so close bars are ties.",
        fontsize=8, color=pal["textSecondary"], ha="left", va="bottom", linespacing=1.4)
    save(fig, path)


# ---------------------------------------------------------------- output


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_manifest(output_dir: Path, paths: list[Path]) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    outputs = manifest.setdefault("outputs", {})
    for path in paths:
        outputs[path.name] = sha256_file(path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return True


def build(run_dir: Path, output_dir: Path) -> list[Path]:
    run = load_run(run_dir)
    pal = horizon_mpl.apply_style("light")
    colors = arm_colors(pal)
    output_dir.mkdir(parents=True, exist_ok=True)
    morris_path = output_dir / MORRIS_NAME
    oat_path = output_dir / OAT_NAME
    make_morris_figure(run, pal, colors, morris_path)
    make_oat_figure(run, pal, colors, oat_path)
    paths = [morris_path, oat_path]
    update_manifest(output_dir, paths)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="sensitivity run folder")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where to write the PNGs (default: the run folder)")
    args = parser.parse_args(argv)
    output_dir = args.output_dir if args.output_dir is not None else args.run_dir
    try:
        written = build(args.run_dir, output_dir)
    except FigureError as err:
        print(f"make_s6_sensitivity_figure: error: {err}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
