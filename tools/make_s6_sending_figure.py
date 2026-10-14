"""Draw the sending-policy figures from a s6-sending run directory.

Usage (from the repo root):

    PYTHONPATH=. ./.venv/bin/python tools/make_s6_sending_figure.py runs/s6-sending \
        [--output-dir DIR]

Question answered: for a fixed radio budget, which sending policy gives the least
tracking error, and does packet loss change the answer?

Writes two PNGs into the output directory (default: the run directory):

    s6_sending_packets.png   median p95 error versus packets per second (log axis)
    s6_sending_bytes.png     the same with on-air bytes per second on the x axis

Panels are the targets (columns) by packet loss (rows), at 100 ms delay and 5 cm
noise, for the capture-time tracker (the "recovery" arm). Periodic points are
joined by a line; event-triggered points are not. Filled markers are float32
positions, open markers are int16 positions.

Every plotted number is read from summary.json. The script only converts meters
to centimeters and fractions to percent for labels. Colors come from
design/horizon_tokens.json through tools/horizon_mpl.py. The same input files
always produce the same PNG bytes. If the output directory has a manifest.json,
the PNG hashes are added to its "outputs" map.
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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

from tools import horizon_mpl  # noqa: E402

EXPERIMENT = "s6-sending"
SLICE = "main"
ARM = "recovery"
MAIN_DELAY_S = 0.1
MAIN_NOISE_M = 0.05
TARGET_ORDER = ("cv", "ct", "jink")
TARGET_TITLES = {
    "cv": "Straight line, 1 m/s",
    "ct": "Circle, 1 m/s, 1 m radius",
    "jink": "Jinking, 1 m/s",
}
LOSS_ROWS = (0.0, 0.30)
DPI = 200
FAMILIES = ("periodic", "event", "event_heartbeat")
FAMILY_MARKER = {"periodic": "o", "event": "s", "event_heartbeat": "D"}
FAMILY_SIZE = {"periodic": 5.0, "event": 4.6, "event_heartbeat": 4.6}
FAMILY_LABEL = {
    "periodic": "Periodic",
    "event": "Event",
    "event_heartbeat": "Event + 1 s heartbeat",
}
FIG_WIDTH = 11.0
FIG_HEIGHT = 8.2
LEFT = 0.075
RIGHT = 0.985
XMODES = {
    "packets": {
        "key": "packets_per_second_median",
        "label": "Packets sent per second (log scale)",
        "file": "s6_sending_packets.png",
        "title": "Which sending policy gives the least tracking error for the packets sent?",
    },
    "bytes": {
        "key": "on_air_bytes_per_second_median",
        "label": "On-air bytes per second (log scale)",
        "file": "s6_sending_bytes.png",
        "title": "Which sending policy gives the least tracking error for the bytes on air?",
    },
}


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
    if not isinstance(doc, dict) or doc.get("experiment") != EXPERIMENT:
        found = doc.get("experiment") if isinstance(doc, dict) else type(doc).__name__
        raise FigureError(f"unexpected experiment {found!r}; this script reads {EXPERIMENT!r}")
    for key in ("config", "cells", "policies"):
        if key not in doc:
            raise FigureError(f"summary.json is missing the {key!r} field")
    if "failure_p95_m" not in doc["config"]:
        raise FigureError("summary.json config is missing 'failure_p95_m'")
    return doc


def family_of(params: dict) -> str:
    if params["kind"] == "periodic":
        return "periodic"
    return "event_heartbeat" if "heartbeat_s" in params else "event"


def encoding_of(params: dict) -> str:
    """float32, int16 at the standard 1 mm step, or int16 at a coarser step."""
    step = params.get("quantization_m")
    if step is None:
        return "f32"
    return "i16" if abs(step - 0.001) < 1e-9 else "i16_coarse"


def point_label(params: dict) -> str:
    if params["kind"] == "periodic":
        return f"{params['rate_hz']:g} Hz"
    return f"{params['delta_m'] * 100:g} cm"


def select_points(doc: dict) -> dict:
    """Map (target, loss) to a list of plotted points for the main slice."""
    out: dict = {}
    for cell in doc["cells"]:
        if cell.get("slice") != SLICE:
            continue
        spec = cell["cell"]
        if abs(spec["delay_s"] - MAIN_DELAY_S) > 1e-9 or abs(spec["noise_std_m"] - MAIN_NOISE_M) > 1e-9:
            continue
        if ARM not in cell["arms"]:
            raise FigureError(f"cell {cell.get('key')} has no {ARM!r} arm")
        params = doc["policies"][cell["policy"]]
        arm = cell["arms"][ARM]
        out.setdefault((spec["target"], round(spec["loss_probability"], 6)), []).append({
            "policy": cell["policy"],
            "params": params,
            "family": family_of(params),
            "encoding": encoding_of(params),
            "budget": cell["budget"],
            "med": arm["p95_median_m"],
            "low": arm["p95_ci_low_m"],
            "high": arm["p95_ci_high_m"],
            "n_seeds": arm["n_seeds"],
        })
    return out


def targets_present(points: dict) -> list[str]:
    found = {key[0] for key in points}
    return [t for t in TARGET_ORDER if t in found] + sorted(found - set(TARGET_ORDER))


def losses_present(points: dict) -> list[float]:
    found = {key[1] for key in points}
    return [p for p in LOSS_ROWS if round(p, 6) in found]


def fmt_g(value: float) -> str:
    return f"{value:g}"


# ---------------------------------------------------------------- drawing


def draw_panel(ax, pts, xkey, pal, color, label_events):
    """Draw one panel: periodic line, then every policy marker with its 95 % interval.

    Periodic points are labelled with their rate in every panel; event points are
    labelled with their drift threshold only where label_events is true (top row).
    """
    periodic = sorted((p for p in pts if p["family"] == "periodic" and p["encoding"] == "f32"),
                      key=lambda p: p["budget"][xkey])
    if len(periodic) > 1:
        ax.plot([p["budget"][xkey] for p in periodic], [p["med"] * 100 for p in periodic],
                color=color, linewidth=1.3, alpha=0.55, zorder=3)
    for p in pts:
        x = p["budget"][xkey]
        y = p["med"] * 100
        fam, enc = p["family"], p["encoding"]
        marker = "x" if enc == "i16_coarse" else FAMILY_MARKER[fam]
        ax.errorbar([x], [y], yerr=[[y - p["low"] * 100], [p["high"] * 100 - y]], fmt="none",
                    ecolor=color, elinewidth=0.7, capsize=1.8, capthick=0.7, alpha=0.7,
                    zorder=3.2)
        if enc == "f32":
            face, edge, width = color, color, 0.0
        else:
            face, edge, width = pal["surface"], color, 1.0
        ax.plot([x], [y], linestyle="none", marker=marker, markersize=FAMILY_SIZE[fam],
                markerfacecolor=face, markeredgecolor=edge, markeredgewidth=width if marker != "x" else 1.2,
                zorder=4)
        if enc == "f32" and (fam == "periodic" or (label_events and fam == "event")):
            above = fam == "periodic"
            ax.annotate(point_label(p["params"]), xy=(x, y), xytext=(5, 5 if above else -5),
                        textcoords="offset points", ha="left", va="bottom" if above else "top",
                        fontsize=7, color=pal["textSecondary"], zorder=5)


def set_log_ticks(ax, lo, hi):
    """Log x axis with plain-number labels at 1, 2, 5 times a power of ten."""
    ticks = []
    decade = 10.0 ** int(-6)
    while decade <= hi * 10:
        ticks += [m * decade for m in (1, 2, 5) if lo <= m * decade <= hi]
        decade *= 10
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def collect_range(points: dict, xkey: str):
    xs, tops = [], []
    for plist in points.values():
        for p in plist:
            xs.append(p["budget"][xkey])
            tops.append(p["high"] * 100)
    return xs, tops


def legend_handles(pal, color, present_families, has_coarse):
    marker_handles = []
    marker_labels = []
    for fam in FAMILIES:
        if fam in present_families:
            marker_handles.append(Line2D([0], [0], linestyle="none", marker=FAMILY_MARKER[fam],
                                         markersize=FAMILY_SIZE[fam], markerfacecolor=color,
                                         markeredgewidth=0, color=color))
            marker_labels.append(FAMILY_LABEL[fam])
    enc_handles = [
        Line2D([0], [0], linestyle="none", marker="o", markersize=5, markerfacecolor=pal["textSecondary"],
               markeredgewidth=0),
        Line2D([0], [0], linestyle="none", marker="o", markersize=5, markerfacecolor=pal["surface"],
               markeredgecolor=pal["textSecondary"], markeredgewidth=1.0),
    ]
    enc_labels = ["float32 (filled)", "int16, 1 mm step (open)"]
    if has_coarse:
        enc_handles.append(Line2D([0], [0], linestyle="none", marker="x", markersize=5,
                                  markeredgecolor=pal["textSecondary"], markeredgewidth=1.2))
        enc_labels.append("int16, 1 cm step (cross)")
    extra_handles = [
        Line2D([0], [0], color=color, linewidth=1.3, alpha=0.55),
        Line2D([0], [0], color=pal["alert"], linewidth=1.0),
    ]
    extra_labels = ["periodic points joined", "failure line"]
    return (
        (marker_handles, marker_labels, "Sending policy (marker shape)"),
        (enc_handles, enc_labels, "Position encoding (fill)"),
        (extra_handles, extra_labels, "Lines"),
    )


def add_legends(fig, groups, pal, y, x_positions):
    for (handles, labels, title), x in zip(groups, x_positions):
        leg = fig.legend(handles, labels, title=title, loc="upper left", bbox_to_anchor=(x, y),
                         alignment="left", handlelength=2.0, labelspacing=0.45, borderaxespad=0,
                         handletextpad=0.7)
        leg.get_title().set_color(pal["textSecondary"])
        leg.get_title().set_fontsize(8.5)


def footnote_text(doc, n_seeds, mode):
    lines = [
        f"Simulated link and sensor, {n_seeds} seeds per cell, median p95 with 95 % interval. "
        f"{fmt_g(MAIN_DELAY_S * 1000)} ms one-way delay, {fmt_g(MAIN_NOISE_M * 100)} cm noise.",
        "A cell fails when at least 50 % of seeds fail (rule proposed, not yet adopted).",
    ]
    if mode == "bytes":
        lines.append(overhead_note(doc))
    return "\n".join(lines)


def overhead_note(doc) -> str:
    for cell in doc["cells"]:
        budget = cell.get("budget") or {}
        if "on_air_bytes_per_packet" in budget and "data_bytes_per_packet" in budget:
            extra = budget["on_air_bytes_per_packet"] - budget["data_bytes_per_packet"]
            return (f"On-air bytes add a fixed {extra:g} B of radio framing to every packet "
                    "(literature-derived, not measured on our boards).")
    raise FigureError("no cell has both on_air_bytes_per_packet and data_bytes_per_packet")


def make_figure(doc, points, mode, pal, path):
    spec = XMODES[mode]
    xkey = spec["key"]
    targets = targets_present(points)
    losses = losses_present(points)
    if not targets or not losses:
        raise FigureError(
            f"no {SLICE!r} cells at delay {MAIN_DELAY_S} s and noise {MAIN_NOISE_M} m")
    color = horizon_mpl.arm_style(ARM, pal)["color"]
    failure_cm = doc["config"]["failure_p95_m"] * 100
    n_seeds = max(p["n_seeds"] for plist in points.values() for p in plist)
    xs, tops = collect_range(points, xkey)
    xlo, xhi = min(xs) / 1.45, max(xs) * 1.45
    ytop = max(failure_cm * 1.3, max(tops) * 1.08)

    ncols, nrows = len(targets), len(losses)
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(nrows, ncols, left=LEFT, right=RIGHT, top=0.845, bottom=0.31,
                          wspace=0.07, hspace=0.30)
    axes = [[None] * ncols for _ in range(nrows)]
    families = set()
    has_coarse = False
    for r, loss in enumerate(losses):
        for c, target in enumerate(targets):
            first = r == 0 and c == 0
            ax = fig.add_subplot(gs[r, c], sharex=axes[0][0] if not first else None,
                                 sharey=axes[0][0] if not first else None)
            axes[r][c] = ax
            pts = points.get((target, round(loss, 6)), [])
            for p in pts:
                families.add(p["family"])
                has_coarse = has_coarse or p["encoding"] == "i16_coarse"
            draw_panel(ax, pts, xkey, pal, color, label_events=(r == 0))
            ax.axhline(failure_cm, color=pal["alert"], linewidth=1.0, zorder=1.5)
            if first:
                ax.annotate(f"failure line (p95 {fmt_g(failure_cm)} cm)", xy=(0.02, failure_cm),
                            xycoords=("axes fraction", "data"), xytext=(0, 3),
                            textcoords="offset points", ha="left", va="bottom", fontsize=7.5,
                            color=pal["textSecondary"])
            ax.set_xscale("log")
            ax.set_xlim(xlo, xhi)
            set_log_ticks(ax, xlo, xhi)
            ax.grid(axis="x", visible=True, which="major")
            if r == 0:
                ax.set_title(TARGET_TITLES.get(target, target), loc="left", fontsize=10)
            if r == nrows - 1:
                ax.set_xlabel(spec["label"], fontsize=8.5)
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(f"{fmt_g(loss * 100)} % packet loss\nmedian p95 error (cm)",
                              fontsize=8.5)
            else:
                ax.tick_params(labelleft=False)
    axes[0][0].set_ylim(0, ytop)
    fig.text(LEFT, 0.974, spec["title"], fontsize=12, color=pal["text"], ha="left", va="top")
    fig.text(LEFT, 0.933, "p95 error of the target estimate, capture-time tracker, by target "
             "motion and packet loss", fontsize=9, color=pal["textSecondary"], ha="left",
             va="top")
    add_legends(fig, legend_handles(pal, color, families, has_coarse), pal, y=0.205,
                x_positions=(LEFT, 0.36, 0.68))
    fig.text(LEFT, 0.025, footnote_text(doc, n_seeds, mode), fontsize=8,
             color=pal["textSecondary"], ha="left", va="bottom", linespacing=1.4)
    save(fig, path)


def save(fig, path: Path):
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------- main


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_manifest(output_dir: Path, written: list[Path]) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    outputs = manifest.setdefault("outputs", {})
    for path in written:
        outputs[path.name] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")


def build(run_dir: Path, output_dir: Path) -> list[Path]:
    doc = load_summary(run_dir)
    points = select_points(doc)
    pal = horizon_mpl.apply_style("light")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for mode in ("packets", "bytes"):
        path = output_dir / XMODES[mode]["file"]
        make_figure(doc, points, mode, pal, path)
        written.append(path)
    update_manifest(output_dir, written)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="run directory containing summary.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where to write the PNGs (default: the run directory)")
    args = parser.parse_args(argv)
    try:
        written = build(args.run_dir, args.output_dir or args.run_dir)
    except FigureError as err:
        print(f"make_s6_sending_figure: error: {err}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
