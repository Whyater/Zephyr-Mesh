"""Create figures from simulated cooperative fusion comparison data."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from sim.estimation.consistency import nis_band

ARMS = ("single", "central", "star_avg", "peer_ci", "peer_naive")
MARKERS = {"single": "o", "central": "s", "star_avg": "^", "peer_ci": "D", "peer_naive": "v"}
CONSISTENCY_YLIM = (0.5, 100.0)
FAIL_RATE_LIMIT = 0.5


def _summary_cell(cells, *, target, loss, arm):
    matches = [cell for cell in cells if cell["target"] == target
               and float(cell["noise_std_m"]) == 0.05
               and float(cell["loss_probability"]) == loss
               and cell["arm"] == arm and cell["ci"] == "fast"]
    if not matches:
        raise ValueError(f"missing summary cell for target={target}, loss={loss}, arm={arm}")
    return sorted(matches, key=lambda cell: int(cell["n_observers"]))


def _is_failure(cell, median) -> bool:
    """True when a cell is drawn as a failure marker instead of a line point."""
    return (cell["fail_rate"]["rate"] >= FAIL_RATE_LIMIT or cell["diverged_count"] > 0
            or median is None)


def _line_values(selected, key, scale=1.0):
    """Return plotted values with NaN gaps at failed cells."""
    values = []
    for cell in selected:
        median = cell[key]["median"]
        values.append(math.nan if _is_failure(cell, median) else scale * median)
    return values


def _mark_failures(axis, failures, colour):
    """Draw alert X markers along the top edge, labelled with the fail rate.

    failures maps N to a list of (arm, fail rate); several arms failing at the
    same N are stacked downwards from the top edge.
    """
    transform = axis.get_xaxis_transform()
    for x, items in failures.items():
        for slot, (arm, rate) in enumerate(items):
            y = 0.95 - 0.07 * slot
            axis.scatter([x], [y], color=colour, marker="X", s=40, zorder=5,
                         transform=transform, clip_on=False)
            axis.annotate(f"{arm} {100.0 * rate:.0f}%", (x, y), xycoords=transform,
                          xytext=(7, 0), textcoords="offset points", ha="left",
                          va="center", fontsize=6.5, color=colour)


def _failed_cells(selected, key):
    """Return (N, fail rate) for every cell of one arm drawn as a failure."""
    return [(int(cell["n_observers"]), cell["fail_rate"]["rate"]) for cell in selected
            if _is_failure(cell, cell[key]["median"])]


def _plot_arm(axis, selected, arm, key, scale, colour):
    x_values = [int(cell["n_observers"]) for cell in selected]
    axis.plot(x_values, _line_values(selected, key, scale), marker=MARKERS[arm], label=arm,
              color=colour, linestyle="--" if arm == "peer_naive" else "-",
              markersize=5, linewidth=1.6)


def _legend(figure, axis, extra_handles):
    handles, labels = axis.get_legend_handles_labels()
    handles = handles + extra_handles
    labels = labels + [handle.get_label() for handle in extra_handles]
    figure.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=8,
                  frameon=False)


def make_figures(directory: Path):
    """Write data-dependent PNG figures using the Horizon palette."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from tools import horizon_mpl

    pal = horizon_mpl.apply_style("light")
    colors = {
        "single": pal["accent"],
        "central": pal["ok"],
        "star_avg": pal["keepOut"],
        "peer_ci": pal["lineStrong"],
        "peer_naive": pal["text"],
    }
    failure_handle = Line2D([], [], color=pal["alert"], marker="X", linestyle="",
                            label="failed or diverged (arm and fail rate shown)")
    directory = Path(directory)
    with (directory / "per_seed.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("per_seed.csv has no rows")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    cells = summary["cells"]
    targets = sorted({cell["target"] for cell in cells})
    xticks = sorted({int(cell["n_observers"]) for cell in cells})

    figure, axes = plt.subplots(1, len(targets), figsize=(max(7.0, 4.2 * len(targets)), 4.6),
                                squeeze=False)
    for axis, target in zip(axes[0], targets):
        per_arm = {arm: _summary_cell(cells, target=target, loss=0.3, arm=arm) for arm in ARMS}
        plotted = [value for arm in ARMS
                   for value in _line_values(per_arm[arm], "p95_error_m", 100.0)
                   if not math.isnan(value)]
        ymax = max(plotted, default=1.0) * 1.35
        failures: dict[int, list] = {}
        for arm in ARMS:
            _plot_arm(axis, per_arm[arm], arm, "p95_error_m", 100.0, colors[arm])
            for x, rate in _failed_cells(per_arm[arm], "p95_error_m"):
                failures.setdefault(x, []).append((arm, rate))
        _mark_failures(axis, failures, pal["alert"])
        axis.set_ylim(0.0, ymax)
        axis.set_title(target)
        axis.set_xlabel("Observers N")
        axis.set_xticks(xticks)
        axis.grid(alpha=0.2)
    axes[0][0].set_ylabel("Median p95 error (cm)")
    _legend(figure, axes[0][0], [failure_handle])
    figure.suptitle("Simulated fusion error by target, noise 0.05 m, loss 0.3", fontsize=11)
    figure.tight_layout(rect=(0, 0.1, 1, 0.94))
    figure.savefig(directory / "fusion_error.png", dpi=150, metadata={"Software": None})
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.6), sharey=True)
    band_low, band_high = nis_band(10, 3, 0.95)
    for axis, loss in zip(axes, (0.0, 0.3)):
        failures = {}
        for arm in ARMS:
            selected = _summary_cell(cells, target="turn", loss=loss, arm=arm)
            _plot_arm(axis, selected, arm, "mean_nees", 1.0, colors[arm])
            for x, rate in _failed_cells(selected, "mean_nees"):
                failures.setdefault(x, []).append((arm, rate))
        axis.axhspan(band_low, band_high, color="lightgray", alpha=0.5,
                     label="95% band, 10-query window")
        axis.set_yscale("log")
        axis.set_ylim(*CONSISTENCY_YLIM)
        _mark_failures(axis, failures, pal["alert"])
        axis.set_title(f"Loss probability {loss:g}")
        axis.set_xlabel("Observers N")
        axis.set_xticks(xticks)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Median mean NEES (unitless, log scale)")
    _legend(figure, axes[0], [failure_handle])
    figure.suptitle("Simulated consistency, turn target, noise 0.05 m", fontsize=11)
    figure.tight_layout(rect=(0, 0.1, 1, 0.94))
    figure.savefig(directory / "fusion_consistency.png", dpi=150, metadata={"Software": None})
    plt.close(figure)
    return directory / "fusion_error.png", directory / "fusion_consistency.png"
