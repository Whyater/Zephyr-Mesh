"""Create figures from simulated cooperative fusion comparison data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from sim.estimation.consistency import nis_band

ARMS = ("single", "central", "star_avg", "peer_ci", "peer_naive")


def _summary_cell(cells, *, target, loss, arm):
    matches = [cell for cell in cells if cell["target"] == target
               and float(cell["noise_std_m"]) == 0.05
               and float(cell["loss_probability"]) == loss
               and cell["arm"] == arm and cell["ci"] == "fast"]
    if not matches:
        raise ValueError(f"missing summary cell for target={target}, loss={loss}, arm={arm}")
    return sorted(matches, key=lambda cell: int(cell["n_observers"]))


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
        "peer_naive": horizon_mpl.mix(pal["accent"], pal["keepOut"], 0.5),
    }
    directory = Path(directory)
    with (directory / "per_seed.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("per_seed.csv has no rows")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    cells = summary["cells"]
    targets = sorted({cell["target"] for cell in cells})
    figure, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.0), squeeze=False)
    for axis, target in zip(axes[0], targets):
        non_failure_values = []
        for arm in ARMS[:-1]:
            selected = _summary_cell(cells, target=target, loss=0.3, arm=arm)
            non_failure_values.extend(cell["p95_error_m"]["median"] for cell in selected
                                      if cell["p95_error_m"]["median"] is not None)
        ymax = max(non_failure_values, default=1.0) * 100.0 * 1.25
        for arm in ARMS:
            selected = _summary_cell(cells, target=target, loss=0.3, arm=arm)
            x_values = [cell["n_observers"] for cell in selected]
            y_values = [None if cell["p95_error_m"]["median"] is None
                        else 100.0 * cell["p95_error_m"]["median"] for cell in selected]
            if arm == "peer_naive":
                points = []
                for cell, value in zip(selected, y_values):
                    failed = (cell["fail_rate"]["rate"] > 0.0 or
                              cell["diverged_count"] > 0 or value is None)
                    points.append(ymax * 0.94 if failed else value)
                axis.plot(x_values, points, marker="o", label=arm, color=colors[arm])
                for x, cell in zip(x_values, selected):
                    failed = (cell["fail_rate"]["rate"] > 0.0 or
                              cell["diverged_count"] > 0 or cell["p95_error_m"]["median"] is None)
                    if failed:
                        rate = 100.0 * cell["fail_rate"]["rate"]
                        axis.scatter([x], [ymax * 0.94], color=pal["alert"], marker="X", zorder=5)
                        axis.annotate(f"diverged / failed\n{rate:.0f}%", (x, ymax * 0.94),
                                      xytext=(0, 8), textcoords="offset points", ha="center",
                                      fontsize=7, color=pal["alert"])
            else:
                axis.plot(x_values, y_values, marker="o", label=arm, color=colors[arm])
        axis.set_ylim(0.0, ymax)
        axis.set_title(target)
        axis.set_xlabel("Observers N")
        axis.set_xticks(sorted({int(cell["n_observers"]) for cell in cells}))
        axis.grid(alpha=0.2)
    axes[0][0].set_ylabel("Median p95 error (cm)")
    axes[0][-1].legend(fontsize=8)
    figure.suptitle("Simulated fusion error, turn setup by target, noise 0.05 m, loss 0.3")
    figure.tight_layout()
    figure.savefig(directory / "fusion_error.png", dpi=150, metadata={"Software": None})
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    band_low, band_high = nis_band(10, 3, 0.95)
    for axis, loss in zip(axes, (0.0, 0.3)):
        for arm in ARMS:
            selected = _summary_cell(cells, target="turn", loss=loss, arm=arm)
            x_values = [cell["n_observers"] for cell in selected]
            y_values = [cell["mean_nees"]["median"] for cell in selected]
            valid = [(x, y) for x, y in zip(x_values, y_values) if y is not None]
            if valid:
                axis.plot([item[0] for item in valid], [item[1] for item in valid],
                          marker="o", label=arm, color=colors[arm])
        axis.axhspan(band_low, band_high, color="lightgray", alpha=0.5,
                     label="95% band, 10-query window")
        axis.set_yscale("log")
        axis.set_title(f"Loss probability {loss:g}")
        axis.set_xlabel("Observers N")
        axis.set_xticks(sorted({int(cell["n_observers"]) for cell in cells}))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Median mean NEES (unitless)")
    handles, labels = axes[-1].get_legend_handles_labels()
    band_handle = Line2D([], [], color="lightgray", linewidth=8, alpha=0.5,
                         label="95% band, 10-query window")
    axes[-1].legend(handles + [band_handle], labels + ["95% band, 10-query window"], fontsize=8)
    figure.suptitle("Simulated consistency, turn setup, noise 0.05 m")
    figure.tight_layout()
    figure.savefig(directory / "fusion_consistency.png", dpi=150, metadata={"Software": None})
    plt.close(figure)
    return directory / "fusion_error.png", directory / "fusion_consistency.png"
