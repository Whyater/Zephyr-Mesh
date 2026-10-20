"""Create simulated decision-authority figures from runner output files."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

KINDS = ("state", "raw", "track", "command", "leaving")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _traffic_sizes(directory: Path) -> dict[str, int]:
    """Read simulator message sizes from traffic metadata, without hard-coded sizes."""
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        traffic = summary.get("traffic", {})
        if isinstance(traffic, dict):
            sizes = {
                kind: int(traffic[kind]["message_bytes"])
                for kind in KINDS
                if isinstance(traffic.get(kind), dict) and "message_bytes" in traffic[kind]
            }
            if sizes:
                return sizes
    raise ValueError("summary.json must include traffic[kind].message_bytes")


def _kind_rate(row: dict[str, str], kind: str) -> float:
    """Read a per-kind bytes-per-second value from one per-seed row."""
    name = f"{kind}_bytes_per_second"
    return float(row[name]) if row.get(name, "") not in (None, "") else 0.0


def _make_outage(
    directory: Path,
    rows: list[dict[str, str]],
    style: dict[str, Any],
    loss: float = 0.1,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    timeline = _read_csv(directory / "timeline.csv")
    available = {
        float(row["loss"])
        for row in timeline
        if row.get("outage") == "10_15"
    }
    if loss not in available:
        raise ValueError(f"timeline.csv is missing outage loss={loss:g}")
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for mode, color in (("ground", style["accent"]), ("onboard", style["ok"])):
        selected = [
            row
            for row in timeline
            if row.get("mode") == mode
            and float(row["loss"]) == loss
            and row.get("outage") == "10_15"
        ]
        if not selected:
            raise ValueError(f"timeline.csv is missing outage cell mode={mode}, loss={loss:g}")
        selected.sort(key=lambda row: float(row["time_s"]))
        axis.plot(
            [float(row["time_s"]) for row in selected],
            [float(row["standoff_error_m"]) for row in selected],
            label=mode,
            color=color,
        )
    axis.axvspan(10.0, 15.0, color="0.75", alpha=0.45, zorder=0)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.75", edgecolor="none", alpha=0.45))
    labels.append("ground link down")
    axis.legend(handles, labels, frameon=False)
    axis.set(
        xlabel="time (s)",
        ylabel="median standoff error (m)",
        title=f"Simulated outage response by decision mode (loss={loss:g})",
    )
    fig.tight_layout()
    fig.savefig(directory / "decisions_outage.png", dpi=150, metadata={"Software": None})
    plt.close(fig)


def _make_traffic(directory: Path, rows: list[dict[str, str]], style: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    sizes = _traffic_sizes(directory)
    modes = ("ground", "onboard")
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(modes))
    bottom = np.zeros(len(modes), dtype=float)
    plotted_kinds: list[str] = []
    colors = {
        "state": style["accent"],
        "raw": style["ok"],
        "track": style["target"],
        "command": style["keepOut"],
        "leaving": style["lineStrong"],
    }
    for kind in KINDS:
        values = [
            float(np.median([_kind_rate(row, kind) for row in rows if row.get("mode") == mode]))
            if any(row.get("mode") == mode for row in rows)
            else 0.0
            for mode in modes
        ]
        if not any(values):
            continue
        axis.bar(x, values, bottom=bottom, label=kind, color=colors.get(kind, "0.5"))
        bottom += np.asarray(values)
        plotted_kinds.append(kind)
    if not plotted_kinds:
        raise ValueError("per_seed.csv has no per-kind bytes-per-second columns")
    axis.set_xticks(x, modes)
    axis.set_ylabel("bytes per second")
    cells = {
        (row.get("mode"), row.get("loss"), row.get("outage"))
        for row in rows
    }
    axis.set_title(
        "Simulated traffic by decision mode\n"
        f"N = 4, each bar pools that mode's {len(cells) // 2} loss and outage cells"
    )
    size_note = ", ".join(f"{kind} {sizes[kind]} B" for kind in plotted_kinds if kind in sizes)
    if size_note:
        axis.set_xlabel(f"decision mode, simulator message sizes: {size_note}")
    else:
        axis.set_xlabel("decision mode")
    axis.legend(frameon=False, title="message kind")
    fig.tight_layout()
    fig.savefig(directory / "decisions_traffic.png", dpi=150, metadata={"Software": None})
    plt.close(fig)


def make_figures(directory: Path, outage_loss: float = 0.1) -> None:
    """Render outage and stacked traffic figures from a completed runner output."""
    import matplotlib

    matplotlib.use("Agg")
    from tools.horizon_mpl import apply_style

    output = Path(directory)
    style = apply_style("light")
    rows = _read_csv(output / "per_seed.csv")
    _make_outage(output, rows, style, loss=outage_loss)
    _make_traffic(output, rows, style)
