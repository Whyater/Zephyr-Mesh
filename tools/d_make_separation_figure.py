"""Create the simulated separation-buffer figures from runner outputs."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

MIN_SEPARATION_M = 0.35
CONTROL_DT_S = 0.05


def _number(row: dict[str, Any], *names: str) -> float | None:
    """Return the first present finite numeric field."""
    for name in names:
        value = row.get(name)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def _boolean(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _dead_reckoning(row: dict[str, Any]) -> bool:
    return _boolean(row.get("dead_reckoning", False))


def _radius(row: dict[str, Any]) -> float | None:
    return _number(row, "avoidance_radius_m")


def _is_baseline(row: dict[str, Any]) -> bool:
    label = str(row.get("pass", "")).strip().lower()
    radius = _radius(row)
    return label in {"baseline", "0", "radius_0.35", "radius0.35"} or (
        radius is not None and abs(radius - MIN_SEPARATION_M) <= 1.0e-9
    )


def _is_censored(entry: dict[str, Any]) -> bool:
    return _boolean(entry.get("censored", False)) or entry.get("required_buffer_m") is None


def _buffer_value(entry: dict[str, Any]) -> float | None:
    value = _number(entry, "required_buffer_m")
    if value is not None:
        return max(0.0, value)
    # Censored entries have no measured buffer.  Their upper marker is drawn
    # separately, so a search limit is never turned into a data value.
    return None


def _envelope_points(data: dict[str, Any]) -> dict[bool, dict[float, dict[float, dict[str, Any]]]]:
    """Return [dead reckoning][speed][delay] worst-case envelope points."""
    points: dict[bool, dict[float, dict[float, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for entry in data.get("envelope", []):
        speed = _number(entry, "speed_mps")
        delay = _number(entry, "delay_s")
        if speed is None or delay is None:
            continue
        dr = _dead_reckoning(entry)
        value = _buffer_value(entry)
        old = points[dr].get(float(speed), {}).get(float(delay))
        current = {"value": value, "censored": _is_censored(entry)}
        old_value = None if old is None else old["value"]
        should_replace = old is None
        if value is not None and (old_value is None or value > float(old_value)):
            should_replace = True
        if current["censored"] and old is not None and not old["censored"]:
            should_replace = True
        if should_replace:
            points[dr][float(speed)][float(delay)] = current
    return points


def _hand_value(entry: dict[str, Any], speed: float, delay: float) -> float:
    value = _number(entry, "hand_buffer_m")
    if value is not None:
        return max(0.0, value)
    return 2.0 * speed * (delay + CONTROL_DT_S)


def _hand_points(data: dict[str, Any], speed: float, delays: list[float]) -> list[float]:
    values: list[float] = []
    for delay in delays:
        candidates = []
        for entry in data.get("entries", []):
            if str(entry.get("kind", "")) != "head_on":
                continue
            e_speed = _number(entry, "speed_mps")
            e_delay = _number(entry, "delay_s")
            if (
                e_speed is None or e_delay is None
                or abs(e_speed - speed) > 1.0e-9
                or abs(e_delay - delay) > 1.0e-9
            ):
                continue
            if _dead_reckoning(entry):
                continue
            loss = _number(entry, "loss_probability")
            noise = _number(entry, "noise_std_m")
            if loss not in (None, 0.0) or noise not in (None, 0.0):
                continue
            candidates.append(_hand_value(entry, speed, delay))
        values.append(float(candidates[0]) if candidates else 2.0 * speed * (delay + CONTROL_DT_S))
    return values


def _baseline_rows(directory: Path) -> list[dict[str, Any]]:
    with (directory / "encounters.csv").open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return [row for row in rows if _is_baseline(row)]


def _available_baseline_settings(rows: list[dict[str, Any]]) -> list[tuple[float, float, bool]]:
    settings = {
        (_number(row, "loss_probability"), _number(row, "noise_std_m"), _dead_reckoning(row))
        for row in rows
    }
    return sorted(
        (float(loss), float(noise), dr)
        for loss, noise, dr in settings
        if loss is not None and noise is not None
    )


def _baseline_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    violations = 0.0
    trials = 0.0
    for row in rows:
        count = _number(row, "violations")
        n = _number(row, "seeds_run")
        if count is not None and n is not None and n > 0:
            violations += count
            trials += n
        else:
            violations += float(_boolean(row.get("violated", row.get("violation", False))))
            trials += 1.0
    return violations / trials if trials else float("nan")


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=150, metadata={"Software": None})


def _buffer_figure(directory: Path, data: dict[str, Any], palette: dict[str, str]) -> list[str]:
    import matplotlib.pyplot as plt

    points = _envelope_points(data)
    all_speeds = sorted({speed for by_speed in points.values() for speed in by_speed})
    if not all_speeds:
        raise ValueError("buffer.json contains no envelope points")
    palette_colors = (
        palette["accent"], palette["target"], palette["ok"], palette["keepOut"]
    )
    colors = {
        speed: color for speed, color in zip(all_speeds, palette_colors)
    }
    for speed in all_speeds:
        colors.setdefault(speed, palette["accent"])
    delays = sorted(
        {
            delay
            for by_speed in points.values()
            for by_delay in by_speed.values()
            for delay in by_delay
        }
    )
    finite_values = [
        float(item["value"])
        for by_speed in points.values()
        for by_delay in by_speed.values()
        for item in by_delay.values()
        if item["value"] is not None
    ]
    top = max(max(finite_values + [0.1]) * 1.12, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for ax, dr in zip(axes, (False, True)):
        for speed in all_speeds:
            series = points.get(dr, {}).get(speed, {})
            xs = [delay for delay in delays if delay in series]
            ys = [
                float(series[delay]["value"])
                if series[delay]["value"] is not None else np.nan
                for delay in xs
            ]
            censored = [bool(series[delay]["censored"]) for delay in xs]
            if not xs:
                continue
            color = colors[speed]
            ax.plot(xs, ys, marker="o", linewidth=1.6, color=color, label=f"{speed:g} m/s")
            if any(censored):
                ax.scatter(
                    [x for x, c in zip(xs, censored) if c],
                    [top] * sum(censored),
                    marker="^",
                    s=48,
                    facecolors=palette["caution"],
                    edgecolors=palette["caution"],
                    label="censored",
                    zorder=4,
                )
            if not dr:
                hand = _hand_points(data, speed, xs)
                ax.plot(
                    xs, hand, linestyle="--", linewidth=0.9, color=color, alpha=0.8,
                    label="hand model (no dead reckoning)",
                )
        ax.set_xlabel("one-way delay (s)")
        ax.set_xticks(delays)
        ax.set_title("Dead reckoning on" if dr else "Dead reckoning off")
        ax.set_ylim(bottom=0.0, top=top)
        ax.grid(True)
    axes[0].set_ylabel("required buffer (m)")
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.suptitle(
        "Simulated required separation buffer\n"
        "Worst case over encounter type, loss, and noise"
    )
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.88))
    _save(fig, directory / "separation_buffer.png")
    plt.close(fig)
    return labels


def _baseline_figure(directory: Path, rows: list[dict[str, Any]], palette: dict[str, str]) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("encounters.csv contains no baseline rows")
    settings = _available_baseline_settings(rows)
    preferred = (0.3, 0.05, False)
    if preferred in settings:
        setting = preferred
    else:
        pilot_run = any(row.get("radius_status") == "not passed (pilot)" for row in rows)
        if not pilot_run:
            raise ValueError("encounters.csv is missing the requested baseline setting")
        setting = settings[0]
    loss, noise, dr = setting
    palette_colors = (
        palette["accent"], palette["target"], palette["ok"], palette["keepOut"]
    )
    selected = [
        row for row in rows
        if abs((_number(row, "loss_probability") or 0.0) - loss) <= 1.0e-9
        and abs((_number(row, "noise_std_m") or 0.0) - noise) <= 1.0e-9
        and _dead_reckoning(row) == dr
    ]
    kinds = sorted({str(row.get("kind", "unknown")) for row in selected})
    delays = sorted(
        {
            _number(row, "delay_s")
            for row in selected
            if _number(row, "delay_s") is not None
        }
    )
    colors = {
        kind: color for kind, color in zip(kinds, palette_colors)
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for kind in kinds:
        values = [
            _baseline_rate([
                row for row in selected
                if str(row.get("kind")) == kind
                and abs((_number(row, "delay_s") or 0.0) - delay) <= 1.0e-9
            ])
            for delay in delays
        ]
        ax.plot(delays, values, marker="o", linewidth=1.6, color=colors[kind], label=kind.replace("_", " "))
    ax.axhline(0.05, color=palette["lineStrong"], linestyle="--", linewidth=1.0, label="5% rule")
    ax.set_xlabel("one-way delay (s)")
    ax.set_ylabel("baseline violation rate")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_xticks(delays)
    setting_label = f"loss {loss:g}, noise {noise:g} m, dead reckoning {'on' if dr else 'off'}"
    suffix = "" if setting == preferred else " (pilot setting shown)"
    ax.set_title(
        f"Simulated baseline violation rate at 0.35 m\n"
        f"Speeds and head-on offsets pooled; {setting_label}{suffix}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, directory / "separation_baseline.png")
    plt.close(fig)


def make_figures(directory: Path) -> None:
    """Render both figures from ``buffer.json`` and ``encounters.csv``."""
    import matplotlib

    matplotlib.use("Agg")
    from tools.horizon_mpl import apply_style

    directory = Path(directory)
    palette = apply_style("light")
    with (directory / "buffer.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    _buffer_figure(directory, data, palette)
    _baseline_figure(directory, _baseline_rows(directory), palette)


if __name__ == "__main__":
    import sys

    make_figures(Path(sys.argv[1]))
