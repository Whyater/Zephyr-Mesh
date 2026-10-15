"""Generate deterministic telemetry and figures for the three S0 v2 cases."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.baseline import KNOWN_LIMITS, run_scenario_v2, scenario_v2_spec

# The v1 limits describe the v1 baseline; this run adds the heading setpoint and
# the sideways and yaw cases, so two of them are replaced here.
_V2_REPLACED = {
    'No heading setpoint: the controller damps yaw rate but does not hold or command a yaw angle.':
        'Optional heading setpoint: the controller holds a yaw angle only when a target yaw is given (this run commands a 90 degree yaw step); with no target yaw it damps yaw rate only.',
    'This artifact only runs the vertical step to (0, 0, 3) m; sideways and yawed cases are covered by tests, not by this run.':
        'This artifact runs three cases: a vertical step to (0, 0, 3) m, a 1 m sideways step, and a 90 degree yaw step. Other cases are covered by tests, not by this run.',
}
V2_LIMITS = [_V2_REPLACED.get(item, item) for item in KNOWN_LIMITS]
from sim.metrics import step_response
from sim.physics.run_manifest import write_manifest
from tools import horizon_mpl


SCENARIOS = ("vertical", "lateral", "yaw")
CSV_COLUMNS = (
    "time", "x", "y", "z", "vx", "vy", "vz",
    "qw", "qx", "qy", "qz", "p", "q", "r",
)


def _write_csv(path: Path, telemetry: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for row in zip(*(telemetry[name] for name in CSV_COLUMNS)):
            writer.writerow([f"{float(value):.17g}" for value in row])


def _yaw(telemetry: dict[str, np.ndarray]) -> np.ndarray:
    qw, qx, qy, qz = (telemetry[key] for key in ("qw", "qx", "qy", "qz"))
    return np.unwrap(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def _metric(name: str, telemetry: dict[str, np.ndarray]) -> dict:
    if name == "vertical":
        signal, goal = telemetry["z"], 3.0
    elif name == "lateral":
        signal, goal = telemetry["x"], 1.0
    else:
        signal, goal = _yaw(telemetry), float(np.pi / 2.0)
    result = step_response(telemetry["time"], signal, goal)
    result["final_error"] = float(goal - signal[-1]) if len(signal) else None
    return result


def _figure(telemetry: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    horizon_mpl.apply_style("light")
    pal = horizon_mpl.palette("light")
    colors = [pal["accent"], pal["info"], pal["target"]]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=False, constrained_layout=True)
    labels = {"vertical": ("z (m)", 3.0), "lateral": ("x (m)", 1.0), "yaw": ("yaw (rad)", np.pi / 2.0)}
    for axis, name, color in zip(axes, SCENARIOS, colors):
        signal = telemetry[name]["z"] if name == "vertical" else telemetry[name]["x"] if name == "lateral" else _yaw(telemetry[name])
        axis.plot(telemetry[name]["time"], signal, color=color, linewidth=1.4, label=name)
        axis.axhline(labels[name][1], color=pal["lineStrong"], linewidth=0.8, linestyle="--")
        axis.set_ylabel(labels[name][0])
        axis.set_title(name.capitalize())
        axis.legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def generate(output_dir: Path, *, dt: float = 0.01, overwrite: bool = False) -> list[Path]:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to write into non-empty directory: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    telemetry = {}
    written = []
    for name in SCENARIOS:
        telemetry[name] = run_scenario_v2(name, dt)
        path = output_dir / f"{name}.csv"
        _write_csv(path, telemetry[name])
        written.append(path)
    report = {
        "schema": "zephyr-s0-baseline-v2-report-1",
        "dt_s": float(dt),
        "scenarios": {name: _metric(name, telemetry[name]) for name in SCENARIOS},
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(report_path)
    figure_path = output_dir / "a_s0_baseline_v2.png"
    _figure(telemetry, figure_path)
    written.append(figure_path)
    runtime = time.perf_counter() - started
    write_manifest(
        output_dir / "manifest.json",
        schema="zephyr-s0-baseline-v2-manifest-1",
        argv=sys.argv[1:],
        config={"dt_s": float(dt), "scenarios": {name: scenario_v2_spec(name, dt) for name in SCENARIOS}},
        seeds=[],
        limits=V2_LIMITS,
        outputs=written,
        output_dir=output_dir,
        replay_command="PYTHONPATH=. " + " ".join(
            shlex.quote(str(argument)) for argument in [sys.executable, *sys.argv]
        ),
        runtime_s=runtime,
    )
    return written + [output_dir / "manifest.json"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args(argv)
    generate(args.output_dir, dt=args.dt, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
