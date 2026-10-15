"""Long-run check of the delay limit near the EXP-021 rule edge.

The pre-registered instability rule looks at a 20 s run and needs the x error
to grow by 1.2 times between the middle and last third. A loop that is only
just unstable grows slowly, so it can pass that rule. This tool flies 80 s
runs near the limit, for a 1 m step and a 1 cm step (small enough to stay in
the linear range), and records, for each 10 s window, the peak 3-D position
error, the peak tilt and the lowest height. A peak error that rises from
window to window means the swing is growing, whatever the 20 s rule said; a
large steady tilt or a falling height means the drone is not flying properly
even if the error does not grow.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sim.controller import GeometricFlightController
from sim.hardware import load_hardware_profile
from sim.physics.control_tolerance import CONTROL_DT_S, _controller_for, _tilt_angle
from sim.physics.run_manifest import write_manifest
from sim.vehicle import ActuatedQuadrotor

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "crazyflie-class": ROOT / "sim/physics/profiles/crazyflie_class.json",
    "esp-fly-class": ROOT / "sim/physics/profiles/esp_fly_class.json",
}
DURATION_S = 80.0
WINDOW_S = 10.0
STEP_START_S = 1.0
# Delays on the 10 ms control grid around each limit.
JOBS = [(arm, delay, amp) for arm, delays in (("b", (0.32, 0.33, 0.34, 0.35, 0.36)), ("c", (0.02, 0.03)))
        for delay in delays for amp in (1.0, 0.01)]


def fly(job):
    profile_id, arm, delay_s, amplitude_m = job
    profile = load_hardware_profile(PROFILES[profile_id])
    vehicle = ActuatedQuadrotor(profile, initial_position=(0.0, 0.0, 1.0),
                                controller=GeometricFlightController(), substeps=1)
    vehicle.actuator.speed_rad_s[:] = profile.hover_speed_rad_s()
    vehicle.controller = _controller_for(arm, delay_s, 0.0, np.random.default_rng(0), vehicle.drone)
    steps = int(round(DURATION_S / CONTROL_DT_S))
    per_window = int(round(WINDOW_S / CONTROL_DT_S))
    errors, tilts, heights = [], [], []
    stopped_at = None
    for k in range(steps):
        t = k * CONTROL_DT_S
        goal_x = amplitude_m if t >= STEP_START_S else 0.0
        vehicle.step((goal_x, 0.0, 1.0), CONTROL_DT_S)
        position = vehicle.drone.position
        error = float(np.linalg.norm(position - np.array((goal_x, 0.0, 1.0))))
        if not math.isfinite(error) or error > 5.0 * max(amplitude_m, 1.0):
            stopped_at = (k + 1) * CONTROL_DT_S
            break
        errors.append(error)
        tilts.append(math.degrees(_tilt_angle(vehicle.drone)))
        heights.append(float(position[2]))
    errors, tilts, heights = np.asarray(errors), np.asarray(tilts), np.asarray(heights)
    peaks, tilt_peaks, lowest = [], [], []
    for w in range(steps // per_window):
        sl = slice(w * per_window, (w + 1) * per_window)
        full = errors[sl].size == per_window
        peaks.append(float(np.max(errors[sl])) if full else None)
        tilt_peaks.append(float(np.max(tilts[sl])) if full else None)
        lowest.append(float(np.min(heights[sl])) if full else None)
    # Compare the last full window with the third one (after the step transient).
    finite = [p for p in peaks if p is not None]
    growth = finite[-1] / peaks[2] if len(finite) == len(peaks) and peaks[2] else None
    return {"profile": profile_id, "arm": arm, "delay_s": delay_s, "step_m": amplitude_m,
            "window_peak_error_m": peaks, "window_peak_tilt_deg": tilt_peaks, "window_lowest_z_m": lowest,
            "stopped_at_s": stopped_at,
            "last_over_third_window_peak": growth,
            "growing": bool(stopped_at is not None or (growth is not None and growth > 1.0))}


def _round(value):
    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    jobs = [(p, arm, d, a) for p in sorted(PROFILES) for arm, d, a in JOBS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(fly, jobs))
    rows.sort(key=lambda r: (r["profile"], r["arm"], r["delay_s"], -r["step_m"]))
    report = _round({"schema": "zephyr-exp-021-long-check-1", "duration_s": DURATION_S, "window_s": WINDOW_S,
                     "note": "3-D position error; growing means the last 10 s window peak exceeds the third window peak, or the run left the 5 m box (5 x step size for steps above 1 m)",
                     "rows": rows})
    (out / "long_check.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out / "manifest.json", schema="zephyr-exp-021-long-check-manifest-1", argv=sys.argv[1:],
                   config={"duration_s": DURATION_S, "window_s": WINDOW_S, "jobs": [list(j) for j in JOBS]},
                   seeds=[0], limits=["simulated only", "literature-derived profiles", "loss 0 only", "no floor modeled"],
                   outputs={"long_check.json": out / "long_check.json"}, output_dir=out,
                   runtime_s=time.perf_counter() - started, n_processes=args.workers, repo=ROOT)


if __name__ == "__main__":
    main()
