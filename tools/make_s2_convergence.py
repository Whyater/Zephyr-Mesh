"""Regenerate runs/s2-physics/convergence.json from the real simulator.

System integrated: ``sim.drone.SixDOFInterceptor``, the full 13-state rigid
body (position, velocity, attitude quaternion, body rates) with gravity,
quadratic drag against a steady wind, a tilted starting attitude, a spinning
body, and a constant body-frame force and torque.  The inputs are held
constant for the whole run, so the only error measured is the RK4 stepping
error, not controller sampling.

There is no closed form for this case, so each fixed-step run is compared
with the same integrator at a much finer step (the reference).  For a
fourth-order method, halving the step divides the error by about 2**4 = 16.

Run from the repository root:
    PYTHONPATH=. ./.venv/bin/python tools/make_s2_convergence.py
This is a numerical check of the integrator. It is not a flight-performance
measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.drone import SixDOFInterceptor
from sim.environment import GRAVITY

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runs" / "s2-physics" / "convergence.json"
DURATION_S = 1.0
STEPS_S = (0.1, 0.05, 0.025, 0.0125)
REFERENCE_DIVISOR = 50  # reference step = smallest step / 50

MASS_KG = 0.45
WIND_MPS = (1.0, 0.5, 0.0)
INITIAL_POSITION_M = (0.0, 0.0, 1.0)
INITIAL_VELOCITY_MPS = (3.0, -1.0, 2.0)
INITIAL_TILT_AXIS = (1.0, 2.0, 3.0)
INITIAL_TILT_RAD = 0.4
INITIAL_BODY_RATE_RAD_S = (0.4, -0.3, 0.8)
BODY_FORCE_N = (0.0, 0.0, 1.1 * MASS_KG * GRAVITY)
BODY_TORQUE_NM = (0.002, -0.001, 0.0005)


def _initial_drone() -> SixDOFInterceptor:
    drone = SixDOFInterceptor(init_pos=INITIAL_POSITION_M, mass=MASS_KG, wind_velocity=WIND_MPS)
    drone.velocity[:] = INITIAL_VELOCITY_MPS
    axis = np.asarray(INITIAL_TILT_AXIS, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = INITIAL_TILT_RAD / 2.0
    drone.quaternion[:] = [np.cos(half), *(np.sin(half) * axis)]
    drone.angular_velocity[:] = INITIAL_BODY_RATE_RAD_S
    return drone


def final_state(dt_s: float) -> np.ndarray:
    """Integrate for DURATION_S with fixed step dt_s and return the 13-state vector."""
    drone = _initial_drone()
    force = np.asarray(BODY_FORCE_N, dtype=float)
    torque = np.asarray(BODY_TORQUE_NM, dtype=float)
    for _ in range(round(DURATION_S / dt_s)):
        drone.step_physics(force, torque, dt_s)
    return drone._state_vector()


def build_convergence() -> dict:
    probe = _initial_drone()
    reference_dt = STEPS_S[-1] / REFERENCE_DIVISOR
    reference = final_state(reference_dt)
    errors = [float(np.linalg.norm(final_state(dt) - reference)) for dt in STEPS_S]
    results = []
    for index, (dt, error) in enumerate(zip(STEPS_S, errors)):
        row = {"dt_s": dt, "absolute_error": error}
        if index + 1 < len(errors):
            row["error_ratio_to_next"] = error / errors[index + 1]
        results.append(row)
    return {
        "schema": "zephyr-s2-convergence-2",
        "fixture": "sim.drone.SixDOFInterceptor full 13-state rigid body, RK4 fixed step vs finer RK4 reference",
        "generator": "tools/make_s2_convergence.py",
        "duration_s": DURATION_S,
        "reference_dt_s": reference_dt,
        "error_metric": "Euclidean norm of the final 13-state difference from the reference",
        "inputs": {
            "mass_kg": MASS_KG,
            "inertia_kg_m2": [probe.I_x, probe.I_y, probe.I_z],
            "drag_coefficient": probe.drag_coefficient,
            "cross_sectional_area_m2": probe.cross_sectional_area,
            "wind_velocity_mps": list(WIND_MPS),
            "initial_position_m": list(INITIAL_POSITION_M),
            "initial_velocity_mps": list(INITIAL_VELOCITY_MPS),
            "initial_tilt_axis": list(INITIAL_TILT_AXIS),
            "initial_tilt_rad": INITIAL_TILT_RAD,
            "initial_body_rate_rad_s": list(INITIAL_BODY_RATE_RAD_S),
            "body_force_n": list(BODY_FORCE_N),
            "body_torque_nm": list(BODY_TORQUE_NM),
        },
        "results": results,
        "interpretation": (
            "A ratio near 16 per halving of dt is the signature of a fourth-order "
            "integrator. These are numerical integration errors of a synthetic "
            "model, not flight-performance measurements."
        ),
    }


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="write convergence.json beneath this directory (default behavior is unchanged)",
    )
    args = parser.parse_args(argv)
    output = args.output if args.output_dir is None else args.output_dir / "convergence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_convergence(), indent=2) + "\n")
    print(output)
    return output


if __name__ == "__main__":
    main()
