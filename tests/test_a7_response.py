"""Fast checks for the EXP-022 disturbance response measurements tool.

These checks exercise the simulator directly for the hand comparisons and
invoke the tool's reduced pilot sweep for reproducibility.  All vehicles are
started with their rotors already at the profile hover speed, as in the
registered experiment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from sim.environment import AIR_DENSITY, GRAVITY
from sim.hardware import load_hardware_profile
from sim.interaction import downwash_wind_at
from sim.physics.disturbances import ConstantWind, GroundEffect
from sim.vehicle import ActuatedQuadrotor


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "sim" / "physics" / "profiles" / "crazyflie_class.json"


def _profile():
    return load_hardware_profile(PROFILE_PATH)


def _trimmed_vehicle(*, initial_position=(0.0, 0.0, 1.0), disturbance=None):
    profile = _profile()
    # Cd=1.0 and the explicit area avoid the constructor's legacy sentinel
    # replacement for the values 1.3 and 0.05.
    vehicle = ActuatedQuadrotor(
        profile,
        initial_position=initial_position,
        drag_coefficient=1.0,
        cross_sectional_area=0.005,
        substeps=1,
        disturbance=disturbance,
    )
    vehicle.actuator.speed_rad_s[:] = profile.hover_speed_rad_s()
    return profile, vehicle


def _simulate(vehicle, target, seconds):
    samples = []
    for _ in range(int(round(seconds / 0.01))):
        samples.append(np.asarray(vehicle.step(target, 0.01).position_m, dtype=float))
    return np.asarray(samples)


def test_steady_wind_offset_matches_hand_formula():
    profile, vehicle = _trimmed_vehicle(
        disturbance=ConstantWind((1.0, 0.0, 0.0)),
    )
    trace = _simulate(vehicle, (0.0, 0.0, 1.0), 20.0)
    simulated_offset = float(np.mean(trace[-500:, 0]))
    kp_xy = float(vehicle.controller.K_p_pos[0])
    hand_offset = 0.5 * AIR_DENSITY * 0.005 * 1.0**2 / (
        profile.total_mass_kg * kp_xy
    )
    np.testing.assert_allclose(simulated_offset, hand_offset, rtol=0.10, atol=1e-6)


def test_zero_wind_preserves_trim():
    _, vehicle = _trimmed_vehicle()
    trace = _simulate(vehicle, (0.0, 0.0, 1.0), 1.0)
    np.testing.assert_allclose(trace[-1], (0.0, 0.0, 1.0), atol=1e-6)


def _ground_effect_hand_offset(target_z: float, rotor_radius: float, kp_z: float) -> float:
    """Solve the steady ground-effect balance used by this test.

    If ``e = z - target_z`` is the upward offset, the controller's commanded
    collective is ``m * (g - Kp_z*e)`` because its position error is
    ``target_z-z``.  The vehicle then multiplies that command by the ground
    scale ``s(z)`` and does not divide the command by ``s``.  Balancing weight
    therefore gives ``e = g*(1 - 1/s(target_z+e))/Kp_z``, with
    ``s(z) = 1/(1-(R/(4z))**2)``.  The implicit dependence on final height is
    retained rather than evaluating the scale only at the target.
    """

    def residual(offset: float) -> float:
        height = target_z + offset
        scale = 1.0 / (1.0 - (rotor_radius / (4.0 * height)) ** 2)
        return offset - GRAVITY * (1.0 - 1.0 / scale) / kp_z

    # The requested heights are at least 2R, so the physical root is the small
    # positive root.  A fixed bisection keeps this test NumPy-only.
    lo, hi = 0.0, 0.25
    assert residual(lo) < 0.0 < residual(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def test_ground_effect_offset_matches_derived_hand_balance():
    profile = _profile()
    target_z = 2.0 * profile.rotor.prop_radius_m
    _, vehicle = _trimmed_vehicle(
        initial_position=(0.0, 0.0, target_z),
        disturbance=GroundEffect(profile.rotor.prop_radius_m),
    )
    trace = _simulate(vehicle, (0.0, 0.0, target_z), 10.0)
    simulated_offset = float(np.mean(trace[-300:, 2]) - target_z)
    hand_offset = _ground_effect_hand_offset(
        target_z,
        profile.rotor.prop_radius_m,
        float(vehicle.controller.K_p_pos[2]),
    )
    np.testing.assert_allclose(simulated_offset, hand_offset, rtol=0.10, atol=1e-5)


def test_downwash_on_axis_matches_momentum_hand_value():
    profile = _profile()
    thrust = profile.total_mass_kg * GRAVITY
    gap = 0.4
    wind = downwash_wind_at(
        (0.0, 0.0, 1.0 + gap),
        thrust,
        profile.rotor.count,
        profile.rotor.prop_radius_m,
        (0.0, 0.0, 1.0),
    )
    disk_area = profile.rotor.count * np.pi * profile.rotor.prop_radius_m**2
    hover_induced_speed = np.sqrt(thrust / (2.0 * AIR_DENSITY * disk_area))
    hand_wind = -2.0 * hover_induced_speed
    np.testing.assert_allclose(wind[2], hand_wind, rtol=0.01, atol=1e-12)


def _run_pilot(output_dir: Path, workers: int) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    # Keep the subprocess independent of a read-only user Matplotlib cache.
    # This is an environment setting for the test run, not a simulator input.
    mpl_config = output_dir.parent / "mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    env["MPL_IGNORE_SYSTEM_FONTS"] = "1"
    env["MPLCONFIGDIR"] = str(mpl_config)
    env["MPLBACKEND"] = "Agg"
    command = [
        sys.executable,
        "tools/a_run_disturbance_response.py",
        "--output-dir",
        str(output_dir),
        "--workers",
        str(workers),
        "--pilot",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "pilot tool failed (stdout/stderr):\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )


def _output_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in sorted(path.iterdir())
        if item.is_file() and item.name != "manifest.json"
    }


def test_pilot_output_is_worker_count_independent(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    third = tmp_path / "third"
    _run_pilot(first, workers=1)
    _run_pilot(second, workers=1)
    _run_pilot(third, workers=2)

    first_files = _output_bytes(first)
    assert {"per_run.csv", "summary.json", "a7_disturbance.png"} <= set(first_files)
    assert first_files == _output_bytes(second)
    assert first_files == _output_bytes(third)

    # Keep the JSON parse in the test so a byte-identical but invalid summary
    # cannot satisfy the determinism check.
    json.loads((first / "summary.json").read_text())
