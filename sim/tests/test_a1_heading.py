"""Checks for the optional heading setpoint in the geometric controller."""

from pathlib import Path

import numpy as np
import pytest

from sim.baseline import BaselineConfig, run_baseline
from sim.controller import GeometricFlightController
from sim.drone import SixDOFInterceptor


def _yaw_from_quaternion(quaternion):
    qw, qx, qy, qz = quaternion
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy**2 + qz**2),
    )


def test_none_heading_path_matches_retained_baseline_bit_for_bit():
    """The omitted heading argument keeps the retained default telemetry exact."""
    artifact = Path(__file__).parents[2] / "sim" / "runs" / "s0-baseline" / "telemetry.csv"
    retained = np.genfromtxt(artifact, delimiter=",", names=True)
    fresh = run_baseline(BaselineConfig())
    assert tuple(retained.dtype.names) == tuple(fresh)
    for name in fresh:
        assert np.array_equal(retained[name], fresh[name]), name


@pytest.mark.parametrize("dt", [0.001, 0.01])
def test_ninety_degree_heading_step_settles_with_small_overshoot(dt):
    target_yaw = np.pi / 2.0
    drone = SixDOFInterceptor(init_pos=(0.0, 0.0, 1.0), drag_coefficient=0.0)
    controller = GeometricFlightController()
    yaws = []
    for _ in range(round(3.0 / dt)):
        force, torque = controller.update_control(
            drone,
            drone.position.copy(),
            dt,
            target_yaw_rad=target_yaw,
        )
        drone.step_physics(force, torque, dt)
        yaws.append(_yaw_from_quaternion(drone.quaternion))
    yaws = np.asarray(yaws)

    overshoot = (float(np.max(yaws)) - target_yaw) / target_yaw
    assert overshoot < 0.01
    error = np.abs(yaws - target_yaw)
    outside = np.flatnonzero(error > 0.02 * target_yaw)
    settling_time = (int(outside[-1]) + 1) * dt if outside.size else 0.0
    assert 0.3 <= settling_time <= 0.8
    assert yaws[-1] == pytest.approx(target_yaw, abs=1e-8)


def test_heading_wraparound_chooses_twenty_degree_short_path():
    current_yaw = np.deg2rad(170.0)
    target_yaw = np.deg2rad(-170.0)
    drone = SixDOFInterceptor()
    drone.quaternion[:] = [np.cos(current_yaw / 2.0), 0.0, 0.0, np.sin(current_yaw / 2.0)]
    _, torque = GeometricFlightController().update_control(
        drone,
        drone.position.copy(),
        0.01,
        target_yaw_rad=target_yaw,
    )
    expected = drone.I_z * 100.0 * np.deg2rad(20.0)
    assert torque[2] == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_heading_setpoint_changes_only_body_z_torque():
    drone = SixDOFInterceptor()
    drone.quaternion[:] = [0.96, 0.12, -0.18, 0.16]
    drone.quaternion[:] /= np.linalg.norm(drone.quaternion)
    controller = GeometricFlightController()
    _, torque_without_heading = controller.update_control(drone, [0.2, -0.1, 1.3], 0.01)
    _, torque_with_heading = controller.update_control(
        drone,
        [0.2, -0.1, 1.3],
        0.01,
        target_yaw_rad=0.0,
    )
    assert np.array_equal(torque_without_heading[:2], torque_with_heading[:2])


def test_heading_setpoint_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="target_yaw_rad"):
        GeometricFlightController().update_control(
            SixDOFInterceptor(), [0.0, 0.0, 1.0], 0.01, target_yaw_rad=np.nan
        )
