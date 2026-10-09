"""Independent integration checks for the actuator-coupled vehicle."""

import numpy as np
import pytest

from sim.scenarios import default_hardware_profile
from sim.vehicle import ActuatedQuadrotor, QuadrotorMixer
from sim.actuator import QuadrotorActuator


def test_hover_mixer_balances_equal_rotors():
    actuator = QuadrotorActuator(default_hardware_profile(0))
    mixer = QuadrotorMixer(actuator)
    result = mixer.mix(0.8, (0.0, 0.0, 0.0))
    assert result.rotor_thrust_n == pytest.approx((0.2, 0.2, 0.2, 0.2))
    assert result.motor_command[0] == pytest.approx(result.motor_command[3])
    assert not result.saturated


def test_excessive_wrench_is_explicitly_saturated():
    actuator = QuadrotorActuator(default_hardware_profile(1))
    mixer = QuadrotorMixer(actuator)
    result = mixer.mix(actuator.profile.estimated_max_thrust_n() * 2.0, (0.0, 0.0, 0.0))
    assert result.saturated
    assert max(result.rotor_thrust_n) <= actuator.profile.estimated_max_thrust_n() / 4.0 + 1e-9
    assert max(result.motor_command) <= 1.0


def test_vehicle_uses_profile_mass_and_produces_finite_coupled_state():
    profile = default_hardware_profile(2)
    vehicle = ActuatedQuadrotor(profile)
    assert vehicle.drone.mass == pytest.approx(profile.total_mass_kg)
    samples = [vehicle.step((0.0, 0.0, 1.0), 0.01) for _ in range(80)]
    final = samples[-1]
    assert np.all(np.isfinite(final.position_m))
    assert np.all(np.isfinite(final.velocity_mps))
    assert final.actuator.total_thrust_n > 0.0
    assert final.actuator.battery_pct < 100.0
    assert any(sample.mixer.saturated for sample in samples) is False


def test_vehicle_rejects_nonfinite_target():
    vehicle = ActuatedQuadrotor(default_hardware_profile(3))
    with pytest.raises(ValueError):
        vehicle.step((0.0, float("nan"), 1.0), 0.01)


def test_vehicle_drag_changes_trajectory_for_same_initial_motion():
    profile = default_hardware_profile(4)
    no_drag = ActuatedQuadrotor(profile, drag_coefficient=0.0)
    with_drag = ActuatedQuadrotor(profile, drag_coefficient=3.0)
    no_drag.drone.velocity[:] = [8.0, 0.0, 0.0]
    with_drag.drone.velocity[:] = [8.0, 0.0, 0.0]

    no_drag_step = no_drag.step((0.0, 0.0, 0.0), 0.02)
    with_drag_step = with_drag.step((0.0, 0.0, 0.0), 0.02)

    assert with_drag_step.position_m[0] < no_drag_step.position_m[0]
    assert with_drag_step.velocity_mps[0] < no_drag_step.velocity_mps[0]


def test_vehicle_horizontal_target_produces_translation_and_tilt():
    vehicle = ActuatedQuadrotor(default_hardware_profile(5))
    samples = [vehicle.step((2.0, 0.0, 1.0), 0.01) for _ in range(200)]
    final = samples[-1]
    assert final.position_m[0] > 0.5
    assert abs(final.position_m[1]) < 0.1
    assert np.linalg.norm(np.asarray(vehicle.drone.quaternion[1:3])) > 1e-3
