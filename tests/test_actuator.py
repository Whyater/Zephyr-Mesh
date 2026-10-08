"""Independent limiting-case checks for the synthetic actuator model."""

import numpy as np
import pytest

from sim.actuator import ActuatorConfig, QuadrotorActuator
from sim.scenarios import default_hardware_profile


def test_zero_command_has_zero_thrust_and_torque():
    actuator = QuadrotorActuator(default_hardware_profile(0))
    state = actuator.step((0, 0, 0, 0), 0.1)
    assert state.total_thrust_n == pytest.approx(0.0)
    assert state.body_torque_nm == pytest.approx((0.0, 0.0, 0.0))
    assert state.battery_pct == pytest.approx(100.0)


def test_first_order_motor_lag_matches_closed_form_rpm():
    config = ActuatorConfig(motor_time_constant_s=0.2)
    profile = default_hardware_profile(0)
    actuator = QuadrotorActuator(profile, config)
    dt = 0.05
    state = actuator.step((1, 1, 1, 1), dt)
    alpha = 1.0 - np.exp(-dt / config.motor_time_constant_s)
    expected = actuator.max_rpm * alpha
    assert state.rpm == pytest.approx((expected,) * 4)
    assert 0.0 < state.total_thrust_n
    later = actuator.step((1, 1, 1, 1), dt)
    assert later.total_thrust_n > state.total_thrust_n


def test_full_command_saturates_and_consumes_energy():
    actuator = QuadrotorActuator(default_hardware_profile(1))
    initial = actuator.last_state
    for _ in range(400):
        state = actuator.step((1, 1, 1, 1), 0.01)
    # Battery sag slightly lowers the asymptotic command below the mechanical limit.
    assert state.rpm == pytest.approx((state.rpm[0],) * 4)
    assert state.rpm[0] > actuator.max_rpm * 0.99
    assert state.total_thrust_n == pytest.approx(sum(state.thrust_n))
    assert state.battery_pct < initial.battery_pct


def test_invalid_commands_and_timestep_are_rejected():
    actuator = QuadrotorActuator(default_hardware_profile(0))
    with pytest.raises(ValueError):
        actuator.step((0, 0, 0), 0.01)
    with pytest.raises(ValueError):
        actuator.step((0, 0, 0, 1.2), 0.01)
    with pytest.raises(ValueError):
        actuator.step((0, 0, 0, 0), 0)
