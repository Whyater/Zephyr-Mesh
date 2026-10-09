"""Independent limiting-case checks for the synthetic actuator model."""

import numpy as np
import pytest

from sim.actuator import ActuatorConfig, QuadrotorActuator
from sim.hardware import HardwareProfile, MotorProfile, PropellerProfile
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


def test_effective_rpm_respects_declared_power_and_torque_limits():
    profile = default_hardware_profile(1)
    actuator = QuadrotorActuator(profile)
    assert actuator.max_rpm <= profile.motor.max_rpm
    assert actuator.max_rpm <= profile.propeller.max_rpm
    assert actuator.max_rpm < min(profile.motor.max_rpm, profile.propeller.max_rpm)
    state = actuator.step((1, 1, 1, 1), 1.0)
    assert max(state.rpm) <= actuator.max_rpm + 1e-9
    assert state.power_w / 4.0 <= profile.motor.max_power_w + 1e-9
    assert max(profile.propeller.estimated_torque_nm(rpm) for rpm in state.rpm) <= profile.motor.max_torque_nm + 1e-9


def test_effective_rpm_isolated_power_torque_and_declared_limits():
    def profile(*, max_power, max_torque, motor_rpm=100_000.0, propeller_rpm=100_000.0):
        return HardwareProfile(
            "isolated",
            MotorProfile("motor", max_torque, max_power, motor_rpm),
            PropellerProfile("prop", 0.1, 0.05, propeller_rpm),
        )

    power_profile = profile(max_power=1.0, max_torque=100.0)
    power_actuator = QuadrotorActuator(power_profile)
    coefficient = power_profile.propeller.power_coefficient * 1.225 * power_profile.propeller.diameter_m**5
    expected_power_rpm = 60.0 * (power_profile.motor.max_power_w / coefficient) ** (1.0 / 3.0)
    assert power_actuator.max_rpm == pytest.approx(expected_power_rpm)

    torque_profile = profile(max_power=1_000.0, max_torque=0.001)
    torque_actuator = QuadrotorActuator(torque_profile)
    expected_torque_rpm = 60.0 * (2.0 * np.pi * torque_profile.motor.max_torque_nm / coefficient) ** 0.5
    assert torque_actuator.max_rpm == pytest.approx(expected_torque_rpm)

    declared_profile = profile(max_power=1_000.0, max_torque=100.0, motor_rpm=5_000.0, propeller_rpm=6_000.0)
    assert QuadrotorActuator(declared_profile).max_rpm == pytest.approx(5_000.0)


def test_depleted_battery_cuts_motor_output_to_zero():
    actuator = QuadrotorActuator(default_hardware_profile(2))
    actuator.energy_wh = 0.0
    state = actuator.step((1, 1, 1, 1), 0.1)
    assert state.rpm == pytest.approx((0.0,) * 4)
    assert state.total_thrust_n == pytest.approx(0.0)
    assert state.power_w == pytest.approx(0.0)


def test_mid_step_battery_depletion_returns_zero_output():
    actuator = QuadrotorActuator(default_hardware_profile(3))
    actuator.energy_wh = 1e-6
    state = actuator.step((1, 1, 1, 1), 0.1)
    assert state.battery_pct == pytest.approx(0.0)
    assert state.rpm == pytest.approx((0.0,) * 4)
    assert state.total_thrust_n == pytest.approx(0.0)
    assert state.power_w == pytest.approx(0.0)


def test_invalid_commands_and_timestep_are_rejected():
    actuator = QuadrotorActuator(default_hardware_profile(0))
    with pytest.raises(ValueError):
        actuator.step((0, 0, 0), 0.01)
    with pytest.raises(ValueError):
        actuator.step((0, 0, 0, 1.2), 0.01)
    with pytest.raises(ValueError):
        actuator.step((0, 0, 0, 0), 0)
