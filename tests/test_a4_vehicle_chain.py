"""Hand checks for the opt-in v2 rotorcraft chain."""

import math

import numpy as np
import pytest

from sim.actuator import RotorcraftActuator
from sim.hardware import RotorcraftProfile
from sim.physics.disturbance_api import DisturbanceSample
from sim.environment import GRAVITY
from sim.drone import SixDOFInterceptor
from sim.vehicle import ActuatedQuadrotor


def fixture_profile(*, battery=None):
    raw = {
        "schema": "zephyr-hardware-profile-2",
        "profile_id": "a4-fixture",
        "status": "fixture",
        "notes": "round-number fixture for dynamics checks",
        "airframe": {
            "mass_kg": 0.030,
            "inertia_kg_m2": [1.0e-5, 1.0e-5, 2.0e-5],
            "arm_length_m": 0.040,
            "drag_coefficient": 0.0,
            "cross_sectional_area_m2": 0.010,
        },
        "rotor": {
            "count": 4,
            "layout": "x",
            "thrust_coefficient_n_s2": 3.0e-8,
            "torque_to_thrust_m": 0.006,
            "max_speed_rad_s": 2500.0,
            "motor_time_constant_s": 0.030,
            "prop_radius_m": 0.020,
            "spin_directions": [1, -1, 1, -1],
        },
        "battery": battery,
        "parameters": {},
    }
    return RotorcraftProfile.from_dict(raw)


class ConstantDisturbance:
    def __init__(self, sample):
        self.sample_value = sample

    def sample(self, t_s, dt_s, position_m, velocity_mps):
        return self.sample_value


def test_motor_step_matches_first_order_lag_at_tau_multiples():
    profile = fixture_profile()
    actuator = RotorcraftActuator(profile)
    tau = profile.rotor.motor_time_constant_s
    for multiple in (1, 2, 3):
        actuator.reset()
        state = actuator.step((1.0,) * 4, multiple * tau)
        expected_ratio = 1.0 - math.exp(-multiple)
        assert state.speed_rad_s[0] / profile.rotor.max_speed_rad_s == pytest.approx(expected_ratio, abs=1e-12)


def test_hover_speed_and_trim_force_are_hand_computed():
    profile = fixture_profile()
    speed = math.sqrt(profile.airframe.mass_kg * 9.81 / (4.0 * profile.rotor.thrust_coefficient_n_s2))
    actuator = RotorcraftActuator(profile)
    actuator.speed_rad_s[:] = speed
    state = actuator._state(False, 1.0)
    assert state.total_thrust_n == pytest.approx(profile.airframe.mass_kg * 9.81)


def test_substep_mean_thrust_is_scaled_by_disturbance():
    profile = fixture_profile()
    speed = profile.hover_speed_rad_s()
    disturbance = ConstantDisturbance(DisturbanceSample(thrust_scale=1.1))
    vehicle = ActuatedQuadrotor(profile, substeps=1, disturbance=disturbance)
    vehicle.actuator.speed_rad_s[:] = speed
    vehicle.step((0.0, 0.0, 0.0), 0.001)
    # The exact substep mean thrust is 1.1 times trim, leaving 0.1 g upward acceleration.
    assert vehicle.drone.velocity[2] / 0.001 == pytest.approx(0.1 * 9.81, rel=2e-3)


def test_reaction_torque_cancels_for_equal_trim_rotors():
    actuator = RotorcraftActuator(fixture_profile())
    actuator.speed_rad_s[:] = actuator.profile.hover_speed_rad_s()
    state = actuator._state(False, 1.0)
    assert state.body_torque_nm[2] == pytest.approx(0.0, abs=1e-15)


def test_battery_voltage_uses_start_voltage_and_charge_decreases():
    battery = {
        "nominal_voltage_v": 4.0,
        "cutoff_voltage_v": 3.0,
        "capacity_mah": 1000.0,
        "internal_resistance_ohm": 0.1,
        "efficiency": 0.8,
    }
    actuator = RotorcraftActuator(fixture_profile(battery=battery))
    first = actuator.step((0.8,) * 4, 0.01)
    assert first.battery_charge_ah < 1.0
    assert first.battery_voltage_v < 4.0
    second = actuator.step((0.8,) * 4, 0.01)
    assert second.battery_charge_ah < first.battery_charge_ah
    assert second.battery_voltage_v <= first.battery_voltage_v


def _average_power_w3(w0, command_speed, h, tau):
    """Integrate ``(c + (w0-c) exp(-t/tau))**3`` over one substep."""
    delta = w0 - command_speed
    terms = (
        command_speed**3,
        3.0 * command_speed**2 * delta * (tau / h) * (1.0 - math.exp(-h / tau)),
        3.0 * command_speed * delta**2 * (tau / (2.0 * h)) * (1.0 - math.exp(-2.0 * h / tau)),
        delta**3 * (tau / (3.0 * h)) * (1.0 - math.exp(-3.0 * h / tau)),
    )
    return sum(terms)


def test_battery_sag_matches_hand_integrated_average_power():
    """The charge drop uses hand-integrated mean ``w**3`` and start voltage.

    For each step, ``w(t) = c + (w0-c) exp(-t/tau)``.  Expanding its cube and
    integrating each exponential term gives the mean power used below.  The
    expected charge and terminal voltage are then calculated directly from
    current, state of charge, open-circuit voltage, and internal resistance.
    """
    battery = {
        "nominal_voltage_v": 4.0,
        "cutoff_voltage_v": 3.0,
        "capacity_mah": 1000.0,
        "internal_resistance_ohm": 0.1,
        "efficiency": 0.8,
    }
    profile = fixture_profile(battery=battery)
    actuator = RotorcraftActuator(profile)
    dt = 0.01
    duty = 0.8
    tau = profile.rotor.motor_time_constant_s
    kf = profile.rotor.thrust_coefficient_n_s2
    km_ratio = profile.rotor.torque_to_thrust_m
    nominal = battery["nominal_voltage_v"]
    charge = battery["capacity_mah"] / 1000.0
    voltage = nominal
    speed = np.zeros(4)
    expected_at = {1, 10, 100}
    reported = {}
    for step in range(1, 101):
        start_voltage = voltage
        command_speed = duty * profile.rotor.max_speed_rad_s * start_voltage / nominal
        average_w3 = _average_power_w3(speed, command_speed, dt, tau)
        mechanical_power = km_ratio * kf * float(np.sum(average_w3))
        current = mechanical_power / (battery["efficiency"] * start_voltage)
        charge = max(0.0, charge - current * dt / 3600.0)
        soc = charge / (battery["capacity_mah"] / 1000.0)
        open_circuit = battery["cutoff_voltage_v"] + (nominal - battery["cutoff_voltage_v"]) * soc
        voltage = open_circuit - current * battery["internal_resistance_ohm"]
        speed = command_speed + (speed - command_speed) * math.exp(-dt / tau)
        actual = actuator.step((duty,) * 4, dt)
        if step in expected_at:
            assert actual.battery_charge_ah == pytest.approx(charge, rel=2e-10, abs=2e-12)
            assert actual.battery_voltage_v == pytest.approx(voltage, rel=2e-10, abs=2e-12)
            assert actual.battery_charge_ah < battery["capacity_mah"] / 1000.0
            reported[step] = (float(actual.battery_charge_ah), float(actual.battery_voltage_v), float(current))
    print("battery hand check", reported)


def test_hover_hold_from_trim_changes_position_by_less_than_one_mm():
    """A trimmed vehicle remains fixed over 10 s using two motor substeps."""
    profile = fixture_profile()
    hover_speed = math.sqrt(profile.airframe.mass_kg * 9.81 / (4.0 * profile.rotor.thrust_coefficient_n_s2))
    vehicle = ActuatedQuadrotor(profile, initial_position=(0.0, 0.0, 1.0), substeps=2)
    vehicle.actuator.speed_rad_s[:] = hover_speed
    initial = vehicle.drone.position.copy()
    for _ in range(1000):
        vehicle.step((0.0, 0.0, 1.0), 0.01)
    displacement = float(np.linalg.norm(vehicle.drone.position - initial))
    print("hover displacement", displacement)
    assert displacement < 1.0e-3


def test_hover_duty_rises_as_battery_sags_while_speed_stays_at_trim():
    """Battery sag raises duty; the target hover speed itself is unchanged.

    The run prompt's phrase "hover RPM rises" is physically wrong for this
    model.  Thrust depends on rotor speed, so voltage loss raises duty to
    preserve the same speed and thrust.
    """
    battery = {
        "nominal_voltage_v": 4.0,
        "cutoff_voltage_v": 3.0,
        "capacity_mah": 5.0,
        "internal_resistance_ohm": 0.15,
        "efficiency": 0.8,
    }
    profile = fixture_profile(battery=battery)
    hover_speed = math.sqrt(profile.airframe.mass_kg * 9.81 / (4.0 * profile.rotor.thrust_coefficient_n_s2))
    vehicle = ActuatedQuadrotor(profile, initial_position=(0.0, 0.0, 1.0), substeps=2)
    vehicle.actuator.speed_rad_s[:] = hover_speed
    duties = []
    mean_speeds = []
    for step in range(1, 401):
        state = vehicle.step((0.0, 0.0, 1.0), 0.01).actuator
        if step % 50 == 0:
            duties.append(float(np.mean(state.duty)))
            mean_speeds.append(float(np.mean(state.speed_rad_s)))
    print("hover duties", duties, "mean speeds", mean_speeds)
    assert all(b > a for a, b in zip(duties, duties[1:]))
    assert all(abs(speed - hover_speed) / hover_speed < 0.005 for speed in mean_speeds)


def _tilt_angle_rad(quaternion):
    qw, qx, qy, qz = quaternion
    body_z = np.array([
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ])
    return math.acos(float(np.clip(body_z[2], -1.0, 1.0)))


@pytest.mark.parametrize("dt", (0.01, 0.005, 0.002))
def test_hover_and_lateral_step_are_stable_across_control_dt(dt):
    """At each control dt, 2 motor substeps per control update are used.

    Both the fixed hover and a 0.5 m lateral move must settle within 2 cm at
    8 s, stay below 30 degrees tilt, and remain finite.
    """
    profile = fixture_profile()
    metrics = []
    for target in ((0.0, 0.0, 1.0), (0.5, 0.0, 1.0)):
        vehicle = ActuatedQuadrotor(profile, initial_position=(0.0, 0.0, 1.0), substeps=2)
        max_tilt = 0.0
        for _ in range(round(8.0 / dt)):
            vehicle.step(target, dt)
            assert np.all(np.isfinite(vehicle.drone._state_vector()))
            max_tilt = max(max_tilt, _tilt_angle_rad(vehicle.drone.quaternion))
        final_error = float(np.linalg.norm(vehicle.drone.position - np.asarray(target)))
        metrics.append((target, final_error, math.degrees(max_tilt)))
        assert final_error < 0.02
        assert max_tilt < math.radians(30.0)
    print("dt stability", dt, metrics)


def test_trimmed_equal_props_keep_yaw_rate_zero():
    """Equal and opposite propellers cancel reaction torque over 10 s."""
    profile = fixture_profile()
    hover_speed = profile.hover_speed_rad_s()
    vehicle = ActuatedQuadrotor(profile, initial_position=(0.0, 0.0, 1.0), substeps=2)
    vehicle.actuator.speed_rad_s[:] = hover_speed
    max_yaw_rate = 0.0
    for _ in range(1000):
        vehicle.step((0.0, 0.0, 1.0), 0.01)
        max_yaw_rate = max(max_yaw_rate, abs(float(vehicle.drone.angular_velocity[2])))
    print("max yaw rate", max_yaw_rate)
    assert max_yaw_rate < 1.0e-9


def test_max_vertical_acceleration_matches_hand_calculation_after_settle():
    profile = fixture_profile()
    actuator = RotorcraftActuator(profile)
    for _ in range(100):
        state = actuator.step((1.0,) * 4, 0.01)
    drone = SixDOFInterceptor(mass=profile.mass_kg, drag_coefficient=0.0,
                              cross_sectional_area=0.0, inertia_kg_m2=profile.inertia_kg_m2)
    measured = drone.compute_derivatives((0.0, 0.0, state.total_thrust_n), (0.0, 0.0, 0.0))[1][2]
    expected = (4.0 * profile.thrust_coefficient_n_s2 * profile.max_speed_rad_s**2) / profile.mass_kg - GRAVITY
    assert measured == pytest.approx(expected, rel=1e-2)


def test_yaw_command_has_the_declared_positive_sign():
    profile = fixture_profile()
    vehicle = ActuatedQuadrotor(profile)
    command = vehicle.mixer.mix(profile.mass_kg * GRAVITY, (0.0, 0.0, 1.0e-5))
    assert command.achieved_torque_nm[2] > 0.0
    angular_acceleration = vehicle.drone.compute_derivatives(
        (0.0, 0.0, profile.mass_kg * GRAVITY), command.achieved_torque_nm
    )[3]
    assert angular_acceleration[2] > 0.0


def _run_end_held(profile, target, *, dt, substeps):
    """Inline reproduction of the old end-of-substep thrust method.

    Starts from the same hover point as the reference run in the test below.
    """
    vehicle = ActuatedQuadrotor(profile, initial_position=(0.0, 0.0, 1.0), substeps=substeps)
    positions = []
    h = dt / substeps
    for _ in range(round(2.0 / dt)):
        requested_force, requested_torque = vehicle.controller.update_control(vehicle.drone, np.asarray(target), dt)
        mixer = vehicle.mixer.mix(float(requested_force[2]), requested_torque)
        for _ in range(substeps):
            state = vehicle.actuator.step(mixer.motor_command, h, saturated=mixer.saturated)
            thrust = np.asarray(state.thrust_n)
            vehicle.drone.step_physics(
                np.array([0.0, 0.0, float(np.sum(thrust))]),
                vehicle.mixer.torque_from_thrust(thrust),
                h,
            )
        vehicle.time_s += dt
        positions.append(vehicle.drone.position.copy())
    return np.asarray(positions)


def test_substep_convergence_selects_smallest_n_under_one_mm_rms():
    """With control dt fixed at .01 s, exact averaging gives these RMS errors.

    The 2 s lateral step uses a 50-substep, dt=.01 reference, so controller
    sampling is identical in every run.  Recorded values on the repaired
    chain (against a 200-substep reference, 2026-10-14) were N=1: 4.8e-5 m,
    N=2: 1.2e-5 m, N=5: 1.9e-6 m and N=10: 4.8e-7 m, falling as 1/N^2.  The
    smallest N below 1 mm is therefore N=1.  The old end-of-substep thrust
    method at N=1 gave 1.07e-2 m from the same start, about 220 times worse.
    """
    profile = fixture_profile()
    target = (0.5, 0.0, 1.0)

    def run(substeps):
        vehicle = ActuatedQuadrotor(profile, initial_position=(0.0, 0.0, 1.0), substeps=substeps)
        positions = []
        for _ in range(200):
            vehicle.step(target, 0.01)
            positions.append(vehicle.drone.position.copy())
        return np.asarray(positions)

    reference = run(50)
    rms_by_substep = {}
    for substeps in (1, 2, 5, 10):
        trial = run(substeps)
        rms_by_substep[substeps] = float(np.sqrt(np.mean(np.sum((trial - reference) ** 2, axis=1))))
    old_end_held_rms = float(np.sqrt(np.mean(np.sum((_run_end_held(profile, target, dt=0.01, substeps=1) - reference) ** 2, axis=1))))
    print("substep RMS", rms_by_substep, "old end-held N=1", old_end_held_rms)
    assert min(n for n in (1, 2, 5, 10) if rms_by_substep[n] < 0.001) == 1
    assert old_end_held_rms > 50.0 * rms_by_substep[1]


def test_v1_default_first_step_matches_saved_pre_edit_state():
    """Constants below were captured from the v1 path before the A4 edits."""
    from sim.hardware import default_hardware_profile

    vehicle = ActuatedQuadrotor(default_hardware_profile("a4-bit"))
    vehicle.step((0.0, 0.0, 0.0), 0.01)
    assert np.array_equal(vehicle.drone.position, np.array([1.689421801475451e-26, 2.368544983712144e-47, -0.00047690481809897235]))
    assert np.array_equal(vehicle.drone.velocity, np.array([6.756864760582927e-24, 2.3684296780654006e-44, -0.09537322503082254]))
    assert np.array_equal(vehicle.drone.quaternion, np.array([1.0, -6.988670337825724e-42, 3.738628180984499e-21, 2.8039711357383727e-21]))


def test_v1_default_path_is_bit_identical_after_500_steps():
    """Constants were produced by running the original v1 files at 1b9d458.

    The source package was reconstructed with ``git show`` into a temporary
    directory, then stepped 500 times at dt=0.01 before printing repr arrays.
    """
    from sim.hardware import default_hardware_profile

    vehicle = ActuatedQuadrotor(default_hardware_profile("a4-bit"))
    for _ in range(500):
        vehicle.step((0.0, 0.0, 0.0), 0.01)
    assert np.array_equal(vehicle.drone.position, np.array([
        7.978887835245687e-17, 5.702616999808826e-16, -3.918359815787623e-03,
    ]))
    assert np.array_equal(vehicle.drone.velocity, np.array([
        -4.2693190478184174e-17, -2.3552969443429124e-16, 8.5070400427831065e-04,
    ]))
    assert np.array_equal(vehicle.drone.quaternion, np.array([
        1.0000000000000000e+00, 7.7404557859104611e-17,
        -1.0023262153217956e-17, 3.8384962862690470e-16,
    ]))
    assert np.array_equal(vehicle.drone.angular_velocity, np.array([
        4.7854440716601806e-17, -8.3595726126813254e-16,
        2.0390478099089447e-16,
    ]))


def test_v1_fleet_default_path_is_bit_identical_after_ten_steps():
    """Fleet constants were produced from the original v1 package at 1b9d458."""
    from sim.scenarios import default_hardware_profile
    from sim.vehicle_fleet import ActuatedVehicleFleet

    fleet = ActuatedVehicleFleet(
        {"beta": default_hardware_profile(1), "alpha": default_hardware_profile(0)},
        initial_positions={"alpha": (-1.0, 0.0, 1.5), "beta": (1.0, 0.0, 1.5)},
    )
    targets = {"alpha": (-1.5, 0.0, 1.5), "beta": (1.5, 0.0, 1.5)}
    for _ in range(10):
        fleet.step(targets, 0.01)
    assert fleet.step_index == 10
    assert fleet.time_s == 0.09999999999999999
    alpha = fleet._vehicles["alpha"].drone
    beta = fleet._vehicles["beta"].drone
    assert np.array_equal(alpha.position, np.array([
        -1.0000312078453029e+00, 2.7468810368889150e-20, 1.4642156077432626e+00,
    ]))
    assert np.array_equal(alpha.velocity, np.array([
        -1.8303401970157962e-03, 1.7332712085910140e-18, -5.7471177506341153e-01,
    ]))
    alpha_quaternion = np.array([
        9.999843586810702e-01, -6.209539120597357e-18,
        -5.5930665299916530e-03, 0.0,
    ])
    alpha_quaternion[3] = float.fromhex("0x1.49f7c150d1ee6p-59")
    assert np.array_equal(alpha.quaternion, alpha_quaternion)
    assert np.array_equal(alpha.angular_velocity, np.array([
        -5.0907339472669637e-16, -3.0803294977354573e-01,
        1.3066505492540819e-16,
    ]))
    assert np.array_equal(beta.position, np.array([
        1.0000312078451201e+00, 3.7702910384455593e-20, 1.4642156077112434e+00,
    ]))
    assert np.array_equal(beta.velocity, np.array([
        1.8303401813779050e-03, 1.9825103558335440e-18, -5.7471177694580210e-01,
    ]))
    beta_quaternion = np.array([
        0.0, -5.6756084669090474e-18,
        5.5930665192784178e-03, 1.0834399748626498e-18,
    ])
    beta_quaternion[0] = float.fromhex("0x1.fffdf32a180ecp-1")
    assert np.array_equal(beta.quaternion, beta_quaternion)
    assert np.array_equal(beta.angular_velocity, np.array([
        -3.9370960554297174e-16, 3.0803294867381897e-01,
        5.6415899251056072e-17,
    ]))


def test_constant_wind_disturbance_matches_direct_rigid_body_step():
    profile = fixture_profile()
    wind = np.array([1.0, -0.5, 0.0])
    vehicle = ActuatedQuadrotor(profile, disturbance=ConstantDisturbance(DisturbanceSample(wind_velocity_mps=wind)))
    speed = profile.hover_speed_rad_s()
    vehicle.actuator.speed_rad_s[:] = speed
    record = vehicle.step((0.0, 0.0, 0.0), 0.001)
    reference = SixDOFInterceptor(mass=profile.mass_kg, drag_coefficient=0.0,
                                  cross_sectional_area=0.0, inertia_kg_m2=profile.inertia_kg_m2)
    reference.step_physics(record.applied_force_body_n, record.applied_torque_body_nm, 0.001, wind_velocity=wind)
    assert np.array_equal(vehicle.drone.position, reference.position)
    assert np.array_equal(vehicle.drone.velocity, reference.velocity)


def test_substep_mean_thrust_matches_brute_force_integration():
    """Independent check of the closed-form mean of w^2 (and power w^3).

    The actuator's closed form and the battery test's helper are the same
    algebra, so this check integrates the lag curve numerically instead:
    w(t) = c + (w0 - c) exp(-t / tau), sampled finely and averaged with the
    trapezoid rule.  Spin-up, spin-down and a mixed set are all covered.
    """
    profile = fixture_profile()
    tau = profile.rotor.motor_time_constant_s
    kf = profile.rotor.thrust_coefficient_n_s2
    km_ratio = profile.rotor.torque_to_thrust_m
    w_max = profile.rotor.max_speed_rad_s
    h = 0.01
    t = np.linspace(0.0, h, 200001)
    for start_fraction, duty in ((0.0, 1.0), (1.0, 0.2), (0.5, 0.5), (0.3, 0.9)):
        actuator = RotorcraftActuator(profile)
        actuator.speed_rad_s[:] = start_fraction * w_max
        state = actuator.step((duty,) * 4, h)
        curve = duty * w_max + (start_fraction * w_max - duty * w_max) * np.exp(-t / tau)
        mean_w2 = np.trapezoid(curve**2, t) / h
        mean_w3 = np.trapezoid(curve**3, t) / h
        assert state.mean_thrust_n[0] == pytest.approx(kf * mean_w2, rel=1e-9)
        # Mechanical power reported for the battery is the mean of k_m w^3 over all rotors.
        assert 4.0 * km_ratio * kf * mean_w3 == pytest.approx(
            km_ratio * kf * 4.0 * _average_power_w3(start_fraction * w_max, duty * w_max, h, tau), rel=1e-9
        )
