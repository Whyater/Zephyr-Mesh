"""Independent rigid-body truth checks for the A1 integrator review."""

import numpy as np

from sim.drone import SixDOFInterceptor
from sim.environment import AIR_DENSITY, GRAVITY


def _advance(drone, force_body, torque_body, duration, dt):
    """Advance exactly ``duration`` seconds using fixed-size steps."""
    count = round(duration / dt)
    assert np.isclose(count * dt, duration)
    for _ in range(count):
        drone.step_physics(force_body, torque_body, dt)


def _rotation_from_quaternion(quaternion):
    """Build the body-to-world matrix from a w-first quaternion."""
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qw * qx),
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ]
    )


def test_free_fall_without_drag_matches_quadratic_solution():
    """A constant gravitational acceleration gives an exact quadratic path."""
    duration = 2.0
    for dt in (0.01, 0.05, 0.1):
        initial_z = 7.0
        drone = SixDOFInterceptor(
            init_pos=(0.0, 0.0, initial_z),
            drag_coefficient=0.0,
        )
        _advance(drone, np.zeros(3), np.zeros(3), duration, dt)
        expected_z = initial_z - 0.5 * GRAVITY * duration**2
        expected_vz = -GRAVITY * duration
        np.testing.assert_allclose(drone.position[2], expected_z, atol=1e-9)
        np.testing.assert_allclose(drone.velocity[2], expected_vz, atol=1e-9)


def test_drag_fall_reaches_terminal_speed_and_matches_tanh_solution():
    """Quadratic drag from rest has the closed-form vertical speed solution."""
    drone = SixDOFInterceptor()
    mass = drone.mass
    drag_coefficient = drone.drag_coefficient
    area = drone.cross_sectional_area
    terminal_speed = np.sqrt(
        2.0 * mass * GRAVITY / (AIR_DENSITY * drag_coefficient * area)
    )
    time_constant = terminal_speed / GRAVITY
    duration = 10.0 * time_constant
    max_closed_form_error = 0.0
    dt = 0.01
    for index in range(round(duration / dt)):
        drone.step_physics(np.zeros(3), np.zeros(3), dt)
        time = (index + 1) * dt
        expected_vz = -terminal_speed * np.tanh(GRAVITY * time / terminal_speed)
        max_closed_form_error = max(
            max_closed_form_error, abs(drone.velocity[2] - expected_vz)
        )
    terminal_relative_error = abs(abs(drone.velocity[2]) - terminal_speed) / terminal_speed
    print(
        "terminal speed",
        terminal_speed,
        "relative error",
        terminal_relative_error,
        "max tanh error",
        max_closed_form_error,
    )
    assert terminal_relative_error < 0.01
    assert max_closed_form_error < 1e-6


def test_torque_free_world_angular_momentum_and_energy_drift():
    """Torque-free motion conserves world angular momentum and kinetic energy."""
    force = np.zeros(3)
    torque = np.zeros(3)
    duration = 2.0
    results = {}
    for inertia in ((0.0023, 0.0023, 0.0040), (0.001, 0.002, 0.003)):
        body_inertia = np.asarray(inertia)
        for dt in (0.01, 0.001):
            drone = SixDOFInterceptor(
                drag_coefficient=0.0,
                inertia_kg_m2=inertia,
            )
            drone.angular_velocity[:] = (3.0, 1.0, 2.0)
            initial_l = body_inertia * drone.angular_velocity
            initial_energy = 0.5 * np.dot(
                drone.angular_velocity, body_inertia * drone.angular_velocity
            )
            max_l_drift = 0.0
            max_energy_drift = 0.0
            for _ in range(round(duration / dt)):
                drone.step_physics(force, torque, dt)
                world_l = _rotation_from_quaternion(drone.quaternion) @ (
                    body_inertia * drone.angular_velocity
                )
                energy = 0.5 * np.dot(
                    drone.angular_velocity, body_inertia * drone.angular_velocity
                )
                max_l_drift = max(
                    max_l_drift,
                    np.linalg.norm(world_l - initial_l) / np.linalg.norm(initial_l),
                )
                max_energy_drift = max(
                    max_energy_drift, abs(energy - initial_energy) / initial_energy
                )
            results[(inertia, dt)] = (max_l_drift, max_energy_drift)
            print(
                "torque-free",
                inertia,
                "dt",
                dt,
                "L drift/s",
                max_l_drift / duration,
                "energy drift/s",
                max_energy_drift / duration,
            )

        fine_l, fine_energy = results[(inertia, 0.001)]
        coarse_l, coarse_energy = results[(inertia, 0.01)]
        assert fine_l / duration < 1e-6
        assert fine_energy / duration < 1e-6
        # The observed fourth-order reduction is expected to be near 10**4.
        l_ratio = coarse_l / fine_l
        energy_ratio = coarse_energy / fine_energy
        print("torque-free drift ratios", inertia, "L", l_ratio, "energy", energy_ratio)
        assert l_ratio > 1000.0
        assert energy_ratio > 1000.0


def test_intermediate_axis_flip_and_principal_axis_stability():
    """The middle principal axis is unstable while the outer axes are stable.

    Euler's rigid-body equation uses ``-cross(omega, I omega)``.  Reversing
    that gyroscopic term swaps the stable and unstable axes, so this test
    checks the sign through the observed Dzhanibekov flip.
    """
    inertia = (1e-3, 2e-3, 3e-3)
    dt = 0.001
    force = np.zeros(3)
    torque = np.zeros(3)
    flip_time = None
    stable_max_angles = {}
    for axis in range(3):
        drone = SixDOFInterceptor(
            mass=0.45,
            drag_coefficient=0.0,
            inertia_kg_m2=inertia,
        )
        initial_rate = np.zeros(3)
        initial_rate[axis] = 10.0
        initial_rate[[0, 2]] += 1e-4
        drone.angular_velocity[:] = initial_rate
        max_angle = 0.0
        for index in range(round(10.0 / dt)):
            drone.step_physics(force, torque, dt)
            if axis == 1 and flip_time is None and drone.angular_velocity[1] < 0.0:
                flip_time = (index + 1) * dt
            unit_axis = np.zeros(3)
            unit_axis[axis] = 1.0
            angle = np.arccos(
                np.clip(
                    np.dot(drone.angular_velocity, unit_axis)
                    / np.linalg.norm(drone.angular_velocity),
                    -1.0,
                    1.0,
                )
            )
            if axis in (0, 2):
                max_angle = max(max_angle, angle)
        if axis in (0, 2):
            stable_max_angles[axis] = max_angle
    print("middle-axis flip time", flip_time, "s")
    print("stable-axis maximum angles", stable_max_angles)
    assert flip_time is not None
    assert flip_time < 5.0
    assert all(angle < np.deg2rad(1.0) for angle in stable_max_angles.values())


def test_hover_trim_stays_fixed_for_sixty_seconds():
    """Exact hover thrust leaves a level, motionless body at rest."""
    drone = SixDOFInterceptor()
    initial_quaternion = drone.quaternion.copy()
    _advance(
        drone,
        np.array([0.0, 0.0, drone.mass * GRAVITY]),
        np.zeros(3),
        60.0,
        0.01,
    )
    assert np.linalg.norm(drone.position) < 1e-9
    assert np.linalg.norm(drone.velocity) < 1e-9
    assert np.linalg.norm(drone.quaternion - initial_quaternion) < 1e-12


def test_quaternion_norm_after_many_steps_is_a_renormalization_restatement():
    """This is a RESTATEMENT of step_physics renormalization, not an independent check."""
    drone = SixDOFInterceptor(drag_coefficient=0.0)
    drone.angular_velocity[:] = (3.0, 1.0, 2.0)
    force = np.zeros(3)
    torque = np.zeros(3)
    max_deviation = 0.0
    for _ in range(100_000):
        drone.step_physics(force, torque, 0.01)
        max_deviation = max(max_deviation, abs(np.linalg.norm(drone.quaternion) - 1.0))
    print("quaternion norm maximum deviation", max_deviation)
    assert abs(np.linalg.norm(drone.quaternion) - 1.0) < 1e-12


def test_drag_fall_error_ratio_is_fourth_order():
    """The tanh solution exposes the expected factor of sixteen per halving."""
    mass = 0.45
    drag_coefficient = 1.3
    area = 0.05
    duration = 0.8
    terminal_speed = np.sqrt(
        2.0 * mass * GRAVITY / (AIR_DENSITY * drag_coefficient * area)
    )
    exact_vz = -terminal_speed * np.tanh(GRAVITY * duration / terminal_speed)
    errors = []
    for dt in (0.08, 0.04, 0.02):
        drone = SixDOFInterceptor(
            mass=mass,
            drag_coefficient=drag_coefficient,
            cross_sectional_area=area,
        )
        _advance(drone, np.zeros(3), np.zeros(3), duration, dt)
        errors.append(abs(drone.velocity[2] - exact_vz))
    ratios = [errors[index] / errors[index + 1] for index in range(2)]
    print("fourth-order drag error ratios", ratios)
    assert all(14.4 < ratio < 17.6 for ratio in ratios)


def test_rk4_stage_quaternion_norm_deviation_for_fast_spin():
    """Measure stage norm drift relevant to optional in-stage normalization."""
    drone = SixDOFInterceptor(drag_coefficient=0.0)
    drone.angular_velocity[:] = (0.0, 0.0, 5.0)
    force = np.zeros(3)
    torque = np.zeros(3)
    dt = 0.01
    max_stage_deviation = 0.0
    for _ in range(round(1.0 / dt)):
        state = drone._state_vector()
        k1 = drone._derivative_vector(state, force, torque, None)
        stage_states = [
            state + 0.5 * dt * k1,
        ]
        k2 = drone._derivative_vector(stage_states[0], force, torque, None)
        stage_states.append(state + 0.5 * dt * k2)
        k3 = drone._derivative_vector(stage_states[1], force, torque, None)
        stage_states.append(state + dt * k3)
        for stage_state in stage_states:
            max_stage_deviation = max(
                max_stage_deviation,
                abs(np.linalg.norm(stage_state[6:10]) - 1.0),
            )
        drone.step_physics(force, torque, dt)
    print("maximum RK4 stage quaternion norm deviation", max_stage_deviation)
    assert 1e-6 < max_stage_deviation < 1e-3
