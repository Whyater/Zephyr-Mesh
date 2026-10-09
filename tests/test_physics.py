"""Independent S2 physics checks against analytic and limiting cases."""
import numpy as np
import pytest

from sim.drone import SixDOFInterceptor
from sim.environment import GRAVITY, calculate_drag, relative_air_velocity


def run(drone, force, duration, dt):
    for _ in range(round(duration / dt)):
        drone.step_physics(force, np.zeros(3), dt)
    return drone


def test_zero_net_force_constant_velocity_matches_analytic_position():
    mass = 0.45
    drone = SixDOFInterceptor(mass=mass, drag_coefficient=0.0)
    drone.velocity[:] = [1.2, -0.4, 0.7]
    initial = drone.position.copy()
    duration, dt = 1.7, 0.017
    run(drone, np.array([0.0, 0.0, mass * GRAVITY]), duration, dt)
    np.testing.assert_allclose(drone.position, initial + drone.velocity * duration, atol=2e-12)
    np.testing.assert_allclose(drone.velocity, [1.2, -0.4, 0.7], atol=2e-12)


def test_constant_acceleration_matches_exact_polynomial():
    mass = 0.45
    acceleration = np.array([0.8, -0.3, 1.1])
    drone = SixDOFInterceptor(mass=mass, drag_coefficient=0.0)
    duration, dt = 1.25, 0.025
    initial_position = drone.position.copy()
    initial_velocity = drone.velocity.copy()
    run(drone, mass * (acceleration + np.array([0.0, 0.0, GRAVITY])), duration, dt)
    np.testing.assert_allclose(drone.position, initial_position + initial_velocity * duration + 0.5 * acceleration * duration**2, atol=2e-12)
    np.testing.assert_allclose(drone.velocity, initial_velocity + acceleration * duration, atol=2e-12)


def test_quadratic_drag_matches_closed_form_speed_and_changes_trajectory():
    mass, speed0 = 0.45, 8.0
    drone = SixDOFInterceptor(mass=mass, drag_coefficient=1.3, cross_sectional_area=0.05)
    drone.velocity[:] = [speed0, 0.0, 0.0]
    duration, dt = 0.8, 0.002
    run(drone, np.array([0.0, 0.0, mass * GRAVITY]), duration, dt)
    coefficient = 0.5 * 1.225 * 1.3 * 0.05 / mass
    expected_speed = speed0 / (1.0 + coefficient * speed0 * duration)
    expected_distance = np.log1p(coefficient * speed0 * duration) / coefficient
    np.testing.assert_allclose(drone.velocity[0], expected_speed, rtol=2e-8)
    np.testing.assert_allclose(drone.position[0], expected_distance, rtol=2e-8)
    assert drone.position[0] < speed0 * duration


def test_drag_is_zero_at_matching_wind_and_opposes_relative_velocity():
    velocity = np.array([3.0, -2.0, 1.0])
    wind = velocity.copy()
    np.testing.assert_allclose(calculate_drag(relative_air_velocity(velocity, wind)), 0.0)

    relative = np.array([2.0, -1.0, 0.5])
    drag = calculate_drag(relative, drag_coefficient=1.3, cross_sectional_area=0.05)
    assert np.dot(drag, relative) < 0.0
    np.testing.assert_allclose(drag / np.linalg.norm(drag), -relative / np.linalg.norm(relative))


def test_hover_force_is_invariant_when_at_rest():
    mass = 0.45
    drone = SixDOFInterceptor(mass=mass)
    run(drone, np.array([0.0, 0.0, mass * GRAVITY]), 1.0, 0.01)
    np.testing.assert_allclose(drone.position, 0.0, atol=2e-12)
    np.testing.assert_allclose(drone.velocity, 0.0, atol=2e-12)


def test_falling_body_rk4_converges_at_fourth_order():
    errors = []
    duration = 1.0
    expected = -0.5 * GRAVITY * duration**2
    for dt in (0.1, 0.05, 0.025):
        drone = run(SixDOFInterceptor(drag_coefficient=0.0), np.zeros(3), duration, dt)
        errors.append(abs(drone.position[2] - expected))
    assert max(errors) < 2e-12


def test_harmonic_oscillator_fixture_has_fourth_order_convergence():
    """Independent scalar RK4 fixture, with force evaluated at each stage."""
    omega = 2.3
    duration = 1.0
    exact = np.cos(omega * duration)

    def integrate(dt):
        x, v = 1.0, 0.0
        for _ in range(round(duration / dt)):
            def deriv(state):
                return np.array([state[1], -(omega * omega) * state[0]])
            state = np.array([x, v])
            k1 = deriv(state)
            k2 = deriv(state + 0.5 * dt * k1)
            k3 = deriv(state + 0.5 * dt * k2)
            k4 = deriv(state + dt * k3)
            x, v = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return abs(x - exact)

    errors = [integrate(dt) for dt in (0.2, 0.1, 0.05)]
    ratios = [errors[i] / errors[i + 1] for i in range(2)]
    assert ratios[0] > 10.0 and ratios[1] > 10.0


def test_rotation_matrix_matches_independent_quaternion_reference():
    """A nonzero angular velocity exercises quaternion stage evaluation."""
    drone = SixDOFInterceptor(drag_coefficient=0.0)
    angle = 0.7
    # Rotation about y, represented independently as a unit quaternion.
    drone.quaternion[:] = [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]
    drone.angular_velocity[:] = [0.0, 1.3, 0.0]
    state = drone._state_vector()
    force = np.array([0.0, 0.0, drone.mass * GRAVITY])
    _, accel, qdot, _ = drone.compute_derivatives(force, np.zeros(3), state=state)
    expected_rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    expected_accel = (expected_rotation @ force + np.array([0.0, 0.0, -drone.mass * GRAVITY])) / drone.mass
    np.testing.assert_allclose(accel, expected_accel)
    np.testing.assert_allclose(qdot, [-0.5 * 1.3 * np.sin(angle / 2), 0.0, 0.5 * 1.3 * np.cos(angle / 2), 0.0])


@pytest.mark.parametrize("kwargs", [
    {"mass": 0.0}, {"mass": np.nan}, {"drag_coefficient": -1.0},
    {"drag_coefficient": np.inf}, {"cross_sectional_area": -1.0},
    {"cross_sectional_area": np.nan}, {"wind_velocity": [0.0, np.inf, 0.0]},
])
def test_constructor_rejects_nonphysical_or_nonfinite_parameters(kwargs):
    with pytest.raises(ValueError):
        SixDOFInterceptor(**kwargs)


def test_force_torque_state_and_wind_must_be_finite():
    drone = SixDOFInterceptor(drag_coefficient=0.0)
    with pytest.raises(ValueError):
        drone.compute_derivatives([np.nan, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        drone.compute_derivatives([0.0, 0.0, 0.0], [0.0, 0.0, np.inf])
    with pytest.raises(ValueError):
        drone.compute_derivatives([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], state=np.full(13, np.nan))
    with pytest.raises(ValueError):
        drone.compute_derivatives([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], wind_velocity=[0.0, np.nan, 0.0])


def test_quaternion_remains_normalized_under_torque():
    drone = SixDOFInterceptor(drag_coefficient=0.0)
    for _ in range(2000):
        drone.step_physics(np.array([0.0, 0.0, drone.mass * GRAVITY]), np.array([0.001, -0.002, 0.0005]), 0.002)
    assert np.isclose(np.linalg.norm(drone.quaternion), 1.0, atol=2e-15)


def test_quadratic_drag_timestep_error_decreases_at_fourth_order_reference():
    mass, speed0, duration = 0.45, 8.0, 0.8
    coefficient = 0.5 * 1.225 * 1.3 * 0.05 / mass
    exact = speed0 / (1.0 + coefficient * speed0 * duration)
    errors = []
    for dt in (0.08, 0.04, 0.02):
        drone = SixDOFInterceptor(mass=mass)
        drone.velocity[0] = speed0
        run(drone, np.array([0.0, 0.0, mass * GRAVITY]), duration, dt)
        errors.append(abs(drone.velocity[0] - exact))
    assert errors[1] < errors[0] / 8.0
    assert errors[2] < errors[1] / 8.0


def test_quadratic_drag_with_wind_preserves_fourth_order_convergence():
    mass, speed0, wind_speed, duration = 0.45, 8.0, 2.0, 0.8
    coefficient = 0.5 * 1.225 * 1.3 * 0.05 / mass
    relative_speed = speed0 - wind_speed
    expected_speed = wind_speed + relative_speed / (1.0 + coefficient * relative_speed * duration)
    expected_distance = wind_speed * duration + np.log1p(coefficient * relative_speed * duration) / coefficient
    errors = []
    for dt in (0.08, 0.04, 0.02):
        drone = SixDOFInterceptor(mass=mass, wind_velocity=(wind_speed, 0.0, 0.0))
        drone.velocity[0] = speed0
        for _ in range(round(duration / dt)):
            drone.step_physics(np.array([0.0, 0.0, mass * GRAVITY]), np.zeros(3), dt)
        errors.append(np.linalg.norm(drone.velocity[[0]] - expected_speed) + abs(drone.position[0] - expected_distance))
    assert errors[1] < errors[0] / 8.0
    assert errors[2] < errors[1] / 8.0
