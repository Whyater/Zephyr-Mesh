"""Independent S2 physics checks against analytic and limiting cases."""
import json
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


def test_physics_model_provenance_is_explicit_and_json_safe():
    drone = SixDOFInterceptor(mass=0.45, drag_coefficient=0.8, cross_sectional_area=0.03, wind_velocity=(1.0, -2.0, 0.5))
    metadata = drone.model_provenance()
    assert metadata["status"] == "synthetic"
    assert metadata["integrator"] == "classical_rk4"
    assert metadata["parameters"]["wind_velocity_mps"] == [1.0, -2.0, 0.5]
    assert "not measured" in metadata["evidence_boundary"]
    json.dumps(metadata, sort_keys=True)
    # Metadata is a standalone JSON-safe snapshot, not a live NumPy view.
    metadata["parameters"]["wind_velocity_mps"][0] = 99.0
    assert drone.wind_velocity[0] == 1.0


def test_hover_force_is_invariant_when_at_rest():
    mass = 0.45
    drone = SixDOFInterceptor(mass=mass)
    run(drone, np.array([0.0, 0.0, mass * GRAVITY]), 1.0, 0.01)
    np.testing.assert_allclose(drone.position, 0.0, atol=2e-12)
    np.testing.assert_allclose(drone.velocity, 0.0, atol=2e-12)


def test_falling_body_rk4_is_exact_for_constant_acceleration():
    """Constant acceleration makes position a quadratic in time, which RK4
    reproduces exactly at any step.  Forward Euler would be off by
    0.5 * g * T * dt (0.49 m at dt 0.1).  This checks exactness, not order;
    the order is measured by the spin and drag tests below."""
    errors = []
    duration = 1.0
    expected = -0.5 * GRAVITY * duration**2
    for dt in (0.1, 0.05, 0.025):
        drone = run(SixDOFInterceptor(drag_coefficient=0.0), np.zeros(3), duration, dt)
        errors.append(abs(drone.position[2] - expected))
    assert max(errors) < 2e-12


def test_spin_quaternion_matches_closed_form_rotation_at_fourth_order():
    """Measure the order of the real sim.drone integrator on an exact answer.

    A torque-free body spinning at r rad/s about its z axis keeps that spin,
    and its attitude quaternion is exactly [cos(r t / 2), 0, 0, sin(r t / 2)].
    The quaternion equation is a harmonic oscillator with frequency r / 2, so
    r = 4.6 rad/s matches the old 2.3 rad/s oscillator fixture.  For a
    fourth-order method, halving dt divides the error by 2**4 = 16.
    """
    rate, duration = 4.6, 1.0
    exact = np.array([np.cos(rate * duration / 2), 0.0, 0.0, np.sin(rate * duration / 2)])
    errors = []
    for dt in (0.2, 0.1, 0.05, 0.025):
        drone = SixDOFInterceptor(drag_coefficient=0.0)
        drone.angular_velocity[:] = [0.0, 0.0, rate]
        run(drone, np.array([0.0, 0.0, drone.mass * GRAVITY]), duration, dt)
        errors.append(np.linalg.norm(drone.quaternion - exact))
    ratios = [errors[i] / errors[i + 1] for i in range(3)]
    # Measured about 15.1, 15.8, 15.9; second order would give about 4.
    assert all(13.0 < ratio < 19.0 for ratio in ratios), ratios


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


def test_quadratic_drag_timestep_error_shows_fourth_order_ratio():
    mass, speed0, duration = 0.45, 8.0, 0.8
    coefficient = 0.5 * 1.225 * 1.3 * 0.05 / mass
    exact = speed0 / (1.0 + coefficient * speed0 * duration)
    errors = []
    for dt in (0.08, 0.04, 0.02):
        drone = SixDOFInterceptor(mass=mass)
        drone.velocity[0] = speed0
        run(drone, np.array([0.0, 0.0, mass * GRAVITY]), duration, dt)
        errors.append(abs(drone.velocity[0] - exact))
    # Closed form v0 / (1 + k v0 t).  Fourth order means a ratio near 16 per
    # halving (measured 15.9 and 16.0); third order would give 8.
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    assert all(13.0 < ratio < 19.0 for ratio in ratios), ratios


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


# ---------------------------------------------------------------------------
# Closed-loop controller checks (controller + RK4 body at the S0 dt).
# ---------------------------------------------------------------------------
from sim.controller import GeometricFlightController


def _yawed_drone(yaw_rad, **kwargs):
    drone = SixDOFInterceptor(**kwargs)
    # Unit quaternion for a rotation of yaw_rad about the vertical axis.
    drone.quaternion[:] = [np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)]
    return drone


def _fly(drone, target, duration, dt):
    controller = GeometricFlightController()
    for _ in range(round(duration / dt)):
        force, torque = controller.update_control(drone, target, dt)
        drone.step_physics(force, torque, dt)
    return drone


@pytest.mark.parametrize("yaw_rad", [0.0, np.pi / 4, np.pi / 2, np.pi])
@pytest.mark.parametrize("target", [(0.2, 0.0, 3.0), (2.0, 0.0, 2.0), (0.0, 2.0, 2.0), (-1.5, 1.0, 2.5)])
def test_lateral_step_converges_at_s0_dt_for_any_initial_yaw(target, yaw_rad):
    """Hand-justified convergence bound for a lateral step at dt = 0.01 s.

    Linearised outer loop per axis: x'' = -Kp x_err - Kd v.  Horizontal axes
    use Kp = 2, Kd = 1.5, so the natural frequency is sqrt(2) = 1.41 rad/s and
    the damping ratio is 1.5 / (2 * 1.41) = 0.53.  The error envelope is
    step * exp(-0.53 * 1.41 * t) / sqrt(1 - 0.53**2) = 2.36 * exp(-0.75 t) m
    for the largest 2 m step, which is 0.0013 m at t = 10 s.  The vertical
    axis (Kp = 3, Kd = 2) decays faster.  The attitude loop adds a short lag,
    so the test allows 0.05 m and 0.05 m/s, about 40 times the envelope.
    Yaw pi is the case where body and ground axes are most opposed.
    """
    drone = _fly(_yawed_drone(yaw_rad), np.array(target), duration=10.0, dt=0.01)
    assert np.linalg.norm(drone.position - np.array(target)) < 0.05
    assert np.linalg.norm(drone.velocity) < 0.05


def test_yaw_rate_damping_matches_sampled_hold_prediction_at_s0_dt():
    """Discrete-time check of the zero-order-hold rate loop.

    With the drone level and at its target, the only torque is yaw-rate
    damping tau_z = -Kd_z * r, held constant over each step.  A constant
    torque makes r change linearly inside the step, which RK4 integrates
    exactly, so r_next = r * (1 - b * dt) with b = Kd_z / I_z.  The loop
    decays only if |1 - b * dt| < 1, which means b * dt < 2.  This test
    measures the per-step factor from the simulator and checks it against
    that rule; it does not read the gains it is checking.
    """
    dt = 0.01
    drone = SixDOFInterceptor(drag_coefficient=0.0)
    drone.angular_velocity[:] = [0.0, 0.0, 1.0]
    controller = GeometricFlightController()
    rates = [drone.angular_velocity[2]]
    for _ in range(20):
        force, torque = controller.update_control(drone, drone.position.copy(), dt)
        drone.step_physics(force, torque, dt)
        rates.append(drone.angular_velocity[2])
    factors = np.array(rates[1:]) / np.array(rates[:-1])
    # One constant factor per step, as the hold-plus-RK4 argument predicts.
    np.testing.assert_allclose(factors, factors[0], rtol=1e-9)
    # Stability rule |1 - b dt| < 1, with margin: the rate must shrink each step
    # without flipping sign (factor between 0 and 1).
    assert 0.0 < factors[0] < 1.0
    assert abs(rates[-1]) < 0.1


def test_profile_vehicle_inertia_matches_hand_estimate_for_78_g_profile():
    """Hand estimate for the 78 g default profile (arm d = 0.10 m).

    This checks the stated model: motor point masses plus a frame plate, with
    d taken as the axis-projected arm (motors at +-d on both body axes, the
    convention in HardwareProfile).  It is not the review's cruder estimate,
    which put the motors at d / sqrt(2) from each axis (about 3e-4 kg m^2).

    Four motor+prop point masses (0.012 kg each) sit at (+-d, +-d), so
    I_x = I_y = 4 * 0.012 * 0.1**2 = 4.8e-4 kg m^2 and I_z = 9.6e-4 kg m^2.
    The 0.03 kg frame is treated as a uniform square plate spanning the motor
    square (side 2d); its moments are integrated numerically here instead of
    using the closed form, giving about 1.0e-4 (x, y) and 2.0e-4 (z).
    """
    from sim.scenarios import default_hardware_profile
    from sim.vehicle import ActuatedQuadrotor

    profile = default_hardware_profile(0)
    d = profile.arm_length_m
    rotor_mass = profile.motor.mass_kg + profile.propeller.mass_kg
    motors = np.array([[d, d], [d, -d], [-d, -d], [-d, d]])
    ix = np.sum(rotor_mass * motors[:, 1] ** 2)
    iy = np.sum(rotor_mass * motors[:, 0] ** 2)
    iz = np.sum(rotor_mass * (motors[:, 0] ** 2 + motors[:, 1] ** 2))
    n = 400
    cells = (np.arange(n) + 0.5) / n * 2 * d - d
    gx, gy = np.meshgrid(cells, cells)
    cell_mass = profile.frame_mass_kg / n**2
    ix += np.sum(cell_mass * gy**2)
    iy += np.sum(cell_mass * gx**2)
    iz += np.sum(cell_mass * (gx**2 + gy**2))
    vehicle = ActuatedQuadrotor(profile)
    np.testing.assert_allclose([vehicle.drone.I_x, vehicle.drone.I_y, vehicle.drone.I_z], [ix, iy, iz], rtol=1e-4)
    # Order-of-magnitude anchor: about 5.8e-4, far below the 450 g frame value.
    assert 5.0e-4 < vehicle.drone.I_x < 7.0e-4


@pytest.mark.parametrize("yaw_rad", [0.0, np.pi / 2, np.pi, -2.0])
def test_tilt_torque_points_the_right_way_in_the_ground_frame(yaw_rad):
    """A target straight ahead in +x needs the roof tipped toward +x.

    Tipping the body z axis toward +x is a rotation about the ground +y axis
    (right-hand rule).  The controller returns a body-frame torque, so mapping
    it back with the body-to-ground rotation must give a mostly +y vector, for
    any starting yaw.  At yaw pi the body y axis points along ground -y, which
    is exactly where an unrotated (ground-frame) tilt error has the wrong sign.
    """
    drone = _yawed_drone(yaw_rad, drag_coefficient=0.0)
    _, torque_body = GeometricFlightController().update_control(drone, [1.0, 0.0, 0.0], 0.01)
    qw, qx, qy, qz = drone.quaternion
    rotation = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy)],
        [2 * (qx*qy + qw*qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qw*qx)],
        [2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2 * (qx**2 + qy**2)],
    ])
    torque_ground = rotation @ torque_body
    assert torque_ground[1] > 0.0
    assert abs(torque_ground[0]) < 1e-9 * abs(torque_ground[1]) + 1e-15


def test_s2_convergence_artifact_regenerates_from_real_integrator():
    """sim/runs/s2-physics/convergence.json must come from tools/make_s2_convergence.py.

    The script integrates sim.drone itself.  As a control, the same system is
    stepped here with forward Euler (one slope per step, written separately
    below): its error ratio per halving is about 2, so a ratio near 16 really
    does identify fourth order rather than any convergent method.
    """
    from pathlib import Path
    from tools import make_s2_convergence as gen

    committed = json.loads((Path(__file__).parents[1] / "sim" / "runs" / "s2-physics" / "convergence.json").read_text())
    fresh = gen.build_convergence()
    assert committed["fixture"].startswith("sim.drone.SixDOFInterceptor")
    assert [row["dt_s"] for row in committed["results"]] == [row["dt_s"] for row in fresh["results"]]
    np.testing.assert_allclose(
        [row["absolute_error"] for row in committed["results"]],
        [row["absolute_error"] for row in fresh["results"]],
        rtol=1e-6,
    )
    ratios = [row["error_ratio_to_next"] for row in fresh["results"][:-1]]
    assert all(14.0 < ratio < 18.0 for ratio in ratios), ratios

    def euler_final(dt):
        drone = gen._initial_drone()
        force, torque = np.asarray(gen.BODY_FORCE_N), np.asarray(gen.BODY_TORQUE_NM)
        for _ in range(round(gen.DURATION_S / dt)):
            state = drone._state_vector()
            state = state + dt * np.concatenate(drone.compute_derivatives(force, torque, state=state))
            drone.position, drone.velocity = state[0:3], state[3:6]
            drone.quaternion = state[6:10] / np.linalg.norm(state[6:10])
            drone.angular_velocity = state[10:13]
        return drone._state_vector()

    reference = gen.final_state(gen.STEPS_S[-1] / gen.REFERENCE_DIVISOR)
    euler_errors = [np.linalg.norm(euler_final(dt) - reference) for dt in gen.STEPS_S[:3]]
    euler_ratios = [euler_errors[i] / euler_errors[i + 1] for i in range(2)]
    assert all(1.6 < ratio < 2.6 for ratio in euler_ratios), euler_ratios
