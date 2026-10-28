"""Independent convergence-order checks for the translational integrator.

The rigid body in sim/drone.py claims to use classical fourth-order
Runge-Kutta (RK4). A fourth-order method has a global error that shrinks
like dt**4, so halving dt should cut the error by about 2**4 = 16. Forward
Euler (first order) would only cut it by about 2, and a second-order method
(midpoint or Heun) by about 4.

What is independent here: the expected answers are closed-form solutions of
the motion equations with quadratic drag, worked out by hand below and
checked by differentiating them. Nothing is computed by calling the code
under test or a copy of it. The order 16 comes from the theory of RK4, not
from a measured run. Existing tests check fourth order on rotation drift and
on a one-axis horizontal coast speed; these tests add the vertical fall with
gravity plus drag (position and speed), and a slanted 3D coast that only
works if drag uses the full speed of the body.
"""
from __future__ import annotations

import numpy as np

from sim.drone import SixDOFInterceptor
from sim.environment import AIR_DENSITY, GRAVITY

# Physical constants used by the hand solutions. g = 9.81 m/s^2 is the
# usual rounded surface gravity and 1.225 kg/m^3 is sea-level air density in
# the standard atmosphere. The simulator states the same two values; the
# guard test below makes sure they still agree, so a failure points at the
# constants and not at the integrator.
G = 9.81
RHO = 1.225

# Airframe numbers passed in explicitly (a 450 g quad with a drag
# coefficient of 1.3 and a 0.05 m^2 frontal area, the simulator defaults).
MASS = 0.45
CD = 1.3
AREA = 0.05

# Quadratic drag force has size 0.5 * rho * Cd * A * speed^2 and points
# against the motion. Dividing by mass gives a deceleration c * speed^2 with
# c = 0.5 * rho * Cd * A / m (units 1/m).
C = 0.5 * RHO * CD * AREA / MASS

# Three step sizes, each half the last. Chosen so the errors sit in the
# range where the dt**4 law holds (coarser steps showed ratios drifting
# toward 13 to 19) but stay far above float round-off: the finest errors are
# around 1e-9, and round-off on numbers of size 10 over 160 steps is near
# 1e-13.
STEP_SIZES = (0.05, 0.025, 0.0125)
DURATION = 2.0

# RK4 theory gives 16 per halving. The band 12 to 20 allows the leftover
# higher-order terms, which still move the ratio a few percent at these
# steps, while a second-order method (about 4) or Euler (about 2) is far
# outside it.
ORDER_BAND = (12.0, 20.0)


def _run(drone, force_body, dt):
    count = round(DURATION / dt)
    assert np.isclose(count * dt, DURATION)
    for _ in range(count):
        drone.step_physics(force_body, np.zeros(3), dt)
    return drone


def test_hand_constants_match_simulator_constants():
    """Guard: the closed forms below assume the simulator's g and rho."""
    assert GRAVITY == G
    assert AIR_DENSITY == RHO


def test_fall_with_drag_converges_at_fourth_order_to_tanh_and_log_cosh():
    """Independent fact: a body dropped from rest with quadratic drag.

    While falling, the downward speed s obeys ds/dt = g - c s^2. With the
    terminal speed vt = sqrt(g / c), the solution is
        s(t) = vt * tanh(g t / vt)
        drop(t) = (vt^2 / g) * ln(cosh(g t / vt)).
    Check by differentiating (AP Calc chain rule):
        d/dt [vt tanh(g t / vt)] = g sech^2(g t / vt) = g (1 - tanh^2)
                                 = g - g s^2 / vt^2 = g - c s^2,
        d/dt [(vt^2 / g) ln cosh(g t / vt)] = vt tanh(g t / vt) = s.
    Both start at 0 when t = 0, matching a drop from rest.

    The error in height and in vertical speed must fall by about 16 each time
    dt is halved. Dropping the drag term, flipping its sign, or using a lower
    order method all break that ratio.
    """
    terminal_speed = np.sqrt(G / C)
    phase = G * DURATION / terminal_speed
    expected_z = -(terminal_speed**2 / G) * np.log(np.cosh(phase))
    expected_vz = -terminal_speed * np.tanh(phase)

    height_errors = []
    speed_errors = []
    for dt in STEP_SIZES:
        drone = SixDOFInterceptor(mass=MASS, drag_coefficient=CD, cross_sectional_area=AREA)
        _run(drone, np.zeros(3), dt)
        height_errors.append(abs(drone.position[2] - expected_z))
        speed_errors.append(abs(drone.velocity[2] - expected_vz))
        # No sideways force or wind, so nothing should move sideways.
        np.testing.assert_allclose(drone.position[:2], 0.0, atol=1e-15)

    for errors in (height_errors, speed_errors):
        assert errors[-1] > 1e-11, f"errors too close to round-off to read an order: {errors}"
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        print("fall error", errors, "ratios", ratios)
        assert all(ORDER_BAND[0] < ratio < ORDER_BAND[1] for ratio in ratios), ratios


def test_slanted_coast_keeps_its_line_and_converges_at_fourth_order():
    """Independent fact: drag acts on the whole speed, along the motion.

    Hover thrust m g along body z (level body) cancels gravity exactly, so the
    only force left is drag, which points straight back along the velocity.
    A force along the velocity cannot turn it, so the body keeps moving on
    its starting line in direction u, and its speed s obeys ds/dt = -c s^2.
    The solution is
        s(t) = s0 / (1 + c s0 t),   distance(t) = ln(1 + c s0 t) / c.
    Check: d/dt [s0 / (1 + c s0 t)] = -c s0^2 / (1 + c s0 t)^2 = -c s^2, and
    d/dt [ln(1 + c s0 t) / c] = s0 / (1 + c s0 t) = s.

    The direction u = (3, 4, 12) / 13 mixes all three axes. If drag were
    applied axis by axis (each component slowed by its own square) the
    direction would bend and the error would not shrink at all. The position
    and velocity error vectors must shrink by about 16 per halving of dt.
    """
    direction = np.array([3.0, 4.0, 12.0]) / 13.0
    start_speed = 12.0
    expected_speed = start_speed / (1.0 + C * start_speed * DURATION)
    expected_distance = np.log1p(C * start_speed * DURATION) / C
    hover_force = np.array([0.0, 0.0, MASS * G])

    position_errors = []
    velocity_errors = []
    for dt in STEP_SIZES:
        drone = SixDOFInterceptor(mass=MASS, drag_coefficient=CD, cross_sectional_area=AREA)
        drone.velocity[:] = start_speed * direction
        _run(drone, hover_force, dt)
        position_errors.append(np.linalg.norm(drone.position - expected_distance * direction))
        velocity_errors.append(np.linalg.norm(drone.velocity - expected_speed * direction))

    for errors in (position_errors, velocity_errors):
        assert errors[-1] > 1e-11, f"errors too close to round-off to read an order: {errors}"
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        print("slanted coast error", errors, "ratios", ratios)
        assert all(ORDER_BAND[0] < ratio < ORDER_BAND[1] for ratio in ratios), ratios
