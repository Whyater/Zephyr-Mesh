"""Independent delay-margin check for the vertical position loop.

Scope: a pure vertical step on the drag-free rigid body, with position and
velocity feedback delayed by ``FeedbackDelayController``.  Attitude stays
exactly level, so the loop is z'' = Kp (z_ref - z(t - d)) - Kd z'(t - d)
with Kp = 3 and Kd = 2.  This does not cover the lateral channel or the
actuated vehicle used by ``find_d_crit``; those add attitude and motor lag.

Reference, derived by hand and typed here (nothing from ``sim`` is used):
the loop gain is L(s) = (Kd s + Kp) exp(-s d) / s**2.  A double integrator
shrinks a wave of frequency w by w**2 and puts it half a cycle (pi rad)
behind; the PD terms scale it by sqrt(Kp**2 + Kd**2 w**2) and pull it
atan(Kd w / Kp) ahead; the delay only puts it w d further behind.
The loop is on the edge when the size is 1 and the total lag is pi:
  size 1:  w**4 - Kd**2 w**2 - Kp**2 = 0, so w*^2 = 2 + sqrt(13),
           w* = 2.3676 rad/s
  lag pi:  w* d = atan(Kd w* / Kp) = 1.00607 rad, so d* = 0.42493 s.
Differentiating s**2 + (Kd s + Kp) exp(-s d) = 0 with respect to d gives
ds/dd = -s**3 / (2 s + Kd exp(-s d) + d s**2).  Its real part at s = j w*,
d = d* is +3.676 per s per s: the oscillation grows for d > d* and decays
for d < d*, with growth rate sigma ~ 3.676 (d - d*).

Sampling correction: the delayed (z, v) is sent on even 10 ms ticks (50 Hz),
so the age of the measurement alternates m and m + 1 ticks (average +0.005 s),
and each command is held for one 10 ms tick (average +0.005 s).  The
simulator's boundary in nominal delay is therefore predicted at
d* - 0.010 = 0.4149 s.  Treating the holds as their average delay is off by
terms of order (w* T)**2 / 24, about 1e-4 in loop gain, which moves the
boundary by about 4e-5 s; linear interpolation of sigma between 0.41 s and
0.42 s adds about 6e-5 s.  The tolerance is 0.0025 s (a quarter tick), about
20 times those errors combined, while a delay off by one 10 ms tick misses
the prediction by about 0.01 s and fails.
"""

from __future__ import annotations

import cmath

import numpy as np

from sim.controller import GeometricFlightController
from sim.drone import SixDOFInterceptor
from sim.physics.control_tolerance import FeedbackDelayController

K_P = 3.0
K_D = 2.0
DT = 0.01
TARGET_Z = 1.02
_CACHE: dict[tuple[float, type], tuple[float, float]] = {}


def _hand_numbers() -> tuple[float, float, float]:
    w_sq = (K_D**2 + np.sqrt(K_D**4 + 4.0 * K_P**2)) / 2.0
    w_star = float(np.sqrt(w_sq))
    phase_margin = float(np.arctan(K_D * w_star / K_P))
    d_star = phase_margin / w_star
    s = 1j * w_star
    slope = float(np.real(-s**3 / (2.0 * s + K_D * cmath.exp(-s * d_star) + d_star * s**2)))
    return w_star, d_star, slope


def _growth_rate(delay_s, *, controller_factory=FeedbackDelayController):
    key = (float(delay_s), controller_factory)
    if key in _CACHE:
        return _CACHE[key]
    drone = SixDOFInterceptor(init_pos=(0.0, 0.0, 1.0), drag_coefficient=0.0)
    state = (drone.position.copy(), drone.velocity.copy())
    controller = controller_factory(float(delay_s), 0.0, np.random.default_rng(0), initial_state=state)
    target = np.array([0.0, 0.0, TARGET_Z])
    times = []
    errors = []
    for k in range(int(40.0 / DT)):
        force, torque = controller.update_control(drone, target, DT)
        drone.step_physics(force, torque, DT)
        times.append((k + 1) * DT)
        errors.append(drone.position[2] - TARGET_Z)
    times = np.asarray(times)
    amplitudes = np.abs(np.asarray(errors))
    peak_indices = np.flatnonzero((amplitudes[1:-1] > amplitudes[:-2]) & (amplitudes[1:-1] >= amplitudes[2:])) + 1
    peak_indices = peak_indices[times[peak_indices] >= 10.0]
    if len(peak_indices) < 3:
        raise AssertionError("not enough post-transient error peaks for a growth-rate fit")
    peak_times = times[peak_indices]
    peak_values = amplitudes[peak_indices]
    sigma = float(np.polyfit(peak_times, np.log(peak_values), 1)[0])
    omega = float(np.pi / np.mean(np.diff(peak_times)))
    _CACHE[key] = (sigma, omega)
    return sigma, omega


def _boundary_error(controller_factory=FeedbackDelayController) -> float:
    _, d_star, _ = _hand_numbers()
    sigma_low, _ = _growth_rate(0.41, controller_factory=controller_factory)
    sigma_high, _ = _growth_rate(0.42, controller_factory=controller_factory)
    d_sim = 0.41 - sigma_low * 0.01 / (sigma_high - sigma_low)
    return abs(d_sim + 0.010 - d_star)


def test_hand_margin_numbers():
    w_star, d_star, slope = _hand_numbers()
    assert abs(w_star - 2.3676) <= 1e-4
    assert abs(d_star - 0.4249) <= 1e-4
    assert abs(slope - 3.676) <= 0.01
    assert np.max(np.real(np.roots([1.0, 2.0, 3.0]))) < 0.0
    controller = GeometricFlightController()
    assert controller.K_p_pos[2] == K_P, "vertical position gain changed; update the independent reference"
    assert controller.K_d_pos[2] == K_D, "vertical velocity gain changed; update the independent reference"


def test_sim_stays_stable_just_below_margin_and_grows_just_above():
    sigma_40, _ = _growth_rate(0.40)
    sigma_41, _ = _growth_rate(0.41)
    sigma_42, _ = _growth_rate(0.42)
    sigma_43, _ = _growth_rate(0.43)
    # 0.41 s and 0.42 s straddle the predicted 0.4149 s boundary.
    assert sigma_40 < sigma_41 < 0.0
    assert 0.0 < sigma_42 < sigma_43


def test_sim_boundary_matches_hand_margin_within_quarter_tick():
    w_star, _, _ = _hand_numbers()
    _, omega_low = _growth_rate(0.41)
    _, omega_high = _growth_rate(0.42)
    assert _boundary_error() <= 0.0025
    assert abs(omega_low - w_star) / w_star <= 0.01
    assert abs(omega_high - w_star) / w_star <= 0.01


def test_growth_rate_slope_matches_hand_slope():
    _, _, slope = _hand_numbers()
    sigma_low, _ = _growth_rate(0.41)
    sigma_high, _ = _growth_rate(0.42)
    assert abs((sigma_high - sigma_low) / 0.01 - slope) / slope <= 0.05


def test_planted_one_tick_shorter_delay_is_caught():
    class OneTickShorterDelay(FeedbackDelayController):
        def __init__(self, delay_s, loss_p, rng, *, initial_state=None, inner=None):
            super().__init__(delay_s, loss_p, rng, initial_state=initial_state, inner=inner)
            self.channel.delay_s = self.delay_s - 0.01

    assert _boundary_error() <= 0.0025
    assert _boundary_error(OneTickShorterDelay) > 0.0025
    assert _growth_rate(0.42, controller_factory=OneTickShorterDelay)[0] < 0.0
