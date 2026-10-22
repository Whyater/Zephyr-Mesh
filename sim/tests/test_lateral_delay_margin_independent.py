"""Independent delay-margin check for the lateral (x-axis) position loop.

Scope: the full motor-driven vehicle used by ``find_d_crit`` (arm "b"), with
position and velocity feedback delayed by ``FeedbackDelayController`` and the
attitude loop, rotor lag and attitude kept fresh.  Both profiles have no drag.
The vertical version of this check is ``test_delay_margin_independent.py``.

Model kept (small motion about hover, x axis only):
  x'' = g * theta                     (tilt makes sideways acceleration)
  theta_des = a_des / g,  a_des = Kp (x_ref - x_meas) - Kd v_meas
  theta'' = lag[Kp_att (theta_des - theta) - Kd_att theta']
with Kp = 2, Kd = 1.5, Kp_att = 100, Kd_att = 18 (per unit inertia) and a
first-order rotor lag tau (0.070 s esp-fly, 0.072 s crazyflie).  Mass,
inertia, g and the thrust coefficient cancel.  Dropped, each zero at first
order for a pure x step: vertical coupling, gyroscopic terms, roll, yaw and
y, and the atan, sine and thrust-projection geometry of the controller.  The
rotor speed-squared law only sets a gain, and it cancels against the mixer's
square-root command.  Drag and battery sag are off in both profiles.

Reference, derived by hand and typed here (nothing from ``sim`` is used):
the loop gain is L(s) = (Kp + Kd s) T_in(s) exp(-s d) / s**2 with
T_in(s) = Kp_att / (tau s**3 + s**2 + Kd_att s + Kp_att), the attitude loop.
A wobble of frequency w comes back the same size when |L(j w)| = 1; with
u = w**2 that is a polynomial, solved below, giving w* = 1.8228 rad/s (one
wobble every 3.45 s).  At w* only timing matters.  Integrating twice puts
the reply half a cycle (180 degrees) late, the velocity term pulls it
atan(Kd w* / Kp) = 53.8 degrees earlier and the attitude loop pushes it
18.5 degrees later, leaving a phase margin of 35.3 degrees (0.616 rad).
A delay d turns cos(w t) into cos(w (t - d)), the same wave w d radians
later, so the margin is used up at d* = 0.6161 rad / 1.8228 rad/s, about
0.338 s.  (In the formulas, s stands for j w, a wave of frequency w, with j
the square root of -1.)  The
rotor lag raises d* by about 2 ms: inside the fast attitude loop it slightly
cuts that loop's lag at 1.8 rad/s.

Sampling correction: the delayed (x, v) is sent on even 10 ms ticks, so the
packet the controller uses is n or n + 1 ticks old: on average 5 ms older
than the nominal delay.  Each command is also held for its 10 ms tick, but
the attitude loop rebuilds the command every tick from a fresh tilt and spin
reading, so that hold acts like 5 ms of extra rotor lag and costs almost
nothing here (+0.15 ms).  The simulator's boundary is therefore predicted at
d* - 0.005 = 0.3330 s.  (The vertical test subtracts 10 ms because its held
thrust goes straight into acceleration with no feedback inside the tick.)
Exact sampled-data models, written separately three times as scratch
scripts that are not kept in this repo, put the boundary at 0.33314 s
(esp-fly) and 0.33319 s (crazyflie) when interpolated like this test does,
so the simple rule is 0.19 ms low (0.154 ms rule residual plus 0.035 ms from
straight-line interpolation; peak-fit and small-amplitude effects are below
1 us).  The healthy simulator measured with this file's own helpers gives
the same 0.19 ms.  The tolerance is 1.0 ms, about 5 times that, while the
smallest planted bug (rotor lag removed) moves the boundary 1.8 ms and
fails.  It is weak on small rotor-lag errors: making the rotor-speed lag
use 1.1 or 0.5 times the time constant moved the boundary 0.4 and 0.7 ms
(passes), while 2 times moved it 2.0 ms (caught).

Growth slope: the loop is on the edge where F(s, d) = s**2 + (Kp + Kd s)
T_in(s) exp(-s d) = 0.  Differentiating with respect to d (implicit
differentiation, with F_d and F_s the derivatives of F with respect to d and
s) gives ds/dd = -F_d / F_s; its real part at s = j w* is about
+2.17 per s per s, so 10 ms past the boundary the wobble grows about 2
percent per second.  The test compares the secant between 0.33 s and 0.34 s,
which sits 0.6 percent below the tangent because the growth rate bends down.

find_d_crit: it reports the smallest 10 ms grid delay at which a 1 m step,
run for 20 s, has its last-third error peak above 1.2 times its middle-third
peak.  That needs a growth rate of ln(1.2) over 3 or 4 half-periods, which is
12 to 16 ms past the linear boundary, so the reported value is at least
0.35 s.  A 1 m step tilts the drone 15 to 30 degrees, and that is
stabilizing (the extra lift tracks the tilt command, which leads the actual
tilt), so it pushes the flag later; the simulator's own 1 m window ratios
sit below the small-step ones, which shows the sign.  Ruling out 0.37 s rests
partly on two small nonlinear planar models written outside this repo, not
on hand reasoning alone.  The simulator's 0.35 s run is about 1 percent under
the 1.2 threshold, so both 0.35 s and 0.36 s are allowed.  The interval
catches a delay one tick too short (it reports 0.37 s) but cannot see one
tick too long (0.35 s); the boundary tests catch both.
"""

from __future__ import annotations

import cmath
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sim.controller import GeometricFlightController
from sim.hardware import load_hardware_profile
from sim.physics import control_tolerance
from sim.physics.control_tolerance import FeedbackDelayController, find_d_crit
from sim.physics.delay_buffer import DelayLossChannel
from sim.vehicle import ActuatedQuadrotor

# Gains and timing typed by hand from the controller source.
K_P_XY = 2.0
K_D_XY = 1.5
K_P_ATT = 100.0
K_D_ATT = 18.0
DT = 0.01

# Motor time constant in seconds for each profile, typed by hand.
PROFILES = {"esp_fly_class": 0.070, "crazyflie_class": 0.072}

# Planted-bug runs use one profile to keep the file affordable.
BUG_PROFILE = "crazyflie_class"

# Healthy boundary error is 0.19 ms; see the module docstring.
TOL_BOUNDARY_S = 0.001
# The wobble frequency drifts -0.71 rad/s per s of delay, so at 0.33 s and
# 0.34 s it sits +0.12 and -0.32 percent from w*.
TOL_OMEGA_REL = 0.005
# The 0.33 to 0.34 s secant sits 0.56 to 0.59 percent below the tangent slope.
TOL_SLOPE_REL = 0.015

# What find_d_crit may report; see the module docstring.
DCRIT_INTERVAL_S = (0.35, 0.36)

_REFERENCE_KEYS = (
    "w_star",
    "phase_margin_rad",
    "d_star_continuous_s",
    "sampling_offset_s",
    "d_boundary_sim_s",
    "slope_per_s2",
    "dcrit_interval_s",
)

_GROWTH_CACHE: dict[tuple, tuple[float, float]] = {}
_DCRIT_CACHE: dict[tuple, dict] = {}


def _hand_reference(tau_s: float) -> dict:
    """Hand-derived lateral margin numbers for motor time constant ``tau_s``."""
    # Size 1: u**2 ((Kp_att - u)**2 + u (Kd_att - tau u)**2) = Kp_att**2 (Kp**2 + Kd**2 u), u = w**2.
    u = np.polynomial.Polynomial([0.0, 1.0])
    size_one = (u**2 * ((K_P_ATT - u) ** 2 + u * (K_D_ATT - tau_s * u) ** 2)
                - K_P_ATT**2 * (K_P_XY**2 + K_D_XY**2 * u))
    positive = [r.real for r in size_one.roots() if abs(r.imag) < 1e-9 and r.real > 0.0]
    assert len(positive) == 1, "expected exactly one gain crossover"
    w_star = math.sqrt(positive[0])
    # Spare lag at w*: the velocity term's lead minus the attitude loop's lag.
    lead = math.atan(K_D_XY * w_star / K_P_XY)
    lag = math.atan2(K_D_ATT * w_star - tau_s * w_star**3, K_P_ATT - w_star**2)
    phase_margin = lead - lag
    d_star = phase_margin / w_star
    offset = DT / 2.0
    # Slope: ds/dd = -F_d / F_s for F(s, d) = s**2 + G(s) exp(-s d).
    s = 1j * w_star
    den = tau_s * s**3 + s**2 + K_D_ATT * s + K_P_ATT
    den_prime = 3.0 * tau_s * s**2 + 2.0 * s + K_D_ATT
    g = (K_P_XY + K_D_XY * s) * K_P_ATT / den
    g_prime = K_D_XY * K_P_ATT / den - (K_P_XY + K_D_XY * s) * K_P_ATT * den_prime / den**2
    e = cmath.exp(-s * d_star)
    slope = (-(-s * g * e) / (2.0 * s + (g_prime - d_star * g) * e)).real
    return {
        "w_star": w_star,
        "phase_margin_rad": phase_margin,
        "d_star_continuous_s": d_star,
        "sampling_offset_s": offset,
        "d_boundary_sim_s": d_star - offset,
        "slope_per_s2": slope,
        "dcrit_interval_s": DCRIT_INTERVAL_S,
        "characteristic_residual": abs(s**2 + g * e),
    }


def _profile_path(name: str) -> str:
    root = Path(__file__).resolve().parents[2]
    return str(root / "sim" / "physics" / "profiles" / f"{name}.json")


def _vehicle(name, delay_s, controller_factory=FeedbackDelayController, motor_tau_s=None, inner=None):
    """Build the arm "b" vehicle exactly as ``run_episode`` does."""
    profile = load_hardware_profile(_profile_path(name))
    vehicle = ActuatedQuadrotor(
        profile,
        initial_position=(0.0, 0.0, 1.0),
        controller=GeometricFlightController(),
        substeps=1,
    )
    vehicle.actuator.speed_rad_s[:] = profile.hover_speed_rad_s()
    if motor_tau_s is not None:
        vehicle.actuator.motor_time_constant_s = float(motor_tau_s)
    drone = vehicle.drone
    vehicle.controller = controller_factory(
        float(delay_s),
        0.0,
        np.random.default_rng(0),
        initial_state=(drone.position.copy(), drone.velocity.copy()),
        inner=inner,
    )
    return vehicle


def _growth_rate(
    name,
    delay_s,
    *,
    controller_factory=FeedbackDelayController,
    motor_tau_s=None,
    inner_factory=None,
    step_m=0.01,
    duration_s=30.0,
    fit_from_s=10.0,
):
    """Return (sigma per s, omega rad/s) of the x error after a small x step."""
    key = (name, float(delay_s), controller_factory, motor_tau_s, inner_factory,
           float(step_m), float(duration_s), float(fit_from_s))
    if key in _GROWTH_CACHE:
        return _GROWTH_CACHE[key]
    inner = None if inner_factory is None else inner_factory()
    vehicle = _vehicle(name, delay_s, controller_factory, motor_tau_s, inner)
    target = (float(step_m), 0.0, 1.0)
    n_steps = round(duration_s / DT)
    times = np.empty(n_steps)
    errors = np.empty(n_steps)
    for k in range(n_steps):
        vehicle.step(target, DT)
        times[k] = (k + 1) * DT
        errors[k] = vehicle.drone.position[0] - step_m
    amplitudes = np.abs(errors)
    peak_indices = np.flatnonzero((amplitudes[1:-1] > amplitudes[:-2]) & (amplitudes[1:-1] >= amplitudes[2:])) + 1
    peak_indices = peak_indices[times[peak_indices] >= fit_from_s]
    if len(peak_indices) < 4:
        raise AssertionError("not enough post-transient error peaks for a growth-rate fit")
    peak_times = times[peak_indices]
    peak_values = amplitudes[peak_indices]
    sigma = float(np.polyfit(peak_times, np.log(peak_values), 1)[0])
    omega = float(np.pi / np.mean(np.diff(peak_times)))
    _GROWTH_CACHE[key] = (sigma, omega)
    return sigma, omega


def _interpolated_boundary(name, lo_s, hi_s, **kwargs):
    """Straight-line zero crossing of sigma between two delays.

    Returns (boundary_s, sigma_lo, sigma_hi, omega_lo, omega_hi).
    """
    sigma_lo, omega_lo = _growth_rate(name, lo_s, **kwargs)
    sigma_hi, omega_hi = _growth_rate(name, hi_s, **kwargs)
    if sigma_hi == sigma_lo:
        raise AssertionError("growth rate does not change between the two delays")
    boundary = lo_s - sigma_lo * (hi_s - lo_s) / (sigma_hi - sigma_lo)
    return boundary, sigma_lo, sigma_hi, omega_lo, omega_hi


def _dcrit(name):
    if name not in _DCRIT_CACHE:
        _DCRIT_CACHE[name] = find_d_crit(_profile_path(name), arm="b", duration_s=20.0)
    return _DCRIT_CACHE[name]


def _grid_delays(d_boundary_s: float) -> tuple[float, float]:
    """The two 10 ms grid delays that straddle ``d_boundary_s``."""
    low_tick = math.floor(d_boundary_s / DT)
    high_tick = math.ceil(d_boundary_s / DT)
    if high_tick == low_tick:
        high_tick = low_tick + 1
    return round(low_tick * DT, 10), round(high_tick * DT, 10)


def _boundary_error(name, **kwargs) -> float:
    reference = _hand_reference(PROFILES[name])
    d_boundary = reference["d_boundary_sim_s"]
    lo_s, hi_s = _grid_delays(d_boundary)
    boundary = _interpolated_boundary(name, lo_s, hi_s, **kwargs)[0]
    return abs(boundary - d_boundary)


class OneTickShorterDelay(FeedbackDelayController):
    """Planted bug (a): the channel delay is silently 10 ms shorter."""

    def __init__(self, delay_s, loss_p, rng, *, initial_state=None, inner=None):
        super().__init__(delay_s, loss_p, rng, initial_state=initial_state, inner=inner)
        self.channel.delay_s = self.delay_s - 0.01


class EveryTickOffers(FeedbackDelayController):
    """Planted bug (b): feedback is offered on every tick, not every even tick."""

    def __init__(self, delay_s, loss_p, rng, *, initial_state=None, inner=None):
        super().__init__(delay_s, loss_p, rng, initial_state=initial_state, inner=inner)
        # The channel only accepts offers on its own sample grid.
        self.channel = DelayLossChannel(self.delay_s, self.loss_p, DT, rng, initial_value=initial_state)

    def update_control(self, drone, target_position, dt, **kwargs):
        # The base class offers on even ticks; add the odd ticks here.
        if self._step_index % 2 == 1:
            self.channel.offer(self._time_s, (drone.position.copy(), drone.velocity.copy()))
        return super().update_control(drone, target_position, dt, **kwargs)


def _inner_with_low_rate_gain():
    """Planted bug (c): inner controller with K_d_att = 16 on roll and pitch."""
    inner = GeometricFlightController()
    inner.K_d_att[:2] = 16.0
    return inner


def _assert_bug_is_caught(**bug):
    assert _boundary_error(BUG_PROFILE) <= TOL_BOUNDARY_S
    assert _boundary_error(BUG_PROFILE, **bug) > TOL_BOUNDARY_S


def test_typed_constants_match_live_code():
    controller = GeometricFlightController()
    assert controller.K_p_pos[0] == K_P_XY, "x position gain changed; update the independent reference"
    assert controller.K_d_pos[0] == K_D_XY, "x velocity gain changed; update the independent reference"
    for axis in (0, 1):
        assert controller.K_p_att[axis] == K_P_ATT, "attitude gain changed; update the independent reference"
        assert controller.K_d_att[axis] == K_D_ATT, "attitude damping changed; update the independent reference"
    for name, tau_s in PROFILES.items():
        with open(_profile_path(name), encoding="utf-8") as handle:
            document = json.load(handle)
        assert document["rotor"]["motor_time_constant_s"] == tau_s
        assert document["airframe"]["drag_coefficient"] == 0.0
        assert document["battery"] is None
        vehicle = _vehicle(name, 0.0)
        assert vehicle.actuator.motor_time_constant_s == tau_s
        assert vehicle.drone.drag_coefficient == 0.0
    assert control_tolerance.CONTROL_DT_S == DT
    assert control_tolerance.SETPOINT_SAMPLE_S == 2 * DT


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_hand_reference_numbers(name):
    tau_s = PROFILES[name]
    reference = _hand_reference(tau_s)
    for key in _REFERENCE_KEYS:
        assert key in reference, f"missing reference key {key}"
    # Values worked out by hand and by two separate derivations.
    expected = {
        "esp_fly_class": (1.822781, 0.6160097, 0.3379505, 2.166616),
        "crazyflie_class": (1.822831, 0.6161268, 0.3380055, 2.167112),
    }[name]
    assert abs(reference["w_star"] - expected[0]) <= 1e-6
    assert abs(reference["phase_margin_rad"] - expected[1]) <= 1e-6
    assert abs(reference["d_star_continuous_s"] - expected[2]) <= 1e-6
    assert abs(reference["slope_per_s2"] - expected[3]) <= 1e-5
    assert reference["sampling_offset_s"] == DT / 2.0
    assert abs(reference["d_boundary_sim_s"] - (expected[2] - 0.005)) <= 1e-6
    # s = j w* with d = d* really solves the characteristic equation.
    assert reference["characteristic_residual"] < 1e-12
    # The attitude loop on its own is stable, and so is the delay-free position loop.
    assert np.max(np.real(np.roots([tau_s, 1.0, K_D_ATT, K_P_ATT]))) < 0.0
    closed = np.polyadd(np.polymul([1.0, 0.0, 0.0], [tau_s, 1.0, K_D_ATT, K_P_ATT]),
                        K_P_ATT * np.array([K_D_XY, K_P_XY]))
    assert np.max(np.real(np.roots(closed))) < 0.0
    assert _grid_delays(reference["d_boundary_sim_s"]) == (0.33, 0.34)


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_small_step_decays_below_and_grows_above(name):
    reference = _hand_reference(PROFILES[name])
    lo_s, hi_s = _grid_delays(reference["d_boundary_sim_s"])
    sigma_lo, _ = _growth_rate(name, lo_s)
    sigma_hi, _ = _growth_rate(name, hi_s)
    assert sigma_lo < 0.0 < sigma_hi


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_linear_boundary_matches_hand(name):
    reference = _hand_reference(PROFILES[name])
    lo_s, hi_s = _grid_delays(reference["d_boundary_sim_s"])
    boundary, _, _, omega_lo, omega_hi = _interpolated_boundary(name, lo_s, hi_s)
    w_star = reference["w_star"]
    assert abs(boundary - reference["d_boundary_sim_s"]) <= TOL_BOUNDARY_S
    assert abs(omega_lo - w_star) / w_star <= TOL_OMEGA_REL
    assert abs(omega_hi - w_star) / w_star <= TOL_OMEGA_REL


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_growth_slope_matches_hand(name):
    reference = _hand_reference(PROFILES[name])
    lo_s, hi_s = _grid_delays(reference["d_boundary_sim_s"])
    _, sigma_lo, sigma_hi, _, _ = _interpolated_boundary(name, lo_s, hi_s)
    slope = reference["slope_per_s2"]
    assert abs((sigma_hi - sigma_lo) / (hi_s - lo_s) - slope) / abs(slope) <= TOL_SLOPE_REL


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_find_d_crit_inside_hand_interval(name):
    lo, hi = _hand_reference(PROFILES[name])["dcrit_interval_s"]
    result = _dcrit(name)
    assert lo <= result["d_crit_s"] <= hi
    assert result["stable_delay_s"] == round(result["d_crit_s"] - 0.01, 10)


def test_planted_one_tick_shorter_delay_is_caught():
    _assert_bug_is_caught(controller_factory=OneTickShorterDelay)


def test_planted_every_tick_feedback_is_caught():
    _assert_bug_is_caught(controller_factory=EveryTickOffers)


def test_planted_low_attitude_damping_is_caught():
    _assert_bug_is_caught(inner_factory=_inner_with_low_rate_gain)


def test_planted_removed_motor_lag_is_caught():
    _assert_bug_is_caught(motor_tau_s=1e-4)


def test_planted_short_delay_moves_find_d_crit_out_of_interval(monkeypatch):
    lo, hi = DCRIT_INTERVAL_S
    assert lo <= _dcrit(BUG_PROFILE)["d_crit_s"] <= hi
    monkeypatch.setattr(control_tolerance, "FeedbackDelayController", OneTickShorterDelay)
    result = find_d_crit(_profile_path(BUG_PROFILE), arm="b", duration_s=20.0)
    assert not lo <= result["d_crit_s"] <= hi
