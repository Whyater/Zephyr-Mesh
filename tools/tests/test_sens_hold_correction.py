"""Tests for tools/sens_hold_correction.py.

The packet-age test is the independent one: it drives the real period-aware
controller with a fake drone whose x position is the clock, so the x the
controller sees tells exactly how old its packet is.  The expected ages are
worked out by hand (see each test), not taken from the tool.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.physics import control_tolerance as ct
from tools import sens_hold_correction as shc

PROFILE = "sim/physics/profiles/esp_fly_class.json"


class _ClockDrone:
    """Fake drone: x position equals the current time, everything else fixed."""

    mass = 0.03
    I_x = I_y = I_z = 1e-5

    def __init__(self):
        self.t = 0.0
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.angular_velocity = np.zeros(3)

    def set_time(self, t):
        self.t = t
        self.position = np.array([t, 0.0, 1.0])


class _RecordingInner:
    def __init__(self):
        self.seen_x = []

    def update_control(self, drone, target_position, dt, **kwargs):
        self.seen_x.append(float(drone.position[0]))


def _ages(period_s, delay_s, n_ticks=400):
    drone = _ClockDrone()
    inner = _RecordingInner()
    ctrl = shc.PeriodFeedbackDelayController(delay_s, 0.0, np.random.default_rng(0), period_s=period_s,
                                             initial_state=(np.full(3, np.nan), np.zeros(3)), inner=inner)
    times = []
    for k in range(n_ticks):
        t = k * 0.01
        drone.set_time(t)
        times.append(t)
        ctrl.update_control(drone, np.zeros(3), 0.01)
    ages = np.array(times) - np.array(inner.seen_x)
    return ages[np.isfinite(ages)]


def test_ticks_per_packet():
    assert shc.ticks_per_packet(0.01) == 1
    assert shc.ticks_per_packet(0.02) == 2
    assert shc.ticks_per_packet(0.1) == 10
    with pytest.raises(ValueError):
        shc.ticks_per_packet(0.015)
    with pytest.raises(ValueError):
        shc.ticks_per_packet(0.0)


def test_packet_age_cycles_like_a_zero_order_hold():
    # Packet every 50 ms, delay 100 ms, read every 10 ms: the packet in use is
    # 100, 110, 120, 130, 140 ms old, then a new one lands.  Mean 120 ms, which is
    # 100 + (50 - 10) / 2.
    ages = _ages(0.05, 0.10)
    assert set(np.round(ages, 6)) == {0.10, 0.11, 0.12, 0.13, 0.14}
    assert ages[:200].mean() == pytest.approx(0.12, abs=1e-9)


def test_packet_age_at_100_hz_has_no_hold():
    # Packet every tick: always exactly the nominal delay old.
    ages = _ages(0.01, 0.07)
    assert np.allclose(ages, 0.07)


def test_patched_controller_matches_stock_at_20ms():
    stock = ct.run_episode(PROFILE, "b", "step", 0.2, 0.0, 0, duration_s=3.0)
    patched = shc.run_at_period(PROFILE, "b", 0.2, 0.02, duration_s=3.0)
    assert stock == patched


def test_period_changes_the_flight():
    slow = shc.run_at_period(PROFILE, "b", 0.2, 0.1, duration_s=3.0)
    fast = shc.run_at_period(PROFILE, "b", 0.2, 0.02, duration_s=3.0)
    assert slow["rms_error_m"] != fast["rms_error_m"]


def test_stream_period_restores_the_factory_after_an_error():
    original = ct._controller_for
    with pytest.raises(RuntimeError), shc.stream_period(0.05):
        assert ct._controller_for is not original
        raise RuntimeError("boom")
    assert ct._controller_for is original


def test_setpoint_controller_sends_on_its_period():
    ctrl = shc.PeriodSetpointDelayController(0.05, 0.0, np.random.default_rng(0), period_s=0.04,
                                             inner=_RecordingInner())
    drone = _ClockDrone()
    for _ in range(100):
        ctrl.update_control(drone, np.zeros(3), 0.01)
    # 100 ticks, one packet every 4 ticks starting at tick 0: 25 packets.
    assert ctrl.channel.offered_count == 25


def test_summarize_scan_flags_a_stable_island():
    verdicts = {0.30: False, 0.31: True, 0.32: False, 0.33: True}
    s = shc.summarize_scan(verdicts)
    assert s["d_crit_s"] == 0.31
    assert s["last_stable_s"] == 0.30
    assert s["monotone"] is False
    assert s["stable_above_first_unstable_s"] == [0.32]
    clean = shc.summarize_scan({0.30: False, 0.31: True, 0.32: True})
    assert clean["monotone"] is True
    assert shc.summarize_scan({0.30: False})["d_crit_s"] is None


def test_fit_hold_recovers_a_known_line():
    # Made-up sim edges on the line d = 0.343 - 0.5 P (hand edge 0.338 s), so the
    # hold is 0.5 P - 5 ms.
    p = [0.01, 0.02, 0.05, 0.1]
    d = [0.343 - 0.5 * x for x in p]
    fit = shc.fit_hold(p, d, 0.338)
    assert fit["hold_slope"] == pytest.approx(0.5)
    assert fit["hold_intercept_s"] == pytest.approx(-0.005)
    assert fit["max_abs_residual_s"] == pytest.approx(0.0, abs=1e-12)


def test_formula_residuals_pick_the_matching_formula():
    p = np.array([0.02, 0.04, 0.06])
    d = 0.338 - (p - 0.01) / 2
    res = shc.formula_residuals(p, d, 0.338)
    assert res["p_minus_c_over_2"]["max_abs_s"] == pytest.approx(0.0, abs=1e-12)
    # The current file's hold is 10 ms bigger, so the sim sits 10 ms above it.
    assert res["main_P_over_2_plus_c_over_2"]["mean_s"] == pytest.approx(0.01)


def test_window_growth_rate_reads_a_known_exponential():
    # A 1.8 rad/s wobble whose size grows as exp(0.02 t): the 10 s window peaks grow
    # by the same factor, so the fitted rate is 0.02 per second.
    t = np.arange(0, 80, 0.01)
    e = 0.01 * np.exp(0.02 * t) * np.abs(np.sin(1.8 * t))
    assert shc.window_growth_rate_per_s(e) == pytest.approx(0.02, abs=1e-3)
    with pytest.raises(ValueError):
        shc.window_growth_rate_per_s(e[:3000])


def test_zero_crossing_interpolates_between_the_bracketing_delays():
    # Rates -0.02 at 0.33 s and +0.02 at 0.34 s: halfway, 0.335 s.
    edge = shc.zero_crossing_edge([0.32, 0.33, 0.34], [-0.05, -0.02, 0.02])
    assert edge["edge_s"] == pytest.approx(0.335)
    assert edge["n_sign_changes"] == 1
    assert shc.zero_crossing_edge([0.32, 0.33], [-0.05, -0.02])["edge_s"] is None


def test_small_step_run_matches_the_long_check_at_20ms(monkeypatch):
    # The long check flies the stock controller; at the stock 20 ms period the
    # small-step loop here must give the same window peaks.
    from tools import a_make_dcrit_long_check as lc

    monkeypatch.setattr(lc, "DURATION_S", 30.0)
    stock = lc.fly(("esp-fly-class", "b", 0.33, 0.01))["window_peak_error_m"]
    errors = shc.small_step_errors(PROFILE, 0.33, 0.02, duration_s=30.0)
    ours = [float(np.max(errors[w * 1000:(w + 1) * 1000])) for w in range(3)]
    assert ours == stock


def test_small_step_run_differs_at_another_period():
    a = shc.small_step_errors(PROFILE, 0.1, 0.05, duration_s=3.0)
    b = shc.small_step_errors(PROFILE, 0.1, 0.02, duration_s=3.0)
    assert a.shape == b.shape == (300,)
    assert not np.array_equal(a, b)
