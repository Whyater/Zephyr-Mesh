"""Checks for tools/sens_tracker_tick_explain.py.

Which checks are independent of the code under test:
- the fast filter against the numeric tool's 6x6 filter (a separate
  implementation that is itself bit for bit with sim/tracker.py),
- the time-shift identity (a property of the math, not of the code),
- the matched-truth generator against the closed-form variances of a
  white-noise-acceleration path (q T for velocity, q T^3 / 3 for position).
The lag value test is a hand calculation.
"""
import numpy as np
import pytest

from tools import sens_tracker_numeric as numeric
from tools import sens_tracker_tick_explain as tx


@pytest.mark.parametrize("scenario", ["straight", "turn"])
def test_fast_filter_reproduces_numeric_tool_traces(scenario):
    seed = numeric.seed_list(1)[0]
    r = tx.x0_check(scenario, seed, 20.0)
    assert r["fast_exact_vs_numeric_f64_exact_m"] < 1e-9
    assert r["fast_next_vs_numeric_tick_next_f64_m"] < 1e-9
    # float32 vs float64: rounding sized, far under a millimetre
    assert r["fast_next_vs_numeric_f32_tick_next_m"] < 1e-3


def test_next_tick_lag_hand_values():
    # capture at 0.1043 s -> next tick 0.11 s, 5.7 ms late; 0.1013 -> 8.7 ms; 0.1073 -> 2.7 ms
    lags = tx.next_tick_times(np.array([0.1043, 0.1013, 0.1073])) - np.array([0.1043, 0.1013, 0.1073])
    assert np.allclose(lags, [0.0057, 0.0087, 0.0027], atol=1e-12)
    assert tx.MEAN_LAG_S == pytest.approx(0.0057)


def test_constant_relabel_is_a_pure_time_shift():
    # Fusing every measurement L later gives, once the start-up transient is
    # gone, the exact filter's estimate from L earlier: est_L(t) = est(t - L).
    seed = numeric.seed_list(1)[0]
    times, z = numeric.make_measurements("straight", 0.05, seed, 40.0)
    x0 = np.zeros(3)
    lag = 0.0057
    a = tx.fast_filter(times, z, q=0.1, r_var=0.0025, x0=x0)
    b = tx.fast_filter(times + lag, z, q=0.1, r_var=0.0025, x0=x0)
    g = np.arange(30.0, 39.0, 0.001)
    assert np.max(np.abs(tx.estimates_at(g, *b) - tx.estimates_at(g - lag, *a))) < 1e-6


def test_matched_truth_has_white_noise_acceleration_variances():
    # Velocity is a random walk: var v(T) = q T. Position integrates it:
    # var p(T) = q T^3 / 3. 600 independent paths x 3 axes, T = 4 s, q = 0.1.
    T, q = 4.0, 0.1
    t = np.linspace(0.05, T, 80)
    ends = np.array([tx.matched_truth(t, s, q_true=q, v0=(0.0, 0.0, 0.0))[-1] for s in range(600)]).ravel()
    var_p = np.var(ends)
    assert var_p == pytest.approx(q * T**3 / 3.0, rel=0.12)
    # velocity from a final small step: finite difference over the last 1e-6 s
    t2 = np.r_[t, T + 1e-6]
    vel = np.array([(lambda p: (p[-1] - p[-2]) / 1e-6)(tx.matched_truth(t2, s, q_true=q, v0=(0.0, 0.0, 0.0)))
                    for s in range(600)]).ravel()
    assert np.var(vel) == pytest.approx(q * T, rel=0.12)


def test_run_case_smoke():
    rows = tx.run_case("matched", numeric.seed_list(1)[0], 6.0, variants=("exact", "next"))
    assert {(r["variant"], r["scoring"]) for r in rows} == {(v, s) for v in ("exact", "next") for s in ("tick", "ms")}
    assert all(np.isfinite(r["p95_m"]) and r["p95_m"] > 0 for r in rows)
