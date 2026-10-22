"""Tests for the D-088 "failed to fly" rule (opt-in, for future runs).

The hand-built trajectories below are written directly with NumPy so they do
not depend on the simulator. Each one flies badly in a way the EXP-021
pre-registered rules miss, and the stricter rule must catch it.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sim.physics.control_tolerance import (
    EXP021_RULES,
    FAILED_TO_FLY_RULES,
    classify_trajectory,
    find_d_crit,
    run_episode,
)

ESP = "sim/physics/profiles/esp_fly_class.json"
CF = "sim/physics/profiles/crazyflie_class.json"

DT = 0.01
N = 2000  # 20 s, like an EXP-021 step run


def _step_flight():
    """A perfect 1 m step along x at t = 1 s, hovering at z = 1 m."""
    t = DT * np.arange(1, N + 1)
    ref = np.zeros((N, 3))
    ref[:, 0] = np.where(t < 1.0, 0.0, 1.0)
    ref[:, 2] = 1.0
    pos = ref.copy()
    tilt = np.full(N, 5.0)
    return t, pos, ref, tilt


def _both(t, pos, ref, tilt):
    old = classify_trajectory(t, pos, ref, tilt, rule_set=EXP021_RULES)
    new = classify_trajectory(t, pos, ref, tilt, rule_set=FAILED_TO_FLY_RULES)
    return old, new


def test_perfect_flight_passes_both_rules():
    old, new = _both(*_step_flight())
    assert old == {"failed": False, "reason": None, "time_s": None}
    assert new == {"failed": False, "reason": None, "time_s": None}


def test_tip_past_45_degrees_fails_only_new_rule():
    t, pos, ref, tilt = _step_flight()
    # Tilt ramps to 60 degrees between 5 and 6 s, then recovers. Position error stays zero.
    ramp = (t >= 5.0) & (t < 6.0)
    tilt[ramp] = 5.0 + 55.0 * (t[ramp] - 5.0)
    old, new = _both(t, pos, ref, tilt)
    assert old["failed"] is False
    assert new["failed"] is True
    assert new["reason"] == "tilt_above_45_deg"
    # First sample above 45 degrees: 5 + 55 (t - 5) > 45 means t > 5.7273 s, so t = 5.73 s.
    assert new["time_s"] == pytest.approx(5.73)


def test_falling_below_zero_height_fails_only_new_rule():
    t, pos, ref, tilt = _step_flight()
    # A 1.4 m drop in the middle third (to z = -0.4 m), then a full recovery.
    dip = (t >= 8.0) & (t < 10.0)
    pos[dip, 2] = 1.0 - 1.4 * np.sin(math.pi * (t[dip] - 8.0) / 2.0)
    old, new = _both(t, pos, ref, tilt)
    assert old["failed"] is False  # error peaks at 1.4 m, under 5 m; x is flat
    assert new["failed"] is True
    assert new["reason"] == "height_below_0_m"
    # 1 - 1.4 sin(pi (t - 8) / 2) < 0 first when sin > 1/1.4: t > 8 + 2 asin(1/1.4) / pi = 8.5063 s.
    assert new["time_s"] == pytest.approx(8.51)


@pytest.mark.parametrize("axis", [1, 2])
def test_growing_y_or_z_error_with_flat_x_fails_only_new_rule(axis):
    t, pos, ref, tilt = _step_flight()
    # A 1 Hz wobble whose amplitude grows from 0.05 m to 0.45 m over 20 s.
    amp = 0.05 + 0.02 * t
    pos[:, axis] += amp * np.sin(2 * math.pi * t)
    old, new = _both(t, pos, ref, tilt)
    assert old["failed"] is False  # x error is exactly zero
    assert new["failed"] is True
    assert new["reason"] == "growing_error"
    assert new["time_s"] == pytest.approx(20.0)


def test_boundaries_are_strict():
    t, pos, ref, tilt = _step_flight()
    tilt[500] = 45.0  # exactly 45 is not "above"
    pos[600, 2] = 0.0  # exactly 0 m is not "below"
    pos[600, 0] = ref[600, 0]
    _, new = _both(t, pos, ref, tilt)
    assert new["failed"] is False


def test_small_growth_under_the_5_cm_floor_does_not_count():
    t, pos, ref, tilt = _step_flight()
    # Grows from 0.005 m to 0.04 m: more than 1.2 times, but never above 0.05 m.
    pos[:, 1] += (0.005 + 0.00175 * t) * np.sin(2 * math.pi * t)
    _, new = _both(t, pos, ref, tilt)
    assert new["failed"] is False


def test_earliest_trigger_wins():
    t, pos, ref, tilt = _step_flight()
    tilt[t >= 3.0] = 50.0
    pos[t >= 2.0, 2] = -0.1
    _, new = _both(t, pos, ref, tilt)
    assert new["reason"] == "height_below_0_m"
    assert new["time_s"] == pytest.approx(2.0)


def test_old_rule_failures_are_also_new_rule_failures():
    t, pos, ref, tilt = _step_flight()
    tilt[t >= 4.0] = 85.0
    old, new = _both(t, pos, ref, tilt)
    assert old["reason"] == "tilt_above_80_deg"
    assert new["reason"] == "tilt_above_45_deg"
    assert new["time_s"] == pytest.approx(4.0)

    t, pos, ref, tilt = _step_flight()
    pos[:, 0] += (0.05 + 0.02 * t) * np.sin(2 * math.pi * t)
    old, new = _both(t, pos, ref, tilt)
    assert old["reason"] == "growing_oscillation"
    assert new["reason"] == "growing_error"

    t, pos, ref, tilt = _step_flight()
    pos[t >= 7.0, 1] = 6.0
    old, new = _both(t, pos, ref, tilt)
    assert old["reason"] == new["reason"] == "max_error_above_5_m"


def test_non_finite_samples_fail_both_rules():
    t, pos, ref, tilt = _step_flight()
    tilt[300] = math.nan
    old, new = _both(t, pos, ref, tilt)
    assert old["reason"] == new["reason"] == "non_finite_state"
    assert new["time_s"] == pytest.approx(t[300])


def test_new_rule_is_never_looser_on_random_flights():
    """Property check: whenever the EXP-021 rules fail a flight, so does the new rule."""
    rng = np.random.default_rng(88)
    checked_old_failures = 0
    for _ in range(300):
        t, pos, ref, tilt = _step_flight()
        n = int(rng.integers(6, N + 1))
        t, pos, ref, tilt = t[:n], pos[:n], ref[:n], tilt[:n]
        growth = rng.uniform(-0.03, 0.08, size=3)
        base = rng.uniform(0.0, 0.3, size=3)
        freq = rng.uniform(0.2, 2.0, size=3)
        for axis in range(3):
            amp = np.clip(base[axis] + growth[axis] * t, 0.0, None)
            pos[:, axis] += amp * np.sin(2 * math.pi * freq[axis] * t + rng.uniform(0, 6))
        tilt = rng.uniform(0.0, 1.0) * 90.0 * np.abs(np.sin(rng.uniform(0.05, 0.5) * t))
        old, new = _both(t, pos, ref, tilt)
        if old["failed"]:
            checked_old_failures += 1
            assert new["failed"], (old, new)
            assert new["time_s"] <= old["time_s"] + 1e-12
    assert checked_old_failures > 20  # make sure the property was actually exercised


def test_classifier_rejects_bad_input():
    t, pos, ref, tilt = _step_flight()
    with pytest.raises(ValueError):
        classify_trajectory(t, pos, ref, tilt, rule_set="something_else")
    with pytest.raises(ValueError):
        classify_trajectory(t, pos[:-1], ref, tilt, rule_set=FAILED_TO_FLY_RULES)
    with pytest.raises(ValueError):
        classify_trajectory(t, pos[:, :2], ref[:, :2], tilt, rule_set=FAILED_TO_FLY_RULES)
    bad_t = t.copy()
    bad_t[10] = math.nan
    with pytest.raises(ValueError):
        classify_trajectory(bad_t, pos, ref, tilt, rule_set=FAILED_TO_FLY_RULES)
    with pytest.raises(ValueError):
        classify_trajectory(t[::-1], pos, ref, tilt, rule_set=FAILED_TO_FLY_RULES)
    with pytest.raises(TypeError):
        classify_trajectory(t, pos, ref, tilt)  # the rule set must be named


# --- Simulator integration (short, targeted runs) ---------------------------------

FTF_KEYS = {"failed_to_fly", "failed_to_fly_reason", "failed_to_fly_time_s", "failed_to_fly_rule_set"}


def test_default_run_episode_output_is_unchanged():
    default = run_episode(CF, "b", "step", 0.1, 0.3, 4, duration_s=3.0)
    explicit = run_episode(CF, "b", "step", 0.1, 0.3, 4, duration_s=3.0, failed_to_fly_rule=False)
    # JSON text compares NaN fields too (NaN != NaN as floats).
    assert json.dumps(default, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    assert not FTF_KEYS & set(default)


def test_opt_in_adds_a_verdict_and_keeps_every_old_field():
    default = run_episode(CF, "b", "step", 0.1, 0.3, 4, duration_s=3.0)
    opted = run_episode(CF, "b", "step", 0.1, 0.3, 4, duration_s=3.0, failed_to_fly_rule=True)
    assert set(opted) - set(default) == FTF_KEYS
    kept = {k: opted[k] for k in default}
    assert json.dumps(kept, sort_keys=True) == json.dumps(default, sort_keys=True)
    assert opted["failed_to_fly"] is False
    assert opted["failed_to_fly_rule_set"] == FAILED_TO_FLY_RULES


def test_esp_fly_slice_c_30_ms_is_stable_by_old_rules_but_failed_to_fly():
    """The D-087 case: 68 degree wobble, 3 m below the (unmodeled) floor, still 'stable'."""
    r = run_episode(ESP, "c", "step", 0.03, 0.0, 0, failed_to_fly_rule=True)
    assert r["unstable"] is False
    assert r["failed_to_fly"] is True
    assert r["failed_to_fly_reason"] in {"tilt_above_45_deg", "height_below_0_m"}
    assert r["whole_run_max_tilt_deg"] > 45.0 or r["whole_run_min_z_m"] < 0.0


def test_find_d_crit_default_is_unchanged_and_opt_in_uses_new_verdict():
    default = find_d_crit(ESP, arm="c", bracket=(0.0, 0.08))
    assert "rule_set" not in default
    assert default["stable_delay_s"] == pytest.approx(0.03)
    strict = find_d_crit(ESP, arm="c", bracket=(0.0, 0.08), failed_to_fly_rule=True)
    assert strict["rule_set"] == FAILED_TO_FLY_RULES
    # The stricter rule can only move the limit down, never up.
    assert strict["unstable_delay_s"] <= default["unstable_delay_s"]
    assert strict["stable_side"]["failed_to_fly"] is False
    assert strict["unstable_side"]["failed_to_fly"] is True
