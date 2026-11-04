"""Checks for tools/pilot_full_chain.py.

The first four tests compare the loop with hand values worked out from the PD gains
(Kp = 2, Kd = 1.5 on the horizontal axes, typed here, not read from the code), so they are
independent checks. test_w1_tracker_matches_s6 only shows the loop reuses the S6 packets and
tracker rule; it restates the implementation and proves nothing about physics.
"""
from __future__ import annotations

import numpy as np
import pytest

from sim.s6_recovery import RecoveryConfig, deliver, run_arms, seed_list
from tools import pilot_full_chain as pilot

KP, KD = 2.0, 1.5  # Crazyflie-class v2 profile position loop, x and y axes
SEED = seed_list(1)[0]
GOLDEN_W1_S6_PASS_TURN_STEADY_P95_M = 1.3957818145058185  # old w1 row, s6_pass turn, first S6 seed


def _settled(trace, after_s=8.0):
    """Follow error once the start transient is gone.

    The PD poles s^2 + 1.5 s + 2 = 0 have real part -0.75, a 1.33 s time constant; the carrot also
    has to close the 0.75 m lead jump at 1.5 - 1.0 = 0.5 m/s. A 0.75 m error left at 2 s shrinks to
    0.75 e^(-6 / 1.33) = 0.008 m by 8 s.
    """
    return trace["follow_m"][trace["times_s"] >= after_s]


def test_zero_cell_straight_lead_cancels_pd_lag():
    # Hand value: a PD loop with velocity damping lags a ramp by v Kd / Kp = 1.0 x 0.75 = 0.75 m.
    _, no_lead = pilot.run_loop("constant_velocity", pilot.CELLS["zero"], lead_s=0.0, seed=SEED, keep_trace=True)
    assert np.median(_settled(no_lead)) == pytest.approx(1.0 * KD / KP, abs=0.05)
    # With the goal 0.75 s ahead the steady lag cancels, from the true state or from the tracker.
    for source in ("truth", "tracker"):
        _, lead = pilot.run_loop("constant_velocity", pilot.CELLS["zero"], lead_s=KD / KP, seed=SEED,
                                 keep_trace=True, goal_source=source)
        assert np.max(_settled(lead)) < 0.03


def test_zero_cell_turn_matches_linear_hand_value():
    # Linear PD loop, attitude ignored: drone / goal = Kp / (Kp - w^2 + j Kd w) at w = v / r = 1 rad/s.
    # A CV goal 0.75 s ahead on the tangent is r (1 + 0.75 j); the target is r. Steady error:
    w = 1.0
    gain = KP / (KP - w**2 + 1j * KD * w)
    hand = abs(gain * (1 + 0.75j * w) - 1)
    assert hand == pytest.approx(0.553, abs=0.002)
    # Goal from the true state and the 1.5 m/s carrot cap lifted, so only the PD loop (and the
    # attitude lag the hand value ignores) is left.
    result, trace = pilot.run_loop("constant_turn", pilot.CELLS["zero"], seed=SEED, keep_trace=True,
                                   goal_source="truth", max_speed_mps=5.0)
    assert np.median(_settled(trace)) == pytest.approx(hand, abs=0.10)
    assert result.failed  # the controller alone breaks the 0.25 m rule on this turn


def test_setpoint_delay_adds_v_times_delay_and_lead_recovers_it():
    # No noise, no loss: a goal that takes 0.2 s to arrive is 0.2 s old, so the drone trails by
    # v x 0.2 = 0.2 m. Leading by the same 0.2 s puts it back on the line.
    cell = pilot.Cell("delay_only", radio_delay_s=0.2)
    _, late = pilot.run_loop("constant_velocity", cell, wiring="w2", seed=SEED, keep_trace=True)
    assert np.median(_settled(late)) == pytest.approx(0.2, abs=0.03)
    result, comp = pilot.run_loop("constant_velocity", cell, wiring="w2", lead_s=KD / KP + 0.2, seed=SEED,
                                  keep_trace=True)
    assert np.max(_settled(comp)) < 0.03
    assert result.non_nominal_fraction == 0.0  # a steady delay never trips the link-age failsafe


def test_long_setpoint_silence_trips_the_hold():
    # Every setpoint lost: the link age passes 0.15 s (hold), then 0.30 s, and the drone lands.
    cell = pilot.Cell("dead_link", loss_probability=1.0)
    result = pilot.run_loop("constant_velocity", cell, wiring="w2", seed=SEED)
    assert result.landed and result.failed


def test_w1_tracker_matches_s6():
    # Restates the implementation: the loop's ground estimate is the S6 recovery estimate.
    config = RecoveryConfig()
    cell = pilot.CELLS["s6_fail"]
    _, trace = pilot.run_loop("constant_turn", cell, wiring="w1", seed=SEED, keep_trace=True)
    packets = deliver("constant_turn", delay_s=cell.total_delay_s, loss_probability=cell.loss_probability,
                      noise_std_m=cell.noise_std_m, seed=SEED, config=config)
    s6 = run_arms(packets, "constant_turn", config)["recovery"]
    n = trace["estimate_m"].size
    np.testing.assert_allclose(trace["estimate_m"], s6.errors_m[:n], rtol=0, atol=1e-12)
    np.testing.assert_allclose(trace["estimate_times_s"], s6.query_times_s[:n], rtol=0, atol=1e-12)


def test_refuses_nonempty_output(tmp_path):
    (tmp_path / "old.txt").write_text("x")
    with pytest.raises(SystemExit):
        pilot.run_pilot(tmp_path, n_seeds=1)



def _fake_rows(cell, motion, seeds):
    rows = []
    for seed in seeds:
        for variant in ("tracker_only",) + pilot.VARIANTS:
            rows.append({"cell": cell, "motion": motion, "variant": variant, "seed": seed, "p95_m": 0.1,
                         "max_m": 0.2, "median_m": 0.05, "p95_steady_m": 0.1, "failed_steady": False,
                         "estimate_p95_m": 0.1, "min_separation_m": 1.4, "non_nominal_fraction": 0.0,
                         "landed": False, "failed": False})
    return rows


def test_resume_reads_the_checkpoint_and_runs_nothing_new(tmp_path, monkeypatch):
    import json

    def no_runs(*args, **kwargs):
        raise AssertionError("a finished group was run again")

    monkeypatch.setattr(pilot, "run_loop", no_runs)
    monkeypatch.setattr(pilot, "tracker_only", no_runs)
    seeds = seed_list(1)
    output = tmp_path / "out"
    partial = tmp_path / "out.partial"
    partial.mkdir()
    with open(partial / pilot.CHECKPOINT, "w") as handle:
        for cell in pilot.CELLS:
            for motion in ("constant_velocity", "constant_turn"):
                handle.write(json.dumps({"cell": cell, "motion": motion, "seeds": seeds,
                                         "rows": _fake_rows(cell, motion, seeds)}) + "\n")
    summary = pilot.run_pilot(output, n_seeds=1)
    assert len(summary["cells_summary"]) == len(pilot.CELLS) * 2 * (1 + len(pilot.VARIANTS))
    assert (output / "per_run.csv").exists() and (output / "summary.json").exists()
    assert not partial.exists() and not (output / pilot.CHECKPOINT).exists()


def test_resume_refuses_a_checkpoint_with_other_seeds(tmp_path):
    import json

    partial = tmp_path / "out.partial"
    partial.mkdir()
    (partial / pilot.CHECKPOINT).write_text(json.dumps({"cell": "zero", "motion": "constant_velocity",
                                                        "seeds": [1, 2], "rows": []}) + "\n")
    with pytest.raises(SystemExit):
        pilot.run_pilot(tmp_path / "out", n_seeds=1)


def test_replot_reads_per_run_csv_back_to_the_same_summary(tmp_path):
    import csv

    rows = [row for cell in pilot.CELLS for motion in ("constant_velocity", "constant_turn")
            for row in _fake_rows(cell, motion, seed_list(2))]
    rows[0].update({"median_m": None, "min_separation_m": None, "non_nominal_fraction": None, "landed": None})
    path = tmp_path / "per_run.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert pilot.read_rows(path) == rows
    assert pilot.summarize(pilot.read_rows(path)) == pilot.summarize(rows)


# Lead goal rule. Hand values first, then the bands fixed before the run.

def test_turn_predict_matches_circle_geometry():
    # Start (1, 0) heading +y at 1 m/s, turning left at 1 rad/s: that is the unit circle round the
    # origin, so a quarter turn lands on (0, 1). The vertical speed just carries on.
    np.testing.assert_allclose(pilot.turn_predict((1.0, 0.0, 0.0), (0.0, 1.0, 0.5), 1.0, np.pi / 2),
                               (0.0, 1.0, 0.25 * np.pi), atol=1e-12)
    # Half a turn from the origin heading +x: the far side of a 1 m circle, 2 m to the left.
    np.testing.assert_allclose(pilot.turn_predict((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0, np.pi), (0.0, 2.0, 0.0),
                               atol=1e-12)
    # No turn: a straight line.
    np.testing.assert_allclose(pilot.turn_predict((1.0, 2.0, 3.0), (0.5, 0.0, 0.0), 0.0, 2.0), (2.0, 2.0, 3.0))


def test_estimate_turn_rate_by_hand():
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    # Heading 0 at t = 0 and 45 degrees at t = 0.5 s: (pi / 4) / 0.5 = pi / 2 rad/s to the left.
    assert pilot.estimate_turn_rate([(0.0, 1.0, 0.0), (0.25, 1.0, 0.4), (0.5, c, s)]) == pytest.approx(np.pi / 2)
    # Across the +-180 degree seam: 179 to -179 degrees is 2 degrees left, not 358 right.
    a, b = np.radians(179.0), np.radians(-179.0)
    rate = pilot.estimate_turn_rate([(0.0, np.cos(a), np.sin(a)), (0.5, np.cos(b), np.sin(b))])
    assert rate == pytest.approx(np.radians(2.0) / 0.5)
    # 162 degrees in 0.5 s is 5.7 rad/s: clipped to 2. Too slow to have a heading: no turn.
    g = 0.9 * np.pi
    assert pilot.estimate_turn_rate([(0.0, 1.0, 0.0), (0.5, np.cos(g), np.sin(g))]) == pilot.TURN_RATE_MAX_RADPS
    assert pilot.estimate_turn_rate([(0.0, 0.1, 0.0), (0.5, 0.0, 0.1)]) == 0.0
    assert pilot.estimate_turn_rate([(0.5, 1.0, 0.0)]) == 0.0


def _circle_trace(lead_s):
    """True-state lead rule on the zero-cell turn, carrot cap lifted; settled samples (t >= 8 s)."""
    _, trace = pilot.run_loop("constant_turn", pilot.CELLS["zero"], seed=SEED, keep_trace=True, goal_source="truth",
                              goal_rule="lead", lead_s=lead_s, max_speed_mps=5.0)
    settled = trace["times_s"] >= 8.0
    # Circle centre (0, 1) plus the (0, -1) standoff: the drone circles the origin in x and y.
    xy = trace["position_m"][settled, :2]
    return np.hypot(xy[:, 0], xy[:, 1]), trace["follow_m"][settled]


@pytest.mark.xfail(strict=True, reason="registered band failed: the sim radius is 1.203 m, not 1.109 m. The "
                   "attitude loop sits inside the position loop, so it stretches the circle as well as delaying it")
def test_pd_follower_on_circle_matches_one_loop_hand_value():
    # Registered before running. A PD follower aimed straight at a point on a 1 m circle at
    # 1 rad/s. Hand, position loop only: drone / goal = Kp / (Kp - w^2 + j Kd w) = 2 / (1 + 1.5 j):
    # radius 1.109 m, 56.3 degrees behind, error 1.00 m.
    gain = KP / (KP - 1.0 + 1j * KD)
    assert abs(gain) == pytest.approx(1.109, abs=0.001)
    assert np.degrees(np.angle(gain)) == pytest.approx(-56.3, abs=0.1)
    assert abs(gain - 1) == pytest.approx(1.000, abs=0.001)
    radius, follow = _circle_trace(0.0)  # lead 0 puts the goal on the target itself
    assert np.median(radius) == pytest.approx(abs(gain), abs=0.03)
    assert 1.00 <= np.median(follow) <= 1.15


def test_pd_follower_on_circle_matches_two_loop_hand_value():
    # Post hoc (written after the registered check above failed), but from gains, not fitted.
    # The attitude loop (K_p_att = 100, K_d_att = 18 per unit inertia) makes the tilt, and so the
    # sideways acceleration, follow its command through A = 100 / (100 - w^2 + 18 j w). At w = 1:
    # gain 0.994, 10.3 degrees late. The position loop closes around it:
    # x = Kp A g / (-w^2 + A (Kp + j Kd w)).
    att = 100.0 / (100.0 - 1.0 + 18.0j)
    gain = KP * att / (-1.0 + att * (KP + 1j * KD))
    assert abs(gain) == pytest.approx(1.203, abs=0.001)
    assert np.degrees(np.angle(gain)) == pytest.approx(-52.6, abs=0.1)
    assert abs(gain - 1) == pytest.approx(0.992, abs=0.001)
    radius, follow = _circle_trace(0.0)
    assert np.median(radius) == pytest.approx(abs(gain), abs=0.03)
    assert np.median(follow) == pytest.approx(abs(gain - 1), abs=0.05)
    # Same model with the goal 0.75 s ahead on the circle (the lead rule): error 0.274 m.
    radius, follow = _circle_trace(KD / KP)
    assert np.median(follow) == pytest.approx(abs(gain * np.exp(0.75j) - 1), abs=0.05)


def test_lead_rule_true_state_turn_matches_hand():
    # Goal 0.75 s ahead on the circle: drone = 1.109 at -56.3 + 43.0 degrees, error 0.268 m; a 0.1 s
    # attitude lag turns it into 0.366 m. Registered band [0.25, 0.42].
    gain = KP / (KP - 1.0 + 1j * KD)
    assert abs(gain * np.exp(1j * KD / KP) - 1) == pytest.approx(0.268, abs=0.001)
    _, trace = pilot.run_loop("constant_turn", pilot.CELLS["zero"], seed=SEED, keep_trace=True, goal_source="truth",
                              goal_rule="lead", lead_s=pilot.CONTROLLER_LAG_S, max_speed_mps=5.0)
    assert 0.25 <= np.median(_settled(trace)) <= 0.42


def test_lead_rule_straight_line_cancels_the_lag():
    # On a straight line the curve prediction is the straight prediction, so the lead rule must do
    # what the velocity rule did: the 0.75 s lead cancels the v Kd / Kp lag.
    assert pilot.CONTROLLER_LAG_S == KD / KP
    for source in ("truth", "tracker"):
        _, trace = pilot.run_loop("constant_velocity", pilot.CELLS["zero"], seed=SEED, keep_trace=True,
                                  goal_source=source, goal_rule="lead", lead_s=pilot.CONTROLLER_LAG_S)
        assert np.max(_settled(trace)) < 0.03


def test_lead_variants_lead_by_radio_delay_plus_controller_lag():
    cell = pilot.CELLS["s6_fail"]
    assert pilot.variant_settings("w2_lead", cell)["lead_s"] == pytest.approx(0.2 + 0.75)
    assert pilot.variant_settings("w2_truth_lead", cell)["lead_s"] == pytest.approx(0.95)
    assert pilot.variant_settings("w1_lead", cell)["lead_s"] == pytest.approx(0.75)  # delay is on the camera path
    with pytest.raises(ValueError):
        pilot.run_loop("constant_velocity", cell, seed=SEED, goal_rule="bogus")


def test_resume_refuses_a_checkpoint_with_other_variants(tmp_path):
    import json

    partial = tmp_path / "out.partial"
    partial.mkdir()
    (partial / pilot.CHECKPOINT).write_text(json.dumps({"cell": "zero", "motion": "constant_velocity",
                                                        "seeds": seed_list(1), "rows": []}) + "\n")
    with pytest.raises(SystemExit):
        pilot.run_pilot(tmp_path / "out", n_seeds=1, with_lead=True)


def test_delay_sweep_refuses_nonempty_output(tmp_path):
    (tmp_path / "old.txt").write_text("x")
    with pytest.raises(SystemExit):
        pilot.run_delay_sweep(tmp_path)


# Switching rule. The two limiting cases are the rule's definition checked end to end: with no
# camera noise the turn-rate estimate is exact, so on a 1 rad/s circle the latch turns on and
# stays on (switch = lead), and on a straight line it is exactly 0 (switch = the old goal).

def test_switch_on_a_constant_rate_circle_equals_the_lead_rule():
    cell = pilot.CELLS["zero"]
    lead = pilot.run_loop("constant_turn", cell, seed=SEED, **pilot.variant_settings("w1_lead", cell))
    switch, trace = pilot.run_loop("constant_turn", cell, seed=SEED, keep_trace=True,
                                   **pilot.variant_settings("w1_switch", cell))
    # The latch needs |w| >= 0.80 rad/s first, so it turns on a little later than the lead rule's
    # estimate starts; that start difference is still about 0.003 m at 6 s.
    assert switch.follow_p95_steady_m == pytest.approx(lead.follow_p95_steady_m, abs=0.01)
    assert all(on for t, _, on in trace["turn_trace"] if t >= 2.0)


def test_switch_on_a_straight_line_equals_the_default_rule():
    cell = pilot.CELLS["zero"]
    for switch_variant, old_variant in (("w1_switch", "w1"), ("w2_switch", "w2_comp")):
        old = pilot.run_loop("constant_velocity", cell, seed=SEED, **pilot.variant_settings(old_variant, cell))
        switch, trace = pilot.run_loop("constant_velocity", cell, seed=SEED, keep_trace=True,
                                       **pilot.variant_settings(switch_variant, cell))
        assert switch == old
        assert not any(on for _, _, on in trace["turn_trace"])


def test_switch_thresholds_must_leave_a_hysteresis_band():
    assert 0.0 < pilot.SWITCH_OFF_RADPS < pilot.SWITCH_ON_RADPS
    with pytest.raises(ValueError):
        pilot.run_loop("constant_velocity", pilot.CELLS["zero"], seed=SEED, goal_rule="switch",
                       switch_on_radps=0.5, switch_off_radps=0.5)


def test_default_noisy_cell_output_is_unchanged():
    # Golden value: the old velocity rule (w1) on the s6_pass turn, first S6 seed, from the code
    # before the switching rule existed. Restates the old output on purpose, to catch any change.
    cell = pilot.CELLS["s6_pass"]
    result = pilot.run_loop("constant_turn", cell, seed=SEED, **pilot.variant_settings("w1", cell))
    assert result.follow_p95_steady_m == pytest.approx(GOLDEN_W1_S6_PASS_TURN_STEADY_P95_M, abs=1e-12)
    assert pilot.SWITCH_VARIANTS[0] not in pilot.VARIANTS  # the default run never adds switch rows
