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
