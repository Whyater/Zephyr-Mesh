import json
from pathlib import Path

import pytest

import sim.investigation as investigation
from sim.adapters import Capabilities
from sim.investigation import run_sweep, SweepConfig

from platform_compare import same_tree

OLD_DEFAULT_ACCELERATIONS = (0.0, 0.5)


def test_sweep_is_deterministic_and_declares_criterion():
    a = run_sweep(delays=(0.0, 0.1), losses=(0.0,), noises=(0.0,), accelerations=(0.0,))
    b = run_sweep(delays=(0.0, 0.1), losses=(0.0,), noises=(0.0,), accelerations=(0.0,))
    assert a == b
    assert len(a["rows"]) == 2
    assert "criterion" in a["rows"][0]
    assert "not flight performance" in a["status"]


def test_zero_delay_zero_noise_constant_velocity_is_bounded():
    report = run_sweep(delays=(0.0,), losses=(0.0,), noises=(0.0,))
    row = report["rows"][0]
    assert row["p95_error_m"] < 0.01
    assert row["held_last_p95_error_m"] > row["p95_error_m"]


def test_latency_and_loss_are_retained_as_inputs():
    report = run_sweep(delays=(0.1,), losses=(0.3,), noises=(0.05,))
    row = report["rows"][0]
    assert row["delay_s"] == 0.1
    assert row["loss_probability"] == 0.3
    assert row["noise_std_m"] == 0.05


def test_arbitrary_delay_is_retained_without_timestep_rounding():
    report = run_sweep(delays=(0.03,), losses=(0.0,), noises=(0.0,))
    row = report["rows"][0]
    assert row["delay_s"] == 0.03
    # The configured delay is retained exactly; a fixed-rate display can only
    # consume it on the first grid sample at or after the receive instant.
    assert abs(row["mean_age_s"] - 0.04) < 1e-12


def test_accelerating_target_exposes_constant_velocity_limit():
    # 0.15 m/s^2 over 8 s peaks at 1.9 m/s, inside the 2.0 m/s vehicle limit.
    report = run_sweep(delays=(0.1,), losses=(0.0,), noises=(0.0,), accelerations=(0.0, 0.15))
    constant, accelerating = report["rows"]
    assert accelerating["target_acceleration_mps2"] == 0.15
    assert accelerating["p95_error_m"] > 2 * constant["p95_error_m"]


def test_delayed_constant_velocity_matches_independent_age_calculation():
    row = run_sweep(config=SweepConfig(duration_s=2), delays=(0.1,), losses=(0,),
                    noises=(0,), accelerations=(0,))["rows"][0]
    # A 0.1 s old observation plus the last 0.08 s of a 0.1 s sample interval
    # gives 0.7 * 0.18 = 0.126 m. Receipt-time prediction converges to a
    # trajectory lagged by 0.1 s, so its steady error remains 0.07 m.
    assert row["mean_age_s"] == pytest.approx(0.1)
    assert row["held_last_max_error_m"] == pytest.approx(0.126)
    assert row["p95_error_m"] == pytest.approx(0.07, abs=0.001)


def test_observation_period_shorter_than_display_grid_keeps_event_timestamps():
    cfg = SweepConfig(duration_s=0.2, truth_dt_s=0.02, observation_period_s=0.005)
    row = run_sweep(config=cfg, delays=(0,), losses=(0,), noises=(0,), accelerations=(0,))["rows"][0]
    assert row["sample_count"] == 11
    assert row["max_error_m"] < 0.004


def test_independent_check_uses_custom_speed_and_observation_period():
    cfg = SweepConfig(target_speed_mps=1.2, observation_period_s=0.25)
    report = run_sweep(config=cfg, delays=(0,), losses=(0,), noises=(0,), accelerations=(0,))
    assert "held-last observation-period lag is speed x observation period = 1.2 m/s x 0.25 s = 0.3 m" in report["independent_check"]


def test_row_labels_follow_declared_criteria():
    cfg = SweepConfig(failure_p95_m=0.01, failure_max_m=0.01)
    report = run_sweep(config=cfg, delays=(0.0,), losses=(0.0,), noises=(0.0,), accelerations=(0.0,))
    for row in report["rows"]:
        expected = "failure-boundary" if row["p95_error_m"] > cfg.failure_p95_m or row["max_error_m"] > cfg.failure_max_m else "within-scenario-envelope"
        assert row["label"] == expected


# ---- S6 repair checks (independent of the implementation) ----------------

def test_naive_error_is_speed_times_delay_by_hand_calculation():
    # Hand calculation: a receive-time tracker lags a constant-speed target by
    # v x delay once it settles. 0.7 m/s x 0.2 s = 0.14 m, 0.7 x 0.1 = 0.07 m.
    # Both delays sit on the 0.02 s display grid, so no extra grid lag appears.
    for delay, expected in ((0.1, 0.07), (0.2, 0.14)):
        row = run_sweep(delays=(delay,), losses=(0.0,), noises=(0.0,), accelerations=(0.0,))["rows"][0]
        assert row["p95_error_m"] == pytest.approx(expected, abs=0.001)


def test_accelerating_lag_matches_speed_at_the_end_times_delay():
    # Hand calculation for a = 0.15, delay 0.2: lag at time t is v(t) x 0.2.
    # The 95th percentile of lag over a uniform run is at t = 0.95 x 8 s:
    # (0.7 + 0.15 x 7.6) x 0.2 = 0.368 m. The final lag is 1.9 x 0.2 = 0.38 m.
    row = run_sweep(delays=(0.2,), losses=(0.0,), noises=(0.0,), accelerations=(0.15,))["rows"][0]
    assert row["p95_error_m"] == pytest.approx(0.368, abs=0.01)
    assert row["max_error_m"] == pytest.approx(0.38, abs=0.01)


# Values produced by the pre-repair inline loop (zero loss, zero noise), so the
# rewired link and sensor path must give the same numbers in the limiting case.
OLD_INLINE_LIMITING_CASE = {
    0.0: (2.080321211406222e-07, 0.055999999999999994),
    0.05: (0.042000000308141106, 0.098),
    0.1: (0.07000000030948728, 0.126),
    0.2: (0.1400000003170898, 0.196),
}


def test_rewired_sweep_reproduces_old_numbers_with_zero_loss_and_zero_noise():
    for delay, (p95, maximum) in OLD_INLINE_LIMITING_CASE.items():
        row = run_sweep(delays=(delay,), losses=(0.0,), noises=(0.0,), accelerations=(0.0,))["rows"][0]
        assert row["p95_error_m"] == pytest.approx(p95, abs=1e-9)
        assert row["max_error_m"] == pytest.approx(maximum, abs=1e-9)


def test_sweep_drives_the_tested_link_and_sensor(monkeypatch):
    calls = {"link": 0, "send": 0, "sensor": 0, "sample": 0}

    class CountingLink(investigation.SimulatedLink):
        def __init__(self, *a, **k):
            calls["link"] += 1
            super().__init__(*a, **k)

        def send(self, *a, **k):
            calls["send"] += 1
            return super().send(*a, **k)

    class CountingSensor(investigation.SensorModel):
        def __init__(self, *a, **k):
            calls["sensor"] += 1
            super().__init__(*a, **k)

        def sample(self, *a, **k):
            calls["sample"] += 1
            return super().sample(*a, **k)

    monkeypatch.setattr(investigation, "SimulatedLink", CountingLink)
    monkeypatch.setattr(investigation, "SensorModel", CountingSensor)
    run_sweep(config=SweepConfig(duration_s=1.0), delays=(0.1,), losses=(0.3,), noises=(0.02,), accelerations=(0.0,))
    assert calls["link"] == 1 and calls["sensor"] == 1
    assert calls["send"] == calls["sample"] == 11  # 0.0 to 1.0 s every 0.1 s


def test_cells_differing_only_in_noise_share_loss_pattern():
    # Common random numbers. Noise 1e-9 vs 2e-9 changes the measurements by far
    # less than a micrometre, so the held-last error can only differ if the
    # loss pattern differs. Separate seeds per cell make it differ by millimetres.
    report = run_sweep(delays=(0.1,), losses=(0.3,), noises=(1e-9, 2e-9), accelerations=(0.0,))
    a, b = report["rows"]
    assert a["noise_std_m"] != b["noise_std_m"]
    assert a["held_last_p95_error_m"] == pytest.approx(b["held_last_p95_error_m"], abs=1e-6)
    assert a["held_last_max_error_m"] == pytest.approx(b["held_last_max_error_m"], abs=1e-6)


def test_cells_differing_only_in_loss_share_noise_draws():
    # With loss 0 and loss 1e-12 (no packet actually dropped) the noise draws
    # must be identical, so the whole row must be identical.
    report = run_sweep(delays=(0.1,), losses=(0.0, 1e-12), noises=(0.05,), accelerations=(0.0,))
    a, b = report["rows"]
    for field in ("p95_error_m", "max_error_m", "held_last_p95_error_m", "mean_age_s"):
        assert a[field] == b[field]


def test_output_text_states_the_estimator_is_naive_and_no_recovery_arm_exists():
    report = run_sweep(delays=(0.2,), losses=(0.0,), noises=(0.0,), accelerations=(0.0,))
    check = report["independent_check"]
    assert "receive-time (naive)" in check
    assert "speed x delay by construction" in check
    assert "v x delay = 0.7 m/s x 0.2 s = 0.14 m" in check
    assert "v x L" not in check  # L was ambiguous (delay vs observation period)
    assert "No recovery arm exists yet" in check
    assert "oracle tuning" in report["criterion_note"]
    assert "One seed per cell" in report["criterion_note"]


def test_default_target_stays_within_reference_vehicle_speed():
    assert investigation.REFERENCE_MAX_SPEED_MPS == Capabilities().max_speed_mps
    cfg = SweepConfig()
    # Hand bound: v(T) = 0.7 + a x 8 must not exceed 2.0, so a <= 1.3 / 8 = 0.1625.
    for a in investigation.DEFAULT_ACCELERATIONS_MPS2:
        assert cfg.target_speed_mps + a * cfg.duration_s <= Capabilities().max_speed_mps
    assert investigation.peak_target_speed(SweepConfig(target_acceleration_mps2=0.5)) == pytest.approx(4.7)
    assert investigation.peak_target_speed(SweepConfig(target_acceleration_mps2=0.15)) == pytest.approx(1.9)


def _failures(report):
    return [row for row in report["rows"] if row["label"] == "failure-boundary"]


def test_old_default_grid_failures_are_all_acceleration_artifacts():
    report = run_sweep(accelerations=OLD_DEFAULT_ACCELERATIONS)
    failures = _failures(report)
    assert len(failures) == 27
    assert {row["target_acceleration_mps2"] for row in failures} == {0.5}
    # The new honesty check flags every one: it still fails with loss and noise off.
    assert all("also fails with zero loss and zero noise" in row["criterion"] for row in failures)
    assert "EXCEEDED" in report["criterion_note"]


def test_new_default_grid_labels_each_failure_with_its_driver():
    report = run_sweep()
    assert "EXCEEDED" not in report["criterion_note"]
    failures = _failures(report)
    # Straight-line target never fails; failures need delay and acceleration.
    assert all(row["target_acceleration_mps2"] == 0.15 and row["delay_s"] >= 0.1 for row in failures)
    for row in failures:
        assert ("also fails with zero loss and zero noise" in row["criterion"]) != (
            "passes with zero loss and zero noise" in row["criterion"])
    # Failures that persist with loss and noise off are delay and motion results,
    # and they must be labelled as such, never as loss or noise tolerance.
    for row in failures:
        if "also fails" in row["criterion"]:
            assert "not loss or noise" in row["criterion"]


def test_committed_example_is_regenerated_by_the_script(tmp_path):
    from tools.build_investigation_report import main

    out = tmp_path / "sweep.json"
    assert main(["--write-sweep", str(out)]) == 0
    committed = Path(__file__).parents[1] / "sim" / "investigation_example.json"
    assert same_tree(json.loads(out.read_text(encoding="utf-8")), json.loads(committed.read_text(encoding="utf-8")))


def test_tracker_is_told_the_true_noise_oracle_tuning(monkeypatch):
    # Behavioural pin for a known limit (oracle noise tuning): the tracker's measurement_std is
    # the true noise (floored at 1e-6), not an independent assumption.
    seen = []

    class SpyTracker(investigation.ConstantVelocityTracker):
        def __init__(self, *a, **k):
            seen.append(k["measurement_std_m"])
            super().__init__(*a, **k)

    monkeypatch.setattr(investigation, "ConstantVelocityTracker", SpyTracker)
    run_sweep(config=SweepConfig(duration_s=1.0), delays=(0.1,), losses=(0.0,),
              noises=(0.0, 0.02, 0.05), accelerations=(0.0,))
    assert seen == [1e-6, 0.02, 0.05]
