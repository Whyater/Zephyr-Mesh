import pytest
from sim.investigation import run_sweep, SweepConfig


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
    report = run_sweep(delays=(0.1,), losses=(0.0,), noises=(0.0,), accelerations=(0.0, 0.5))
    constant, accelerating = report["rows"]
    assert accelerating["target_acceleration_mps2"] == 0.5
    assert accelerating["label"] == "failure-boundary"
    assert accelerating["p95_error_m"] > constant["p95_error_m"]


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
    assert "1.2 m/s x 0.25 s = 0.3 m" in report["independent_check"]


def test_row_labels_follow_declared_criteria():
    cfg = SweepConfig(failure_p95_m=0.01, failure_max_m=0.01)
    report = run_sweep(config=cfg, delays=(0.0,), losses=(0.0,), noises=(0.0,), accelerations=(0.0,))
    for row in report["rows"]:
        expected = "failure-boundary" if row["p95_error_m"] > cfg.failure_p95_m or row["max_error_m"] > cfg.failure_max_m else "within-scenario-envelope"
        assert row["label"] == expected
