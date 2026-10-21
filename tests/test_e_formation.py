"""EXP-034 formation mission checks."""
from __future__ import annotations

import math

import pytest

from sim.missions.api import Scenario
from sim.missions.engine import run_mission
from sim.missions.log import validate_log
from platform_compare import same_tree
from tests.reference_e.hand_checks import first_order_loop_jitter_std


def _scenario(seed: int, *, duration_s: float = 8.0, **params) -> Scenario:
    return Scenario(
        mission="formation",
        seed=seed,
        duration_s=duration_s,
        adapter="point_mass_offboard",
        preset="clean",
        params=params,
    )


def test_point_mass_hand_check_matches_first_order_reference():
    sigma = 0.05
    result = run_mission(_scenario(1, duration_s=30.0, fix_noise_std_m=sigma, fix_latency_s=0.0, fix_rate_hz=30.0))
    measured = result.metrics["per_axis_std_m"]
    reference = first_order_loop_jitter_std(sigma, 1.0 / 30.0, 0.25)
    deviation = (measured - reference) / reference
    print(f"per_axis_std_m={measured:.9f} reference={reference:.9f} deviation={deviation:+.1%}")
    print("The point-mass acceleration cap and discrete control loop can offset the ideal first-order value.")
    assert measured == pytest.approx(reference, rel=0.15)


@pytest.mark.parametrize("rule", ["coop", "local"])
def test_point_mass_rms_matches_hand_formula_within_10_percent(rule):
    # 3-D RMS of a first-order loop on held fixes: sqrt(3) x the per-axis std.
    # Measured on seed 1, 30 s (RMS / reference): sigma 0.02 gives 1.089 in
    # both modes; sigma 0.05 gives 1.102 coop and 1.099 local; sigma 0.10
    # gives 1.179 coop and 1.093 local, because Run D's rule then clips the
    # noise-driven velocity toward neighbours on many steps.
    sigma = 0.02
    result = run_mission(_scenario(1, duration_s=30.0, fix_noise_std_m=sigma, fix_latency_s=0.0,
                                   fix_rate_hz=30.0, separation_rule=rule))
    reference = math.sqrt(3.0) * first_order_loop_jitter_std(sigma, 1.0 / 30.0, 0.25)
    ratio = result.metrics["rms_error_m"] / reference
    print(f"{rule} rms={result.metrics['rms_error_m']:.9f} reference={reference:.9f} ratio={ratio:.4f}")
    assert ratio == pytest.approx(1.0, abs=0.10)


def test_local_separation_rule_reproduces_pre_run_d_metrics():
    # Recorded from the code before the coop rule was added (seed 1, 30 s,
    # sigma 0.10, latency 0.2 s).  Frames and events also matched.
    before = {
        'downwash_wind_max_mps': 0.0, 'duration_s': 30.0, 'end_reason': 'timeout', 'guard_activations': 0,
        'min_separation_m': 1.7196263535089085, 'p95_error_m': 0.17113029234671245,
        'per_axis_std_m': 0.05893817087362754, 'rms_error_m': 0.10383258171137955,
    }
    result = run_mission(_scenario(1, duration_s=30.0, fix_noise_std_m=0.10, fix_latency_s=0.2,
                                   separation_rule="local"))
    assert same_tree(result.metrics, before)


def test_error_grows_with_fix_noise():
    low = run_mission(_scenario(1, fix_noise_std_m=0.01)).metrics["rms_error_m"]
    high = run_mission(_scenario(1, fix_noise_std_m=0.10)).metrics["rms_error_m"]
    print(f"sigma=0.01 rms={low:.9f}; sigma=0.10 rms={high:.9f}")
    assert high > low


def test_latency_raises_error_for_three_seeds():
    increases = []
    for seed in (1, 2, 3):
        no_latency = run_mission(_scenario(seed, fix_noise_std_m=0.05, fix_latency_s=0.0)).metrics["rms_error_m"]
        delayed = run_mission(_scenario(seed, fix_noise_std_m=0.05, fix_latency_s=0.2)).metrics["rms_error_m"]
        increases.append(delayed > no_latency)
        print(f"seed={seed} latency0={no_latency:.9f} latency0.2={delayed:.9f}")
    assert all(increases)


def test_clean_minimum_separation_stays_above_1_5_m():
    for seed in (1, 2, 3):
        metrics = run_mission(_scenario(seed, fix_noise_std_m=0.02)).metrics
        assert metrics["min_separation_m"] > 1.5


def test_physics_offboard_reports_metrics():
    scenario = Scenario(
        mission="formation",
        seed=7,
        duration_s=6.0,
        adapter="physics_offboard",
        preset="clean",
        params={"fix_noise_std_m": 0.02},
    )
    result = run_mission(scenario)
    print(f"physics_offboard rms={result.metrics['rms_error_m']:.9f}")
    assert result.end_reason == "timeout"
    assert result.metrics["rms_error_m"] is not None
    assert result.metrics["p95_error_m"] is not None
    assert result.metrics["per_axis_std_m"] is not None
    assert result.metrics["downwash_wind_max_mps"] == 0.0


def test_determinism_and_log_validation():
    scenario = _scenario(11, duration_s=6.0, fix_noise_std_m=0.02)
    first = run_mission(scenario, log=True)
    second = run_mission(scenario, log=True)
    assert first.log is not None
    assert second.log is not None
    validate_log(first.log)
    validate_log(second.log)
    assert first.metrics == second.metrics
    assert first.log == second.log
