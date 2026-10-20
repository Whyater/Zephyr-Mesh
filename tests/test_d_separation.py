"""Independent checks for the simulated separation-buffer encounter model."""
from __future__ import annotations

import numpy as np
import pytest

from sim.coop.separation import (
    EncounterSpec,
    _linear_segment_minimum,
    hand_buffer_m,
    run_encounter,
)


@pytest.mark.parametrize("speed_mps", (0.5, 1.0, 2.0))
def test_perfect_information_head_on_keeps_distance_and_reaches_goals(speed_mps):
    spec = EncounterSpec(
        kind="head_on",
        speed_mps=speed_mps,
        delay_s=0.0,
        loss_probability=0.0,
        noise_std_m=0.0,
        dead_reckoning=False,
        avoidance_radius_m=0.35,
        share_rate_hz=20.0,
    )
    result = run_encounter(spec, seed=17)
    assert result["min_center_distance_m"] >= 0.35 - 1.0e-6
    assert result["reached_goals"] is True


def test_perfect_information_crossing_keeps_distance_and_reaches_goals():
    spec = EncounterSpec(
        kind="crossing",
        speed_mps=1.0,
        delay_s=0.0,
        loss_probability=0.0,
        noise_std_m=0.0,
        dead_reckoning=False,
        avoidance_radius_m=0.35,
        share_rate_hz=20.0,
    )
    result = run_encounter(spec, seed=17)
    assert result["min_center_distance_m"] >= 0.35 - 1.0e-6
    assert result["reached_goals"] is True


def test_perfect_information_ring_swap_keeps_every_pair_separated_and_reaches_goals():
    spec = EncounterSpec(
        kind="ring_swap",
        speed_mps=1.0,
        delay_s=0.0,
        loss_probability=0.0,
        noise_std_m=0.0,
        dead_reckoning=False,
        avoidance_radius_m=0.35,
        share_rate_hz=20.0,
    )
    result = run_encounter(spec, seed=17)
    assert result["min_center_distance_m"] >= 0.35 - 1.0e-6
    assert result["reached_goals"] is True


def test_ring_swap_at_default_share_rate_is_recorded_without_a_pass_claim():
    spec = EncounterSpec(
        kind="ring_swap",
        speed_mps=1.0,
        delay_s=0.0,
        loss_probability=0.0,
        noise_std_m=0.0,
        dead_reckoning=False,
        avoidance_radius_m=0.35,
        share_rate_hz=10.0,
    )
    result = run_encounter(spec, seed=17)
    assert np.isfinite(result["min_center_distance_m"])
    assert isinstance(result["reached_goals"], bool)


def test_avoidance_disabled_matches_hand_geometry():
    head_on = EncounterSpec(
        kind="head_on",
        speed_mps=1.0,
        delay_s=0.0,
        loss_probability=0.0,
        noise_std_m=0.0,
        dead_reckoning=False,
        avoidance_radius_m=0.35,
        lateral_offset_m=0.1,
        avoid=False,
    )
    crossing = EncounterSpec(
        kind="crossing",
        speed_mps=1.0,
        delay_s=0.0,
        loss_probability=0.0,
        noise_std_m=0.0,
        dead_reckoning=False,
        avoidance_radius_m=0.35,
        avoid=False,
    )
    assert run_encounter(head_on, seed=17)["min_center_distance_m"] == pytest.approx(0.1, abs=1.0e-6)
    crossing_result = run_encounter(crossing, seed=17)
    assert crossing_result["min_center_distance_m"] == pytest.approx(0.0, abs=1.0e-6)
    assert crossing_result["ellipsoid_violated"] is True


def test_stale_position_is_delay_times_speed_behind_true_position():
    delay = 0.2
    speed = 1.0
    spec = EncounterSpec(
        kind="head_on",
        speed_mps=speed,
        delay_s=delay,
        loss_probability=0.0,
        noise_std_m=0.0,
        dead_reckoning=False,
        avoidance_radius_m=0.35,
        avoid=False,
    )
    result = run_encounter(spec, seed=17)
    trace = result["perceived_trace"]
    checked = 0
    for item in trace:
        age = float(item["age_s"])
        if age <= 0.0:
            continue
        # The first few samples are acceleration, and the final samples slow
        # down near the goal.  The middle of the segment is constant velocity.
        if not 0.3 <= float(item["send_time_s"]) <= 2.0:
            continue
        true_position = np.asarray(item["true_position"], dtype=float)
        perceived_position = np.asarray(item["perceived_position"], dtype=float)
        sent_velocity = np.asarray(item["perceived_velocity_mps"], dtype=float)
        displacement = true_position - perceived_position
        # Once acceleration has reached the requested cruise speed, the path is
        # a straight segment and stale position differs by v times its age.
        if not np.isclose(np.linalg.norm(sent_velocity), speed, atol=1.0e-12):
            continue
        assert np.linalg.norm(displacement) == pytest.approx(speed * age, abs=1.0e-9)
        checked += 1
    assert checked > 0, "the encounter did not expose a straight-segment stale sample"


def test_zero_delay_message_ages_are_control_step_aligned():
    spec = EncounterSpec(
        kind="head_on",
        speed_mps=1.0,
        delay_s=0.0,
        loss_probability=0.0,
        noise_std_m=0.0,
        dead_reckoning=False,
        avoidance_radius_m=0.35,
        share_rate_hz=10.0,
        control_dt_s=0.05,
        avoid=False,
    )
    ages = run_encounter(spec, seed=17)["message_age_samples_s"]
    assert ages
    assert all(any(np.isclose(age, expected, atol=1.0e-12) for expected in (0.0, 0.05))
               for age in ages)


def test_exact_segment_minimum_catches_a_crossing_between_samples():
    start = np.array((-1.0, 0.2, 0.0))
    end = np.array((1.0, -0.2, 0.0))
    sampled_minimum = min(np.linalg.norm(start), np.linalg.norm(end))
    assert sampled_minimum == pytest.approx(np.sqrt(1.04), abs=1.0e-12)
    assert _linear_segment_minimum(start, end) == pytest.approx(0.0, abs=1.0e-12)


def test_exact_scaled_segment_minimum_uses_ellipsoid_coordinates():
    start = np.array((-0.48, 0.12, 0.0))
    end = np.array((0.48, -0.12, 0.0))
    scales = np.array((0.48, 0.24, 0.60))
    assert _linear_segment_minimum(start, end, scales) == pytest.approx(0.0, abs=1.0e-12)


def test_hand_buffer_example():
    spec = EncounterSpec(
        kind="head_on",
        speed_mps=1.0,
        delay_s=0.2,
        loss_probability=0.0,
        noise_std_m=0.0,
        dead_reckoning=False,
        avoidance_radius_m=0.35,
    )
    assert hand_buffer_m(spec) == pytest.approx(2.0 * (0.2 + 0.05), abs=1.0e-12)
