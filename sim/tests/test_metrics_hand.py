"""Independent hand-calculated checks for every public metric."""

import math

import numpy as np
import pytest

from sim.metrics import (
    controller_recovery_time,
    failure_label,
    miss_distance_time,
    packet_age,
    packet_delivery,
    reacquisition_time,
    step_response,
    tracking_error,
)


def test_first_order_rise_time_is_tau_ln_nine():
    dt = 1e-3
    t = np.arange(0.0, 12.0 + dt / 2, dt)
    for tau in (0.1, 0.5, 2.0):
        y = 1.0 - np.exp(-t / tau)
        at_ten = step_response(t, y, 1.0, rise_fraction=0.1)["rise_time_s"]
        at_ninety = step_response(t, y, 1.0, rise_fraction=0.9)["rise_time_s"]
        assert at_ninety - at_ten == pytest.approx(tau * math.log(9.0), abs=2 * dt, rel=0.005)


@pytest.mark.parametrize("zeta, wn", [(0.2, 2.0), (0.5, 5.0), (0.7, 1.5), (0.9, 10.0)])
def test_underdamped_overshoot_and_settling_have_analytic_answers(zeta, wn):
    dt = 1e-3
    t = np.arange(0.0, 40.0 + dt / 2, dt)
    wd = wn * math.sqrt(1.0 - zeta * zeta)
    phi = math.acos(zeta)
    y = 1.0 - np.exp(-zeta * wn * t) / math.sqrt(1.0 - zeta * zeta) * np.sin(wd * t + phi)
    result = step_response(t, y, 1.0, tolerance=0.02)

    expected_overshoot = 100.0 * math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta * zeta))
    assert result["overshoot_pct"] == pytest.approx(expected_overshoot, rel=0.005, abs=0.1)

    # The dense analytic signal is the independent reference for the sampled metric.
    outside = np.flatnonzero(np.abs(y - 1.0) > 0.02)
    expected_settling = None if not len(outside) else t[outside[-1]] + dt
    assert result["settling_time_s"] == pytest.approx(expected_settling, abs=2 * dt)
    # 4/(zeta*wn) is a documented envelope estimate, so use it only as a sanity bound.
    estimate = 4.0 / (zeta * wn)
    assert abs(result["settling_time_s"] - estimate) / result["settling_time_s"] < 0.35


def test_overdamped_response_has_no_overshoot():
    t = np.arange(0.0, 10.0, 1e-3)
    y = 1.0 - np.exp(-t)
    assert step_response(t, y, 1.0)["overshoot_pct"] == 0.0


def test_tracking_error_rms_and_max_match_sine_and_ramp_math():
    dt = 1e-3
    periods = 8
    t = np.arange(0.0, periods * 2.0 * math.pi + dt / 2, dt)
    amplitude = 3.0
    sine_result = tracking_error(np.zeros_like(t), amplitude * np.sin(t))
    assert sine_result["rms_m"] == pytest.approx(amplitude / math.sqrt(2.0), abs=2e-3)

    v, k, end = 2.0, 4.0, 5.0
    ramp_t = np.arange(0.0, end + dt / 2, dt)
    reference = v * ramp_t
    estimate = reference - (v / k) * (1.0 - np.exp(-k * ramp_t))
    ramp_result = tracking_error(reference, estimate)
    expected_end_error = (v / k) * (1.0 - math.exp(-k * end))
    assert ramp_result["max_m"] == pytest.approx(expected_end_error, abs=2 * dt)


def test_packet_delivery_counts_known_sent_received_duplicate_and_reorder():
    rows = [
        {"sender_id": "a", "receiver_id": "b", "seq": 0, "sent": True, "received": True},
        {"sender_id": "a", "receiver_id": "b", "seq": 3, "sent": True},
        {"sender_id": "a", "receiver_id": "b", "seq": 2, "sent": True, "received": True},
        {"sender_id": "a", "receiver_id": "b", "seq": 1, "sent": True, "received": True},
        {"sender_id": "a", "receiver_id": "b", "seq": 1, "received": True},
    ]
    result = packet_delivery(rows)
    assert result["sent_unique"] == 4
    assert result["received_unique"] == 3
    assert result["lost_unique"] == 1
    assert result["loss_rate"] == pytest.approx(1.0 / 4.0)
    assert result["duplicates"] == 1 and result["out_of_order"] == 1


def test_packet_age_is_exact_clock_difference():
    assert packet_age(12.75, 10.25) == pytest.approx(2.5)


def test_reacquisition_and_controller_recovery_are_first_valid_times():
    t = np.arange(0.0, 5.0, 0.5)
    assert reacquisition_time(t, [1, 0, 0, 1, 1, 1, 1, 1, 1, 1], consecutive=2) == {
        "time_s": 1.0,
        "censored": False,
    }
    assert controller_recovery_time(t, [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]) == {
        "time_s": 1.5,
        "censored": False,
    }


def test_miss_distance_and_failure_label_match_closed_form_values():
    distance = [5.0, 3.0, 1.0, 2.0]
    result = miss_distance_time([0.0, 1.0, 2.0, 3.0], distance, capture_radius=1.5)
    assert result == {"miss_distance_m": 1.0, "time_to_capture_s": 2.0, "censored": False}
    assert failure_label(tracking_error_m=2.5, max_error_m=2.0, scenario="hand") == {
        "scenario": "hand",
        "status": "failure",
        "criterion": "declared scenario thresholds",
    }
    assert failure_label(tracking_error_m=1.0, max_error_m=2.0, scenario="hand")["status"] == "pass"


def test_step_response_never_settles_and_empty_inputs_are_explicitly_censored():
    t = np.arange(0.0, 5.0, 1e-3)
    never_settles = 1.0 + 0.1 * np.sin(t)
    result = step_response(t, never_settles, 1.0, tolerance=0.02)
    assert result["settling_time_s"] is None and result["censored"]["settling_time_s"]
    empty = step_response([], [], 1.0)
    assert all(empty["censored"].values())
    with pytest.raises(ValueError):
        step_response([0.0, 1.0], [0.0, float("nan")], 1.0)
