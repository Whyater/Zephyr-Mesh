"""Hand-worked checks for sim/vision_noise_fit.py.

Every expected number below is worked out by hand in the comments, not by
calling the fit's own helpers.
"""
import hashlib
import json
import math

import pytest

from sim.sensor import SensorConfig
from sim.vision_noise_fit import FIT_SCHEMA, fit_vision_noise
from sim.vision_trace import VisionTrace
from tools.fit_vision_noise import main as fit_cli

Z95 = 1.959963984540054  # standard normal 97.5th percentile


def _det(i, t, residual, truth=(10.0, -5.0, 2.0), u=0.0, subject="drone-1", frame="camera", processing=None):
    record = {
        "observation_id": f"o{i}", "frame_id": f"f{i}", "subject_id": subject, "capture_time_s": t,
        "outcome": "detected", "position_m": [truth[k] + residual[k] for k in range(3)], "position_frame": frame,
        "ground_truth": {"position_m": list(truth), "uncertainty_m": u, "method": "test"},
    }
    if processing is not None:
        record["processing_time_s"] = processing
    return record


def _miss(i, t, subject="drone-1"):
    return {"observation_id": f"o{i}", "frame_id": f"f{i}", "subject_id": subject, "capture_time_s": t,
            "outcome": "missed", "miss_reason": "test"}


def _unknown(i, t, subject="drone-1"):
    return {"observation_id": f"o{i}", "frame_id": f"f{i}", "subject_id": subject, "capture_time_s": t,
            "outcome": "unknown", "unknown_reason": "test"}


def _raw(observations, clock="shared_monotonic"):
    return {
        "schema": "zephyr-vision-trace-1", "trace_id": "hand-case", "status": "synthetic", "clock_domain": clock,
        "capture": {"source_type": "camera", "source_id": "test", "device_ids": ["host"], "model": "test"},
        "observations": observations,
    }


def _trace(observations, clock="shared_monotonic"):
    return VisionTrace.from_dict(_raw(observations, clock))


def _availability(pattern):
    """Build a log from a string: A = detected (with truth), D = missed, ? = unknown."""
    records = []
    for i, ch in enumerate(pattern):
        t = 1.0 + 0.1 * i
        if ch == "A":
            records.append(_det(i, t, (0.01 * (i % 3), 0.0, -0.01 * (i % 2))))
        elif ch == "D":
            records.append(_miss(i, t))
        else:
            records.append(_unknown(i, t))
    return _trace(records)


# Residuals (x, y, z): (1, 0, -1), (2, 0, 1), (3, 2, -1), (4, 2, 1)
RESIDUALS = [(1.0, 0.0, -1.0), (2.0, 0.0, 1.0), (3.0, 2.0, -1.0), (4.0, 2.0, 1.0)]


def _residual_trace(u=0.0):
    # Listed out of time order on purpose; the fit must sort by capture time.
    order = [2, 0, 3, 1]
    return _trace([_det(i, 1.0 + 0.1 * i, RESIDUALS[i], u=u) for i in order])


def test_bias_and_spread_by_hand():
    fit = fit_vision_noise(_residual_trace(), n_bootstrap=0)
    # x: 1,2,3,4 -> mean 2.5; squared deviations 2.25+0.25+0.25+2.25 = 5; var = 5/3
    # y: 0,0,2,2 -> mean 1;   squared deviations 1+1+1+1 = 4;           var = 4/3
    # z: -1,1,-1,1 -> mean 0; squared deviations 1+1+1+1 = 4;           var = 4/3
    assert fit["sensor_config"]["bias_m"] == pytest.approx([2.5, 1.0, 0.0])
    assert fit["per_axis"]["std_m"] == pytest.approx([math.sqrt(5 / 3), math.sqrt(4 / 3), math.sqrt(4 / 3)])
    # pooled = sqrt((5/3 + 4/3 + 4/3) / 3) = sqrt(13/9) = sqrt(13)/3
    assert fit["sensor_config"]["noise_std_m"] == pytest.approx(math.sqrt(13) / 3)
    # anisotropy = sqrt(5/3) / sqrt(4/3) = sqrt(5/4)
    assert fit["per_axis"]["anisotropy_ratio"] == pytest.approx(math.sqrt(1.25))
    assert fit["input"]["counts"] == {"observed": 4, "detected": 4, "detected_with_truth": 4, "missed": 0, "unknown": 0}


def test_analytic_bias_interval_and_truth_uncertainty_by_hand():
    # x residuals in time order 1,2,3,4; deviations -1.5,-0.5,0.5,1.5; sum of squares 5.
    # lag-1 products: 0.75 - 0.25 + 0.75 = 1.25, so r = 1.25 / 5 = 0.25.
    # n_eff = 4 * 0.75 / 1.25 = 2.4. Standard error = sqrt(5/3) / sqrt(2.4) = sqrt(5/7.2) = 0.8333...
    se = math.sqrt(5 / 7.2)
    fit = fit_vision_noise(_residual_trace(), n_bootstrap=0)
    rec = fit["uncertainty"]["bias_m_x"]
    assert se == pytest.approx(5 / 6)
    assert rec["ci_low"] == pytest.approx(2.5 - Z95 * se)
    assert rec["ci_high"] == pytest.approx(2.5 + Z95 * se)
    # Truth 1-sigma 0.1 m added in quadrature: half = sqrt((z*se)^2 + (z*0.1)^2) = z*sqrt(25/36 + 0.01)
    fit_u = fit_vision_noise(_residual_trace(u=0.1), n_bootstrap=0)
    half = Z95 * math.sqrt(25 / 36 + 0.01)
    assert fit_u["uncertainty"]["bias_m_x"]["ci_high"] == pytest.approx(2.5 + half)
    assert fit_u["uncertainty"]["bias_m_x"]["ci_low"] == pytest.approx(2.5 - half)
    assert fit_u["sensor_config"]["bias_m"] == pytest.approx([2.5, 1.0, 0.0])


def test_wilson_interval_by_hand():
    # 8 detected, 2 missed: p = 0.2, n = 10, z^2 = 3.841459
    # denom = 1 + z^2/10 = 1.3841459; center = (0.2 + z^2/20) / denom = 0.3920729 / 1.3841459 = 0.283260
    # half = (z / denom) * sqrt(0.2*0.8/10 + z^2/400) = 1.416011 * sqrt(0.0256036) = 0.226577
    fit = fit_vision_noise(_availability("AAAADAAAAD"), n_bootstrap=0)
    rec = fit["uncertainty"]["dropout_probability"]
    assert rec["estimate"] == pytest.approx(0.2)
    assert rec["ci_low"] == pytest.approx(0.283260 - 0.226577, abs=2e-6)
    assert rec["ci_high"] == pytest.approx(0.283260 + 0.226577, abs=2e-6)
    assert rec["n"] == 10


def test_transition_counts_by_hand():
    # A A A D D D A A A A -> steps AA AA AD DD DD DA AA AA AA
    # n_AA = 5, n_AD = 1, n_DD = 2, n_DA = 1
    # s = 1/6; P(D->A) = 1/3; e = (1/3) / (1 - 1/6) = (1/3) / (5/6) = 0.4
    fit = fit_vision_noise(_availability("AAADDDAAAA"), dropout_mode="burst", n_bootstrap=0)
    assert fit["transition_counts"] == {"available_to_available": 5, "available_to_dropped": 1,
                                        "dropped_to_available": 1, "dropped_to_dropped": 2}
    assert fit["sensor_config"]["burst_start_probability"] == pytest.approx(1 / 6)
    assert fit["sensor_config"]["burst_end_probability"] == pytest.approx(0.4)
    assert fit["sensor_config"]["mode"] == "burst"
    # dropout fraction 3/10
    assert fit["sensor_config"]["dropout_probability"] == pytest.approx(0.3)


def test_burst_end_is_clipped_to_one():
    # A A D D A A A D A -> AA AD DD DA AA AA AD DA: n_AA=3, n_AD=2, n_DD=1, n_DA=2
    # s = 2/5; P(D->A) = 2/3; e = (2/3)/(3/5) = 10/9 -> clipped to 1
    fit = fit_vision_noise(_availability("AADDAAADA"), dropout_mode="burst", n_bootstrap=0)
    assert fit["sensor_config"]["burst_start_probability"] == pytest.approx(0.4)
    assert fit["sensor_config"]["burst_end_probability"] == 1.0


def test_unknown_breaks_transition_counting():
    # A D ? D A -> segments [A D] and [D A]: one AD and one DA, and no DD across the unknown.
    # Dropout: 2 missed / (2 detected + 2 missed) = 0.5; the unknown is not in the denominator.
    fit = fit_vision_noise(_availability("AD?DA"), n_bootstrap=0)
    assert fit["transition_counts"] == {"available_to_available": 0, "available_to_dropped": 1,
                                        "dropped_to_available": 1, "dropped_to_dropped": 0}
    assert fit["sensor_config"]["dropout_probability"] == pytest.approx(0.5)
    assert fit["uncertainty"]["dropout_probability"]["n"] == 4
    assert fit["input"]["counts"]["unknown"] == 1


def test_latency_mean_and_std_on_three_records():
    # latencies 0.01, 0.02, 0.06 -> mean 0.03; deviations -0.02, -0.01, 0.03;
    # squares 0.0004 + 0.0001 + 0.0009 = 0.0014; / 2 = 0.0007; std = sqrt(0.0007)
    records = [
        _det(0, 1.0, (0.0, 0.0, 0.0), processing=1.01),
        _det(1, 1.1, (0.1, 0.0, 0.0), processing=1.12),
        _det(2, 1.2, (0.0, 0.1, 0.0), processing=1.26),
    ]
    fit = fit_vision_noise(_trace(records), n_bootstrap=0)
    assert fit["scenario_delay"]["measurement_delay_s"] == pytest.approx(0.03)
    assert fit["scenario_delay"]["measurement_jitter_s"] == pytest.approx(math.sqrt(0.0007))
    assert fit["uncertainty"]["measurement_delay_s"]["n"] == 3
    assert "scenario_delay_null_reason" not in fit


def test_non_shared_clock_gives_null_delay_with_reason():
    records = [_det(i, 1.0 + 0.1 * i, RESIDUALS[i], processing=1.05 + 0.1 * i) for i in range(4)]
    fit = fit_vision_noise(_trace(records, clock="device_and_host"), n_bootstrap=0)
    assert fit["scenario_delay"] is None
    assert "device_and_host" in fit["scenario_delay_null_reason"]
    assert fit["uncertainty"]["measurement_delay_s"]["estimate"] is None


def test_no_processing_time_gives_null_delay():
    fit = fit_vision_noise(_residual_trace(), n_bootstrap=0)
    assert fit["scenario_delay"] is None
    assert fit["scenario_delay_null_reason"]


def test_one_latency_record_gives_null_delay_with_reason():
    records = [_det(0, 1.0, (0.0, 0.0, 0.0), processing=1.02), _det(1, 1.1, (0.1, 0.0, 0.0))]
    fit = fit_vision_noise(_trace(records), n_bootstrap=0)
    assert fit["scenario_delay"] is None
    reason = fit["scenario_delay_null_reason"]
    assert "standard deviation needs at least 2 records" in reason
    assert "found 1" in reason
    assert fit["uncertainty"]["measurement_delay_s"]["estimate"] is None
    assert fit["uncertainty"]["measurement_jitter_s"]["n"] == 1


# D D ? A ? A -> segments [D D], [A], [A]. Only one step is counted: D -> D.
# n_AA = 0, n_AD = 0, n_DA = 0, n_DD = 1. No step leaves the available state, so s
# cannot be estimated, and e = P(D->A)/(1 - s) needs s, so e cannot be either.
UNIDENTIFIED_BURST = "DD?A?A"


def test_unidentifiable_burst_raises_in_burst_mode():
    with pytest.raises(ValueError) as info:
        fit_vision_noise(_availability(UNIDENTIFIED_BURST), dropout_mode="burst", n_bootstrap=0)
    message = str(info.value)
    assert "burst_start_probability is not identifiable" in message
    assert "burst_end_probability is not identifiable" in message
    assert "AA=0, AD=0, DA=0, DD=1" in message


def test_burst_end_unidentifiable_when_no_step_leaves_dropped_state():
    # A A A D -> AA AA AD: s = 1/3 is fine, but no step leaves the dropped state.
    with pytest.raises(ValueError, match="burst_end_probability is not identifiable") as info:
        fit_vision_noise(_availability("AAAD"), dropout_mode="burst", n_bootstrap=0)
    assert "burst_start_probability" not in str(info.value)
    fit = fit_vision_noise(_availability("AAAD"), n_bootstrap=0)
    assert fit["sensor_config"]["burst_start_probability"] == pytest.approx(1 / 3)
    assert fit["sensor_config"]["burst_end_probability"] == 1.0
    assert fit["uncertainty"]["burst_end_probability"]["estimate"] is None


def test_unidentifiable_burst_falls_back_to_defaults_in_independent_mode():
    fit = fit_vision_noise(_availability(UNIDENTIFIED_BURST), n_bootstrap=0)
    assert fit["transition_counts"] == {"available_to_available": 0, "available_to_dropped": 0,
                                        "dropped_to_available": 0, "dropped_to_dropped": 1}
    # SensorConfig defaults, not fitted numbers.
    assert fit["sensor_config"]["burst_start_probability"] == 0.0
    assert fit["sensor_config"]["burst_end_probability"] == 1.0
    for key in ("burst_start_probability", "burst_end_probability"):
        rec = fit["uncertainty"][key]
        assert rec["estimate"] is None and rec["ci_low"] is None and rec["ci_high"] is None
        assert rec["method"].startswith("not identifiable:")
    assert any("not identifiable" in text for text in fit["assumptions"])
    # The independent fit itself is unchanged: 2 missed / (2 detected + 2 missed) = 0.5.
    assert fit["sensor_config"]["dropout_probability"] == pytest.approx(0.5)
    assert fit["sensor_config"]["mode"] == "independent"
    SensorConfig(**fit["sensor_config"])
    json.dumps(fit, allow_nan=False)


def test_sensor_config_constructs_in_both_modes():
    for mode in ("independent", "burst"):
        fit = fit_vision_noise(_availability("AAADDDAAAA"), dropout_mode=mode, n_bootstrap=50)
        assert set(fit["sensor_config"]) == {"noise_std_m", "bias_m", "random_walk_std_m", "dropout_probability",
                                             "burst_start_probability", "burst_end_probability", "mode"}
        config = SensorConfig(**fit["sensor_config"])
        assert config.mode == mode
        assert config.random_walk_std_m == 0.0
        # The burst MLE is reported in independent mode too.
        assert fit["sensor_config"]["burst_start_probability"] == pytest.approx(1 / 6)


def test_fewer_than_two_residuals_raises():
    with pytest.raises(ValueError, match="at least 2"):
        fit_vision_noise(_trace([_det(0, 1.0, (0.0, 0.0, 0.0)), _miss(1, 1.1), _miss(2, 1.2)]))


def test_mixed_position_frame_raises():
    records = [_det(0, 1.0, (0.0, 0.0, 0.0)), _det(1, 1.1, (0.1, 0.0, 0.0)), _det(2, 1.2, (0.0, 0.1, 0.0), frame="world")]
    with pytest.raises(ValueError, match="position_frame"):
        fit_vision_noise(_trace(records))


def test_multiple_subjects_need_a_choice():
    records = [_det(i, 1.0 + 0.1 * i, RESIDUALS[i]) for i in range(4)]
    records += [_det(10 + i, 1.0 + 0.1 * i, RESIDUALS[i], subject="drone-2") for i in range(2)]
    trace = _trace(records)
    with pytest.raises(ValueError, match="subject_id"):
        fit_vision_noise(trace)
    fit = fit_vision_noise(trace, subject_id="drone-1", n_bootstrap=0)
    assert fit["input"]["subject_id"] == "drone-1"
    assert fit["sensor_config"]["bias_m"] == pytest.approx([2.5, 1.0, 0.0])
    with pytest.raises(ValueError, match="not in the trace"):
        fit_vision_noise(trace, subject_id="drone-9")


def test_bad_arguments_raise():
    trace = _residual_trace()
    with pytest.raises(ValueError, match="n_bootstrap"):
        fit_vision_noise(trace, n_bootstrap=-1)
    with pytest.raises(ValueError, match="dropout_mode"):
        fit_vision_noise(trace, dropout_mode="bursty")
    with pytest.raises(ValueError, match="confidence_level"):
        fit_vision_noise(trace, confidence_level=1.0)


def _json_types_only(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_types_only(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_json_types_only(v) for v in value)
    return type(value) in (str, int, float, bool, type(None))


def test_bootstrap_output_is_deterministic_and_json_safe():
    trace = _availability("AAADDAAAAADAAAADDDAAAAAAADAAAA")
    a = fit_vision_noise(trace, dropout_mode="burst", n_bootstrap=200, seed=7)
    b = fit_vision_noise(trace, dropout_mode="burst", n_bootstrap=200, seed=7)
    assert json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
    assert _json_types_only(a)
    assert a["schema"] == FIT_SCHEMA
    for name, rec in a["uncertainty"].items():
        if rec["estimate"] is None:
            continue
        assert rec["ci_low"] <= rec["estimate"] + 1e-12, name
        assert rec["estimate"] <= rec["ci_high"] + 1e-12, name


def test_cli_writes_json_with_input_hash(tmp_path):
    records = [_det(i, 1.0 + 0.1 * i, RESIDUALS[i], processing=1.03 + 0.1 * i + 0.001 * i) for i in range(4)]
    records.append(_miss(4, 1.4))
    # A detection after the miss gives one step out of the dropped state, so burst mode is identifiable.
    records.append(_det(5, 1.5, (0.0, 0.0, 0.0)))
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(_raw(records)), encoding="utf-8")
    out = tmp_path / "out" / "fit.json"
    assert fit_cli([str(source), "--output", str(out), "--bootstrap", "40", "--dropout-mode", "burst"]) == 0

    def no_constants(token):
        raise AssertionError(f"non-finite token {token}")

    fit = json.loads(out.read_text(encoding="utf-8"), parse_constant=no_constants)
    assert fit["schema"] == FIT_SCHEMA
    assert fit["input"]["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert fit["sensor_config"]["mode"] == "burst"
    assert fit["scenario_delay"]["measurement_delay_s"] == pytest.approx(0.0315)
    SensorConfig(**fit["sensor_config"])


def test_cli_reports_errors_with_exit_code_two(tmp_path):
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(_raw([_det(0, 1.0, (0.0, 0.0, 0.0))])), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        fit_cli([str(source), "--bootstrap", "0"])
    assert excinfo.value.code == 2
