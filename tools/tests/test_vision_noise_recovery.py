from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sim.detection_log import detection_log_lines, parse_detection_log_lines
from sim.sensor import SensorConfig
from sim.vision_noise_fit import fit_vision_noise
from tools.make_detection_fixture import make_fixture


def _fixture(**kwargs):
    config_values = {"noise_std_m": 0.04, "bias_m": (0.02, -0.01, 0.05),
                     "dropout_probability": 0.1}
    config_values.update(kwargs.pop("config", {}))
    config = SensorConfig(**config_values)
    return make_fixture(n_frames=20_000, frame_rate_hz=30, sensor_config=config, seed=7,
                        latency_mean_s=0.03, latency_std_s=0.008,
                        trajectory=kwargs.pop("trajectory", "circle"))


def _check_ci(fit, key, truth):
    item = fit["uncertainty"][key]
    assert item["ci_low"] <= truth <= item["ci_high"]


def test_independent_dropout_recovery() -> None:
    trace = _fixture()
    fit = fit_vision_noise(trace, n_bootstrap=300, seed=7)
    n = fit["input"]["counts"]["detected_with_truth"]
    # Pooled three-axis noise has SE 0.04/sqrt(6(n-1)); five SE gives the tolerance.
    assert abs(fit["sensor_config"]["noise_std_m"] - 0.04) < 5 * 0.04 / np.sqrt(6 * (n - 1))
    for axis, truth in zip("xyz", (0.02, -0.01, 0.05)):
        assert abs(fit["sensor_config"]["bias_m"]["xyz".index(axis)] - truth) < 5 * 0.04 / np.sqrt(n)
        _check_ci(fit, f"bias_m_{axis}", truth)
    # A Bernoulli proportion has SE sqrt(p(1-p)/N), with N all frames.
    assert abs(fit["sensor_config"]["dropout_probability"] - 0.1) < 5 * np.sqrt(0.1 * 0.9 / 20_000)
    # The latency mean has SE std/sqrt(N). For lognormal latency, w=1+CV^2 and
    # kurtosis k=w^4+2w^3+3w^2-3; SE(sample std) is about sigma*sqrt((k-1)/(4N)).
    assert abs(fit["scenario_delay"]["measurement_delay_s"] - 0.03) < 5 * 0.008 / np.sqrt(20_000)
    w = 1 + (0.008 / 0.03) ** 2
    k = w**4 + 2 * w**3 + 3 * w**2 - 3
    latency_tol = 5 * 0.008 * np.sqrt((k - 1) / (4 * 20_000))
    assert abs(fit["scenario_delay"]["measurement_jitter_s"] - 0.008) < latency_tol
    for key, truth in (("noise_std_m", 0.04), ("dropout_probability", 0.1),
                       ("measurement_delay_s", 0.03), ("measurement_jitter_s", 0.008)):
        _check_ci(fit, key, truth)


def test_burst_recovery() -> None:
    trace = _fixture(config={"mode": "burst", "dropout_probability": 0.0,
                             "burst_start_probability": 0.02, "burst_end_probability": 0.25})
    fit = fit_vision_noise(trace, dropout_mode="burst", n_bootstrap=300, seed=7)
    n_d = fit["input"]["counts"]["missed"]
    n_a = fit["input"]["counts"]["detected"]
    p = 0.25 * (1 - 0.02)
    tol_s = 5 * np.sqrt(0.02 * 0.98 / n_a)
    # e is estimated only from dropped steps (about 1500 of 20000). P(drop->avail) = e(1-s).
    tol_e = 5 * np.sqrt(p * (1 - p) / n_d) / (1 - 0.02)
    assert abs(fit["sensor_config"]["burst_start_probability"] - 0.02) < tol_s
    assert abs(fit["sensor_config"]["burst_end_probability"] - 0.25) < tol_e
    _check_ci(fit, "burst_start_probability", 0.02)
    _check_ci(fit, "burst_end_probability", 0.25)
    cfg = SensorConfig(**fit["sensor_config"])
    assert cfg.mode == "burst"


def test_ci_coverage_over_40_seeds() -> None:
    covered = {"noise_std_m": 0, "dropout_probability": 0,
               "bias_m_x": 0, "bias_m_y": 0, "bias_m_z": 0}
    for seed in range(40):
        config = SensorConfig(noise_std_m=0.04, bias_m=(0.02, -0.01, 0.05), dropout_probability=0.1)
        fit = fit_vision_noise(make_fixture(n_frames=1_000, frame_rate_hz=30, sensor_config=config,
                                            seed=seed, latency_mean_s=0.03, latency_std_s=0.008),
                               n_bootstrap=200, seed=seed)
        for key, truth in (("noise_std_m", 0.04), ("dropout_probability", 0.1),
                           ("bias_m_x", 0.02), ("bias_m_y", -0.01), ("bias_m_z", 0.05)):
            covered[key] += int(fit["uncertainty"][key]["ci_low"] <= truth <= fit["uncertainty"][key]["ci_high"])
    # Bias axes pool 120 trials. At true 95%, falling below 106 is about 1e-3;
    # at 85% coverage this catches the problem about 80% of the time.
    assert sum(covered[key] for key in ("bias_m_x", "bias_m_y", "bias_m_z")) >= 106
    # For 40 trials, SD is about 1.38, so 32 is about 4.3 SD below 38. This
    # guards against badly under-covering intervals, not small miscalibration.
    assert covered["noise_std_m"] >= 32
    assert covered["dropout_probability"] >= 32


def test_static_trajectory_recovery() -> None:
    config = SensorConfig(noise_std_m=0.04, bias_m=(0.02, -0.01, 0.05), dropout_probability=0.1)
    fit = fit_vision_noise(make_fixture(n_frames=5_000, frame_rate_hz=30, sensor_config=config, seed=7,
                                        latency_mean_s=0.03, latency_std_s=0.008, trajectory="static"), seed=7)
    n = fit["input"]["counts"]["detected_with_truth"]
    # Static truth still permits bias and noise estimates; use five sampling SEs.
    assert np.allclose(fit["sensor_config"]["bias_m"], config.bias_m, atol=5 * 0.04 / np.sqrt(n))
    assert abs(fit["sensor_config"]["noise_std_m"] - 0.04) < 5 * 0.04 / np.sqrt(6 * (n - 1))


def test_zero_latency_jitter() -> None:
    config = SensorConfig(noise_std_m=0.04, bias_m=(0.02, -0.01, 0.05), dropout_probability=0.1)
    fit = fit_vision_noise(make_fixture(n_frames=2_000, frame_rate_hz=30, sensor_config=config, seed=7,
                                        latency_mean_s=0.03, latency_std_s=0), seed=7)
    assert fit["scenario_delay"]["measurement_jitter_s"] == pytest.approx(0.0, abs=1e-9)
    assert fit["scenario_delay"]["measurement_delay_s"] == pytest.approx(0.03)


def test_hand_worked_tiny_case() -> None:
    # Residual axes are [0,1,2,3,4], [-1,0,1,2,3], [2,3,4,5,6]: means 2, 1, 4;
    # each sample std is sqrt(2.5), so pooled sample noise_std_m is sqrt(2.5); dropout is 1/6.
    jsonl = '''{"schema":"zephyr-detection-log-1","trace_id":"tiny","status":"synthetic","clock_domain":"shared_monotonic","capture":{"source_type":"camera","source_id":"x","device_ids":["d"],"model":"m","resolution_px":{"width":1,"height":1},"frame_rate_hz":1}}
{"observation_id":"0","frame_id":"0","subject_id":"s","capture_time_s":1,"processing_time_s":1.01,"outcome":"detected","position_m":[0,-1,2],"position_frame":"camera","ground_truth":{"position_m":[0,0,0],"uncertainty_m":0,"method":"x"}}
{"observation_id":"1","frame_id":"1","subject_id":"s","capture_time_s":2,"processing_time_s":2.01,"outcome":"detected","position_m":[1,0,3],"position_frame":"camera","ground_truth":{"position_m":[0,0,0],"uncertainty_m":0,"method":"x"}}
{"observation_id":"2","frame_id":"2","subject_id":"s","capture_time_s":3,"processing_time_s":3.01,"outcome":"detected","position_m":[2,1,4],"position_frame":"camera","ground_truth":{"position_m":[0,0,0],"uncertainty_m":0,"method":"x"}}
{"observation_id":"3","frame_id":"3","subject_id":"s","capture_time_s":4,"processing_time_s":4.01,"outcome":"detected","position_m":[3,2,5],"position_frame":"camera","ground_truth":{"position_m":[0,0,0],"uncertainty_m":0,"method":"x"}}
{"observation_id":"4","frame_id":"4","subject_id":"s","capture_time_s":5,"processing_time_s":5.01,"outcome":"detected","position_m":[4,3,6],"position_frame":"camera","ground_truth":{"position_m":[0,0,0],"uncertainty_m":0,"method":"x"}}
{"observation_id":"5","frame_id":"5","subject_id":"s","capture_time_s":6,"outcome":"missed","miss_reason":"x"}'''
    fit = fit_vision_noise(parse_detection_log_lines(jsonl.splitlines()), n_bootstrap=0)
    assert np.allclose(fit["sensor_config"]["bias_m"], (2, 1, 4))
    assert fit["sensor_config"]["noise_std_m"] == pytest.approx(np.sqrt(2.5))
    assert fit["input"]["counts"]["detected"] == 5
    assert fit["scenario_delay"]["measurement_delay_s"] == pytest.approx(0.01)
    assert fit["scenario_delay"]["measurement_jitter_s"] == pytest.approx(0.0, abs=1e-12)
    assert fit["input"]["counts"]["missed"] == 1
    assert fit["sensor_config"]["dropout_probability"] == pytest.approx(1 / 6)


def test_cli_round_trip(tmp_path) -> None:
    from tools.fit_vision_noise import main
    from tools.make_detection_fixture import main as make_main
    path = tmp_path / "fixture.jsonl"
    out = tmp_path / "fit.json"
    make_main([str(path), "--frames", "200", "--rate", "30", "--noise-std", "0.04", "--bias", "0.02", "-0.01", "0.05",
               "--dropout", "0.1", "--latency-mean", "0.03", "--latency-std", "0.008", "--seed", "7"])
    assert main([str(path), "--output", str(out), "--bootstrap", "0"]) == 0
    result = json.loads(out.read_text())
    assert result["schema"] == "zephyr-vision-noise-fit-1"
    assert result["input"]["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_different_seeds_produce_different_logs() -> None:
    traces = [make_fixture(n_frames=30, frame_rate_hz=30,
                           sensor_config=SensorConfig(noise_std_m=0.04, bias_m=(0.02, -0.01, 0.05),
                                                       dropout_probability=0.1),
                           seed=seed, latency_mean_s=0.03, latency_std_s=0.008)
              for seed in (7, 8)]
    assert detection_log_lines(traces[0]) != detection_log_lines(traces[1])
