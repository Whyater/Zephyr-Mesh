from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sim.sensor import SensorConfig
from tools.make_detection_fixture import main as make_main
from tools.make_detection_fixture import make_fixture


def test_fixture_is_deterministic() -> None:
    config = SensorConfig(noise_std_m=0.04, bias_m=(0.02, -0.01, 0.05), dropout_probability=0.1)
    a = make_fixture(n_frames=200, frame_rate_hz=30, sensor_config=config, seed=7,
                     latency_mean_s=0.03, latency_std_s=0.008)
    b = make_fixture(n_frames=200, frame_rate_hz=30, sensor_config=config, seed=7,
                     latency_mean_s=0.03, latency_std_s=0.008)
    c = make_fixture(n_frames=200, frame_rate_hz=30, sensor_config=config, seed=8,
                     latency_mean_s=0.03, latency_std_s=0.008)
    assert a.as_dict() == b.as_dict()
    assert a.as_dict() != c.as_dict()


def test_fixture_statistics_match_parameters() -> None:
    n = 20_000
    config = SensorConfig(noise_std_m=0.04, bias_m=(0.02, -0.01, 0.05), dropout_probability=0.1)
    trace = make_fixture(n_frames=n, frame_rate_hz=30, sensor_config=config, seed=7,
                         latency_mean_s=0.03, latency_std_s=0.008)
    misses = [o for o in trace.observations if o.outcome == "missed"]
    detected = [o for o in trace.observations if o.outcome == "detected"]
    assert abs(len(misses) / n - 0.1) < 5 * np.sqrt(0.1 * 0.9 / n)
    latencies = np.array([o.processing_time_s - o.capture_time_s for o in trace.observations])
    assert abs(latencies.mean() - 0.03) < 5 * 0.008 / np.sqrt(n)
    # For a lognormal sample, SE(std) is about sigma*sqrt((kurtosis-1)/(4N)).
    w = 1 + (0.008 / 0.03) ** 2
    k = w**4 + 2 * w**3 + 3 * w**2 - 3
    latency_tol = 5 * 0.008 * np.sqrt((k - 1) / (4 * n))
    assert abs(latencies.std(ddof=1) - 0.008) < latency_tol
    residuals = np.array([np.array(o.position_m) - np.array(o.ground_truth["position_m"])
                          for o in detected])
    assert np.all(np.abs(residuals.mean(axis=0) - np.array(config.bias_m))
                  < 5 * 0.04 / np.sqrt(len(detected)))
    # Pooling three independent axes gives 6(n-1) degrees of freedom.
    centered_residuals = residuals - np.array(config.bias_m)
    pooled_noise = centered_residuals.reshape(-1).std(ddof=1)
    noise_tol = 5 * 0.04 / np.sqrt(6 * (len(detected) - 1))
    assert abs(pooled_noise - 0.04) < noise_tol


def test_burst_mode_has_geometric_miss_runs() -> None:
    end_probability = 0.25
    config = SensorConfig(mode="burst", dropout_probability=0.0,
                          burst_start_probability=0.05,
                          burst_end_probability=end_probability)
    trace = make_fixture(n_frames=40_000, frame_rate_hz=30, sensor_config=config, seed=7,
                         latency_mean_s=0.03, latency_std_s=0.0)
    lengths: list[int] = []
    current = 0
    for observation in trace.observations:
        if observation.outcome == "missed":
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    assert lengths
    mean_length = float(np.mean(lengths))
    # A geometric run has variance (1-e)/e^2; use observed run count for its SE.
    standard_error = math.sqrt((1 - end_probability) / end_probability**2 / len(lengths))
    assert abs(mean_length - 1 / end_probability) < 5 * standard_error


def test_static_trajectory_has_constant_truth() -> None:
    trace = make_fixture(n_frames=100, frame_rate_hz=30,
                         sensor_config=SensorConfig(), seed=7,
                         latency_mean_s=0.03, latency_std_s=0.0, trajectory="static")
    truths = {tuple(observation.ground_truth["position_m"])
              for observation in trace.observations}
    assert truths == {(0.3, -0.2, 2.5)}


def test_cli_sidecar_matches_header_parameters(tmp_path) -> None:
    output = tmp_path / "fixture.jsonl"
    make_main([str(output), "--frames", "12", "--rate", "30", "--noise-std", "0.04",
               "--bias", "0.02", "-0.01", "0.05", "--dropout", "0.1",
               "--latency-mean", "0.03", "--latency-std", "0.008", "--seed", "7",
               "--trajectory", "static"])
    sidecar = json.loads(output.with_suffix(".params.json").read_text(encoding="utf-8"))
    header = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    prefix = "Parameters: "
    assert header["notes"].startswith("Synthetic fixture from sim.sensor.SensorModel; not camera data. " + prefix)
    notes_parameters = json.loads(header["notes"].split(prefix, 1)[1])
    assert notes_parameters == sidecar


def test_constant_latency_when_std_zero() -> None:
    trace = make_fixture(n_frames=100, frame_rate_hz=30, sensor_config=SensorConfig(), seed=7,
                         latency_mean_s=0.03, latency_std_s=0.0)
    latencies = np.array([o.processing_time_s - o.capture_time_s for o in trace.observations])
    assert np.all(latencies == pytest.approx(0.03, abs=1e-12))
