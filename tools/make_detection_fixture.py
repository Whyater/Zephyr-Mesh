"""Generate a deterministic synthetic detection log."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sim.sensor import SensorConfig, SensorModel
from sim.vision_trace import VisionObservation, VisionTrace


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _fixture_parameters(*, sensor_config: SensorConfig, n_frames: int,
                        frame_rate_hz: float, seed: int, latency_mean_s: float,
                        latency_std_s: float, trajectory: str) -> dict:
    params = asdict(sensor_config)
    params.update(n_frames=n_frames, frame_rate_hz=frame_rate_hz, seed=seed,
                  latency_mean_s=latency_mean_s, latency_std_s=latency_std_s,
                  trajectory=trajectory)
    return _jsonable(params)


def make_fixture(*, n_frames: int, frame_rate_hz: float, sensor_config: SensorConfig,
                 seed: int, latency_mean_s: float, latency_std_s: float,
                 trajectory: str = "circle") -> VisionTrace:
    """Build a trace using SensorModel and synthetic timing."""
    if (n_frames < 0 or not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0
            or not math.isfinite(latency_mean_s) or latency_mean_s <= 0
            or not math.isfinite(latency_std_s) or latency_std_s < 0):
        raise ValueError("frame count, frame rate, and latency parameters are invalid")
    if trajectory not in {"circle", "static"}:
        raise ValueError("trajectory must be circle or static")
    model = SensorModel(sensor_config, seed=seed)
    confidence_seed, latency_seed = np.random.SeedSequence(seed).spawn(2)
    confidence_rng = np.random.default_rng(confidence_seed)
    latency_rng = np.random.default_rng(latency_seed)
    sigma2 = float(np.log1p((latency_std_s / latency_mean_s) ** 2))
    mu = float(np.log(latency_mean_s) - sigma2 / 2.0)
    observations: list[VisionObservation] = []
    for frame in range(n_frames):
        capture_time = 1.0 + frame / frame_rate_hz
        if trajectory == "static":
            truth = np.array((0.3, -0.2, 2.5), dtype=float)
        else:
            angle = 2.0 * np.pi * capture_time / 10.0
            truth = np.array((np.cos(angle), np.sin(angle), 2.0), dtype=float)
        sample = model.sample(truth, capture_time)
        latency = float(latency_rng.lognormal(mu, np.sqrt(sigma2)))
        common = {"observation_id": f"obs-{frame:06d}", "frame_id": f"frame-{frame:06d}",
                  "subject_id": "subject-1", "capture_time_s": capture_time,
                  "processing_time_s": capture_time + latency}
        gt = {"position_m": truth.tolist(), "uncertainty_m": 0.0, "method": "synthetic"}
        if sample.available:
            observations.append(VisionObservation(
                **common, outcome="detected", confidence=float(confidence_rng.uniform(0.5, 1.0)),
                position_m=sample.measurement_m, position_frame="camera", ground_truth=gt))
        else:
            observations.append(VisionObservation(**common, outcome="missed",
                                                   miss_reason="synthetic_dropout", ground_truth=gt))
    params = _fixture_parameters(sensor_config=sensor_config, n_frames=n_frames,
                                 frame_rate_hz=frame_rate_hz, seed=seed,
                                 latency_mean_s=latency_mean_s,
                                 latency_std_s=latency_std_s, trajectory=trajectory)
    notes = ("Synthetic fixture from sim.sensor.SensorModel; not camera data. Parameters: "
             + json.dumps(params, sort_keys=True))
    return VisionTrace(
        trace_id=f"synthetic-fixture-{seed}", status="synthetic", clock_domain="shared_monotonic",
        observations=tuple(observations), source="synthetic-fixture", notes=notes,
        capture={"source_type": "camera", "source_id": "synthetic-fixture",
                 "device_ids": ["synthetic-host"], "model": "sim.sensor.SensorModel",
                 "resolution_px": {"width": 1280, "height": 720}, "frame_rate_hz": frame_rate_hz},
    )


def _write_log(trace: VisionTrace, path: Path) -> None:
    try:
        from sim.detection_log import write_detection_log
    except ImportError:
        manifest = trace.as_dict()
        manifest.pop("schema", None)
        manifest.pop("observations", None)
        lines = [{"schema": "zephyr-detection-log-1", **manifest}]
        lines.extend(observation.as_dict() for observation in trace.observations)
        rendered = "\n".join(json.dumps(item, sort_keys=True, allow_nan=False,
                                            separators=(",", ":")) for item in lines) + "\n"
        path.write_text(rendered, encoding="utf-8")
    else:
        write_detection_log(trace, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--noise-std", type=float, required=True)
    parser.add_argument("--bias", nargs=3, type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--mode", choices=("independent", "burst"), default="independent")
    parser.add_argument("--burst-start", type=float)
    parser.add_argument("--burst-end", type=float)
    parser.add_argument("--latency-mean", type=float, required=True)
    parser.add_argument("--latency-std", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trajectory", choices=("circle", "static"), default="circle")
    args = parser.parse_args(argv)
    if args.mode == "independent" and (args.burst_start is not None or args.burst_end is not None):
        parser.error("--burst-start and --burst-end require --mode burst")
    if args.mode == "burst" and args.dropout != 0:
        parser.error("--dropout must be zero in burst mode")
    burst_start = 0.0 if args.burst_start is None else args.burst_start
    burst_end = 1.0 if args.burst_end is None else args.burst_end
    try:
        config = SensorConfig(noise_std_m=args.noise_std, bias_m=tuple(args.bias),
                              dropout_probability=args.dropout,
                              burst_start_probability=burst_start,
                              burst_end_probability=burst_end, mode=args.mode)
        trace = make_fixture(n_frames=args.frames, frame_rate_hz=args.rate, sensor_config=config,
                             seed=args.seed, latency_mean_s=args.latency_mean,
                             latency_std_s=args.latency_std, trajectory=args.trajectory)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_log(trace, args.output)
    params = _fixture_parameters(sensor_config=config, n_frames=args.frames,
                                 frame_rate_hz=args.rate, seed=args.seed,
                                 latency_mean_s=args.latency_mean,
                                 latency_std_s=args.latency_std, trajectory=args.trajectory)
    sidecar = args.output.with_suffix(".params.json")
    sidecar.write_text(
        json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
