"""Check the latency tool's stated error on many synthetic recordings (no camera).

Regenerates the coverage and bias numbers quoted in sim/vision_latency.py:

    PYTHONPATH=. .venv/bin/python tools/vision_latency_calibration.py coverage --runs 300
    PYTHONPATH=. .venv/bin/python tools/vision_latency_calibration.py bias --runs 80

coverage: random setups (latency 0 to 300 ms, 15/30/60 fps, exposure 2 to 33 ms,
jitter 0/10/30 ms, 10 s each). Counts how often the stated error covers the
true mean latency. About 6 minutes for 300 runs on one core.

Not covered: the automatic strip box (the runs pass the true box), a display lag
that varies during a run, and the default 2000 bootstrap resamples (coverage
uses 1000, bias uses 200).

bias: the average signed error over random 30/60 fps setups, with no jitter.
"""
from __future__ import annotations

import argparse

import numpy as np

from sim.vision_latency import Settings, SynthConfig, analyze, synth_recording


def coverage(runs: int) -> None:
    rng = np.random.default_rng(2026)
    misses, ratios, errors = 0, [], []
    guard = guard_misses = 0
    for i in range(runs):
        cfg = SynthConfig(latency_s=float(rng.uniform(0, 0.3)), fps=float(rng.choice([15, 30, 60])),
                          exposure_s=float(rng.uniform(0.002, 0.033)),
                          latency_jitter_s=float(rng.choice([0, 0, 0.01, 0.03])),
                          noise_std=float(rng.uniform(2, 10)), duration_s=10.0, seed=1000 + i,
                          camera_phase_s=float(rng.uniform(0, 0.03)))
        rec = synth_recording(cfg)
        result = analyze(rec.frames, rec.t_read_s, rec.log,
                         Settings(display_lag_s=(0, 0), bootstrap=1000, roi=cfg.strip_box, seed=i))
        a = result["apparent_latency_s"]
        error = a["mean"] - float(np.mean(rec.true_latency_s))
        guarded = a["stated_error_parts"]["locking_guard"] > 0
        guard += guarded
        errors.append(error)
        ratios.append(abs(error) / a["stated_error"])
        if abs(error) > a["stated_error"]:
            misses += 1
            guard_misses += guarded
            print(f"miss: run {i}, {cfg.fps:.0f} fps, latency {1000 * cfg.latency_s:.0f} ms, "
                  f"error {1000 * error:+.2f} ms, stated {1000 * a['stated_error']:.2f} ms")
    print(f"runs {runs}, covered {runs - misses} ({100 * (runs - misses) / runs:.1f}%), "
          f"worst |error|/stated {max(ratios):.2f}, worst |error| {1000 * max(map(abs, errors)):.2f} ms, "
          f"locking guard on in {guard} runs ({guard_misses} of the misses)")


def bias(runs: int) -> None:
    rng = np.random.default_rng(7)
    errors, stated = [], []
    for i in range(runs):
        cfg = SynthConfig(latency_s=float(rng.uniform(0, 0.3)), fps=float(rng.choice([30, 60])),
                          exposure_s=float(rng.uniform(0.002, 0.033)), duration_s=10.0, seed=5000 + i,
                          camera_phase_s=float(rng.uniform(0, 0.03)))
        rec = synth_recording(cfg)
        result = analyze(rec.frames, rec.t_read_s, rec.log,
                         Settings(display_lag_s=(0, 0), bootstrap=200, roi=cfg.strip_box))
        errors.append(result["apparent_latency_s"]["mean"] - float(np.mean(rec.true_latency_s)))
        stated.append(result["apparent_latency_s"]["stated_error"])
    e = np.array(errors)
    print(f"runs {runs}, mean signed error {1000 * e.mean():+.3f} +/- {1000 * e.std() / np.sqrt(runs):.3f} ms, "
          f"error std {1000 * e.std():.3f} ms, mean stated error {1000 * np.mean(stated):.3f} ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["coverage", "bias"])
    parser.add_argument("--runs", type=int, default=300)
    args = parser.parse_args(argv)
    (coverage if args.check == "coverage" else bias)(args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
