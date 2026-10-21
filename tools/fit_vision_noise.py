#!/usr/bin/env python3
"""Fit the sim's measurement noise parameters from one detection log with ground truth.

This fits one log to the sim's sensor model. It is not a camera model, and a
synthetic or fixture log says nothing about a real camera.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.vision_noise_fit import DROPOUT_MODES, fit_vision_noise
from sim.vision_trace import VisionTrace


def _load(path: Path, source_bytes: bytes) -> VisionTrace:
    if path.suffix.lower() == ".jsonl":
        # Imported here so this tool still imports while the reader is absent.
        from sim.detection_log import read_detection_log

        return read_detection_log(path)
    return VisionTrace.from_dict(json.loads(source_bytes.decode("utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="zephyr-detection-log-1 .jsonl file or zephyr-vision-trace-1 JSON file")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument("--subject", help="subject_id to fit (required when the log has more than one)")
    parser.add_argument("--dropout-mode", choices=DROPOUT_MODES, default="independent")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples; 0 gives normal-approximation intervals")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() == args.log.resolve():
            raise ValueError("output must differ from the input log")
        source_bytes = args.log.read_bytes()
        trace = _load(args.log, source_bytes)
        fit = fit_vision_noise(trace, subject_id=args.subject, dropout_mode=args.dropout_mode,
                               n_bootstrap=args.bootstrap, seed=args.seed)
        fit["input"]["input_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        rendered = json.dumps(fit, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, ValueError, OverflowError, UnicodeDecodeError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
