#!/usr/bin/env python3
"""Summarize a validated Zephyr Mesh vision trace without fitting a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sim.vision_metrics import summarize_vision_trace
from sim.vision_trace import VisionTrace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="zephyr-vision-trace-1 JSON file")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() == args.trace.resolve():
            raise ValueError("output must differ from the input trace")
        source_bytes = args.trace.read_bytes()
        summary = summarize_vision_trace(VisionTrace.from_dict(json.loads(source_bytes)))
        summary["provenance"]["input_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
