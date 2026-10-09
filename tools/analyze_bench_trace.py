#!/usr/bin/env python3
"""Summarize a validated Zephyr Mesh bench trace without fitting a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from sim.bench_metrics import summarize_bench_trace
from sim.bench_trace import BenchTrace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="zephyr-bench-trace-1 JSON file")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() == args.trace.resolve():
            raise ValueError("output must differ from the input trace")
        source_bytes = args.trace.read_bytes()
        source = json.loads(source_bytes)
        schema = json.loads((Path(__file__).parents[1] / "sim" / "bench_trace_schema.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(source)
        summary = summarize_bench_trace(BenchTrace.from_dict(source))
        summary["provenance"]["input_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, ValueError, OverflowError, json.JSONDecodeError, ValidationError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
