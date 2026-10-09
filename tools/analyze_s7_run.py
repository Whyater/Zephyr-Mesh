#!/usr/bin/env python3
"""Summarize a validated S7 replay without claiming physical performance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from sim.s7_metrics import summarize_s7_run

MAX_RUN_BYTES = 16 * 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="zephyr-s7-swarm-run-1 JSON file")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() == args.run.resolve():
            raise ValueError("output must differ from input")
        if args.run.stat().st_size > MAX_RUN_BYTES:
            raise ValueError(f"input exceeds {MAX_RUN_BYTES // (1024 * 1024)} MiB")
        raw = args.run.read_bytes()
        def reject_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result
        document = json.loads(raw, object_pairs_hook=reject_duplicate_keys, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid constant {value}")))
        schema = json.loads((Path(__file__).parents[1] / "sim" / "s7_run_schema.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(document)
        summary = summarize_s7_run(document)
        summary["provenance"]["input_sha256"] = hashlib.sha256(raw).hexdigest()
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
