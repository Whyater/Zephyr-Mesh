#!/usr/bin/env python3
"""Validate and normalize a Zephyr Mesh hardware-profile document."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from sim.hardware import HardwareProfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="zephyr-hardware-profile-1 JSON file")
    parser.add_argument("--output", type=Path, help="optional normalized JSON output path")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() == args.profile.resolve():
            raise ValueError("output must differ from the input profile")
        source = json.loads(args.profile.read_text(encoding="utf-8"))
        schema_path = Path(__file__).parents[1] / "sim" / "hardware_profile_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(source)
        profile = HardwareProfile.from_dict(source)
        rendered = json.dumps(profile.as_document(), indent=2, sort_keys=True, allow_nan=False) + "\n"
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
