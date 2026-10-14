#!/usr/bin/env python3
"""Validate and normalize a Zephyr Mesh hardware-profile document."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from sim.hardware import (
    HARDWARE_PROFILE_SCHEMA,
    HARDWARE_PROFILE_SCHEMA_V2,
    HardwareProfile,
    RotorcraftProfile,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="Zephyr hardware profile JSON file")
    parser.add_argument("--output", type=Path, help="optional normalized JSON output path")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() == args.profile.resolve():
            raise ValueError("output must differ from the input profile")
        source = json.loads(args.profile.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise ValueError("hardware profile must be a JSON object")
        schema_name = "hardware_profile_schema_v2.json" if source.get("schema") == HARDWARE_PROFILE_SCHEMA_V2 else "hardware_profile_schema.json"
        schema_path = Path(__file__).parents[1] / "sim" / schema_name
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(source)
        if source.get("schema") == HARDWARE_PROFILE_SCHEMA:
            profile = HardwareProfile.from_dict(source)
        elif source.get("schema") == HARDWARE_PROFILE_SCHEMA_V2:
            profile = RotorcraftProfile.from_dict(source)
        else:
            raise ValueError(f"unsupported hardware profile schema: {source.get('schema')!r}")
        rendered = json.dumps(profile.as_document(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        if isinstance(profile, RotorcraftProfile):
            print(f"hover_speed_rad_s: {profile.hover_speed_rad_s():.9g}")
            print(f"hover_thrust_per_rotor_n: {profile.airframe.mass_kg * 9.81 / profile.rotor.count:.9g}")
            print(f"thrust_to_weight: {profile.thrust_to_weight():.9g}")
            print(f"max_vertical_acceleration_mps2: {profile.thrust_to_weight() * 9.81 - 9.81:.9g}")
    except (OSError, ValueError, OverflowError, json.JSONDecodeError, ValidationError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
