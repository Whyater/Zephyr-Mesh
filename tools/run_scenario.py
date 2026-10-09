#!/usr/bin/env python3
"""Run a bounded S7 JSON scenario and write a canonical synthetic replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Make ``python tools/run_scenario.py`` work from any current directory while
# keeping the module importable as ``tools.run_scenario``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.scenario_runner import (
    MAX_OUTPUT_BYTES,
    ScenarioConfigError,
    load_json_config,
    run_scenario,
    serialize_run,
    summary,
)


def _write_output(path: Path, data: bytes, *, config_path: Path) -> None:
    """Write only to a new regular file path, with a bounded payload."""
    resolved = path.resolve()
    if resolved == config_path.resolve():
        raise ScenarioConfigError("output must differ from config")
    immutable_replay = Path(__file__).resolve().parents[1] / "runs" / "s7-swarm" / "run.json"
    if resolved == immutable_replay:
        raise ScenarioConfigError("the checked-in S7 replay is immutable")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ScenarioConfigError("output must be a regular file when it already exists")
    if len(data) > MAX_OUTPUT_BYTES:
        raise ScenarioConfigError(f"output exceeds {MAX_OUTPUT_BYTES // (1024 * 1024)} MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON scenario config")
    parser.add_argument("--output", type=Path, help="canonical JSON output path")
    parser.add_argument("--summary", action="store_true", help="print a compact JSON summary")
    args = parser.parse_args(argv)
    try:
        raw = load_json_config(args.config)
        document = run_scenario(raw)
        data = serialize_run(document)
        if args.output is not None:
            _write_output(args.output, data, config_path=args.config)
        elif not args.summary:
            sys.stdout.buffer.write(data)
        if args.summary:
            rendered = json.dumps(summary(document), sort_keys=True, separators=(",", ":"))
            print(rendered)
    except (OSError, ScenarioConfigError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
