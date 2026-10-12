#!/usr/bin/env python3
"""Create a portable evidence report from an S6 investigation sweep.

With ``--write-sweep PATH`` it instead regenerates the sweep JSON itself
(the committed ``sim/investigation_example.json``) from ``sim.investigation``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from sim.investigation import run_sweep
from desktop.evidence import EvidenceError, build_evidence_report, write_evidence_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", help="zephyr-s6-sweep-1 JSON file")
    parser.add_argument("--output", type=Path, help="portable report JSON path")
    parser.add_argument("--write-sweep", type=Path, metavar="PATH",
                        help="regenerate the default sweep JSON at PATH and exit")
    args = parser.parse_args(argv)
    if args.write_sweep is not None:
        args.write_sweep.write_text(json.dumps(run_sweep(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    if args.source is None or args.output is None:
        parser.error("source and --output are required unless --write-sweep is used")
    try:
        if args.output.resolve() == args.source.resolve():
            raise EvidenceError("output must differ from source")
        report = build_evidence_report(args.source, "investigation")
        schema = json.loads((Path(__file__).parents[1] / "desktop" / "evidence_report_schema.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(report)
        write_evidence_report(args.output, report)
    except (EvidenceError, OSError, ValidationError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
