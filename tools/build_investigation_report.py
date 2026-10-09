#!/usr/bin/env python3
"""Create a portable evidence report from an S6 investigation sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from desktop.evidence import EvidenceError, build_evidence_report, write_evidence_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="zephyr-s6-sweep-1 JSON file")
    parser.add_argument("--output", type=Path, required=True, help="portable report JSON path")
    args = parser.parse_args(argv)
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
