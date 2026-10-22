#!/usr/bin/env python3
"""Regenerate the checked-in S7 presentation golden file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desktop.presentation import present_run


def main(argv: list[str] | None = None) -> int:
    """Write or check the canonical presentation output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "runs/s7-swarm/run.json")
    parser.add_argument("--output", type=Path, default=ROOT / "tools/design/golden/s7_presentation.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    run = json.loads(args.input.read_text(encoding="utf-8"))
    expected = json.dumps(present_run(run), indent=2, sort_keys=True) + "\n"
    if args.check:
        return 0 if args.output.exists() and args.output.read_text(encoding="utf-8") == expected else 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
