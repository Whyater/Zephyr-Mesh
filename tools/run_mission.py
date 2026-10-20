"""Run one simulated mission from a scenario file and save its log and metrics.

    PYTHONPATH=. python tools/run_mission.py sim/missions/examples/tag_straight.json --out out_dir
    PYTHONPATH=. python tools/run_mission.py scenario.json --adapter physics --out out_dir

Writes ``mission_log.json`` (schema ``zephyr-mission-log-1``) and
``metrics.json`` into ``--out`` (which must be new or empty).  Everything is
simulated: link loss patterns are synthetic and vehicle parameters are
literature-derived.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from sim.missions.api import ADAPTER_KINDS, Scenario
from sim.missions.engine import run_mission
from sim.missions.log import write_log


def _code_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                             check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unrecorded"
    return rev or "unrecorded"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one simulated mission (headless).")
    parser.add_argument("scenario", help="scenario JSON file")
    parser.add_argument("--out", required=True, help="new or empty output directory")
    parser.add_argument("--adapter", choices=ADAPTER_KINDS, help="override the scenario's adapter")
    parser.add_argument("--seed", type=int, help="override the scenario's seed")
    parser.add_argument("--code-revision", help="revision string to record (default: git HEAD)")
    args = parser.parse_args(argv)

    scenario = Scenario.load(args.scenario)
    changes = {}
    if args.adapter:
        changes["adapter"] = args.adapter
    if args.seed is not None:
        changes["seed"] = args.seed
    if changes:
        scenario = scenario.replace(**changes)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    revision = args.code_revision or _code_revision()
    result = run_mission(scenario, log=True, code_revision=revision)
    write_log(out / "mission_log.json", result.log)
    (out / "metrics.json").write_text(
        json.dumps(result.log["metrics"], indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"end_reason": result.end_reason, "steps": result.steps,
                      "metrics": result.log["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
