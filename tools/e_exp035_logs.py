"""EXP-035: mission log integrity checks on canonical simulated missions.

Usage (repo root):

    PYTHONPATH=. ./.venv/bin/python tools/e_exp035_logs.py --output runs/e-exp035-logs

For every example scenario in sim/missions/examples/ and both adapters of its
kind (point_mass and physics, or the offboard pair for the formation), the
script runs the mission with the first EXP-035 seed and checks three things:

1. Regeneration: running the same scenario twice gives byte-identical log files.
2. Schema: the written log validates against sim/missions/mission_log_schema.json.
3. Replay: the independent reader (sim/missions/replay_reader.py) recomputes the
   metrics from the log alone.  Metrics the log carries in full ("exact") must
   agree with the mission's values; the others are listed as not applicable,
   with the reader's reason and its log-only estimate beside the mission value.

Everything is simulated.  Logs are written under logs/ in the output directory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from sim.missions.api import Scenario
from sim.missions.engine import run_mission
from sim.missions.log import load_log, validate_log, write_log
from sim.missions.replay_reader import compare
from tools.c_common import REPO_ROOT, argv_or_default, prepare_output, write_json, write_manifest
from tools.e_common import MISSION_LIMITS, MISSION_SOURCES, replay_command, seeds_for, write_rows

EXAMPLES_DIR = REPO_ROOT / "sim" / "missions" / "examples"
ONBOARD_ADAPTERS = ("point_mass", "physics")
OFFBOARD_ADAPTERS = ("point_mass_offboard", "physics_offboard")
SOURCES = MISSION_SOURCES + (
    "sim/missions/log.py",
    "sim/missions/mission_log_schema.json",
    "sim/missions/replay_reader.py",
    "sim/missions/hover.py",
    "sim/missions/tag.py",
    "sim/missions/sweep.py",
    "sim/missions/relay.py",
    "sim/missions/monitor.py",
    "sim/missions/formation.py",
    "tools/e_exp035_logs.py",
)
PREDICTION = (
    "all pass after at most one fix; the most likely mismatch is a metric "
    "computed on 100 Hz state that the 20 Hz log cannot reproduce."
)
PREDICTION_SOURCE = "pre-registered before the runs (EXP-035 section of the private pre-registration)"


def _git_revision() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unrecorded"
    return out or "unrecorded"


def canonical_cases(seed: int) -> list[dict]:
    """One case per example scenario and adapter, in a fixed order."""
    cases = []
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        base = Scenario.load(path)
        adapters = OFFBOARD_ADAPTERS if base.adapter in OFFBOARD_ADAPTERS else ONBOARD_ADAPTERS
        for adapter in adapters:
            cases.append({
                "name": f"{path.stem}_{adapter}",
                "example": path.name,
                "scenario": base.replace(seed=seed, adapter=adapter),
            })
    return cases


def check_case(case: dict, logs_dir: Path, revision: str) -> tuple[dict, list[dict]]:
    """Run one case twice, validate it and replay it; return a summary and comparison rows."""
    scenario = case["scenario"]
    log_path = logs_dir / f"{case['name']}.json"
    first = run_mission(scenario, log=True, code_revision=revision)
    write_log(log_path, first.log)
    with tempfile.TemporaryDirectory() as scratch:
        again_path = Path(scratch) / "again.json"
        second = run_mission(scenario, log=True, code_revision=revision)
        write_log(again_path, second.log)
        identical = again_path.read_bytes() == log_path.read_bytes()
    loaded = load_log(log_path)
    try:
        validate_log(loaded)
        schema_error = None
    except ValueError as exc:
        schema_error = str(exc)
    rows = []
    for row in compare(loaded):
        rows.append({
            "case": case["name"],
            "metric": row["metric"],
            "exact": bool(row["exact"]),
            "agrees": row["agrees"],
            "mission_value": json.dumps(row["mission_value"], sort_keys=True),
            "reader_value": json.dumps(row["reader_value"], sort_keys=True),
            "reason": row["reason"],
        })
    exact_rows = [r for r in rows if r["exact"]]
    disagreements = [r for r in exact_rows if r["agrees"] is not True]
    summary = {
        "case": case["name"],
        "example": case["example"],
        "mission": scenario.mission,
        "adapter": scenario.adapter,
        "seed": int(scenario.seed),
        "end_reason": first.end_reason,
        "regenerates_byte_identical": bool(identical),
        "schema_valid": schema_error is None,
        "schema_error": schema_error,
        "exact_metrics": len(exact_rows),
        "exact_agree": len(exact_rows) - len(disagreements),
        "exact_disagreements": [{"metric": r["metric"], "mission_value": r["mission_value"],
                                 "reader_value": r["reader_value"]} for r in disagreements],
        "not_applicable": sorted(r["metric"] for r in rows if not r["exact"]),
        "passed": bool(identical and schema_error is None and not disagreements),
    }
    return summary, rows


def run(output: Path, argv: list[str]) -> dict:
    output = prepare_output(Path(output))
    logs_dir = output / "logs"
    logs_dir.mkdir()
    start = time.perf_counter()
    seed = seeds_for(35, 1)[0]
    revision = _git_revision()
    summaries = []
    all_rows = []
    for case in canonical_cases(seed):
        summary, rows = check_case(case, logs_dir, revision)
        summaries.append(summary)
        all_rows.extend(rows)
    write_rows(output / "integrity.csv", all_rows,
               first_columns=("case", "metric", "exact", "agrees", "mission_value", "reader_value", "reason"))
    result = {
        "experiment": "EXP-035",
        "simulated": True,
        "seed": seed,
        "cases": summaries,
        "all_passed": all(s["passed"] for s in summaries),
        "prediction": PREDICTION,
        "prediction_source": PREDICTION_SOURCE,
        "notes": [
            "Logs record frames at 20 Hz; metrics computed by the missions on 100 Hz state, on "
            "per-packet radio timing or on internal estimator state are declared not applicable, "
            "with the reader's reason and its log-only estimate in integrity.csv.",
            "Logs carry the git revision they were made with, so they regenerate byte for byte only "
            "from the same commit.",
        ],
        "limits": list(MISSION_LIMITS),
    }
    write_json(output / "summary.json", result)
    log_names = sorted(f"logs/{p.name}" for p in logs_dir.iterdir())
    write_manifest(output, schema="zephyr-exp035-1", argv=argv,
                   command=replay_command("tools/e_exp035_logs.py", argv),
                   config={"seed": seed, "examples": sorted(p.name for p in EXAMPLES_DIR.glob("*.json"))},
                   seeds=[seed], outputs=["integrity.csv", "summary.json", *log_names],
                   limits=list(MISSION_LIMITS), runtime_s=time.perf_counter() - start, n_processes=1,
                   hashed_sources=list(SOURCES))
    return result


def main(argv: list[str] | None = None) -> int:
    argv = argv_or_default(argv)
    parser = argparse.ArgumentParser(description="EXP-035 mission log integrity checks (simulated).")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.output, argv)
    for case in result["cases"]:
        print(f"{case['case']}: identical={case['regenerates_byte_identical']} schema={case['schema_valid']} "
              f"exact {case['exact_agree']}/{case['exact_metrics']}")
    print("all passed" if result["all_passed"] else "NOT all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
