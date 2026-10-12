#!/usr/bin/env python3
"""Build the macOS native replay fixture from the canonical S7 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs" / "s7-swarm" / "run.json"
DEFAULT_OUTPUT = ROOT / "macos" / "ZephyrMeshApp" / "Sources" / "ZephyrMeshApp" / "Resources" / "demo_swarm.json"
sys.path.insert(0, str(ROOT))
from desktop.presentation import present_run


def agent_projection(agent: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only canonical telemetry fields needed by the native scene."""
    return {
        "id": agent["agent_id"],
        "position": agent.get("position_m"),
        "velocity": agent.get("velocity_mps"),
        "profile_id": agent.get("profile_id"),
        "active": agent.get("active"),
        "neighbor_count": agent.get("neighbor_count"),
        "min_neighbor_distance_m": agent.get("min_neighbor_distance_m"),
        "constraint_flags": agent.get("constraint_flags", []),
        "target_estimate_m": agent.get("target_estimate_m"),
        "fused_target_m": agent.get("fused_target_m"),
        "target_source_time_s": agent.get("target_source_time_s"),
        "target_age_s": agent.get("target_age_s"),
    }


def build(run: Mapping[str, Any]) -> dict[str, Any]:
    """Build the native projection and embed the shared presentation."""
    presentation = present_run(run)
    steps = run["steps"]
    first = steps[0]
    return {
        "schema": "zephyr-s7-native-fixture-3",
        "run_id": run["scenario_id"],
        "seed": run["seed"],
        "command_authority": run["command_authority"],
        "failsafe": run["failsafe"],
        "dt_s": run["dt_s"],
        "evidence_boundary": run["evidence_boundary"],
        "generation_command": run["generation_command"],
        "status": run["status"],
        "about": presentation["about"],
        "drones": [agent_projection(agent) for agent in first["agents"]],
        "frames": [
            {
                "step_index": step["step_index"],
                "time": step["time_s"],
                "target_position": step.get("target_position_m"),
                "active_count": step.get("active_count"),
                "drones": [agent_projection(agent) for agent in step["agents"]],
                "presentation": presentation["frames"][index],
            }
            for index, step in enumerate(steps)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Build a fixture JSON file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    run = json.loads(args.input.read_text(encoding="utf-8"))
    fixture = build(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
