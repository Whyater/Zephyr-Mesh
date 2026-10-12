#!/usr/bin/env python3
"""Build the macOS native replay fixture from the canonical S7 run."""
from __future__ import annotations

import argparse
import hashlib
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


# Fields a viewer might expect that the S7 run schema does not carry. The native
# About view lists them so a reader never mistakes a blank for a measured zero.
UNAVAILABLE = (
    "Live radio link",
    "Camera or vision input",
    "Battery state",
    "Flight controller commands",
    "Measured flight performance",
)


def profile_projection(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the hardware profile parts shown under the inspector Details."""
    return {
        "profile_id": profile["profile_id"],
        "status": profile.get("status"),
        "motor_count": profile["motor_count"],
        "arm_length_m": profile["arm_length_m"],
        "total_mass_kg": profile["total_mass_kg"],
        "motor_part_id": profile["motor"]["part_id"],
        "propeller_part_id": profile["propeller"]["part_id"],
        "propeller_diameter_m": profile["propeller"]["diameter_m"],
    }


def keep_out_volumes(step: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return the keep-out spheres the simulator enforced for one step.

    ``sim.swarm`` appends a target keep-out sphere (center at the target,
    radius from the scenario constraints) to the configured spheres on every
    step; the run snapshot only records the configured ones, so the target
    sphere is restored here from the manifest that produced the run.
    """
    volumes = [
        {"center_m": list(zone["center_m"]), "radius_m": zone["radius_m"], "label": zone["label"]}
        for zone in step.get("keep_out_spheres", [])
    ]
    radius = (manifest or {}).get("constraints", {}).get("target_keepout_radius_m")
    target = step.get("target_position_m")
    if radius is not None and target is not None:
        volumes.append({"center_m": list(target), "radius_m": radius, "label": "target_keepout"})
    return volumes


def run_name(run: Mapping[str, Any], agent_count: int) -> str:
    """Name the run in plain words, for example ``50-drone ring, seed 17``."""
    parts = str(run["scenario_id"]).split("-")
    formation = "ring" if "ring" in parts else "swarm"
    return f"{agent_count}-drone {formation}, seed {run['seed']}"


def build(run: Mapping[str, Any], manifest: Mapping[str, Any] | None = None, run_sha256: str = "") -> dict[str, Any]:
    """Build the native projection and embed the shared presentation."""
    presentation = present_run(run)
    steps = run["steps"]
    first = steps[0]
    return {
        "schema": "zephyr-s7-native-fixture-3",
        "run_id": run["scenario_id"],
        "run_name": run_name(run, len(first["agents"])),
        "source": {
            "run_schema": run["schema"],
            "run_sha256": run_sha256,
            "presentation_schema": presentation["schema"],
        },
        "unavailable": list(UNAVAILABLE),
        "profiles": [profile_projection(profile) for profile in run.get("agent_profiles", [])],
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
                "keep_out_volumes": keep_out_volumes(step, manifest),
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
    raw = args.input.read_bytes()
    run = json.loads(raw.decode("utf-8"))
    manifest_path = args.input.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    fixture = build(run, manifest, hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
