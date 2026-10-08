"""Generate the checked-in S7 50-agent fixture and manifest.

This is a reproducibility helper. It does not contact hardware and it does not
turn the scenario parameters into measured flight performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.link import LinkConfig
from sim.scenarios import make_ring_swarm_config, scenario_manifest
from sim.swarm import SwarmSimulator


def generate(output_dir: Path, *, agent_count: int = 50, steps: int = 6, seed: int = 17) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_ring_swarm_config(
        agent_count=agent_count,
        seed=seed,
        target_link=LinkConfig(delay_s=0.08, loss_model="independent", loss_probability=0.08),
        neighbor_link=LinkConfig(delay_s=0.04, loss_model="independent", loss_probability=0.03),
    )
    manifest = scenario_manifest(config, scenario_id=f"s7-ring-{agent_count}-seed-{seed}")
    manifest["reproducibility"] = {
        "command": f"PYTHONPATH=. ./.venv/bin/python tools/generate_s7_fixture.py --output {output_dir}",
        "module": "sim.scenarios.make_ring_swarm_config + sim.swarm.SwarmSimulator",
        "steps": steps,
    }
    run = SwarmSimulator(config).run(steps).as_dict()
    run["scenario_id"] = manifest["scenario_id"]
    run["generation_command"] = manifest["reproducibility"]["command"]
    run["status"] = "synthetic deterministic replay; not flight performance"
    run["command_authority"] = "replay only"
    run["failsafe"] = "synthetic constraints only; live failsafe unavailable"
    manifest["command_authority"] = "replay only"
    manifest["failsafe"] = "synthetic constraints only; live failsafe unavailable"
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/s7-swarm"))
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    generate(args.output, agent_count=args.agents, steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
