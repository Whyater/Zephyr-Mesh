"""Write a deterministic cooperative S7 simulation to a new directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from sim.coop.simulator import CoopSwarmConfig, CoopSwarmSimulator, to_replay_v1


def _round(value):
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, list):
        return [_round(x) for x in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def _write(path: Path, value: dict) -> None:
    text = json.dumps(_round(value), sort_keys=True, indent=1, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_hash(root: Path, tool: Path) -> str:
    h = hashlib.sha256()
    for path in sorted((root / "sim" / "coop").glob("*.py")) + [tool]:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def generate(output: Path, *, agents: int = 4, seed: int = 23, duration: float = 20.0) -> None:
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = CoopSwarmConfig(n_agents=agents, seed=seed, duration_s=duration)
    run = CoopSwarmSimulator(config).run().as_dict()
    replay = to_replay_v1(run)
    run_path = output / "run.json"
    replay_path = output / "replay_v1.json"
    _write(run_path, run)
    _write(replay_path, replay)
    root = Path(__file__).resolve().parents[1]
    tool_path = Path(__file__).resolve()
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True, encoding="utf-8").stdout.strip()
        tree_state = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=False, capture_output=True, text=True, encoding="utf-8").stdout.strip()
    except OSError:
        revision, tree_state = "unavailable", "unavailable"
    target_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0])
    phase_seeds = [[seed, 3000 + i] for i in range(1, agents)]
    manifest = {
        "schema": "zephyr-s7-coop-manifest-1",
        "config": run["config"],
        "seeds": {"run": seed, "target_derivation": {"sequence": [seed, 1], "state_index": 0, "value": target_seed}, "sensors": [[seed, 100 + i] for i in range(agents)], "observation_phases": phase_seeds, "links": {"base": [seed, 1000], "route_order": "sorted directed node pairs"}},
        "code_sha256": _code_hash(root, tool_path),
        "outputs": {"run.json": _sha(run_path), "replay_v1.json": _sha(replay_path)},
        "limits": "Simulated only: kinematic point-mass agents, abstract links, literature-derived thresholds, and no hardware.",
        "generation_command": f"PYTHONPATH=. {{python}} tools/d_generate_s7_coop.py --output {{out}} --agents {agents} --seed {seed} --duration {duration:g}",
        "volatile": {"runtime_s": time.perf_counter() - started, "git_revision": revision, "git_tree_state": tree_state, "platform": platform.platform()},
    }
    _write(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    generate(args.output, agents=args.agents, seed=args.seed, duration=args.duration)


if __name__ == "__main__":
    main()
