"""Shared plumbing for the cooperative swarm experiment scripts (tools/d_run_*.py).

Every script writes into a NEW directory and finishes with a manifest.json
that maps each data and figure file to its sha256. runtime_s, git_revision,
git_tree_state, platform and n_processes are volatile and are left out of
byte-identity checks; everything under "outputs" must regenerate byte for
byte from the same command.

All results are simulated. Nothing is measured.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path, PurePath

import numpy as np

from tools.c_common import (  # noqa: F401  (re-exported for the d_ scripts)
    REPO_ROOT,
    VOLATILE_MANIFEST_FIELDS,
    _git,
    argv_or_default,
    prepare_output,
    sha256_file,
    sha256_json,
    write_json,
)
from tools.c_common import seeds_from as _seeds_from

# One base per experiment so seed lists never overlap between experiments.
BASE_SEED = 20261015
EXPERIMENT_OFFSETS = {
    "fusion": 23,
    "tolerance": 24,
    "separation": 25,
    "dropout": 26,
    "decisions": 27,
    "consensus": 28,
    "s7-coop": 29,
}

SIMULATED_LIMITS = ("Simulated only: no drone, camera or radio hardware was used, and nothing here is measured.",)
MOVING_AGENT_LIMITS = (
    "Agents are kinematic point masses with speed and acceleration caps, not flight dynamics.",
    "Sensors are simulated Gaussian noise on the true target position; no camera or vision code is involved.",
)
RADIO_LIMITS = (
    "Radio parameters are literature-derived (sim/radio/params.py), with ranges; none are measured.",
)


def seeds_for(experiment: str, n: int) -> list[int]:
    """Seed list for one experiment: SeedSequence(BASE_SEED + offset).generate_state(n)."""
    return _seeds_from(BASE_SEED + EXPERIMENT_OFFSETS[experiment], n)


def default_processes() -> int:
    """Leave headroom for other sessions sharing the machine: at most half the cores, at least one."""
    return max(1, (os.cpu_count() or 2) // 2)


def write_manifest(output: Path, *, schema: str, argv: list[str], command: str, config: dict, seeds: list[int],
                   outputs: list[str], limits: list[str], runtime_s: float, n_processes: int,
                   hashed_sources: list[str], extra: dict | None = None) -> dict:
    """Write manifest.json. ``limits`` must describe this experiment; only the generic
    "simulated only" line is added in front. Use MOVING_AGENT_LIMITS and RADIO_LIMITS
    where they apply."""
    porcelain = _git(["status", "--porcelain"])
    manifest = {
        "schema": schema,
        "argv": list(argv),
        "replay_command": command,
        "config": config,
        "config_sha256": sha256_json(config),
        "seeds": [int(s) for s in seeds],
        "limits": list(SIMULATED_LIMITS) + list(limits),
        "code_sha256": {PurePath(p).as_posix(): sha256_file(REPO_ROOT / p) for p in hashed_sources},
        "git_revision": _git(["rev-parse", "HEAD"]),
        "git_tree_state": "unknown" if porcelain is None else ("dirty" if porcelain else "clean"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "n_processes": int(n_processes),
        "runtime_s": float(runtime_s),
        "outputs": {name: sha256_file(Path(output) / name) for name in outputs},
    }
    if extra:
        manifest.update(extra)
    write_json(Path(output) / "manifest.json", manifest)
    return manifest
