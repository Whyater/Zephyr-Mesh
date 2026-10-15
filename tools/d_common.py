"""Shared plumbing for the cooperative swarm experiment scripts (tools/d_run_*.py).

Every script writes into a NEW directory and finishes with a manifest.json
that maps each data and figure file to its sha256. runtime_s, git_revision,
git_tree_state, platform and n_processes are volatile and are left out of
byte-identity checks; everything under "outputs" must regenerate byte for
byte from the same command.

All results are simulated: kinematic point-mass agents, simulated sensors,
abstract links or Run C's literature-derived radio model. Nothing is measured.
"""
from __future__ import annotations

import os

from tools.c_common import (  # noqa: F401  (re-exported for the d_ scripts)
    VOLATILE_MANIFEST_FIELDS,
    argv_or_default,
    prepare_output,
    sha256_file,
    sha256_json,
    write_json,
    write_manifest,
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

SIMULATED_LIMITS = (
    "Agents are kinematic point masses with speed and acceleration caps, not flight dynamics.",
    "Sensors are simulated Gaussian noise on the true target position; no camera or vision code is involved.",
)


def seeds_for(experiment: str, n: int) -> list[int]:
    """Seed list for one experiment: SeedSequence(BASE_SEED + offset).generate_state(n)."""
    return _seeds_from(BASE_SEED + EXPERIMENT_OFFSETS[experiment], n)


def default_processes() -> int:
    """Leave headroom for other sessions sharing the machine: at most half the cores, at least one."""
    return max(1, (os.cpu_count() or 2) // 2)
