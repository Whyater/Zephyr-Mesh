"""Shared plumbing for the radio experiment scripts (EXP-013 to EXP-018).

Every script writes its outputs into a NEW directory, then a manifest.json
that maps each data and figure file to its sha256.  The fields runtime_s,
git_revision, git_tree_state, platform and n_processes are volatile and are
left out of byte-identity checks; everything listed under "outputs" must
regenerate byte for byte from the same command.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VOLATILE_MANIFEST_FIELDS = ("runtime_s", "git_revision", "git_tree_state", "platform", "n_processes")
SIMULATED_LIMITS = (
    "Simulated only: no ESP32 or drone hardware was used.",
    "Radio parameters are literature-derived (sim/radio/params.py), with ranges; none are measured.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL, cwd=REPO_ROOT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_output(output: Path) -> Path:
    """Refuse to touch an existing results directory; sim/runs/ directories are never edited."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} already exists and is not empty; write to a new directory")
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def default_processes() -> int:
    """Wave 1 runs share the machine: use at most a third of the cores."""
    return max(1, (os.cpu_count() or 3) // 3)


def write_manifest(output: Path, *, schema: str, argv: list[str], command: str, config: dict, seeds: list[int],
                   outputs: list[str], limits: list[str], runtime_s: float, n_processes: int,
                   hashed_sources: list[str], extra: dict | None = None) -> dict:
    porcelain = _git(["status", "--porcelain"])
    manifest = {
        "schema": schema,
        "argv": list(argv),
        "replay_command": command,
        "config": config,
        "config_sha256": sha256_json(config),
        "seeds": [int(s) for s in seeds],
        "limits": list(SIMULATED_LIMITS) + list(limits),
        "code_sha256": {p: sha256_file(REPO_ROOT / p) for p in hashed_sources},
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


def seeds_from(base_seed: int, n: int) -> list[int]:
    return [int(v) for v in np.random.SeedSequence(base_seed).generate_state(n)]


def argv_or_default(argv: list[str] | None) -> list[str]:
    return list(sys.argv[1:] if argv is None else argv)
