"""Small, reproducible manifests for generated physics artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


VOLATILE_FIELDS = [
    "runtime_s",
    "git_revision",
    "git_tree_state",
    "platform",
    "python",
    "numpy",
    "n_processes",
]


def canonical_json_sha256(value) -> str:
    """Hash JSON using the canonical ordering used for configuration values."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_details(repo: Path) -> tuple[str | None, str]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return revision, "dirty" if status.strip() else "clean"
    except (OSError, subprocess.CalledProcessError):
        return None, "unavailable"


def _output_hashes(outputs, output_dir: Path | None) -> dict[str, str]:
    if outputs is None:
        return {}
    if isinstance(outputs, Mapping):
        pairs = outputs.items()
    else:
        paths = list(outputs)
        pairs = ((Path(path).name, path) for path in paths)
    result = {}
    for name, value in pairs:
        path = Path(value)
        # Callers pass either names relative to the output directory or paths
        # that already include it (for example a relative "sim/runs/x/data.csv").
        if not path.is_absolute() and output_dir is not None and not path.is_file():
            path = output_dir / path
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(name)] = file_sha256(path)
    return dict(sorted(result.items()))


def build_manifest(
    *,
    schema: str,
    argv: Sequence[str] | None = None,
    config: Mapping | None = None,
    seeds: Sequence | None = None,
    limits: Sequence[str] | None = None,
    outputs=None,
    output_dir: str | Path | None = None,
    replay_command: str | None = None,
    runtime_s: float | None = None,
    n_processes: int | None = None,
    repo: str | Path | None = None,
) -> dict:
    """Build a manifest without modifying repository state.

    ``outputs`` may be a mapping from artifact names to paths or an iterable
    of paths.  The manifest itself should be written after all other outputs,
    so it is not included in the digest map.
    """
    if not schema:
        raise ValueError("schema must be non-empty")
    config_dict = dict(config or {})
    argv_list = list(sys.argv[1:] if argv is None else argv)
    repo_path = Path(repo or Path(__file__).resolve().parents[2])
    revision, tree_state = _git_details(repo_path)
    if replay_command is None:
        replay_command = " ".join(shlex.quote(str(arg)) for arg in argv_list)
    manifest = {
        "schema": str(schema),
        "argv": argv_list,
        "git_revision": revision,
        "git_tree_state": tree_state,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "runtime_s": None if runtime_s is None else float(runtime_s),
        "config": config_dict,
        "config_sha256": canonical_json_sha256(config_dict),
        "seeds": list(seeds or []),
        "limits": [str(limit) for limit in (limits or [])],
        "outputs": _output_hashes(outputs, Path(output_dir) if output_dir else None),
        "replay_command": str(replay_command),
        "volatile_fields": list(VOLATILE_FIELDS),
        "n_processes": None if n_processes is None else int(n_processes),
    }
    return manifest


def write_manifest(path: str | Path, **kwargs) -> Path:
    """Build and write a sorted JSON manifest with a trailing newline."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if "output_dir" not in kwargs:
        kwargs["output_dir"] = destination.parent
    document = build_manifest(**kwargs)
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


__all__ = ["VOLATILE_FIELDS", "build_manifest", "canonical_json_sha256", "file_sha256", "write_manifest"]
