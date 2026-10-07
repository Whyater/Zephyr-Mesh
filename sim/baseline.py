"""Reproducible S0 hover baseline and run manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sim.controller import GeometricFlightController
from sim.drone import SixDOFInterceptor


@dataclass(frozen=True)
class BaselineConfig:
    dt: float = 0.01
    total_time: float = 7.0
    target_position: tuple[float, float, float] = (0.0, 0.0, 3.0)
    initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0)


def run_baseline(config: BaselineConfig = BaselineConfig()) -> dict[str, np.ndarray]:
    """Run the current controller and physics implementation without plotting."""
    if config.dt <= 0 or config.total_time <= 0:
        raise ValueError("dt and total_time must be positive")
    steps_float = config.total_time / config.dt
    steps = int(round(steps_float))
    if not np.isclose(steps, steps_float):
        raise ValueError("total_time must be an integer multiple of dt")

    drone = SixDOFInterceptor(init_pos=config.initial_position)
    controller = GeometricFlightController()
    target = np.asarray(config.target_position, dtype=float)
    rows = []
    for step in range(steps):
        time = step * config.dt
        force_body, torque_body = controller.update_control(drone, target, config.dt)
        drone.step_physics(force_body, torque_body, config.dt)
        rows.append((
            time,
            *drone.position,
            *drone.velocity,
            *drone.quaternion,
            *drone.angular_velocity,
        ))
    values = np.asarray(rows, dtype=float)
    names = (
        "time", "x", "y", "z", "vx", "vy", "vz",
        "qw", "qx", "qy", "qz", "p", "q", "r",
    )
    return {name: values[:, index] for index, name in enumerate(names)}


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_revision() -> tuple[str | None, str, dict]:
    repo = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
        staged = bool(subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--cached", "--name-only"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip())
        unstaged = bool(subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--name-only"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()) or any(line.startswith("??") for line in status)
        return revision, "dirty" if status else "clean", {
            "staged": staged,
            "unstaged": unstaged,
            "entries": status,
        }
    except (OSError, subprocess.CalledProcessError):
        return None, "unavailable", {"staged": None, "unstaged": None, "entries": []}


def build_manifest(config: BaselineConfig, *, command=None, output_dir=None, telemetry=None) -> dict:
    """Return run metadata needed to reproduce an S0 result."""
    revision, tree_state, git_status = _repo_revision()
    repo = Path(__file__).resolve().parents[1]
    argv = list(sys.argv)
    invocation = [sys.executable, *argv]
    command_text = command or " ".join(shlex.quote(arg) for arg in invocation)
    replay_command = f"cd {shlex.quote(str(repo))} && {command_text}"
    target = np.asarray(config.target_position)
    predicted_vz = 3.0 * target[2] * config.dt
    actual_vz = None if telemetry is None else float(telemetry["vz"][0])
    return {
        "schema": "zephyr-s0-baseline-1",
        "config": asdict(config),
        "command": command_text,
        "executable": sys.executable,
        "argv": argv,
        "invocation": invocation,
        "replay_command": replay_command,
        "cwd": os.getcwd(),
        "output_dir": str(Path(output_dir).resolve()) if output_dir else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_revision": revision,
        "git_tree_state": tree_state,
        "git_status": git_status,
        "random_seed": None,
        "known_issues": [
            "step_physics is labeled RK4 but currently behaves as first-order Euler",
            "calculate_drag is not applied by the current physics step",
        ],
        "status": "baseline measurement hook; not flight performance",
        "prediction": "Initial vertical acceleration is 9 m/s^2, so vz at the first recorded sample should be 3*target_z*dt.",
        "independent_check": {"name": "initial vertical acceleration hand check", "expected_vz_first_sample": float(predicted_vz), "actual_vz_first_sample": actual_vz, "passed": bool(actual_vz is not None and np.isclose(actual_vz, predicted_vz))},
    }


def write_baseline(output_dir: str | Path, config: BaselineConfig = BaselineConfig(), *, telemetry=None, command=None) -> Path:
    """Write ``telemetry.csv`` and ``manifest.json`` into *output_dir*."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if telemetry is None:
        telemetry = run_baseline(config)
    csv_path = destination / "telemetry.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        names = tuple(telemetry)
        writer.writerow(names)
        writer.writerows(zip(*(telemetry[name] for name in names)))
    manifest = build_manifest(config, command=command, output_dir=destination, telemetry=telemetry)
    manifest["artifacts"] = {"telemetry.csv": {"path": str(csv_path.resolve()), "sha256": _sha256(csv_path)}}
    manifest_path = destination / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    (destination / "manifest.sha256").write_text(f"{_sha256(manifest_path)}  manifest.json\n")
    return csv_path


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/s0-baseline")
    print(write_baseline(output))
