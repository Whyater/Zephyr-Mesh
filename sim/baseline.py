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


_V2_SCENARIOS = {
    "vertical": {
        "total_time": 7.0,
        "initial_position": (0.0, 0.0, 0.0),
        "target_position": (0.0, 0.0, 3.0),
        "target_yaw_rad": None,
    },
    "lateral": {
        "total_time": 10.0,
        "initial_position": (0.0, 0.0, 1.0),
        "target_position": (1.0, 0.0, 1.0),
        "target_yaw_rad": None,
    },
    "yaw": {
        "total_time": 3.0,
        "initial_position": (0.0, 0.0, 1.0),
        "target_position": (0.0, 0.0, 1.0),
        "target_yaw_rad": float(np.pi / 2.0),
    },
}


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
        # Each row holds the state after the step, so label it with the time
        # at the end of the step.  The initial state at t = 0 is the config.
        time = (step + 1) * config.dt
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


def run_scenario_v2(name: str, dt: float = 0.01) -> dict[str, np.ndarray]:
    """Run one of the additive S0 v2 scenarios.

    The historical :func:`run_baseline` path is deliberately separate so its
    floating point operation order and output remain unchanged.
    """
    if name not in _V2_SCENARIOS:
        raise ValueError(f"unknown v2 scenario {name!r}; expected one of {tuple(_V2_SCENARIOS)}")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    spec = _V2_SCENARIOS[name]
    steps_float = spec["total_time"] / dt
    steps = int(round(steps_float))
    if not np.isclose(steps, steps_float):
        raise ValueError("scenario duration must be an integer multiple of dt")
    drone = SixDOFInterceptor(init_pos=spec["initial_position"])
    controller = GeometricFlightController()
    target = np.asarray(spec["target_position"], dtype=float)
    rows = []
    for step in range(steps):
        time = (step + 1) * dt
        force_body, torque_body = controller.update_control(
            drone, target, dt, target_yaw_rad=spec["target_yaw_rad"]
        )
        drone.step_physics(force_body, torque_body, dt)
        rows.append((
            time, *drone.position, *drone.velocity, *drone.quaternion,
            *drone.angular_velocity,
        ))
    values = np.asarray(rows, dtype=float)
    names = (
        "time", "x", "y", "z", "vx", "vy", "vz",
        "qw", "qx", "qy", "qz", "p", "q", "r",
    )
    return {name: values[:, index] for index, name in enumerate(names)}


def scenario_v2_spec(name: str, dt: float = 0.01) -> dict:
    """Return the serializable configuration used by :func:`run_scenario_v2`."""
    if name not in _V2_SCENARIOS:
        raise ValueError(f"unknown v2 scenario {name!r}")
    spec = dict(_V2_SCENARIOS[name])
    spec["name"] = name
    spec["dt"] = float(dt)
    return spec


# Real, current limits of this baseline and the models it runs.  Keep this list
# in step with the code; an empty list would claim there are none.
KNOWN_LIMITS = (
    "Sampled-controller limit: attitude rate damping of 18 per second needs dt < 2/18 = 0.111 s "
    "(necessary condition; the controller rejects larger steps). Sideways steps are tested to converge "
    "at dt 0.01 s for four targets and four starting yaw angles, not at every dt below the limit.",
    "No heading setpoint: the controller damps yaw rate but does not hold or command a yaw angle.",
    "No ground plane: the body can descend below z = 0 without contact.",
    "No integral action: a steady thrust shortfall leaves a small steady position error.",
    "Default inertia (0.0023, 0.0023, 0.0040 kg m^2) is a fixed estimate for a 450 g 5-inch frame and "
    "is not derived from mass. Hardware-profile vehicles use a point-mass plus plate estimate. Neither is measured.",
    "Drag coefficient, reference area, and controller gains are synthetic scenario inputs, not measured values.",
    "This artifact only runs the vertical step to (0, 0, 3) m; sideways and yawed cases are covered by tests, not by this run.",
    "RK4 stages use the attitude quaternion without renormalizing it inside a step (it is renormalized after each step). "
    "The effect is small: the measured convergence ratio is still about 16 per halving of dt.",
)


def _portable(path) -> str:
    """Return *path* relative to the repository root when it lies inside it.

    Absolute machine paths (home folders, temporary checkouts) are not useful
    evidence in a shared repository, so paths inside the repo are recorded
    from its root.  Paths outside the repo are kept as given.
    """
    repo = Path(__file__).resolve().parents[1]
    candidate = Path(os.path.abspath(path))
    for base in (repo, Path(os.path.abspath(Path(__file__).parents[1]))):
        for probe in (candidate, candidate.parent.resolve() / candidate.name):
            try:
                relative = probe.relative_to(base)
            except ValueError:
                continue
            return relative.as_posix() or "."
    return str(path)


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
    argv = list(sys.argv)
    executable = _portable(sys.executable)
    invocation = [executable, *argv]
    command_text = command or " ".join(shlex.quote(arg) for arg in invocation)
    cwd = _portable(os.getcwd())
    replay_command = command_text if cwd == "." else f"cd {shlex.quote(cwd)} && {command_text}"
    target = np.asarray(config.target_position)
    predicted_vz = 3.0 * target[2] * config.dt
    actual_vz = None if telemetry is None else float(telemetry["vz"][0])
    # At t=0 the drone is stationary, so drag is exactly zero and the
    # controller's initial vertical acceleration is 3*target_z m/s^2.  The
    # first-step prediction is therefore a limiting-case check, not a claim
    # that the controlled, drag-inclusive step is exactly Euler.
    check_tolerance = max(1e-5, 0.02 * abs(predicted_vz))
    return {
        "schema": "zephyr-s0-baseline-1",
        "config": asdict(config),
        "command": command_text,
        "executable": executable,
        "argv": argv,
        "invocation": invocation,
        "replay_command": replay_command,
        "cwd": cwd,
        "output_dir": _portable(output_dir) if output_dir else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_revision": revision,
        "git_tree_state": tree_state,
        "git_status": git_status,
        "random_seed": None,
        "known_issues": list(KNOWN_LIMITS),
        "status": "post-S2 baseline measurement hook; not flight performance",
        "physics_revision": "S2 true RK4 with quadratic drag; inertia-scaled attitude gains with body-frame tilt error",
        "time_semantics": "time is the end of the step that produced each row's state; the t = 0 state is config.initial_position at rest",
        "prediction": "Initial vertical acceleration is 9 m/s^2, so vz at the first recorded sample should be 3*target_z*dt.",
        "independent_check": {
            "name": "initial vertical acceleration limiting-case check",
            "expected_vz_first_sample": float(predicted_vz),
            "actual_vz_first_sample": actual_vz,
            "absolute_tolerance": float(check_tolerance),
            "method": "stationary initial state implies zero drag; compare first-step result to a*dt",
            "passed": bool(actual_vz is not None and abs(actual_vz - predicted_vz) <= check_tolerance),
        },
        "artifact_status": "frozen historical fixture",
        "provenance_note": (
            "The recorded revision and tree state describe the run that wrote this file; uncommitted "
            "changes listed in git_status were present. No match to any later worktree is claimed. "
            "Paths inside the repository are recorded relative to its root."
        ),
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
    manifest["artifacts"] = {"telemetry.csv": {"path": _portable(csv_path), "sha256": _sha256(csv_path)}}
    manifest_path = destination / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    (destination / "manifest.sha256").write_text(f"{_sha256(manifest_path)}  manifest.json\n")
    return csv_path


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/s0-baseline")
    print(write_baseline(output))
