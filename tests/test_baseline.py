import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from sim.baseline import BaselineConfig, build_manifest, run_baseline, write_baseline
from sim.controller import GeometricFlightController
from sim.drone import SixDOFInterceptor


def test_baseline_is_deterministic_and_has_expected_samples():
    config = BaselineConfig(dt=0.1, total_time=0.3)
    first = run_baseline(config)
    second = run_baseline(config)
    assert len(first["time"]) == 3
    assert np.array_equal(first["z"], second["z"])
    assert np.all(np.isfinite(first["z"]))


def test_manifest_records_s2_physics_and_lists_real_known_limits():
    """Field presence plus the known limits; it cannot detect controller bugs.

    Sideways and yawed closed-loop behaviour is checked in test_physics.py.
    """
    manifest = build_manifest(BaselineConfig())
    assert manifest["schema"] == "zephyr-s0-baseline-1"
    limits = " ".join(manifest["known_issues"])
    for phrase in ("dt < 2/18", "No ground plane", "450 g", "vertical step", "No heading setpoint"):
        assert phrase in limits, phrase
    assert manifest["config"]["dt"] == 0.01
    assert manifest["executable"]
    assert manifest["invocation"][0] == manifest["executable"]
    assert manifest["replay_command"].endswith(manifest["command"])
    # Paths inside the repo are portable (relative), never this machine's absolute path.
    assert not manifest["cwd"].startswith("/")
    assert set(manifest["git_status"]) == {"staged", "unstaged", "entries"}
    check = manifest["independent_check"]
    assert check["name"].endswith("limiting-case check")
    assert check["method"].startswith("stationary initial state")
    assert check["passed"] is False  # no telemetry was supplied to build_manifest


def test_writer_emits_machine_readable_outputs(tmp_path):
    write_baseline(tmp_path, BaselineConfig(dt=0.1, total_time=0.3))
    assert (tmp_path / "telemetry.csv").read_text().startswith("time,x,y,z")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["config"]["total_time"] == 0.3
    assert manifest["independent_check"]["passed"]
    assert (tmp_path / "manifest.sha256").exists()


def test_time_column_labels_the_state_it_sits_beside():
    """Limiting case: from rest with acceleration a, speed is v = a * t.

    The controller's first vertical acceleration is Kp_z * 3 m = 9 m/s^2 and
    drag is zero at rest, so on the first row vz / time must be about 9.  If
    the row were labelled with the step's start time (t = 0) this fails.
    """
    telemetry = run_baseline(BaselineConfig(dt=0.01, total_time=0.05))
    assert telemetry["time"][0] == pytest.approx(0.01)
    assert telemetry["vz"][0] / telemetry["time"][0] == pytest.approx(9.0, rel=1e-3)
    np.testing.assert_allclose(np.diff(telemetry["time"]), 0.01)


def test_retained_default_baseline_matches_fresh_rerun():
    """The checked-in S0 telemetry remains reproducible after code changes."""
    artifact = Path(__file__).parents[1] / "runs" / "s0-baseline" / "telemetry.csv"
    retained = np.genfromtxt(artifact, delimiter=",", names=True)
    fresh = run_baseline(BaselineConfig())
    assert tuple(retained.dtype.names) == tuple(fresh)
    for name in fresh:
        assert np.array_equal(retained[name], fresh[name]), name

    manifest_path = artifact.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert manifest["artifacts"]["telemetry.csv"]["sha256"] == digest

def test_controller_rejects_invalid_inputs_before_producing_forces():
    controller = GeometricFlightController()
    drone = SixDOFInterceptor()
    with pytest.raises(ValueError):
        controller.update_control(drone, [0.0, 0.0, 1.0], 0.0)
    with pytest.raises(ValueError):
        controller.update_control(drone, [0.0, np.nan, 1.0], 0.01)

def test_controller_never_commands_negative_rotor_thrust_when_inverted():
    controller = GeometricFlightController()
    drone = SixDOFInterceptor()
    drone.quaternion[:] = [0.0, 1.0, 0.0, 0.0]  # 180 degrees about x
    force_body, _ = controller.update_control(drone, [0.0, 0.0, 3.0], 0.01)
    assert force_body[2] == 0.0
