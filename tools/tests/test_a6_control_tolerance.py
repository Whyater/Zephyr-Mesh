"""Real simulator checks for the EXP-021 delay and loss harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim.controller import GeometricFlightController
from sim.hardware import load_hardware_profile
from sim.physics import control_tolerance
from sim.physics.control_tolerance import _controller_for, reference_position, run_episode
from sim.vehicle import ActuatedQuadrotor
import tools.a_run_control_tolerance as run_control
from tools.a_run_control_tolerance import build_cells, load_config, run_sweep


PROFILE = "sim/physics/profiles/crazyflie_class.json"
PROFILES = (
    "sim/physics/profiles/esp_fly_class.json",
    PROFILE,
)
DT = control_tolerance.CONTROL_DT_S


def _vehicle(arm: str, delay: float, loss: float, seed: int = 0):
    profile = load_hardware_profile(PROFILE)
    vehicle = ActuatedQuadrotor(
        profile,
        initial_position=(0.0, 0.0, 1.0),
        controller=GeometricFlightController(),
        substeps=1,
    )
    vehicle.actuator.speed_rad_s[:] = profile.hover_speed_rad_s()
    vehicle.controller = _controller_for(
        arm, delay, loss, np.random.default_rng(seed), vehicle.drone
    )
    return vehicle


def _trajectory(arm: str, reference: str, delay: float, loss: float, *, seconds: float = 3.0):
    vehicle = _vehicle(arm, delay, loss)
    states = []
    for index in range(int(round(seconds / DT))):
        vehicle.step(reference_position(reference, index * DT), DT)
        states.append(
            (
                vehicle.drone.position.copy(),
                vehicle.drone.velocity.copy(),
                vehicle.drone.quaternion.copy(),
                vehicle.drone.angular_velocity.copy(),
            )
        )
    return states


def test_config_grid_and_common_seed_list():
    config = load_config()
    assert config["schema"].startswith("zephyr-a6")
    assert config["seeds"] == list(range(30))
    cells = build_cells(False)
    assert cells
    assert len(cells) == 178
    assert all(cell["loss_probability"] == 0 for cell in cells if cell["arm"] == "c")
    assert {cell["profile"] for cell in cells} == {
        "sim/physics/profiles/esp_fly_class.json",
        PROFILE,
    }
    assert config["pilot"]["jobs"]
    assert config["pilot"]["seeds"] == [0]


def test_dcrit_schema_records_grid_and_does_not_duplicate_loss_maps():
    schema_path = Path(__file__).parents[2] / "sim" / "physics" / "d_crit_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert {"schema", "pilot", "episode_duration_s", "effective_delay_grid_s", "profiles"} <= set(
        schema["required"]
    )
    zero_loss = schema["$defs"]["zero_loss_dcrit"]
    assert "effective_delay_grid_s" in zero_loss["required"]
    assert zero_loss["properties"]["resolution_s"]["const"] == pytest.approx(0.01)
    arm_b = schema["$defs"]["profile"]["properties"]["arm_b"]["properties"]
    assert "p_gt_0" in arm_b
    assert "p>0" not in arm_b
    assert "smallest_unstable_grid_delay_s" not in arm_b
    assert "unstable_rate_interval_at_d_crit" not in arm_b


@pytest.mark.parametrize("arm", ["a", "b", "c"])
def test_zero_delay_zero_loss_matches_plain_actuated_quadrotor(arm):
    wrapped = _trajectory(arm, "step", 0.0, 0.0)
    profile = load_hardware_profile(PROFILE)
    plain = ActuatedQuadrotor(
        profile,
        initial_position=(0.0, 0.0, 1.0),
        controller=GeometricFlightController(),
        substeps=1,
    )
    plain.actuator.speed_rad_s[:] = profile.hover_speed_rad_s()
    for index, expected in enumerate(wrapped):
        plain.step(reference_position("step", index * DT), DT)
        actual = (plain.drone.position, plain.drone.velocity, plain.drone.quaternion, plain.drone.angular_velocity)
        for got, want in zip(actual, expected):
            assert np.array_equal(got, want), f"arm {arm}, step {index}"


def test_arm_a_ramp_lag_follows_hand_slope():
    for delay in (0.0, 0.2, 0.4):
        vehicle = _vehicle("a", delay, 0.0)
        lag = []
        for index in range(800):
            vehicle.step(reference_position("ramp", index * DT), DT)
            if (index + 1) * DT >= 6.0:
                lag.append(reference_position("ramp", (index + 1) * DT)[0] - vehicle.drone.position[0])
        observed = float(np.mean(lag))
        expected = 0.5 * (0.75 + delay)
        assert observed == pytest.approx(expected, rel=0.15)


@pytest.mark.parametrize("profile", PROFILES)
def test_arm_a_worst_case_never_unstable(profile):
    for seed in (0, 1, 2):
        result = run_episode(profile, "a", "step", 0.5, 0.5, seed, duration_s=8.0)
        assert result["unstable"] is False


@pytest.mark.parametrize("profile", PROFILES)
def test_arm_b_stability_boundary(profile):
    stable = run_episode(profile, "b", "step", 0.1, 0.0, 0, duration_s=8.0)
    unstable = run_episode(profile, "b", "step", 0.5, 0.0, 0, duration_s=8.0)
    assert stable["unstable"] is False
    assert unstable["unstable"] is True


@pytest.mark.parametrize("profile", PROFILES)
def test_slice_c_stability_boundary(profile):
    stable = run_episode(profile, "c", "step", 0.0, 0.0, 0, duration_s=8.0)
    unstable = run_episode(profile, "c", "step", 0.08, 0.0, 0, duration_s=8.0)
    assert stable["unstable"] is False
    assert unstable["unstable"] is True


def test_setpoint_delay_uses_floor_of_50_hz_reference_grid():
    received = []

    class SpyController(GeometricFlightController):
        def update_control(self, drone, target_position, dt, **kwargs):
            received.append(np.asarray(target_position, dtype=float).copy())
            return super().update_control(drone, target_position, dt, **kwargs)

    profile = load_hardware_profile(PROFILE)
    vehicle = ActuatedQuadrotor(
        profile,
        initial_position=(0.0, 0.0, 1.0),
        controller=GeometricFlightController(),
        substeps=1,
    )
    vehicle.actuator.speed_rad_s[:] = profile.hover_speed_rad_s()
    delay = 0.05
    vehicle.controller = _controller_for("a", delay, 0.0, np.random.default_rng(3), vehicle.drone)
    vehicle.controller.inner = SpyController()
    for index in range(300):
        vehicle.step(reference_position("ramp", index * DT), DT)
    assert len(received) == 300
    for index, target in enumerate(received):
        now = index * DT
        sample_index = int(np.floor((now - delay + 1e-9) / control_tolerance.SETPOINT_SAMPLE_S))
        expected_time = max(0, sample_index) * control_tolerance.SETPOINT_SAMPLE_S
        assert np.array_equal(target, reference_position("ramp", expected_time)), index


def test_run_episode_exception_propagates(monkeypatch):
    def broken_update(self, drone, target_position, dt, **kwargs):
        raise RuntimeError("intentional controller bug")

    monkeypatch.setattr(GeometricFlightController, "update_control", broken_update)
    with pytest.raises(RuntimeError, match="intentional controller bug"):
        run_episode(PROFILE, "a", "step", 0.0, 0.0, 0, duration_s=0.2)


def test_run_episode_is_deterministic():
    first = run_episode(PROFILE, "b", "step", 0.1, 0.3, 7, duration_s=3.0)
    second = run_episode(PROFILE, "b", "step", 0.1, 0.3, 7, duration_s=3.0)
    assert first == second


def test_pilot_outputs_are_deterministic_and_worker_independent(tmp_path):
    outputs = []
    for suffix, workers in (("one", 1), ("two", 1), ("parallel", 2)):
        output = tmp_path / suffix
        run_sweep(output, pilot=True, workers=workers)
        outputs.append(output)
    for name in ("per_run.csv", "summary.json", "d_crit.json", "a6_control_tolerance.png"):
        baseline = (outputs[0] / name).read_bytes()
        assert all((output / name).read_bytes() == baseline for output in outputs[1:]), name
    assert (outputs[0] / "manifest.json").exists()
    summary = json.loads((outputs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["median_interval_method"] == (
        "order-statistic (binomial) interval, sim.s6_recovery.median_interval"
    )
    assert "bootstrap_seed" not in summary
    dcrit = json.loads((outputs[0] / "d_crit.json").read_text(encoding="utf-8"))
    assert dcrit["pilot"] is True
    assert dcrit["episode_duration_s"] == pytest.approx(6.0)
    assert dcrit["effective_delay_grid_s"] == pytest.approx(0.01)
    for profile, entry in dcrit["profiles"].items():
        assert entry["slice_c"]["hand_attitude_whole_state"]
        assert entry["arm_b"]["p0"]["resolution_s"] == pytest.approx(0.01)
        assert entry["slice_c"]["resolution_s"] == pytest.approx(0.01)
        assert entry["arm_b"]["p0"]["episode_duration_s"] == pytest.approx(6.0)
        assert entry["slice_c"]["episode_duration_s"] == pytest.approx(6.0)
    header = (outputs[0] / "per_run.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "effective_delay_s" in header


def test_nonpilot_single_cell_checkpoint_and_resume(monkeypatch, tmp_path):
    """A full-grid cell id must come from the cell, not result metadata."""
    cell = {
        "profile": PROFILE,
        "profile_id": "crazyflie-class",
        "reference": "step",
        "arm": "c",
        "delay_s": 0.0,
        "loss_probability": 0.0,
    }
    calls = []

    def fake_episode(profile, arm, reference, delay, loss, seed, *, duration_s=None, scoring_windows=None):
        calls.append((profile, arm, reference, delay, loss, seed))
        return {
            "duration_s": 0.2,
            "rms_error_m": 0.1,
            "max_error_m": 0.2,
            "unstable": False,
            "unstable_reason": None,
            "unstable_time_s": None,
        }

    class InlinePool:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, function, values):
            return [function(value) for value in values]

    monkeypatch.setattr(run_control, "build_cells", lambda pilot=False: [dict(cell)])
    monkeypatch.setattr(run_control, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(control_tolerance, "run_episode", fake_episode)
    monkeypatch.setattr(
        run_control,
        "_dcrit",
        lambda rows, pilot: {
            "schema": "zephyr-mesh-exp-021-d-crit-1",
            "pilot": bool(pilot),
            "episode_duration_s": 20.0,
            "effective_delay_grid_s": 0.01,
            "profiles": {},
        },
    )
    def fake_plot(output, summaries, dcrit):
        (output / "a6_control_tolerance.png").write_bytes(b"test image")

    monkeypatch.setattr(run_control, "_write_plot", fake_plot)

    output = tmp_path / "single-cell"
    first = run_control.run_sweep(output, pilot=False, workers=1)
    checkpoint = next((output / "cells").glob("*.json"))
    assert checkpoint.is_file()
    assert (output / "per_run.csv").is_file()
    assert (output / "summary.json").is_file()
    assert first["rows"] == 1
    assert len(calls) == 1

    second = run_control.run_sweep(output, pilot=False, workers=1, resume=True)
    assert second["rows"] == 1
    assert len(calls) == 1


def test_nonempty_requires_resume(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "old").write_text("x")
    with pytest.raises(FileExistsError):
        run_sweep(output, pilot=True)
