"""Independent checks for the optional actuator-coupled fleet fixture."""

import numpy as np
import pytest

from sim.scenarios import default_hardware_profile
from sim.vehicle_fleet import ActuatedVehicleFleet


def make_fleet() -> ActuatedVehicleFleet:
    return ActuatedVehicleFleet(
        {"beta": default_hardware_profile(1), "alpha": default_hardware_profile(0)},
        initial_positions={"alpha": (-1.0, 0.0, 1.5), "beta": (1.0, 0.0, 1.5)},
    )


def test_fleet_returns_stable_per_agent_vehicle_telemetry():
    fleet = make_fleet()
    result = fleet.step({"beta": (1.5, 0.0, 1.5), "alpha": (-1.5, 0.0, 1.5)}, 0.01)
    assert [vehicle.agent_id for vehicle in result.vehicles] == ["alpha", "beta"]
    assert result.step_index == 0
    assert result.time_s == 0.0
    assert all(vehicle.vehicle.actuator.power_w >= 0.0 for vehicle in result.vehicles)
    assert all(np.all(np.isfinite(vehicle.vehicle.position_m)) for vehicle in result.vehicles)
    assert result.vehicles[0].profile_id != result.vehicles[1].profile_id


def test_fleet_provenance_is_actuator_coupled_without_swarm_or_radio_claims():
    metadata = make_fleet().model_provenance()
    assert metadata["status"] == "synthetic"
    assert metadata["actuator_coupled"] is True
    assert metadata["coordination_model"] == "none"
    assert metadata["radio_model"] == "none"
    assert set(metadata["agents"]) == {"alpha", "beta"}
    assert "no swarm" in metadata["evidence_boundary"]


def test_invalid_target_is_rejected_before_any_vehicle_moves():
    fleet = make_fleet()
    with pytest.raises(ValueError):
        fleet.step({"alpha": (0.0, 0.0, 1.5), "beta": (np.nan, 0.0, 1.5)}, 0.01)
    assert fleet.step_index == 0
    assert fleet.time_s == 0.0


def test_fleet_reset_restores_time_and_vehicle_state():
    fleet = make_fleet()
    fleet.step({"alpha": (0.0, 0.0, 1.5), "beta": (0.0, 0.0, 1.5)}, 0.01)
    fleet.reset()
    assert fleet.step_index == 0
    assert fleet.time_s == 0.0
    result = fleet.step({"alpha": (-1.0, 0.0, 1.5), "beta": (1.0, 0.0, 1.5)}, 0.01)
    assert all(vehicle.vehicle.actuator.total_thrust_n > 0.0 for vehicle in result.vehicles)


def test_fleet_vehicles_move_toward_their_targets_with_hand_computed_bounds():
    """Each vehicle gets a 0.5 m sideways step from its start.

    Hand calculation for the horizontal loop (Kp = 2, Kd = 1.5): natural
    frequency w = sqrt(2) = 1.414 rad/s, damping ratio z = 0.53, damped
    frequency wd = w * sqrt(1 - z**2) = 1.20 rad/s.  The step error is
    0.5 * exp(-z w t) * (cos(wd t) + z / sqrt(1 - z**2) * sin(wd t)), which
    is 0.22 m at t = 1 s and 0.0003 m at t = 10 s.  Attitude lag, motor lag
    and drag can only slow the start, so at 1 s the error must be at least the
    hand value minus a 0.02 m tolerance (0.20 m), and well below the untouched
    0.5 m (window 0.20 to 0.45 m).  The 10 s bound is 0.05 m to allow
    millimetres of altitude droop.
    """
    fleet = make_fleet()
    targets = {"alpha": (-0.5, 0.0, 1.5), "beta": (1.0, 0.5, 1.5)}
    dt = 0.01
    history = [fleet.step(targets, dt) for _ in range(1000)]

    def distance(step, agent):
        vehicle = next(item for item in step.vehicles if item.agent_id == agent)
        return float(np.linalg.norm(np.asarray(vehicle.vehicle.position_m) - targets[agent]))

    for agent in ("alpha", "beta"):
        assert distance(history[0], agent) == pytest.approx(0.5, abs=1e-3)
        assert 0.20 < distance(history[99], agent) < 0.45
        assert distance(history[-1], agent) < 0.05
