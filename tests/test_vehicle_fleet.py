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
