"""Optional actuator-coupled fleet fixture.

This module runs one :class:`ActuatedQuadrotor` per named agent so actuator,
battery, controller, and 6-DOF telemetry can be inspected together.  It is a
separate layer from :mod:`sim.swarm`: no radio links, formation estimator, or
S7 point-mass frames are consumed or produced here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

from sim.hardware import HardwareProfile
from sim.vehicle import ActuatedQuadrotor, VehicleStep


def _target(value: Iterable[float], *, agent_id: str) -> tuple[float, float, float]:
    try:
        array = np.asarray(tuple(value), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target position for {agent_id} must be a finite 3-vector") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"target position for {agent_id} must be a finite 3-vector")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class FleetVehicleStep:
    """One named agent's actuator-coupled vehicle observation."""

    agent_id: str
    profile_id: str
    vehicle: VehicleStep

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "profile_id": self.profile_id,
            "vehicle": self.vehicle.as_dict(),
        }


@dataclass(frozen=True)
class VehicleFleetStep:
    """Deterministic snapshot from the optional actuator-coupled fleet."""

    step_index: int
    time_s: float
    vehicles: tuple[FleetVehicleStep, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "time_s": self.time_s,
            "vehicles": [vehicle.as_dict() for vehicle in self.vehicles],
        }


class ActuatedVehicleFleet:
    """Run independent actuator-coupled vehicles under external targets.

    Target positions are supplied by the caller.  This deliberate boundary
    keeps actuator telemetry useful without implying that this fixture models
    swarm links, formation coordination, or measured flight performance.
    """

    def __init__(
        self,
        profiles: Mapping[str, HardwareProfile],
        *,
        initial_positions: Mapping[str, Iterable[float]] | None = None,
        drag_coefficient: float = 1.3,
        cross_sectional_area: float = 0.05,
        wind_velocity: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> None:
        if not isinstance(profiles, Mapping) or not profiles:
            raise ValueError("profiles must be a non-empty mapping")
        if any(not isinstance(agent_id, str) or not agent_id for agent_id in profiles):
            raise ValueError("profile ids must be non-empty strings")
        for agent_id, profile in profiles.items():
            if not isinstance(profile, HardwareProfile):
                raise ValueError(f"profile for {agent_id} must be a HardwareProfile")
            if profile.motor_count != 4:
                raise ValueError(f"profile for {agent_id} must describe four motors")
        self._profiles = {agent_id: profiles[agent_id] for agent_id in sorted(profiles)}
        positions = {} if initial_positions is None else initial_positions
        if not isinstance(positions, Mapping):
            raise ValueError("initial_positions must be a mapping")
        if set(positions) - set(self._profiles):
            raise ValueError("initial_positions contains an unknown agent")
        self._vehicles: dict[str, ActuatedQuadrotor] = {}
        for agent_id, profile in self._profiles.items():
            initial = positions.get(agent_id, (0.0, 0.0, 0.0))
            try:
                position = _target(initial, agent_id=agent_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(str(exc)) from exc
            self._vehicles[agent_id] = ActuatedQuadrotor(
                profile,
                initial_position=position,
                drag_coefficient=drag_coefficient,
                cross_sectional_area=cross_sectional_area,
                wind_velocity=wind_velocity,
            )
        self.time_s = 0.0
        self.step_index = 0

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(self._vehicles)

    def model_provenance(self) -> dict[str, Any]:
        """Return the explicit boundary for this optional fleet layer."""
        return {
            "status": "synthetic",
            "model": "independent_actuator_coupled_vehicle_fleet",
            "actuator_coupled": True,
            "coordination_model": "none",
            "radio_model": "none",
            "agents": {
                agent_id: {
                    "profile_id": self._profiles[agent_id].profile_id,
                    "profile_status": self._profiles[agent_id].status,
                }
                for agent_id in self.agent_ids
            },
            "evidence_boundary": (
                "Synthetic actuator, battery, controller, and 6-DOF telemetry; "
                "targets are externally supplied and no swarm or measured-flight "
                "claim is made."
            ),
        }

    def _validated_targets(self, targets: Mapping[str, Iterable[float]]) -> dict[str, tuple[float, float, float]]:
        if not isinstance(targets, Mapping) or set(targets) != set(self._vehicles):
            raise ValueError("targets must contain exactly one finite position for each agent")
        return {
            agent_id: _target(targets[agent_id], agent_id=agent_id)
            for agent_id in self.agent_ids
        }

    def step(self, targets: Mapping[str, Iterable[float]], dt_s: float) -> VehicleFleetStep:
        """Advance every independent vehicle once in stable agent order."""
        dt_s = float(dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        validated = self._validated_targets(targets)
        observations = tuple(
            FleetVehicleStep(
                agent_id=agent_id,
                profile_id=self._profiles[agent_id].profile_id,
                vehicle=self._vehicles[agent_id].step(validated[agent_id], dt_s),
            )
            for agent_id in self.agent_ids
        )
        result = VehicleFleetStep(self.step_index, self.time_s, observations)
        self.time_s += dt_s
        self.step_index += 1
        return result

    def reset(self, positions: Mapping[str, Iterable[float]] | None = None) -> None:
        """Reset all vehicles and optional positions, time, and step index."""
        positions = {} if positions is None else positions
        if not isinstance(positions, Mapping) or set(positions) - set(self._vehicles):
            raise ValueError("positions contains an unknown agent")
        validated = {
            agent_id: _target(value, agent_id=agent_id)
            for agent_id, value in positions.items()
        }
        for agent_id, vehicle in self._vehicles.items():
            position = validated.get(agent_id)
            vehicle.reset(position)
        self.time_s = 0.0
        self.step_index = 0
