"""Reusable Stage 7 scenario factories and machine-readable manifests."""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

from sim.hardware import HardwareProfile, default_hardware_profile as _default_hardware_profile
from sim.link import LinkConfig
from sim.swarm import AgentSpec, SwarmConfig


def default_hardware_profile(index: int) -> HardwareProfile:
    """Create a uniquely identified, replaceable quad profile.

    The numeric values are transparent scenario defaults.  The identifiers are
    intentionally unique so a later parts registry can substitute measured
    motor and propeller records per vehicle without changing swarm logic.
    """
    if isinstance(index, bool) or int(index) != index or int(index) < 0:
        raise ValueError("index must be a nonnegative integer")
    index = int(index)
    return _default_hardware_profile(f"drone-{index + 1:03d}")


def make_ring_swarm_config(
    agent_count: int = 50,
    *,
    radius_m: float = 6.0,
    altitude_m: float = 1.5,
    seed: int = 0,
    target_link: LinkConfig | None = None,
    neighbor_link: LinkConfig | None = None,
) -> SwarmConfig:
    """Create a deterministic ring of uniquely identified agents.

    ``agent_count=50`` is a stress fixture for manifests and coordination
    bookkeeping.  It is not a claim that the current controller or radio can
    fly fifty vehicles.
    """
    if isinstance(agent_count, bool) or int(agent_count) != agent_count or int(agent_count) < 1:
        raise ValueError("agent_count must be a positive integer")
    agent_count = int(agent_count)
    radius_m = float(radius_m)
    altitude_m = float(altitude_m)
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    if not math.isfinite(altitude_m) or altitude_m < 0.0:
        raise ValueError("altitude_m must be finite and nonnegative")
    target = (0.0, 0.0, altitude_m)
    agents = []
    for index in range(agent_count):
        angle = 2.0 * math.pi * index / agent_count
        offset = (radius_m * math.cos(angle), radius_m * math.sin(angle), 0.0)
        agents.append(AgentSpec(
            agent_id=f"drone-{index + 1:03d}",
            position_m=(offset[0], offset[1], altitude_m),
            slot_offset_m=offset,
            hardware_profile=default_hardware_profile(index),
        ))
    return SwarmConfig(
        agents=tuple(agents),
        target_position_m=target,
        dt_s=0.05,
        min_separation_m=0.35,
        target_keepout_radius_m=0.75,
        target_link=target_link or LinkConfig(loss_model="none"),
        neighbor_link=neighbor_link or LinkConfig(loss_model="none"),
        seed=seed,
    )


def scenario_manifest(config: SwarmConfig, *, scenario_id: str = "s7-ring") -> dict[str, Any]:
    """Return a portable manifest with assumptions and per-agent hardware IDs."""
    if not isinstance(config, SwarmConfig):
        raise ValueError("config must be a SwarmConfig")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    agents = []
    for agent in config.agents:
        agents.append({
            "agent_id": agent.agent_id,
            "position_m": list(agent.position_m),
            "velocity_mps": list(agent.velocity_mps),
            "slot_offset_m": list(agent.slot_offset_m),
            "hardware_profile": None if agent.hardware_profile is None else agent.hardware_profile.as_dict(),
        })
    return {
        "schema": "zephyr-s7-scenario-manifest-1",
        "scenario_id": scenario_id,
        "seed": config.seed,
        "agent_count": len(config.agents),
        "target_position_m": list(config.target_position_m),
        "target_velocity_mps": list(config.target_velocity_mps),
        "dt_s": config.dt_s,
        "constraints": {
            "min_separation_m": config.min_separation_m,
            "target_keepout_radius_m": config.target_keepout_radius_m,
            "ground_plane_z_m": config.ground_plane_z_m,
        },
        "link_models": {
            "target": asdict(config.target_link),
            "neighbor": asdict(config.neighbor_link),
            "target_overrides": {
                agent_id: asdict(link) for agent_id, link in config.target_link_overrides
            },
        },
        "agents": agents,
        "evidence_boundary": (
            "Coordination and hardware-parameter fixture only; numeric motor and propeller "
            "values are scenario inputs, not measured flight performance."
        ),
    }
