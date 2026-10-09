"""Independent checks for the Stage 7 multi-agent coordination fixture."""

import numpy as np
import pytest

from sim.link import LinkConfig
from sim.swarm import AgentSpec, KeepOutSphere, SwarmConfig, SwarmSimulator


def two_agent_config(**overrides):
    values = dict(
        agents=(
            AgentSpec("a", (-0.8, 0.0, 1.5), slot_offset_m=(-0.8, 0.0, 0.0)),
            AgentSpec("b", (0.8, 0.0, 1.5), slot_offset_m=(0.8, 0.0, 0.0)),
        ),
        target_position_m=(0.0, 0.0, 1.5),
        target_keepout_radius_m=0.4,
        min_separation_m=0.5,
        seed=12,
    )
    values.update(overrides)
    return SwarmConfig(**values)


def test_seeded_run_is_repeatable_and_serializable():
    link = LinkConfig(delay_s=0.08, delay_jitter_s=0.01, loss_probability=0.25)
    config = two_agent_config(target_link=link, neighbor_link=link)
    first = SwarmSimulator(config).run(18).as_dict()
    second = SwarmSimulator(config).run(18).as_dict()
    assert first == second
    assert first["schema"] == "zephyr-s7-swarm-run-1"
    assert len(first["steps"]) == 18
    assert first["link_events"]


def test_model_provenance_keeps_point_mass_separate_from_actuator_and_energy():
    simulator = SwarmSimulator(two_agent_config())
    metadata = simulator.model_provenance()
    assert metadata["status"] == "synthetic"
    assert metadata["model"] == "bounded_point_mass_coordination"
    assert metadata["actuator_coupled"] is False
    assert set(metadata["agents"]) == {"a", "b"}
    for agent in metadata["agents"].values():
        assert agent["actuator_model"] == "profile declaration only"
        assert agent["energy_telemetry"] == "not simulated"
        assert agent["declared_battery_capacity_wh"] > 0.0
    assert "not simulated" in metadata["evidence_boundary"]


def test_target_link_loss_is_explicit_and_holds_without_estimate():
    config = two_agent_config(
        target_link=LinkConfig(loss_model="independent", loss_probability=1.0),
        neighbor_link=LinkConfig(loss_model="none"),
    )
    sim = SwarmSimulator(config)
    result = sim.run(4)
    assert all(state.target_estimate_m is None for state in result.steps[-1].agents)
    assert all(
        event["loss_reason"] == "independent_loss"
        for event in result.link_events
        if event["sender"] == "__target__"
    )
    assert not any("target_keepout" in state.constraint_flags for state in result.steps[-1].agents)


def test_neighbor_packets_can_recover_target_estimate_after_target_dropout():
    # Agent a receives the target, while b loses every direct target packet.
    # A no-loss neighbor link lets b use a's advertised estimate.
    config = two_agent_config(
        target_link=LinkConfig(loss_model="independent", loss_probability=1.0),
        target_link_overrides=(("a", LinkConfig(loss_model="none")),),
        neighbor_link=LinkConfig(loss_model="none"),
    )
    sim = SwarmSimulator(config)
    result = sim.run(3)
    b = result.steps[-1].agents[1]
    assert b.target_estimate_m is None
    assert b.fused_target_m == pytest.approx((0.0, 0.0, 1.5))
    assert any(event["sender"] == "__target__" and event["receiver"] == "b" and event["loss_reason"]
               for event in result.link_events)


def test_collision_and_keepout_projection_are_reported():
    config = SwarmConfig(
        agents=(
            AgentSpec("a", (0.0, 0.0, 1.0)),
            AgentSpec("b", (0.0, 0.0, 1.0)),
        ),
        target_position_m=(0.0, 0.0, 1.0),
        target_keepout_radius_m=0.6,
        min_separation_m=0.8,
        keep_out_spheres=(KeepOutSphere((0.0, 0.0, 1.0), 0.9, "wall"),),
        seed=2,
    )
    step = SwarmSimulator(config).step()
    positions = {state.agent_id: np.asarray(state.position_m) for state in step.agents}
    assert np.linalg.norm(positions["a"] - positions["b"]) >= 0.8 - 1e-9
    assert all(np.linalg.norm(pos - np.array([0.0, 0.0, 1.0])) >= 0.9 - 1e-9 for pos in positions.values())
    assert all("separation" in state.constraint_flags for state in step.agents)
    assert all("wall" in state.constraint_flags for state in step.agents)


def test_keep_out_sphere_center_at_linear_velocity_is_deterministic():
    sphere = KeepOutSphere((1.0, -2.0, 0.5), 0.4, "moving-wall", (0.5, 0.0, -0.25))
    assert sphere.center_at(0.0) == (1.0, -2.0, 0.5)
    assert sphere.center_at(2.0) == (2.0, -2.0, 0.0)
    for velocity in ((50.1, 0.0, 0.0), (-50.1, 0.0, 0.0)):
        with pytest.raises(ValueError):
            KeepOutSphere((1.0, -2.0, 0.5), 0.4, "too-fast", velocity)


def test_dropout_schedule_removes_one_sender_but_keeps_other_active():
    sim = SwarmSimulator(two_agent_config())
    result = sim.run(5, dropout_schedule={2: ("b",)})
    assert result.steps[1].active_count == 2
    assert result.steps[2].active_count == 1
    assert result.steps[-1].agents[1].active is False
    # No packets are attempted from the dropped agent after the schedule step.
    assert all(not (event["sender"] == "b" and event["send_time"] >= 0.1) for event in result.link_events)


def test_config_rejects_duplicate_ids_and_invalid_geometry():
    with pytest.raises(ValueError):
        SwarmConfig(agents=(AgentSpec("a", (0, 0, 1)), AgentSpec("a", (1, 0, 1))))
    with pytest.raises(ValueError):
        SwarmConfig(agents=(AgentSpec("a", (0, 0, 1)),), min_separation_m=-1)
    with pytest.raises(ValueError):
        KeepOutSphere((0, 0, 0), -0.1)
