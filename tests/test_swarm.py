"""Independent checks for the Stage 7 multi-agent coordination fixture."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from sim.link import LinkConfig
from sim.swarm import AgentSpec, KeepOutSphere, SwarmConfig, SwarmSimulator

from platform_compare import same_tree


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


# --- Fixture reproducibility (generator versus schema versus retained file) ---

ROOT = Path(__file__).parents[1]
DOCUMENTED_GENERATION_COMMAND = (
    "PYTHONPATH=. ./.venv/bin/python tools/generate_s7_fixture.py --output sim/runs/s7-swarm"
)


def _regenerate_default_fixture(destination: Path) -> tuple[dict, dict]:
    """Run the documented generator with its documented defaults."""
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "generate_s7_fixture.py"), "--output", str(destination)],
        cwd=ROOT, env=env, check=True, capture_output=True,
    )
    return (
        json.loads((destination / "run.json").read_text(encoding="utf-8")),
        json.loads((destination / "manifest.json").read_text(encoding="utf-8")),
    )


@pytest.fixture(scope="module")
def regenerated_default_fixture(tmp_path_factory):
    return _regenerate_default_fixture(tmp_path_factory.mktemp("s7-regen"))


def test_generator_output_validates_against_the_s7_schema(regenerated_default_fixture):
    run, _ = regenerated_default_fixture
    schema = json.loads((ROOT / "sim" / "s7_run_schema.json").read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(run))
    assert len(errors) == 0, f"{len(errors)} schema errors, first: {errors[0].message if errors else ''}"


def test_retained_s7_run_and_manifest_equal_a_fresh_regeneration(regenerated_default_fixture):
    run, manifest = regenerated_default_fixture
    retained_run = json.loads((ROOT / "sim" / "runs" / "s7-swarm" / "run.json").read_text(encoding="utf-8"))
    retained_manifest = json.loads((ROOT / "sim" / "runs" / "s7-swarm" / "manifest.json").read_text(encoding="utf-8"))
    # The command string embeds the output directory, so it is the only field
    # allowed to differ. The retained copy must carry the documented command.
    assert retained_run.pop("generation_command") == DOCUMENTED_GENERATION_COMMAND
    assert retained_manifest["reproducibility"].pop("command") == DOCUMENTED_GENERATION_COMMAND
    run.pop("generation_command")
    manifest["reproducibility"].pop("command")
    assert same_tree(run, retained_run)
    assert same_tree(manifest, retained_manifest)


# --- provenance validation and fused-target reuse ---

def test_model_provenance_rejects_a_missing_profile_with_value_error():
    simulator = SwarmSimulator(two_agent_config())
    # AgentSpec fills in a default profile, so reach past it the way a bad caller could.
    object.__setattr__(simulator._agents["a"].spec, "hardware_profile", None)
    with pytest.raises(ValueError, match="hardware profile"):
        simulator.model_provenance()


def test_fused_target_is_computed_once_per_active_agent_per_step(monkeypatch):
    simulator = SwarmSimulator(two_agent_config())
    calls = []
    original = SwarmSimulator._fused_target

    def counting(self, agent):
        calls.append(agent.spec.agent_id)
        return original(self, agent)

    monkeypatch.setattr(SwarmSimulator, "_fused_target", counting)
    simulator.step()
    assert sorted(calls) == ["a", "b"]


# Digest of the run below, computed with sim/swarm.py as it was BEFORE the fused-target
# cleanup (git show 0595dc3:sim/swarm.py copied over a copy of sim/) together with
# the current sim/link.py, in the same environment. It is a literal on purpose:
# a test that recomputed the expected value with the new code would only restate it.
OLD_SWARM_RUN_SHA256 = "9213e3f5a13f1fb905d0946faac2f8568af83ad6b36ea186cb10ef9e2f536b9d"


@pytest.mark.skipif(sys.platform != "darwin", reason="the digest was recorded on macOS; other platforms differ in the last digits")
def test_run_with_jitter_loss_dropout_and_keep_out_matches_pre_cleanup_digest():
    from dataclasses import replace
    import hashlib

    from sim.scenarios import make_ring_swarm_config

    config = make_ring_swarm_config(
        agent_count=8, radius_m=1.0, seed=5,
        target_link=LinkConfig(delay_s=0.1, delay_jitter_s=0.02, loss_probability=0.3),
        neighbor_link=LinkConfig(delay_s=0.05, delay_jitter_s=0.01, loss_probability=0.2),
    )
    config = replace(
        config,
        target_velocity_mps=(0.3, 0.0, 0.0),
        keep_out_spheres=(KeepOutSphere((0.5, 0.5, 1.5), 0.4, "pole", (0.0, 0.1, 0.0)),),
    )
    run = SwarmSimulator(config).run(40, dropout_schedule={10: ["drone-002"]}).as_dict()
    # The scenario must actually exercise constraints, or the digest proves little.
    assert any(state["constraint_flags"] for step in run["steps"] for state in step["agents"])
    digest = hashlib.sha256(json.dumps(run, sort_keys=True).encode("utf-8")).hexdigest()
    assert digest == OLD_SWARM_RUN_SHA256
