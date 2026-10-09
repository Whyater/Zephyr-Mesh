"""Contract and deterministic-run checks for the portable S7 scenario runner."""

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from desktop.replay import ReplayModel
from sim.scenario_runner import (
    JSON_SAFE_INTEGER_MAX,
    ScenarioConfigError,
    load_json_config,
    run_scenario,
    scenario_hash,
    serialize_run,
    to_replay_document,
    validate_scenario_config,
)


def config(**overrides):
    value = {
        "agent_count": 2,
        "radius_m": 2.0,
        "altitude_m": 1.5,
        "steps": 3,
        "seed": 7,
        "target_link": {"delay_s": 0.0, "delay_jitter_s": 0.0, "loss_probability": 0.0},
        "neighbor_link": {"delay_s": 0.0, "delay_jitter_s": 0.0, "loss_probability": 0.0},
    }
    value.update(overrides)
    return value


def test_same_seed_is_byte_deterministic():
    first = serialize_run(run_scenario(config(), code_revision="test"))
    second = serialize_run(run_scenario(config(), code_revision="test"))
    assert first == second


def test_checked_in_example_config_generates_a_valid_run():
    example = Path(__file__).parents[1] / "sim" / "scenario_example.json"
    result = run_scenario(load_json_config(example), code_revision="test")
    assert result["parameters"]["scenario_id"] == "s7-example"
    assert len(result["frames"]) == result["parameters"]["steps"]
    assert serialize_run(result).endswith(b"\n")


def test_canonical_parameter_bytes_define_cross_language_hash_contract():
    result = run_scenario(config(), code_revision="test")
    canonical = result["parameters_canonical"]
    assert canonical.endswith("\n")
    assert json.loads(canonical) == result["parameters"]
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == result["scenario_hash"]


def test_zero_delay_known_answer_delivers_target_before_first_motion():
    result = run_scenario(config(steps=1), code_revision="test")
    first_agent = result["frames"][0]["agents"][0]
    assert first_agent["target_estimate_m"] == [0.0, 0.0, 1.5]
    assert first_agent["fused_target_m"] == [0.0, 0.0, 1.5]
    assert all(event["receive_time"] == event["send_time"] for event in result["link_events"])


def test_one_agent_target_motion_has_independent_time_answer():
    result = run_scenario(config(agent_count=1, steps=2, target_velocity_mps=[0.5, 0.0, 0.0]), code_revision="test")
    assert result["frames"][0]["target_position_m"] == [0.0, 0.0, 1.5]
    assert result["frames"][1]["target_position_m"] == [0.025, 0.0, 1.5]


def test_delayed_target_packet_age_is_nonnegative_and_bounded_by_delay():
    result = run_scenario(config(agent_count=1, steps=6, target_link={"delay_s": 0.1, "loss_probability": 0.0}), code_revision="test")
    ages = [frame["agents"][0]["target_age_s"] for frame in result["frames"]]
    observed = [age for age in ages if age is not None]
    assert observed
    assert all(age >= 0.0 for age in observed)
    assert max(observed) <= 0.1 + 1e-12
    for frame in result["frames"]:
        state = frame["agents"][0]
        if state["target_age_s"] is not None:
            assert state["target_age_s"] == pytest.approx(frame["time_s"] - state["target_source_time_s"])
    received = [event["packet_age"] for event in result["link_events"] if event["outcome"] == "received"]
    assert received == [pytest.approx(0.1)] * len(received)


def test_deterministic_loss_is_explicit_in_every_target_event():
    result = run_scenario(config(agent_count=1, steps=4, target_link={"loss_probability": 1.0}), code_revision="test")
    target_events = [event for event in result["link_events"] if event["sender"] == "__target__"]
    assert target_events
    assert all(event["outcome"] == "lost" and event["loss_reason"] == "independent_loss" for event in target_events)


def test_keep_out_sphere_invariant_is_reported_in_frames():
    result = run_scenario(config(agent_count=1, steps=2, keep_out_spheres=[{"center_m": [2.0, 0.0, 1.5], "radius_m": 0.5, "label": "wall"}]), code_revision="test")
    for frame in result["frames"]:
        position = frame["agents"][0]["position_m"]
        distance = ((position[0] - 2.0) ** 2 + position[1] ** 2 + (position[2] - 1.5) ** 2) ** 0.5
        assert distance >= 0.5 - 1e-12
        if distance <= 0.5 + 1e-12:
            assert "wall" in frame["agents"][0]["constraint_flags"]


def test_moving_keep_out_snapshots_and_invariant_are_deterministic():
    sphere = {"center_m": [2.0, 0.0, 1.5], "radius_m": 0.5, "label": "moving-wall", "velocity_mps": [0.5, 0.0, 0.0]}
    result = run_scenario(config(agent_count=1, steps=3, keep_out_spheres=[sphere]), code_revision="test")
    snapshots = [frame["keep_out_spheres"][0]["center_m"] for frame in result["frames"]]
    assert snapshots == [[2.0, 0.0, 1.5], [2.025, 0.0, 1.5], [2.05, 0.0, 1.5]]
    dt = result["parameters"]["dt_s"]
    for index, frame in enumerate(result["frames"]):
        center = frame["keep_out_spheres"][0]["center_m"]
        assert center == [2.0 + 0.5 * index * dt, 0.0, 1.5]
        position = frame["agents"][0]["position_m"]
        distance = sum((position[index] - center[index]) ** 2 for index in range(3)) ** 0.5
        assert distance >= 0.5 - 1e-12


def test_malformed_config_and_bounds_are_rejected():
    with pytest.raises(ScenarioConfigError):
        validate_scenario_config({"agent_count": 2})
    with pytest.raises(ScenarioConfigError):
        validate_scenario_config(config(agent_count=0))
    with pytest.raises(ScenarioConfigError):
        validate_scenario_config(config(steps=10_001))
    with pytest.raises(ScenarioConfigError):
        validate_scenario_config(config(agent_count=100, steps=100))
    with pytest.raises(ScenarioConfigError):
        validate_scenario_config(config(unexpected=True))
    with pytest.raises(ScenarioConfigError):
        validate_scenario_config(config(target_link={"delay_s": 0.0, "delay_jitter_s": 0.1}))
    with pytest.raises(ScenarioConfigError):
        validate_scenario_config(config(target_link={"loss_model": "none", "loss_probability": 0.1}))
    with pytest.raises(ScenarioConfigError):
        validate_scenario_config(config(target_link={"loss_model": "burst", "loss_probability": 0.1}))
    with pytest.raises(ScenarioConfigError):
        validate_scenario_config(config(target_link={"packet_duration_s": 0.1}))
    with pytest.raises(ScenarioConfigError):
        validate_scenario_config(config(keep_out_spheres=[{"center_m": [0.0, 0.0, 1.0], "radius_m": 0.5, "velocity_mps": [50.1, 0.0, 0.0]}]))


def test_hash_changes_when_a_scenario_parameter_changes():
    first = run_scenario(config(), code_revision="test")
    second = run_scenario(config(radius_m=2.1), code_revision="test")
    assert first["scenario_hash"] == scenario_hash(first["parameters"])
    assert second["scenario_hash"] != first["scenario_hash"]


def test_json_safe_seed_boundary_is_accepted_by_runner_and_desktop(tmp_path):
    from desktop.scenario_run import load_scenario_run

    result = run_scenario(config(seed=JSON_SAFE_INTEGER_MAX, steps=1), code_revision="test")
    path = tmp_path / "safe-seed.json"
    path.write_bytes(serialize_run(result))
    loaded = load_scenario_run(path)
    assert loaded.parameters["seed"] == JSON_SAFE_INTEGER_MAX
    with pytest.raises(ScenarioConfigError):
        validate_scenario_config(config(seed=JSON_SAFE_INTEGER_MAX + 1))


def test_evidence_boundary_is_explicit_and_profiles_are_synthetic():
    result = run_scenario(config(), code_revision="test")
    boundary = result["evidence_boundary"].lower()
    assert "synthetic" in boundary
    assert "not" in boundary or "does not establish" in boundary
    assert result["provenance"]["status"] == "synthetic deterministic replay; read-only"
    assert all(profile["status"] in {"synthetic", "fixture"} for profile in result["profiles"])


def test_converter_loads_new_envelope_in_existing_desktop_replay_model():
    scenario = run_scenario(config(), code_revision="test")
    replay = to_replay_document(scenario)
    model = ReplayModel(replay)
    integrated_model = ReplayModel.from_scenario_document(scenario)
    schema = json.loads((Path(__file__).parents[1] / "sim" / "s7_run_schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(replay)
    assert model.frame_count == scenario["parameters"]["steps"]
    assert integrated_model.frame_count == model.frame_count
    assert model.scenario_id == scenario["parameters"]["scenario_id"]
    assert model.frame.agents[0].position_m == tuple(scenario["frames"][0]["agents"][0]["position_m"])


def test_nested_output_validation_rejects_nonfinite_agent_value():
    scenario = run_scenario(config(), code_revision="test")
    scenario["frames"][0]["agents"][0]["position_m"][0] = float("nan")
    with pytest.raises(ScenarioConfigError):
        serialize_run(scenario)
    malformed = run_scenario(config(), code_revision="test")
    malformed["parameters"] = []
    with pytest.raises(ScenarioConfigError):
        serialize_run(malformed)


def test_integrity_and_link_mutations_are_rejected_across_boundaries():
    scenario = run_scenario(config(), code_revision="test")
    mutated_payload = json.loads(json.dumps(scenario))
    mutated_payload["frames"][0]["agents"][0]["position_m"][0] += 0.1
    with pytest.raises(ScenarioConfigError):
        serialize_run(mutated_payload)
    mutated_digest = json.loads(json.dumps(scenario))
    mutated_digest["payload_sha256"] = "0" * 64
    with pytest.raises(ScenarioConfigError):
        serialize_run(mutated_digest)
    mutated_link = json.loads(json.dumps(scenario))
    mutated_link["parameters"]["target_link"]["delay_jitter_s"] = 0.1
    mutated_link["scenario_hash"] = scenario_hash(mutated_link["parameters"])
    mutated_link["parameters_canonical"] = (json.dumps(mutated_link["parameters"], sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    with pytest.raises(ScenarioConfigError):
        serialize_run(mutated_link)
    mutated_revision = json.loads(json.dumps(scenario))
    mutated_revision["provenance"]["code_revision"] = 3
    with pytest.raises(ScenarioConfigError):
        serialize_run(mutated_revision)


def test_payload_digest_rejects_pairwise_frame_mutation():
    scenario = run_scenario(config(), code_revision="test")
    scenario["frames"][0]["agents"][0]["position_m"][0] += 0.01
    with pytest.raises(ScenarioConfigError, match="payload"):
        serialize_run(scenario)


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        ("frames", lambda value: value + 0.01),
        ("link_events", lambda value: "lost" if value == "received" else "received"),
        ("profiles", lambda value: "fixture" if value == "synthetic" else "synthetic"),
    ],
)
def test_payload_digest_rejects_each_payload_section_mutation(section, mutate):
    scenario = run_scenario(config(), code_revision="test")
    if section == "frames":
        scenario[section][0]["time_s"] = mutate(scenario[section][0]["time_s"])
    elif section == "link_events":
        scenario[section][0]["outcome"] = mutate(scenario[section][0]["outcome"])
    else:
        scenario[section][0]["status"] = mutate(scenario[section][0]["status"])
    with pytest.raises(ScenarioConfigError, match="payload"):
        serialize_run(scenario)


def test_json_loader_rejects_duplicate_keys_and_symlink(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"agent_count": 2, "agent_count": 3}', encoding="utf-8")
    with pytest.raises(ScenarioConfigError):
        load_json_config(config_path)
    link = tmp_path / "link.json"
    link.symlink_to(config_path)
    with pytest.raises(ScenarioConfigError):
        load_json_config(link)
