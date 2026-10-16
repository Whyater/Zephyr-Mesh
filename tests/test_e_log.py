"""Checks for the simulated mission log format (writer, validator, recorder)."""
import copy
import json
import math
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from platform_compare import same_tree

from sim.failsafe import FailsafeConfig
from sim.missions.api import Scenario
from sim.missions.log import (
    MissionLogRecorder,
    build_log,
    canonical_json,
    load_log,
    round_floats,
    scenario_sha256,
    validate_log,
    write_log,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "sim" / "missions" / "mission_log_example.json"
SCHEMA = ROOT / "sim" / "missions" / "mission_log_schema.json"
HOVER_SCENARIO = ROOT / "sim" / "missions" / "examples" / "hover.json"


def real_scenario(**changes):
    """A scenario dictionary in the exact shape Scenario.as_dict() writes."""
    return Scenario.load(HOVER_SCENARIO).replace(**changes).as_dict()


def make_log(**overrides):
    """A small valid log: 3 frames, two drones, no estimates."""
    agents = ["a0", "a1"]
    series = {
        agent: {
            "position_m": [[0, 0, 1]] * 3,
            "velocity_mps": [[0, 0, 0]] * 3,
            "failsafe_state": ["nominal"] * 3,
            "authority": ["mission"] * 3,
            "link_age_s": [0.0] * 3,
            "estimate_position_m": None,
            "estimate_age_s": None,
        }
        for agent in agents
    }
    fields = dict(
        mission="tag",
        scenario=real_scenario(),
        code_revision="test",
        adapter="point_mass",
        preset="clean",
        seed=1,
        policy=FailsafeConfig().as_dict(),
        limits=["simulated"],
        dt_s=0.01,
        log_every=1,
        agents=agents,
        frames={"time_s": [0, 0.1, 0.2], "target_position_m": None, "agents": series},
        events=[],
        link_stats={},
        metrics={},
    )
    fields.update(overrides)
    return build_log(**fields)


def per_agent_frame():
    return {
        "position_m": [0, 0, 0],
        "velocity_mps": [0, 0, 0],
        "failsafe_state": "nominal",
        "authority": "mission",
        "link_age_s": 0.0,
        "estimate_position_m": None,
        "estimate_age_s": None,
    }


def test_schema_is_valid_and_example_validates():
    jsonschema.Draft202012Validator.check_schema(json.loads(SCHEMA.read_text(encoding="utf-8")))
    validate_log(json.loads(EXAMPLE.read_text(encoding="utf-8")))
    assert load_log(EXAMPLE)["simulated"] is True


def test_write_is_repeatable_and_round_trips(tmp_path):
    log = make_log()
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    write_log(first, log)
    write_log(second, log)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n") and b"\r" not in first.read_bytes()
    assert same_tree(load_log(first), log)


# Issue 9: the policy block accepts FailsafeConfig().as_dict() and nothing else.
def test_policy_accepts_failsafe_config_dict():
    validate_log(make_log(policy=FailsafeConfig().as_dict()))
    slow = FailsafeConfig(degraded_action="slow", weak_rule="age_or_loss", loss_window_packets=5)
    validate_log(make_log(policy=slow.as_dict()))


def test_policy_rejects_unknown_missing_and_bad_values():
    policy = FailsafeConfig().as_dict()
    with pytest.raises(ValueError, match="policy"):
        validate_log(make_log(policy={**policy, "extra": 1}))
    missing = dict(policy)
    del missing["weak_rule"]
    with pytest.raises(ValueError, match="policy"):
        validate_log(make_log(policy=missing))
    for key, bad in [("degraded_action", "land"), ("loss_window_packets", 0),
                     ("low_battery_pct", 101), ("weak_loss_fraction", 1.5),
                     ("slow_speed_mps", 0), ("weak_rule", "other"), ("land_after_s", -1)]:
        with pytest.raises(ValueError, match="policy"):
            validate_log(make_log(policy={**policy, key: bad}))


# Issue 10: round the scenario first, then hash it.
def test_build_log_hashes_the_rounded_scenario():
    params = {"gain": 0.123456789, "rate": [1.00000049, 2]}
    log = make_log(scenario=real_scenario(params=params))
    assert log["scenario"]["params"] == {"gain": 0.123457, "rate": [1.0, 2]}
    assert log["scenario_sha256"] == scenario_sha256(log["scenario"])
    validate_log(log)


def test_hash_is_order_independent_and_sensitive():
    assert scenario_sha256({"a": 1, "b": 2}) == scenario_sha256({"b": 2, "a": 1})
    assert scenario_sha256({"a": 1}) != scenario_sha256({"a": 2})
    assert canonical_json({"b": [1, 2], "a": "é"}) == '{"a":"\\u00e9","b":[1,2]}'


def test_wrong_hash_is_rejected():
    log = make_log()
    log["scenario_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scenario_sha256"):
        validate_log(log)


# Issue 11: write_log rounds before validating; round_floats handles NumPy.
def test_write_log_rounds_and_nulls_non_finite(tmp_path):
    log = make_log()
    log["metrics"] = {"nan": float("nan"), "inf": math.inf, "x": 0.1234567891, "neg": -0.0}
    log["frames"]["time_s"] = [0, 0.1000004, 0.2]
    path = tmp_path / "log.json"
    write_log(path, log)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["metrics"] == {"nan": None, "inf": None, "x": 0.123457, "neg": 0.0}
    assert saved["frames"]["time_s"] == [0, 0.1, 0.2]
    assert "-0.0" not in path.read_text(encoding="utf-8")
    assert log["metrics"]["x"] == 0.1234567891  # the caller's dict is not changed


def test_round_floats_converts_numpy_and_tuples():
    out = round_floats({
        "a": np.array([[1.23456789, np.nan], [0.0, -0.0]]),
        "b": np.float64(0.5000004),
        "c": np.int64(7),
        "d": np.bool_(True),
        "e": (1, 2.0000004),
        "f": np.float32(0.25),
    })
    assert out == {"a": [[1.234568, None], [0.0, 0.0]], "b": 0.5, "c": 7, "d": True,
                   "e": [1, 2.0], "f": 0.25}
    assert type(out["b"]) is float and type(out["c"]) is int and type(out["d"]) is bool
    assert type(out["a"][0][0]) is float
    json.dumps(out, allow_nan=False)


# Issue 12: the recorder's output validates end to end.
def test_recorder_output_validates_without_estimates():
    recorder = MissionLogRecorder(["b", "a"], 0.01, 1)
    for i in range(11):
        recorder.frame(i, i * 0.01, None, {"a": per_agent_frame(), "b": per_agent_frame()})
    log = recorder.to_log(
        mission="hover", scenario=real_scenario(), code_revision="x", adapter="point_mass", preset="clean",
        seed=1, policy=FailsafeConfig().as_dict(), limits=[], link_stats={}, metrics={})
    validate_log(log)
    assert log["agents"] == ["a", "b"]
    assert len(log["frames"]["time_s"]) == 11
    assert log["frames"]["target_position_m"] is None
    assert log["frames"]["agents"]["a"]["estimate_position_m"] is None
    assert log["frames"]["agents"]["a"]["estimate_age_s"] is None


def test_recorder_keeps_partial_estimates_and_events():
    recorder = MissionLogRecorder(["a"], 0.01, 5)
    for i in range(11):
        row = per_agent_frame()
        if i == 10:
            row["estimate_position_m"] = np.array([1.0, 2.0, 3.0])
            row["estimate_age_s"] = 0.05
        recorder.frame(i, i * 0.01, [0.5, 0, 1], {"a": row})
    recorder.event(0.05, "replan", "a", {"gain": np.float64(0.5), "ids": (1, 2)})
    recorder.event(0.1, "mission_end", None)
    log = recorder.to_log(
        mission="sweep", scenario=real_scenario(), code_revision="x", adapter="physics", preset="clean",
        seed=2, policy=FailsafeConfig().as_dict(), limits=[], link_stats={}, metrics={})
    validate_log(log)
    assert log["frames"]["time_s"] == [0, 0.05, 0.1]
    series = log["frames"]["agents"]["a"]
    assert series["estimate_position_m"] == [None, None, [1.0, 2.0, 3.0]]
    assert series["estimate_age_s"] == [None, None, 0.05]
    assert log["events"][0]["detail"] == {"gain": 0.5, "ids": [1, 2]}


def test_partial_build_log_for_events_only():
    # The engine builds just the events through build_log.
    out = build_log(events=[{"time_s": np.float64(0.1), "kind": "landed", "agent": None, "detail": {}}])
    assert out["events"][0]["time_s"] == 0.1
    assert "scenario_sha256" not in out


def test_mission_enum():
    for mission in ["tag", "sweep", "relay", "monitor", "formation", "failsafe", "hover"]:
        validate_log(make_log(mission=mission))
    for bad in ["failsafe_hold", "Tag", ""]:
        with pytest.raises(ValueError, match="mission"):
            validate_log(make_log(mission=bad))


def test_estimate_and_target_columns_accept_null_entries():
    log = make_log()
    log["frames"]["target_position_m"] = [[1, 2, 3], None, [1, 2, 4]]
    series = log["frames"]["agents"]["a0"]
    series["estimate_position_m"] = [None, [0, 0, 1], None]
    series["estimate_age_s"] = [None, 0.1, None]
    series["link_age_s"] = [None, 0.1, 0.2]
    validate_log(log)


def test_extra_keys_and_bad_lengths_rejected():
    log = make_log()
    log["frames"]["agents"]["a0"]["oops"] = 1
    with pytest.raises(ValueError):
        validate_log(log)
    log = make_log()
    log["frames"]["agents"]["a0"]["authority"] = []
    with pytest.raises(ValueError, match="authority"):
        validate_log(log)
    log = make_log()
    log["frames"]["agents"]["a1"]["estimate_age_s"] = [None]
    with pytest.raises(ValueError, match="estimate_age_s"):
        validate_log(log)
    log = make_log()
    log["extra"] = 1
    with pytest.raises(ValueError):
        validate_log(log)


def test_target_position_length_checked():
    log = make_log()
    log["frames"]["target_position_m"] = [[0, 0, 1], [0, 0, 1]]
    with pytest.raises(ValueError, match="target_position_m"):
        validate_log(log)


def test_frame_times_must_increase():
    for times in ([0, 0.1, 0.1], [0, 0.2, 0.1]):
        log = make_log()
        log["frames"]["time_s"] = times
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_log(log)


def test_events_must_be_sorted_known_kind_and_known_agent():
    good = [{"time_s": 0.1, "kind": "landed", "agent": "a0", "detail": {}},
            {"time_s": 0.2, "kind": "mission_end", "agent": None, "detail": {}}]
    validate_log(make_log(events=good))
    with pytest.raises(ValueError, match="sorted"):
        validate_log(make_log(events=list(reversed(good))))
    with pytest.raises(ValueError, match="kind"):
        validate_log(make_log(events=[{"time_s": 0, "kind": "unknown", "agent": None, "detail": {}}]))
    with pytest.raises(ValueError, match="agent"):
        validate_log(make_log(events=[{"time_s": 0, "kind": "landed", "agent": "zz", "detail": {}}]))


def test_event_detail_is_free_form_but_json_ready():
    detail = {"anything": [1, {"deep": "yes"}], "other_key": None}
    validate_log(make_log(events=[{"time_s": 0, "kind": "replan", "agent": None, "detail": detail}]))
    log = make_log(events=[{"time_s": 0, "kind": "replan", "agent": None, "detail": {}}])
    log["events"][0]["detail"] = {"bad": {1, 2}}
    with pytest.raises(ValueError, match="JSON-ready"):
        validate_log(log)
    log["events"][0]["detail"] = {"bad": float("nan")}
    with pytest.raises(ValueError, match="JSON-ready"):
        validate_log(log)


def test_validate_raises_value_error_when_many_things_are_wrong():
    # Several unrelated problems at once still give one ValueError (best_match, no sorting).
    log = make_log()
    log["frames"]["agents"]["a0"]["position_m"][1] = [0, 0]
    log["metrics"] = {"bad": [1]}
    log["seed"] = "one"
    with pytest.raises(ValueError):
        validate_log(log)


def test_agents_must_be_sorted_and_match_frames():
    log = make_log()
    log["agents"] = ["a1", "a0"]
    with pytest.raises(ValueError, match="sorted"):
        validate_log(log)
    log = make_log()
    del log["frames"]["agents"]["a1"]
    with pytest.raises(ValueError, match="frames.agents"):
        validate_log(log)


def test_validate_does_not_change_input():
    log = make_log()
    before = copy.deepcopy(log)
    validate_log(log)
    assert log == before


# Validation gaps: non-finite numbers anywhere, bad policy, recorder output.
@pytest.mark.parametrize("path", [
    ("policy", "low_battery_pct"),
    ("frames", "time_s", 1),
    ("frames", "agents", "a0", "position_m", 1, 0),
    ("frames", "agents", "a1", "link_age_s", 2),
    ("dt_s",),
])
def test_non_finite_numbers_rejected_anywhere(path):
    for bad in (float("nan"), math.inf):
        log = make_log()  # built by hand below: round_floats would turn these into null
        node = log
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = bad
        with pytest.raises(ValueError, match="non-finite"):
            validate_log(log)


def test_policy_with_land_before_degraded_rejected():
    policy = {**FailsafeConfig().as_dict(), "degraded_after_s": 0.5, "land_after_s": 0.1}
    with pytest.raises(ValueError, match="^policy: "):
        validate_log(make_log(policy=policy))


def test_recorder_validates_what_it_returns():
    recorder = MissionLogRecorder(["a"], 0.01, 1)
    recorder.frame(0, 0.0, None, {"a": per_agent_frame()})
    header = dict(
        mission="hover", scenario=real_scenario(), code_revision="x", adapter="point_mass",
        preset="clean", seed=1, policy=FailsafeConfig().as_dict(), limits=[], link_stats={},
        metrics={})
    validate_log(recorder.to_log(**header))
    with pytest.raises(ValueError, match="adapter"):
        recorder.to_log(**{**header, "adapter": "bogus"})
    with pytest.raises(ValueError, match="scenario"):
        recorder.to_log(**{**header, "scenario": {"x": 1}})


# Closed scenario block.
def test_scenario_from_api_round_trips_with_outages_and_policy():
    scenario = real_scenario(
        outages=[{"start_s": 1.0, "end_s": None, "streams": None},
                 {"start_s": 2.0, "end_s": 3.5, "streams": ["cmd:a0", "tel:a0"]}],
        policy={"land_after_s": 0.5, "degraded_action": "slow"},
        params={"start_m": [2.0, 2.0, 1.5], "note": None, "nested": {"k": [1, "two", True]}},
        measurement_delay_s=0.0123456789)
    log = make_log(scenario=scenario)
    validate_log(log)
    assert log["scenario"]["outages"][1]["streams"] == ["cmd:a0", "tel:a0"]
    assert log["scenario"]["measurement_delay_s"] == 0.012346


def test_scenario_rejects_unknown_missing_and_mistyped_keys():
    good = real_scenario()
    for scenario in [
        {**good, "extra": 1},
        {key: value for key, value in good.items() if key != "preset"},
        {**good, "seed": "one"},
        {**good, "seed": -1},
        {**good, "adapter": "bogus"},
        {**good, "duration_s": 0},
        {**good, "command_delay_s": -0.1},
        {**good, "outages": {}},
    ]:
        with pytest.raises(ValueError, match="scenario"):
            validate_log(make_log(scenario=scenario))


def test_scenario_outage_is_closed():
    for outage in [
        {"start_s": 1.0, "end_s": None, "streams": None, "extra": 1},
        {"end_s": 2.0, "streams": None},
        {"start_s": -1.0, "end_s": None, "streams": None},
        {"start_s": 1.0, "end_s": "later", "streams": None},
        {"start_s": 1.0, "end_s": None, "streams": [1]},
    ]:
        with pytest.raises(ValueError, match="scenario.outages"):
            validate_log(make_log(scenario={**real_scenario(), "outages": [outage]}))


def test_scenario_policy_is_a_closed_partial_override():
    validate_log(make_log(scenario=real_scenario(policy={})))
    validate_log(make_log(scenario=real_scenario(policy={"weak_rule": "age_or_loss"})))
    for policy in [{"extra": 1}, {"degraded_action": "land"}, {"loss_window_packets": 0},
                   {"slow_speed_mps": "fast"}, {"low_battery_pct": 101}]:
        with pytest.raises(ValueError, match="scenario.policy"):
            validate_log(make_log(scenario={**real_scenario(), "policy": policy}))


def test_dynamic_objects_reject_empty_keys_but_allow_any_json():
    with pytest.raises(ValueError, match="scenario.params"):
        validate_log(make_log(scenario=real_scenario(params={"": 1})))
    with pytest.raises(ValueError, match="detail"):
        validate_log(make_log(events=[{"time_s": 0, "kind": "replan", "agent": None, "detail": {"": 1}}]))
    with pytest.raises(ValueError, match="metrics"):
        validate_log(make_log(metrics={"": 1}))
    with pytest.raises(ValueError, match="link_stats"):
        validate_log(make_log(link_stats={"": {"sent": 0, "delivered": 0, "lost_loss": 0, "lost_outage": 0}}))
    validate_log(make_log(metrics={"end_reason": "timeout", "ok": True, "x": None, "y": 1.5}))
