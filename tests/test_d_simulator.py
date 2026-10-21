"""Checks for the simulated cooperative S7 run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from jsonschema import Draft202012Validator

from sim.coop.avoidance import AvoidanceConfig
from sim.coop.simulator import (
    CoopLinkConfig,
    CoopSwarmConfig,
    CoopSwarmSimulator,
    _count_disagreements,
)
from sim.estimation.base import EstimatorSettings
from sim.estimation.factory import make_estimator
from sim.sensor import SensorConfig, SensorModel


ROOT = Path(__file__).resolve().parents[1]


def _generate(path: Path, agents=4):
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.run([sys.executable, "tools/d_generate_s7_coop.py", "--output", str(path), "--agents", str(agents), "--seed", "23", "--duration", "20"], cwd=ROOT, check=True, env=env, encoding="utf-8")


def test_generator_is_byte_stable(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _generate(a)
    _generate(b)
    assert (a / "run.json").read_bytes() == (b / "run.json").read_bytes()
    assert (a / "replay_v1.json").read_bytes() == (b / "replay_v1.json").read_bytes()
    ma, mb = json.loads((a / "manifest.json").read_text(encoding="utf-8")), json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    ma.pop("volatile")
    mb.pop("volatile")
    assert ma == mb


def test_both_schemas_validate(tmp_path):
    out = tmp_path / "run"
    _generate(out)
    for name, schema in (("run.json", "sim/s7_coop_run_schema.json"), ("replay_v1.json", "sim/s7_run_schema.json")):
        value = json.loads((out / name).read_text(encoding="utf-8"))
        validator = Draft202012Validator(json.loads((ROOT / schema).read_text(encoding="utf-8")))
        assert list(validator.iter_errors(value)) == []


def test_demo_moves_and_re_spaces(tmp_path):
    out = tmp_path / "run"
    _generate(out)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    positions = [np.asarray(s["agents"][1]["position_m"]) for s in run["steps"]]
    assert sum(np.linalg.norm(positions[i] - positions[i - 1]) > 1e-12 for i in range(1, len(positions))) >= 0.9 * (len(positions) - 1)
    truth = [np.asarray(s["target_truth_position_m"]) for s in run["steps"]]
    assert all(np.linalg.norm(truth[i] - truth[i - 1]) > 1e-12 for i in range(1, len(truth)))
    s16 = min(run["steps"], key=lambda s: abs(s["time_s"] - 16.0))
    for agent in s16["agents"][1:]:
        assert agent["believed_alive"] == ["drone-002", "drone-003", "drone-004"]


def test_single_observer_zero_noise_matches_separate_filter():
    config = CoopSwarmConfig(n_agents=1, duration_s=2.0, noise_std_m=0.0, fusion="none", dropouts=(), link=CoopLinkConfig(delay_s=0.0, loss_probability=0.0, loss_model="none"))
    simulator = CoopSwarmSimulator(config)
    run = simulator.run().as_dict()
    standalone = make_estimator(config.estimator, EstimatorSettings(), simulator._target_start)
    sensor = SensorModel(SensorConfig(noise_std_m=0.0), seed=int(np.random.SeedSequence([config.seed, 100]).generate_state(1)[0]))
    phase = 0.0
    next_capture = phase
    for step in run["steps"]:
        t = float(step["time_s"])
        while next_capture <= t + 1e-12:
            capture = round(next_capture, 12)
            truth = simulator._truth(capture)[0]
            sample = sensor.sample(truth, capture)
            if sample.available:
                standalone.update(sample.measurement_m, capture)
            next_capture += 1.0 / config.sensor_rate_hz
        expected = standalone.predict_at(t).position_m
        actual = np.asarray(step["agents"][0]["target_estimate_m"])
        assert np.max(np.abs(actual - expected)) <= 1e-12


def test_send_time_age_delay_regression():
    config = CoopSwarmConfig(
        n_agents=3,
        duration_s=3.0,
        noise_std_m=0.0,
        fusion="ci",
        dropouts=(),
        link=CoopLinkConfig(delay_s=0.3, loss_probability=0.0, loss_model="none"),
    )
    run = CoopSwarmSimulator(config).run().as_dict()
    errors = [row["truth_error_m"] for step in run["steps"] if step["time_s"] >= 2.0 for row in step["agents"] if row["active"]]
    group = [step["gs_group_error_m"] for step in run["steps"] if step["time_s"] >= 2.0 and step["gs_group_error_m"] is not None]
    assert np.percentile(errors, 95) < 0.05
    assert np.percentile(group, 95) < 0.05


def test_ground_controller_matches_onboard_pd():
    common = dict(n_agents=4, duration_s=10.0, noise_std_m=0.0, dropouts=(), fusion="central", link=CoopLinkConfig(delay_s=0.0, loss_probability=0.0, loss_model="none"))
    onboard = CoopSwarmSimulator(CoopSwarmConfig(**common, decision="onboard")).run().as_dict()
    ground = CoopSwarmSimulator(CoopSwarmConfig(**common, decision="ground")).run().as_dict()
    def median(value):
        return np.median([a["standoff_error_m"] for s in value["steps"] if s["time_s"] >= 8.0 for a in s["agents"] if a["active"]])
    assert abs(median(onboard) - median(ground)) < 0.02


def test_silent_agent_sends_nothing_after_drop():
    config = CoopSwarmConfig(n_agents=3, duration_s=3.0, fusion="central", dropouts=(("drone-001", 1.0, "silent"),))
    run = CoopSwarmSimulator(config).run().as_dict()
    assert not [event for event in run["link_events"] if event["sender"] == "drone-001" and event["send_time"] >= 1.0]


def test_landed_leaving_and_gs_fuser_logging():
    config = CoopSwarmConfig(n_agents=3, duration_s=2.0, fusion="central", dropouts=(("drone-001", 1.0, "landed"),), link=CoopLinkConfig(delay_s=0.3, loss_probability=0.0, loss_model="none"))
    run = CoopSwarmSimulator(config).run().as_dict()
    leaving = [event for event in run["link_events"] if event["kind"] == "leaving"]
    assert len(leaving) == 3
    leaving_arrival = max(event["receive_time"] for event in leaving if event["receive_time"] is not None)
    assert 1.29 <= leaving_arrival <= 1.31
    survivor = min(run["steps"], key=lambda step: abs(step["time_s"] - 1.5))
    assert "drone-001" not in survivor["agents"][1]["believed_alive"]
    assert all("gs_fuser_estimate_m" in step and "gs_fuser_error_m" in step for step in run["steps"])
    assert run["summary"]["gs_fuser_error_max_m"] is not None


def test_traffic_is_bucketed_by_kind_and_ci_mode_is_recorded():
    config = CoopSwarmConfig(n_agents=2, duration_s=2.0, fusion="central", dropouts=(), ci_mode="fast")
    run = CoopSwarmSimulator(config).run().as_dict()
    assert run["config"]["ci_mode"] == "fast"
    assert set(run["summary"]["traffic"]) == {"state", "raw", "track", "command", "leaving"}
    assert all("kind" in event for event in run["link_events"])
    shares = 21
    assert run["summary"]["traffic"]["state"]["messages"] == 4 * shares
    assert 40 <= run["summary"]["traffic"]["raw"]["messages"] <= 42
    assert run["summary"]["traffic"]["track"]["messages"] == 2 * shares
    assert run["summary"]["traffic"]["raw"]["bytes"] == 56 * run["summary"]["traffic"]["raw"]["messages"]


def test_track_messages_only_for_central_fusion():
    config = CoopSwarmConfig(n_agents=2, duration_s=1.0, fusion="ci", dropouts=())
    run = CoopSwarmSimulator(config).run().as_dict()
    assert run["summary"]["traffic"]["track"]["messages"] == 0


def test_ground_command_carries_plan_and_outage_timeout_is_bounded():
    config = CoopSwarmConfig(
        n_agents=3,
        duration_s=4.0,
        decision="ground",
        fusion="central",
        dropouts=(),
        gs_outage=((2.0, 3.0),),
        link=CoopLinkConfig(delay_s=0.05, loss_probability=0.0, loss_model="none"),
    )
    run = CoopSwarmSimulator(config).run().as_dict()
    command_events = [event for event in run["link_events"] if event["kind"] == "command"]
    assert command_events
    healthy = [a for step in run["steps"] if 1.0 <= step["time_s"] < 2.0 for a in step["agents"] if a["active"]]
    assert healthy and all(len(row["believed_assignment"]) == len(row["believed_alive"]) for row in healthy)
    timeout_steps = [step["time_s"] for step in run["steps"] if any("command_timeout" in a["flags"] for a in step["agents"] if a["active"])]
    assert timeout_steps and min(timeout_steps) >= 2.0
    assert not [step for step in run["steps"] if step["time_s"] < 2.0 and any("command_timeout" in a["flags"] for a in step["agents"] if a["active"])]




def test_perfect_six_observers_keep_separation():
    config = CoopSwarmConfig(
        n_agents=6,
        duration_s=8.0,
        dropouts=(),
        link=CoopLinkConfig(
            delay_s=0.0, loss_probability=0.0, loss_model="none"
        ),
        avoidance=AvoidanceConfig(ellipsoid_radii=(0.12, 0.12, 0.30)),
    )
    run = CoopSwarmSimulator(config).run().as_dict()
    assert run["summary"]["separation_violations_count"] == 0
    assert run["summary"]["min_pairwise_separation_m"] >= 0.35 - 1e-6


def test_simulator_determinism_and_banned_words():
    config = CoopSwarmConfig(n_agents=2, duration_s=1.0, dropouts=(), link=CoopLinkConfig(delay_s=0.0, loss_probability=0.0, loss_model="none"))
    first = CoopSwarmSimulator(config).run().as_dict()
    second = CoopSwarmSimulator(config).run().as_dict()
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)
    text = (ROOT / "sim/coop/simulator.py").read_text(encoding="utf-8").lower()
    for word in ("intercept", "kill", "attack", "weapon", "payload", "strike", "targeting"):
        assert word not in text


def test_replay_timing_and_frozen_swarm_schema():
    config = CoopSwarmConfig(n_agents=1, duration_s=1.0, log_every=2, dropouts=(), link=CoopLinkConfig(delay_s=0.0, loss_probability=0.0, loss_model="none"))
    from sim.coop.simulator import to_replay_v1
    replay = to_replay_v1(CoopSwarmSimulator(config).run())
    assert replay["dt_s"] == config.control_dt_s * config.log_every
    frozen = json.loads((ROOT / "runs/s7-swarm/run.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "sim/s7_run_schema.json").read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(frozen)) == []


def test_a_disagreement_helper_hand_cases():
    views_a = {
        "a": ("a", "b", "c"), "b": ("a", "b", "c"), "c": ("c",)
    }
    assignments_a = {
        "a": {"a": 0, "b": 1, "c": 2},
        "b": {"a": 0, "b": 2, "c": 1},
        "c": {"c": 0},
    }
    assert _count_disagreements(views_a, assignments_a) == 1
    assignments_b = dict(assignments_a)
    assignments_b["b"] = assignments_b["a"]
    assert _count_disagreements(views_a, assignments_b) == 0
    views_c = {"a": ("a", "b"), "b": ("a", "b"), "c": ("a", "b", "c")}
    assignments_c = dict(assignments_a)
    assignments_c["b"] = assignments_c["a"]
    assert _count_disagreements(views_c, assignments_c) == 0


def _dropout_run(mode):
    config = CoopSwarmConfig(
        n_agents=4,
        duration_s=20.0,
        log_every=1,
        dropouts=(("drone-001", 10.0, mode),),
        link=CoopLinkConfig(
            delay_s=0.05, loss_probability=0.0, loss_model="none"
        ),
    )
    return CoopSwarmSimulator(config).run().as_dict()


def _first_missing(run):
    result = {}
    for step in run["steps"]:
        for row in step["agents"]:
            if row["active"] and row["agent_id"] != "drone-001":
                if step["time_s"] >= 10.0 and "drone-001" not in row["believed_alive"]:
                    result.setdefault(row["agent_id"], step["time_s"])
    return result


def test_b_silent_dropout_timing_and_standoff():
    run = _dropout_run("silent")
    detections = _first_missing(run)
    assert all(10.45 < value <= 10.70 for value in detections.values())
    assert all(
        row["standoff_error_m"] <= 0.3
        for step in run["steps"]
        if step["time_s"] >= 16.0
        for row in step["agents"]
        if row["active"] and row["agent_id"] != "drone-001"
    )


def test_c_landed_dropout_is_faster_than_silent():
    landed = _first_missing(_dropout_run("landed"))
    silent = _first_missing(_dropout_run("silent"))
    assert all(10.05 <= value <= 10.10 for value in landed.values())
    assert all(landed[key] < silent[key] - 0.3 for key in landed)


def test_d_lost_leaving_detection_uses_last_state_receipt():
    config = CoopSwarmConfig(
        n_agents=4,
        seed=1,
        duration_s=7.0,
        log_every=1,
        dropouts=(("drone-001", 5.0, "landed"),),
        link=CoopLinkConfig(
            delay_s=0.05, loss_probability=0.5, loss_model="independent"
        ),
    )
    run = CoopSwarmSimulator(config).run().as_dict()
    for receiver in config.agent_ids[1:] + ("gs",):
        leaving = next(
            event for event in run["link_events"]
            if event["kind"] == "leaving" and event["receiver"] == receiver
        )
        state_receipts = [
            event["receive_time"]
            for event in run["link_events"]
            if event["kind"] == "state"
            and event["sender"] == "drone-001"
            and event["receiver"] == receiver
            and event["receive_time"] is not None
        ]
        threshold = (
            leaving["receive_time"]
            if leaving["outcome"] == "received"
            else max(state_receipts) + 0.5
        )
        observed = next(
            step["time_s"] for step in run["steps"]
            if step["time_s"] >= threshold
            and (
                receiver == "gs"
                or next(
                    row for row in step["agents"]
                    if row["agent_id"] == receiver
                )["active"]
            )
            and (
                receiver == "gs"
                or "drone-001" not in next(
                    row for row in step["agents"]
                    if row["agent_id"] == receiver
                )["believed_alive"]
            )
        )
        assert observed >= threshold - 1e-9


def test_e_antipodal_swap_has_no_separation_violation(monkeypatch):
    import sim.coop.simulator as simulator_module
    original_reallocate = simulator_module.reallocate

    def swap(ids, positions, slots, previous, *, switch_penalty_m):
        if len(ids) == 6:
            return {aid: (index + 3) % 6 for index, aid in enumerate(ids)}
        return original_reallocate(
            ids, positions, slots, previous, switch_penalty_m=switch_penalty_m
        )

    monkeypatch.setattr(simulator_module, "reallocate", swap)
    config = CoopSwarmConfig(
        n_agents=6,
        duration_s=12.0,
        target="cv",
        target_params=(("speed", 1e-6),),
        initial_radius_m=1.5,
        standoff_radius_m=1.5,
        keep_out_radius_m=0.0,
        noise_std_m=0.0,
        dropouts=(),
        link=CoopLinkConfig(delay_s=0.0, loss_probability=0.0, loss_model="none"),
    )
    run = CoopSwarmSimulator(config).run().as_dict()
    assert run["summary"]["spherical_violations_count"] == 0
    assert run["summary"]["separation_violations_count"] == 0
    assert run["summary"]["min_pairwise_separation_m"] >= 0.35 - 1e-6
    step = min(run["steps"], key=lambda row: abs(row["time_s"] - 10.0))
    positions = {
        row["agent_id"]: np.asarray(row["position_m"])
        for row in step["agents"] if row["active"]
    }
    ids = sorted(positions)
    truth = np.asarray(step["target_truth_position_m"])
    assert all(
        np.linalg.norm(
            positions[ids[i]] + positions[ids[i + 3]] - 2.0 * truth
        ) <= 0.3
        for i in range(3)
    )


def test_f_ground_plan_matches_latest_fresh_command():
    config = CoopSwarmConfig(
        decision="ground",
        fusion="central",
        duration_s=20.0,
        dropouts=(("drone-001", 10.0, "landed"),),
        link=CoopLinkConfig(delay_s=0.05, loss_probability=0.3),
    )
    simulator = CoopSwarmSimulator(config)
    run = simulator.run().as_dict()
    mismatches = 0
    for step in run["steps"]:
        for row in step["agents"]:
            if not row["active"]:
                continue
            commands = [
                event for event in run["link_events"]
                if event["kind"] == "command"
                and event["receiver"] == row["agent_id"]
                and event["receive_time"] is not None
                and event["receive_time"] <= step["time_s"] + 1e-9
            ]
            if not commands or step["time_s"] - commands[-1]["receive_time"] > 0.5 + 1e-9:
                continue
            message = simulator._messages[("gs", row["agent_id"], commands[-1]["seq"])]
            mismatches += message["slot_index"] != row["slot_index"]
    assert mismatches == 0


def test_g_central_fallback_during_gs_outage():
    config = CoopSwarmConfig(
        fusion="central", decision="onboard", duration_s=18.0,
        gs_outage=((10.0, 15.0),), dropouts=(), log_every=1,
    )
    run = CoopSwarmSimulator(config).run().as_dict()
    rows = [
        row for step in run["steps"] if 10.0 <= step["time_s"] <= 15.0
        for row in step["agents"] if row["active"]
    ]
    assert max(row["target_age_s"] for row in rows) < 0.6
    assert max(row["truth_error_m"] for row in rows) < 0.2


def test_h_outage_timeout_and_standoff_growth():
    for decision in ("ground", "onboard"):
        config = CoopSwarmConfig(
            decision=decision, fusion="central", duration_s=18.0,
            gs_outage=((10.0, 15.0),), dropouts=(), log_every=1,
            link=CoopLinkConfig(delay_s=0.05, loss_probability=0.0, loss_model="none"),
        )
        run = CoopSwarmSimulator(config).run().as_dict()
        timeout = [
            step["time_s"] for step in run["steps"]
            if any("command_timeout" in row["flags"] for row in step["agents"] if row["active"])
        ]
        if decision == "ground":
            assert 10.45 <= min(timeout) <= 10.55
            assert max(timeout) <= 15.25
        values = [
            row["standoff_error_m"] for step in run["steps"]
            if 14.0 <= step["time_s"] <= 15.0
            for row in step["agents"] if row["active"]
        ]
        if decision == "ground":
            assert np.median(values) > 1.0
        else:
            assert np.median(values) < 0.3


def test_i_traffic_counts_and_bytes_after_leaving():
    config = CoopSwarmConfig(
        n_agents=3, duration_s=1.0, decision="ground", fusion="central",
        dropouts=(("drone-001", 0.5, "landed"),),
        link=CoopLinkConfig(delay_s=0.05, loss_probability=0.0, loss_model="none"),
    )
    run = CoopSwarmSimulator(config).run().as_dict()
    traffic = run["summary"]["traffic"]
    expected = {"state": 71, "track": 33, "command": 25, "leaving": 3}
    for kind, count in expected.items():
        assert traffic[kind]["messages"] == count
        assert traffic[kind]["bytes"] == count * traffic[kind]["message_bytes"]
    assert all(event["kind"] in expected or event["kind"] == "raw" for event in run["link_events"])


def test_j_gs_fuser_matches_target_estimate():
    config = CoopSwarmConfig(
        n_agents=1, fusion="central", duration_s=3.0, dropouts=(),
        link=CoopLinkConfig(delay_s=0.0, loss_probability=0.0, loss_model="none"),
    )
    run = CoopSwarmSimulator(config).run().as_dict()
    for step in run["steps"]:
        assert np.max(
            np.abs(
                np.asarray(step["gs_fuser_estimate_m"])
                - np.asarray(step["agents"][0]["target_estimate_m"])
            )
        ) <= 1e-12


def test_k_every_non_silent_agent_moves_most_logged_steps():
    run = CoopSwarmSimulator(CoopSwarmConfig(log_every=1)).run().as_dict()
    for aid in CoopSwarmConfig().agent_ids:
        rows = [
            np.asarray(row["position_m"])
            for step in run["steps"]
            for row in step["agents"]
            if row["agent_id"] == aid and row["active"]
            and (aid != "drone-001" or step["time_s"] < 10.0)
        ]
        moved = sum(np.linalg.norm(rows[i] - rows[i - 1]) > 1e-12 for i in range(1, len(rows)))
        assert moved >= 0.9 * max(0, len(rows) - 1)
