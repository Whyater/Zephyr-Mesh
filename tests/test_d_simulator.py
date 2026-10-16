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
from sim.coop.simulator import CoopLinkConfig, CoopSwarmConfig, CoopSwarmSimulator
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
    ma, mb = json.loads((a / "manifest.json").read_text()), json.loads((b / "manifest.json").read_text())
    ma.pop("volatile")
    mb.pop("volatile")
    assert ma == mb


def test_both_schemas_validate(tmp_path):
    out = tmp_path / "run"
    _generate(out)
    for name, schema in (("run.json", "sim/s7_coop_run_schema.json"), ("replay_v1.json", "sim/s7_run_schema.json")):
        value = json.loads((out / name).read_text())
        validator = Draft202012Validator(json.loads((ROOT / schema).read_text()))
        assert list(validator.iter_errors(value)) == []


def test_demo_moves_and_re_spaces(tmp_path):
    out = tmp_path / "run"
    _generate(out)
    run = json.loads((out / "run.json").read_text())
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
    config = CoopSwarmConfig(n_agents=6, duration_s=8.0, dropouts=(), link=CoopLinkConfig(delay_s=0.0, loss_probability=0.0, loss_model="none"), avoidance=AvoidanceConfig(ellipsoid_radii=(0.12, 0.12, 0.30)))
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
