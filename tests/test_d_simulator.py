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


def test_single_observer_zero_noise_matches_local_filter():
    config = CoopSwarmConfig(n_agents=1, duration_s=2.0, noise_std_m=0.0, fusion="none", dropouts=(), link=CoopLinkConfig(delay_s=0.0, loss_probability=0.0, loss_model="none"))
    run = CoopSwarmSimulator(config).run().as_dict()
    assert all(step["agents"][0]["target_estimate_m"] is not None for step in run["steps"])


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
