"""Checks for deterministic S0 and S2 v2 artifact generators."""

import hashlib
import json
from pathlib import Path

import numpy as np

from sim.baseline import run_scenario_v2
from sim.physics.run_manifest import canonical_json_sha256, write_manifest
from tools.a_make_s0_baseline_v2 import generate as generate_s0
from tools.a_make_s2_physics_v2 import generate as generate_s2


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_config_hash_and_output_hashes(tmp_path):
    artifact = tmp_path / "data.bin"
    artifact.write_bytes(b"manifest test")
    path = write_manifest(
        tmp_path / "manifest.json",
        schema="test-manifest-1",
        argv=["example"],
        config={"b": 2, "a": 1},
        seeds=[],
        limits=["synthetic"],
        outputs=[artifact],
        output_dir=tmp_path,
        replay_command="example",
        runtime_s=0.1,
    )
    document = json.loads(path.read_text())
    assert document["config_sha256"] == canonical_json_sha256({"a": 1, "b": 2})
    assert document["outputs"]["data.bin"] == _digest(artifact)
    assert path.read_bytes().endswith(b"\n")
    assert "runtime_s" in document["volatile_fields"]


def test_s0_v2_outputs_are_byte_identical(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    generate_s0(first)
    generate_s0(second)
    names = {path.name for path in first.iterdir()} - {"manifest.json"}
    assert names
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_lateral_step_has_expected_linearized_response(tmp_path):
    generate_s0(tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    lateral = report["scenarios"]["lateral"]
    # The attitude loop adds lag to x'' + 1.5x' + 2x = 2, so allow a broad
    # band around the hand linearization's 13.6 percent overshoot.
    assert 10.0 <= lateral["overshoot_pct"] <= 25.0
    assert 0.3 <= lateral["rise_time_s"] <= 0.8


def test_scenario_time_and_s2_artifacts(tmp_path):
    telemetry = run_scenario_v2("yaw", 0.01)
    assert len(telemetry["time"]) == 300
    assert telemetry["time"][0] == 0.01
    first = tmp_path / "s2-first"
    second = tmp_path / "s2-second"
    generate_s2(first)
    generate_s2(second)
    for name in ("convergence.json", "a_s2_convergence.png"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    document = json.loads((first / "convergence.json").read_text())
    ratios = [row["error_ratio_to_next"] for row in document["results"][:-1]]
    assert ratios[0] > 10.0
