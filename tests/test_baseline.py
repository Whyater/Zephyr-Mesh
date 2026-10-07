import json

import numpy as np

from sim.baseline import BaselineConfig, build_manifest, run_baseline, write_baseline


def test_baseline_is_deterministic_and_has_expected_samples():
    config = BaselineConfig(dt=0.1, total_time=0.3)
    first = run_baseline(config)
    second = run_baseline(config)
    assert len(first["time"]) == 3
    assert np.array_equal(first["z"], second["z"])
    assert np.all(np.isfinite(first["z"]))


def test_manifest_records_current_known_issues():
    manifest = build_manifest(BaselineConfig())
    assert manifest["schema"] == "zephyr-s0-baseline-1"
    assert len(manifest["known_issues"]) == 2
    assert manifest["config"]["dt"] == 0.01


def test_writer_emits_machine_readable_outputs(tmp_path):
    write_baseline(tmp_path, BaselineConfig(dt=0.1, total_time=0.3))
    assert (tmp_path / "telemetry.csv").read_text().startswith("time,x,y,z")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["config"]["total_time"] == 0.3
    assert manifest["independent_check"]["passed"]
    assert (tmp_path / "manifest.sha256").exists()
