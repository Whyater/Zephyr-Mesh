"""Tests for the monitor-mission sensor noise side run."""
from __future__ import annotations

import csv
import re

import numpy as np
import pytest

from sim.missions import monitor
from sim.sensor import SensorConfig, SensorModel
from tools import sens_monitor_noise as sens
from tools.e_common import seeds_for


def test_monitor_still_has_both_literals_once():
    """The wrapper patches two 0.5 literals; fail loudly if monitor.py moved or changed them."""
    text = (sens.ROOT / "sim" / "missions" / "monitor.py").read_text(encoding="utf-8")
    assert len(re.findall(r"SensorConfig\(noise_std_m=0\.5\)", text)) == 1
    assert len(re.findall(r"EstimatorSettings\(measurement_std_m=0\.5\)", text)) == 1


def test_job_counts_labels_and_seeds():
    jobs = sens.build_noise_jobs()
    per_arm = 2 * 2 * sens.N_SEEDS
    assert len(jobs) == per_arm * (4 + 2 + 3 + 3)
    assert len(sens.build_noise_jobs(include_posthoc=False)) == per_arm * 4
    seeds = sorted({job["scenario"]["seed"] for job in jobs})
    assert seeds == sorted(seeds_for(33, 50)[:30])
    for job in jobs:
        labels, scenario = job["labels"], job["scenario"]
        assert labels["adapter"] == "point_mass" and scenario["adapter"] == "point_mass"
        assert labels["delay_s"] == 0.0 and scenario["measurement_delay_s"] == 0.0
        assert scenario["params"]["path"] == labels["path"] and scenario["preset"] == labels["preset"]
        assert job["noise"] == {"sensor_sigma_m": labels["sensor_sigma_m"],
                                "estimator_sigma_m": labels["estimator_sigma_m"]}
    split = [j for j in jobs if j["labels"]["arm"] == "posthoc_sensor"]
    assert {j["labels"]["estimator_sigma_m"] for j in split} == {0.5}
    split = [j for j in jobs if j["labels"]["arm"] == "posthoc_estimator"]
    assert {j["labels"]["sensor_sigma_m"] for j in split} == {0.5}


def test_patch_restores_real_classes_and_checks_literal():
    with sens._Patch(2.0, 3.0) as patch:
        assert monitor.SensorConfig(noise_std_m=0.5).noise_std_m == 2.0
        assert monitor.EstimatorSettings(measurement_std_m=0.5).measurement_std_m == 3.0
        with pytest.raises(RuntimeError):
            monitor.SensorConfig(noise_std_m=0.4)
    assert patch.sensor_built == [2.0] and patch.estimator_built == [3.0]
    assert monitor.SensorConfig is SensorConfig
    assert monitor.EstimatorSettings is sens._real_estimator_settings


def test_patched_sensor_noise_has_the_requested_spread():
    """Independent of the recorder: sample the sensor the patched factory builds and measure its std."""
    with sens._Patch(2.0, 0.5):
        config = monitor.SensorConfig(noise_std_m=0.5)
    model = SensorModel(config, seed=11)
    truth = np.array([10.0, 20.0, 15.0])
    errors = np.array([model.sample(truth, 0.1 * k).measurement_m - truth for k in range(4000)])
    assert np.std(errors) == pytest.approx(2.0, rel=0.05)


def test_one_run_records_effective_sigma_and_matches_registered_row():
    """A sigma 0.5 run equals its registered row; a sigma 2.0 run of the same seed records 2.0 for both values."""
    jobs = [job for job in sens.build_noise_jobs(include_posthoc=False, n_seeds=1)
            if job["labels"]["path"] == "crossing" and job["labels"]["preset"] == "clean"]
    rows = [sens.run_noise_job(job) for job in jobs]
    by_sigma = {row["sensor_sigma_m"]: row for row in rows}
    assert by_sigma[2.0]["effective_sensor_sigma_m"] == 2.0
    assert by_sigma[2.0]["effective_estimator_sigma_m"] == 2.0
    assert by_sigma[2.0]["n_sensor_configs_built"] == 2
    assert sens.effective_sigma_check(rows)["passed"]
    control = sens.control_check(rows)
    assert control["n_control_rows"] == 1 and control["passed"]
    assert control["registered_csv_sha256"] == sens.REGISTERED_CSV_SHA256


def test_control_reports_a_tampered_row(tmp_path):
    """The control is not vacuous: one changed value in the reference file is caught."""
    jobs = [job for job in sens.build_noise_jobs(include_posthoc=False, n_seeds=1)
            if job["labels"]["sensor_sigma_m"] == 0.5 and job["labels"]["path"] == "crossing"
            and job["labels"]["preset"] == "clean"]
    rows = [sens.run_noise_job(job) for job in jobs]
    with open(sens.REGISTERED_CSV, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        reference = list(reader)
    seed = str(rows[0]["seed"])
    for row in reference:
        if (row["adapter"], row["path"], row["preset"], row["delay_s"], row["seed"]) == \
                ("point_mass", "crossing", "clean", "0.0", seed):
            row["alerts_raised"] = str(int(row["alerts_raised"]) + 1)
    tampered = tmp_path / "exp033.csv"
    with open(tampered, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(reference)
    result = sens.control_check(rows, tampered)
    assert not result["passed"]
    assert result["n_mismatched_values"] == 1
    assert result["first_mismatches"][0]["column"] == "alerts_raised"


def test_first_failing_sigma():
    def cell(arm, sigma, path, false_count, missed_count=0):
        return {"arm": arm, "sensor_sigma_m": sigma, "estimator_sigma_m": sigma, "path": path,
                "metrics": {"false_alert": {"count": false_count}, "missed_alert": {"count": missed_count}}}
    cells = [cell("planned", 1.0, "skimming", 0), cell("planned", 2.0, "skimming", 0),
             cell("posthoc_matched", 4.0, "skimming", 3), cell("posthoc_matched", 8.0, "skimming", 9),
             cell("planned", 2.0, "crossing", 0, 1)]
    assert sens.first_failing_sigma(cells, ("planned",), "sensor_sigma_m") == {"crossing": 2.0, "skimming": None}
    both = sens.first_failing_sigma(cells, ("planned", "posthoc_matched"), "sensor_sigma_m")
    assert both == {"crossing": 2.0, "skimming": 4.0}


def test_refuses_output_inside_sim_runs():
    with pytest.raises(SystemExit):
        sens.run_sweep(sens.ROOT / "sim" / "runs" / "n11-should-not-exist")
    assert not (sens.ROOT / "sim" / "runs" / "n11-should-not-exist").exists()
