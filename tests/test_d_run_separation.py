"""Output and reproducibility checks for the separation-buffer runner."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from sim.coop.separation import EncounterSpec, run_encounter
from sim.s6_recovery import wilson_interval
from tools.d_common import seeds_for
from tools.d_run_separation import MIN_SEPARATION_M, search_radius_table, separation_grid
from tools.d_make_separation_figure import make_figures

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/Users/akashi0281/Desktop/Zephyr Mesh/Zephyr-Mesh/.venv/bin/python"


def _run_pilot(path: Path) -> str:
    command = [PYTHON, "tools/d_run_separation.py", "--output", str(path), "--pilot", "--processes", "2"]
    return subprocess.run(command, cwd=ROOT, env={"PYTHONPATH": "."}, check=True,
                          capture_output=True, text=True).stdout


def test_search_logic_and_wilson_rule():
    assert wilson_interval(0, 75)[1] <= 0.05
    assert wilson_interval(1, 75)[1] > 0.05
    table = [{"avoidance_radius_m": 0.35, "violations": 1, "seeds_run": 75, "first_violation_seed_index": 3},
             {"avoidance_radius_m": 0.40, "violations": 0, "seeds_run": 75, "first_violation_seed_index": None}]
    result = search_radius_table(table)
    assert result["required_buffer_m"] == 0.05 and result["failing_radius_m"] == 0.35
    assert result["censored"] is False and result["max_radius_searched_m"] == 0.4
    assert result["first_violation_seed_index"] == 3
    censored = search_radius_table([{"avoidance_radius_m": 0.35, "violations": 1, "seeds_run": 1}])
    assert censored["censored"] is True and censored["required_buffer_m"] is None
    assert censored["max_radius_searched_m"] == 1.5


def test_grid_is_fixed_and_ring_first():
    full = separation_grid()
    assert full[0]["kind"] == "ring_swap"
    assert any(c["loss_probability"] == 0.1 for c in full)
    assert any(c["kind"] == "head_on" and c["speed_mps"] == 0.5 and c["delay_s"] == 0.0 and c["lateral_offset_m"] == 0.0 for c in full)
    assert any(c["kind"] == "head_on" and c["lateral_offset_m"] == 0.1 for c in full)
    assert len(full) == 4 * 2 * 2 + 2 * 3 * 6 * 3 * 2 * 2 + 3 * 6 * 3 * 2 * 2


def test_pilot_is_byte_stable_and_schema_valid(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _run_pilot(first)
    _run_pilot(second)
    a = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    b = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert a["n_processes"] <= 4 and b["n_processes"] <= 4
    for key in ("runtime_s", "git_revision", "git_tree_state", "platform", "n_processes", "argv"):
        a.pop(key, None)
        b.pop(key, None)
    assert a == b
    for name in a["outputs"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    for name in ("separation_buffer.png", "separation_baseline.png"):
        content = (first / name).read_bytes()
        assert len(content) > 10_000 and content[:8] == b"\x89PNG\r\n\x1a\n"
    value = json.loads((first / "buffer.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "sim/coop/buffer_schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(value)) == []
    assert value["note"] and all("description" in spec for spec in schema["properties"].values())
    for entry in value["entries"]:
        assert "kinds_included" not in entry
    assert all("kinds_included" in row for row in value["envelope"])


def test_one_written_row_recomputed_from_run_encounter(tmp_path):
    output = tmp_path / "pilot"
    _run_pilot(output)
    with (output / "encounters.csv").open(encoding="utf-8", newline="") as handle:
        row = next(item for item in csv.DictReader(handle) if item["kind"] == "head_on")
    condition = EncounterSpec(kind=row["kind"], speed_mps=float(row["speed_mps"]), delay_s=float(row["delay_s"]),
                              loss_probability=float(row["loss_probability"]), noise_std_m=float(row["noise_std_m"]),
                              dead_reckoning=row["dead_reckoning"].lower() == "true",
                              avoidance_radius_m=float(row["avoidance_radius_m"]),
                              lateral_offset_m=float(row["lateral_offset_m"]))
    seeds = seeds_for("separation", 2)
    results = [run_encounter(condition, seed) for seed in seeds]
    assert int(row["violations"]) == sum(bool(result["violated"]) for result in results)
    assert int(row["seeds_run"]) == len(results)
    assert float(row["min_center_distance_m"]) == min(float(result["min_center_distance_m"]) for result in results)


def test_printed_estimate_matches_measured_formula(tmp_path):
    from tools.d_run_separation import N_SEEDS, RADIUS_COUNT

    output = tmp_path / "estimate"
    stdout = _run_pilot(output)
    line = next(line for line in stdout.splitlines() if "full-run worst-case estimate:" in line)
    printed = float(line.split("full-run worst-case estimate: ", 1)[1].split(" CPU s", 1)[0])
    rates_text = line.split("measured CPU s per encounter by kind: ", 1)[1]
    rates = {item.split("=", 1)[0]: float(item.split("=", 1)[1])
             for item in rates_text.split(", ")}
    full = separation_grid()
    counts = Counter(cell["kind"] for cell in full)
    expected_worst = sum(rates[kind] * count * N_SEEDS * RADIUS_COUNT
                         for kind, count in counts.items())
    expected_best = sum(rates[kind] * count * N_SEEDS
                        for kind, count in counts.items())
    best_line = next(line for line in stdout.splitlines() if "full-run best-case estimate:" in line)
    printed_best = float(best_line.split("full-run best-case estimate: ", 1)[1].split(" CPU s", 1)[0])
    assert printed == pytest.approx(round(expected_worst, 1), abs=0.05)
    assert printed_best == pytest.approx(round(expected_best, 1), abs=0.05)


def test_processes_are_capped(tmp_path):
    import tools.d_run_separation as runner

    captured = {}
    conditions = separation_grid(pilot=False)

    def fake_run_jobs(jobs, target, *, processes, checkpoint_dir, fingerprint, progress):
        captured["processes"] = processes
        return [{"baseline": {"seeds_run": 75}} for _ in jobs]

    original_run_jobs = runner.run_jobs
    original_write_outputs = runner._write_outputs
    runner.run_jobs = fake_run_jobs
    runner._write_outputs = lambda *args, **kwargs: None
    try:
        runner.run(tmp_path / "cap", processes=6)
    finally:
        runner.run_jobs = original_run_jobs
        runner._write_outputs = original_write_outputs
    assert len(conditions) > 0
    assert captured["processes"] == 4


def test_buffer_figure_changes_when_buffer_data_change(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_pilot(first)
    shutil.copytree(first, second)
    buffer_path = second / "buffer.json"
    data = json.loads(buffer_path.read_text(encoding="utf-8"))
    assert data["envelope"]
    data["envelope"][0]["required_buffer_m"] = 0.99
    data["envelope"][0]["censored"] = False
    buffer_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_figures(second)
    assert (first / "separation_buffer.png").read_bytes() != (second / "separation_buffer.png").read_bytes()
