"""Fast checks for the sending experiment sweep and its registered policies."""
from __future__ import annotations

import json
import shlex

import pytest

from sim.estimation.base import ArmSpec, NAIVE
from sim.estimation.engine import CellSpec, deliver_cell, run_cell
from sim.s6_recovery import RecoveryConfig, seed_list
from sim.estimation import sweep
from tools import run_s6_sending as runner


def _cell_spec(record: dict) -> CellSpec:
    fields = set(CellSpec.__dataclass_fields__)
    values = {key: value for key, value in record.items() if key in fields}
    return CellSpec(**values)


def _small_cells() -> list[dict]:
    cells = runner.build_cells(pilot=True)
    assert cells
    return cells[: min(3, len(cells))]


def test_registered_grid_size_and_unique_keys():
    cells = runner.build_cells()
    assert len(cells) == 504
    assert {record["slice"] for record in cells} == {"main"}
    keys = [_cell_spec(record).key() for record in cells]
    assert len(set(keys)) == len(keys)
    pilot = runner.build_cells(pilot=True)
    assert pilot
    assert {record["slice"] for record in pilot} == {"main"}
    assert len(pilot) < len(cells)


def test_registered_arms_have_exact_names():
    arms = runner.build_arms()
    assert [arm.name for arm in arms] == ["naive", "recovery", "imm"]
    assert all(isinstance(arm, ArmSpec) for arm in arms)
    assert arms[-1].model == "imm"


def test_naive_arm_is_shared_receive_stamped_baseline():
    arms = runner.build_arms()
    assert arms[0] == NAIVE
    assert arms[0].stamping == "receive"


def test_manifest_replay_command_includes_runner_path(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(runner, "build_cells", lambda pilot=False: [])
    monkeypatch.setattr(runner, "run_jobs", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "write_manifest", lambda *args, **kwargs: captured.update(kwargs) or {})
    args = ["--output", str(tmp_path / "out"), "--pilot", "--seeds", "2"]
    runner.run_sweep(tmp_path / "out", pilot=True, seeds_count=2, no_checkpoint=True, argv=args)
    assert captured["extra"]["replay_command"] == "PYTHONPATH=. ./.venv/bin/python tools/run_s6_sending.py " + shlex.join(args)


def test_job_result_is_json_serializable():
    record = _small_cells()[0]
    result = runner.job_cell(record)
    json.dumps(result, allow_nan=False, sort_keys=True)


def test_small_sweep_outputs_are_byte_identical(tmp_path, monkeypatch):
    cells = _small_cells()
    monkeypatch.setattr(runner, "build_cells", lambda pilot=False: cells)
    first = tmp_path / "first"
    second = tmp_path / "second"
    runner.run_sweep(first, pilot=True, seeds_count=2, processes=1, no_checkpoint=True)
    runner.run_sweep(second, pilot=True, seeds_count=2, processes=2, no_checkpoint=True)
    names = ["summary.json", "cells.csv", "per_seed.csv.gz"]
    assert sweep.compare_outputs(first, second, names) == {name: True for name in names}
    assert (first / "manifest.json").exists()


def test_per50_float_matches_unspecified_sender_for_each_seed():
    record = next(item for item in runner.build_cells() if item.get("policy") == "per50_f32")
    sent = _cell_spec(record)
    baseline = CellSpec(**{name: getattr(sent, name) for name in CellSpec.__dataclass_fields__ if name != "sender"})
    arms = runner.build_arms()
    seeds = seed_list(2)
    sent_result = run_cell(sent, arms, seeds, RecoveryConfig())
    base_result = run_cell(baseline, arms, seeds, RecoveryConfig())
    for arm in ("naive", "recovery", "imm"):
        left = [row["p95_error_m"] for row in sent_result["per_seed"] if row["arm"] == arm]
        right = [row["p95_error_m"] for row in base_result["per_seed"] if row["arm"] == arm]
        assert left == pytest.approx(right, abs=1e-12)


def test_per10_budget_is_ten_packets_per_second_before_loss():
    record = next(item for item in runner.build_cells() if item.get("policy") == "per10_f32")
    result = runner.job_cell({**record, "seeds": seed_list(2)})
    budget = result["budget"]
    assert budget["packets_per_second_median"] == pytest.approx(10.0, rel=0.01)
    assert budget["data_bytes_per_second_median"] == pytest.approx(180.0, rel=0.01)
    assert budget["on_air_bytes_per_second_median"] == pytest.approx(610.0, rel=0.01)
    for row in result["per_seed"]:
        assert {"sent_count", "packets_per_second", "data_bytes_per_second", "on_air_bytes_per_second"} <= row.keys()


def test_event_zero_delta_matches_per50_quantized_policy():
    event_record = next(item for item in runner.build_cells() if item.get("policy") == "evt2cm_f32")
    event = _cell_spec(event_record)
    periodic = CellSpec(**{name: getattr(event, name) for name in CellSpec.__dataclass_fields__ if name != "sender"},
                        sender=(("kind", "periodic"), ("rate_hz", 50.0)))
    event_zero = CellSpec(**{name: getattr(event, name) for name in CellSpec.__dataclass_fields__ if name != "sender"},
                          sender=(("delta_m", 0.0), ("kind", "event")))
    arms = runner.build_arms()
    seeds = seed_list(2)
    a = run_cell(event_zero, arms, seeds, RecoveryConfig())
    b = run_cell(periodic, arms, seeds, RecoveryConfig())
    for arm in ("naive", "recovery", "imm"):
        av = [row["p95_error_m"] for row in a["per_seed"] if row["arm"] == arm]
        bv = [row["p95_error_m"] for row in b["per_seed"] if row["arm"] == arm]
        assert av == pytest.approx(bv, abs=1e-12)
