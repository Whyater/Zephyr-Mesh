"""Checks for the late-packet re-filtering runner."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sim.s6_recovery import RecoveryConfig, seed_list, simulate_run
from sim.estimation.engine import CellSpec

import tools.run_s6_retro as retro


PYTHON = sys.executable


def _cell(slice_name: str, target: str = "cv", *, delay: float | None = None,
          jitter: float | None = None, loss: float | None = None) -> dict:
    for c in retro.build_cells(False):
        if c["slice"] != slice_name or c.get("target") != target:
            continue
        if delay is not None and float(c["delay_s"]) != delay:
            continue
        if jitter is not None and float(c["jitter_s"]) != jitter:
            continue
        if loss is not None and float(c["loss_probability"]) != loss:
            continue
        return c
    raise AssertionError("requested cell is absent")


def _job_result(cell: dict, n: int = 3) -> dict:
    """Call the public job function, accepting the two common job encodings."""
    seeds = seed_list(n)
    candidates = [
        {**cell, "seeds": n},
        {"cell": cell, "seeds": n},
        {**cell, "n_seeds": n},
        {"cell": cell, "n_seeds": n},
        {**cell, "seeds": seeds},
        {"cell": cell, "seeds": seeds},
    ]
    errors = []
    for job in candidates:
        try:
            result = retro.job_cell(job)
            json.dumps(result, allow_nan=False)
            return result
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(exc)
    raise AssertionError(f"could not call job_cell using supported encodings: {errors!r}")


def _per_seed_arm(result: dict, arm_name: str) -> list[dict]:
    rows = result["per_seed"]
    out = []
    for row in rows:
        if isinstance(row.get("arms"), dict) and arm_name in row["arms"]:
            value = row["arms"][arm_name]
        elif row.get("arm") == arm_name:
            value = row
        elif arm_name in row and isinstance(row[arm_name], dict):
            value = row[arm_name]
        else:
            continue
        out.append(value)
    return out


def test_grid_size_slices_and_unique_keys():
    cells = retro.build_cells(False)
    assert len(cells) == 56
    assert {c["slice"] for c in cells} == {"J0", "J1", "J2"}
    assert {s: sum(c["slice"] == s for c in cells) for s in ("J0", "J1", "J2")} == {
        "J0": 8, "J1": 24, "J2": 24
    }
    assert len({c["key"] if "key" in c else CellSpec(**{k: v for k, v in c.items() if k != "slice"}).key() for c in cells}) == 56
    pilot = retro.build_cells(True)
    assert len(pilot) == 5
    assert {c["slice"] for c in pilot} == {"J0", "J1", "J2"}
    assert [a.name for a in retro.build_arms()] == ["naive", "recovery", "retro"]


def test_job_result_is_json_serializable_and_census_rules():
    j0 = _job_result(_cell("J0", delay=0.1, loss=0.0), 1)
    json.dumps(j0, allow_nan=False)
    assert j0["reorder_census"]["fraction"] == 0
    j1 = _job_result(_cell("J1", target="cv", delay=0.3, jitter=0.15, loss=0.0), 1)
    assert j1["reorder_census"]["fraction"] > 0
    assert j1["reorder_census"]["delivered"] >= j1["reorder_census"]["count"]


def test_census_reports_per_seed_summary_and_rows():
    result = _job_result(_cell("J1", target="cv", delay=0.3, jitter=0.15, loss=0.0), 3)
    census = result["reorder_census"]
    fractions = census["per_seed"]
    assert len(fractions) == 3
    assert census["mean"] == pytest.approx(sum(fractions) / len(fractions))
    assert census["min"] == min(fractions)
    assert census["max"] == max(fractions)
    rows = [row for row in result["per_seed"] if row["arm"] == "recovery"]
    assert [row["reorder_fraction"] for row in rows] == fractions


def test_j0_retro_and_recovery_match_per_seed():
    result = _job_result(_cell("J0", delay=0.3, loss=0.3), 3)
    retro_rows = _per_seed_arm(result, "retro")
    recovery_rows = _per_seed_arm(result, "recovery")
    assert len(retro_rows) == len(recovery_rows) == 3
    for a, b in zip(retro_rows, recovery_rows):
        assert a["p95_error_m"] == pytest.approx(b["p95_error_m"], abs=0, rel=0)
        assert a["max_error_m"] == pytest.approx(b["max_error_m"], abs=0, rel=0)


def test_j1_late_count_matches_direct_fused_sequence_check():
    cell = _cell("J1", target="cv", delay=0.3, jitter=0.15, loss=0.0)
    diagnostics = retro._late_drop_diagnostics(
        CellSpec(**{k: v for k, v in cell.items() if k != "slice"}),
        seed_list(3),
        RecoveryConfig(),
        retro.retro_window_for(CellSpec(**{k: v for k, v in cell.items() if k != "slice"})),
        retro.EstimatorSettings(),
    )
    assert len(diagnostics) == 3
    retro_arm = next(arm for arm in retro.build_arms() if arm.name == "retro")
    for seed, row in zip(seed_list(3), diagnostics):
        spec = CellSpec(**{k: v for k, v in cell.items() if k != "slice"})
        bundle = retro.deliver_cell(spec, seed, RecoveryConfig(), retro.EstimatorSettings())
        detailed = retro.run_arm_detailed(bundle, spec, retro_arm, RecoveryConfig(), retro.EstimatorSettings())
        fused = {packet.seq for packet in detailed.fused}
        newest = float("-inf")
        expected = 0
        for delivery in bundle.deliveries:
            if delivery.receive_time_s > RecoveryConfig().duration_s + retro.TIME_TOLERANCE_S:
                continue
            stamp = float(delivery.stamp_s)
            if (delivery.seq not in fused and newest != float("-inf")
                    and stamp >= newest - retro.retro_window_for(spec)):
                expected += 1
            if delivery.seq in fused:
                newest = max(newest, stamp)
        assert row["late_within_window_dropped"] == expected


def test_s6_cross_check_first_three_seeds():
    cell = _cell("J1", target="cv", delay=0.3, jitter=0.15, loss=0.0)
    result = _job_result(cell, 3)
    seeds = seed_list(3)
    expected = []
    for seed in seeds:
        arms = simulate_run("constant_velocity", delay_s=0.3, loss_probability=0.0,
                            noise_std_m=0.05, seed=seed, config=RecoveryConfig(),
                            loss_model="independent", jitter_s=0.15)
        expected.append((arms["naive"].p95_error_m, arms["recovery"].p95_error_m))
    naive = _per_seed_arm(result, "naive")
    recovery = _per_seed_arm(result, "recovery")
    assert len(naive) == len(recovery) == 3
    for i, (n, r) in enumerate(zip(naive, recovery)):
        assert n["p95_error_m"] == pytest.approx(expected[i][0], abs=1e-12)
        assert r["p95_error_m"] == pytest.approx(expected[i][1], abs=1e-12)


def test_small_sweep_outputs_are_byte_identical(tmp_path, monkeypatch):
    cells = retro.build_cells(False)[:2]
    monkeypatch.setattr(retro, "build_cells", lambda pilot=False: cells)
    run = getattr(retro, "run_sweep", None)
    if run is None:
        pytest.skip("runner exposes only its command line interface")

    def invoke(out: Path):
        sig = inspect.signature(run)
        kwargs = {"processes": 1}
        if "seeds_count" in sig.parameters:
            kwargs["seeds_count"] = 2
        elif "seeds" in sig.parameters:
            kwargs["seeds"] = 2
        if "checkpoint_dir" in sig.parameters:
            kwargs["checkpoint_dir"] = None
        if "no_checkpoint" in sig.parameters:
            kwargs["no_checkpoint"] = True
        run(out, **kwargs)

    a, b = tmp_path / "a", tmp_path / "b"
    invoke(a)
    invoke(b)
    names = ["summary.json", "cells.csv", "per_seed.csv.gz"]
    assert all((a / name).is_file() for name in names)
    assert all((b / name).is_file() for name in names)
    from sim.estimation.sweep import compare_outputs
    assert all(compare_outputs(a, b, names).values())
    manifest = json.loads((a / "manifest.json").read_text())
    assert manifest["replay_command"].startswith(
        "PYTHONPATH=. ./.venv/bin/python tools/run_s6_retro.py "
    )
    summary = json.loads((a / "summary.json").read_text())
    assert summary["failure_rule"] == {
        "p95_m": RecoveryConfig().failure_p95_m,
        "max_m": RecoveryConfig().failure_max_m,
        "cell_rule": "a cell fails when at least 50 % of seeds fail",
        "cell_rule_status": "proposed, not yet adopted",
    }
