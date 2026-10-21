"""Proposal check: EXP-016 gives the same bytes at 1 and several --workers.

The runner's independent cells go through sim.ordered_pool.ordered_map, which
returns results in job order. These tests compare files by sha256, so they
catch any change in row order, column order or number formatting.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from sim.ordered_pool import ordered_map
from tools import c_exp016_heartbeat as runner

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("summary.json", "false_alarms.csv", "detection_delay.csv", "interference.csv",
           "thresholds.json", "exp016_false_alarms.png", "exp016_rule_300ms.png")


def _hashes(directory: Path) -> dict[str, str]:
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in OUTPUTS}


def _registered() -> dict:
    entries = json.loads((ROOT / "tools" / "reproduce_registry.json").read_text(encoding="utf-8"))
    return next(e for e in entries if e["run_dir"] == "runs/c5-heartbeat")


def test_pilot_is_byte_identical_for_one_and_three_workers(tmp_path: Path) -> None:
    serial, parallel = tmp_path / "w1", tmp_path / "w3"
    runner.main(["--output", str(serial), "--pilot", "--workers", "1"])
    runner.main(["--output", str(parallel), "--pilot", "--workers", "3"])
    assert _hashes(serial) == _hashes(parallel)


def test_interference_cells_match_in_process_and_in_pool() -> None:
    # The pilot has one interference cell, which never reaches the pool, so
    # run several tiny cells here through both paths.
    jobs = [{"margin": m, "dur": 1, "rise": r, "pilot": True} for m in (20, 30) for r in (10, 20)]
    serial = ordered_map("tools.c_exp016_heartbeat:_interference_job", jobs, workers=1)
    pooled = ordered_map("tools.c_exp016_heartbeat:_interference_job", jobs, workers=2)
    assert json.dumps(serial) == json.dumps(pooled)
    assert [(row["margin_db"], row["noise_rise_db"]) for row in pooled] == [(20, 10), (20, 20), (30, 10), (30, 20)]


def test_registered_command_uses_the_serial_default() -> None:
    # The registered replay passes no --workers, so it runs the default (1 worker).
    assert "--workers" not in _registered()["command"]


@pytest.mark.skipif(os.environ.get("ZEPHYR_FULL_REPRO") != "1",
                    reason="full EXP-016 run takes minutes; set ZEPHYR_FULL_REPRO=1")
@pytest.mark.parametrize("workers", [1, 4])
def test_full_run_reproduces_registered_hashes(tmp_path: Path, workers: int) -> None:
    out = tmp_path / f"w{workers}"
    runner.main(["--output", str(out), "--workers", str(workers)])
    expected = _registered()["outputs"]
    assert {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in expected} == expected
