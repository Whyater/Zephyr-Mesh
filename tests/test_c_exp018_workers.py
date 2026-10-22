"""Proposal check: EXP-018 gives the same bytes on the old pool path and on the
opt-in sim.ordered_pool path at 1 and 4 workers (files compared by sha256)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tools import c_exp018_recovery as runner

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("summary.json", "recovery.csv", "sweeps.csv", "exp018_coverage.png", "exp018_design.png")


def _hashes(directory: Path) -> dict[str, str]:
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in OUTPUTS}


def _registered() -> dict:
    entries = json.loads((ROOT / "tools" / "reproduce_registry.json").read_text(encoding="utf-8"))
    return next(e for e in entries if e["run_dir"] == "sim/runs/c-exp018-recovery-r2")


def test_pilot_is_byte_identical_for_legacy_and_opt_in_workers(tmp_path: Path) -> None:
    legacy, one, four = tmp_path / "legacy", tmp_path / "w1", tmp_path / "w4"
    runner.main(["--output", str(legacy), "--pilot", "--processes", "2"])
    runner.main(["--output", str(one), "--pilot", "--workers", "1"])
    runner.main(["--output", str(four), "--pilot", "--workers", "4"])
    assert _hashes(legacy) == _hashes(one) == _hashes(four)


def test_registered_command_is_legacy_default() -> None:
    assert "--workers" not in _registered()["command"]


@pytest.mark.skipif(os.environ.get("ZEPHYR_FULL_REPRO") != "1",
                    reason="full EXP-018 run takes minutes; set ZEPHYR_FULL_REPRO=1")
@pytest.mark.parametrize("workers", [1, 4])
def test_full_run_reproduces_registered_hashes(tmp_path: Path, workers: int) -> None:
    out = tmp_path / f"w{workers}"
    runner.main(["--output", str(out), "--workers", str(workers)])
    expected = _registered()["outputs"]
    assert _hashes(out) == {name: expected[name] for name in OUTPUTS}
