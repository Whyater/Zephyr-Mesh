"""EXP-031 checks for the simulated area sweep mission.

The tests exercise the public mission runner and compare the clean limiting
case with the independent area-rate hand check.  All scenarios are simulated.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pytest

from sim.missions.api import Scenario
from sim.missions.engine import run_mission
from sim.missions.log import validate_log

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_e import hand_checks as hc  # noqa: E402


_SEED_BASE = 20261031


@pytest.fixture(scope="module", autouse=True)
def _report_module_wall_time():
    started = time.perf_counter()
    yield
    elapsed = time.perf_counter() - started
    print(f"[wall] test_e_sweep.py total: {elapsed:.3f}s")


@pytest.fixture(autouse=True)
def _report_test_wall_time(request):
    started = time.perf_counter()
    yield
    elapsed = time.perf_counter() - started
    print(f"[wall] {request.node.name}: {elapsed:.3f}s")


def _scenario(*, seed=_SEED_BASE, preset="clean", dropout="none", mode="delta", duration_s=120.0):
    return Scenario(
        mission="sweep",
        seed=seed,
        duration_s=duration_s,
        preset=preset,
        params={"dropout": dropout, "message_mode": mode},
    )


def _run(*, seed=_SEED_BASE, preset="clean", dropout="none", mode="delta", log=False):
    return run_mission(
        _scenario(seed=seed, preset=preset, dropout=dropout, mode=mode),
        log=log,
    )


def test_clean_hand_check_and_completion_time():
    """Clean sweeping must take at least the area-rate lower bound."""
    reference_s = hc.coverage_time_s(800, 3, 4.0, 2.0)
    result = _run(preset="clean", dropout="none", mode="delta")
    t95_s = result.metrics["t95_s"]
    t100_s = result.metrics["t100_s"]
    assert t95_s is not None
    assert t100_s is not None
    assert t95_s >= reference_s * 0.95
    assert reference_s <= t100_s <= 50.0
    print(f"hand coverage reference={reference_s:.6f}s t95_s={t95_s:.6f}s t100_s={t100_s:.6f}s")


def test_no_dropout_loss_does_not_change_duplication():
    clean = _run(preset="clean", dropout="none", mode="delta")
    iid_30 = _run(preset="iid_30", dropout="none", mode="delta")
    clean_fraction = float(clean.metrics["duplicated_fraction"])
    iid_fraction = float(iid_30.metrics["duplicated_fraction"])
    print(f"duplicated_fraction clean={clean_fraction:.6f} iid_30={iid_fraction:.6f}")
    assert abs(clean_fraction - iid_fraction) <= 0.01


def test_clean_dropout_replans_and_finishes_without_belief_error():
    result = _run(preset="clean", dropout="one", mode="delta")
    assert int(result.metrics["replans"]) >= 1
    assert result.metrics["t100_s"] is not None
    assert int(result.metrics["belief_error_cells"]) == 0
    print(
        "clean dropout replans={} t100_s={:.6f} belief_error_cells={}".format(
            result.metrics["replans"], result.metrics["t100_s"], result.metrics["belief_error_cells"]
        )
    )


def test_iid30_dropout_delta_vs_cumulative():
    delta_rows = []
    cumulative_rows = []
    for offset in range(3):
        seed = _SEED_BASE + offset
        delta = _run(seed=seed, preset="iid_30", dropout="one", mode="delta")
        cumulative = _run(seed=seed, preset="iid_30", dropout="one", mode="cumulative")
        delta_row = (float(delta.metrics["duplicated_fraction"]), int(delta.metrics["belief_error_cells"]))
        cumulative_row = (
            float(cumulative.metrics["duplicated_fraction"]),
            int(cumulative.metrics["belief_error_cells"]),
        )
        delta_rows.append(delta_row)
        cumulative_rows.append(cumulative_row)
        print(f"seed={seed} delta duplicated_fraction={delta_row[0]:.6f} "
              f"belief_error_cells={delta_row[1]}")
        print(f"seed={seed} cumulative duplicated_fraction={cumulative_row[0]:.6f} "
              f"belief_error_cells={cumulative_row[1]}")
    assert sum(row[1] for row in cumulative_rows) <= sum(row[1] for row in delta_rows)


def test_sweep_log_is_deterministic_and_valid():
    first = _run(seed=_SEED_BASE, preset="clean", dropout="none", mode="delta", log=True)
    try:
        validate_log(first.log)
    except ValueError as exc:
        # Keep this skip narrow: it is only for an old schema that has not
        # rebuilt its mission enum to include sweep yet.
        if "mission" in str(exc) and "sweep" in str(exc):
            pytest.skip(f"log schema does not yet accept sweep: {exc}")
        raise
    second = _run(seed=_SEED_BASE, preset="clean", dropout="none", mode="delta", log=True)
    validate_log(second.log)
    assert json.dumps(first.log, sort_keys=True) == json.dumps(second.log, sort_keys=True)

