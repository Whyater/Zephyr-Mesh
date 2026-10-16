"""Shared plumbing for the mission experiment scripts (EXP-029 Part B to EXP-034).

Everything here is simulated.  A script builds a list of jobs (one job = one
scenario dict plus the cell labels it belongs to), runs them with
``run_jobs`` on at most two worker processes (other simulation runs share the
machine), and writes the rows with ``write_rows``.  Rows come back in job
order whatever the number of processes, so the CSV files regenerate byte for
byte.  Wall times are volatile and never go into a data file.

Seeds: EXP-029 + k uses ``seed_list(n, base=20261029 + k)``, the same seed list
for every cell and arm of that experiment, so arms are compared on the same
random draws.
"""
from __future__ import annotations

import csv
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from sim import s6_recovery
from sim.missions.api import Scenario
from sim.missions.engine import run_mission
from tools.c_common import default_processes

EXP_BASE_SEED = 20261029  # EXP-029; EXP-029 + k adds k
MAX_PROCESSES = 2
MISSION_LIMITS = (
    "Simulated only: no drone or radio hardware was used.",
    "Link loss patterns are synthetic (named presets in sim/missions/presets.py), never measured.",
    "Vehicle parameters are literature-derived (point mass, or the crazyflie-class physics profile).",
)
MISSION_SOURCES = (
    "sim/adapters.py",
    "sim/failsafe.py",
    "sim/authority.py",
    "sim/missions/api.py",
    "sim/missions/comms.py",
    "sim/missions/engine.py",
    "sim/missions/estimate_feed.py",
    "sim/missions/physics_adapter.py",
    "sim/missions/presets.py",
    "sim/missions/stack.py",
    "tools/e_common.py",
)


def base_seed(exp_number: int) -> int:
    if not 29 <= int(exp_number) <= 35:
        raise ValueError("Run E experiments are EXP-029 to EXP-035")
    return EXP_BASE_SEED + int(exp_number) - 29


def seeds_for(exp_number: int, n: int) -> list[int]:
    """The per-run seeds of one experiment (identical for every cell and arm)."""
    return s6_recovery.seed_list(int(n), base_seed(exp_number))


def processes(requested: int | None = None) -> int:
    """Worker count: the request, capped at MAX_PROCESSES; default min(2, cores // 3)."""
    if requested is None:
        return max(1, min(MAX_PROCESSES, default_processes()))
    if int(requested) < 1:
        raise ValueError("processes must be at least 1")
    return min(MAX_PROCESSES, int(requested))


def make_job(labels: dict, scenario: dict) -> dict:
    """One unit of work.  ``labels`` are the cell coordinates copied into the row."""
    Scenario.from_dict(scenario)  # fail early on a bad scenario, in the parent process
    return {"labels": dict(labels), "scenario": dict(scenario)}


def run_job(job: dict) -> dict:
    """Run one scenario and return a flat row: labels, seed, then the mission's metrics."""
    scenario = Scenario.from_dict(job["scenario"])
    result = run_mission(scenario, log=False)
    row: dict[str, Any] = dict(job["labels"])
    row["seed"] = int(scenario.seed)
    row["adapter"] = scenario.adapter
    for key in sorted(result.metrics):
        if key in row:
            raise ValueError(f"metric {key!r} collides with a cell label")
        row[key] = result.metrics[key]
    return row


def run_jobs(jobs: list[dict], n_processes: int = 1, worker: Callable[[dict], dict] = run_job,
             chunksize: int = 4) -> list[dict]:
    """Run every job; the returned rows are in job order for any process count."""
    n_processes = processes(n_processes)
    if n_processes == 1 or len(jobs) <= 1:
        return [worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=n_processes) as pool:
        return list(pool.map(worker, jobs, chunksize=max(1, int(chunksize))))


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("rows must not contain inf or nan; use None for a missing value")
        return repr(value)
    return str(value)


def write_rows(path: Path, rows: list[dict], first_columns: Iterable[str] = ()) -> list[str]:
    """Write rows as CSV (LF, UTF-8).  Columns: ``first_columns`` then the rest sorted.

    None is written as an empty cell, booleans as true/false, floats with repr
    (shortest exact form) so the file is byte-stable.
    """
    first = [c for c in first_columns]
    rest = sorted({key for row in rows for key in row} - set(first))
    columns = first + rest
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell_text(row.get(c)) for c in columns])
    return columns


def group_rows(rows: list[dict], keys: Iterable[str]) -> dict[tuple, list[dict]]:
    """Rows grouped by the given label keys, groups in first-seen order."""
    keys = tuple(keys)
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return groups


def rate_summary(flags: list[bool]) -> dict:
    """Count, rate and Wilson 95 % interval of a list of booleans."""
    n = len(flags)
    k = sum(1 for f in flags if f)
    low, high = s6_recovery.wilson_interval(k, n)
    return {"n": n, "count": k, "rate": k / n, "wilson_low": low, "wilson_high": high}


def median_summary(values: list[float | None]) -> dict:
    """Median and its 95 % order-statistic interval over the non-missing values."""
    present = [float(v) for v in values if v is not None]
    if not present:
        return {"n": 0, "median": None, "low": None, "high": None}
    median, low, high = s6_recovery.median_interval(present)
    return {"n": len(present), "median": median, "low": low, "high": high}
