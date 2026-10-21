"""Tests for sim.ordered_pool (proposal): order, plain-data jobs, worker cap, serial path."""
from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from sim import ordered_pool
from sim.ordered_pool import check_plain, effective_workers, ordered_map

FN = "sim.ordered_pool:_selftest_job"


def _jobs(n: int, *, scramble: bool = False) -> list[dict]:
    # With scramble, early jobs sleep longest, so they finish last in a pool.
    return [{"seed": 7, "index": i, "sleep_s": (0.05 * (n - i) if scramble else 0)} for i in range(n)]


def _without_pid(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "pid"} for row in rows]


def test_results_come_back_in_submission_order_even_when_finish_order_differs() -> None:
    rows = ordered_map(FN, _jobs(6, scramble=True), workers=3)
    assert [row["index"] for row in rows] == list(range(6))
    assert len({row["pid"] for row in rows}) > 1  # really ran in more than one worker
    assert os.getpid() not in {row["pid"] for row in rows}


def test_one_worker_and_four_workers_give_identical_pickled_results() -> None:
    serial = ordered_map(FN, _jobs(8), workers=1)
    pooled = ordered_map(FN, _jobs(8), workers=4)
    assert {row["pid"] for row in serial} == {os.getpid()}
    # Same values, same types (numpy scalar kept, tuple kept), same key order.
    assert pickle.dumps(_without_pid(serial)) == pickle.dumps(_without_pid(pooled))
    assert type(serial[0]["first"]) is np.float64 and type(serial[0]["values"]) is tuple
    assert list(serial[0]) == ["index", "pid", "z", "values", "first"]


def test_results_match_a_plain_loop_reference() -> None:
    # Independent reference: draw the same streams directly, without the helper.
    expected = [float(np.random.default_rng(np.random.SeedSequence([7, i])).normal(size=5).sum()) for i in range(5)]
    assert [row["z"] for row in ordered_map(FN, _jobs(5), workers=2)] == expected


def test_serial_path_copies_jobs_like_a_process_boundary() -> None:
    jobs = _jobs(3)
    ordered_map(FN, jobs, workers=1)
    assert all("touched" not in job for job in jobs)


@pytest.mark.parametrize("bad", [np.random.default_rng(1), np.random.SeedSequence(1), np.int64(3),
                                 {1: "non-string key"}, [object()], {"rng": np.random.default_rng(1)}])
def test_jobs_must_be_plain_data(bad: object) -> None:
    with pytest.raises(TypeError):
        ordered_map(FN, [bad], workers=1)


def test_check_plain_accepts_nested_plain_data() -> None:
    check_plain({"seed": 3, "cond": {"p": 0.1, "label": "x", "d": None}, "ds": (1.0, 2.0), "ok": True, "l": [1, [2]]})


def test_worker_count_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ordered_pool.os, "cpu_count", lambda: 8)
    assert effective_workers(1, 100) == 1
    assert effective_workers(16, 100) == ordered_pool.MAX_WORKERS == 4
    assert effective_workers(4, 2) == 2
    assert effective_workers(4, 0) == 1
    monkeypatch.setattr(ordered_pool.os, "cpu_count", lambda: 2)
    assert effective_workers(4, 100) == 2
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            effective_workers(bad, 10)  # type: ignore[arg-type]


def test_bad_function_name_fails_before_any_work() -> None:
    with pytest.raises(ValueError):
        ordered_map("sim.ordered_pool._selftest_job", _jobs(2), workers=2)
    with pytest.raises(AttributeError):
        ordered_map("sim.ordered_pool:no_such_job", _jobs(2), workers=2)


def test_empty_job_list() -> None:
    assert ordered_map(FN, [], workers=4) == []
