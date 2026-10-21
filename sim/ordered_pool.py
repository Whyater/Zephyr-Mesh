"""Run independent jobs on a few worker processes and get the results back in
submission order (proposal, opt-in).

The rules that keep a parallel run byte-identical to a serial one:

1. Every job is built before anything runs, and it is plain data (numbers,
   strings, None, lists, tuples, dicts with string keys). A seed must be a
   number inside the job, never a Generator or SeedSequence object, so the job
   function has to build its own random streams from the job's own keys.
2. The job function is named as "module:function" and lives at module level, so
   a fresh worker can import it. It writes no files and changes no globals.
3. Results come back in the order the jobs were given, never in finish order.
4. One worker is not a different code path: the parent calls the same
   ``_run_one`` that the workers call, on the same pickled bytes, so a job sees
   a fresh copy of its input and the parent gets a fresh copy of the result.
   That copies the data only: in the parent a job can still see the parent's
   globals (imported modules, environment, caches), while a pooled worker starts
   fresh. Rule 2 is what makes that difference harmless.

Workers are started with "spawn" on every OS, so Linux behaves like macOS and
Windows (no copy of the parent's memory, including any global random state).
The worker count is capped at MAX_WORKERS, the core count and the job count;
``effective_workers`` gives the number that will really be used, for manifests.
"""
from __future__ import annotations

import importlib
import multiprocessing
import os
import pickle
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

MAX_WORKERS = 4

_PLAIN_SCALARS = (type(None), bool, int, float, str)


def effective_workers(requested: int, n_jobs: int) -> int:
    """The worker count ordered_map will use: min(requested, MAX_WORKERS, cores, jobs), at least 1."""
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError(f"workers must be a whole number of at least 1, got {requested!r}")
    return max(1, min(requested, MAX_WORKERS, os.cpu_count() or 1, n_jobs))


def check_plain(value: Any, where: str = "job") -> None:
    """Raise TypeError unless value is plain data (see rule 1 in the module docstring)."""
    if type(value) in _PLAIN_SCALARS:
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            check_plain(item, f"{where}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{where} has a non-string key {key!r}")
            check_plain(item, f"{where}[{key!r}]")
        return
    raise TypeError(f"{where} is {type(value).__name__}, not plain data; convert it (for example int(seed))")


def _resolve(fn: str) -> Callable[[Any], Any]:
    module_name, separator, attribute = fn.partition(":")
    if not separator or not module_name or not attribute or "." in attribute or "<" in attribute:
        raise ValueError(f"fn must look like 'module:function', got {fn!r}")
    target = getattr(importlib.import_module(module_name), attribute)
    if not callable(target):
        raise TypeError(f"{fn!r} is not callable")
    return target


def _run_one(payload: bytes) -> bytes:
    """Run one job. The parent (one worker) and the pool workers both call this."""
    fn, job = pickle.loads(payload)
    return pickle.dumps(_resolve(fn)(job), protocol=pickle.HIGHEST_PROTOCOL)


def ordered_map(fn: str, jobs: Iterable[Any], *, workers: int = 1) -> list[Any]:
    """Return [fn(job) for job in jobs], in job order, using up to `workers` processes."""
    jobs = list(jobs)
    for index, job in enumerate(jobs):
        check_plain(job, f"jobs[{index}]")
    _resolve(fn)  # fail in the parent on a bad name, before any work starts
    payloads = [pickle.dumps((fn, job), protocol=pickle.HIGHEST_PROTOCOL) for job in jobs]
    count = effective_workers(workers, len(jobs))
    if count == 1:
        raw = [_run_one(payload) for payload in payloads]
    else:
        pool = ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"))
        try:
            raw = list(pool.map(_run_one, payloads))  # map keeps submission order
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
    return [pickle.loads(item) for item in raw]


def _selftest_job(job: dict[str, Any]) -> dict[str, Any]:
    """Tiny job for the unit tests (module level, so spawned workers can import it).

    sleep_s pauses first, so later jobs can finish before earlier ones;
    fail=True raises, so tests can check that an error reaches the caller.
    """
    import time

    import numpy as np

    if job.get("fail"):
        raise RuntimeError(f"selftest job {job['index']} failed on purpose")
    if job.get("sleep_s"):
        time.sleep(float(job["sleep_s"]))
    rng = np.random.default_rng(np.random.SeedSequence([int(job["seed"]), int(job["index"])]))
    draws = rng.normal(size=int(job.get("n", 5)))
    job["touched"] = True  # must not leak back to the caller's job
    return {"index": job["index"], "pid": os.getpid(), "z": float(draws.sum()),
            "values": tuple(float(x) for x in draws), "first": draws[0]}
