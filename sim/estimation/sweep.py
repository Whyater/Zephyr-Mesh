"""Generic checkpointed sweep runner and output writers.

A sweep is a list of independent jobs (plain dicts) and one function that turns
a job into a JSON-serializable result. This module runs the jobs, optionally
across several processes, and can save each finished job to disk so an
interrupted run resumes instead of starting over. Results always come back in
job order, so the output does not depend on which process finished first.

The writers (JSON, CSV, gzipped CSV, manifest) are deterministic: the same rows
give the same bytes, so two runs can be compared with a hash. Results are from
the simulation only, never from flight hardware.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import io
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY_PREFIX = "PYTHONPATH=. ./.venv/bin/python "

# Keys write_manifest always fills in itself; `extra` may not overwrite them.
_RESERVED_MANIFEST_KEYS = frozenset({
    "schema", "argv", "python", "numpy", "platform", "git_revision", "git_tree_state",
    "runtime_s", "n_processes", "outputs",
})
# Keys that describe the sweep itself and come from `extra` (with neutral defaults).
_EXTRA_MANIFEST_DEFAULTS: dict[str, Any] = {
    "n_cells": None, "n_runs": None, "n_seeds": None, "config_sha256": None, "seeds": [], "limits": [],
}

ProgressCallback = Callable[[int, int, float], None]


# ---------------------------------------------------------------------------
# Hashing and provenance
# ---------------------------------------------------------------------------

def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Compact, key-sorted JSON used for hashing (not for files people read)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _git_value(args: list[str]) -> str | None:
    """Return git's output, "" for empty output, or None when git cannot answer."""
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL, cwd=REPO_ROOT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_provenance() -> dict[str, str | None]:
    """Git revision and tree state now.

    Call this before a run writes anything, so the new output files cannot make
    the tree look dirty, then pass the result to write_manifest.
    """
    porcelain = _git_value(["status", "--porcelain"])
    return {
        "git_revision": _git_value(["rev-parse", "HEAD"]) or None,
        "git_tree_state": "unknown" if porcelain is None else ("dirty" if porcelain else "clean"),
    }


# ---------------------------------------------------------------------------
# Running jobs
# ---------------------------------------------------------------------------

def default_processes() -> int:
    """A third of the cores (at least one), leaving the machine usable."""
    return max(1, (os.cpu_count() or 1) // 3)


def _resolve(fn: str) -> Callable[[dict[str, Any]], Any]:
    module_name, separator, attribute = fn.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"fn must look like 'module:function', got {fn!r}")
    target: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise ValueError(f"{fn!r} is not callable")
    return target


def _call(fn: str, job: dict[str, Any]) -> Any:
    # Resolved inside the worker so only the name crosses the process boundary.
    return _resolve(fn)(job)


def _normalize(result: Any) -> Any:
    """Round-trip through JSON with sorted keys so fresh and checkpointed results
    are identical, key order included (checkpoints are stored with sorted keys,
    and callers may build CSV columns from key order)."""
    return json.loads(json.dumps(result, sort_keys=True))


def _library_stamp() -> str:
    """numpy and scipy versions: a library upgrade can change results, so it must
    invalidate checkpoints just like a code change does."""
    import scipy

    return f"numpy {np.__version__}; scipy {scipy.__version__}"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_checkpoint(path: Path, job_sha: str, fingerprint: str) -> tuple[bool, Any]:
    if not path.is_file():
        return False, None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False, None
    if (not isinstance(data, dict) or data.get("fingerprint") != fingerprint
            or data.get("job_sha256") != job_sha or data.get("libraries") != _library_stamp()):
        return False, None
    if "result" not in data:
        return False, None
    return True, data["result"]


def _stderr_progress(done: int, total: int, elapsed_s: float) -> None:
    print(f"[sweep] {done}/{total} jobs done, {elapsed_s:.1f} s elapsed", file=sys.stderr, flush=True)


def run_jobs(jobs: Sequence[dict[str, Any]], fn: str, *, processes: int | None = None,
             checkpoint_dir: str | os.PathLike[str] | None = None, fingerprint: str = "",
             progress: ProgressCallback | None | bool = None) -> list[Any]:
    """Run fn(job) for every job and return the results in job order.

    fn is "module:function"; it must take one job dict and return a
    JSON-serializable value. processes=1 runs in this process, more uses a
    process pool. With checkpoint_dir, each finished job is saved to
    <checkpoint_dir>/<sha256(job json)[:16]>.json together with the fingerprint,
    the job's hash and the hash of the whole job list. A rerun loads a job
    instead of recomputing it when its file exists with the same fingerprint.
    Change the fingerprint whenever the code or config behind fn changes.

    progress(done, total, elapsed_s) is called after every finished job
    (cached jobs count as finished). None prints a line to stderr, False is silent.
    """
    jobs = list(jobs)
    libraries = _library_stamp()
    if processes is None:
        processes = default_processes()
    if processes < 1:
        raise ValueError("processes must be at least 1")
    _resolve(fn)  # fail fast in the parent on a bad name
    if progress is None:
        report: ProgressCallback | None = _stderr_progress
    elif progress is False:
        report = None
    else:
        report = progress  # type: ignore[assignment]

    # Fail on unserializable jobs now, not after hours of compute.
    job_shas = [sha256_json(job) for job in jobs]
    job_list_sha = sha256_json(job_shas)
    directory: Path | None = None
    if checkpoint_dir is not None:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(index: int) -> Path:
        assert directory is not None
        return directory / f"{job_shas[index][:16]}.json"

    total = len(jobs)
    results: list[Any] = [None] * total
    pending: list[int] = []
    done = 0
    start = time.perf_counter()
    for index in range(total):
        if directory is not None:
            hit, cached = _load_checkpoint(checkpoint_path(index), job_shas[index], fingerprint)
            if hit:
                results[index] = cached
                done += 1
                if report is not None:
                    report(done, total, time.perf_counter() - start)
                continue
        pending.append(index)

    def finish(index: int, raw: Any) -> None:
        nonlocal done
        results[index] = _normalize(raw)
        if directory is not None:
            record = {"fingerprint": fingerprint, "libraries": libraries, "job_sha256": job_shas[index],
                      "job_list_sha256": job_list_sha, "result": results[index]}
            _write_atomic(checkpoint_path(index), json.dumps(record, sort_keys=True) + "\n")
        done += 1
        if report is not None:
            report(done, total, time.perf_counter() - start)

    if processes == 1 or len(pending) <= 1:
        for index in pending:
            finish(index, _call(fn, jobs[index]))
    else:
        pool = ProcessPoolExecutor(max_workers=min(processes, len(pending)))
        try:
            futures = {pool.submit(_call, fn, jobs[index]): index for index in pending}
            waiting = set(futures)
            while waiting:
                finished, waiting = wait(waiting, return_when=FIRST_COMPLETED)
                for future in finished:
                    finish(futures[future], future.result())
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
    return results


def _selftest_job(job: dict[str, Any]) -> dict[str, Any]:
    """Tiny job used by the unit tests (kept here so workers can import it).

    Optional keys: sleep_s (pause first, to scramble completion order) and
    marker_dir (leave one empty file per call, so tests can count calls).
    """
    if job.get("sleep_s"):
        time.sleep(float(job["sleep_s"]))
    marker_dir = job.get("marker_dir")
    if marker_dir:
        Path(marker_dir, f"{os.getpid()}-{uuid.uuid4().hex}.called").touch()
    x = job["x"]
    return {"x": x, "square": x * x, "tenth": 0.1 * x}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_json(path: str | os.PathLike[str], obj: Any) -> None:
    """Key-sorted, 2-space-indented JSON with a trailing newline."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _csv_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def _write_rows(handle: Any, rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(list(columns))
    for row in rows:
        writer.writerow([_csv_value(row.get(column)) for column in columns])


def write_csv(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    """CSV with the given columns; floats are written with repr so they round-trip."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        _write_rows(handle, rows, columns)


def write_csv_gz(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    """Gzipped CSV with no timestamp or file name in the header.

    The same rows always give the same bytes, which is what makes the file
    hashable and comparable across runs.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
            _write_rows(text, rows, columns)
            text.flush()
            text.detach()


def write_manifest(out_dir: str | os.PathLike[str], *, outputs: Sequence[str], argv: Sequence[str],
                   n_processes: int, runtime_s: float, extra: dict[str, Any] | None = None, schema: str,
                   provenance: dict[str, str | None] | None = None) -> dict[str, Any]:
    """Write out_dir/manifest.json and return it.

    outputs are file names inside out_dir; each is recorded with its sha256.
    The keys match sim/runs/s6-recovery/manifest.json plus config_sha256, seeds and
    limits. extra supplies the sweep-specific values (n_cells, n_runs, n_seeds,
    config_sha256, seeds, limits, and optionally replay_command or other keys);
    it cannot overwrite the keys this function fills in. Pass provenance from
    capture_provenance() taken before the run; otherwise git is asked now,
    when the new outputs may already make the tree look dirty.
    """
    directory = Path(out_dir)
    extra = dict(extra or {})
    clash = _RESERVED_MANIFEST_KEYS.intersection(extra)
    if clash:
        raise ValueError(f"extra may not set reserved manifest keys: {sorted(clash)}")
    state = provenance if provenance is not None else capture_provenance()
    argv_list = [str(item) for item in argv]
    manifest: dict[str, Any] = {
        "schema": schema,
        "replay_command": DEFAULT_REPLAY_PREFIX + " ".join(shlex.quote(item) for item in argv_list),
        "argv": argv_list,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_revision": state.get("git_revision"),
        "git_tree_state": state.get("git_tree_state", "unknown"),
        "runtime_s": runtime_s,
        "n_processes": n_processes,
        "outputs": {name: sha256_file(directory / name) for name in outputs},
    }
    for key, default in _EXTRA_MANIFEST_DEFAULTS.items():
        manifest[key] = default
    manifest.update(extra)
    write_json(directory / "manifest.json", manifest)
    return manifest


def compare_outputs(dir_a: str | os.PathLike[str], dir_b: str | os.PathLike[str],
                    names: Iterable[str]) -> dict[str, bool]:
    """For each file name, True when both directories hold it with identical bytes."""
    result: dict[str, bool] = {}
    for name in names:
        a, b = Path(dir_a) / name, Path(dir_b) / name
        result[name] = a.is_file() and b.is_file() and sha256_file(a) == sha256_file(b)
    return result
