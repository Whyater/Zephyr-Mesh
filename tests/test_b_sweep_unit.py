"""Unit tests for the checkpointed sweep runner and output writers."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from sim.estimation import sweep

FN = "sim.estimation.sweep:_selftest_job"
S6_MANIFEST_KEYS = {
    "argv", "git_revision", "git_tree_state", "n_cells", "n_processes", "n_runs", "n_seeds", "numpy",
    "outputs", "platform", "python", "replay_command", "runtime_s", "schema",
}


def toy(job: dict) -> dict:
    """Job function defined in this module, to check module:function lookup."""
    return {"double": 2 * job["x"]}


def calls(marker_dir: Path) -> int:
    return len(list(marker_dir.glob("*.called")))


def make_jobs(marker_dir: Path | None = None, n: int = 6) -> list[dict]:
    jobs = []
    for x in range(n):
        job = {"x": x, "sleep_s": 0.05 * (n - x)}  # first jobs finish last
        if marker_dir is not None:
            job["marker_dir"] = str(marker_dir)
        jobs.append(job)
    return jobs


def test_serial_and_parallel_results_are_identical_and_in_job_order():
    jobs = make_jobs()
    serial = sweep.run_jobs(jobs, FN, processes=1, progress=False)
    parallel = sweep.run_jobs(jobs, FN, processes=2, progress=False)
    assert serial == parallel
    assert [r["x"] for r in parallel] == list(range(6))
    assert parallel[3] == {"x": 3, "square": 9, "tenth": 0.1 * 3}


def test_function_defined_in_a_test_module_resolves():
    out = sweep.run_jobs([{"x": 1}, {"x": 2}], "tests.test_b_sweep_unit:toy", processes=2, progress=False)
    assert out == [{"double": 2}, {"double": 4}]


def test_checkpoint_resume_skips_finished_jobs(tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    ckpt = tmp_path / "ckpt"
    jobs = make_jobs(markers, n=4)
    first = sweep.run_jobs(jobs, FN, processes=2, checkpoint_dir=ckpt, fingerprint="v1", progress=False)
    assert calls(markers) == 4
    assert len(list(ckpt.glob("*.json"))) == 4
    again = sweep.run_jobs(jobs, FN, processes=2, checkpoint_dir=ckpt, fingerprint="v1", progress=False)
    assert calls(markers) == 4
    assert again == first
    # One job removed: only that one is recomputed.
    victim = ckpt / f"{sweep.sha256_json(jobs[2])[:16]}.json"
    victim.unlink()
    sweep.run_jobs(jobs, FN, processes=1, checkpoint_dir=ckpt, fingerprint="v1", progress=False)
    assert calls(markers) == 5
    # The record holds the fingerprint and both hashes.
    record = json.loads(victim.read_text())
    assert record["fingerprint"] == "v1"
    assert record["job_sha256"] == sweep.sha256_json(jobs[2])
    assert len(record["job_list_sha256"]) == 64


def test_changed_fingerprint_recomputes(tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    ckpt = tmp_path / "ckpt"
    jobs = make_jobs(markers, n=3)
    sweep.run_jobs(jobs, FN, processes=1, checkpoint_dir=ckpt, fingerprint="v1", progress=False)
    assert calls(markers) == 3
    sweep.run_jobs(jobs, FN, processes=1, checkpoint_dir=ckpt, fingerprint="v2", progress=False)
    assert calls(markers) == 6
    sweep.run_jobs(jobs, FN, processes=1, checkpoint_dir=ckpt, fingerprint="v2", progress=False)
    assert calls(markers) == 6


def test_corrupt_checkpoint_is_recomputed(tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    ckpt = tmp_path / "ckpt"
    jobs = make_jobs(markers, n=2)
    sweep.run_jobs(jobs, FN, processes=1, checkpoint_dir=ckpt, progress=False)
    (ckpt / f"{sweep.sha256_json(jobs[0])[:16]}.json").write_text("{not json")
    sweep.run_jobs(jobs, FN, processes=1, checkpoint_dir=ckpt, progress=False)
    assert calls(markers) == 3
    assert not list(ckpt.glob("*.tmp")) and not list(ckpt.glob(".*"))


def test_progress_callback_counts_every_job(tmp_path):
    seen: list[tuple[int, int]] = []
    jobs = make_jobs(n=3)
    sweep.run_jobs(jobs, FN, processes=2, checkpoint_dir=tmp_path, progress=lambda d, t, e: seen.append((d, t)))
    assert seen == [(1, 3), (2, 3), (3, 3)]
    seen.clear()  # fully cached rerun still reports
    sweep.run_jobs(jobs, FN, processes=1, checkpoint_dir=tmp_path, progress=lambda d, t, e: seen.append((d, t)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        sweep.run_jobs([{"x": 1}], FN, processes=0)
    with pytest.raises(ValueError):
        sweep.run_jobs([{"x": 1}], "no_colon_here")
    with pytest.raises(ModuleNotFoundError):
        sweep.run_jobs([{"x": 1}], "no_such_module_zz:f")


def test_job_error_propagates_from_pool():
    with pytest.raises(KeyError):
        sweep.run_jobs([{"x": 1}, {"y": 2}], FN, processes=2, progress=False)


def test_default_processes_is_a_third_of_cores_at_least_one():
    assert sweep.default_processes() >= 1


ROWS = [
    {"a": 1, "b": 0.1, "c": True, "d": None, "e": "x,y"},
    {"a": np.int64(2), "b": np.float64(1 / 3), "c": False, "d": 1e-17, "e": "z"},
]
COLS = ["a", "b", "c", "d", "e", "missing"]


def test_csv_formats_values_exactly(tmp_path):
    path = tmp_path / "t.csv"
    sweep.write_csv(path, ROWS, COLS)
    lines = path.read_text().split("\n")
    assert lines[0] == "a,b,c,d,e,missing"
    assert lines[1] == '1,0.1,true,,"x,y",'
    assert lines[2] == f"2,{1 / 3!r},false,1e-17,z,"
    assert float(lines[2].split(",")[1]) == 1 / 3  # round-trips


def test_write_csv_gz_is_byte_identical_and_matches_plain_csv(tmp_path):
    a, b = tmp_path / "a.csv.gz", tmp_path / "sub" / "b.csv.gz"
    sweep.write_csv_gz(a, ROWS, COLS)
    sweep.write_csv_gz(b, ROWS, COLS)
    assert a.read_bytes() == b.read_bytes()
    raw = a.read_bytes()
    assert raw[4:8] == b"\x00\x00\x00\x00"  # mtime 0
    assert b"a.csv" not in raw[:40]  # no file name in header
    plain = tmp_path / "plain.csv"
    sweep.write_csv(plain, ROWS, COLS)
    assert gzip.decompress(raw) == plain.read_bytes()


def test_write_json_format(tmp_path):
    path = tmp_path / "o.json"
    sweep.write_json(path, {"b": 1, "a": [1, 2]})
    assert path.read_text() == json.dumps({"a": [1, 2], "b": 1}, indent=2) + "\n"


def test_manifest_has_every_key_and_correct_hashes(tmp_path):
    sweep.write_json(tmp_path / "summary.json", {"k": 1})
    sweep.write_csv_gz(tmp_path / "rows.csv.gz", ROWS, COLS)
    manifest = sweep.write_manifest(
        tmp_path, outputs=["summary.json", "rows.csv.gz"], argv=["--output", "sim/runs/x y", "--seeds", "5"],
        n_processes=2, runtime_s=1.5, schema="zephyr-test-manifest-1",
        extra={"n_cells": 3, "n_runs": 15, "n_seeds": 5, "config_sha256": "ab" * 32, "seeds": [1, 2, 3, 4, 5],
               "limits": ["simulation only"]})
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest
    assert (tmp_path / "manifest.json").read_text() == json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    assert set(manifest) == S6_MANIFEST_KEYS | {"config_sha256", "seeds", "limits"}
    assert manifest["schema"] == "zephyr-test-manifest-1"
    assert manifest["argv"] == ["--output", "sim/runs/x y", "--seeds", "5"]
    assert manifest["replay_command"].endswith("--output 'sim/runs/x y' --seeds 5")
    assert manifest["n_processes"] == 2 and manifest["runtime_s"] == 1.5
    assert manifest["seeds"] == [1, 2, 3, 4, 5] and manifest["limits"] == ["simulation only"]
    assert manifest["numpy"] == np.__version__
    assert manifest["git_tree_state"] in {"clean", "dirty", "unknown"}
    assert set(manifest["outputs"]) == {"summary.json", "rows.csv.gz"}
    assert manifest["outputs"]["summary.json"] == sweep.sha256_file(tmp_path / "summary.json")
    assert len(manifest["outputs"]["rows.csv.gz"]) == 64


def test_manifest_defaults_and_reserved_keys(tmp_path):
    sweep.write_json(tmp_path / "s.json", {})
    prov = {"git_revision": "abc", "git_tree_state": "clean"}
    manifest = sweep.write_manifest(tmp_path, outputs=["s.json"], argv=[], n_processes=1, runtime_s=0.0,
                                    schema="x", provenance=prov)
    assert manifest["git_revision"] == "abc" and manifest["git_tree_state"] == "clean"
    assert set(manifest) == S6_MANIFEST_KEYS | {"config_sha256", "seeds", "limits"}
    with pytest.raises(ValueError):
        sweep.write_manifest(tmp_path, outputs=["s.json"], argv=[], n_processes=1, runtime_s=0.0, schema="x",
                             extra={"python": "2.7"})
    with pytest.raises(FileNotFoundError):
        sweep.write_manifest(tmp_path, outputs=["nope.json"], argv=[], n_processes=1, runtime_s=0.0, schema="x")


def test_compare_outputs_by_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "same.txt").write_bytes(b"1")
    (b / "same.txt").write_bytes(b"1")
    (a / "diff.txt").write_bytes(b"1")
    (b / "diff.txt").write_bytes(b"2")
    (a / "only_a.txt").write_bytes(b"1")
    assert sweep.compare_outputs(a, b, ["same.txt", "diff.txt", "only_a.txt", "neither.txt"]) == {
        "same.txt": True, "diff.txt": False, "only_a.txt": False, "neither.txt": False}


def unsorted_keys(job):
    """Returns keys deliberately out of alphabetical order (z before a)."""
    return {"z_last": job["x"], "a_first": -job["x"], "rows": [{"k": 1, "b": 2}]}


def test_fresh_and_resumed_results_are_identical_including_key_order(tmp_path):
    """A checkpoint is stored with sorted keys. If fresh results kept their original key
    order, a resumed run would produce CSV columns in a different order than a fresh run
    (seen with the atlas sweep). Both must now come out the same."""
    import json
    jobs = [{"x": 1}, {"x": 2}]
    fn = "tests.test_b_sweep_unit:unsorted_keys"
    fresh = sweep.run_jobs(jobs, fn, processes=1, checkpoint_dir=tmp_path, fingerprint="v", progress=False)
    resumed = sweep.run_jobs(jobs, fn, processes=1, checkpoint_dir=tmp_path, fingerprint="v", progress=False)
    assert json.dumps(fresh) == json.dumps(resumed)
    assert list(fresh[0]) == ["a_first", "rows", "z_last"]
