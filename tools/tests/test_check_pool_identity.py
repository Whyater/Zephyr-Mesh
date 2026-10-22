"""Tests for pure helpers in check_pool_identity."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools import check_pool_identity


def test_registry_output_lookup_from_json_file(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            [
                {
                    "run_dir": "sim/runs/c5-heartbeat",
                    "outputs": {"one.csv": "a"},
                }
            ]
        ),
        encoding="utf-8",
    )
    assert check_pool_identity.registry_outputs(path, "exp016") == {"one.csv": "a"}


def test_real_registry_lists_both_runners() -> None:
    registry = json.loads((check_pool_identity.REGISTRY_PATH).read_text(encoding="utf-8"))
    assert len(check_pool_identity.registry_outputs(registry, "exp016")) == 7
    assert len(check_pool_identity.registry_outputs(registry, "exp018")) == 5
    with pytest.raises(check_pool_identity.CheckerError):
        check_pool_identity.registry_outputs(registry, "exp099")


def test_identity_verdict_all_same() -> None:
    hashes = {
        "workers1": {"a": "aaa", "b": "bbb"},
        "workers4": {"a": "aaa", "b": "bbb"},
    }
    identical, detail = check_pool_identity.identity_verdict(hashes)
    assert identical is True
    assert detail == {}


def test_identity_verdict_names_differing_file() -> None:
    hashes = {
        "workers1": {"a": "aaa", "b": "bbb"},
        "workers4": {"a": "ccc", "b": "bbb"},
    }
    identical, detail = check_pool_identity.identity_verdict(hashes)
    assert identical is False
    assert "a" in detail


def test_registry_match() -> None:
    matched, detail = check_pool_identity.registry_match(
        {"a": "aaa", "b": "bbb"}, {"a": "aaa", "b": "bbb"}
    )
    assert matched is True
    assert detail == {}


def test_registry_match_mismatch() -> None:
    matched, detail = check_pool_identity.registry_match({"a": "bad"}, {"a": "aaa"})
    assert matched is False
    assert detail["a"] == "hash differs"


def test_registry_match_missing_file() -> None:
    matched, detail = check_pool_identity.registry_match({}, {"a": "aaa"})
    assert matched is False
    assert detail["a"] == "missing file"


@pytest.mark.skipif(os.environ.get("ZEPHYR_POOL_PILOT") != "1", reason="pilot is opt-in")
def test_exp016_pilot(tmp_path: Path) -> None:
    exit_code = check_pool_identity.main(
        ["--runner", "exp016", "--pilot", "--keep", str(tmp_path)]
    )
    assert exit_code == 0


def test_manifest_processes_reads_the_runner_manifest(tmp_path: Path) -> None:
    assert check_pool_identity._manifest_processes(tmp_path) is None
    (tmp_path / "manifest.json").write_text(json.dumps({"n_processes": 3}), encoding="utf-8")
    assert check_pool_identity._manifest_processes(tmp_path) == 3
    (tmp_path / "manifest.json").write_text("not json", encoding="utf-8")
    assert check_pool_identity._manifest_processes(tmp_path) is None
