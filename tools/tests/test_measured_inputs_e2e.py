"""End-to-end test for tools/rehearse_measured_inputs.py (simulated inputs only).

The inputs come from three generators written separately from the code that reads them:
tools/make_espnow_echo_fixture.py, tools/make_detection_fixture.py and tools/vision_latency.py synth.
So every number checked here is compared with a value the generator chose (or, for the latency
recording, with its answer key), never with another output of the reading code.

Nothing here is a radio or camera measurement. The chain runs once for the whole file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tools import rehearse_measured_inputs as rehearsal

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "sim" / "runs"


def _listing(folder: Path) -> list[str]:
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*")) if folder.exists() else []


@pytest.fixture(scope="module")
def rehearsed(tmp_path_factory):
    work = tmp_path_factory.mktemp("rehearsal")
    runs_before = _listing(RUNS_DIR)
    start = time.perf_counter()
    result = rehearsal.run_rehearsal(work)
    result["wall_s"] = time.perf_counter() - start
    result["runs_before"] = runs_before
    return result


def _by_name(result, prefix):
    rows = [c for c in result["checks"] if c["name"].startswith(prefix)]
    assert rows, f"no check starting with {prefix!r}"
    return rows


def test_every_check_of_the_rehearsal_passes(rehearsed):
    failed = [f"{c['name']}: {c['detail']}" for c in rehearsed["checks"] if not c["ok"]]
    assert not failed, "\n".join(failed)


@pytest.mark.parametrize("prefix,count", [("radio ", 7), ("vision ", 6), ("latency ", 4)])
# Counts: 6 radio values + the radio INTAKE.md status row; 5 vision values + the vision INTAKE.md status row.
def test_each_group_actually_ran_its_checks(rehearsed, prefix, count):
    rows = _by_name(rehearsed, prefix)
    assert len(rows) == count


def test_chain_runs_in_under_three_minutes(rehearsed):
    assert rehearsed["wall_s"] < 180
    assert set(rehearsed["seconds"]) == {"radio intake", "vision intake", "combined rerun"}


def test_artifacts_are_synthetic_in_the_files_themselves(rehearsed):
    """Read the written files directly instead of trusting the rehearsal's own status checks."""
    work = rehearsed["work"]
    assert json.loads((work / "radio-intake" / "fit.json").read_text(encoding="utf-8"))["source_trace"]["status"] == "synthetic"
    assert json.loads((work / "vision-intake" / "fits" / "cal-a.fit.json").read_text(encoding="utf-8"))["input"]["status"] == "synthetic"
    link = json.loads((work / "vision-intake" / "links" / "lat-a.link.json").read_text(encoding="utf-8"))
    assert link["status"] == "synthetic"
    summary = json.loads((work / "combined" / "summary.json").read_text(encoding="utf-8"))
    assert summary["input_status"] == "synthetic" and summary["complete"] is True
    assert "synthetic" in summary["status"]
    assert "measured" not in summary["input_status"]


def test_recovered_delay_is_not_the_nominal_knob(rehearsed):
    """The latency recording's jitter is half-normal, so its true mean sits above the typed 120 ms."""
    key = _by_name(rehearsed, "latency answer key mean")[0]
    assert key["ok"]
    link = json.loads((rehearsed["work"] / "vision-intake" / "links" / "lat-a.link.json").read_text(encoding="utf-8"))
    upper = link["components"]["camera"]["latency_s"]["upper"]
    assert abs(upper - rehearsal.LATENCY_NOMINAL_MS / 1e3) > 0.004


def test_the_interval_check_can_fail():
    """A truth outside the stated interval must be reported as a failure (the check is not a rubber stamp)."""
    assert rehearsal._inside("x", 1.0, 0.0, 2.0)["ok"]
    assert not rehearsal._inside("x", 3.0, 0.0, 2.0)["ok"]
    assert not rehearsal._inside("x", -0.1, 0.0, 2.0)["ok"]


def test_the_status_check_can_fail(tmp_path):
    bad = tmp_path / "a.json"
    bad.write_text(json.dumps({"status": "measured"}), encoding="utf-8")
    missing = tmp_path / "b.json"
    missing.write_text(json.dumps({"x": 1}), encoding="utf-8")
    good = tmp_path / "c.json"
    good.write_text(json.dumps({"source_trace": {"status": "synthetic"}}), encoding="utf-8")
    flagged = rehearsal._all_synthetic([bad, missing, good])
    assert len(flagged) == 2 and not any("c.json" in item for item in flagged)


def test_nothing_is_written_under_sim_runs_and_the_registry_is_unchanged(rehearsed):
    assert _listing(RUNS_DIR) == rehearsed["runs_before"]
    row = _by_name(rehearsed, "registry file unchanged")[0]
    assert row["ok"]


def test_report_names_every_check_and_has_no_local_paths(rehearsed):
    text = "\n".join(rehearsal.report_lines(rehearsed))
    for check in rehearsed["checks"]:
        assert check["name"] in text
    assert "/" + "Users" not in text and "C:" + "\\" not in text
    assert "\u2014" not in text
    assert "not a measurement" in text


def test_workdir_must_be_empty(tmp_path, capsys):
    (tmp_path / "leftover.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        rehearsal.main(["--workdir", str(tmp_path)])
