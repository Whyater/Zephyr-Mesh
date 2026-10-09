from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sim.s7_metrics import _reacquisition, summarize_s7_run
from tools.analyze_s7_run import main as analyze_s7_main

ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs/s7-swarm/run.json"


def test_s7_summary_reports_observed_contract_without_performance_claims():
    document = json.loads(RUN.read_text(encoding="utf-8"))
    summary = summarize_s7_run(document)
    assert summary["schema"] == "zephyr-s7-swarm-summary-1"
    assert summary["run"]["frame_count"] == 6
    assert summary["run"]["agent_count"] == 50
    assert summary["link_events"]["attempted"] == 15000
    assert summary["link_events"]["received"] == 14583
    assert summary["link_events"]["lost"] == 417
    assert summary["estimation"]["fused_target_present"] == 198
    assert "not" in summary["evidence_boundary"]


def test_s7_summary_is_deterministic_and_schema_fixture_is_valid():
    document = json.loads(RUN.read_text(encoding="utf-8"))
    assert summarize_s7_run(document) == summarize_s7_run(document)
    schema = json.loads((ROOT / "sim/s7_run_schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)


def test_s7_schema_rejects_extra_top_level_field():
    document = json.loads(RUN.read_text(encoding="utf-8"))
    document["unexpected"] = True
    schema = json.loads((ROOT / "sim/s7_run_schema.json").read_text(encoding="utf-8"))
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(document)


def test_s7_recovery_distinguishes_bracketed_and_censored_gaps():
    bracketed = _reacquisition([True, False, False, True], 0.05)
    assert bracketed["bracketed_intervals"]["count"] == 1
    assert bracketed["bracketed_intervals"]["mean"] == pytest.approx(0.10)
    assert bracketed["censored_outages"] == 0
    assert _reacquisition([False, True], 0.05)["censored_outages"] == 1
    assert _reacquisition([True, False], 0.05)["censored_outages"] == 1
    assert _reacquisition([True, None, True], 0.05)["censored_outages"] == 1


def test_s7_schema_rejects_contradictory_link_outcome():
    document = json.loads(RUN.read_text(encoding="utf-8"))
    document["link_events"][0]["loss_reason"] = "contradiction"
    schema = json.loads((ROOT / "sim/s7_run_schema.json").read_text(encoding="utf-8"))
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(document)


def test_s7_cli_rejects_duplicate_nonfinite_and_oversize_input(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"zephyr-s7-swarm-run-1","schema":"zephyr-s7-swarm-run-1"}', encoding="utf-8")
    with pytest.raises(SystemExit):
        analyze_s7_main([str(duplicate)])
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema":"zephyr-s7-swarm-run-1","x":NaN}', encoding="utf-8")
    with pytest.raises(SystemExit):
        analyze_s7_main([str(nonfinite)])
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{" + b" " * (16 * 1024 * 1024))
    with pytest.raises(SystemExit):
        analyze_s7_main([str(oversized)])
