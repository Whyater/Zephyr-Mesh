from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from desktop.evidence import EvidenceError, build_evidence_report, load_evidence_report, write_evidence_report


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("kind", "source"),
    [
        ("espnow", "sim/trace_example.json"),
        ("vision", "sim/vision_trace_example.json"),
        ("hardware", "sim/hardware_profile_example.json"),
        ("bench", "sim/bench_trace_example.json"),
        ("investigation", "sim/investigation_example.json"),
    ],
)
def test_report_round_trip_preserves_fixture_boundary(tmp_path: Path, kind: str, source: str):
    report = build_evidence_report(ROOT / source, kind)
    destination = tmp_path / f"{kind}.json"
    write_evidence_report(destination, report)
    loaded = load_evidence_report(destination)
    assert loaded == report
    assert loaded["status"] == ("synthetic" if kind == "investigation" else "fixture")
    assert loaded["source"]["sha256"] == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    assert loaded["limitations"]


def test_report_rejects_unknown_fields_and_wrong_schema(tmp_path: Path):
    report = build_evidence_report(ROOT / "sim/trace_example.json", "espnow")
    report["extra"] = True
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvidenceError, match="unknown field"):
        load_evidence_report(path)

    report.pop("extra")
    report["source"]["schema"] = "zephyr-vision-trace-1"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvidenceError, match="source.schema"):
        load_evidence_report(path)


def test_report_rejects_bad_digest_nonfinite_and_oversize(tmp_path: Path):
    report = build_evidence_report(ROOT / "sim/trace_example.json", "espnow")
    report["source"]["sha256"] = "A" * 64
    path = tmp_path / "bad-digest.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvidenceError, match="lowercase"):
        load_evidence_report(path)

    report["source"]["sha256"] = "0" * 64
    report["summary"]["bad"] = float("nan")
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvidenceError, match="invalid JSON|non-finite"):
        load_evidence_report(path)

    path.write_bytes(b"{" + b"x" * (8 * 1024 * 1024))
    with pytest.raises(EvidenceError, match="exceeds"):
        load_evidence_report(path)


def test_report_rejects_duplicate_keys_status_contradiction_and_deep_summary(tmp_path: Path):
    report = build_evidence_report(ROOT / "sim/trace_example.json", "espnow")
    report["summary"]["provenance"]["status"] = "measured"
    path = tmp_path / "contradiction.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvidenceError, match="status"):
        load_evidence_report(path)

    path.write_text('{"schema":"zephyr-evidence-report-1","schema":"zephyr-evidence-report-1"}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="duplicate JSON key"):
        load_evidence_report(path)

    report = build_evidence_report(ROOT / "sim/trace_example.json", "espnow")
    nested: object = None
    for _ in range(66):
        nested = {"next": nested}
    report["summary"]["deep"] = nested
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvidenceError, match="nesting"):
        load_evidence_report(path)


def test_builder_rejects_raw_schema_violation(tmp_path: Path):
    raw = json.loads((ROOT / "sim/trace_example.json").read_text(encoding="utf-8"))
    raw["packets"][0]["unexpected"] = True
    source = tmp_path / "bad-trace.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(EvidenceError, match="additional properties|unexpected"):
        build_evidence_report(source, "espnow")


def test_investigation_report_retains_synthetic_status_and_grid():
    report = build_evidence_report(ROOT / "sim/investigation_example.json", "investigation")
    assert report["status"] == "synthetic"
    assert report["source"]["schema"] == "zephyr-s6-sweep-1"
    assert len(report["summary"]["rows"]) == 72
    assert report["summary"]["provenance"]["source_status"].startswith("synthetic investigation")


def test_report_schema_accepts_emitted_investigation_envelope():
    report = build_evidence_report(ROOT / "sim/investigation_example.json", "investigation")
    schema = json.loads((ROOT / "desktop/evidence_report_schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(report)
