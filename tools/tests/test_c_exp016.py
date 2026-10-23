"""Fast checks for the EXP-016 heartbeat and degraded-link runner.

The two numerical values in the first test are hand calculations from D9.
The remaining checks exercise the runner's simulation, threshold selection,
schema validation, and deterministic pilot output.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError

from sim.radio.channel import GilbertElliott
from sim.radio.heartbeat import (
    false_alarm_rate_ge_per_hour,
)
from tools import c_exp016_heartbeat as runner


def _expected_rate(kind: str, h: float, timeout: float, *, p: float = 0.1,
                   duration_s: float = 0.5) -> float:
    """Call the runner's public expected-rate helper with its stable API."""
    return float(runner.expected_rate(kind, h, timeout, p=p, duration_s=duration_s))


def test_d9_expected_rates_match_hand_numbers():
    # D9: 3600 * 10 * 0.9 * 0.1**3 = 32.4 alarms/hour.
    assert _expected_rate("independent", 10, 0.3, p=0.1) == pytest.approx(32.4, rel=0.005)
    # Classic Gilbert example: p ~= 0.0222, r = 0.2, m*=3, about 461/hour.
    ge = GilbertElliott(0.0222, 0.2)
    hand = false_alarm_rate_ge_per_hour(ge, 10, 0.3)
    assert hand == pytest.approx(461.0, abs=3.0)
    # The runner's condition API takes mean loss.  A mean loss of 0.1 with
    # a 0.5 s burst at 10 Hz gives r = 0.2 and p_gb = 0.0222...
    assert _expected_rate("bursty", 10, 0.3, p=0.1, duration_s=0.5) == pytest.approx(hand, rel=0.005)


def test_simulated_cell_agrees_with_closed_form():
    # This is deliberately a long single cell, independent losses, so the
    # count has enough samples for the preregistered 4-sigma comparison.
    result = runner.simulate_false_alarm_cell(
        h=10, timeout_s=0.3, condition="independent", p=0.1,
        duration_hours=50.0, seed=16016,
    )
    expected = _expected_rate("independent", 10, 0.3, p=0.1)
    simulated = float(result["simulated_rate_per_hour"] if isinstance(result, dict)
                      else result.simulated_rate_per_hour)
    count = float(result["simulated_count"] if isinstance(result, dict)
                  else result.simulated_count)
    expected_count = expected * 50.0
    assert abs(count - expected_count) <= 4.0 * np.sqrt(max(expected_count, 1.0))
    assert simulated == pytest.approx(count / 50.0)


def test_threshold_selection_uses_both_rates():
    rows = [
        {"timeout_s": 0.1, "simulated_rate_per_hour": 3.0, "expected_rate_per_hour": 2.0,
         "mean_delay_s": 0.08, "p95_delay_s": 0.1},
        {"timeout_s": 0.2, "simulated_rate_per_hour": 0.8, "expected_rate_per_hour": 0.7,
         "mean_delay_s": 0.18, "p95_delay_s": 0.2},
        {"timeout_s": 0.3, "simulated_rate_per_hour": 0.0, "expected_rate_per_hour": 0.02,
         "mean_delay_s": 0.28, "p95_delay_s": 0.3},
    ]
    one = runner.select_threshold(rows, threshold_per_hour=1.0)
    assert one["timeout_s"] == pytest.approx(0.2)
    # At 0.01/hour, the first row satisfying both is the third row.
    hundred = runner.select_threshold(rows, threshold_per_hour=0.01)
    assert hundred is None
    rows[-1]["expected_rate_per_hour"] = 0.005
    hundred = runner.select_threshold(rows, threshold_per_hour=0.01)
    assert hundred["timeout_s"] == pytest.approx(0.3)


def test_pilot_thresholds_validate_and_reject_extra_field(tmp_path: Path):
    out = tmp_path / "pilot"
    runner.main(["--output", str(out), "--pilot", "--processes", "1"])
    schema = json.loads(Path("sim/radio/thresholds_schema.json").read_text(encoding="utf-8"))
    thresholds = json.loads((out / "thresholds.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(thresholds)
    bad = json.loads(json.dumps(thresholds))
    bad["records"][0]["unexpected"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(bad)


def test_pilot_is_byte_identical(tmp_path: Path):
    first, second = tmp_path / "one", tmp_path / "two"
    argv = ["--output", str(first), "--pilot", "--processes", "1"]
    runner.main(argv)
    runner.main(["--output", str(second), "--pilot", "--processes", "1"])
    a = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    b = sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    assert a == b
    # Every data and figure file must match byte for byte.  The manifest is
    # compared without its volatile fields (runtime, git state, platform).
    from tools.c_common import VOLATILE_MANIFEST_FIELDS
    for rel in a:
        if rel.name == "manifest.json":
            ma, mb = (json.loads((d / rel).read_text(encoding="utf-8")) for d in (first, second))
            for key in VOLATILE_MANIFEST_FIELDS:
                ma.pop(key, None); mb.pop(key, None)
            assert ma == mb
            continue
        assert hashlib.sha256((first / rel).read_bytes()).digest() == hashlib.sha256((second / rel).read_bytes()).digest()


def test_fixed_detector_pilot_changes_only_the_detector_outputs(tmp_path: Path):
    # --detector lost-as-low-snr must leave every heartbeat output (they
    # never use the detector) byte-identical to the legacy run, and record the
    # detector choice in the summary and manifest.
    legacy, fixed = tmp_path / "legacy", tmp_path / "fixed"
    runner.main(["--output", str(legacy), "--pilot", "--processes", "1"])
    runner.main(["--output", str(fixed), "--pilot", "--processes", "1", "--detector", "lost-as-low-snr"])
    for name in ("false_alarms.csv", "detection_delay.csv", "exp016_false_alarms.png", "exp016_rule_300ms.png"):
        assert (legacy / name).read_bytes() == (fixed / name).read_bytes(), name
    # thresholds.json differs only in its replay command, which names the detector.
    t_legacy, t_fixed = (json.loads((d / "thresholds.json").read_text(encoding="utf-8")) for d in (legacy, fixed))
    assert t_fixed["generating_command"].endswith("--output {out} --detector lost-as-low-snr")
    assert "--detector" not in t_legacy["generating_command"]
    t_fixed["generating_command"] = t_legacy["generating_command"]
    assert t_fixed == t_legacy
    summary = json.loads((fixed / "summary.json").read_text(encoding="utf-8"))
    assert "lost-as-low-snr" in summary["interference_method"]
    assert "lost-as-low-snr" not in json.loads((legacy / "summary.json").read_text(encoding="utf-8"))["interference_method"]
    manifest = json.loads((fixed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["argv"][-2:] == ["--detector", "lost-as-low-snr"]
    assert manifest["config"]["detector"] == "lost-as-low-snr"
    assert manifest["replay_command"].endswith("--detector lost-as-low-snr")
    with pytest.raises(SystemExit):
        runner.main(["--output", str(tmp_path / "bad"), "--pilot", "--detector", "median"])
