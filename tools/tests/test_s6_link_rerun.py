import csv
import hashlib
import json

import pytest

from sim.s6_recovery import RecoveryConfig, seed_list, simulate_run
from tools.fit_espnow_log import fit_echo_trace
from tools.import_espnow_log import read_echo_logs
from tools.make_espnow_echo_fixture import EchoParams, write_fixture
from tools.s6_link_rerun import LinkRerunError, describe_input, parse_link, run_rerun, status_label


def test_parse_s6_deliver_block_and_small_json():
    fit = {"schema": "zephyr-espnow-fit-1", "s6_deliver": {"delay_s": 0.01, "jitter_s": 0.002,
        "loss_probability": 0.05, "loss_model": "independent"}}
    assert parse_link(fit) == {"delay_s": 0.01, "jitter_s": 0.002, "loss_probability": 0.05, "loss_model": "independent"}
    assert parse_link({"delay_s": 0.01, "jitter_s": 0, "loss": 0.05})["loss_probability"] == 0.05


@pytest.mark.parametrize("value", [{}, {"delay_s": -1, "loss_probability": 0},
                                    {"delay_s": 0, "loss_probability": 2},
                                    {"delay_s": 0, "loss_probability": 0, "loss_model": "x"}])
def test_reject_malformed_input(value):
    with pytest.raises(LinkRerunError):
        parse_link(value)


def test_echo_fixture_fit_and_pilot_output(tmp_path):
    fixture = tmp_path / "fixture"
    write_fixture(fixture, EchoParams(n_packets=80, burst_start=0.0, burst_end=1.0, return_loss=0.0), 4)
    trace = read_echo_logs(fixture / "sender.csv", fixture / "echoer.csv", trace_id="fixture", status="fixture")
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps(fit_echo_trace(trace, n_boot=10, seed=2)))
    out = tmp_path / "result"
    manifest = run_rerun(fit_path, out, pilot=True, seeds_count=2)
    assert manifest["input_sha256"] == hashlib.sha256(fit_path.read_bytes()).hexdigest()
    assert (out / "manifest.json").exists()
    assert len(json.loads((out / "summary.json").read_text())["cells"]) == 4


def test_burst_fit_runs_and_preserves_burst_length(tmp_path):
    source = tmp_path / "burst.json"
    source.write_text(json.dumps({"delay_s": 0.01, "loss_probability": 0.2,
                                  "loss_model": "burst", "burst_mean_length_packets": 4}))
    manifest = run_rerun(source, tmp_path / "burst-out", pilot=True, seeds_count=2)
    assert manifest["link"]["burst_mean_length_packets"] == 4.0


def test_zero_link_matches_simulation_per_seed(tmp_path):
    source = tmp_path / "zero.json"
    source.write_text(json.dumps({"delay_s": 0, "jitter_s": 0, "loss_probability": 0}))
    out = tmp_path / "zero-out"
    run_rerun(source, out, pilot=True, seeds_count=2)
    with (out / "per_seed.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    config = RecoveryConfig()
    seeds = seed_list(2)
    for motion in ("constant_velocity", "constant_turn"):
        expected = [simulate_run(motion, delay_s=0, loss_probability=0, noise_std_m=0,
                                 seed=seed, config=config)["recovery"].p95_error_m for seed in seeds]
        actual = [float(row["p95_error_m"]) for row in rows
                  if row["motion"] == motion and row["noise_std_m"] == "0.0" and row["arm"] == "recovery"]
        assert actual == expected


@pytest.mark.parametrize("link", [
    {"delay_s": 0.01, "loss_probability": 0},
    {"delay_s": 0, "loss_probability": 0.2},
    {"delay_s": 0.01, "jitter_s": 0.01, "loss_probability": 0},
])
def test_each_nonzero_link_setting_changes_results(tmp_path, link):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"delay_s": 0, "jitter_s": 0, "loss_probability": 0}))
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(link))
    run_rerun(baseline, tmp_path / "base-out", pilot=True, seeds_count=2)
    run_rerun(changed, tmp_path / "changed-out", pilot=True, seeds_count=2)
    base = (tmp_path / "base-out" / "per_seed.csv").read_bytes()
    altered = (tmp_path / "changed-out" / "per_seed.csv").read_bytes()
    assert altered != base


def test_same_input_is_byte_identical(tmp_path):
    source = tmp_path / "link.json"
    source.write_text(json.dumps({"delay_s": 0.01, "jitter_s": 0.001, "loss_probability": 0.05}))
    one, two = tmp_path / "one", tmp_path / "two"
    run_rerun(source, one, pilot=True, seeds_count=2)
    run_rerun(source, two, pilot=True, seeds_count=2)
    for name in ("summary.json", "per_seed.csv", "manifest.json"):
        assert (one / name).read_bytes() == (two / name).read_bytes()


MEASURED_PROVENANCE = {
    "instrument": {"kind": "esp32-echo-pair", "model": "test-board", "settings": {"channel": 1}},
    "calibration_ref": None, "calibration_ref_null_reason": "test trace",
    "raw_sha256": "a" * 64, "code_revision": "test", "captured_at": "2026-10-27T12:00:00-06:00",
    "operator": "test operator", "ground_truth": None, "ground_truth_null_reason": "test trace",
}


def _fit_file(tmp_path, status, trace_id, **trace_kwargs):
    fixture = tmp_path / f"fixture-{status}"
    write_fixture(fixture, EchoParams(n_packets=80, burst_start=0.0, burst_end=1.0, return_loss=0.0), 4)
    trace = read_echo_logs(fixture / "sender.csv", fixture / "echoer.csv", trace_id=trace_id, status=status,
                           **trace_kwargs)
    path = tmp_path / f"fit-{status}.json"
    path.write_text(json.dumps(fit_echo_trace(trace, n_boot=10, seed=2)))
    return path


def test_measured_fit_is_labelled_measured_with_trace_id_and_hash(tmp_path):
    # Truth: the trace was imported with status "measured" and trace id "bench-7". The fixture
    # generator makes the packets; only the declared status matters for the label.
    fit_path = _fit_file(tmp_path, "measured", "bench-7", source="bench capture",
                         provenance=MEASURED_PROVENANCE)
    out = tmp_path / "measured-out"
    manifest = run_rerun(fit_path, out, pilot=True, seeds_count=2)
    summary = json.loads((out / "summary.json").read_text())
    assert "measured" in summary["status"] and "synthetic" not in summary["status"]
    assert "sensor simulated" in summary["status"]
    assert "no hardware data" not in summary["limits"]
    expected_hash = hashlib.sha256(fit_path.read_bytes()).hexdigest()
    assert summary["provenance"] == {"input_kind": "espnow-fit", "input_status": "measured",
                                     "trace_id": "bench-7", "input_file": fit_path.name,
                                     "input_sha256": expected_hash}
    assert manifest["provenance"] == summary["provenance"]


@pytest.mark.parametrize("status", ["synthetic", "fixture"])
def test_synthetic_and_fixture_fits_stay_synthetic(tmp_path, status):
    # Truth: the trace was imported with this status and trace id "gen-1"; neither is a measurement.
    fit_path = _fit_file(tmp_path, status, "gen-1")
    raw = fit_path.read_bytes()
    provenance = describe_input(json.loads(raw), raw, fit_path.name)
    assert provenance["input_kind"] == "espnow-fit"
    assert provenance["input_status"] == status
    assert provenance["trace_id"] == "gen-1"
    label, limits = status_label(provenance)
    assert "measured" not in label and "measured" not in " ".join(limits)
    assert label == "synthetic investigation: simulated link and sensor"  # unchanged from before provenance
    assert "no hardware data" in limits


def test_small_link_json_carries_its_declared_status(tmp_path):
    # Truth: a small link JSON that declares status "measured" and trace id "cam-3".
    source = tmp_path / "link.json"
    source.write_text(json.dumps({"delay_s": 0.12, "jitter_s": 0.008, "loss_probability": 0,
                                  "status": "measured", "trace_id": "cam-3"}))
    out = tmp_path / "small-out"
    run_rerun(source, out, pilot=True, seeds_count=2)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "investigation: link from measured input, sensor simulated"
    assert summary["provenance"]["input_kind"] == "link-json"
    assert summary["provenance"]["trace_id"] == "cam-3"


@pytest.mark.parametrize("data", [
    {"delay_s": 0.01, "loss_probability": 0},                                   # no status at all
    {"delay_s": 0.01, "loss_probability": 0, "status": "Measured"},             # not an exact known status
    {"delay_s": 0.01, "loss_probability": 0, "status": ["measured"]},           # not a string
    {"schema": "zephyr-espnow-fit-1", "s6_deliver": {"delay_s": 0, "loss_probability": 0}},  # fit, no source_trace
    {"schema": "zephyr-espnow-fit-1", "s6_deliver": {"delay_s": 0, "loss_probability": 0},
     "source_trace": {"trace_id": 5, "status": "real"}},                       # fit, odd source_trace
])
def test_missing_or_unrecognised_status_is_labelled_unknown(data):
    # Truth: none of these inputs declares one of measured, synthetic or fixture.
    raw = json.dumps(data).encode()
    provenance = describe_input(data, raw, "in.json")
    assert provenance["input_status"] == "unknown"
    assert provenance["trace_id"] is None
    assert provenance["input_sha256"] == hashlib.sha256(raw).hexdigest()
    label, limits = status_label(provenance)
    assert "unknown" in label and "measured" not in label
    assert "synthetic" not in label
