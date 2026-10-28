"""Tests for tools/intake_vision.py (no OpenCV, no camera).

The shared run uses a synthetic detection log with a known noise (0.03 m),
known bias and known latency, plus a synthetic latency recording. The checks
compare the intake's numbers with that known truth. The synthetic latency
jitter is abs(normal(0, sigma)), so its truth is the realized mean and std from
the recording's answer key (about 126.6 ms and 4.8 ms for 120 and 8), not the
two command-line numbers.
"""
import hashlib
import json
import re
import time

import numpy as np
import pytest

from sim.detection_log import read_detection_log, write_detection_log
from sim.vision_latency import SynthConfig, synth_recording
from sim.vision_trace import VisionTrace
from tools import intake_vision as tool
from tools import make_detection_fixture as fixture_cli
from tools import vision_latency as latency_cli
from tools.s6_measured_rerun import build_rerun_cells

NOISE = 0.03
BIAS = (0.02, -0.01, 0.05)
FIXTURE_LATENCY_S = 0.04
MEASURED_LATENCY_META = {"schema": "zephyr-vision-latency-run-1", "status": "measured", "n_bits": 8,
                         "opencv_version": "4.x", "platform": "bench-test-platform",
                         "clock": "time.monotonic (same clock for frames and display)"}
PROCESSING = {"schema": "zephyr-processing-time-1", "median_s": 0.015, "p95_s": 0.021, "n": 300,
              "status": "synthetic"}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_log(path, *, seed=3, frames=600, noise=NOISE):
    fixture_cli.main([str(path), "--frames", str(frames), "--rate", "30", "--noise-std", str(noise),
                      "--bias", *(str(b) for b in BIAS), "--dropout", "0.1",
                      "--latency-mean", str(FIXTURE_LATENCY_S), "--latency-std", "0.01", "--seed", str(seed)])


def _make_latency(run_dir, *, latency_ms=120, jitter_ms=8):
    assert latency_cli.main(["synth", str(run_dir), "--latency-ms", str(latency_ms),
                             "--jitter-ms", str(jitter_ms)]) == 0


def _intake(*args):
    return tool.main([str(a) for a in args])


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    log = root / "cal-a.jsonl"
    _make_log(log)
    run = root / "lat-a"
    _make_latency(run)
    truth = synth_recording(SynthConfig(latency_s=0.120, latency_jitter_s=0.008, seed=1)).true_latency_s
    return root, log, run, truth


@pytest.fixture(scope="module")
def pilot(bench):
    """One shared pilot intake: a log, a latency run, a processing JSON."""
    root, log, run, _ = bench
    proc = root / "proc.json"
    proc.write_text(json.dumps(PROCESSING))
    out = root / "intake-out"
    before = {p: _sha(p) for p in (log, proc, *run.iterdir())}
    start = time.perf_counter()
    code = _intake("--log", log, "--latency-run", run, "--processing", proc, "--output", out, "--pilot")
    elapsed = time.perf_counter() - start
    assert code == 0
    return {"out": out, "elapsed": elapsed, "before": before, "proc": proc}


def _fit(pilot_out, name="cal-a"):
    return json.loads((pilot_out / "fits" / f"{name}.fit.json").read_text())


def test_fit_recovers_the_known_noise_and_bias_inside_its_own_interval(pilot):
    fit = _fit(pilot["out"])
    unc = fit["uncertainty"]
    pooled = unc["noise_std_m"]
    assert pooled["ci_low"] <= NOISE <= pooled["ci_high"]
    assert pooled["estimate"] == pytest.approx(NOISE, rel=0.10)
    for axis, truth in zip("xyz", BIAS):
        rec = unc[f"bias_m_{axis}"]
        assert rec["ci_low"] <= truth <= rec["ci_high"], axis
    delay = unc["measurement_delay_s"]
    assert delay["ci_low"] <= FIXTURE_LATENCY_S <= delay["ci_high"]


def test_link_matches_the_recordings_answer_key_not_the_nominal_knobs(bench, pilot):
    _, _, _, truth = bench
    link = json.loads((pilot["out"] / "links" / "lat-a.link.json").read_text())
    analysis = json.loads((pilot["out"] / "latency" / "lat-a.latency.json").read_text())
    stated = analysis["apparent_latency_s"]["stated_error"]
    camera_upper = link["components"]["camera"]["latency_s"]["upper"]
    true_mean, true_std = float(np.mean(truth)), float(np.std(truth, ddof=1))
    assert true_mean <= camera_upper <= true_mean + 2 * stated
    assert abs(camera_upper - 0.120) > 0.004  # the nominal 120 ms would be wrong
    # With a processing median of 15 ms the link delay is exactly 15 ms above the camera upper bound.
    assert link["delay_s"] == pytest.approx(camera_upper + 0.015, abs=1e-12)
    cam_jitter = link["components"]["camera"]["jitter_s"]
    assert true_std * 0.8 <= cam_jitter <= true_std + 0.005  # straddle estimate reads a little high
    assert abs(cam_jitter - 0.008) > 0.002  # the nominal 8 ms would be wrong


def test_everything_stays_synthetic(pilot):
    out = pilot["out"]
    text = (out / "INTAKE.md").read_text()
    assert re.search(r"^Overall status: synthetic$", text, re.MULTILINE)
    assert json.loads((out / "links" / "lat-a.link.json").read_text())["status"] == "synthetic"
    assert _fit(out)["input"]["status"] == "synthetic"
    summary = json.loads((out / "s6" / "cal-a" / "pooled" / "summary.json").read_text())
    assert "synthetic" in summary["status"]
    assert "measured fit" not in summary["status"]


def test_pilot_is_fast_and_does_not_touch_inputs(pilot):
    assert pilot["elapsed"] < 120
    for path, digest in pilot["before"].items():
        assert _sha(path) == digest, path.name
    # Nothing is written next to the inputs (no latency.json in the run folder).
    assert not (pilot["proc"].parent / "lat-a" / "latency.json").exists()


def test_intake_md_lists_every_input_and_output_hash(pilot):
    out = pilot["out"]
    text = (out / "INTAKE.md").read_text()
    rows = dict(re.findall(r"^\| ([^|]+?) \| `?([0-9a-f]{64})`? \|", text, re.MULTILINE))
    # inputs: the log, the processing JSON, and every file of the latency run
    assert rows["cal-a.jsonl"] == _sha(out.parent / "cal-a.jsonl")
    assert rows["proc.json"] == _sha(pilot["proc"])
    for name in ("frames.npz", "display_log.csv", "run_meta.json"):
        assert rows[f"lat-a/{name}"] == _sha(out.parent / "lat-a" / name)
    # outputs: every non-volatile file the intake wrote
    written = [p for p in out.rglob("*") if p.is_file()
               and p.name not in ("INTAKE.md", "manifest.json", "README.txt")]
    assert written
    for path in written:
        key = path.relative_to(out).as_posix()
        assert rows[key] == _sha(path), key
    # the volatile files are named as excluded, not hashed
    assert "manifest.json" in text and "volatile" in text


def test_intake_md_has_no_local_paths_or_em_dashes(pilot):
    text = (pilot["out"] / "INTAKE.md").read_text()
    assert "/" + "Users" not in text and str(pilot["out"].parent) not in text
    assert "\u2014" not in text


def test_intake_md_reports_fitted_values_and_assumed_display_lag(pilot):
    text = (pilot["out"] / "INTAKE.md").read_text()
    assert "noise_std_m" in text and "bias" in text and "dropout" in text
    assert "display lag" in text.lower() and "assumption" in text.lower()
    assert "tracker" in text.lower()  # states what the rerun does not apply


def test_pilot_uses_a_subset_of_the_registered_main_slice(pilot):
    manifest = json.loads((pilot["out"] / "s6" / "cal-a" / "manifest.json").read_text())
    assert manifest["cell_filter_used"] is True
    assert manifest["seeds_count"] == 1
    assert manifest["processes"] == 1
    full = len(build_rerun_cells(NOISE))
    assert 0 < manifest["n_cells"]["pooled"] < full


def test_pilot_filter_picks_both_motions_and_a_loss_free_and_lossy_cell():
    cells = build_rerun_cells(NOISE)
    chosen = [c for c in cells if tool.pilot_filter(c)]
    assert len(chosen) == 8
    assert {c["motion"] for c in chosen} == {c["motion"] for c in cells}
    assert {c["loss_probability"] for c in chosen} == {0.0, 0.30}
    assert {c["delay_s"] for c in chosen} == {0.0, 0.10}


def test_two_runs_give_identical_intake_md(bench, tmp_path):
    _, log, run, _ = bench
    texts = []
    for name in ("one", "two"):
        out = tmp_path / name
        assert _intake("--log", log, "--latency-run", run, "--output", out, "--pilot") == 0
        texts.append((out / "INTAKE.md").read_bytes())
    assert texts[0] == texts[1]


def test_measured_log_without_provenance_fails_loudly_and_writes_nothing(bench, tmp_path, capsys, monkeypatch):
    _, log, run, _ = bench

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("the fit ran before the provenance check")

    monkeypatch.setattr(tool.fit_vision_noise, "main", must_not_run)
    lines = log.read_text().splitlines()
    header = json.loads(lines[0])
    header["status"] = "measured"
    header["source"] = "bench-camera"
    header.pop("provenance", None)
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join([json.dumps(header, sort_keys=True), *lines[1:]]) + "\n")
    out = tmp_path / "out"
    code = _intake("--log", bad, "--latency-run", run, "--output", out, "--pilot")
    assert code != 0
    err = capsys.readouterr().err
    assert "bad.jsonl" in err and "measured" in err and "provenance" in err
    assert "this input was not used" in err
    assert not out.exists()


def test_measured_latency_run_without_recorder_metadata_fails_loudly(bench, tmp_path, capsys):
    _, log, _, _ = bench
    run = tmp_path / "lat-m"
    _make_latency(run)
    (run / "run_meta.json").write_text(json.dumps({"schema": "zephyr-vision-latency-run-1",
                                                   "status": "measured", "n_bits": 8}))
    out = tmp_path / "out"
    code = _intake("--log", log, "--latency-run", run, "--output", out, "--pilot")
    assert code != 0
    err = capsys.readouterr().err
    assert "lat-m" in err and "measured" in err and "provenance" in err
    assert not out.exists()


def test_measured_latency_run_with_recorder_metadata_is_accepted_but_stays_synthetic_overall(bench, tmp_path):
    _, log, _, _ = bench
    run = tmp_path / "lat-m"
    _make_latency(run)
    (run / "run_meta.json").write_text(json.dumps(MEASURED_LATENCY_META))
    out = tmp_path / "out"
    assert _intake("--log", log, "--latency-run", run, "--output", out, "--pilot") == 0
    link = json.loads((out / "links" / "lat-m.link.json").read_text())
    assert link["components"]["camera"]["status"] == "measured"
    assert link["status"] == "measured"  # camera-only link from a measured run
    text = (out / "INTAKE.md").read_text()
    assert re.search(r"^Overall status: synthetic$", text, re.MULTILINE)  # the log is still synthetic


def _measured_log(tmp_path, source_log):
    trace = read_detection_log(source_log)
    doc = trace.as_dict()
    doc["status"] = "measured"
    doc["source"] = "bench-camera"
    doc["provenance"] = {
        "instrument": {"kind": "camera", "model": "bench-cam", "settings": {}},
        "calibration_ref": None, "calibration_ref_null_reason": "test",
        "raw_sha256": "b" * 64, "code_revision": "abc123", "captured_at": "2026-10-30T09:30:00-06:00",
        "operator": "bench operator", "ground_truth": None, "ground_truth_null_reason": "fixture truth in observations"}
    path = tmp_path / "measured.jsonl"
    write_detection_log(VisionTrace.from_dict(doc), path)
    return path


def test_all_measured_inputs_give_overall_measured(bench, tmp_path):
    _, log, _, _ = bench
    mlog = _measured_log(tmp_path, log)
    run = tmp_path / "lat-m"
    _make_latency(run)
    (run / "run_meta.json").write_text(json.dumps(MEASURED_LATENCY_META))
    proc = tmp_path / "proc.json"
    proc.write_text(json.dumps({**PROCESSING, "status": "measured"}))
    out = tmp_path / "out"
    assert _intake("--log", mlog, "--latency-run", run, "--processing", proc, "--output", out, "--pilot") == 0
    text = (out / "INTAKE.md").read_text()
    assert re.search(r"^Overall status: measured$", text, re.MULTILINE)
    assert _fit(out, "measured")["input"]["status"] == "measured"


def test_one_synthetic_input_makes_the_whole_intake_synthetic(bench, tmp_path):
    _, log, _, _ = bench
    mlog = _measured_log(tmp_path, log)
    run = tmp_path / "lat-s"
    _make_latency(run)  # synthetic
    out = tmp_path / "out"
    assert _intake("--log", mlog, "--latency-run", run, "--output", out, "--pilot") == 0
    assert re.search(r"^Overall status: synthetic$", (out / "INTAKE.md").read_text(), re.MULTILINE)


def test_missing_status_makes_the_intake_unknown_not_measured(bench, tmp_path):
    _, log, _, _ = bench
    run = tmp_path / "lat-u"
    _make_latency(run)
    (run / "run_meta.json").unlink()
    out = tmp_path / "out"
    assert _intake("--log", log, "--latency-run", run, "--output", out, "--pilot") == 0
    link = json.loads((out / "links" / "lat-u.link.json").read_text())
    assert link["components"]["camera"]["status"] == "unknown"


def test_refuses_output_inside_sim_runs(bench, capsys):
    _, log, run, _ = bench
    code = _intake("--log", log, "--latency-run", run, "--output", tool.REPO_ROOT / "sim" / "runs" / "intake-test",
                   "--pilot")
    assert code != 0
    assert "sim/runs" in capsys.readouterr().err
    assert not (tool.REPO_ROOT / "sim" / "runs" / "intake-test").exists()


def test_refuses_a_non_empty_output_folder(bench, tmp_path, capsys):
    _, log, run, _ = bench
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    assert _intake("--log", log, "--latency-run", run, "--output", out, "--pilot") != 0
    assert "not empty" in capsys.readouterr().err
    assert (out / "keep.txt").read_text() == "keep"
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_two_inputs_with_the_same_name_are_refused(bench, tmp_path, capsys):
    _, log, run, _ = bench
    other = tmp_path / "elsewhere"
    other.mkdir()
    _make_log(other / "cal-a.jsonl", seed=4)
    out = tmp_path / "out"
    assert _intake("--log", log, "--log", other / "cal-a.jsonl", "--latency-run", run, "--output", out,
                   "--pilot") != 0
    assert "cal-a" in capsys.readouterr().err
    assert not out.exists()


def test_processing_without_a_latency_run_is_refused(bench, tmp_path, capsys):
    _, log, _, _ = bench
    proc = tmp_path / "proc.json"
    proc.write_text(json.dumps(PROCESSING))
    out = tmp_path / "out"
    assert _intake("--log", log, "--processing", proc, "--output", out, "--pilot") != 0
    assert "latency" in capsys.readouterr().err
    assert not out.exists()


def test_no_inputs_is_refused(tmp_path, capsys):
    assert _intake("--output", tmp_path / "out", "--pilot") != 0
    assert "at least one" in capsys.readouterr().err


def test_a_failing_fit_removes_the_folder_it_created(bench, tmp_path, capsys):
    _, log, run, _ = bench
    broken = tmp_path / "broken.jsonl"
    broken.write_text("not json\n")
    out = tmp_path / "out"
    assert _intake("--log", log, "--log", broken, "--latency-run", run, "--output", out, "--pilot") != 0
    assert "broken.jsonl" in capsys.readouterr().err
    assert not out.exists()


def test_two_logs_get_two_fits_and_two_reruns(bench, tmp_path):
    _, log, _, _ = bench
    second = tmp_path / "cal-b.jsonl"
    _make_log(second, seed=5, noise=0.06)
    out = tmp_path / "out"
    assert _intake("--log", log, "--log", second, "--output", out, "--pilot", "--bracket", "pooled") == 0
    assert _fit(out, "cal-a")["sensor_config"]["noise_std_m"] == pytest.approx(0.03, rel=0.1)
    assert _fit(out, "cal-b")["sensor_config"]["noise_std_m"] == pytest.approx(0.06, rel=0.1)
    assert (out / "s6" / "cal-a" / "pooled" / "summary.json").exists()
    assert (out / "s6" / "cal-b" / "pooled" / "summary.json").exists()


def test_no_latency_run_means_no_link_section_and_no_link_file(bench, tmp_path):
    _, log, _, _ = bench
    out = tmp_path / "out"
    assert _intake("--log", log, "--output", out, "--pilot", "--bracket", "pooled") == 0
    assert not (out / "links").exists()
    assert "no latency run" in (out / "INTAKE.md").read_text().lower()


def test_one_processing_time_is_added_to_every_latency_run(bench, tmp_path):
    _, log, run, _ = bench
    second = tmp_path / "lat-b"
    _make_latency(second, latency_ms=80, jitter_ms=4)
    proc = tmp_path / "proc.json"
    proc.write_text(json.dumps(PROCESSING))
    out = tmp_path / "out"
    assert _intake("--log", log, "--latency-run", run, "--latency-run", second, "--processing", proc,
                   "--output", out, "--pilot", "--bracket", "pooled") == 0
    uppers = []
    for name in ("lat-a", "lat-b"):
        link = json.loads((out / "links" / f"{name}.link.json").read_text())
        upper = link["components"]["camera"]["latency_s"]["upper"]
        assert link["delay_s"] == pytest.approx(upper + 0.015, abs=1e-12)
        uppers.append(upper)
    assert uppers[0] - uppers[1] > 0.03  # the 120 ms and 80 ms recordings stay distinguishable
    assert (out / "INTAKE.md").read_text().count("### lat-") == 2


def test_overall_unknown_when_nothing_is_synthetic_and_a_status_is_missing(bench, tmp_path):
    _, log, _, _ = bench
    mlog = _measured_log(tmp_path, log)
    run = tmp_path / "lat-u"
    _make_latency(run)
    (run / "run_meta.json").unlink()
    out = tmp_path / "out"
    assert _intake("--log", mlog, "--latency-run", run, "--output", out, "--pilot", "--bracket", "pooled") == 0
    assert re.search(r"^Overall status: unknown$", (out / "INTAKE.md").read_text(), re.MULTILINE)


def test_a_processing_file_without_status_keeps_an_otherwise_measured_intake_unknown(bench, tmp_path):
    _, log, _, _ = bench
    mlog = _measured_log(tmp_path, log)
    run = tmp_path / "lat-m"
    _make_latency(run)
    (run / "run_meta.json").write_text(json.dumps(MEASURED_LATENCY_META))
    proc = tmp_path / "proc.json"
    proc.write_text(json.dumps({k: v for k, v in PROCESSING.items() if k != "status"}))
    out = tmp_path / "out"
    assert _intake("--log", mlog, "--latency-run", run, "--processing", proc, "--output", out, "--pilot",
                   "--bracket", "pooled") == 0
    assert re.search(r"^Overall status: unknown$", (out / "INTAKE.md").read_text(), re.MULTILINE)


def test_a_fixture_log_is_shown_as_fixture_and_never_counts_as_measured(bench, tmp_path):
    _, log, _, _ = bench
    doc = read_detection_log(log).as_dict()
    doc["status"] = "fixture"
    path = tmp_path / "fx.jsonl"
    write_detection_log(VisionTrace.from_dict(doc), path)
    out = tmp_path / "out"
    assert _intake("--log", path, "--output", out, "--pilot", "--bracket", "pooled") == 0
    text = (out / "INTAKE.md").read_text()
    assert re.search(r"^\| fx.jsonl \| [0-9a-f]{64} \| fixture \|$", text, re.MULTILINE)
    assert re.search(r"^Overall status: synthetic$", text, re.MULTILINE)


def test_a_corrupt_recording_gives_a_plain_error_and_leaves_no_output(bench, tmp_path, capsys):
    _, log, _, _ = bench
    run = tmp_path / "lat-bad"
    _make_latency(run)
    (run / "frames.npz").write_bytes(b"this is not an npz file")
    for existing in (False, True):
        out = tmp_path / f"out-{existing}"
        if existing:
            out.mkdir()
        assert _intake("--log", log, "--latency-run", run, "--output", out, "--pilot", "--bracket", "pooled") == 2
        err = capsys.readouterr().err
        assert "error:" in err and "lat-bad" in err and "Traceback" not in err
        assert out.exists() == existing
        if existing:
            assert list(out.iterdir()) == []  # a folder the user made is emptied, not deleted


def test_a_bad_processing_file_is_refused_before_any_stage_runs(bench, tmp_path, capsys, monkeypatch):
    _, log, run, _ = bench

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("a stage ran before the processing file was checked")

    monkeypatch.setattr(tool.fit_vision_noise, "main", must_not_run)
    proc = tmp_path / "proc.json"
    proc.write_text(json.dumps({"median_s": 0.01}))  # no schema, no p95_s, no n
    out = tmp_path / "out"
    assert _intake("--log", log, "--latency-run", run, "--processing", proc, "--output", out, "--pilot") == 2
    assert "proc.json" in capsys.readouterr().err
    assert not out.exists()


def test_intake_md_says_the_pilot_is_not_a_result_and_the_interval_is_approximate(pilot):
    text = (pilot["out"] / "INTAKE.md").read_text()
    assert "It is not a result." in text
    assert "approximate interval" in text


@pytest.mark.parametrize("seeds, subset, expected", [
    (1, True, "one-seed medians on a subset of cells"),
    (3, True, "3-seed medians on a subset of cells"),
    (1, False, "one-seed medians on the full main slice"),
])
def test_pilot_note_matches_the_real_seeds_and_cells(seeds, subset, expected):
    note = tool._pilot_note(seeds, subset)
    assert expected in note
    assert "not a result" in note
