"""Tests for tools/aruco_processing_time.py.

The statistics and the writer are pure and use an injected timer, so every
number below is checked against a hand calculation. The OpenCV path (frames or
video through the real detector) needs the camera environment and is skipped in
the repo's .venv: it is UNTESTED until it has been run there.
"""
import inspect
import json
from pathlib import Path

import pytest

from tools import aruco_processing_time as tool
from tools import vision_latency_to_link as link_tool

GOOD_PROV = {"operator": "A. Tester", "captured_at": "2026-10-30T10:00:00-06:00",
             "code_revision": "abc1234", "raw_sha256": "ab" * 32}


class FakeTimer:
    """Returns scripted start/stop pairs so each timed call has an exact known duration."""

    def __init__(self, durations):
        self.values = []
        t = 100.0
        for d in durations:
            self.values += [t, t + d]
            t += d + 0.5  # a gap between calls that must not leak into the result
        self.calls = 0

    def __call__(self):
        value = self.values[self.calls]
        self.calls += 1
        return value


def _durations(ms_values):
    return [v / 1000 for v in ms_values]


# ------------------------------------------------------------ statistics by hand

def test_twenty_frames_give_the_hand_median_and_nearest_rank_p95():
    timings = _durations(range(1, 21))  # 1..20 ms
    stats = tool.summarize(timings)
    assert stats["n"] == 20
    assert stats["median_s"] == pytest.approx(0.0105)  # mean of the 10th and 11th values
    assert stats["p95_s"] == pytest.approx(0.019)  # ceil(0.95 * 20) = 19th value
    assert stats["min_s"] == pytest.approx(0.001)
    assert stats["max_s"] == pytest.approx(0.020)


def test_order_of_the_timings_does_not_matter():
    shuffled = _durations([7, 1, 20, 3, 14, 9, 2, 18, 5, 11, 16, 4, 13, 8, 19, 6, 15, 10, 12, 17])
    assert tool.summarize(shuffled) == tool.summarize(_durations(range(1, 21)))


def test_odd_count_median_is_the_middle_value_and_one_frame_is_allowed():
    assert tool.summarize(_durations([5, 1, 9]))["median_s"] == pytest.approx(0.005)
    one = tool.summarize([0.012])
    assert one["median_s"] == one["p95_s"] == 0.012 and one["n"] == 1


def test_a_slow_outlier_moves_p95_but_not_the_median():
    base = _durations([10] * 99 + [10])
    spiky = _durations([10] * 90 + [100] * 10)  # 10 slow frames of 100
    assert tool.summarize(base)["p95_s"] == pytest.approx(0.010)
    assert tool.summarize(spiky)["median_s"] == pytest.approx(0.010)
    assert tool.summarize(spiky)["p95_s"] == pytest.approx(0.100)  # rank 95 of 100 is a slow frame


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")], [-0.001], [True], ["0.1"]])
def test_bad_timings_are_refused(bad):
    with pytest.raises(tool.ProcessingTimeError):
        tool.summarize(bad)


# ------------------------------------------------------------ timing loop

def test_time_calls_measures_only_the_detector_call():
    durations = _durations([3, 5, 9])
    timer = FakeTimer(durations)
    seen = []
    out = tool.time_calls(["a", "b", "c"], seen.append, timer=timer)
    assert out == pytest.approx(durations)
    assert seen == ["a", "b", "c"]
    assert timer.calls == 6  # two timer reads per frame, nothing else


def test_warmup_frames_run_but_are_not_counted():
    durations = _durations([50, 40, 3, 5])  # the first two are slow warm-up frames
    timer = FakeTimer(durations)
    seen = []
    out = tool.time_calls([1, 2, 3, 4], seen.append, timer=timer, warmup=2)
    assert seen == [1, 2, 3, 4]  # the detector still ran on the warm-up frames
    assert out == pytest.approx([0.003, 0.005])


def test_max_frames_stops_early_and_too_few_frames_is_an_error():
    timer = FakeTimer(_durations([1, 2, 3, 4]))
    assert len(tool.time_calls(range(10), lambda f: None, timer=timer, max_frames=3)) == 3
    with pytest.raises(tool.ProcessingTimeError, match="no frames"):
        tool.time_calls([], lambda f: None, timer=FakeTimer([]))
    with pytest.raises(tool.ProcessingTimeError, match="warm-up"):
        tool.time_calls([1, 2], lambda f: None, timer=FakeTimer(_durations([1, 2])), warmup=2)


def test_after_callback_runs_outside_the_timed_window_with_each_result():
    timer = FakeTimer(_durations([1, 2, 3]))
    log = []

    def after(index, result):
        log.append((index, result, timer.calls))  # timer.calls shows both reads of this frame are already done

    out = tool.time_calls(["a", "b", "c"], lambda f: f.upper(), timer=timer, warmup=1, after=after)
    assert log == [(1, "A", 2), (2, "B", 4), (3, "C", 6)]
    assert out == pytest.approx([0.002, 0.003])


def test_a_timer_that_runs_backwards_is_refused():
    with pytest.raises(tool.ProcessingTimeError, match="backwards"):
        tool.time_calls([1], lambda f: None, timer=iter([5.0, 4.0]).__next__)


# ------------------------------------------------------------ hash and report

def test_raw_hash_is_the_sha256_of_nine_decimal_text_lines():
    import hashlib
    text = "0.001000000\n0.002500000\n"
    assert tool.timings_sha256([0.001, 0.0025]) == hashlib.sha256(text.encode()).hexdigest()
    assert tool.timings_sha256([0.001, 0.0026]) != tool.timings_sha256([0.001, 0.0025])


def _report(timings=None, **over):
    timings = _durations(range(1, 21)) if timings is None else timings
    kwargs = dict(status="measured", cv2_version="5.0.0", dictionary="DICT_4X4_50",
                  machine={"platform": "test-os", "machine": "arm64"},
                  source={"kind": "frames", "name": "frames", "frames_read": 22},
                  warmup_frames=2, operator="A. Tester", captured_at="2026-10-30T10:00:00-06:00",
                  code_revision="abc1234")
    kwargs.update(over)
    return tool.build_report(timings, **kwargs)


def test_report_carries_stats_cv2_version_machine_and_provenance():
    report = _report()
    assert report["schema"] == "zephyr-processing-time-1"
    assert report["status"] == "measured"
    assert (report["median_s"], report["p95_s"], report["n"]) == (pytest.approx(0.0105), pytest.approx(0.019), 20)
    assert report["cv2_version"] == "5.0.0"
    assert report["machine"]["platform"] == "test-os"
    prov = report["provenance"]
    assert prov["operator"] == "A. Tester" and prov["code_revision"] == "abc1234"
    assert prov["raw_sha256"] == tool.timings_sha256(_durations(range(1, 21)))
    assert "stage" in report and "detect" in report["stage"]


def test_a_measured_report_passes_the_reader_and_keeps_the_numbers():
    parsed = link_tool._processing_from_json(_report())
    assert parsed["status"] == "measured"
    assert parsed["delay_s"] == pytest.approx(0.0105) and parsed["n"] == 20


def test_building_a_measured_report_without_provenance_fields_is_refused():
    for field in ("operator", "captured_at", "code_revision"):
        with pytest.raises(tool.ProcessingTimeError, match=field):
            _report(**{field: ""})


def test_report_json_has_no_absolute_local_path(tmp_path):
    out = tmp_path / "pt.json"
    tool.write_report(out, _report(), _durations(range(1, 21)))
    text = out.read_text(encoding="utf-8")
    assert "/Users/" not in text and str(tmp_path) not in text


# ------------------------------------------------------------ writer and check

def test_write_then_check_round_trips(tmp_path):
    out = tmp_path / "pt.json"
    timings = _durations(range(1, 21))
    tool.write_report(out, _report(timings), timings)
    sidecar = tmp_path / "pt.timings.csv"
    assert sidecar.is_file()
    assert len(sidecar.read_text(encoding="utf-8").split()) == 21  # header + 20 rows
    assert tool.check_report(out) == []


def test_write_refuses_to_overwrite(tmp_path):
    out = tmp_path / "pt.json"
    timings = _durations(range(1, 21))
    tool.write_report(out, _report(timings), timings)
    with pytest.raises(tool.ProcessingTimeError, match="exists"):
        tool.write_report(out, _report(timings), timings)


def test_check_catches_an_edited_timing_or_median(tmp_path):
    out = tmp_path / "pt.json"
    timings = _durations(range(1, 21))
    tool.write_report(out, _report(timings), timings)
    sidecar = tmp_path / "pt.timings.csv"
    sidecar.write_text(sidecar.read_text(encoding="utf-8").replace("0.010000000", "0.011000000"), encoding="utf-8")
    problems = tool.check_report(out)
    assert any("sha256" in p for p in problems)
    # an edited median in the JSON (sidecar restored) is also caught
    tool.write_timings_csv(sidecar, timings, overwrite=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    data["median_s"] = 0.001
    out.write_text(json.dumps(data), encoding="utf-8")
    assert any("median_s" in p for p in tool.check_report(out))


def test_check_without_the_sidecar_says_so(tmp_path):
    out = tmp_path / "pt.json"
    timings = _durations(range(1, 21))
    tool.write_report(out, _report(timings), timings)
    (tmp_path / "pt.timings.csv").unlink()
    assert any("timings file" in p for p in tool.check_report(out))


# ------------------------------------------------------------ the reader refuses unproven "measured"

def _bare(status="measured", **extra):
    base = {"schema": "zephyr-processing-time-1", "median_s": 0.015, "p95_s": 0.021, "n": 300, "status": status}
    base.update(extra)
    return base


def test_reader_refuses_measured_without_provenance():
    with pytest.raises(link_tool.LinkConversionError, match="provenance"):
        link_tool._processing_from_json(_bare())


@pytest.mark.parametrize("field", ["operator", "captured_at", "code_revision", "raw_sha256"])
def test_reader_names_the_missing_provenance_field(field):
    prov = {k: v for k, v in GOOD_PROV.items() if k != field}
    with pytest.raises(link_tool.LinkConversionError, match=field):
        link_tool._processing_from_json(_bare(provenance=prov))


@pytest.mark.parametrize("field,value", [("operator", ""), ("operator", "  "), ("captured_at", 5),
                                         ("raw_sha256", "xyz"), ("raw_sha256", "ab" * 31), ("raw_sha256", "AB" * 32)])
def test_reader_refuses_empty_or_malformed_provenance_values(field, value):
    with pytest.raises(link_tool.LinkConversionError, match=field):
        link_tool._processing_from_json(_bare(provenance=dict(GOOD_PROV, **{field: value})))


def test_reader_refuses_a_provenance_that_is_not_an_object():
    with pytest.raises(link_tool.LinkConversionError, match="provenance"):
        link_tool._processing_from_json(_bare(provenance="trust me"))


def test_reader_accepts_measured_with_full_provenance():
    parsed = link_tool._processing_from_json(_bare(provenance=GOOD_PROV))
    assert parsed["status"] == "measured"


@pytest.mark.parametrize("status", ["synthetic", None])
def test_reader_does_not_demand_provenance_for_synthetic_or_unlabelled_files(status):
    data = _bare()
    if status is None:
        data.pop("status")
    else:
        data["status"] = status
    parsed = link_tool._processing_from_json(data)
    assert parsed["status"] in ("synthetic", "unknown")


# ------------------------------------------------------------ command line (pure parts)

def test_check_command_prints_ok_and_returns_zero(tmp_path, capsys):
    out = tmp_path / "pt.json"
    timings = _durations(range(1, 21))
    tool.write_report(out, _report(timings), timings)
    assert tool.main(["check", str(out)]) == 0
    assert "ok" in capsys.readouterr().out


def test_check_reports_a_problem_for_json_that_is_not_an_object(tmp_path, capsys):
    out = tmp_path / "pt.json"
    out.write_text("[]", encoding="utf-8")
    assert tool.main(["check", str(out)]) == 1
    assert "not an object" in capsys.readouterr().err


def test_measure_needs_an_explicit_status_and_a_known_dictionary(tmp_path, capsys):
    base = ["measure", str(tmp_path), "--out", str(tmp_path / "o.json"), "--operator", "A"]
    with pytest.raises(SystemExit):
        tool.main(base)  # no --status
    with pytest.raises(SystemExit):
        tool.main(base + ["--status", "measured", "--dictionary", "DICT_NOPE"])
    with pytest.raises(SystemExit):
        tool.main(base + ["--status", "measured", "--hfov-deg", "200"])


def test_check_command_fails_on_a_missing_file(tmp_path, capsys):
    assert tool.main(["check", str(tmp_path / "nope.json")]) != 0
    assert "not found" in capsys.readouterr().err


def test_measure_without_opencv_explains_instead_of_crashing(tmp_path, capsys, monkeypatch):
    def no_cv2():
        raise tool.ProcessingTimeError("OpenCV (cv2) is not installed in this Python.")
    monkeypatch.setattr(tool, "_load_cv2", no_cv2)
    frames = tmp_path / "frames"
    frames.mkdir()
    code = tool.main(["measure", str(frames), "--out", str(tmp_path / "o.json"), "--operator", "A",
                     "--status", "measured"])
    assert code != 0 and "OpenCV" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


# ------------------------------------------------------------ measure() wiring with a fake OpenCV

def test_measure_wiring_with_a_fake_detector_writes_a_checked_file(tmp_path, monkeypatch, capsys):
    """Runs measure() end to end with stand-ins for cv2 and the detector. It proves the wiring (warm-up,
    counts, files, provenance), not the real OpenCV cost."""
    import types

    import numpy as np

    from tools import capture_aruco as cap

    class FakeCv2:
        __version__ = "fake-1.0"
        IMREAD_COLOR = 1

        @staticmethod
        def imread(path, flag):
            assert flag == 1  # color, like the live BGR frames
            # the file name's last digit decides whether the fake detector "finds" a marker
            return np.full((48, 64), int(Path(path).stem[-1]), dtype=np.uint8)

        @staticmethod
        def getNumThreads():
            return 4

    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(8):
        (folder / f"f{i}.png").write_bytes(b"x")
    (folder / "._f0.png").write_bytes(b"x")  # a macOS resource-fork leftover must be ignored
    monkeypatch.setattr(tool, "_load_cv2", lambda: FakeCv2)
    real_build, real_analyze = cap.build_context, cap.analyze_frame

    def fake_build(*args, **kwargs):
        inspect.signature(real_build).bind(*args, **kwargs)  # fails if the real signature drifts
        return types.SimpleNamespace(**kwargs)

    def fake_analyze(*args, **kwargs):
        inspect.signature(real_analyze).bind(*args, **kwargs)
        frame = args[0]
        return types.SimpleNamespace(outcome="detected" if frame[0, 0] % 2 == 0 else "missed")

    monkeypatch.setattr(cap, "build_context", fake_build)
    monkeypatch.setattr(cap, "analyze_frame", fake_analyze)
    monkeypatch.setattr(cap, "git_revision", lambda: "rev-for-test")
    out = tmp_path / "pt.json"
    assert tool.main(["measure", str(folder), "--out", str(out), "--operator", "A. Tester", "--warmup", "2",
                      "--status", "measured"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n"] == 6 and report["warmup_frames"] == 2
    assert report["source"] == {"kind": "frames", "name": "frames", "frames_read": 8, "frame_size_px": [64, 48]}
    assert report["frames_with_marker"] == 3  # frames 2..7 counted, of which 2, 4, 6 are even
    assert report["cv2_version"] == "fake-1.0" and report["machine"]["opencv_threads"] == 4
    assert report["provenance"]["code_revision"] == "rev-for-test" and report["status"] == "measured"
    assert report["median_s"] >= 0 and report["p95_s"] >= report["median_s"]
    assert tool.check_report(out) == []
    assert link_tool._processing_from_json(report)["n"] == 6
    assert "warning" in capsys.readouterr().err  # not every timed frame had the marker
    assert tool.main(["measure", str(folder), "--out", str(out), "--operator", "A",
                      "--status", "measured"]) == 2  # no overwrite


# ------------------------------------------------------------ the OpenCV path (skipped in .venv)

def test_cv2_end_to_end_on_generated_marker_frames(tmp_path):  # pragma: no cover - needs OpenCV
    cv2 = pytest.importorskip("cv2", reason="OpenCV is only in the camera environment; this path is UNTESTED here")
    import numpy as np
    from tools import capture_aruco as cap
    dictionary = cap.get_dictionary(cv2, "DICT_4X4_50")
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 200, borderBits=1)
    frame = np.full((480, 640), 255, dtype=np.uint8)
    frame[140:340, 220:420] = marker
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(8):
        cv2.imwrite(str(folder / f"f{i:03d}.png"), frame)
    out = tmp_path / "pt.json"
    assert tool.main(["measure", str(folder), "--out", str(out), "--operator", "A", "--warmup", "2",
                      "--status", "synthetic"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n"] == 6 and report["median_s"] > 0 and report["cv2_version"] == cv2.__version__
    assert report["frames_with_marker"] == 6  # counted over the timed frames only
    assert tool.check_report(out) == []
