"""Tests for the latency command-line tool and the capture loop (no OpenCV, no camera).

The capture loop test uses a fake screen and a fake camera running in real
time: the fake camera returns whatever the fake screen showed a known time ago.
The recording then goes through the analysis, which must find that delay.
"""
import bisect
import json
import threading
import time

import numpy as np
import pytest

from sim.vision_latency import Settings, analyze, pattern_strip_box, read_display_log
from tools import vision_latency as cli
from tools import vision_latency_run as runner


def test_synth_then_analyze_round_trip(tmp_path, capsys):
    run_dir = tmp_path / "run"
    assert cli.main(["synth", str(run_dir), "--latency-ms", "90", "--display-lag-ms", "20",
                     "--duration-s", "12", "--seed", "4"]) == 0
    assert cli.main(["analyze", str(run_dir), "--display-lag-ms", "0,40"]) == 0
    result = json.loads((run_dir / "latency.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert result["status"] == "synthetic"
    a = result["apparent_latency_s"]
    truth = meta["injected"]["true_mean_latency_ms"] / 1000 + 0.020
    assert abs(a["mean"] - truth) <= a["stated_error"]
    assert result["camera_latency_s"]["low"] <= truth - 0.020 <= result["camera_latency_s"]["high"]
    assert set(result["inputs"]) == {"frames.npz", "display_log.csv", "run_meta.json"}
    samples = (run_dir / "latency.samples.csv").read_text().splitlines()
    assert samples[0] == "frame,t_read_s,seq,straddle,lo_s,hi_s,mid_s"
    assert len(samples) - 1 == result["frames_used"]
    out = capsys.readouterr().out
    assert "upper bound on camera latency" in out


def test_synth_and_analyze_with_a_refreshing_screen(tmp_path, capsys):
    run_dir = tmp_path / "run"
    assert cli.main(["synth", str(run_dir), "--latency-ms", "90", "--vsync-hz", "60",
                     "--duration-s", "12", "--seed", "4"]) == 0
    assert cli.main(["analyze", str(run_dir), "--display-lag-ms", "0,20", "--refresh-hz", "60"]) == 0
    result = json.loads((run_dir / "latency.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["injected"]["vsync_hz"] == 60.0
    assert result["settings"]["display_refresh_hz"] == 60.0
    assert result["apparent_latency_s"]["stated_error_parts"]["refresh_phase"] == pytest.approx(1 / 120)
    # The refresh delay (0 to 16.7 ms) is display lag, inside the assumed 0 to 20 ms.
    camera = meta["injected"]["true_mean_latency_ms"] / 1000
    assert result["camera_latency_s"]["low"] <= camera <= result["camera_latency_s"]["high"]
    assert "60 Hz screen refresh" in capsys.readouterr().out


def test_preview_writes_a_png(tmp_path):
    run_dir = tmp_path / "run"
    cli.main(["synth", str(run_dir), "--duration-s", "12"])
    assert cli.main(["preview", str(run_dir)]) == 0
    assert (run_dir / "preview.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_analyze_reports_errors_instead_of_crashing(tmp_path, capsys):
    assert cli.main(["analyze", str(tmp_path / "missing")]) == 2
    assert "error" in capsys.readouterr().err


class FakeScreen:
    """Remembers what was shown and when, on the real monotonic clock."""

    def __init__(self):
        self.times: list[float] = []
        self.images: list[np.ndarray] = []
        self.lock = threading.Lock()

    def show(self, image):
        with self.lock:
            self.times.append(time.monotonic())
            self.images.append(image)

    def at(self, t):
        with self.lock:
            i = bisect.bisect_right(self.times, t) - 1
            return self.images[i] if i >= 0 else None


class FakeCamera:
    """Delivers a frame every 1/fps s. Each frame shows the screen as it was latency_s ago."""

    def __init__(self, screen, fps, latency_s):
        self.screen, self.period, self.latency = screen, 1.0 / fps, latency_s
        self.next = time.monotonic() + self.period

    def read(self):
        now = time.monotonic()
        if self.next > now:
            time.sleep(self.next - now)
        self.next += self.period
        image = self.screen.at(time.monotonic() - self.latency)
        return None if image is None else image.copy()


def test_capture_loop_with_fake_camera_recovers_the_delay(tmp_path):
    screen = FakeScreen()
    camera = FakeCamera(screen, fps=30.0, latency_s=0.150)
    rec = runner.record(camera.read, lambda f: f[::2, ::2], screen.show, lambda: False,
                        n_bits=6, seconds=4.0, hold_s=(0.040, 0.070), screen_size=(320, 180), seed=7)
    assert rec.camera_error == ""
    # t_shown is logged before the draw, so the screen never changes before the logged time.
    shown = np.array([d[2] for d in rec.display])
    assert np.all(np.array(screen.times[:len(shown)]) >= shown)

    runner.save(rec, tmp_path, {"status": "synthetic", "n_bits": 6})
    data = np.load(tmp_path / "frames.npz")
    log = read_display_log(tmp_path / "display_log.csv", 6)
    x0, y0, x1, y1 = pattern_strip_box(320, 180)
    roi = (x0 // 2, y0 // 2, x1 // 2, y1 // 2)
    result = analyze(data["frames"], data["t_read_s"], log,
                     Settings(n_bits=6, roi=roi, display_lag_s=(0.0, 0.0), max_latency_s=1.0, bootstrap=500))
    a = result["apparent_latency_s"]
    assert result["frames_used"] >= 0.8 * result["frames_total"]
    # The fake camera's frame is the screen at (read time - 0.150 s), give or take
    # thread scheduling. The fake shutter is instant, so the apparent latency is
    # 0.150 s plus a few ms of scheduling noise; allow 10 ms for a busy machine.
    assert abs(a["mean"] - 0.150) <= max(a["stated_error"], 0.010)
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert 25 <= meta["measured_fps"] <= 35
    # The draw-call timing column is read back and summarised.
    assert result["draw_call_ms"] is not None and result["draw_call_ms"]["median"] < 50


def test_runner_refuses_an_existing_folder_and_too_few_bits(tmp_path, capsys):
    with pytest.raises(SystemExit):
        runner.main(["--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        runner.main(["--out", str(tmp_path / "new"), "--bits", "6"])
