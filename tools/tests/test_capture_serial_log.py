"""Tests for tools/capture_serial_log.py.

No real port is ever opened. A pseudo-terminal pair from os.openpty() stands in
for the device: the test writes into the master end, and the tool reads the
slave end exactly as it would read /dev/cu.usbmodem*.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX serial only (no termios or openpty)")
termios = pytest.importorskip("termios")

from tools import capture_serial_log as csl  # noqa: E402

BANNER = b"ets Jun  8 2016 00:22:57\r\n\r\nrst:0x1 (POWERON_RESET),boot:0x13\r\n"
# Every byte value, so any line-discipline change (CR/LF mapping, 0x03 signal,
# 0x04 EOF, 0x11/0x13 flow control) shows up as a mismatch.
ALL_BYTES = bytes(range(256))
LINES = b"".join(b"seq,%d,t_us,%d\r\n" % (i, 1000 * i) for i in range(200))
FIXTURE = BANNER + ALL_BYTES + LINES


def _wait_raw(fd: int, timeout: float = 5.0) -> None:
    """Wait until the terminal behind fd is in raw mode (the tool has set it)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        lflag = termios.tcgetattr(fd)[3]
        if not (lflag & termios.ICANON) and not (lflag & termios.ECHO):
            return
        time.sleep(0.01)
    raise AssertionError("tool never switched the port to raw mode")


def _feed(master: int, data: bytes, chunk: int = 97) -> None:
    for i in range(0, len(data), chunk):
        os.write(master, data[i : i + chunk])
        time.sleep(0.001)


@pytest.fixture()
def pty_pair():
    master, slave = os.openpty()
    name = os.ttyname(slave)
    yield master, slave, name
    for fd in (master, slave):
        try:
            os.close(fd)
        except OSError:
            pass


def test_capture_equals_fed_bytes_and_sidecar_hash(tmp_path, pty_pair):
    master, slave, name = pty_pair
    out = tmp_path / "sender_raw.txt"
    stop = threading.Event()
    done = threading.Event()

    def writer():
        _wait_raw(slave)
        _feed(master, FIXTURE)
        time.sleep(0.2)
        done.set()
        stop.set()

    t = threading.Thread(target=writer)
    t.start()
    result = csl.capture(name, 115200, out, stop_event=stop, seconds=30)
    stop.set()
    t.join()

    assert done.is_set()
    got = out.read_bytes()
    assert got == FIXTURE
    assert result["bytes"] == len(FIXTURE)
    assert result["sha256"] == hashlib.sha256(FIXTURE).hexdigest()
    assert result["stop_reason"] == "interrupt"

    side = json.loads(Path(str(out) + ".json").read_text())
    assert side["schema"] == "zephyr-serial-capture-1"
    assert side["port"] == name
    assert side["baud"] == 115200
    assert side["bytes"] == len(FIXTURE)
    assert side["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert side["raw_file"] == "sender_raw.txt"
    assert side["started_utc"] <= side["stopped_utc"]
    assert side["duration_s"] >= 0
    # the sidecar must not leak a local home folder
    assert "/Users/" not in json.dumps(side)


def test_seconds_stop_and_empty_capture(tmp_path, pty_pair):
    _, _, name = pty_pair
    out = tmp_path / "quiet.bin"
    t0 = time.monotonic()
    result = csl.capture(name, 115200, out, seconds=0.4)
    assert 0.35 <= time.monotonic() - t0 < 3.0
    assert result["stop_reason"] == "seconds"
    assert result["bytes"] == 0
    assert result["sha256"] == hashlib.sha256(b"").hexdigest()
    assert out.read_bytes() == b""


def test_never_sends_bytes(tmp_path, pty_pair):
    """Whatever the tool does, nothing may appear on the device's input side."""
    master, _, name = pty_pair
    csl.capture(name, 115200, tmp_path / "x.bin", seconds=0.2)
    os.set_blocking(master, False)
    with pytest.raises(BlockingIOError):
        os.read(master, 4096)


def test_main_runs_end_to_end_with_seconds(tmp_path, pty_pair, capsys):
    master, slave, name = pty_pair
    out = tmp_path / "echoer_raw.txt"

    def writer():
        _wait_raw(slave)
        _feed(master, FIXTURE)

    t = threading.Thread(target=writer)
    t.start()
    rc = csl.main([name, "--baud", "115200", "--out", str(out), "--seconds", "1.5"])
    t.join()
    assert rc == 0
    assert out.read_bytes() == FIXTURE
    assert json.loads(Path(str(out) + ".json").read_text())["stop_reason"] == "seconds"
    assert str(len(FIXTURE)) in capsys.readouterr().out


def test_device_closed_is_recorded_not_crashed(tmp_path):
    master, slave = os.openpty()
    name = os.ttyname(slave)
    out = tmp_path / "gone.bin"

    def writer():
        _wait_raw(slave)
        os.write(master, b"partial line")
        time.sleep(0.2)
        os.close(master)

    t = threading.Thread(target=writer)
    t.start()
    result = csl.capture(name, 115200, out, seconds=5)
    t.join()
    os.close(slave)
    assert out.read_bytes() == b"partial line"
    assert result["stop_reason"] == "device-closed"
    assert result["bytes"] == len(b"partial line")


def test_refuses_to_overwrite(tmp_path, pty_pair):
    _, _, name = pty_pair
    out = tmp_path / "keep.bin"
    out.write_bytes(b"earlier capture")
    with pytest.raises(FileExistsError):
        csl.capture(name, 115200, out, seconds=0.1)
    assert out.read_bytes() == b"earlier capture"
    assert not Path(str(out) + ".json").exists()


def test_refuses_non_tty_and_missing_port(tmp_path):
    plain = tmp_path / "not_a_port.txt"
    plain.write_bytes(b"x")
    with pytest.raises(csl.CaptureError):
        csl.capture(str(plain), 115200, tmp_path / "a.bin", seconds=0.1)
    with pytest.raises(csl.CaptureError):
        csl.capture(str(tmp_path / "missing"), 115200, tmp_path / "b.bin", seconds=0.1)
    assert not (tmp_path / "a.bin").exists()
    assert not (tmp_path / "b.bin").exists()


def test_unsupported_baud_is_refused_clearly(tmp_path, pty_pair):
    _, _, name = pty_pair
    with pytest.raises(csl.CaptureError, match="74880"):
        csl.capture(name, 74880, tmp_path / "c.bin", seconds=0.1)
    assert not (tmp_path / "c.bin").exists()


def test_main_returns_nonzero_on_refusal(tmp_path, capsys):
    rc = csl.main([str(tmp_path / "missing"), "--out", str(tmp_path / "d.bin"), "--seconds", "0.1"])
    assert rc == 2
    assert "missing" in capsys.readouterr().err


def test_bad_seconds_and_baud_are_refused(tmp_path, pty_pair):
    _, _, name = pty_pair
    for bad in (float("nan"), 0, -1):
        with pytest.raises(csl.CaptureError):
            csl.capture(name, 115200, tmp_path / "s.bin", seconds=bad)
    with pytest.raises(csl.CaptureError):
        csl.capture(name, 0, tmp_path / "s.bin", seconds=0.1)
    assert not (tmp_path / "s.bin").exists()


def test_existing_sidecar_is_not_overwritten(tmp_path, pty_pair):
    _, _, name = pty_pair
    side = tmp_path / "g.bin.json"
    side.write_text("earlier sidecar")
    with pytest.raises(FileExistsError):
        csl.capture(name, 115200, tmp_path / "g.bin", seconds=0.1)
    assert side.read_text() == "earlier sidecar"
    assert not (tmp_path / "g.bin").exists()


def test_port_failure_leaves_no_output_file(tmp_path):
    with pytest.raises(csl.CaptureError):
        csl.capture(str(tmp_path / "missing"), 115200, tmp_path / "h.bin", seconds=0.1)
    assert not (tmp_path / "h.bin").exists()


def test_raw_mode_clears_flags_that_alter_or_generate_bytes(tmp_path, pty_pair):
    """Start from a deliberately dirty line setup: the capture must still be exact."""
    master, slave, name = pty_pair
    attrs = termios.tcgetattr(slave)
    attrs[0] |= termios.INLCR | termios.IGNCR | termios.ICRNL | termios.IXON | termios.IXOFF | termios.ISTRIP
    attrs[3] |= termios.ICANON | termios.ECHO | termios.ISIG
    termios.tcsetattr(slave, termios.TCSANOW, attrs)
    out = tmp_path / "dirty.bin"
    stop = threading.Event()

    def writer():
        _wait_raw(slave)
        _feed(master, FIXTURE)
        time.sleep(0.2)
        stop.set()

    t = threading.Thread(target=writer)
    t.start()
    csl.capture(name, 115200, out, stop_event=stop, seconds=30)
    stop.set()
    t.join()
    assert out.read_bytes() == FIXTURE
    after = termios.tcgetattr(slave)
    banned = termios.INLCR | termios.IGNCR | termios.ICRNL | termios.IXON | termios.IXOFF | termios.ISTRIP
    assert after[0] & banned == 0
    assert after[3] & (termios.ICANON | termios.ECHO | termios.ISIG) == 0
    assert after[2] & termios.CSIZE == termios.CS8


def test_bytes_waiting_when_stop_arrives_are_still_kept(tmp_path, pty_pair):
    master, _, name = pty_pair
    out = tmp_path / "drain.bin"
    fd = csl.open_port(name, 115200)
    out_fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        payload = b"x" * 512  # a pty holds only about 1 KB unread
        os.write(master, payload)
        stop = threading.Event()
        stop.set()  # stop is already requested when the loop starts
        digest = hashlib.sha256()
        total, reason = csl._read_loop(fd, out_fd, digest, None, stop)
    finally:
        os.close(fd)
        os.close(out_fd)
    assert reason == "interrupt"
    assert total == len(payload) == len(out.read_bytes())
    assert digest.hexdigest() == hashlib.sha256(payload).hexdigest()


def test_ctrl_c_stops_cleanly_and_writes_sidecar(tmp_path, pty_pair):
    import signal

    master, slave, name = pty_pair
    out = tmp_path / "ctrlc.bin"
    main_thread = threading.get_ident()
    # A shell can start the test run with SIGINT ignored; make Ctrl-C behave normally here.
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)

    def writer():
        _wait_raw(slave)
        os.write(master, b"before interrupt")
        time.sleep(0.3)
        signal.pthread_kill(main_thread, signal.SIGINT)

    t = threading.Thread(target=writer)
    t.start()
    try:
        result = csl.capture(name, 115200, out, seconds=20)
    finally:
        t.join()
        signal.signal(signal.SIGINT, previous)
    assert result["stop_reason"] == "interrupt"
    assert out.read_bytes() == b"before interrupt"
    assert json.loads(Path(str(out) + ".json").read_text())["bytes"] == len(b"before interrupt")


def test_main_exit_code_4_when_device_goes_away(tmp_path):
    master, slave = os.openpty()
    name = os.ttyname(slave)

    def writer():
        _wait_raw(slave)
        os.write(master, b"abc")
        time.sleep(0.2)
        os.close(master)

    t = threading.Thread(target=writer)
    t.start()
    rc = csl.main([name, "--out", str(tmp_path / "i.bin"), "--seconds", "5"])
    t.join()
    os.close(slave)
    assert rc == 4
    assert (tmp_path / "i.bin").read_bytes() == b"abc"


def test_main_restores_sigterm_handler(tmp_path, pty_pair):
    import signal

    _, _, name = pty_pair
    before = signal.getsignal(signal.SIGTERM)
    csl.main([name, "--out", str(tmp_path / "j.bin"), "--seconds", "0.1"])
    assert signal.getsignal(signal.SIGTERM) == before


def test_extract_hook_reports_missing_module(tmp_path, pty_pair, monkeypatch, capsys):
    _, _, name = pty_pair
    monkeypatch.setitem(sys.modules, "tools.extract_echo_log", None)  # forces ImportError
    out = tmp_path / "e.bin"
    rc = csl.main([name, "--out", str(out), "--seconds", "0.2", "--extract"])
    assert rc == 3
    assert out.exists()  # the raw capture is kept
    assert "extract_echo_log" in capsys.readouterr().err


def test_extract_hook_calls_module_when_present(tmp_path, pty_pair, monkeypatch):
    _, _, name = pty_pair
    calls = []
    fake = types.ModuleType("tools.extract_echo_log")
    fake.main = lambda argv: calls.append(list(argv)) or 0
    monkeypatch.setitem(sys.modules, "tools.extract_echo_log", fake)
    out = tmp_path / "f.bin"
    rc = csl.main([name, "--out", str(out), "--seconds", "0.2", "--extract"])
    assert rc == 0
    assert len(calls) == 1
    assert str(out) in calls[0]


def test_missing_out_folder_is_a_clean_start_failure(tmp_path, pty_pair, capsys):
    _, _, name = pty_pair
    rc = csl.main([name, "--out", str(tmp_path / "no-such-dir" / "d.bin"), "--seconds", "0.1"])
    assert rc == 2
    assert "capture_serial_log:" in capsys.readouterr().err


def test_without_termios_main_says_posix_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(csl, "termios", None)
    rc = csl.main(["/dev/null", "--out", str(tmp_path / "d.bin"), "--seconds", "0.1"])
    assert rc == 2
    assert "POSIX" in capsys.readouterr().err
    assert not (tmp_path / "d.bin").exists()
