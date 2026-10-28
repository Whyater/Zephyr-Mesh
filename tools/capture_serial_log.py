#!/usr/bin/env python3
r"""Record the raw bytes coming out of a USB serial port into a file.

This replaces `stty ... raw; cat /dev/cu.usbmodemXXXX > file`, which works but
keeps no record of the port, the baud rate, the times or a checksum. This tool
writes the same raw bytes plus a small sidecar JSON next to them.

It uses only the Python standard library (termios, os, select), because
pyserial is not installed in the repo's environment.

What it does NOT do
-------------------
- It never sends a byte to the device. The port is opened read-only.
- It never flashes a board and runs no device commands.
- It never overwrites an earlier capture: if the output file exists, it stops.
- Tests never open a real port (they use a pseudo-terminal pair).

Usage (run from the repo folder)
--------------------------------
Find the port, then record until Ctrl-C or for a fixed time:

    ls /dev/cu.usbmodem*
    python3 tools/capture_serial_log.py /dev/cu.usbmodemXXXX \
        --out "$HOME/zephyr-data/radio/sender_raw.txt" --seconds 120

Output
------
    sender_raw.txt        the raw bytes, exactly as they arrived
    sender_raw.txt.json   sidecar: port, baud, start and stop time (UTC),
                          duration, byte count, sha256, why it stopped

Opening a board's USB serial port can reset it on some boards, so the boot
banner may appear at the start of the file. That is real device output and is
kept.

Exit codes
----------
    0  capture finished (time ran out or Ctrl-C)
    2  refused or failed to start (bad port, unsupported baud, file exists)
    3  --extract was asked for but the extractor is not available
    4  the device went away during the capture (the bytes so far are kept)
    5  writing the output file failed (for example the disk is full)

If several apply, the first one found is reported (a missing extractor hides 4).

Optional --extract hook
-----------------------
With --extract the tool calls tools.extract_echo_log.main([raw, "--out", clean])
after the capture, if that module exists. The extractor is a separate tool that
is not part of this change, and its command line is not confirmed: this hook
has only been tested against a stand-in module. If the module is missing the
raw capture and sidecar are still kept and the tool exits with code 3.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import select
import signal
import sys
import termios
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "zephyr-serial-capture-1"
READ_SIZE = 4096
POLL_S = 0.1


class CaptureError(Exception):
    """The capture could not start (bad port, unsupported baud)."""


def _speed_constant(baud: int) -> int:
    const = getattr(termios, f"B{baud}", None)
    if const is None:
        raise CaptureError(
            f"baud {baud} has no termios constant on this system; "
            "use a standard rate such as 115200"
        )
    return const


def _make_raw(attrs: list, speed: int) -> list:
    """Return termios attributes for a byte-transparent 8N1 receive-only line.

    Every flag that could change, drop or generate bytes is cleared by name,
    so the result does not depend on which flags the Python version's own
    raw-mode helper happens to touch. In particular software flow control
    (IXON, IXOFF) is off, because IXOFF lets the kernel send XOFF bytes
    to the device.
    """
    iflag, oflag, cflag, lflag = attrs[0], attrs[1], attrs[2], attrs[3]
    iflag &= ~(
        termios.IGNBRK | termios.BRKINT | termios.PARMRK | termios.ISTRIP
        | termios.INLCR | termios.IGNCR | termios.ICRNL | termios.IXON
        | termios.IXOFF | termios.IXANY | termios.INPCK | termios.IGNPAR
    )
    oflag &= ~termios.OPOST
    cflag &= ~(termios.CSIZE | termios.PARENB | termios.PARODD | termios.CSTOPB | getattr(termios, "CRTSCTS", 0))
    cflag |= termios.CS8 | termios.CLOCAL | termios.CREAD
    cflag &= ~termios.HUPCL  # do not drop the modem lines when we close
    lflag &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG | termios.IEXTEN)
    cc = list(attrs[6])
    cc[termios.VMIN] = 1
    cc[termios.VTIME] = 0
    return [iflag, oflag, cflag, lflag, speed, speed, cc]


def open_port(port: str, baud: int) -> int:
    """Open `port` read-only in raw mode at `baud`. Returns the file descriptor.

    Raw mode turns off every byte translation (CR/LF mapping, echo, signal and
    flow-control characters), so the file holds exactly what the device sent.
    Bytes that arrive in the instant between the open and the mode change are
    handled under the old settings, so start recording before you reset a board.
    """
    if baud <= 0:
        raise CaptureError(f"baud must be positive, got {baud}")
    speed = _speed_constant(baud)
    try:
        fd = os.open(port, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError as exc:
        raise CaptureError(f"cannot open {port}: {exc.strerror or exc}") from exc
    try:
        if not os.isatty(fd):
            raise CaptureError(f"{port} is not a serial device (not a terminal)")
        termios.tcsetattr(fd, termios.TCSANOW, _make_raw(termios.tcgetattr(fd), speed))
    except termios.error as exc:
        os.close(fd)
        raise CaptureError(f"cannot configure {port}: {exc}") from exc
    except CaptureError:
        os.close(fd)
        raise
    return fd


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_all(out_fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(out_fd, view) :]


def _drain(fd, out_fd, digest):
    total = 0
    while select.select([fd], [], [], 0)[0]:
        try:
            data = os.read(fd, READ_SIZE)
        except OSError:
            break
        if not data:
            break
        _write_all(out_fd, data)
        digest.update(data)
        total += len(data)
    return total


def _read_loop(fd, out_fd, digest, deadline, stop_event):
    """Copy bytes from the port to the file. Returns (byte_count, stop_reason)."""
    total = 0
    reason = "interrupt"
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                reason = "interrupt"
                break
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                reason = "seconds"
                break
            wait = POLL_S if remaining is None else min(POLL_S, remaining)
            if not select.select([fd], [], [], wait)[0]:
                continue
            try:
                data = os.read(fd, READ_SIZE)
            except BlockingIOError:
                continue
            except OSError:
                reason = "device-closed"
                break
            if not data:
                reason = "device-closed"
                break
            _write_all(out_fd, data)
            digest.update(data)
            total += len(data)
        if reason != "device-closed":
            # Bytes that arrived just before the stop are still in the buffer.
            total += _drain(fd, out_fd, digest)
    except KeyboardInterrupt:
        reason = "interrupt"
    except OSError:
        reason = "write-error"  # for example the disk is full
    return total, reason


def capture(port, baud, out_path, seconds=None, stop_event=None):
    """Record `port` to `out_path` and write `<out_path>.json`. Returns the sidecar dict.

    Stops after `seconds`, when `stop_event` is set, on Ctrl-C, or when the
    device goes away. Raises CaptureError if the port cannot be used and
    FileExistsError if `out_path` or its sidecar already exists. The output
    files are created before the port is opened, so a refused run never
    touches the device.
    """
    out_path = Path(out_path)
    if seconds is not None and not (math.isfinite(seconds) and seconds > 0):
        raise CaptureError(f"--seconds must be a positive number, got {seconds}")
    if baud <= 0:
        raise CaptureError(f"baud must be positive, got {baud}")
    _speed_constant(baud)
    if sidecar_path(out_path).exists():
        raise FileExistsError(f"{sidecar_path(out_path)} already exists")
    out_fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        try:
            fd = open_port(port, baud)
        except CaptureError:
            os.close(out_fd)
            out_fd = None
            out_path.unlink()  # we created it a moment ago and it is empty
            raise
        try:
            digest = hashlib.sha256()
            started = _now_utc()
            t0 = time.monotonic()
            deadline = None if seconds is None else t0 + float(seconds)
            total, reason = _read_loop(fd, out_fd, digest, deadline, stop_event)
            duration = time.monotonic() - t0
            stopped = _now_utc()
        finally:
            os.close(fd)
    finally:
        if out_fd is not None:
            os.close(out_fd)
    if reason == "write-error":
        # A failed write can leave the file longer than the running hash knows.
        data = out_path.read_bytes()
        total, digest = len(data), hashlib.sha256(data)
    sidecar = {
        "schema": SCHEMA,
        "port": str(port),
        "baud": int(baud),
        "raw_file": out_path.name,
        "started_utc": started,
        "stopped_utc": stopped,
        "duration_s": round(duration, 3),
        "bytes": total,
        "sha256": digest.hexdigest(),
        "stop_reason": reason,
    }
    write_sidecar(out_path, sidecar, exclusive=True)
    return sidecar


def sidecar_path(out_path) -> Path:
    return Path(str(out_path) + ".json")


def write_sidecar(out_path, sidecar: dict, exclusive: bool = False) -> None:
    text = json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(sidecar_path(out_path), flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def run_extract(raw_path, clean_path):
    """Optional hook: run the log extractor if it exists. Returns its exit code.

    Raises ImportError if tools.extract_echo_log is not available.
    The extractor's command line is not confirmed (see the module docstring).
    """
    # Run as `python3 tools/capture_serial_log.py`, sys.path starts at tools/,
    # so the repo root is added to let `tools.extract_echo_log` be found.
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.append(root)
    try:
        module = importlib.import_module("tools.extract_echo_log")
    except ModuleNotFoundError as exc:
        if exc.name in ("tools", "tools.extract_echo_log"):
            raise ImportError("tools.extract_echo_log is not available") from exc
        raise  # the extractor exists but one of its own imports is missing
    return int(module.main([str(raw_path), "--out", str(clean_path)]) or 0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record raw serial bytes to a file with a metadata sidecar (read-only, stdlib only)."
    )
    p.add_argument("port", help="serial device, for example /dev/cu.usbmodemXXXX")
    p.add_argument("--baud", type=int, default=115200, help="baud rate (default 115200)")
    p.add_argument("--out", required=True, help="raw output file (must not exist yet)")
    p.add_argument("--seconds", type=float, default=None, help="stop after this many seconds (default: until Ctrl-C)")
    p.add_argument("--extract", action="store_true", help="after capture, run tools.extract_echo_log if it exists")
    p.add_argument("--extract-out", default=None, help="clean-log path for --extract (default: <out>.clean)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Treat a plain `kill` like Ctrl-C so the sidecar is still written.
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        side = capture(args.port, args.baud, args.out, seconds=args.seconds)
    except (CaptureError, FileExistsError) as exc:
        print(f"capture_serial_log: {exc}", file=sys.stderr)
        return 2
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    print(
        f"captured {side['bytes']} bytes in {side['duration_s']} s "
        f"({side['stop_reason']}) -> {args.out}\n  sha256 {side['sha256']}"
    )
    rc = 0
    if side["stop_reason"] == "device-closed":
        print("capture_serial_log: the device went away; bytes so far are kept", file=sys.stderr)
        rc = 4
    elif side["stop_reason"] == "write-error":
        print("capture_serial_log: writing the file failed (disk full?); bytes so far are kept", file=sys.stderr)
        rc = 5
    if args.extract:
        clean = args.extract_out or str(args.out) + ".clean"
        try:
            code = run_extract(args.out, clean)
        except ImportError:
            print(
                "capture_serial_log: tools.extract_echo_log is not available; "
                "the raw capture and sidecar are kept",
                file=sys.stderr,
            )
            return 3
        side["extract"] = {
            "module": "tools.extract_echo_log",
            "exit_code": code,
            "output": Path(clean).name,
        }
        write_sidecar(args.out, side)
        if code != 0:
            return code
    return rc


if __name__ == "__main__":
    sys.exit(main())
