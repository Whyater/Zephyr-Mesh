"""Read ESP-NOW echo logs into trace dictionaries.

This is a proposal reader for the plain-text format described in
``docs/ESPNOW_ECHO_LOG.md``. It never repairs or skips lines: the first error
stops parsing and reports the file and the line it came from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim.trace import EspNowTrace

FORMAT_LINE = "# zephyr-espnow-echo-log-1"


class EchoLogError(ValueError):
    """A malformed echo log, with a file label and one-based line number."""


@dataclass(frozen=True)
class SenderLog:
    header: dict[str, str]
    rows: tuple[tuple[int, int, int | None], ...]
    raw_sha256: str
    label: str = "sender"
    header_lines: tuple[tuple[str, int], ...] = ()
    row_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class EchoerLog:
    header: dict[str, str]
    rows: tuple[tuple[int, int, int | None, int | None], ...]
    raw_sha256: str
    label: str = "echoer"
    header_lines: tuple[tuple[str, int], ...] = ()
    row_lines: tuple[int, ...] = ()


_HEADER_KEYS = {"role", "board", "peer", "session", "firmware", "payload_bytes", "channel", "rate_hz", "phy", "note"}
# esp_timer_get_time() is a signed 64-bit count, so 19 digits is the most a
# real stamp can have. The cap also stops Python from choking on huge numbers.
_MAX_DIGITS = 19
_DIGITS = re.compile(rf"[0-9]{{1,{_MAX_DIGITS}}}")
_RSSI = re.compile(r"0|-[1-9][0-9]{0,2}")
_RATE = re.compile(r"[0-9]{1,9}(\.[0-9]{1,9})?")
_SENDER_COLS = "seq,tx_us,echo_us"
_ECHOER_COLS = "seq,rx_us,echo_tx_us,rssi_dbm"


def _error(label: str, line: int, message: str) -> EchoLogError:
    return EchoLogError(f"{label}:{line}: {message}")


def _lines(text: str, label: str) -> list[tuple[int, str]]:
    """Split on ``\\n`` only, so every other character stays inside its line.

    ``str.splitlines`` would also break on characters such as ``\\x0b`` or
    ``\\u2028`` and report the wrong line. A missing final newline means the
    capture was probably cut off mid-row, so that is an error too.
    """
    if text == "":
        raise _error(label, 1, "empty file")
    if text.startswith("\ufeff"):
        raise _error(label, 1, "UTF-8 byte order mark is not allowed")
    if not text.endswith("\n"):
        raise _error(label, text.count("\n") + 1, "last line has no newline (the capture may be cut off)")
    result: list[tuple[int, str]] = []
    for number, content in enumerate(text.split("\n")[:-1], 1):
        content = content.removesuffix("\r")
        if "\t" in content:
            raise _error(label, number, "tab characters are not allowed")
        if content == "":
            raise _error(label, number, "blank lines are not allowed")
        result.append((number, content))
    return result


def _parse_header(text: str, label: str, role: str) -> tuple[dict[str, str], dict[str, int], list[tuple[int, str]]]:
    """Return the header, the line of each header key, and the data lines."""
    lines = _lines(text, label)
    if lines[0][1] != FORMAT_LINE:
        raise _error(label, lines[0][0], f"first line must be {FORMAT_LINE}")
    header: dict[str, str] = {}
    key_lines: dict[str, int] = {}
    index = 1
    while index < len(lines) and lines[index][1].startswith("# "):
        number, content = lines[index]
        body = content[2:]
        if body.count("=") != 1:
            raise _error(label, number, "header line must be # key=value")
        key, value = body.split("=", 1)
        if not key or not value or key.strip() != key or value.strip() != value:
            raise _error(label, number, "header line must have no spaces around = and a non-empty value")
        if not value.isprintable():
            raise _error(label, number, "header value contains a control character")
        if key in header:
            raise _error(label, number, f"duplicate header key {key}")
        if key not in _HEADER_KEYS:
            raise _error(label, number, f"unknown header key {key}")
        header[key] = value
        key_lines[key] = number
        index += 1
    if index >= len(lines):
        raise _error(label, lines[-1][0] + 1, "missing column header")
    column_line, columns = lines[index]
    required = {"role", "board", "peer", "session"} | ({"firmware"} if role == "sender" else set())
    missing = sorted(required - set(header))
    if missing:
        raise _error(label, column_line, f"missing required header key(s): {', '.join(missing)}")
    if header["role"] != role:
        raise _error(label, key_lines["role"], f"role must be {role}")
    if header["board"] == header["peer"]:
        raise _error(label, key_lines["peer"], "peer must differ from board")
    if "payload_bytes" in header and not _DIGITS.fullmatch(header["payload_bytes"]):
        raise _error(label, key_lines["payload_bytes"], "payload_bytes must be a nonnegative integer")
    if "channel" in header and (not _DIGITS.fullmatch(header["channel"]) or not 1 <= int(header["channel"]) <= 14):
        raise _error(label, key_lines["channel"], "channel must be an integer from 1 to 14")
    if "rate_hz" in header and (not _RATE.fullmatch(header["rate_hz"]) or float(header["rate_hz"]) <= 0):
        raise _error(label, key_lines["rate_hz"], "rate_hz must be a positive decimal number such as 50 or 12.5")
    expected = _SENDER_COLS if role == "sender" else _ECHOER_COLS
    if columns != expected:
        raise _error(label, column_line, f"column header must be exactly {expected}")
    data = lines[index + 1:]
    if not data:
        raise _error(label, column_line + 1, "no packets after the column header")
    return header, key_lines, data


def _number(value: str, label: str, line: int, field: str) -> int:
    if not _DIGITS.fullmatch(value):
        raise _error(label, line, f"{field} must be 1 to {_MAX_DIGITS} plain digits")
    return int(value)


def parse_sender_log(text: str, label: str = "sender") -> SenderLog:
    header, key_lines, rows = _parse_header(text, label, "sender")
    parsed: list[tuple[int, int, int | None]] = []
    previous_seq: int | None = None
    previous_tx: int | None = None
    for line, content in rows:
        fields = content.split(",")
        if len(fields) != 3:
            raise _error(label, line, f"expected 3 fields, got {len(fields)}")
        seq = _number(fields[0], label, line, "seq")
        tx = _number(fields[1], label, line, "tx_us")
        echo = None if fields[2] == "" else _number(fields[2], label, line, "echo_us")
        if previous_seq is not None and seq != previous_seq + 1:
            raise _error(label, line, f"seq jumps from {previous_seq} to {seq}; it must go up by exactly 1 (lost serial lines?)")
        if previous_tx is not None and tx <= previous_tx:
            raise _error(label, line, "tx_us must always increase")
        if echo is not None and echo < tx:
            raise _error(label, line, "echo_us cannot precede tx_us")
        previous_seq, previous_tx = seq, tx
        parsed.append((seq, tx, echo))
    return SenderLog(header, tuple(parsed), hashlib.sha256(text.encode("utf-8")).hexdigest(), label,
                     tuple(key_lines.items()), tuple(line for line, _ in rows))


def parse_echoer_log(text: str, label: str = "echoer") -> EchoerLog:
    header, key_lines, rows = _parse_header(text, label, "echoer")
    parsed: list[tuple[int, int, int | None, int | None]] = []
    previous_rx: int | None = None
    seen: set[int] = set()
    for line, content in rows:
        fields = content.split(",")
        if len(fields) != 4:
            raise _error(label, line, f"expected 4 fields, got {len(fields)}")
        seq = _number(fields[0], label, line, "seq")
        rx = _number(fields[1], label, line, "rx_us")
        echo_tx = None if fields[2] == "" else _number(fields[2], label, line, "echo_tx_us")
        rssi = None
        if fields[3] != "":
            if not _RSSI.fullmatch(fields[3]) or not -128 <= int(fields[3]) <= 0:
                raise _error(label, line, "rssi_dbm must be an integer from -128 to 0")
            rssi = int(fields[3])
        if previous_rx is not None and rx < previous_rx:
            raise _error(label, line, "rx_us must never decrease")
        if echo_tx is not None and echo_tx < rx:
            raise _error(label, line, "echo_tx_us cannot precede rx_us")
        if seq in seen and echo_tx is not None:
            raise _error(label, line, "a repeated arrival must have an empty echo_tx_us (only the first arrival is echoed)")
        seen.add(seq)
        previous_rx = rx
        parsed.append((seq, rx, echo_tx, rssi))
    return EchoerLog(header, tuple(parsed), hashlib.sha256(text.encode("utf-8")).hexdigest(), label,
                     tuple(key_lines.items()), tuple(line for line, _ in rows))


def _row_line(log: SenderLog | EchoerLog, index: int) -> int:
    return log.row_lines[index] if log.row_lines else index + 1


def _key_line(log: SenderLog | EchoerLog, key: str) -> int:
    return dict(log.header_lines).get(key, 1)


def logs_to_trace_dict(sender: SenderLog, echoer: EchoerLog | None, *, trace_id: str, status: str,
                       source: str | None = None, provenance: dict | None = None, notes: str | None = None) -> dict:
    """Cross-check the two logs and build a ``sender_round_trip`` trace dict.

    Each packet's echoer data comes from its FIRST arrival; later arrivals of
    the same ``seq`` become extra records marked ``duplicate``.
    """
    sh = sender.header
    first: dict[int, tuple[int, int | None, int | None]] = {}
    repeats: dict[int, list[tuple[int, int | None]]] = {}
    out_of_order: set[int] = set()
    if echoer is not None:
        eh = echoer.header
        if eh["peer"] != sh["board"]:
            raise _error(echoer.label, _key_line(echoer, "peer"), f"peer={eh['peer']} but the sender's board is {sh['board']}")
        if eh["board"] != sh["peer"]:
            raise _error(echoer.label, _key_line(echoer, "board"), f"board={eh['board']} but the sender's peer is {sh['peer']}")
        sender_seqs = {row[0] for row in sender.rows}
        highest: int | None = None
        for i, (seq, rx, echo_tx, rssi) in enumerate(echoer.rows):
            if seq not in sender_seqs:
                raise _error(echoer.label, _row_line(echoer, i), f"seq {seq} does not appear in the sender file")
            if seq in first:
                repeats.setdefault(seq, []).append((rx, rssi))
                continue
            first[seq] = (rx, echo_tx, rssi)
            if highest is not None and seq < highest:
                out_of_order.add(seq)
            highest = seq if highest is None else max(highest, seq)
    low, high = (min(first), max(first)) if first else (None, None)
    packets: list[dict[str, Any]] = []
    for i, (seq, tx, echo) in enumerate(sender.rows):
        base = {"packet_id": f"{sh['session']}:{seq}", "sender_id": sh["board"], "receiver_id": sh["peer"],
                "sender_session_id": sh["session"], "seq": seq, "send_time_s": tx / 1e6}
        arrival = first.get(seq)
        if echo is not None:
            if echoer is not None:
                if arrival is None or arrival[1] is None:
                    raise _error(sender.label, _row_line(sender, i),
                                 f"echo logged for seq {seq} but the echoer log shows no echo for it (a serial log is incomplete)")
                hold = arrival[1] - arrival[0]
                if hold > echo - tx:
                    raise _error(sender.label, _row_line(sender, i),
                                 f"echoer held seq {seq} for {hold} us, longer than the sender's whole round trip of {echo - tx} us")
            packet = {**base, "outcome": "received", "echo_time_s": echo / 1e6}
            if arrival is not None:
                packet["receive_time_s"] = arrival[0] / 1e6
                packet["rssi_dbm"] = arrival[2]
                packet["echo_turnaround_s"] = (arrival[1] - arrival[0]) / 1e6
        elif arrival is not None:
            packet = {**base, "outcome": "received", "receive_time_s": arrival[0] / 1e6, "rssi_dbm": arrival[2]}
        elif echoer is None:
            packet = {**base, "outcome": "unknown", "unknown_reason": "no_echoer_log"}
        elif seq < low:
            packet = {**base, "outcome": "unknown", "unknown_reason": "before_echoer_range"}
        elif seq > high:
            packet = {**base, "outcome": "unknown", "unknown_reason": "after_echoer_range"}
        else:
            packet = {**base, "outcome": "lost", "loss_reason": "not_in_echoer_log"}
        if seq in out_of_order:
            packet["out_of_order"] = True
        if "payload_bytes" in sh:
            packet["payload_bytes"] = int(sh["payload_bytes"])
        packets.append(packet)
        for k, (rx, rssi) in enumerate(repeats.get(seq, []), 1):
            packets.append({**base, "packet_id": f"{sh['session']}:{seq}:dup{k}", "outcome": "received",
                            "receive_time_s": rx / 1e6, "duplicate": True, "rssi_dbm": rssi})
    capture = {"transport": "ESP-NOW", "device_ids": [sh["board"], sh["peer"]], "firmware": sh["firmware"],
               "channel": int(sh["channel"]) if "channel" in sh else None, "phy": sh.get("phy", "unrecorded")}
    raw: dict[str, Any] = {"schema": "zephyr-espnow-trace-1", "trace_id": trace_id, "status": status,
                           "clock_domain": "sender_round_trip", "capture": capture, "packets": packets}
    for key, value in (("source", source), ("provenance", provenance), ("notes", notes)):
        if value is not None:
            raw[key] = value
    return raw


def _read_logs(sender_path: str | Path, echoer_path: str | Path | None) -> tuple[SenderLog, EchoerLog | None]:
    with Path(sender_path).open(encoding="utf-8", newline="") as handle:
        sender = parse_sender_log(handle.read(), Path(sender_path).name)
    echoer = None
    if echoer_path is not None:
        with Path(echoer_path).open(encoding="utf-8", newline="") as handle:
            echoer = parse_echoer_log(handle.read(), Path(echoer_path).name)
    return sender, echoer


def read_echo_logs(sender_path: str | Path, echoer_path: str | Path | None = None, *, trace_id: str, status: str,
                   source: str | None = None, provenance: dict | None = None, notes: str | None = None) -> EspNowTrace:
    """Read, cross-check, and validate one echo capture (echoer log optional)."""
    sender, echoer = _read_logs(sender_path, echoer_path)
    raw = logs_to_trace_dict(sender, echoer, trace_id=trace_id, status=status, source=source,
                             provenance=provenance, notes=notes)
    return EspNowTrace.from_dict(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert ESP-NOW echo logs to a zephyr-espnow-trace-1 JSON file.")
    parser.add_argument("--sender", required=True)
    parser.add_argument("--echoer")
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--status", required=True, choices=["measured", "synthetic", "fixture"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--source")
    parser.add_argument("--notes")
    parser.add_argument("--operator")
    parser.add_argument("--captured-at")
    parser.add_argument("--code-revision")
    parser.add_argument("--instrument-model")
    args = parser.parse_args(argv)
    try:
        sender, echoer = _read_logs(args.sender, args.echoer)
        provenance = None
        if args.status == "measured":
            required = ["operator", "captured_at", "code_revision", "instrument_model", "source"]
            missing = [name for name in required if not getattr(args, name)]
            if missing:
                raise ValueError("measured status requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
            provenance = {
                "instrument": {"kind": "esp32-echo-pair", "model": args.instrument_model, "settings": {
                    "sender_log_sha256": sender.raw_sha256,
                    "echoer_log_sha256": echoer.raw_sha256 if echoer else None,
                    "sender_header": sender.header,
                    "echoer_header": echoer.header if echoer else None}},
                "calibration_ref": None,
                "calibration_ref_null_reason": "echo timing has no external calibration; one-way delay is rtt/2 under a symmetry assumption",
                "raw_sha256": sender.raw_sha256,
                "code_revision": args.code_revision,
                "captured_at": args.captured_at,
                "operator": args.operator,
                "ground_truth": None,
                "ground_truth_null_reason": "no independent clock reference; see docs/ESPNOW_ECHO_LOG.md",
            }
        raw = logs_to_trace_dict(sender, echoer, trace_id=args.trace_id, status=args.status, source=args.source,
                                 provenance=provenance, notes=args.notes)
        trace = EspNowTrace.from_dict(raw)
        Path(args.out).write_text(json.dumps(trace.as_dict(), indent=2, allow_nan=False) + "\n",
                                  encoding="utf-8", newline="\n")
        return 0
    except (ValueError, OSError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
