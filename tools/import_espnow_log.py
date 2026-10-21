"""Read ESP-NOW echo logs into trace dictionaries.

This is a proposal reader for the plain-text format described in
``docs/ESPNOW_ECHO_LOG.md``. It never repairs or skips lines: the first error
stops parsing and reports its source line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim.trace import EspNowTrace


class EchoLogError(ValueError):
    """A malformed echo log, with a file label and one-based line number."""


@dataclass(frozen=True)
class SenderLog:
    header: dict[str, str]
    rows: tuple[tuple[int, int, int | None], ...]
    raw_sha256: str


@dataclass(frozen=True)
class EchoerLog:
    header: dict[str, str]
    rows: tuple[tuple[int, int, int | None, int | None], ...]
    raw_sha256: str


_HEADER_KEYS = {"role", "board", "peer", "session", "firmware", "payload_bytes", "channel", "rate_hz", "phy", "note"}
_DIGITS = re.compile(r"^[0-9]+$")
_RSSI = re.compile(r"^-?[0-9]+$")
_SENDER_COLS = "seq,tx_us,echo_us"
_ECHOER_COLS = "seq,rx_us,echo_tx_us,rssi_dbm"


def _error(label: str, line: int, message: str) -> EchoLogError:
    return EchoLogError(f"{label}:{line}: {message}")


def _lines(text: str, label: str) -> list[tuple[int, str]]:
    if text == "":
        raise _error(label, 1, "empty file")
    if text.startswith("\ufeff"):
        raise _error(label, 1, "UTF-8 BOM is not allowed")
    result: list[tuple[int, str]] = []
    raw_lines = text.splitlines(keepends=True)
    for number, raw in enumerate(raw_lines, 1):
        if raw.endswith("\n"):
            content = raw[:-1]
            content = content.removesuffix("\r")
        else:
            content = raw
        if "\t" in content:
            raise _error(label, number, "tab characters are not allowed")
        if content == "":
            raise _error(label, number, "blank lines are not allowed")
        result.append((number, content))
    return result


def _parse_header(text: str, label: str, role: str) -> tuple[dict[str, str], list[tuple[int, str]]]:
    lines = _lines(text, label)
    if lines[0][1] != "# zephyr-espnow-echo-log-1":
        raise _error(label, lines[0][0], "first line must be # zephyr-espnow-echo-log-1")
    header: dict[str, str] = {}
    index = 1
    while index < len(lines) and lines[index][1].startswith("# "):
        number, content = lines[index]
        body = content[2:]
        if "=" not in body or body.count("=") != 1:
            raise _error(label, number, "header line must be # key=value")
        key, value = body.split("=", 1)
        if not key or not value or key.strip() != key or value.strip() != value:
            raise _error(label, number, "header line must have no spaces around = and a non-empty value")
        if key in header:
            raise _error(label, number, f"duplicate header key {key}")
        if key not in _HEADER_KEYS:
            raise _error(label, number, f"unknown header key {key}")
        header[key] = value
        index += 1
    required = {"role", "board", "peer", "session"}
    if role == "sender":
        required.add("firmware")
    missing = sorted(required - set(header))
    if missing:
        raise _error(label, index + 1, f"missing required header key(s): {', '.join(missing)}")
    if header.get("role") != role:
        line = next((n for n, c in lines[1:index + 1] if c == "# role=" + header.get("role", "")), index)
        raise _error(label, line, f"role must be {role}")
    if "payload_bytes" in header and not _DIGITS.fullmatch(header["payload_bytes"]):
        line = next(n for n, c in lines if c == "# payload_bytes=" + header["payload_bytes"])
        raise _error(label, line, "payload_bytes must be a nonnegative integer")
    if "channel" in header and (not _DIGITS.fullmatch(header["channel"]) or not 1 <= int(header["channel"]) <= 14):
        line = next(n for n, c in lines if c == "# channel=" + header["channel"])
        raise _error(label, line, "channel must be an integer from 1 to 14")
    if "rate_hz" in header:
        try:
            rate = float(header["rate_hz"])
        except ValueError:
            rate = math.nan
        if not math.isfinite(rate) or rate <= 0:
            line = next(n for n, c in lines if c == "# rate_hz=" + header["rate_hz"])
            raise _error(label, line, "rate_hz must be a positive finite number")
    if index >= len(lines):
        raise _error(label, index + 1, "missing column header")
    expected = _SENDER_COLS if role == "sender" else _ECHOER_COLS
    number, columns = lines[index]
    if columns != expected:
        raise _error(label, number, f"column header must be exactly {expected}")
    return header, lines[index + 1 :]


def _number(value: str, label: str, line: int, field: str) -> int:
    if not _DIGITS.fullmatch(value):
        raise _error(label, line, f"{field} must be digits only")
    return int(value)


def parse_sender_log(text: str, label: str = "sender") -> SenderLog:
    header, rows = _parse_header(text, label, "sender")
    parsed: list[tuple[int, int, int | None]] = []
    previous_seq: int | None = None
    previous_tx: int | None = None
    for line, content in rows:
        fields = content.split(",")
        if len(fields) != 3:
            raise _error(label, line, "wrong field count")
        seq = _number(fields[0], label, line, "seq")
        tx = _number(fields[1], label, line, "tx_us")
        echo = None if fields[2] == "" else _number(fields[2], label, line, "echo_us")
        if previous_seq is not None and seq != previous_seq + 1:
            raise _error(label, line, "seq must increase by exactly 1")
        if previous_tx is not None and tx <= previous_tx:
            raise _error(label, line, "tx_us must always increase")
        if echo is not None and echo < tx:
            raise _error(label, line, "echo_us cannot precede tx_us")
        previous_seq, previous_tx = seq, tx
        parsed.append((seq, tx, echo))
    if not parsed:
        raise _error(label, rows[-1][0] + 1 if rows else 1, "no packets")
    return SenderLog(header, tuple(parsed), hashlib.sha256(text.encode("utf-8")).hexdigest())


def parse_echoer_log(text: str, label: str = "echoer") -> EchoerLog:
    header, rows = _parse_header(text, label, "echoer")
    parsed: list[tuple[int, int, int | None, int | None]] = []
    previous_rx: int | None = None
    for line, content in rows:
        fields = content.split(",")
        if len(fields) != 4:
            raise _error(label, line, "wrong field count")
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
        previous_rx = rx
        parsed.append((seq, rx, echo_tx, rssi))
    return EchoerLog(header, tuple(parsed), hashlib.sha256(text.encode("utf-8")).hexdigest())


def logs_to_trace_dict(sender: SenderLog, echoer: EchoerLog | None, *, trace_id: str, status: str, source: str | None = None, provenance: dict | None = None, notes: str | None = None) -> dict:
    sh, eh = sender.header, echoer.header if echoer else None
    if echoer is not None:
        if eh["peer"] != sh["board"] or sh["peer"] != eh["board"]:
            raise _error("echoer", len(echoer.header) + 2, "sender and echoer peer/board headers do not agree")
        sender_seqs = {row[0] for row in sender.rows}
        for i, (seq, *_rest) in enumerate(echoer.rows):
            if seq not in sender_seqs:
                raise _error("echoer", len(echoer.header) + 3 + i, "seq does not appear in sender file")
        echoed_by_echoer = {row[0] for row in echoer.rows if row[2] is not None}
        for i, row in enumerate(sender.rows):
            if row[2] is not None and row[0] not in echoed_by_echoer:
                raise _error("sender", len(sender.header) + 3 + i, "sender echo has no matching echoer echo")
    arrivals: dict[int, list[tuple[int, int | None, int | None]]] = {}
    if echoer:
        for seq, rx, etx, rssi in echoer.rows:
            arrivals.setdefault(seq, []).append((rx, etx, rssi))
    min_echo, max_echo = (min(arrivals), max(arrivals)) if arrivals else (None, None)
    packets: list[dict[str, Any]] = []
    for seq, tx, echo in sender.rows:
        base = {"packet_id": f"{sh['session']}:{seq}", "sender_id": sh["board"], "receiver_id": sh["peer"], "sender_session_id": sh["session"], "seq": seq, "send_time_s": tx / 1e6}
        arr = arrivals.get(seq, [])
        if echo is not None:
            p = {**base, "outcome": "received", "echo_time_s": echo / 1e6}
            if arr:
                first = arr[0]
                p["receive_time_s"] = first[0] / 1e6
                p["rssi_dbm"] = first[2]
                if first[1] is not None:
                    p["echo_turnaround_s"] = (first[1] - first[0]) / 1e6
        elif arr:
            first = arr[0]
            p = {**base, "outcome": "received", "receive_time_s": first[0] / 1e6, "rssi_dbm": first[2]}
        elif echoer is None or min_echo is None or seq < min_echo or seq > max_echo:
            reason = "no_echoer_log" if echoer is None else ("before_echoer_range" if seq < min_echo else "after_echoer_range")
            p = {**base, "outcome": "unknown", "unknown_reason": reason}
        else:
            p = {**base, "outcome": "lost", "loss_reason": "not_in_echoer_log"}
        if sh.get("payload_bytes") is not None:
            p["payload_bytes"] = int(sh["payload_bytes"])
        packets.append(p)
        for k, (rx, _etx, rssi) in enumerate(arr[1:], 1):
            dup = {**base, "packet_id": f"{sh['session']}:{seq}:dup{k}", "outcome": "received", "receive_time_s": rx / 1e6, "duplicate": True, "rssi_dbm": rssi}
            packets.append(dup)
    # Mark first-arrival out-of-order against earlier arrival records.
    if echoer:
        logical = {p["seq"]: p for p in packets if not p.get("duplicate")}
        first_seen: set[int] = set()
        highest: int | None = None
        for seq, *_ in echoer.rows:
            if seq in first_seen:
                continue
            first_seen.add(seq)
            if highest is not None and highest > seq:
                logical[seq]["out_of_order"] = True
            highest = seq if highest is None else max(highest, seq)
    capture = {"transport": "ESP-NOW", "device_ids": [sh["board"], sh["peer"]], "firmware": sh["firmware"], "channel": int(sh["channel"]) if sh.get("channel") else None, "phy": sh.get("phy", "unrecorded")}
    raw: dict[str, Any] = {"schema": "zephyr-espnow-trace-1", "trace_id": trace_id, "status": status, "clock_domain": "sender_round_trip", "capture": capture, "packets": packets}
    if source is not None: raw["source"] = source
    if provenance is not None: raw["provenance"] = provenance
    if notes is not None: raw["notes"] = notes
    return raw


def _read_logs(sender_path: str | Path, echoer_path: str | Path | None) -> tuple[SenderLog, EchoerLog | None]:
    with Path(sender_path).open(encoding="utf-8", newline="") as handle:
        sender = parse_sender_log(handle.read(), Path(sender_path).name)
    echoer = None
    if echoer_path is not None:
        with Path(echoer_path).open(encoding="utf-8", newline="") as handle:
            echoer = parse_echoer_log(handle.read(), Path(echoer_path).name)
    return sender, echoer


def _build_trace(sender: SenderLog, echoer: EchoerLog | None, sender_path: str | Path,
                 echoer_path: str | Path | None, **kwargs: Any) -> EspNowTrace:
    """Cross-check the two logs, then validate the trace with ``sim.trace``."""
    try:
        raw = logs_to_trace_dict(sender, echoer, **kwargs)
    except EchoLogError as exc:
        # logs_to_trace_dict labels rows "sender"/"echoer"; name the real file.
        message = str(exc)
        if message.startswith("sender:"):
            raise EchoLogError(f"{Path(sender_path).name}:{message.split(':', 1)[1]}") from exc
        if message.startswith("echoer:") and echoer_path is not None:
            raise EchoLogError(f"{Path(echoer_path).name}:{message.split(':', 1)[1]}") from exc
        raise
    return EspNowTrace.from_dict(raw)


def read_echo_logs(sender_path: str | Path, echoer_path: str | Path | None = None, *, trace_id: str, status: str, source: str | None = None, provenance: dict | None = None, notes: str | None = None) -> EspNowTrace:
    sender, echoer = _read_logs(sender_path, echoer_path)
    return _build_trace(sender, echoer, sender_path, echoer_path, trace_id=trace_id, status=status,
                        source=source, provenance=provenance, notes=notes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
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
    try:
        args = parser.parse_args(argv)
        sender, echoer = _read_logs(args.sender, args.echoer)
        if args.status == "measured":
            required = ["operator", "captured_at", "code_revision", "instrument_model", "source"]
            missing = [x for x in required if not getattr(args, x)]
            if missing:
                raise ValueError("measured status requires: " + ", ".join("--" + x.replace("_", "-") for x in missing))
            provenance = {"instrument": {"kind": "esp32-echo-pair", "model": args.instrument_model, "settings": {"sender_log_sha256": sender.raw_sha256, "echoer_log_sha256": echoer.raw_sha256 if echoer else None, "sender_header": sender.header, "echoer_header": echoer.header if echoer else None}}, "calibration_ref": None, "calibration_ref_null_reason": "echo timing has no external calibration; one-way delay is rtt/2 under a symmetry assumption", "raw_sha256": sender.raw_sha256, "code_revision": args.code_revision, "captured_at": args.captured_at, "operator": args.operator, "ground_truth": None, "ground_truth_null_reason": "no independent clock reference; see docs/ESPNOW_ECHO_LOG.md"}
        else:
            provenance = None
        trace = _build_trace(sender, echoer, args.sender, args.echoer, trace_id=args.trace_id, status=args.status,
                             source=args.source, provenance=provenance, notes=args.notes)
        Path(args.out).write_text(json.dumps(trace.as_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        return 0
    except (EchoLogError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
