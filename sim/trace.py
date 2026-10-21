"""Measured ESP-NOW trace schema and replay helpers.

The schema keeps raw observations separate from the stochastic link fixture.
It can describe a measured capture today, while synthetic examples remain
explicitly labelled and cannot be mistaken for radio evidence. Replay keeps
the observed receive outcome and timestamps instead of fitting a distribution.

Clock domains (``clock_domain``):

* ``shared_monotonic``: send and receive stamps come from one clock, so
  one-way age ``receive_time_s - send_time_s`` is a real measurement.
* ``device_and_host`` and ``unknown``: stamps come from different clocks; no
  one-way age is computed.
* ``sender_round_trip`` (echo mode, a proposal on this branch): the sender
  stamps ``send_time_s`` and ``echo_time_s`` on its own clock, the receiver
  bounces the packet straight back, and ``receive_time_s`` (when present) is
  the receiver's own clock, which is NOT comparable with the sender's.
  ``echo_turnaround_s`` is how long the receiver held the packet before
  echoing it, measured on the receiver's clock (a difference of two stamps on
  one clock, so the unknown offset between the boards cancels). The round
  trip is ``rtt = echo_time_s - send_time_s - echo_turnaround_s``.

  One-way delay is then ESTIMATED as ``rtt / 2``. That rests on an
  assumption, not a measurement: the outbound leg (sender to receiver) and the
  return leg take the same time on average. If the outbound leg is really
  ``d_out`` and the return ``d_back``, then ``rtt / 2 = (d_out + d_back) / 2``,
  so the estimate is off by ``(d_back - d_out) / 2`` and nothing in an echo
  log can detect that error. A shared wired clock edge between the two boards
  is the only way to check it. Clock drift between the boards does not matter
  here because each difference uses one board's clock only, except that the
  receiver's turnaround is scaled by its clock-rate error (parts per million
  of a sub-millisecond hold, which is negligible).

  ``sim.link.replay_trace`` treats an echo trace like ``device_and_host``: it
  keeps the receiver-clock arrival order and computes no one-way age. A
  received packet known only from its echo has no arrival time, so replay
  stops with an error. Fit echo traces with ``tools/fit_espnow_log.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

TRACE_SCHEMA = "zephyr-espnow-trace-1"
TRACE_STATUSES = {"measured", "synthetic", "fixture"}
OUTCOMES = {"received", "lost", "unknown"}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[^ ]+(?:Z|[+-]\d{2}:\d{2})$")


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown field(s): {', '.join(unknown)}")


_PACKET_FIELDS = {
    "packet_id", "sender_id", "receiver_id", "sender_session_id", "seq", "send_time_s",
    "outcome", "receive_time_s", "callback_time_s", "rssi_dbm", "payload_bytes",
    "retry_count", "loss_reason", "unknown_reason", "duplicate", "out_of_order",
    "echo_time_s", "echo_turnaround_s",
}
CLOCK_DOMAINS = {"shared_monotonic", "device_and_host", "unknown", "sender_round_trip"}
_TRACE_FIELDS = {
    "schema", "trace_id", "status", "clock_domain", "source", "code_revision",
    "clock_uncertainty_s", "notes", "provenance", "capture", "packets",
}
_CAPTURE_FIELDS = {"transport", "device_ids", "firmware", "channel", "phy"}


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _optional_float(value: Any, field: str) -> float | None:
    return None if value is None else _finite_nonnegative(value, field)


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean or null")
    return value


def _sha256(value: Any, field: str) -> str:
    value = _required_text(value, field)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a 64-hex-character sha256 string")
    return value.lower()


def _timestamp(value: Any, field: str) -> str:
    value = _required_text(value, field)
    if not _TIMESTAMP_RE.fullmatch(value):
        raise ValueError(f"{field} must be an ISO 8601 timestamp with a UTC offset")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO 8601 timestamp with a UTC offset") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset")
    return value


_PROVENANCE_FIELDS = {
    "instrument", "calibration_ref", "calibration_ref_null_reason", "raw_sha256",
    "code_revision", "captured_at", "operator", "ground_truth", "ground_truth_null_reason",
}
_PROVENANCE_REQUIRED = {"instrument", "calibration_ref", "raw_sha256", "code_revision", "captured_at", "operator", "ground_truth"}


def _provenance(value: Any) -> dict[str, Any]:
    """Validate the shared capture-provenance block (same rules as the vision and bench traces)."""

    if not isinstance(value, Mapping):
        raise ValueError("provenance must be an object")
    _reject_unknown(value, _PROVENANCE_FIELDS, "provenance")
    missing = sorted(_PROVENANCE_REQUIRED - set(value))
    if missing:
        raise ValueError(f"provenance is missing required field(s): {', '.join(missing)}")
    instrument = value["instrument"]
    if not isinstance(instrument, Mapping):
        raise ValueError("provenance.instrument must be an object")
    _reject_unknown(instrument, {"kind", "model", "settings"}, "provenance.instrument")
    if set(instrument) != {"kind", "model", "settings"}:
        raise ValueError("provenance.instrument requires kind, model, and settings")
    if not isinstance(instrument["settings"], Mapping):
        raise ValueError("provenance.instrument.settings must be an object")
    result: dict[str, Any] = {
        "instrument": {
            "kind": _required_text(instrument["kind"], "provenance.instrument.kind"),
            "model": _required_text(instrument["model"], "provenance.instrument.model"),
            "settings": dict(instrument["settings"]),
        }
    }
    calibration = value["calibration_ref"]
    if calibration is None:
        result["calibration_ref"] = None
        result["calibration_ref_null_reason"] = _required_text(value.get("calibration_ref_null_reason"), "provenance.calibration_ref_null_reason")
    else:
        result["calibration_ref"] = _sha256(calibration, "provenance.calibration_ref")
        if "calibration_ref_null_reason" in value:
            raise ValueError("calibration_ref_null_reason is only allowed when calibration_ref is null")
    result["raw_sha256"] = _sha256(value["raw_sha256"], "provenance.raw_sha256")
    result["code_revision"] = _required_text(value["code_revision"], "provenance.code_revision")
    result["captured_at"] = _timestamp(value["captured_at"], "provenance.captured_at")
    result["operator"] = _required_text(value["operator"], "provenance.operator")
    ground_truth = value["ground_truth"]
    if ground_truth is None:
        result["ground_truth"] = None
        result["ground_truth_null_reason"] = _required_text(value.get("ground_truth_null_reason"), "provenance.ground_truth_null_reason")
    else:
        if not isinstance(ground_truth, Mapping):
            raise ValueError("provenance.ground_truth must be an object or null")
        _reject_unknown(ground_truth, {"method", "values", "uncertainty"}, "provenance.ground_truth")
        if set(ground_truth) != {"method", "values", "uncertainty"}:
            raise ValueError("provenance.ground_truth requires method, values, and uncertainty")
        if not isinstance(ground_truth["values"], Mapping) or not isinstance(ground_truth["uncertainty"], Mapping):
            raise ValueError("provenance.ground_truth values and uncertainty must be objects")
        result["ground_truth"] = {
            "method": _required_text(ground_truth["method"], "provenance.ground_truth.method"),
            "values": dict(ground_truth["values"]),
            "uncertainty": dict(ground_truth["uncertainty"]),
        }
        if "ground_truth_null_reason" in value:
            raise ValueError("ground_truth_null_reason is only allowed when ground_truth is non-null")
    return result


@dataclass(frozen=True)
class EspNowPacket:
    """One attempted packet observation from a capture.

    ``send_time_s`` and ``receive_time_s`` use the clock described by the
    parent trace. A null receive time is retained for a loss or an incomplete
    observation. ``packet_id`` should remain stable across raw-log exports;
    sequence numbers alone can repeat after a device reboot.

    ``echo_time_s`` and ``echo_turnaround_s`` exist only for echo captures
    (``clock_domain = "sender_round_trip"``, see the module docstring). An
    echo proves the outbound packet arrived, so a packet with an echo must be
    ``received``. A ``received`` packet with no echo means the return leg was
    lost (or the receiver did not echo). In that domain a ``received`` packet
    may omit ``receive_time_s`` when an echo shows it arrived.

    ``seq`` should be an unwrapped counter within one ``sender_session_id``.
    A narrow hardware counter (uint8, uint16) repeats values, and replay then
    derives false duplicate flags; either unwrap it when exporting or supply
    explicit ``duplicate`` and ``out_of_order`` flags.
    """

    packet_id: str
    sender_id: str
    receiver_id: str
    sender_session_id: str
    seq: int
    send_time_s: float
    outcome: str
    receive_time_s: float | None = None
    callback_time_s: float | None = None
    rssi_dbm: float | None = None
    payload_bytes: int | None = None
    retry_count: int | None = None
    loss_reason: str | None = None
    unknown_reason: str | None = None
    duplicate: bool | None = None
    out_of_order: bool | None = None
    echo_time_s: float | None = None
    echo_turnaround_s: float | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], clock_domain: str | None = None) -> "EspNowPacket":
        """Validate one packet.

        ``clock_domain`` is the parent trace's domain. It only relaxes or adds
        rules for ``sender_round_trip``; every other domain behaves as before.
        """
        if not isinstance(raw, Mapping):
            raise ValueError("packet must be an object")
        _reject_unknown(raw, _PACKET_FIELDS, "packet")
        packet_id = _required_text(raw.get("packet_id"), "packet_id")
        sender_id = _required_text(raw.get("sender_id"), "sender_id")
        receiver_id = _required_text(raw.get("receiver_id"), "receiver_id")
        sender_session_id = _required_text(raw.get("sender_session_id"), "sender_session_id")
        seq = raw.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise ValueError("seq must be a nonnegative integer")
        outcome = _required_text(raw.get("outcome"), "outcome")
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
        receive_time_s = _optional_float(raw.get("receive_time_s"), "receive_time_s")
        callback_time_s = _optional_float(raw.get("callback_time_s"), "callback_time_s")
        echo_time_s = _optional_float(raw.get("echo_time_s"), "echo_time_s")
        echo_turnaround_s = _optional_float(raw.get("echo_turnaround_s"), "echo_turnaround_s")
        echo_mode = clock_domain == "sender_round_trip"
        if not echo_mode and (echo_time_s is not None or echo_turnaround_s is not None):
            raise ValueError("echo_time_s and echo_turnaround_s are only allowed when clock_domain is sender_round_trip")
        if echo_turnaround_s is not None and echo_time_s is None:
            raise ValueError("echo_turnaround_s requires echo_time_s")
        if echo_time_s is not None and outcome != "received":
            raise ValueError("a packet with echo_time_s must be received (the echo proves it arrived)")
        if outcome == "received" and receive_time_s is None and not (echo_mode and echo_time_s is not None):
            raise ValueError("received packets require receive_time_s")
        if outcome != "received" and receive_time_s is not None:
            raise ValueError(f"{outcome} packets cannot have receive_time_s")
        payload_bytes = raw.get("payload_bytes")
        if payload_bytes is not None and (isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes < 0):
            raise ValueError("payload_bytes must be a nonnegative integer or null")
        retry_count = raw.get("retry_count")
        if retry_count is not None and (isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0):
            raise ValueError("retry_count must be a nonnegative integer or null")
        loss_reason = raw.get("loss_reason")
        if loss_reason is not None:
            loss_reason = _required_text(loss_reason, "loss_reason")
        if outcome == "lost" and loss_reason is None:
            raise ValueError("lost packets require loss_reason")
        unknown_reason = raw.get("unknown_reason")
        if unknown_reason is not None:
            unknown_reason = _required_text(unknown_reason, "unknown_reason")
        if outcome == "unknown" and unknown_reason is None:
            raise ValueError("unknown packets require unknown_reason")
        if outcome != "unknown" and unknown_reason is not None:
            raise ValueError(f"{outcome} packets cannot have unknown_reason")
        if outcome != "lost" and loss_reason is not None:
            raise ValueError(f"{outcome} packets cannot have loss_reason")
        rssi_dbm = raw.get("rssi_dbm")
        if rssi_dbm is not None:
            if isinstance(rssi_dbm, bool) or not isinstance(rssi_dbm, (int, float)):
                raise ValueError("rssi_dbm must be finite or null")
            rssi_dbm = float(rssi_dbm)
            if not math.isfinite(rssi_dbm):
                raise ValueError("rssi_dbm must be finite or null")
        # send_time_s is checked here, after rssi_dbm, as before echo mode existed,
        # so non-echo traces report errors in the same order.
        send_time_s = _finite_nonnegative(raw.get("send_time_s"), "send_time_s")
        if echo_time_s is not None:
            if echo_time_s < send_time_s:
                raise ValueError("echo_time_s cannot precede send_time_s (both are on the sender's clock)")
            if echo_turnaround_s is not None and echo_turnaround_s > echo_time_s - send_time_s:
                raise ValueError("echo_turnaround_s cannot exceed echo_time_s - send_time_s")
        return cls(
            packet_id=packet_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            sender_session_id=sender_session_id,
            seq=seq,
            send_time_s=send_time_s,
            outcome=outcome,
            receive_time_s=receive_time_s,
            callback_time_s=callback_time_s,
            rssi_dbm=rssi_dbm,
            payload_bytes=payload_bytes,
            retry_count=retry_count,
            loss_reason=loss_reason,
            unknown_reason=unknown_reason,
            duplicate=_optional_bool(raw.get("duplicate"), "duplicate"),
            out_of_order=_optional_bool(raw.get("out_of_order"), "out_of_order"),
            echo_time_s=echo_time_s,
            echo_turnaround_s=echo_turnaround_s,
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Echo fields are left out when unused so non-echo traces serialize
        # exactly as they did before echo mode existed.
        for key in ("echo_time_s", "echo_turnaround_s"):
            if data[key] is None:
                data.pop(key)
        return data

    def round_trip_s(self) -> float | None:
        """Sender-clock round trip minus the receiver's hold time, or None.

        Only meaningful for ``sender_round_trip`` captures. A missing
        turnaround is treated as zero, which makes the round trip (and the
        ``rtt / 2`` one-way estimate) too long by the receiver's hold time.
        """
        if self.echo_time_s is None:
            return None
        return self.echo_time_s - self.send_time_s - (self.echo_turnaround_s or 0.0)


@dataclass(frozen=True)
class EspNowTrace:
    """Validated trace manifest plus ordered packet observations."""

    trace_id: str
    status: str
    clock_domain: str
    packets: tuple[EspNowPacket, ...]
    capture: dict[str, Any]
    source: str | None = None
    code_revision: str | None = None
    clock_uncertainty_s: float | None = None
    notes: str | None = None
    provenance: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EspNowTrace":
        if not isinstance(raw, Mapping):
            raise ValueError("trace must be an object")
        _reject_unknown(raw, _TRACE_FIELDS, "trace")
        if raw.get("schema") != TRACE_SCHEMA:
            raise ValueError(f"schema must be {TRACE_SCHEMA}")
        status = _required_text(raw.get("status"), "status")
        if status not in TRACE_STATUSES:
            raise ValueError(f"status must be one of {sorted(TRACE_STATUSES)}")
        clock_domain = _required_text(raw.get("clock_domain"), "clock_domain")
        if clock_domain not in CLOCK_DOMAINS:
            raise ValueError("clock_domain must be shared_monotonic, device_and_host, unknown, or sender_round_trip")
        capture = raw.get("capture")
        if not isinstance(capture, Mapping):
            raise ValueError("capture must be an object")
        _reject_unknown(capture, _CAPTURE_FIELDS, "capture")
        if capture.get("transport") != "ESP-NOW":
            raise ValueError('capture.transport must be "ESP-NOW"')
        device_ids = capture.get("device_ids")
        if not isinstance(device_ids, list) or not device_ids or any(not isinstance(value, str) or not value.strip() for value in device_ids):
            raise ValueError("capture.device_ids must be a non-empty string array")
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("capture.device_ids must be unique")
        firmware = _required_text(capture.get("firmware"), "capture.firmware")
        channel = capture.get("channel")
        if channel is not None and (isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 14):
            raise ValueError("capture.channel must be an integer from 1 to 14 or null")
        phy = _required_text(capture.get("phy"), "capture.phy")
        capture = {"transport": "ESP-NOW", "device_ids": list(device_ids), "firmware": firmware, "channel": channel, "phy": phy}
        packets_raw = raw.get("packets")
        if not isinstance(packets_raw, list):
            raise ValueError("packets must be an array")
        packets = tuple(EspNowPacket.from_dict(packet, clock_domain) for packet in packets_raw)
        packet_ids = [packet.packet_id for packet in packets]
        if len(packet_ids) != len(set(packet_ids)):
            raise ValueError("packet_id values must be unique")
        if clock_domain == "shared_monotonic":
            for packet in packets:
                if packet.receive_time_s is not None and packet.receive_time_s < packet.send_time_s:
                    raise ValueError("shared_monotonic receive_time_s cannot precede send_time_s")
                if packet.callback_time_s is not None and packet.callback_time_s < packet.send_time_s:
                    raise ValueError("shared_monotonic callback_time_s cannot precede send_time_s")
                if packet.callback_time_s is not None and packet.receive_time_s is not None and packet.callback_time_s < packet.receive_time_s:
                    raise ValueError("shared_monotonic callback_time_s cannot precede receive_time_s")
        if status == "measured" and raw.get("source") is None:
            raise ValueError("measured traces require source")
        provenance = _provenance(raw["provenance"]) if "provenance" in raw else None
        if status == "measured" and provenance is None:
            raise ValueError("measured traces require provenance")
        uncertainty = _optional_float(raw.get("clock_uncertainty_s"), "clock_uncertainty_s")
        return cls(
            trace_id=_required_text(raw.get("trace_id"), "trace_id"),
            status=status,
            clock_domain=clock_domain,
            packets=packets,
            capture=capture,
            source=None if raw.get("source") is None else _required_text(raw["source"], "source"),
            code_revision=None if raw.get("code_revision") is None else _required_text(raw["code_revision"], "code_revision"),
            clock_uncertainty_s=uncertainty,
            notes=None if raw.get("notes") is None else _required_text(raw["notes"], "notes"),
            provenance=provenance,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "EspNowTrace":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema": TRACE_SCHEMA,
            "trace_id": self.trace_id,
            "status": self.status,
            "clock_domain": self.clock_domain,
            "capture": dict(self.capture),
            "packets": [packet.as_dict() for packet in self.packets],
        }
        for key in ("source", "code_revision", "clock_uncertainty_s", "notes", "provenance"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


def validate_trace(raw: Mapping[str, Any]) -> EspNowTrace:
    """Validate a decoded JSON object without fitting or changing observations."""

    return EspNowTrace.from_dict(raw)


def iter_received(trace: EspNowTrace) -> Iterable[EspNowPacket]:
    """Yield received observations in callback order, falling back to receive time."""

    def order_key(packet: EspNowPacket) -> tuple[float, str]:
        return (packet.callback_time_s if packet.callback_time_s is not None else (packet.receive_time_s or 0.0), packet.packet_id)

    return iter(sorted((packet for packet in trace.packets if packet.outcome == "received"), key=order_key))
