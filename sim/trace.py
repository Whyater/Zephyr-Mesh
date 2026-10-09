"""Measured ESP-NOW trace schema and replay helpers.

The schema keeps raw observations separate from the stochastic link fixture.
It can describe a measured capture today, while synthetic examples remain
explicitly labelled and cannot be mistaken for radio evidence. Replay keeps
the observed receive outcome and timestamps instead of fitting a distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

TRACE_SCHEMA = "zephyr-espnow-trace-1"
TRACE_STATUSES = {"measured", "synthetic", "fixture"}
OUTCOMES = {"received", "lost", "unknown"}


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


@dataclass(frozen=True)
class EspNowPacket:
    """One attempted packet observation from a capture.

    ``send_time_s`` and ``receive_time_s`` use the clock described by the
    parent trace. A null receive time is retained for a loss or an incomplete
    observation. ``packet_id`` should remain stable across raw-log exports;
    sequence numbers alone can repeat after a device reboot.
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

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EspNowPacket":
        if not isinstance(raw, Mapping):
            raise ValueError("packet must be an object")
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
        if outcome == "received" and receive_time_s is None:
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
        return cls(
            packet_id=packet_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            sender_session_id=sender_session_id,
            seq=seq,
            send_time_s=_finite_nonnegative(raw.get("send_time_s"), "send_time_s"),
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
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EspNowTrace":
        if not isinstance(raw, Mapping):
            raise ValueError("trace must be an object")
        if raw.get("schema") != TRACE_SCHEMA:
            raise ValueError(f"schema must be {TRACE_SCHEMA}")
        status = _required_text(raw.get("status"), "status")
        if status not in TRACE_STATUSES:
            raise ValueError(f"status must be one of {sorted(TRACE_STATUSES)}")
        clock_domain = _required_text(raw.get("clock_domain"), "clock_domain")
        if clock_domain not in {"shared_monotonic", "device_and_host", "unknown"}:
            raise ValueError("clock_domain must be shared_monotonic, device_and_host, or unknown")
        capture = raw.get("capture")
        if not isinstance(capture, Mapping):
            raise ValueError("capture must be an object")
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
        packets = tuple(EspNowPacket.from_dict(packet) for packet in packets_raw)
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
        for key in ("source", "code_revision", "clock_uncertainty_s", "notes"):
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
