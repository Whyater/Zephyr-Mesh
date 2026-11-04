"""Wire format for the drone-to-drone state message (simulation only).

The cooperative simulator (``sim.coop.simulator``) shares a ``"state"``
message between drones: the sender's own position and velocity, its target
track (a 6-value mean and a 6x6 covariance), the time that track refers to,
and optionally a slot claim.  Inside the sim that message is a Python dict and
the radio only sees a byte count.  Real ESP-NOW firmware needs fixed bytes, so
this module defines them.  It is not wired into any simulator or registered
run, and no firmware implements it yet.

Layout (version 1, little-endian, no padding, 161 bytes)::

    offset size type     field
         0    1 uint8    version (1)
         1    1 uint8    message type (1 = state)
         2    1 uint8    sender id, 0..254 (255 reserved)
         3    1 uint8    flags: bit 0 = slot claim present; other bits must be 0
         4    4 uint32   sequence number (same 32-bit space as sim.radio.seq)
         8    8 uint64   sender send time, microseconds
        16    4 int32    track source time minus send time, microseconds
        20   12 3xf32    own position, m
        32   12 3xf32    own velocity, m/s
        44   24 6xf32    target track mean (x, y, z, vx, vy, vz)
        68   84 21xf32   target track covariance, upper triangle, row by row
       152    1 uint8    slot index 0..31, or 255 when no claim
       153    4 uint32   slot members bitmask (bit k = drone id k, ids 0..31)
       157    4 uint32   CRC-32 (zlib / IEEE 802.3) of bytes 0..156

Why 250: the ESP-IDF ESP-NOW guide gives ``ESP_NOW_MAX_DATA_LEN`` = 250 bytes
for ESP-NOW v1.0 and ``ESP_NOW_MAX_DATA_LEN_V2`` = 1470 bytes for v2.0, and
says v1.0 receivers truncate or drop longer v2.0 packets.  Staying at or
under 250 keeps both versions working.  Source:
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html

The 802.11 frame check sequence already catches radio bit errors, so the CRC
here mainly catches truncation, a sender and receiver built with different
layouts, and memory bugs.  Whether an ESP32 ROM CRC routine produces the same
value as ``zlib.crc32`` has not been checked.

Float32 keeps 24 significant bits, so a value in [512, 1024) m moves by at
most 2**-15 m (0.0305 mm) on the round trip.  Times are whole microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence
import zlib

import numpy as np


ESPNOW_MAX_PAYLOAD_BYTES = 250
PACKET_VERSION = 1
MSG_TYPE_STATE = 1

FLAG_SLOT_CLAIM = 0x01
RESERVED_SENDER_ID = 255
NO_SLOT = 255
MAX_SLOT_INDEX = 31
MAX_MEMBER_ID = 31

POSITION_LIMIT_M = 1000.0
VELOCITY_LIMIT_MPS = 100.0
VARIANCE_LIMIT = 1.0e6
_CORRELATION_SLACK = 1.0e-6

_BODY = struct.Struct("<BBBBIQi33fBI")
_CRC = struct.Struct("<I")
STATE_PACKET_BYTES = _BODY.size + _CRC.size

_UPPER = np.triu_indices(6)
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1

assert STATE_PACKET_BYTES <= ESPNOW_MAX_PAYLOAD_BYTES


class PacketError(ValueError):
    """A packet that must not be used.  ``reason`` says which check failed:
    ``"length"``, ``"version"``, ``"checksum"``, ``"type"`` or ``"range"``."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


@dataclass(frozen=True)
class StateMessage:
    sender_id: int
    seq: int
    send_time_s: float
    source_time_s: float
    position_m: Sequence[float]
    velocity_mps: Sequence[float]
    track_mean: Sequence[float]
    track_cov: Sequence[Sequence[float]]
    slot_index: int | None = None
    slot_members: tuple[int, ...] = ()


def crc32(data: bytes) -> int:
    """CRC-32 as used by zlib, Ethernet and PNG."""
    return zlib.crc32(data) & 0xFFFFFFFF


def _int_in(value, low: int, high: int, name: str) -> int:
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        raise PacketError("range", f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise PacketError("range", f"{name}={value} outside [{low}, {high}]")
    return value


def _check_floats(floats: np.ndarray) -> None:
    """Range checks shared by the encoder and decoder (on float32 values)."""
    if not np.all(np.isfinite(floats)):
        raise PacketError("range", "non-finite float")
    pos, vel, mean, upper = floats[0:3], floats[3:6], floats[6:12], floats[12:33]
    if np.any(np.abs(pos) > POSITION_LIMIT_M) or np.any(np.abs(mean[:3]) > POSITION_LIMIT_M):
        raise PacketError("range", f"position beyond {POSITION_LIMIT_M} m")
    if np.any(np.abs(vel) > VELOCITY_LIMIT_MPS) or np.any(np.abs(mean[3:]) > VELOCITY_LIMIT_MPS):
        raise PacketError("range", f"velocity beyond {VELOCITY_LIMIT_MPS} m/s")
    cov = np.zeros((6, 6))
    cov[_UPPER] = upper
    diag = np.diag(cov)
    if np.any(diag < 0.0) or np.any(diag > VARIANCE_LIMIT):
        raise PacketError("range", "covariance diagonal outside [0, 1e6]")
    # A covariance entry can never exceed the geometric mean of its two
    # variances (a correlation above 1).  This is a cheap sanity check, not a
    # full positive-definiteness test.
    limit = np.sqrt(np.outer(diag, diag)) * (1.0 + _CORRELATION_SLACK)
    if np.any(np.abs(cov) > limit):
        raise PacketError("range", "covariance entry implies correlation above 1")


def _check_slots(flags: int, slot_index: int, mask: int) -> None:
    if flags & ~FLAG_SLOT_CLAIM:
        raise PacketError("range", f"reserved flag bits set: {flags:#04x}")
    if flags & FLAG_SLOT_CLAIM:
        if slot_index > MAX_SLOT_INDEX:
            raise PacketError("range", f"slot index {slot_index} above {MAX_SLOT_INDEX}")
    elif slot_index != NO_SLOT or mask != 0:
        raise PacketError("range", "slot fields filled without the slot flag")


def encode_state(msg: StateMessage) -> bytes:
    """Pack ``msg`` into exactly ``STATE_PACKET_BYTES`` bytes.

    Raises ``PacketError`` (a ``ValueError``) for anything the decoder would
    reject, so a packet this function returns always decodes.
    """
    sender = _int_in(msg.sender_id, 0, RESERVED_SENDER_ID - 1, "sender_id")
    seq = _int_in(msg.seq, 0, _UINT32_MAX, "seq")
    send_s = float(msg.send_time_s)
    source_s = float(msg.source_time_s)
    if not (math.isfinite(send_s) and math.isfinite(source_s)) or send_s < 0.0:
        raise PacketError("range", "send time must be finite and nonnegative")
    send_us = _int_in(round(send_s * 1e6), 0, _UINT64_MAX, "send_time_us")
    offset_us = _int_in(round(source_s * 1e6) - send_us, _INT32_MIN, _INT32_MAX, "source offset us")

    cov = np.asarray(msg.track_cov, dtype=float)
    if cov.shape != (6, 6):
        raise PacketError("range", f"track_cov must be 6x6, got {cov.shape}")
    parts = [np.asarray(msg.position_m, dtype=float), np.asarray(msg.velocity_mps, dtype=float),
             np.asarray(msg.track_mean, dtype=float)]
    if [p.shape for p in parts] != [(3,), (3,), (6,)]:
        raise PacketError("range", "position and velocity need 3 values, track_mean needs 6")
    floats = np.concatenate(parts + [cov[_UPPER]]).astype(np.float32)
    _check_floats(floats)

    if msg.slot_index is None:
        if msg.slot_members:
            raise PacketError("range", "slot_members given without slot_index")
        flags, slot_index, mask = 0, NO_SLOT, 0
    else:
        flags = FLAG_SLOT_CLAIM
        slot_index = _int_in(msg.slot_index, 0, MAX_SLOT_INDEX, "slot_index")
        mask = 0
        for member in msg.slot_members:
            mask |= 1 << _int_in(member, 0, MAX_MEMBER_ID, "slot member id")
    _check_slots(flags, slot_index, mask)

    body = _BODY.pack(PACKET_VERSION, MSG_TYPE_STATE, sender, flags, seq, send_us, offset_us,
                      *floats.tolist(), slot_index, mask)
    return body + _CRC.pack(crc32(body))


def decode_state(data: bytes | bytearray | memoryview) -> StateMessage:
    """Unpack and check one packet.  Raises ``PacketError`` on any problem.

    Check order: version, length, checksum, message type, then field ranges.
    """
    data = bytes(data)
    if not data:
        raise PacketError("length", "empty packet")
    if data[0] != PACKET_VERSION:
        raise PacketError("version", f"version {data[0]}, expected {PACKET_VERSION}")
    if len(data) != STATE_PACKET_BYTES:
        raise PacketError("length", f"{len(data)} bytes, expected {STATE_PACKET_BYTES}")
    body = data[:_BODY.size]
    (stored,) = _CRC.unpack(data[_BODY.size:])
    if crc32(body) != stored:
        raise PacketError("checksum", "CRC-32 mismatch")

    fields = _BODY.unpack(body)
    _version, msg_type, sender, flags, seq, send_us, offset_us = fields[:7]
    floats = np.asarray(fields[7:40], dtype=np.float32)
    slot_index, mask = fields[40], fields[41]
    if msg_type != MSG_TYPE_STATE:
        raise PacketError("type", f"unknown message type {msg_type}")
    if sender == RESERVED_SENDER_ID:
        raise PacketError("range", "sender id 255 is reserved")
    _check_slots(flags, slot_index, mask)
    _check_floats(floats)

    values = floats.astype(float)
    cov = np.zeros((6, 6))
    cov[_UPPER] = values[12:33]
    cov = np.triu(cov) + np.triu(cov, 1).T
    has_slot = bool(flags & FLAG_SLOT_CLAIM)
    return StateMessage(
        sender_id=sender,
        seq=seq,
        send_time_s=send_us / 1e6,
        source_time_s=(send_us + offset_us) / 1e6,
        position_m=tuple(values[0:3].tolist()),
        velocity_mps=tuple(values[3:6].tolist()),
        track_mean=tuple(values[6:12].tolist()),
        track_cov=tuple(tuple(row) for row in cov.tolist()),
        slot_index=slot_index if has_slot else None,
        slot_members=tuple(k for k in range(MAX_MEMBER_ID + 1) if mask >> k & 1) if has_slot else (),
    )
