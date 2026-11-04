"""Tests for the ESP-NOW state packet wire format (sim.radio.packet).

Several checks here do not reuse the module's own struct strings: the byte
offsets, the total size, the CRC check value and the float32 error bounds are
worked out by hand below so they can catch a wrong layout, not just restate it.
"""

import math
import struct
import zlib

import numpy as np
import pytest

from sim.radio.packet import (
    ESPNOW_MAX_PAYLOAD_BYTES,
    MSG_TYPE_STATE,
    PACKET_VERSION,
    STATE_PACKET_BYTES,
    PacketError,
    StateMessage,
    crc32,
    decode_state,
    encode_state,
)
from sim.radio.seq import SEQ_MOD, SequenceTracker, seq_next


def _cov(scale=1.0):
    a = np.arange(36, dtype=float).reshape(6, 6) / 100.0
    return scale * (a @ a.T + np.eye(6))


def _message(**overrides):
    fields = dict(
        sender_id=3,
        seq=0x01020304,
        send_time_s=12.345678,
        source_time_s=12.300001,
        position_m=(1.5, -2.25, 3.0),
        velocity_mps=(0.1, -0.2, 0.3),
        track_mean=(10.0, 20.0, 3.0, 0.7, 0.0, -0.1),
        track_cov=_cov(),
        slot_index=None,
        slot_members=(),
    )
    fields.update(overrides)
    return StateMessage(**fields)


def _reseal(raw: bytes) -> bytes:
    """Replace the last 4 bytes with a correct CRC-32 of the rest."""
    body = raw[:-4]
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


# ---- size and layout (hand-worked, not read from the module) ------------

def test_size_hand_count_and_espnow_limit():
    header = 1 + 1 + 1 + 1 + 4 + 8 + 4  # version, type, sender, flags, seq, send us, source offset us
    floats = (3 + 3 + 6 + 21) * 4  # own pos, own vel, track mean, cov upper triangle
    slots = 1 + 4  # slot index, member bitmask
    crc = 4
    assert STATE_PACKET_BYTES == header + floats + slots + crc == 161
    assert ESPNOW_MAX_PAYLOAD_BYTES == 250
    assert len(encode_state(_message())) == STATE_PACKET_BYTES <= ESPNOW_MAX_PAYLOAD_BYTES


def test_header_bytes_are_little_endian_at_fixed_offsets():
    raw = encode_state(_message())
    assert raw[0] == PACKET_VERSION == 1
    assert raw[1] == MSG_TYPE_STATE == 1
    assert raw[2] == 3  # sender id
    assert raw[3] == 0  # flags: no slot claim
    assert raw[4:8] == bytes([0x04, 0x03, 0x02, 0x01])  # seq, little-endian
    assert int.from_bytes(raw[8:16], "little") == 12_345_678  # send time in microseconds
    assert int.from_bytes(raw[16:20], "little", signed=True) == -45_677  # source - send, microseconds
    assert struct.unpack("<f", raw[20:24])[0] == 1.5  # first float: own x
    assert raw[-4:] == struct.pack("<I", zlib.crc32(raw[:-4]))


def test_crc_matches_standard_check_value():
    # CRC-32 (IEEE 802.3, the zlib one) of "123456789" is 0xCBF43926 by definition.
    assert crc32(b"123456789") == 0xCBF43926


# ---- round trip and precision ---------------------------------------------

def test_round_trip_exact_fields():
    msg = _message(slot_index=2, slot_members=(0, 3, 7))
    out = decode_state(encode_state(msg))
    assert out.sender_id == 3
    assert out.seq == 0x01020304
    assert out.slot_index == 2
    assert out.slot_members == (0, 3, 7)
    assert out.send_time_s == pytest.approx(12.345678, abs=0.5e-6)
    assert out.source_time_s == pytest.approx(12.300001, abs=1e-6)
    np.testing.assert_array_equal(out.position_m, np.float32([1.5, -2.25, 3.0]))
    cov = np.asarray(out.track_cov)
    np.testing.assert_array_equal(cov, cov.T)


def test_float32_position_error_is_within_half_ulp_in_mm():
    # A float32 has 24 significant bits.  For |x| in [512, 1024) m the spacing
    # between float32 values is 2**(9 - 23) m, so rounding moves a value by at
    # most half that: 2**-15 m = 0.0305 mm.  For |x| in [1, 2) m it is
    # 2**-24 m = 0.0000596 mm.
    rng = np.random.default_rng(22)
    worst_mm = 0.0
    for _ in range(200):
        pos = rng.uniform(512.0, 999.0, 3) * rng.choice([-1.0, 1.0], 3)
        mean = np.r_[rng.uniform(-999.0, 999.0, 3), rng.uniform(-99.0, 99.0, 3)]
        out = decode_state(encode_state(_message(position_m=pos, track_mean=mean)))
        err_mm = np.abs(np.asarray(out.position_m) - pos) * 1000.0
        worst_mm = max(worst_mm, float(err_mm.max()))
        assert np.all(err_mm <= 2.0**-15 * 1000.0)
        bound = np.array([2.0 ** (math.floor(math.log2(abs(v))) - 24) for v in mean])
        assert np.all(np.abs(np.asarray(out.track_mean) - mean) <= bound)
    assert worst_mm <= 0.0306
    assert worst_mm > 0.0  # float32 really does lose something at this range

    small = np.array([1.0 + 1e-7, 1.333333333, -1.999999])
    out = decode_state(encode_state(_message(position_m=small)))
    assert np.all(np.abs(np.asarray(out.position_m) - small) * 1000.0 <= 2.0**-24 * 1000.0)


def test_covariance_round_trip_relative_error():
    cov = _cov(scale=0.04)
    out = np.asarray(decode_state(encode_state(_message(track_cov=cov))).track_cov)
    assert np.all(np.abs(out - cov) <= np.abs(cov) * 2.0**-24 + 1e-30)


def test_lower_triangle_is_ignored_upper_is_sent():
    cov = _cov()
    lopsided = cov.copy()
    lopsided[1, 0] += 5.0  # lower triangle only
    out = decode_state(encode_state(_message(track_cov=lopsided)))
    assert np.asarray(out.track_cov)[1, 0] == np.float32(cov[0, 1])


# ---- sequence number wraparound ------------------------------------------

def test_sequence_wraparound_through_packets():
    seqs = [SEQ_MOD - 2]
    for _ in range(3):
        seqs.append(seq_next(seqs[-1]))
    assert seqs == [SEQ_MOD - 2, SEQ_MOD - 1, 0, 1]
    tracker = SequenceTracker(window=64)
    statuses = []
    for s in seqs:
        decoded = decode_state(encode_state(_message(seq=s)))
        assert decoded.seq == s
        statuses.append(tracker.observe("d3", decoded.seq).status)
    assert statuses == ["new"] * 4
    with pytest.raises(ValueError):
        encode_state(_message(seq=SEQ_MOD))
    with pytest.raises(ValueError):
        encode_state(_message(seq=-1))


# ---- decoder rejections ---------------------------------------------------

def test_rejects_wrong_version():
    raw = bytearray(encode_state(_message()))
    raw[0] = 2
    with pytest.raises(PacketError) as info:
        decode_state(_reseal(bytes(raw)))
    assert info.value.reason == "version"


@pytest.mark.parametrize("cut", [0, 1, 160, 162, 250])
def test_rejects_wrong_length(cut):
    raw = encode_state(_message())
    data = raw[:cut] if cut <= len(raw) else raw + bytes(cut - len(raw))
    with pytest.raises(PacketError) as info:
        decode_state(data)
    assert info.value.reason == "length"


@pytest.mark.parametrize("index", [2, 30, 100, 160])
def test_rejects_bad_checksum(index):
    raw = bytearray(encode_state(_message()))
    raw[index] ^= 0x01
    with pytest.raises(PacketError) as info:
        decode_state(bytes(raw))
    assert info.value.reason == "checksum"


def _field_offset(name):
    # Hand offsets: header 20 bytes, then 33 float32 values, then slot fields.
    floats = {"pos_x": 0, "vel_x": 3, "mean_x": 6, "mean_vx": 9, "cov00": 12, "cov01": 13}
    if name in floats:
        return 20 + 4 * floats[name]
    return {"slot_index": 152, "slot_mask": 153}[name]


@pytest.mark.parametrize(
    "name,value",
    [
        ("pos_x", 1000.5),
        ("pos_x", float("nan")),
        ("vel_x", -100.5),
        ("mean_x", float("inf")),
        ("mean_vx", 250.0),
        ("cov00", -1.0),
        ("cov00", 2e6),
        ("cov01", 1e3),  # |c01| far above sqrt(c00 * c11): not a covariance
    ],
)
def test_rejects_out_of_range_floats(name, value):
    raw = bytearray(encode_state(_message()))
    off = _field_offset(name)
    raw[off:off + 4] = struct.pack("<f", value)
    with pytest.raises(PacketError) as info:
        decode_state(_reseal(bytes(raw)))
    assert info.value.reason == "range"


@pytest.mark.parametrize(
    "offset,value,reason",
    [
        (1, 9, "type"),  # unknown message type
        (2, 255, "range"),  # sender id 255 is reserved
        (3, 0x02, "range"),  # reserved flag bit set
        (3, 0x00, "range"),  # flag clear but slot fields filled (set below)
    ],
)
def test_rejects_bad_header_values(offset, value, reason):
    raw = bytearray(encode_state(_message(slot_index=1, slot_members=(1,))))
    raw[offset] = value
    with pytest.raises(PacketError) as info:
        decode_state(_reseal(bytes(raw)))
    assert info.value.reason == reason


def test_rejects_slot_index_out_of_range():
    raw = bytearray(encode_state(_message(slot_index=1, slot_members=(1,))))
    raw[_field_offset("slot_index")] = 40
    with pytest.raises(PacketError) as info:
        decode_state(_reseal(bytes(raw)))
    assert info.value.reason == "range"


def test_encoder_refuses_values_the_decoder_would_reject():
    with pytest.raises(ValueError):
        encode_state(_message(position_m=(1001.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        encode_state(_message(sender_id=255))
    with pytest.raises(ValueError):
        encode_state(_message(slot_index=None, slot_members=(1,)))
    with pytest.raises(ValueError):
        encode_state(_message(slot_index=3, slot_members=(32,)))
    with pytest.raises(ValueError):
        encode_state(_message(send_time_s=-0.001))
    with pytest.raises(ValueError):
        encode_state(_message(track_cov=np.eye(5)))


def test_decode_accepts_bytearray_and_memoryview():
    raw = encode_state(_message())
    assert decode_state(bytearray(raw)).seq == decode_state(memoryview(raw)).seq
