"""Hand check of tools/espnow_airtime_budget.py against numbers worked on paper.

None of the expected numbers below come from the tool or from sim/radio.  Each one
is built from a published rule, and the comment next to it says which:

- ESP-NOW frame wrapper: Espressif's ESP-NOW guide, "Frame Format" section,
  https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html
  (24 B MAC header, 1 B category, 3 B organization id, 4 B random value,
  7 B vendor element header, 4 B FCS, and "The default ESP-NOW bit rate is 1 Mbps").
- PHY timing: IEEE 802.11 TXTIME rules as quoted in the Linux kernel's
  ieee80211_frame_duration(),
  https://raw.githubusercontent.com/torvalds/linux/master/net/mac80211/util.c
  DSSS/CCK: TXTIME = PreambleLength + PLCPHeaderTime + Ceiling(LENGTH x 8 / DATARATE),
  with 144 + 48 us long preamble or 72 + 24 us short.  ERP-OFDM: 16 us preamble +
  4 us SIGNAL + 4 us x Ceiling((16 + 8 x LENGTH + 6) / (4 x DATARATE)) + 6 us signal
  extension.
"""
from __future__ import annotations

import math

import pytest

from sim.radio.packet import ESPNOW_MAX_PAYLOAD_BYTES, STATE_PACKET_BYTES
from tools import espnow_airtime_budget as ab


# The state packet, field by field from the layout table in sim/radio/packet.py:
# version, type, sender, flags (1 B each), seq 4, send time 8, source offset 4,
# 33 float32 values (3 position + 3 velocity + 6 track mean + 21 covariance),
# slot index 1, slot mask 4, CRC-32 4.
PAYLOAD = 4 * 1 + 4 + 8 + 4 + 33 * 4 + 1 + 4 + 4
WRAPPER = 24 + 1 + 3 + 4 + 7 + 4
MPDU_BITS = 8 * (PAYLOAD + WRAPPER)  # 204 bytes on the air

# 1 Mbit/s DSSS, long preamble: 192 us, then one bit per microsecond
DATA_1M = 144 + 48 + MPDU_BITS / 1.0
ACK_1M = 144 + 48 + 8 * 14  # ACK is 14 bytes, same rate
SIFS = 10
DIFS = SIFS + 2 * 20  # two 20 us DSSS slots
MEAN_BACKOFF = 31 / 2 * 20  # half of CWmin = 31 slots


def test_packet_size_and_wrapper():
    assert PAYLOAD == STATE_PACKET_BYTES == 161
    assert PAYLOAD <= ESPNOW_MAX_PAYLOAD_BYTES == 250
    assert WRAPPER == ab.espnow_overhead_bytes() == 43
    assert DATA_1M == 1824


@pytest.mark.parametrize(
    "drones, rate_hz, busy, busy_access, unicast, aloha",
    [
        # 50 frames/s x 1824 us; one sender cannot collide with itself
        (1, 50.0, 0.0912, 0.1092, 0.1069, 0.0),
        # 250 frames/s; each of 4 other drones overlaps with chance 2 x 1824 us x 50 Hz = 0.1824
        (5, 50.0, 0.456, 0.546, 0.5345, 1 - (1 - 0.1824) ** 4),
        # 1000 frames/s asks for 1.824 s of air every second: the channel cannot carry it
        (10, 100.0, 1.824, 2.184, 2.138, 1 - (1 - 0.3648) ** 9),
    ],
)
def test_default_rate_cases_by_hand(drones, rate_hz, busy, busy_access, unicast, aloha):
    frames = drones * rate_hz
    # sanity: the table values above really are frames x time
    assert busy == pytest.approx(frames * DATA_1M * 1e-6)
    assert busy_access == pytest.approx(frames * (DATA_1M + DIFS + MEAN_BACKOFF) * 1e-6)
    assert unicast == pytest.approx(frames * (DATA_1M + SIFS + ACK_1M) * 1e-6)

    b = ab.cell(drones, rate_hz, "broadcast")
    u = ab.cell(drones, rate_hz, "unicast_gs")
    # pass band set before comparing: 0.1 % for the default rate
    assert b["frames_per_s"] == frames
    assert b["utilization"] == pytest.approx(busy, rel=1e-3)
    assert b["utilization_with_access"] == pytest.approx(busy_access, rel=1e-3)
    assert u["utilization"] == pytest.approx(unicast, rel=1e-3)
    assert b["aloha_collision"] == pytest.approx(aloha, rel=1e-3, abs=1e-9)


def _ofdm(rate_mbps, bits):
    return 16 + 4 + 4 * math.ceil((16 + bits + 6) / (4 * rate_mbps)) + 6


@pytest.mark.parametrize(
    "name, expected_us",
    [
        ("2M_S", 72 + 24 + math.ceil(MPDU_BITS / 2)),       # 912
        ("11M_S", 72 + 24 + math.ceil(MPDU_BITS / 11)),     # 96 + ceil(148.36) = 245
        ("6M OFDM", _ofdm(6, MPDU_BITS)),                   # 69 symbols -> 302
        ("24M OFDM", _ofdm(24, MPDU_BITS)),                 # 18 symbols -> 98
        ("54M OFDM", _ofdm(54, MPDU_BITS)),                 # 8 symbols -> 58
    ],
)
def test_other_rate_frame_times_by_hand(name, expected_us):
    mode = next(m for m in ab.MODES if m.name == name)
    # the DSSS/CCK rule rounds the payload time up to a whole microsecond, so the
    # answer is exact, not just close
    assert ab.data_us(PAYLOAD, mode) == pytest.approx(expected_us, abs=1e-9)


def test_cck_ack_rounds_up_too():
    mode = next(m for m in ab.MODES if m.name == "11M_S")
    # 14-byte ACK at 11 Mbit/s short preamble: 96 + ceil(112 / 11 = 10.18) = 107
    assert ab.ack_us(mode) == pytest.approx(96 + 11, abs=1e-9)
