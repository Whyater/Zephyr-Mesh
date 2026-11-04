"""Hand-worked checks that pin two airtime facts from the 802.11 standard.

1. ERP-OFDM signal extension (2.4 GHz 802.11g/n OFDM frames).
   TXTIME = T_PREAMBLE + T_SIGNAL + T_SYM x N_SYM + signal extension, where the
   signal extension is 6 us of silence after every 2.4 GHz OFDM frame. It lets the
   receiver finish decoding and still meet the 10 us SIFS of the 2.4 GHz band
   (5 GHz OFDM uses a 16 us SIFS instead). Sources: IEEE 802.11 ERP clause
   (clause 19 in 802.11-2007 and 802.11-2012), as quoted in the Linux mac80211
   ``ieee80211_frame_duration`` comment ("TXTIME = T_PREAMBLE + T_SIGNAL + T_SYM x
   N_SYM + Signal Ext"; "802.11g - 19.8.4: aSIFSTime = 10 usec + signal ext = 6
   usec") and IEEE 802.11 document 11-08/0981r1 ("signal extension time is added to
   the calculation of TXTIME"). The standard text itself was not opened.

2. The 1 Mbit/s unicast ceiling with the standard ACK.
   An ACK answers at the highest basic rate that is not faster than the data
   frame. For 1 Mbit/s data that can only be 1 Mbit/s, the lowest 802.11b rate,
   so the ACK is 14 B x 8 = 112 us plus the 192 us long preamble = 304 us.
   (The rate rule is the 802.11 control-response rate rule; the exact clause
   number was not opened, and whether the ESP32 follows it for ESP-NOW is
   UNCONFIRMED. 1 Mbit/s is the slowest rate, so the ACK cannot be slower.)
   A 250 B ESP-NOW payload is 250 + 43 = 293 B = 2344 us + 192 us = 2536 us on air.
   Data + SIFS + ACK = 2536 + 10 + 304 = 2850 us, so one sender can complete at
   most 1 / 2850 us = 350.9 unicast exchanges per second even with no DIFS and no
   backoff.
"""
from __future__ import annotations

import math

import pytest

from sim.radio import mac as m
from sim.radio.params import PARAMS, default_phy


def test_ofdm_6mbps_75_bytes_includes_signal_extension():
    # 16 SERVICE + 600 data + 6 tail = 622 bits; 24 bits per 4 us symbol -> 26 symbols
    # 20 us preamble+SIGNAL + 104 us + 6 us signal extension = 130 us
    phy = m.PhyConfig(rate_mbps=6.0, modulation="ofdm")
    assert m.airtime_s(75, 6.0, phy) == pytest.approx(130e-6, abs=1e-12)
    assert m.data_airtime_s(32, phy) == pytest.approx(130e-6, abs=1e-12)


def test_ofdm_54mbps_state_frame_and_24mbps_ack():
    # 204 B state frame: 22 + 1632 = 1654 bits / 216 per symbol -> 8 symbols: 20 + 32 + 6 = 58 us
    phy = m.PhyConfig(rate_mbps=54.0, modulation="ofdm", ack_rate_mbps=24.0)
    assert m.airtime_s(204, 54.0, phy) == pytest.approx(58e-6, abs=1e-12)
    # 14 B ACK at 24 Mbit/s: 22 + 112 = 134 bits / 96 -> 2 symbols: 20 + 8 + 6 = 34 us
    assert m.ack_airtime_s(phy) == pytest.approx(34e-6, abs=1e-12)


def test_dsss_airtime_has_no_signal_extension():
    # Limiting case: the extension belongs to OFDM only. 204 B at 1 Mbit/s long: 192 + 1632 = 1824 us
    phy = default_phy()
    assert m.airtime_s(204, 1.0, phy) == pytest.approx(1824e-6, abs=1e-12)
    short = m.PhyConfig(rate_mbps=11.0, preamble="short")
    assert m.airtime_s(204, 11.0, short) == pytest.approx(96e-6 + 1632e-6 / 11, abs=1e-12)


def test_signal_extension_is_the_same_6us_at_every_ofdm_rate():
    # Pull the extension out by subtraction: airtime minus preamble minus whole symbols.
    for rate in (6.0, 9.0, 12.0, 18.0, 24.0, 36.0, 48.0, 54.0):
        phy = m.PhyConfig(rate_mbps=rate, modulation="ofdm")
        symbols = math.ceil((16 + 8 * 204 + 6) / (4 * rate))
        assert m.airtime_s(204, rate, phy) - (20e-6 + 4e-6 * symbols) == pytest.approx(6e-6, abs=1e-12)


def _unicast_ceiling_pps(payload_bytes: int) -> float:
    phy = default_phy()
    exchange = m.data_airtime_s(payload_bytes, phy) + phy.sifs_s + m.ack_airtime_s(phy)
    return 1.0 / exchange


def test_one_mbps_unicast_ceiling_by_hand():
    phy = default_phy()
    assert phy.rate_mbps == 1.0 and phy.ack_rate_mbps == 1.0 and phy.modulation == "dsss"
    assert m.data_airtime_s(250, phy) == pytest.approx(2536e-6, abs=1e-12)
    assert m.ack_airtime_s(phy) == pytest.approx(304e-6, abs=1e-12)
    assert _unicast_ceiling_pps(250) == pytest.approx(1 / 2850e-6)  # 350.877 per second


def test_published_250b_unicast_check_value_is_flagged_above_the_ceiling():
    """The published 1 Mbit/s unicast figure (357 per second) is above the 350.9 ceiling.

    It cannot be a valid check for the airtime model as labelled, so the entry must
    say so (status ``contradicted``) and keep the published number untouched.
    """
    v = PARAMS["measured_throughput_250b_pps"]
    assert v.value == 357  # the source's number (89250 B/s / 250 B), unchanged
    ceiling = _unicast_ceiling_pps(250)
    assert v.value > ceiling
    assert v.status == "contradicted"
    assert "350.9" in v.note


def test_every_packets_per_second_check_value_respects_the_unicast_ceiling_or_is_flagged():
    for name, v in PARAMS.items():
        if v.units != "packets/s" or v.value is None:
            continue
        if v.value > _unicast_ceiling_pps(250):
            assert v.status == "contradicted", name
