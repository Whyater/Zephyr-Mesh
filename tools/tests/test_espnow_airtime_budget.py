"""Tests for tools/espnow_airtime_budget.py.

Expected frame times are worked by hand from the 802.11 timing rules written in the
comments, not read back from the tool.  The engine runs (ALOHA and DCF) are checked
against the closed-form ALOHA formula and against limiting cases.
"""
from __future__ import annotations

import json
import math

import pytest

from sim.radio.packet import STATE_PACKET_BYTES
from sim.radio.params import PARAMS
from tools import espnow_airtime_budget as ab


def mode(name):
    return next(m for m in ab.MODES if m.name == name)


def test_overhead_is_the_espressif_frame_table_summed_by_hand():
    # 24 MAC header + 1 category + 3 OUI + 4 random + 7 vendor element header + 4 FCS
    assert ab.espnow_overhead_bytes() == 24 + 1 + 3 + 4 + 7 + 4 == 43
    assert ab.espnow_overhead_bytes() == PARAMS["espnow_overhead_bytes"].value
    assert STATE_PACKET_BYTES == 161


def test_state_frame_at_default_rate_by_hand():
    # 161 + 43 = 204 bytes = 1632 bits at 1 bit/us, plus the 192 us long preamble and header
    assert ab.data_us(161, ab.DEFAULT_MODE) == pytest.approx(192 + 1632)
    # ACK: 14 bytes = 112 bits at 1 Mbit/s plus 192 us
    assert ab.ack_us(ab.DEFAULT_MODE) == pytest.approx(304)
    ex = ab.exchange(161, ab.DEFAULT_MODE, "unicast_gs")
    assert ex.busy_us == pytest.approx(1824 + 10 + 304)
    # DIFS = SIFS + 2 slots = 10 + 40; mean backoff = 31 / 2 slots of 20 us = 310
    assert ex.access_us == pytest.approx(50 + 310)
    assert ab.exchange(161, ab.DEFAULT_MODE, "broadcast").sifs_ack_us == 0


def test_other_rates_by_hand():
    # 11 Mbit/s short preamble: 96 + 1632 / 11 = 96 + 148.36, rounded up to a whole us
    assert ab.data_us(161, mode("11M_S")) == pytest.approx(96 + 149)
    # 2 Mbit/s short preamble: 96 + 816; ACK 96 + 112 / 2 = 152
    assert ab.data_us(161, mode("2M_S")) == pytest.approx(912)
    assert ab.ack_us(mode("2M_S")) == pytest.approx(152)
    # 6 Mbit/s OFDM: 24 data bits per 4 us symbol; 22 + 1632 = 1654 bits -> 69 symbols
    # 20 us preamble+SIGNAL + 276 us + 6 us signal extension
    assert ab.data_us(161, mode("6M OFDM")) == pytest.approx(20 + 69 * 4 + 6)
    # 54 Mbit/s: 216 bits per symbol, 1654 / 216 = 7.66 -> 8 symbols
    assert ab.data_us(161, mode("54M OFDM")) == pytest.approx(20 + 8 * 4 + 6)
    # its ACK goes at 24 Mbit/s: 22 + 112 = 134 bits / 96 -> 2 symbols
    assert ab.ack_us(mode("54M OFDM")) == pytest.approx(20 + 2 * 4 + 6)


def test_ack_rate_rule():
    got = {m.name: m.ack_rate_mbps for m in ab.MODES}
    assert got == {"1M_L (default)": 1.0, "2M_S": 2.0, "11M_S": 11.0,
                   "6M OFDM": 6.0, "24M OFDM": 24.0, "54M OFDM": 24.0}


def test_short_preamble_at_1mbps_is_rejected():
    with pytest.raises(ValueError):
        ab.PhyMode("bad", "dsss", 1.0, True)


def test_hand_checked_cell_eight_drones_50hz_broadcast():
    row = ab.cell(8, 50.0, "broadcast")
    assert row["frames_per_s"] == 400
    assert row["utilization"] == pytest.approx(400 * 1824e-6)  # 0.7296
    # one other drone overlaps with chance 2 * 1824 us * 50 Hz = 0.1824; seven others
    assert row["aloha_collision"] == pytest.approx(1 - (1 - 0.1824) ** 7)
    assert row["aloha_collision"] == pytest.approx(0.7558, abs=5e-4)


def test_aloha_limits():
    assert ab.aloha_collision(1, 50.0, 2000.0)["periodic"] == 0.0
    a = ab.aloha_collision(5, 25.0, 2138.0)
    # (1 - x) ** k <= exp(-k x), so the periodic form is never below the Poisson form
    assert a["periodic"] >= a["poisson"] > 0
    # tiny load: both reduce to 2 (n - 1) f D
    tiny = ab.aloha_collision(3, 1.0, 1.0)
    assert tiny["periodic"] == pytest.approx(2 * 2 * 1.0 * 1e-6, rel=1e-5)
    with pytest.raises(ValueError):
        ab.aloha_collision(0, 10.0, 100.0)


def test_unicast_holds_the_air_longer_than_broadcast():
    b = ab.cell(5, 25.0, "broadcast")
    u = ab.cell(5, 25.0, "unicast_gs")
    assert u["utilization"] - b["utilization"] == pytest.approx(125 * (10 + 304) * 1e-6)
    assert u["aloha_collision"] > b["aloha_collision"]


def test_crosscheck_dsss_matches_sim_radio_and_ofdm_differs_by_signal_extension():
    for r in ab.crosscheck_sim_radio():
        if "OFDM" in r["mode"]:
            # sim/radio/mac.py leaves out the 6 us ERP signal extension
            assert r["data_us_here"] - r["data_us_sim_radio"] == pytest.approx(6.0)
            assert r["ack_us_here"] - r["ack_us_sim_radio"] == pytest.approx(6.0)
        else:
            # this file rounds DSSS/CCK frame time up to a whole microsecond and
            # sim/radio/mac.py does not, so they can differ by less than 1 us
            for key in ("data", "ack"):
                gap = r[f"{key}_us_here"] - r[f"{key}_us_sim_radio"]
                assert 0.0 <= gap < 1.0


def test_engine_aloha_matches_the_closed_form():
    # 3 drones at 50 Hz, broadcast: closed form 1 - (1 - 0.1824) ** 2 = 0.3315
    sim = ab.simulate_collision(3, 50.0, "broadcast", carrier_sense=False, sim_seconds=20.0, seed=5)
    expect = 1 - (1 - 0.1824) ** 2
    sd = math.sqrt(expect * (1 - expect) / sim["frames"])
    assert sim["frames"] > 2700
    assert abs(sim["loss_fraction"] - expect) < 4 * sd


def test_engine_dcf_limits():
    lone = ab.simulate_collision(1, 50.0, "unicast_gs", carrier_sense=True, sim_seconds=4.0)
    assert lone["lost"] == 0 and lone["frames"] > 100
    # alone on the air, every frame goes right after one DIFS = 10 + 2 x 20 us
    assert lone["wait_mean_ms"] == pytest.approx(0.050, abs=1e-6)
    assert lone["wait_p95_ms"] == pytest.approx(0.050, abs=1e-6)
    dcf = ab.simulate_collision(5, 25.0, "broadcast", carrier_sense=True, sim_seconds=10.0)
    # listening before talking must beat the no-listening bound by a wide margin
    assert dcf["loss_fraction"] < 0.5 * ab.cell(5, 25.0, "broadcast")["aloha_collision"]
    # the engine's busy share matches the closed-form utilization (frames do not queue up)
    assert dcf["busy_fraction"] == pytest.approx(ab.cell(5, 25.0, "broadcast")["utilization"], rel=0.1)


def test_main_writes_json(tmp_path, capsys):
    out = tmp_path / "budget.json"
    assert ab.main(["--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["table_1M"]) == 2 * 4 * 3
    assert data["payload_bytes"] == 161
    for c in data["constants"].values():
        assert c["source"] and c["status"] in ("CONFIRMED", "standard", "modelling-choice", "UNCONFIRMED")
    assert "1 Mbit/s long preamble" in capsys.readouterr().out
