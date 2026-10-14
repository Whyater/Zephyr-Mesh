"""Channel-access rules of the MAC engine, each checked against a hand-derived value.

These pin four 802.11 DCF behaviours that decide how much contention costs:
same-slot collisions, deferring to ACKs, keeping backoff progress, and backing
off between back-to-back frames.  Expected values come from counting slots by
hand, not from the engine.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim.radio import mac as m
from sim.radio.channel import Channel, ChannelConfig

PHY = m.PhyConfig()
DIFS = PHY.difs_s
SLOT = PHY.slot_s
DATA_32 = m.data_airtime_s(32, PHY)          # 792 us at 1 Mbps, long preamble
ACK = m.ack_airtime_s(PHY)                    # 304 us


def _sim(positions, mac_cfg, seed, phy=PHY):
    channel = Channel(ChannelConfig(), np.random.default_rng(0))
    return m.MacSimulator(positions, channel, phy, mac_cfg, seed=seed)


def test_same_slot_backoff_collides_with_probability_one_over_cw_plus_one():
    # Two senders each owe a backoff drawn uniformly from {0..31} after the same
    # DIFS.  They collide only if they draw the same slot: P = 32 * (1/32)^2 = 1/32.
    # A larger draw freezes when the smaller one starts, so it never collides later.
    positions = {"a": (0.0, 0.0, 1.0), "b": (1.0, 0.0, 1.0), "gs": (0.5, 1.0, 1.0)}
    cfg = m.MacConfig(immediate_access=False)
    n_trials = 4000
    collided = 0
    for seed in range(n_trials):
        sim = _sim(positions, cfg, seed)
        sim.enqueue("a", m.Frame("a", m.BROADCAST, 32, 0.0), 0.0)
        sim.enqueue("b", m.Frame("b", m.BROADCAST, 32, 0.0), 0.0)
        sim.run_until(0.05)
        at_gs = [r.outcome for r in sim.rx_records if r.rx == "gs"]
        assert len(at_gs) == 2
        collided += at_gs[0] == "collision"
        assert (at_gs[0] == "collision") == (at_gs[1] == "collision")
    p = 1 / 32
    sd = math.sqrt(n_trials * p * (1 - p))
    assert abs(collided - n_trials * p) <= 4 * sd


def test_node_defers_until_ack_ends_plus_difs():
    # a unicasts to b at t = 0 (immediate access: data starts at DIFS).  c gets a
    # frame while a's data is on air, so it must wait for the ACK
    # (data end + SIFS + ACK airtime) and then a full DIFS.  Backoff is off, so
    # the start time is exact.
    positions = {"a": (0.0, 0.0, 1.0), "b": (2.0, 0.0, 1.0), "c": (1.0, 1.0, 1.0)}
    cfg = m.MacConfig(backoff=False)
    sim = _sim(positions, cfg, seed=1)
    fa = sim.enqueue("a", m.Frame("a", "b", 32, 0.0), 0.0)
    sim.run_until(DIFS + 0.0002)
    fc = sim.enqueue("c", m.Frame("c", m.BROADCAST, 32, sim.now_s), sim.now_s)
    sim.run_until(0.05)
    data_end = DIFS + DATA_32
    ack_end = data_end + PHY.sifs_s + ACK
    assert sim.frame_records[fa].status == "acked"
    assert sim.frame_records[fa].tries == 1
    assert sim.frame_records[fc].first_tx_s == pytest.approx(ack_end + DIFS, abs=1e-9)


@pytest.mark.parametrize("seed", range(30))
def test_backoff_progress_survives_unsensed_traffic_and_new_arrivals(seed):
    # a owes a backoff (immediate access off).  Neither of these may move a's
    # transmission start: (1) c, 1000 m away and below the CCA threshold at a
    # (about -90 dBm < -82 dBm), sends a frame that ends during a's countdown;
    # (2) a second frame joins a's queue mid-countdown.  A long slot (200 us)
    # makes the countdown span c's whole transmission.
    phy = m.PhyConfig(slot_s=200e-6)
    positions = {"a": (0.0, 0.0, 1.0), "gs": (1.0, 0.0, 1.0), "c": (1000.0, 0.0, 1.0)}
    cfg = m.MacConfig(immediate_access=False)

    def first_start(with_c: bool, with_second: bool) -> float:
        sim = _sim(positions, cfg, seed, phy)
        fid = sim.enqueue("a", m.Frame("a", m.BROADCAST, 32, 0.0), 0.0)  # a draws first
        if with_c:
            sim.enqueue("c", m.Frame("c", m.BROADCAST, 32, 1e-4), 1e-4)
        if with_second:
            sim.run_until(phy.difs_s + 0.5 * phy.slot_s)
            sim.enqueue("a", m.Frame("a", m.BROADCAST, 32, sim.now_s), sim.now_s)
        sim.run_until(0.1)
        return sim.frame_records[fid].first_tx_s

    alone = first_start(False, False)
    assert first_start(True, False) == alone
    assert first_start(False, True) == alone


def test_back_to_back_frames_back_off_between_them():
    # Two frames queued at once.  The first goes after DIFS (immediate access).
    # The second owes a backoff of k slots, k uniform on {0..31}: start =
    # end1 + DIFS + k * slot, so E[k] = 15.5 and Var[k] = (32^2 - 1) / 12.
    positions = {"a": (0.0, 0.0, 1.0), "gs": (1.0, 0.0, 1.0)}
    cfg = m.MacConfig()
    ks = []
    for seed in range(2000):
        sim = _sim(positions, cfg, seed)
        f1 = sim.enqueue("a", m.Frame("a", m.BROADCAST, 32, 0.0), 0.0)
        f2 = sim.enqueue("a", m.Frame("a", m.BROADCAST, 32, 0.0), 0.0)
        sim.run_until(0.05)
        assert sim.frame_records[f1].first_tx_s == pytest.approx(DIFS, abs=1e-12)
        gap = sim.frame_records[f2].first_tx_s - (DIFS + DATA_32) - DIFS
        k = gap / SLOT
        assert abs(k - round(k)) < 1e-6 and 0 <= round(k) <= 31
        ks.append(round(k))
    sd_mean = math.sqrt((32 ** 2 - 1) / 12) / math.sqrt(len(ks))
    assert abs(np.mean(ks) - 15.5) <= 4 * sd_mean
