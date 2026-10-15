import math

import numpy as np

from sim.radio.mac import (
    BROADCAST,
    Frame,
    MacConfig,
    MacSimulator,
    PhyConfig,
    airtime_s,
    ack_airtime_s,
    data_airtime_s,
    frame_bytes,
)


class _Sample:
    def __init__(self, rssi=-50.0, per=0.0):
        self.rssi_dbm = rssi
        self.snr_db = rssi + 95.0
        self.per = per


class FakeChannel:
    sensitivity_dbm = -98.0

    def __init__(self, per=0.0, ack_per=0.0, rssi=-50.0):
        self.per = per
        self.ack_per = ack_per
        self.rssi = rssi

    def sample(self, tx, rx, distance_m, t_s, *, frame_bytes, walls=0, bodies=0,
               noise_floor_dbm=None, extra_loss_db=0.0):
        p = self.ack_per if frame_bytes == 14 else self.per
        return _Sample(self.rssi, p)

    def received_power_dbm(self, tx, rx, distance_m, t_s, *, walls=0, bodies=0,
                           extra_loss_db=0.0):
        return self.rssi


def _sim(nodes=("a", "b"), channel=None, mac=None, seed=1):
    positions = {n: (float(i), 0.0, 0.0) for i, n in enumerate(nodes)}
    return MacSimulator(positions, channel or FakeChannel(), PhyConfig(),
                        mac or MacConfig(backoff=False), seed=seed)


def test_d6_airtime_hand_numbers():
    p = PhyConfig()
    assert frame_bytes(32, p) == 75
    assert airtime_s(75, 1.0, p) == 792e-6
    assert math.isclose(airtime_s(107, 1.0, p), 1048e-6, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(airtime_s(293, 1.0, p), 2536e-6, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(ack_airtime_s(p), 304e-6, rel_tol=0, abs_tol=1e-15)
    ofdm = PhyConfig(modulation="ofdm", rate_mbps=6.0)
    assert math.isclose(airtime_s(75, 6.0, ofdm), 124e-6, rel_tol=0, abs_tol=1e-15)
    short = PhyConfig(rate_mbps=2.0, preamble="short")
    assert math.isclose(airtime_s(75, 2.0, short), 396e-6, rel_tol=0, abs_tol=1e-15)


def test_no_contention_delay_is_difs_plus_data():
    p = PhyConfig()
    delay = 37e-6
    sim = MacSimulator({"a": (0, 0, 0), "b": (1, 0, 0)}, FakeChannel(), p,
                       MacConfig(backoff=False, processing_delay_s=delay), seed=2)
    sim.enqueue("a", Frame("a", "b", 32, 0), 0)
    sim.run_until(0.01)
    rx = sim.rx_records[0]
    assert rx.deliver_s == p.difs_s + data_airtime_s(32, p) + delay


def test_two_simultaneous_aloha_senders_collide():
    mac = MacConfig(carrier_sense=False, backoff=False)
    sim = _sim(nodes=("a", "b", "c"), mac=mac)
    sim.enqueue("a", Frame("a", "c", 32, 0), 0)
    sim.enqueue("b", Frame("b", "c", 32, 0), 0)
    sim.run_until(0.01)
    assert {r.outcome for r in sim.rx_records if r.rx == "c"} == {"collision"}


def test_back_to_back_frames_do_not_collide():
    sim = _sim()
    p = PhyConfig()
    d = data_airtime_s(32, p)
    sim.enqueue("a", Frame("a", "b", 32, 0), 0)
    sim.enqueue("a", Frame("a", "b", 32, 1), p.difs_s + d)
    sim.run_until(0.01)
    assert [r.outcome for r in sim.rx_records] == ["delivered", "delivered"]


def test_broadcast_is_never_retried_and_queue_drops():
    mac = MacConfig(carrier_sense=False, backoff=False, queue_capacity=1)
    sim = _sim(nodes=("a", "b"), channel=FakeChannel(per=1.0), mac=mac)
    i0 = sim.enqueue("a", Frame("a", BROADCAST, 32, 0), 0)
    sim.enqueue("a", Frame("a", BROADCAST, 32, 0), 1e-6)
    i1 = sim.enqueue("a", Frame("a", BROADCAST, 32, 0), 2e-6)
    sim.run_until(0.01)
    assert sim.frame_records[i0].status == "sent"
    assert sim.frame_records[i0].tries == 1
    assert sim.frame_records[i1].status == "queue_drop"


def test_retries_match_independent_loss_probability():
    q = 0.3
    k = 3
    n = 20000
    sim = _sim(channel=FakeChannel(per=q), mac=MacConfig(backoff=False, unicast_retry_limit=k,
                                                         queue_capacity=n), seed=9)
    for i in range(n):
        sim.enqueue("a", Frame("a", "b", 1, 0, seq=i), 0)
    sim.run_until(100.0)
    delivered = sum(r.outcome == "delivered" for r in sim.rx_records)
    expected = 1.0 - q ** (k + 1)
    observed = delivered / n
    sd = math.sqrt(expected * (1 - expected) / n)
    assert abs(observed - expected) <= 4 * sd


def test_ack_loss_causes_duplicate_delivery():
    sim = _sim(channel=FakeChannel(per=0.0, ack_per=1.0),
               mac=MacConfig(backoff=False, unicast_retry_limit=1))
    sim.enqueue("a", Frame("a", "b", 32, 0), 0)
    sim.run_until(0.01)
    assert [r.outcome for r in sim.rx_records] == ["delivered", "delivered"]
    assert sim.frame_records[0].status == "retry_exhausted"


def test_incremental_run_and_seed_are_deterministic():
    def run(split):
        mac = MacConfig(backoff=True, carrier_sense=True)
        s = _sim(channel=FakeChannel(), mac=mac, seed=44)
        for i in range(8):
            s.enqueue("a", Frame("a", "b", 20, 0, seq=i), i * 0.001)
        if split:
            s.run_until(0.003)
        s.run_until(0.1)
        return s.rx_records, s.frame_records

    a = run(False)
    b = run(True)
    assert a == b
