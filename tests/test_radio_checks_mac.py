"""Independent checks for the medium access (MAC) simulator, written from math only.

Every expected value below is hand-derived from the Run C derivations (D5, D6, D7)
and the written API contract. None of it was read from, or tuned to, the code in
sim/radio/mac.py. The simulator talks to a fake channel defined in this file, so
the radio physics is out of the picture and only the MAC rules are tested.

If the module does not exist yet, every test except test_module_exists is skipped.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import numpy as np
import pytest


@pytest.fixture(scope="module")
def m():
    return pytest.importorskip("sim.radio.mac")


def test_module_exists():
    """The MAC module must be importable (plain import, no skip)."""
    importlib.import_module("sim.radio.mac")


# ---------------------------------------------------------------------------
# Fake channel
# ---------------------------------------------------------------------------


@dataclass
class _FakeSample:
    rssi_dbm: float
    snr_db: float
    per: float
    shadow_db: float = 0.0
    path_loss_db: float = 0.0


class FakeChannel:
    """A channel with fixed numbers: rssi -50 dBm, SNR 45 dB, chosen PER.

    Frames longer than 20 bytes are data frames (PER = per_data). Frames of 20
    bytes or less are ACKs (the contract's ACK is 14 bytes), with PER = per_ack.
    """

    sensitivity_dbm = -98.0

    def __init__(self, per_data: float = 0.0, per_ack: float = 0.0):
        self.per_data = per_data
        self.per_ack = per_ack

    def sample(self, tx, rx, distance_m, t_s, *, frame_bytes, walls=0, bodies=0,
               noise_floor_dbm=None, extra_loss_db=0.0):
        per = self.per_data if frame_bytes > 20 else self.per_ack
        return _FakeSample(rssi_dbm=-50.0, snr_db=45.0, per=per)

    def received_power_dbm(self, tx, rx, distance_m, t_s, *, walls=0, bodies=0,
                           extra_loss_db=0.0):
        return -50.0


def _positions(names=("a", "gs"), spacing_m=5.0):
    return {n: (spacing_m * i, 0.0, 0.0) for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# D6: airtime by hand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload,expected_s", [(32, 792e-6), (64, 1048e-6), (250, 2536e-6)])
def test_d6_dsss_data_airtime(m, payload, expected_s):
    """D6 (hand-derived): 802.11b, 1 Mbps, long preamble. MPDU = payload + 43 bytes.
    32 B: 75 B = 600 bits -> 600 us + 192 us = 792 us. 64 B: 107 B -> 856 + 192 = 1048 us.
    250 B: 293 B -> 2344 + 192 = 2536 us. Tolerance 1e-12 s."""
    phy = m.PhyConfig()
    assert m.data_airtime_s(payload, phy) == pytest.approx(expected_s, abs=1e-12)
    n_mpdu = payload + 43
    assert m.airtime_s(n_mpdu, phy.rate_mbps, phy) == pytest.approx(expected_s, abs=1e-12)


def test_d6_ack_airtime(m):
    """D6 (hand-derived): 14 B at 1 Mbps = 112 bits = 112 us + 192 us PLCP = 304 us."""
    assert m.ack_airtime_s(m.PhyConfig()) == pytest.approx(304e-6, abs=1e-12)


def test_d6_ofdm_airtime(m):
    """D6 (hand-derived): 802.11g, 6 Mbps, 4 us symbols carry 24 bits. 75 B -> 16 + 600 + 6 =
    622 bits -> ceil(622/24) = 26 symbols -> 20 us + 26 * 4 us = 124 us."""
    phy = m.PhyConfig(rate_mbps=6.0, modulation="ofdm")
    assert m.airtime_s(75, 6.0, phy) == pytest.approx(124e-6, abs=1e-12)
    assert m.data_airtime_s(32, phy) == pytest.approx(124e-6, abs=1e-12)


def test_d6_short_preamble_2mbps(m):
    """D6 (hand-derived): DSSS, short preamble (96 us), 2 Mbps, 75 B = 600 bits -> 300 us.
    Total 396 us."""
    phy = m.PhyConfig(rate_mbps=2.0, preamble="short")
    assert m.airtime_s(75, 2.0, phy) == pytest.approx(396e-6, abs=1e-12)
    assert m.data_airtime_s(32, phy) == pytest.approx(396e-6, abs=1e-12)


def test_d6_difs_default(m):
    """D6 / contract (hand-derived): DIFS = SIFS + 2 slots = 10 us + 2 * 20 us = 50 us."""
    phy = m.PhyConfig()
    assert phy.sifs_s == pytest.approx(10e-6, abs=1e-15)
    assert phy.slot_s == pytest.approx(20e-6, abs=1e-15)
    assert phy.difs_s == pytest.approx(50e-6, abs=1e-12)


def test_d6_frame_bytes_limit(m):
    """Contract (hand-derived): frame_bytes = payload + 43; 250 B payload is allowed (293 B),
    251 B raises ValueError because max_payload_bytes is 250."""
    phy = m.PhyConfig()
    assert phy.max_payload_bytes == 250
    assert m.frame_bytes(250, phy) == 293
    assert m.frame_bytes(32, phy) == 75
    with pytest.raises(ValueError):
        m.frame_bytes(251, phy)
    with pytest.raises(ValueError):
        m.data_airtime_s(251, phy)


# ---------------------------------------------------------------------------
# D5: single sender, no contention
# ---------------------------------------------------------------------------


def _single_sender_sim(m, seed=1):
    phy = m.PhyConfig()
    mac = m.MacConfig(carrier_sense=True, backoff=False, immediate_access=True,
                      processing_delay_s=0.0007)
    sim = m.MacSimulator(_positions(), FakeChannel(), phy, mac, seed=seed)
    return sim, phy


def test_d5_single_sender_delay(m):
    """D5 (hand-derived): one broadcast of 32 B at t = 1.0 on an idle medium, backoff off.
    delay = DIFS (50 us) + data airtime (792 us) + processing (0.7 ms) = 1.542 ms, within 1e-9.

    The DIFS term reflects the rule that a frame must wait for the medium to have been idle
    for DIFS before sending. The contract's immediate_access wording says a frame reaching
    head of line after the medium has ALREADY been idle for >= DIFS goes out right away.
    If the module documents that rule (a frame arriving after a long idle period does not
    wait a further DIFS), the delay would be 0.7 ms + 792 us and this check will flag the
    difference on purpose, so the rule is a visible, deliberate choice and not an accident."""
    sim, phy = _single_sender_sim(m)
    f = m.Frame(src="a", dst=m.BROADCAST, payload_bytes=32, created_s=1.0)
    fid = sim.enqueue("a", f, 1.0)
    sim.run_until(5.0)
    recs = [r for r in sim.rx_records if r.frame_id == fid and r.rx == "gs"]
    assert len(recs) == 1
    r = recs[0]
    assert r.outcome == "delivered"
    expected = phy.difs_s + 792e-6 + 0.0007
    assert r.deliver_s - r.created_s == pytest.approx(expected, abs=1e-9)
    assert r.tx_end_s - r.tx_start_s == pytest.approx(792e-6, abs=1e-9)
    assert r.deliver_s - r.tx_end_s == pytest.approx(0.0007, abs=1e-9)
    fr = sim.frame_records[fid]
    assert fr.status == "sent" and fr.tries == 1


def test_d5_single_sender_chain_identical_delays(m):
    """D5 (hand-derived): 10 broadcasts enqueued 0.1 s apart (far apart compared with the
    ~1.5 ms frame time), each on an idle medium, so every delay is the same
    DIFS + 792 us + 0.7 ms = 1.542 ms (within 1e-9, same DIFS-rule caveat as the single-frame
    check). The spread between delays must be below 1e-9."""
    sim, phy = _single_sender_sim(m, seed=2)
    ids = []
    for k in range(10):
        t = 1.0 + 0.1 * k
        ids.append(sim.enqueue("a", m.Frame(src="a", dst=m.BROADCAST, payload_bytes=32,
                                            created_s=t), t))
    sim.run_until(10.0)
    expected = phy.difs_s + 792e-6 + 0.0007
    delays = []
    for fid in ids:
        recs = [r for r in sim.rx_records if r.frame_id == fid and r.rx == "gs"]
        assert len(recs) == 1 and recs[0].outcome == "delivered"
        delays.append(recs[0].deliver_s - recs[0].created_s)
    assert max(delays) - min(delays) < 1e-9
    for d in delays:
        assert d == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# D5: M/D/1 through the MAC
# ---------------------------------------------------------------------------

_MD1_CACHE: dict = {}


def _md1_run(m, rho, n_arrivals, seed):
    key = (rho, n_arrivals, seed)
    if key in _MD1_CACHE:
        return _MD1_CACHE[key]
    phy = m.PhyConfig()
    # immediate_access=False so EVERY frame waits one DIFS: service time is the constant
    # D = DIFS + data airtime. (With immediate access, a frame on an idle medium skips the
    # DIFS and the service time is no longer constant, so M/D/1 would not apply.)
    mac = m.MacConfig(carrier_sense=True, backoff=False, immediate_access=False,
                      queue_capacity=100000)
    big_d = phy.difs_s + m.data_airtime_s(32, phy)
    lam = rho / big_d
    sim = m.MacSimulator(_positions(), FakeChannel(), phy, mac, seed=seed)
    stop_s = n_arrivals / lam
    sim.add_poisson_source("a", m.BROADCAST, lam, 32, start_s=0.0, stop_s=stop_s)
    sim.run_until(stop_s + 5.0)
    recs = sorted((r for r in sim.frame_records.values() if r.src == "a"),
                  key=lambda r: r.frame_id)
    _MD1_CACHE[key] = (recs, big_d)
    return recs, big_d


def _batch_means_check(values, theory, n_batches=20, warmup_frac=0.05):
    v = np.asarray(values, dtype=float)
    v = v[int(len(v) * warmup_frac):]
    size = len(v) // n_batches
    b = v[: size * n_batches].reshape(n_batches, size).mean(axis=1)
    mean = b.mean()
    se = b.std(ddof=1) / math.sqrt(n_batches)
    return mean, se


@pytest.mark.parametrize("rho,n_arrivals", [(0.5, 100_000), (0.8, 200_000)])
def test_d5_md1_queue_wait(m, rho, n_arrivals):
    """D5 (hand-derived, Pollaczek-Khinchine for deterministic service):
        Wq = rho / (2 mu (1 - rho)) = rho * D / (2 (1 - rho)),  mu = 1/D,
    with D = DIFS + data airtime(32 B) = 50 us + 792 us = 842 us.
    rho = 0.5 -> Wq = 421 us; rho = 0.8 -> Wq = 1684 us.
    Mapping: queue_wait_s is the time from enqueue until the frame becomes head of line and
    its DIFS wait begins, so the service time (DIFS + airtime) is the constant D.
    Method: broadcast Poisson source, 20 batch means after a 5 % warmup, tolerance 4 batch-mean
    standard errors. Arrivals reduced from the 4 * 10^5 suggested to keep runtime low
    (1e5 at rho 0.5, 2e5 at rho 0.8); batch means keep the test honest about correlation."""
    recs, big_d = _md1_run(m, rho, n_arrivals, seed=11)
    assert len(recs) >= 0.99 * n_arrivals
    assert all(r.status == "sent" for r in recs)
    waits = [r.queue_wait_s for r in recs]
    assert all(w is not None and w >= -1e-12 for w in waits)
    theory = rho * big_d / (2.0 * (1.0 - rho))
    mean, se = _batch_means_check(waits, theory)
    assert abs(mean - theory) <= 4.0 * se, (mean, theory, se)


@pytest.mark.parametrize("rho,n_arrivals", [(0.5, 100_000), (0.8, 200_000)])
def test_d5_md1_wait_from_first_tx(m, rho, n_arrivals):
    """D5 (hand-derived): same M/D/1 run, but waiting time rebuilt from the timestamps:
    (first_tx_s - created_s) - DIFS, because with backoff off the only thing between
    becoming head of line and sending is one DIFS. This does not depend on how the module
    defines queue_wait_s. Theory and tolerance as in the queue_wait_s check."""
    recs, big_d = _md1_run(m, rho, n_arrivals, seed=11)
    difs = big_d - 792e-6
    waits = [r.first_tx_s - r.created_s - difs for r in recs]
    theory = rho * big_d / (2.0 * (1.0 - rho))
    mean, se = _batch_means_check(waits, theory)
    assert abs(mean - theory) <= 4.0 * se, (mean, theory, se)


# ---------------------------------------------------------------------------
# D7: retries
# ---------------------------------------------------------------------------


def _exact_distribution(q, a, k):
    """Exact tree of per-try outcomes. Each try: data lost (prob q), or data arrives but the
    ACK is lost (prob (1-q) a), or both arrive (prob (1-q)(1-a), sender stops). The sender
    stops on the first success or after k+1 tries. Returns (prob, tries, deliveries, acked)."""
    out = []
    stack = [(1.0, 0, 0)]
    while stack:
        p, t, d = stack.pop()
        t += 1
        for pp, dd, ok in ((q, 0, False), ((1 - q) * a, 1, False), ((1 - q) * (1 - a), 1, True)):
            if pp == 0.0:
                continue
            if ok:
                out.append((p * pp, t, d + dd, True))
            elif t == k + 1:
                out.append((p * pp, t, d + dd, False))
            else:
                stack.append((p * pp, t, d + dd))
    return out


def _mean_var(dist, fn):
    mean = sum(d[0] * fn(d) for d in dist)
    var = sum(d[0] * fn(d) ** 2 for d in dist) - mean ** 2
    return mean, var


def test_d7_helper_matches_closed_form():
    """D7 (hand-derived): with no ACK loss the exact tree must reproduce the closed forms
    1 - q^(k+1) and (1 - q^(k+1)) / (1 - q). For q = 0.3, k = 3: 0.9919 and 1.4170."""
    dist = _exact_distribution(0.3, 0.0, 3)
    assert sum(d[0] for d in dist) == pytest.approx(1.0, abs=1e-12)
    acked = sum(d[0] for d in dist if d[3])
    assert acked == pytest.approx(1 - 0.3 ** 4, abs=1e-12)
    mean_tries, _ = _mean_var(dist, lambda d: d[1])
    assert mean_tries == pytest.approx((1 - 0.3 ** 4) / (1 - 0.3), abs=1e-12)
    assert mean_tries == pytest.approx(1.417, abs=1e-12)


_RETRY_CACHE: dict = {}


def _retry_run(m, per_data, per_ack, k, n=20_000, seed=21):
    key = (per_data, per_ack, k, n, seed)
    if key in _RETRY_CACHE:
        return _RETRY_CACHE[key]
    phy = m.PhyConfig()
    mac = m.MacConfig(carrier_sense=True, backoff=True, unicast_retry_limit=k,
                      queue_capacity=1000)
    sim = m.MacSimulator(_positions(), FakeChannel(per_data, per_ack), phy, mac, seed=seed)
    ids = []
    for i in range(n):
        t = 0.05 * (i + 1)
        ids.append(sim.enqueue("a", m.Frame(src="a", dst="gs", payload_bytes=32, created_s=t), t))
    sim.run_until(0.05 * (n + 1) + 10.0)
    deliveries = {fid: 0 for fid in ids}
    for r in sim.rx_records:
        if r.rx == "gs" and r.dst == "gs" and r.outcome == "delivered" and r.frame_id in deliveries:
            deliveries[r.frame_id] += 1
    out = (sim, ids, np.array([deliveries[fid] for fid in ids]))
    _RETRY_CACHE[key] = out
    return out


def _binom_sd(p, n):
    return math.sqrt(p * (1 - p) / n)


def test_d7_retries_acked_fraction(m):
    """D7 (hand-derived): data loss q = 0.3 per try, ACKs never lost, k = 3 retries (4 tries).
    P(acked) = 1 - q^4 = 1 - 0.0081 = 0.9919. Frames sent 50 ms apart (no contention),
    n = 2 * 10^4. Tolerance 4 binomial standard deviations (sd about 6.3e-4)."""
    n = 20_000
    sim, ids, deliveries = _retry_run(m, 0.3, 0.0, 3, n)
    statuses = [sim.frame_records[i].status for i in ids]
    assert set(statuses) <= {"acked", "retry_exhausted"}
    frac = statuses.count("acked") / n
    p = 1 - 0.3 ** 4
    assert abs(frac - p) <= 4 * _binom_sd(p, n), (frac, p)


def test_d7_retries_mean_tries_and_no_duplicates(m):
    """D7 (hand-derived): mean tries = 1 + q + q^2 + q^3 = 1.4170 (q = 0.3, k = 3), with the
    exact variance of the truncated geometric (P(T=1..3) = .7, .21, .063; P(T=4) = .027):
    E[T^2] = 2.539, Var = 0.5311, so the standard error over 2 * 10^4 frames is 5.2e-3.
    Tolerance 4 standard errors. With no ACK loss the receiver gets each frame at most once
    (the sender stops at the first ACK), so there must be zero duplicate deliveries, and
    the fraction delivered at least once equals 0.9919 within 4 binomial sd."""
    n = 20_000
    sim, ids, deliveries = _retry_run(m, 0.3, 0.0, 3, n)
    tries = np.array([sim.frame_records[i].tries for i in ids], dtype=float)
    dist = _exact_distribution(0.3, 0.0, 3)
    mean, var = _mean_var(dist, lambda d: d[1])
    assert mean == pytest.approx(1.417, abs=1e-12)
    assert var == pytest.approx(0.531111, abs=1e-5)
    assert abs(tries.mean() - mean) <= 4 * math.sqrt(var / n), (tries.mean(), mean)
    assert tries.min() >= 1 and tries.max() <= 4
    assert int((deliveries >= 2).sum()) == 0
    p = 1 - 0.3 ** 4
    assert abs((deliveries >= 1).mean() - p) <= 4 * _binom_sd(p, n)


def test_d7_zero_retries(m):
    """D7 (hand-derived): k = 0 (a single try), q = 0.3, ACK never lost: P(acked) = 1 - q = 0.7,
    every frame has exactly 1 try. n = 2 * 10^4, tolerance 4 binomial sd (sd about 3.2e-3)."""
    n = 20_000
    sim, ids, deliveries = _retry_run(m, 0.3, 0.0, 0, n)
    recs = [sim.frame_records[i] for i in ids]
    assert all(r.tries == 1 for r in recs)
    assert {r.status for r in recs} <= {"acked", "retry_exhausted"}
    frac = sum(r.status == "acked" for r in recs) / n
    assert abs(frac - 0.7) <= 4 * _binom_sd(0.7, n), frac


def test_d7_ack_loss_duplicates_and_at_least_once(m):
    """D7 (hand-derived): data loss q = 0.3 AND ACK loss a = 0.3, k = 3 (4 tries). Each try
    ends in one of three ways: data lost (0.3), data arrives but ACK lost (0.7 * 0.3 = 0.21),
    both arrive (0.7 * 0.7 = 0.49, sender stops). The tree is enumerated exactly.
    - Receiver gets the frame at least once with prob 1 - q^4 = 0.9919 (no delivery means no
      ACK, so all four tries run, each failing with prob 0.3). Tolerance 4 binomial sd.
    - Sender sees success with prob 1 - 0.51^4 = 0.9323. Tolerance 4 binomial sd.
    - Duplicates exist: P(delivered twice or more) is about 0.30 from the tree, and total
      deliveries exceed the number of distinct frames delivered.
    - Mean tries and mean deliveries per frame match the exact tree within 4 standard errors."""
    n = 20_000
    sim, ids, deliveries = _retry_run(m, 0.3, 0.3, 3, n)
    dist = _exact_distribution(0.3, 0.3, 3)
    assert sum(d[0] for d in dist) == pytest.approx(1.0, abs=1e-12)

    p_once = 1 - 0.3 ** 4
    assert abs((deliveries >= 1).mean() - p_once) <= 4 * _binom_sd(p_once, n)

    p_ack = 1 - 0.51 ** 4
    exact_ack = sum(d[0] for d in dist if d[3])
    assert exact_ack == pytest.approx(p_ack, abs=1e-12)
    acked = np.array([sim.frame_records[i].status == "acked" for i in ids])
    assert abs(acked.mean() - p_ack) <= 4 * _binom_sd(p_ack, n)

    distinct = int((deliveries >= 1).sum())
    assert int(deliveries.sum()) > distinct
    p_dup, _ = _mean_var(dist, lambda d: 1.0 if d[2] >= 2 else 0.0)
    dup = (deliveries >= 2).mean()
    assert dup > 0.0
    assert abs(dup - p_dup) <= 4 * _binom_sd(p_dup, n), (dup, p_dup)

    tries = np.array([sim.frame_records[i].tries for i in ids], dtype=float)
    mt, vt = _mean_var(dist, lambda d: d[1])
    assert abs(tries.mean() - mt) <= 4 * math.sqrt(vt / n), (tries.mean(), mt)
    md, vd = _mean_var(dist, lambda d: d[2])
    assert abs(deliveries.mean() - md) <= 4 * math.sqrt(vd / n), (deliveries.mean(), md)


# ---------------------------------------------------------------------------
# Broadcast and conservation
# ---------------------------------------------------------------------------


def test_broadcast_never_retries(m):
    """Contract (hand-derived rule, no statistics): a broadcast frame is never ACKed and never
    retried, even when unicast_retry_limit = 3 and the channel loses 30 % of data frames.
    Every broadcast FrameRecord has tries == 1 and status "sent"; every RxRecord has
    attempt == 0. Three nodes, low load, no queue drops."""
    phy = m.PhyConfig()
    mac = m.MacConfig(unicast_retry_limit=3, queue_capacity=10000)
    names = ("a", "b", "gs")
    sim = m.MacSimulator(_positions(names), FakeChannel(0.3, 0.3), phy, mac, seed=5)
    for node in ("a", "b"):
        sim.add_poisson_source(node, m.BROADCAST, 20.0, 32, start_s=0.0, stop_s=20.0)
    sim.run_until(30.0)
    recs = list(sim.frame_records.values())
    assert len(recs) > 400
    assert all(r.status == "sent" for r in recs)
    assert all(r.tries == 1 for r in recs)
    assert all(r.attempt == 0 for r in sim.rx_records)


def test_conservation_one_terminal_status_per_frame(m):
    """Conservation (hand-derived rule): every enqueued frame ends with exactly one
    FrameRecord in exactly one terminal status after the event queue drains. Mixed unicast
    and broadcast traffic from three nodes into a tiny queue (capacity 3) at a rate above
    the channel capacity forces queue drops, and a lossy channel (30 % data and ACK loss,
    k = 2) forces retry exhaustion. Checks: ids returned by enqueue map to distinct records,
    no record is "pending", unicast ends acked / retry_exhausted / queue_drop, broadcast ends
    sent / queue_drop, per-status counts sum to the number of records, and tries is within
    1..k+1 for every frame that was not dropped from the queue."""
    k = 2
    phy = m.PhyConfig()
    mac = m.MacConfig(unicast_retry_limit=k, queue_capacity=3)
    names = ("a", "b", "gs")
    sim = m.MacSimulator(_positions(names), FakeChannel(0.3, 0.3), phy, mac, seed=9)
    manual = {}
    for i in range(900):
        node = ("a", "b")[i % 2]
        dst = "gs" if i % 3 else m.BROADCAST
        t = 0.001 * (i // 2)
        fid = sim.enqueue(node, m.Frame(src=node, dst=dst, payload_bytes=32, created_s=t), t)
        manual[fid] = dst
    sim.add_poisson_source("gs", "a", 400.0, 32, start_s=0.0, stop_s=1.0)
    sim.add_poisson_source("a", m.BROADCAST, 400.0, 32, start_s=0.0, stop_s=1.0)
    sim.run_until(1.0e4)

    assert len(set(manual)) == len(manual)
    assert set(manual) <= set(sim.frame_records)
    terminal_ok = {"acked", "sent", "queue_drop", "retry_exhausted"}
    counts: dict[str, int] = {}
    for fid, r in sim.frame_records.items():
        assert r.frame_id == fid
        assert r.status in terminal_ok, (fid, r.status)
        counts[r.status] = counts.get(r.status, 0) + 1
        if r.dst == m.BROADCAST:
            assert r.status in {"sent", "queue_drop"}
        else:
            assert r.status in {"acked", "retry_exhausted", "queue_drop"}
        if r.status != "queue_drop":
            assert 1 <= r.tries <= k + 1
            if r.dst == m.BROADCAST:
                assert r.tries == 1
    assert sum(counts.values()) == len(sim.frame_records)
    assert counts.get("queue_drop", 0) > 0
    assert counts.get("retry_exhausted", 0) > 0
    assert counts.get("acked", 0) > 0 and counts.get("sent", 0) > 0
