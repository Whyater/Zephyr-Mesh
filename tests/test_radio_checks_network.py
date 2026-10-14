"""Independent checks for sim.radio.network, written from the documented semantics and hand derivations.

Nothing here was derived by running the network code.  Every expected number has
its hand derivation in a comment next to it.

Fake channel.  A link graph decides who can hear whom.  A neighbour pair has
RSSI -40 dBm, SNR 55 dB and a packet error rate that the test chooses (0 unless a
test needs loss).  A non-neighbour pair has RSSI -150 dBm, which is far below the
sensitivity point minus 10 dB (-108 dBm), below the interference floor
(-100 dBm) and below the CCA threshold (-82 dBm): such nodes neither hear nor
sense nor disturb each other.  Positions are arbitrary distinct points because
the fake channel ignores distance.  Data frames and ACK frames are told apart by
frame size: an ACK is 14 bytes, a data frame with a 32 byte payload is 75 bytes.

Timing constants (D6, 802.11b DSSS 1 Mbps, long preamble):
  DIFS = SIFS + 2 slots = 10 + 2 * 20 = 50 us
  data airtime, 32 B payload: (32 + 43) bytes = 600 bits = 600 us, plus 192 us PLCP = 792 us
  ACK airtime: 14 bytes = 112 bits = 112 us, plus 192 us = 304 us

Why the exact flood checks are collision free (see each test for the details):
in a chain, or in any graph where every transmission is triggered by the end of
exactly one earlier transmission, transmissions happen one after another and
never overlap, so no copy is ever destroyed.  Where two forwarders do transmit at
the same instant (ring, grid) the test either shows that the destroyed copies
only reach nodes that already have the message, or derives by hand exactly which
copies are destroyed.
"""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

network = pytest.importorskip("sim.radio.network")
mac = pytest.importorskip("sim.radio.mac")

DIFS_S = 50e-6
DATA_AIR_32_S = 792e-6
PAYLOAD = 32
ACK_BYTES = 14
EXACT_S = 1e-9


# ---------------------------------------------------------------------------
# fake channel and helpers
# ---------------------------------------------------------------------------

class GraphChannel:
    """Connectivity from an explicit neighbour graph (see the module docstring)."""

    sensitivity_dbm = -98.0

    def __init__(self, edges, data_per=0.0, ack_per=0.0):
        self.links = {frozenset(e) for e in edges}
        self.data_per = float(data_per)
        self.ack_per = float(ack_per)
        self.cfg = SimpleNamespace(noise_floor_dbm=-95.0, sensitivity_dbm=-98.0)
        self._far = SimpleNamespace(rssi_dbm=-150.0, snr_db=-55.0, per=1.0, shadow_db=0.0, path_loss_db=160.0)

    def _linked(self, tx, rx):
        return frozenset((tx, rx)) in self.links

    def sample(self, tx, rx, distance_m, t_s, *, frame_bytes=0, walls=0, bodies=0,
               noise_floor_dbm=None, extra_loss_db=0.0):
        if not self._linked(tx, rx):
            return self._far
        per = self.ack_per if frame_bytes <= ACK_BYTES else self.data_per
        return SimpleNamespace(rssi_dbm=-40.0, snr_db=55.0, per=per, shadow_db=0.0, path_loss_db=50.0)

    def received_power_dbm(self, tx, rx, distance_m, t_s, *, walls=0, bodies=0, extra_loss_db=0.0):
        return -40.0 if self._linked(tx, rx) else -150.0


def chain_edges(n, prefix="n"):
    return [(f"{prefix}{i}", f"{prefix}{i + 1}") for i in range(n - 1)]


def adjacency(edges):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def bfs_dist(edges, src):
    """Graph distances from src, my own breadth first search."""
    adj = adjacency(edges)
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def make_network(edges, topology, *, seed=1, gateway=None, relay=None, ttl=4, gossip_g=0.5,
                 forward_delay_s=0.0, gateway_forwards=False, backoff=False,
                 processing_delay_s=0.0, retry_limit=0, data_per=0.0, ack_per=0.0, extra_nodes=()):
    nodes = sorted({n for e in edges for n in e} | set(extra_nodes))
    positions = {n: (3.0 * i, 0.0, 0.0) for i, n in enumerate(nodes)}
    channel = GraphChannel(edges, data_per=data_per, ack_per=ack_per)
    phy = mac.PhyConfig()
    mac_cfg = mac.MacConfig(carrier_sense=True, backoff=backoff, unicast_retry_limit=retry_limit,
                            processing_delay_s=processing_delay_s)
    kwargs = dict(topology=topology, gateway=gateway if gateway is not None else nodes[-1], ttl=ttl,
                  gossip_g=gossip_g, forward_delay_s=forward_delay_s, gateway_forwards=gateway_forwards)
    if relay is not None:
        kwargs["relay"] = relay
    net = network.Network(positions, channel, phy, mac_cfg, network.NetworkConfig(**kwargs), seed=seed)
    return net


def reached_by_message(net):
    out = {}
    for (origin, seq, rx), rec in net.receptions.items():
        out.setdefault((origin, seq), {})[rx] = rec
    return out


def assert_reception_keys_consistent(net):
    """The dict key must match the record, and no node may record its own message."""
    for (origin, seq, rx), rec in net.receptions.items():
        assert (rec.origin, rec.origin_seq, rec.rx) == (origin, seq, rx)
        assert rx != origin
    # distinct keys <=> distinct records per (origin, seq, rx)
    assert len(net.receptions) == len({(r.origin, r.origin_seq, r.rx) for r in net.receptions.values()})


# ---------------------------------------------------------------------------
# 1. flooding reach and hop count on a chain (D10)
# ---------------------------------------------------------------------------
#
# Chain n0 - n1 - n2 - n3 - n4, origin n0.  Graph distances: n1:1, n2:2, n3:3, n4:4.
#
# Why no collision can happen: n0 transmits alone.  n1 enqueues its copy only
# when it has received n0's frame, so n1's transmission starts after n0's ended.
# n2 transmits only after hearing n1, and so on.  Each transmission is triggered
# by the end of the previous one, so no two transmissions ever overlap, with or
# without random backoff.  Hidden terminals (n1 and n3 cannot sense each other)
# do not matter because they are never on air together.
#
# Forwarders: every reached node with distance < ttl (the rule "h < ttl"), and the
# gateway only if gateway_forwards.  Transmissions = 1 (origin) + forwarders.
#
# Cases (gateway, forwards, ttl) -> (reached node indices, data transmissions):
#   gateway n4, forwards True,  ttl 4: reached 1..4; forwarders n1,n2,n3 (n4 has d = 4, not < 4); tx = 1 + 3 = 4
#   gateway n4, forwards True,  ttl 6: reached 1..4; forwarders n1..n4; tx = 1 + 4 = 5
#   gateway n4, forwards False, ttl 6: reached 1..4; forwarders n1,n2,n3; tx = 1 + 3 = 4
#   gateway n1, forwards False, ttl 6: n1 hears the origin and is the gateway, does not forward;
#                                      nobody else hears anything; reached {1}; tx = 1
#   gateway n1, forwards True,  ttl 6: same as the first ttl 6 case; tx = 5

CHAIN_CASES = [
    ("n4", True, 4, {1, 2, 3, 4}, 4),
    ("n4", True, 6, {1, 2, 3, 4}, 5),
    ("n4", False, 6, {1, 2, 3, 4}, 4),
    ("n1", False, 6, {1}, 1),
    ("n1", True, 6, {1, 2, 3, 4}, 5),
]


@pytest.mark.parametrize("backoff", [False, True])
@pytest.mark.parametrize("gateway,forwards,ttl,reached_idx,expected_tx", CHAIN_CASES)
def test_flood_chain_reach_hops_and_transmissions(gateway, forwards, ttl, reached_idx, expected_tx, backoff):
    edges = chain_edges(5)
    net = make_network(edges, "flood", gateway=gateway, ttl=ttl, gateway_forwards=forwards,
                       backoff=backoff, seed=11)
    key = net.originate("n0", 0.0, PAYLOAD)
    assert key == ("n0", 0)
    net.run_until(0.5)

    dist = bfs_dist(edges, "n0")
    got = reached_by_message(net)[key] if net.receptions else {}
    assert set(got) == {f"n{i}" for i in reached_idx}
    for node, rec in got.items():
        assert rec.hops == dist[node]          # first copy arrives along the shortest path in a chain
        assert rec.hops <= ttl
        assert rec.created_s == 0.0
        assert rec.first_rx_s > 0.0
    assert ("n0", 0, "n0") not in net.receptions   # the origin never records its own message
    assert net.data_transmissions() == expected_tx
    assert net.forwarded_frames() == expected_tx - 1   # every frame except the origin's own is a forward
    assert_reception_keys_consistent(net)
    # one Reception per reached node, nothing else
    assert len(net.receptions) == len(reached_idx)


# ---------------------------------------------------------------------------
# 1b. flooding on a 5-node ring (a graph with a cycle, odd length)
# ---------------------------------------------------------------------------
#
# Ring n0-n1-n2-n3-n4-n0, origin n0.  Distances: n1 = 1, n4 = 1, n2 = 2, n3 = 2.
# backoff False, so equal waiting times mean equal start times.
#
# Timeline: n0 transmits alone.  n1 and n4 (not neighbours of each other, so they
# do not interact) both receive at the same instant and transmit at the same
# instant.  n2 hears only n1 (n2's neighbours are n1 and n3; n3 is silent), and
# n3 hears only n4.  So n2 and n3 each get a clean first copy: hops 2.
# Then n2 and n3 transmit at the same instant.  They are neighbours but both
# have only just been triggered, so both start together and each cannot receive
# while it transmits.  That destroys only copies that nobody needs: n2 and n3
# already have the message.  n1 hears n2 and n4 hears n3 (duplicates, ignored).
# The origin hears n1 and n4 together (a collision at the origin, irrelevant).
#
# Receptions: n1 hops 1, n4 hops 1, n2 hops 2, n3 hops 2.  Origin: none.
# Transmissions: ttl 4: every reached node has d < 4, so all 4 forward: 1 + 4 = 5.
#                ttl 2: n2, n3 have d = 2, not < 2, so only n1, n4 forward: 1 + 2 = 3.

@pytest.mark.parametrize("ttl,expected_tx", [(4, 5), (2, 3)])
def test_flood_ring_reach_and_hops(ttl, expected_tx):
    edges = [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n0")]
    net = make_network(edges, "flood", gateway="n2", ttl=ttl, gateway_forwards=True, backoff=False, seed=5)
    key = net.originate("n0", 0.0, PAYLOAD)
    net.run_until(0.5)
    got = reached_by_message(net)[key]
    assert {n: r.hops for n, r in got.items()} == {"n1": 1, "n4": 1, "n2": 2, "n3": 2}
    assert net.data_transmissions() == expected_tx
    assert_reception_keys_consistent(net)
    assert len(net.receptions) == 4


# ---------------------------------------------------------------------------
# 1c. flooding on a 3x3 grid: hand-derived collisions
# ---------------------------------------------------------------------------
#
# Nodes g<row><col>, 4-neighbour grid, origin g00 (the corner), carrier sense on,
# backoff off, ttl 5, gateway g22 with gateway_forwards True.  BFS distances:
#   row 0: 0 1 2   row 1: 1 2 3   row 2: 2 3 4
#
# With no backoff, nodes triggered at the same instant start at the same instant,
# and a receiver that hears two overlapping frames loses both (no capture).
# Hand simulation:
#  gen 1: g00 transmits alone.  g01 and g10 hear it (hops 1) and both transmit at
#         the same instant (they are not neighbours, so they cannot sense each other).
#         g02 hears only g01: clean, hops 2.  g20 hears only g10: clean, hops 2.
#         g11 hears both g01 and g10 at once: COLLISION, no reception.
#  gen 2: g02 and g20 transmit together.  g12 hears only g02: hops 3.
#         g21 hears only g20: hops 3.  (g01, g10 hear duplicates.)
#  gen 3: g12 and g21 transmit together.  g11 hears both: COLLISION.
#         g22 hears both: COLLISION.  Nobody else is new.
#  g11 and g22 are never reached, although they are connected to the origin.
# Reached: g01:1 g10:1 g02:2 g20:2 g12:3 g21:3.  All six have d < ttl (5), so all
# six forward: transmissions = 1 + 6 = 7.

def _grid_edges():
    edges = []
    for r in range(3):
        for c in range(3):
            if c + 1 < 3:
                edges.append((f"g{r}{c}", f"g{r}{c + 1}"))
            if r + 1 < 3:
                edges.append((f"g{r}{c}", f"g{r + 1}{c}"))
    return edges


def test_flood_grid_simultaneous_forwarders_collide_at_merge_nodes():
    edges = _grid_edges()
    net = make_network(edges, "flood", gateway="g22", ttl=5, gateway_forwards=True, backoff=False, seed=3)
    key = net.originate("g00", 0.0, PAYLOAD)
    net.run_until(0.5)
    got = reached_by_message(net)[key]
    assert {n: r.hops for n, r in got.items()} == {"g01": 1, "g10": 1, "g02": 2, "g20": 2, "g12": 3, "g21": 3}
    assert "g11" not in got and "g22" not in got
    assert net.data_transmissions() == 7
    assert_reception_keys_consistent(net)


@pytest.mark.parametrize("seed", range(10))
def test_flood_grid_with_backoff_only_provable_properties(seed):
    # With random backoff the hidden-terminal overlaps are random, so reach is not
    # provable.  What is provable: a copy moves one edge per transmission, so a
    # reception can never have fewer hops than the graph distance; nobody beyond
    # ttl hops hears anything; the origin records nothing; and the first hop is
    # always clean (the origin transmits alone), so both of its neighbours get hops 1.
    edges = _grid_edges()
    ttl = 3
    net = make_network(edges, "flood", gateway="g22", ttl=ttl, gateway_forwards=True, backoff=True, seed=seed)
    key = net.originate("g00", 0.0, PAYLOAD)
    net.run_until(0.5)
    dist = bfs_dist(edges, "g00")
    got = reached_by_message(net)[key]
    assert got["g01"].hops == 1 and got["g10"].hops == 1
    for node, rec in got.items():
        assert rec.hops >= dist[node]
        assert rec.hops <= ttl
        assert dist[node] <= ttl
        assert node != "g00"
    assert_reception_keys_consistent(net)


# ---------------------------------------------------------------------------
# 2. TTL cap
# ---------------------------------------------------------------------------
#
# Chain of 6 nodes n0..n5, ttl 2, origin n0.  Distances n1:1, n2:2, n3:3, n4:4, n5:5.
# n1 hears the origin (hops 1 < 2): forwards.  n2 hears n1 (hops 2, not < 2):
# receives but does not forward.  So n3, n4, n5 never hear anything.
# Reached {n1, n2}; transmissions = origin + n1 = 2.  The gateway (n5) is never
# reached, so gateway_forwards is irrelevant (set True).

def test_ttl_cap_stops_flood():
    edges = chain_edges(6)
    net = make_network(edges, "flood", gateway="n5", ttl=2, gateway_forwards=True, backoff=False, seed=2)
    key = net.originate("n0", 0.0, PAYLOAD)
    net.run_until(0.5)
    got = reached_by_message(net)[key]
    assert {n: r.hops for n, r in got.items()} == {"n1": 1, "n2": 2}
    assert net.data_transmissions() == 2
    assert net.forwarded_frames() == 1
    assert_reception_keys_consistent(net)


# ---------------------------------------------------------------------------
# 3. chain delay (D10)
# ---------------------------------------------------------------------------
#
# 4 hop chain n0..n4, backoff off, carrier sense on, per 0, processing delay p = 1 ms,
# forward_delay 0, one message created at t = 0.25 s.  Flood uses broadcast, so there
# is no ACK to wait for.
#
# Per hop: the frame becomes head of line, waits DIFS (50 us) of idle medium, then
# transmits for 792 us (backoff off, so nothing else).  It is delivered to the next
# node p = 1 ms after the end of the transmission, and the next node enqueues at that
# moment.  The next node's DIFS counts from the moment its frame became head of line
# (documented rule), and the medium has been idle for p already, so it still waits a
# full DIFS.  Nobody else is on air at that time (previous transmitter finished
# p earlier).  So each hop takes t_hop = 50 us + 792 us + 1000 us = 1.842 ms.
# Reception at hop j: first_rx - created = j * 1.842 ms, so hop 4: 7.368 ms.

def test_chain_delay_is_hops_times_difs_airtime_processing():
    p = 1e-3
    t_hop = DIFS_S + DATA_AIR_32_S + p      # 1.842e-3
    created = 0.25
    net = make_network(chain_edges(5), "flood", gateway="n4", ttl=6, gateway_forwards=True,
                       backoff=False, processing_delay_s=p, seed=7)
    key = net.originate("n0", created, PAYLOAD)
    net.run_until(created + 0.5)
    got = reached_by_message(net)[key]
    for j in range(1, 5):
        rec = got[f"n{j}"]
        assert rec.hops == j
        assert rec.created_s == created
        assert rec.first_rx_s - created == pytest.approx(j * t_hop, abs=EXACT_S)
    assert got["n4"].first_rx_s - created == pytest.approx(7.368e-3, abs=EXACT_S)


def test_chain_delay_with_forward_delay():
    # A forwarder enqueues its copy forward_delay_s after its first reception.  The origin
    # has no such delay.  So hop j costs t_hop, plus (j - 1) forward delays in total.
    # (Assumption: the delay is counted from the delivery instant, first_rx_s.)
    p, f = 1e-3, 0.5e-3
    t_hop = DIFS_S + DATA_AIR_32_S + p
    net = make_network(chain_edges(5), "flood", gateway="n4", ttl=6, gateway_forwards=True,
                       backoff=False, processing_delay_s=p, forward_delay_s=f, seed=7)
    key = net.originate("n0", 0.0, PAYLOAD)
    net.run_until(0.5)
    got = reached_by_message(net)[key]
    for j in range(1, 5):
        assert got[f"n{j}"].first_rx_s == pytest.approx(j * t_hop + (j - 1) * f, abs=EXACT_S)


# ---------------------------------------------------------------------------
# 4. gossip on a line (D10)
# ---------------------------------------------------------------------------
#
# Line n0..n4, origin n0, gossip_g 0.5, ttl 6 (>= 4), per 0.  Node j is reached iff
# nodes 1..j-1 all chose to forward: P = g^(j-1) = 1, 0.5, 0.25, 0.125 for j = 1..4.
# Transmissions are one after another (each triggered by the previous one), so
# nothing collides.  The gateway is the last node with gateway_forwards True so that
# the gateway flag changes nothing about reach.
# Over N = 2000 seeds the count of runs reaching node j is Binomial(N, P).
# Tolerance 4 sd, sd = sqrt(N P (1 - P)):  j=1: 0 (must be exactly 2000);
# j=2: sqrt(2000*.25) = 22.4 -> 89.4; j=3: sqrt(2000*.1875) = 19.4 -> 77.5;
# j=4: sqrt(2000*.125*.875) = 14.8 -> 59.2.

N_GOSSIP = 2000


def _gossip_reach(seed, nodes=5, g=0.5):
    net = make_network(chain_edges(nodes), "gossip", gateway=f"n{nodes - 1}", ttl=nodes + 2,
                       gossip_g=g, gateway_forwards=True, backoff=False, seed=seed)
    net.originate("n0", 0.0, PAYLOAD)
    net.run_until(0.05)
    return tuple(("n0", 0, f"n{j}") in net.receptions for j in range(1, nodes))


def test_gossip_line_reach_probability():
    counts = np.zeros(4, dtype=int)
    for seed in range(N_GOSSIP):
        reach = _gossip_reach(seed)
        # reach is nested: node j is reached only if node j-1 was reached
        for j in range(1, 4):
            assert (not reach[j]) or reach[j - 1]
        counts += np.array(reach, dtype=int)
    g = 0.5
    for j in range(1, 5):
        p = g ** (j - 1)
        sd = (N_GOSSIP * p * (1.0 - p)) ** 0.5
        assert abs(counts[j - 1] - N_GOSSIP * p) <= 4.0 * sd + 1e-9, (j, counts[j - 1], N_GOSSIP * p)


# ---------------------------------------------------------------------------
# 5. star with retries (D7)
# ---------------------------------------------------------------------------
#
# One drone d0 and the gateway gs are neighbours.  Data frames (75 B) have per q = 0.3;
# ACKs (14 B) have per 0.  Retry limit 3 (up to 4 tries).  Messages are 0.1 s apart,
# far longer than any message needs (4 tries of about 1.2 ms plus backoff of a few
# ms), so there is no queueing or contention.
#
# Delivery: the message reaches gs unless all 4 tries are lost: 1 - q^4 = 1 - 0.0081 = 0.9919.
#   sd of the ratio over n = 4000: sqrt(0.9919 * 0.0081 / 4000) = 1.417e-3; tolerance 4 sd = 5.67e-3.
# Tries per message T: P(T = i) = q^(i-1) (1 - q) for i = 1..3, P(T = 4) = q^3 (the 4th try
#   is the last whether it works or not).
#   E[T] = 1 + q + q^2 + q^3 = 1 + 0.3 + 0.09 + 0.027 = 1.417 = (1 - q^4) / (1 - q) = 0.9919 / 0.7.
#   E[T^2] = 1*0.7 + 4*0.21 + 9*0.063 + 16*0.027 = 0.7 + 0.84 + 0.567 + 0.432 = 2.539
#   Var[T] = 2.539 - 1.417^2 = 2.539 - 2.007889 = 0.531111; sd of the mean over 4000 messages
#   = sqrt(0.531111 / 4000) = 0.011523; tolerance 4 sd = 0.0461.
# (data_transmissions counts data frames only, not ACKs.)

def test_star_retries_delivery_and_tries():
    q, n, spacing = 0.3, 4000, 0.1
    edges = [("d0", "gs")]
    net = make_network(edges, "star", gateway="gs", retry_limit=3, data_per=q, ack_per=0.0,
                       backoff=True, seed=21)
    for k in range(n):
        key = net.originate("d0", k * spacing, PAYLOAD)
        assert key == ("d0", k)                         # origin_seq counts 0, 1, 2, ...
    net.run_until(n * spacing + 1.0)

    assert len(net.messages) == n
    delivered = sum(1 for k in range(n) if ("d0", k, "gs") in net.receptions)
    p_deliver = 1.0 - q ** 4                           # 0.9919
    sd = (p_deliver * (1 - p_deliver) / n) ** 0.5
    assert abs(delivered / n - p_deliver) <= 4.0 * sd

    mean_tries = net.data_transmissions() / n
    expected_tries = (1.0 - q ** 4) / (1.0 - q)        # 1.417
    sd_tries = (0.531111 / n) ** 0.5
    assert abs(mean_tries - expected_tries) <= 4.0 * sd_tries

    # star has no forwarding, and the only receiver is the gateway, one hop away
    assert all(r.rx == "gs" and r.hops == 1 for r in net.receptions.values())
    assert net.forwarded_frames() == 0
    assert_reception_keys_consistent(net)

    s = net.summary(0.0, n * spacing)
    assert s["n_messages"] == n
    assert s["n_delivered"] == delivered
    assert s["delivery_ratio"] == pytest.approx(delivered / n)
    assert s["transmissions_per_message"] == pytest.approx(mean_tries)


# ---------------------------------------------------------------------------
# 6. relay
# ---------------------------------------------------------------------------
#
# Gateway G, relay R, drones A and B.  Links: G-A, G-R, A-R, R-B.  So A hears G and
# R; B hears only R; G hears A and R but not B.
# Message from A (t = 0): A broadcasts (hops 0).  G and R both hear it: G records hops 1
#   at DIFS + 792 us; R records hops 1 as well.  R then unicasts a copy to G (frame hops
#   1, so reception hops 2); G already has the message and ignores the copy.
#   Receptions for (A, 0): at G (hops 1) and at R (hops 1).  Exactly one at G.
# Message from B (t = 0.1): B broadcasts; only R hears it (hops 1).  R unicasts to G
#   (frame hops 1); G records hops 2.  Receptions for (B, 0): at R (hops 1), at G (hops 2).
# Timing of B's copy at G (processing 0, backoff off, forwarded payload the same size):
#   B starts after DIFS, ends 50 + 792 us later; R enqueues then, waits DIFS, transmits 792 us;
#   G delivers at the end: created + 2 * (50 us + 792 us) = created + 1.684 ms.
#   A's copy at G: created + 842 us.
# Data transmissions: A, R(for A), B, R(for B) = 4 (ACKs are not data frames).
# Frames enqueued by forwarders: 2.

def test_relay_paths_and_duplicate_suppression():
    edges = [("G", "A"), ("G", "R"), ("A", "R"), ("R", "B")]
    net = make_network(edges, "relay", gateway="G", relay="R", backoff=False, seed=9)
    ka = net.originate("A", 0.0, PAYLOAD)
    kb = net.originate("B", 0.1, PAYLOAD)
    assert ka == ("A", 0) and kb == ("B", 0)
    net.run_until(0.5)

    a_recv = reached_by_message(net)[ka]
    b_recv = reached_by_message(net)[kb]
    assert {n: r.hops for n, r in a_recv.items()} == {"G": 1, "R": 1}
    assert {n: r.hops for n, r in b_recv.items()} == {"R": 1, "G": 2}
    assert len(net.receptions) == 4                      # exactly one record per (message, node)
    assert a_recv["G"].first_rx_s == pytest.approx(DIFS_S + DATA_AIR_32_S, abs=EXACT_S)
    assert b_recv["G"].first_rx_s - 0.1 == pytest.approx(2.0 * (DIFS_S + DATA_AIR_32_S), abs=EXACT_S)
    assert net.data_transmissions() == 4
    assert net.forwarded_frames() == 2
    assert_reception_keys_consistent(net)


# ---------------------------------------------------------------------------
# 7. duplicate suppression at the upper layer
# ---------------------------------------------------------------------------
# The flood, ring and relay tests above already check that the number of Reception
# records equals the number of distinct reached nodes.  Here the MAC itself hands the
# upper layer several copies: data per 0 but ACK per a = 0.5, retry limit 3.  The
# receiver gets the data on the first try every time (so every message is delivered),
# but the sender sees an ACK only half the time and retries, so gs gets duplicates at
# the MAC level.  P(a message has at least 2 delivered copies) = P(first ACK lost) = 0.5,
# so about 200 of 400 messages.  The check needs at least 100 (10 sd below the mean).
# The network must still record exactly one Reception per message, and its
# first_rx_s must be the earliest of the MAC delivery times.

def test_mac_level_duplicates_are_suppressed_and_first_copy_wins():
    n, spacing = 400, 0.1
    net = make_network([("d0", "gs")], "star", gateway="gs", retry_limit=3, data_per=0.0, ack_per=0.5,
                       backoff=True, seed=33)
    for k in range(n):
        net.originate("d0", k * spacing, PAYLOAD)
    net.run_until(n * spacing + 1.0)

    copies = {}
    for r in net.sim.rx_records:
        if r.rx == "gs" and r.outcome == "delivered" and r.origin == "d0" and r.origin_seq is not None:
            copies.setdefault(r.origin_seq, []).append(r.deliver_s)
    assert sum(1 for v in copies.values() if len(v) > 1) >= 100     # duplicates really happened

    assert len(net.receptions) == n
    assert_reception_keys_consistent(net)
    for k in range(n):
        rec = net.receptions[("d0", k, "gs")]
        assert rec.first_rx_s == pytest.approx(min(copies[k]), abs=1e-12)


# ---------------------------------------------------------------------------
# 8. determinism
# ---------------------------------------------------------------------------

def _lossy_star_receptions(seed):
    net = make_network([("d0", "gs")], "star", gateway="gs", retry_limit=1, data_per=0.5, ack_per=0.0,
                       backoff=True, seed=seed)
    for k in range(100):
        net.originate("d0", k * 0.05, PAYLOAD)
    net.run_until(6.0)
    return dict(net.receptions), net.data_transmissions()


def test_same_seed_same_results_and_different_seed_differs():
    a = _lossy_star_receptions(5)
    b = _lossy_star_receptions(5)
    assert a == b
    c = _lossy_star_receptions(6)
    assert a != c           # 100 messages with random backoff and 50 % loss: the runs cannot match


def test_gossip_same_seed_identical_and_seeds_differ():
    first = [_gossip_reach(s, nodes=6) for s in range(20)]
    second = [_gossip_reach(s, nodes=6) for s in range(20)]
    assert first == second
    # P(all 20 runs give one identical pattern) is below 2^-19, so at least two patterns must appear
    assert len(set(first)) > 1


def test_gossip_same_seed_identical_receptions_dict():
    def run(seed):
        net = make_network(chain_edges(6), "gossip", gateway="n5", ttl=8, gossip_g=0.7,
                           gateway_forwards=True, backoff=True, seed=seed)
        for k in range(30):
            net.originate("n0", k * 0.05, PAYLOAD)
        net.run_until(2.0)
        return dict(net.receptions), net.data_transmissions()
    assert run(4) == run(4)
