"""Cross-check of sim.radio.mac.MacSimulator against an independent reference.

The reference (sim/tests/radio_reference.py) was written from the probability alone.
Here the MAC engine is run in its ALOHA limiting mode and in a single-sender
queueing setup, and compared with (1) the closed forms and (2) the reference.

ALOHA mode: carrier_sense False, backoff False, huge queues, no retries, every
frame a broadcast heard by a gateway "gs".  N sender nodes each carry a Poisson
source of rate G / (T N), so the total offered load is G = lambda T.  A frame is
delivered at gs iff nothing else overlaps it at gs (pure) or shares its slot
(slotted).

Fake channel: every link that involves "gs" has RSSI -40 dBm, SNR 55 dB, PER 0.
Sender-to-sender links are set far below sensitivity (-150 dBm) so the engine
does not write 200 records per frame for receivers nobody counts (that would be
tens of millions of records).  This changes nothing at gs, which is the only
receiver counted.

Finite-population bias (why N is large).  Each sender is half-duplex: if a new
arrival comes while its own previous frame is still on air, the frame waits for
that frame to end.  So frames are not exactly Poisson per node.  A fraction of
about 1 - exp(-G/N) of arrivals is affected (about 0.5 % at G = 1, N = 200).  The
effect on P(success) is first order in 1/N: a node never overlaps itself, so the
other-node load is G (1 - 1/N) and pure P(success) is about exp(-2G) (1 + 2G/N);
slotted about exp(-G) (1 + G/N).  The bias budget used below is that first-order
value, and test_finite_population_bias_is_below_quarter_of_tolerance checks with
the reference that, at the N and frame count used here, the true bias is below
one quarter of the 4-sigma tolerance.  The MAC is also compared against the
reference run with the SAME finite N, which removes this bias entirely.

Tolerances: 4 standard errors (batch means over 20 batches of consecutive
frames, so neighbour correlation is included automatically).  Combined sd of two
independent estimates is the root of the sum of squares.

M/D/1 mode: one sender "n0", carrier_sense True, backoff False,
immediate_access True, Poisson source to BROADCAST.  Two intervals matter:

* Contract rule (2026-10-14): every frame waits DIFS of idle medium after it
  becomes head of line, even on a long-idle medium.  So
  start_k = max(arrival_k, end_{k-1}) + difs, and s_k = start_k - difs obeys
  s_k = max(arrival_k, s_{k-1} + D) with D = difs + data airtime: the Lindley
  recursion with service time D.
* queue_wait_s = time from creation until head of line = s_k - arrival_k, which
  is exactly the M/D/1 waiting time; its mean is the Pollaczek-Khinchine value
  Wq = rho D / (2 (1 - rho)).
* W = first_tx_s - created_s = queue_wait_s + difs, so its mean is Wq + difs.
"""

import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mac = pytest.importorskip("sim.radio.mac")

from sim.tests import radio_reference as ref  # noqa: E402

PAYLOAD = 32
N_NODES = 150            # senders in ALOHA mode (see the bias note above)
M_FRAMES = 25_000        # frames scored per ALOHA run (engine cost is about N * frames * 3 us)
N_BATCHES = 20
WARM_T = 20.0            # frame durations skipped at the start and the end of the run
REF_SCALE = 4            # the reference runs this many times longer (smaller sd)
EXACT_TOL_S = 1e-9


class FlatChannel:
    """Fake channel: -40 dBm on every link that involves 'gs', else far below sensitivity."""

    sensitivity_dbm = -98.0

    def __init__(self, gateway="gs"):
        self.gateway = gateway
        self.cfg = SimpleNamespace(noise_floor_dbm=-95.0, sensitivity_dbm=-98.0)
        # shared, never mutated: saves an allocation per (transmission, receiver) pair
        self._good = SimpleNamespace(rssi_dbm=-40.0, snr_db=55.0, per=0.0, shadow_db=0.0, path_loss_db=50.0)
        self._bad = SimpleNamespace(rssi_dbm=-150.0, snr_db=-55.0, per=1.0, shadow_db=0.0, path_loss_db=160.0)

    def sample(self, tx, rx, distance_m, t_s, *, frame_bytes=0, **kwargs):
        return self._good if (tx == self.gateway or rx == self.gateway) else self._bad

    def received_power_dbm(self, tx, rx, distance_m, t_s, **kwargs):
        return -40.0 if (tx == self.gateway or rx == self.gateway) else -150.0


def _positions(names):
    pos = {"gs": (0.0, 0.0, 0.0)}
    for i, name in enumerate(names):
        ang = 2.0 * np.pi * i / max(len(names), 1)
        pos[name] = (10.0 * float(np.cos(ang)), 10.0 * float(np.sin(ang)), 0.0)
    return pos


# ---------------------------------------------------------------------------
# ALOHA mode
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _mac_aloha(G, slotted, n_nodes=N_NODES, n_frames=M_FRAMES, seed=2026):
    """Run the engine; return per-frame arrays for every frame (sorted by creation)."""
    phy = mac.PhyConfig()
    T = mac.data_airtime_s(PAYLOAD, phy)
    cfg = mac.MacConfig(carrier_sense=False, backoff=False, slotted=slotted,
                        slot_len_s=T if slotted else None,
                        queue_capacity=100_000, unicast_retry_limit=0)
    names = [f"n{i}" for i in range(n_nodes)]
    sim = mac.MacSimulator(_positions(names), FlatChannel(), phy, cfg, seed=seed)
    rate = G / (T * n_nodes)
    window_s = n_frames * T / G
    horizon = WARM_T * T + window_s + WARM_T * T
    for name in names:
        sim.add_poisson_source(name, mac.BROADCAST, rate, PAYLOAD, start_s=0.0, stop_s=horizon)
    sim.run_until(horizon + 200.0 * T)  # drain: per-node utilisation is tiny
    at_gs = {}
    for r in sim.rx_records:
        if r.rx == "gs":
            assert r.frame_id not in at_gs, "broadcast frame recorded twice at gs"
            at_gs[r.frame_id] = r
    assert len(at_gs) == len(sim.frame_records), "every transmitted frame must have a record at gs"
    fids = sorted(at_gs)
    created = np.array([sim.frame_records[f].created_s for f in fids])
    start = np.array([at_gs[f].tx_start_s for f in fids])
    end = np.array([at_gs[f].tx_end_s for f in fids])
    ok = np.array([at_gs[f].outcome == "delivered" for f in fids])
    node = np.array([int(sim.frame_records[f].src[1:]) for f in fids])
    order = np.argsort(created, kind="stable")
    return {"T": T, "created": created[order], "start": start[order], "end": end[order],
            "success": ok[order], "node": node[order],
            "t0": WARM_T * T, "t1": WARM_T * T + window_s}


def _window_mean_se(created, success, t0, t1):
    m = (created >= t0) & (created < t1)
    mean, se = ref.batch_means(success[m].astype(float), N_BATCHES)
    return mean, se, int(m.sum())


def _reference_finite_n(G, slotted, n_nodes=N_NODES, n_frames=M_FRAMES, seed=77):
    """Reference run with the same N, windowed in frame durations (T = 1)."""
    window = REF_SCALE * n_frames / G
    res = ref.finite_n_aloha(G, n_nodes, WARM_T + window + WARM_T, ref.make_rng(seed), slotted=slotted)
    return ref.window_success(res, WARM_T, WARM_T + window, N_BATCHES)


def _closed_form(G, slotted):
    return np.exp(-G) if slotted else np.exp(-2.0 * G)


def _bias_budget(G, slotted, n_nodes=N_NODES):
    """First-order finite-population bias of P(success): p * 2G/N (pure), p * G/N (slotted)."""
    return _closed_form(G, slotted) * (1.0 if slotted else 2.0) * G / n_nodes


CASES = [(False, 0.5), (False, 1.0), (True, 1.0), (True, 2.0)]
IDS = [f"{'slotted' if s else 'pure'}-G{g}" for s, g in CASES]


@pytest.mark.parametrize("slotted,G", CASES, ids=IDS)
def test_mac_aloha_success_matches_closed_form_and_reference(slotted, G):
    r = _mac_aloha(G, slotted)
    mac_p, mac_se, n = _window_mean_se(r["created"], r["success"], r["t0"], r["t1"])
    assert n > 0.95 * M_FRAMES

    closed = _closed_form(G, slotted)
    budget = _bias_budget(G, slotted)
    ref_p, ref_se, _ = _reference_finite_n(G, slotted)

    # 1. MAC against the closed form: 4 sd plus the known finite-population bias
    assert abs(mac_p - closed) <= 4.0 * mac_se + budget, (mac_p, closed, mac_se, budget)
    # 2. the reference against the closed form, same allowance
    assert abs(ref_p - closed) <= 4.0 * ref_se + budget, (ref_p, closed, ref_se, budget)
    # 3. MAC against the reference with the same N: no bias allowance needed
    assert abs(mac_p - ref_p) <= 4.0 * np.hypot(mac_se, ref_se), (mac_p, ref_p, mac_se, ref_se)
    # throughput S = G * p, the quantity in the textbook formulas
    S_closed = G * closed
    assert abs(G * mac_p - S_closed) <= G * (4.0 * mac_se + budget)
    assert mac_se < 0.01  # the check is not vacuous


@pytest.mark.parametrize("slotted,G", CASES, ids=IDS)
def test_finite_population_bias_is_below_quarter_of_tolerance(slotted, G):
    """With this N the true bias (measured on a very long reference run) is under
    one quarter of the 4-sigma tolerance of the MAC runs, i.e. under one sd."""
    frames = 3_000_000
    window = frames / G
    res = ref.finite_n_aloha(G, N_NODES, window + 2 * WARM_T, ref.make_rng(99), slotted=slotted)
    p, se, _ = ref.window_success(res, WARM_T, WARM_T + window, N_BATCHES)
    if slotted:
        nominal_sd = ref.simulate_slotted_aloha(G, M_FRAMES, ref.make_rng(5))["se_fraction"]
    else:
        nominal_sd = ref.pure_aloha_fraction_se(G, M_FRAMES)
    bias = abs(p - _closed_form(G, slotted))
    assert bias - 3.0 * se <= nominal_sd, (bias, se, nominal_sd)  # quarter of 4 sd = 1 sd


@pytest.mark.parametrize("slotted,G", [(False, 0.5), (True, 1.0)], ids=["pure-G0.5", "slotted-G1.0"])
def test_mac_per_frame_outcomes_match_reference_classifier(slotted, G):
    """Exact, frame by frame: given the start times the engine chose, the independent
    classifier predicts which frames reach gs.  No statistics involved."""
    r = _mac_aloha(G, slotted)
    T = r["T"]
    assert np.all(r["start"] >= r["created"] - EXACT_TOL_S)
    assert np.allclose(r["end"] - r["start"], T, rtol=0, atol=EXACT_TOL_S)
    # half-duplex: one node's transmissions never overlap each other
    for i in np.unique(r["node"])[:50]:
        s = np.sort(r["start"][r["node"] == i])
        assert np.all(np.diff(s) >= T * (1.0 - 1e-9))
    if slotted:
        predicted, slot = ref.classify_slotted(r["start"], T)  # raises if not on the slot grid
    else:
        predicted = ref.classify_pure(r["start"], T)
    wrong = np.flatnonzero(predicted != r["success"])
    assert len(wrong) == 0, f"{len(wrong)} of {len(predicted)} frames disagree, first at {r['start'][wrong[:3]]}"
    assert 0.0 < r["success"].mean() < 1.0


# ---------------------------------------------------------------------------
# M/D/1 mode
# ---------------------------------------------------------------------------

MD1_BATCH = 10_000
MD1_WARMUP = 2_000


@lru_cache(maxsize=None)
def _mac_md1(rho, seed=2027):
    phy = mac.PhyConfig()
    T = mac.data_airtime_s(PAYLOAD, phy)
    difs = phy.difs_s
    D = difs + T
    lam = rho / D
    n = MD1_WARMUP + N_BATCHES * MD1_BATCH
    horizon = n / lam
    cfg = mac.MacConfig(carrier_sense=True, backoff=False, immediate_access=True,
                        queue_capacity=100_000, unicast_retry_limit=0)
    sim = mac.MacSimulator(_positions(["n0"]), FlatChannel(), phy, cfg, seed=seed)
    sim.add_poisson_source("n0", mac.BROADCAST, lam, PAYLOAD, start_s=0.0, stop_s=horizon)
    sim.run_until(horizon + 500.0 * D)
    recs = [sim.frame_records[f] for f in sorted(sim.frame_records)]
    assert all(fr.status == "sent" for fr in recs)
    assert all(fr.queue_wait_s is not None and fr.first_tx_s is not None for fr in recs)
    return {"T": T, "difs": difs, "D": D, "lam": lam,
            "created": np.array([fr.created_s for fr in recs]),
            "first_tx": np.array([fr.first_tx_s for fr in recs]),
            "queue_wait": np.array([fr.queue_wait_s for fr in recs])}


@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_mac_single_sender_path_matches_lindley_exactly(rho):
    """Feed the engine's own arrival times to the independent recursion: every frame's
    transmission start and head-of-line wait must agree to a nanosecond."""
    r = _mac_md1(rho)
    starts = ref.single_sender_starts(r["created"], r["T"], r["difs"])
    np.testing.assert_allclose(r["first_tx"], starts, rtol=0, atol=EXACT_TOL_S)
    hol = ref.head_of_line_waits(r["created"], starts, r["T"])
    np.testing.assert_allclose(r["queue_wait"], hol, rtol=0, atol=EXACT_TOL_S)
    assert r["queue_wait"].max() > 0.0  # the queue really did build up


@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_mac_md1_mean_wait_matches_formula_and_reference(rho):
    r = _mac_md1(rho)
    T, difs, D, lam = r["T"], r["difs"], r["D"], r["lam"]
    formula = ref.md1_mean_wait(lam, D)  # Wq = rho D / (2 (1 - rho))

    n = MD1_WARMUP + N_BATCHES * MD1_BATCH
    ref_arr = np.cumsum(ref.make_rng(88).exponential(1.0 / lam, size=n))
    ref_start = ref.single_sender_starts(ref_arr, T, difs)
    ref_w = (ref_start - ref_arr)[MD1_WARMUP:]
    ref_h = ref.head_of_line_waits(ref_arr, ref_start, T)[MD1_WARMUP:]
    ref_w_mean, ref_w_se = ref.batch_means(ref_w, N_BATCHES)
    ref_h_mean, ref_h_se = ref.batch_means(ref_h, N_BATCHES)

    # queue_wait_s (arrival to head of line) is the M/D/1 waiting time itself
    mac_h_mean, mac_h_se = ref.batch_means(r["queue_wait"][MD1_WARMUP:], N_BATCHES)
    assert abs(mac_h_mean - formula) <= 4.0 * mac_h_se
    assert abs(ref_h_mean - formula) <= 4.0 * ref_h_se
    assert abs(mac_h_mean - ref_h_mean) <= 4.0 * np.hypot(mac_h_se, ref_h_se)

    # W = first_tx_s - created_s adds exactly one DIFS to every frame
    mac_w_mean, mac_w_se = ref.batch_means((r["first_tx"] - r["created"])[MD1_WARMUP:], N_BATCHES)
    assert abs(mac_w_mean - (formula + difs)) <= 4.0 * mac_w_se
    assert abs(mac_w_mean - ref_w_mean) <= 4.0 * np.hypot(mac_w_se, ref_w_se)
    assert mac_h_se < 0.05 * formula  # not vacuous
