"""Independent checks of the realized loss rate and delay of the packet links.

Plain-words derivation (no linear algebra needed)

Independent loss: each packet is a coin flip that is lost with chance p. Out of
N packets the lost fraction wobbles around p with standard error
sqrt(p(1-p)/N); we accept anything within 3.29 standard errors (a 99.9 percent
band), so a correct link fails by chance about once in a thousand, and the
seeds are fixed so the result repeats.

Burst loss: the link is either in a good stretch or a lost stretch. This
code's burst model is NOT the textbook two-state chain. Here each delivered
packet starts a burst with chance s, and the burst-start packet is itself lost.
A lost packet is followed by a delivery only if the burst ends (chance e) and
the next start draw also misses (chance 1 - s). Write a = s for "delivered, then
lost" and b = e(1 - s) for "lost, then delivered". Balancing the flow between
the two states gives the long-run lost fraction a/(a + b) = s/(s + e - s*e),
and the mean lost run is 1/b = 1/(e(1 - s)). The textbook chain (start packet
delivered, p/(p + q) and 1/q) would give different answers: with s = 0.05 and
e = 0.25 the code's rule gives 0.17391 and 4.2105 while the textbook gives
0.16667 and 4.000. Neighbouring packets are correlated (a burst drags the next
packet with it), so the sampling noise of the lost fraction is larger than for
coin flips by the factor (1 + lam)/(1 - lam), where lam = 1 - a - b, and the
band is widened by the square root of that factor. A run of consecutive losses
ending in a delivery is a geometric length with mean 1/b and standard
deviation sqrt(1 - b)/b.

Cooperative abstract links: six routes, each a long stream of packets at 10 Hz.
Loss per route is again a binomial count (with the band split over six routes
so all six together still fail by chance only once in a thousand), delay is a
lognormal with a stated mean, and a packet sent in the last second is dropped
as too late at the end of the run. For the burst version the start chance that
makes the long-run loss equal a target P is s = P*e/(1 - P*(1 - e)), obtained
by solving P = s/(s + e - s*e) for s.

Every expected value is a literal or comes from numpy or the standard library;
nothing is imported from sim/ except the link objects under test.
"""
from __future__ import annotations

import numpy as np

from sim.coop.net import LinkSpec, TrafficSpec, make_deliveries
from sim.link import LinkConfig, SimulatedLink

Z = 3.2905           # 99.9 percent two-sided normal quantile
Z_SIX_ROUTES = 3.7648  # 99.9 percent split over six routes (Bonferroni)


def _lost_flags(config, n, seed):
    link = SimulatedLink(config, seed=seed)
    return np.array([link.send("A", "B", i, i * 0.01).receive_time is None for i in range(n)])


def _closed_runs(flags):
    """Lengths of runs of True that end at a False (a run cut off by the end is dropped)."""
    runs, current = [], 0
    for lost in flags:
        if lost:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    return np.array(runs)


def _after(flags, previous):
    """Outcomes of the packets whose previous packet had the given outcome."""
    return flags[1:][flags[:-1] == previous]


def test_independent_link_loss_is_bernoulli_at_the_configured_rate():
    """Independent fact: N Bernoulli(p) draws give a lost fraction with SE sqrt(p(1-p)/N)."""
    n = 20_000
    for p in (0.05, 0.3, 0.9):
        flags = _lost_flags(LinkConfig(loss_model="independent", loss_probability=p), n, seed=601)
        assert abs(flags.mean() - p) <= Z * np.sqrt(p * (1 - p) / n), p
        if p == 0.3:
            # No memory: after a lost packet the next one is still lost with chance p.
            # Expected count of such packets is p * n = 6,000.
            after_loss = _after(flags, True)
            assert abs(after_loss.mean() - p) <= Z * np.sqrt(p * (1 - p) / (p * n))
    # Limiting cases: uniform draws lie in [0, 1), so p = 0 never loses and p = 1 always does.
    assert not _lost_flags(LinkConfig(loss_model="independent", loss_probability=0.0), 2000, 602).any()
    assert _lost_flags(LinkConfig(loss_model="independent", loss_probability=1.0), 2000, 603).all()


def test_burst_link_matches_its_two_state_chain():
    """Independent fact: hand-derived chain for sim/link.py's burst rule.

    A delivered packet is followed by a loss with chance a = s (start draw).
    A lost packet is followed by a delivery only if the burst ends (e) and the
    next start draw misses (1 - s): b = e(1 - s). Long-run lost fraction
    a / (a + b) = 0.17391, mean lost run 1 / b = 4.2105. The textbook chain
    (start packet delivered) would give 0.16667, 0.7500 and 4.000 instead.
    Correlation lambda = 1 - a - b widens the fraction's variance by
    (1 + lambda) / (1 - lambda) = 5.957 over binomial.
    """
    s, e, n = 0.05, 0.25, 200_000
    flags = _lost_flags(LinkConfig(loss_model="burst", burst_start_probability=s,
                                   burst_end_probability=e), n, seed=604)
    a, b = s, e * (1 - s)
    pi = a / (a + b)
    lam = 1 - a - b
    fraction_sd = np.sqrt(pi * (1 - pi) / n * (1 + lam) / (1 - lam))
    assert abs(flags.mean() - pi) <= Z * fraction_sd
    # Each step out of a state is a fresh coin (Markov property).
    assert abs(_after(flags, True).mean() - (1 - b)) <= Z * np.sqrt(b * (1 - b) / (pi * n))
    assert abs(_after(flags, False).mean() - a) <= Z * np.sqrt(a * (1 - a) / ((1 - pi) * n))
    # Closed runs are independent geometric lengths with SD sqrt(1 - b) / b.
    runs = _closed_runs(flags)
    assert len(runs) > 0.9 * n * pi * b
    assert abs(runs.mean() - 1 / b) <= Z * np.sqrt(1 - b) / b / np.sqrt(len(runs))


OBSERVERS = tuple(f"o{i}" for i in range(1, 7))


def test_coop_abstract_routes_deliver_the_configured_fraction_and_delay():
    """Independent facts: binomial loss per route, lognormal mean 0.1 s, end cut-off.

    Six routes, 6001 packets each (0.0 to 600.0 s at 10 Hz), loss 0.3.
    A packet sent at or before 599.0 s is late only if its delay exceeds 1 s,
    which for a lognormal with mean 0.1 s and std 0.02 s has chance about 1e-30.
    """
    p, delay, jitter, duration = 0.3, 0.1, 0.02, 600.0
    table = make_deliveries(TrafficSpec("star", OBSERVERS, rate_hz=10.0, duration_s=duration),
                            LinkSpec("abstract", delay, p, jitter_s=jitter), 605)
    all_rows = []
    for name in OBSERVERS:
        rows = table[(name, "gs")]
        assert len(rows) == 6001
        link_lost = np.array([row.loss_reason == "link" for row in rows])
        assert abs(link_lost.mean() - p) <= Z_SIX_ROUTES * np.sqrt(p * (1 - p) / len(rows)), name
        all_rows.extend(rows)
    pooled = np.mean([row.loss_reason == "link" for row in all_rows])
    assert abs(pooled - p) <= Z * np.sqrt(p * (1 - p) / len(all_rows))
    assert all(row.send_time_s > duration - 1.0 for row in all_rows if row.loss_reason == "late_end")
    latencies = np.array([row.receive_time_s - row.send_time_s for row in all_rows
                          if row.receive_time_s is not None and row.send_time_s <= duration - 1.0])
    assert latencies.min() > 0.0
    assert abs(latencies.mean() - delay) <= Z * jitter / np.sqrt(len(latencies))


def test_coop_abstract_burst_hits_the_requested_loss_and_merged_run_length():
    """Independent fact: the same chain with e = 1/5 and the start chance that
    should make the long-run loss 0.3: s = P e / (1 - P (1 - e)) = 0.078947.

    Then b = e (1 - s) = 0.18421 and the mean lost run is 1 / b = 5.4286, not
    1 / e = 5, because a burst can end and restart with no delivery between.
    Six routes of 30,001 packets each; correlation widens the band by 6.600.
    """
    target, e = 0.3, 0.2
    table = make_deliveries(TrafficSpec("star", OBSERVERS, rate_hz=10.0, duration_s=3000.0),
                            LinkSpec("abstract", 0.0, target, loss_model="burst"), 606)
    s = target * e / (1 - target * (1 - e))
    a, b = s, e * (1 - s)
    lam = 1 - a - b
    flags_by_route = []
    for name in OBSERVERS:
        rows = table[(name, "gs")]
        assert {row.loss_reason for row in rows} <= {None, "link"}
        flags_by_route.append(np.array([row.receive_time_s is None for row in rows]))
    pooled = np.concatenate(flags_by_route)
    fraction_sd = np.sqrt(target * (1 - target) / len(pooled) * (1 + lam) / (1 - lam))
    assert abs(pooled.mean() - target) <= Z * fraction_sd
    runs = np.concatenate([_closed_runs(flags) for flags in flags_by_route])
    assert len(runs) > 0.9 * len(pooled) * target * b
    assert abs(runs.mean() - 1 / b) <= Z * np.sqrt(1 - b) / b / np.sqrt(len(runs))
