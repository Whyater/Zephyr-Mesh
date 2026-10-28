"""Do tools/burst_bridge.py and tools/fit_espnow_log.py agree on whether a log is bursty?

Both tools read the same statistic: the burstiness z from ``fit_gilbert`` in
tools/calibrate_radio_from_trace.py (observed mean complete loss run against
the independent-loss value 1 / (1 - q)).  They apply different cutoffs:

- burst_bridge ``_verdict``: two-sided, |z| < 2.0 is "consistent with
  independent loss", z >= 2.0 is "bursty", z <= -2.0 is "anti-bursty";
- fit_espnow_log ``fit_loss``: one-sided, the burst model is chosen when
  z > norm.ppf(confidence) and the fitted end probability is below 1, else
  the independent model.  At the default confidence 0.95 the cutoff is 1.645.

So the two tools give opposite answers on the same log when z falls between
the two cutoffs.  The tests below pin that band with synthetic logs whose z is
worked out by hand, and check that nothing else separates the tools (same
pieces, same z).  They describe the tools as they are on purpose: if the
cutoffs are ever unified, the disagreement tests here must change with them.

Every log here is synthetic, built from a pattern written in this file or a
numpy-driven chain with known transition chances.  Nothing is a radio
measurement.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from sim.trace import EspNowTrace
from tools.burst_bridge import (
    VERDICT_ANTI,
    VERDICT_BURSTY,
    VERDICT_IID,
    analyze,
    read_trace_json,
)
from tools.fit_espnow_log import fit_loss

BRIDGE_CUTOFF = 2.0  # tools/burst_bridge.py IID_Z_CUTOFF, written out so a change is noticed here


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _packet(session: str, seq: int, outcome: str, *, duplicate: bool = False) -> dict:
    t = 0.1 * seq
    pkt = {"packet_id": f"{session}:{seq}" + (":dup" if duplicate else ""), "sender_id": "A", "receiver_id": "B",
           "sender_session_id": session, "seq": seq, "send_time_s": t, "outcome": outcome}
    if outcome == "received":
        pkt.update(receive_time_s=t + 0.002, echo_turnaround_s=0.0001, echo_time_s=t + 0.004, rssi_dbm=-50)
    elif outcome == "lost":
        pkt["loss_reason"] = "not_in_echoer_log"
    else:
        pkt["unknown_reason"] = "outside_echoer_log"
    if duplicate:
        pkt["duplicate"] = True
    return pkt


def trace_doc(packets: list[dict]) -> dict:
    return {
        "schema": "zephyr-espnow-trace-1", "trace_id": "burst-verdict-consistency", "status": "synthetic",
        "clock_domain": "sender_round_trip", "source": "tools/tests/test_burst_verdict_consistency.py",
        "notes": "synthetic loss pattern; not a radio measurement",
        "capture": {"transport": "ESP-NOW", "device_ids": ["A", "B"], "firmware": "synthetic", "channel": 1,
                    "phy": "synthetic"},
        "packets": packets,
    }


def doc_from_lost(lost) -> dict:
    """One unbroken session: 1 = lost, 0 = received."""
    return trace_doc([_packet("s", i, "lost" if v else "received") for i, v in enumerate(lost)])


def run_both(doc: dict, confidence: float = 0.95) -> tuple[dict, dict]:
    """The same trace document through each tool's own reader and loss analysis."""
    fit = fit_loss(EspNowTrace.from_dict(doc), confidence=confidence, seed=0)
    pieces, _ = read_trace_json(doc)
    return fit, analyze(pieces)


def bridge_says(res: dict) -> str:
    v = res["verdict"]
    for label in (VERDICT_BURSTY, VERDICT_ANTI, VERDICT_IID):
        if v.startswith(label):
            return {VERDICT_BURSTY: "bursty", VERDICT_ANTI: "anti", VERDICT_IID: "independent"}[label]
    raise AssertionError(f"unexpected verdict {v!r}")


def fit_says(fit: dict) -> str:
    return "bursty" if fit["model_choice"]["loss_model"] == "burst" else "independent"


def pattern_log(n_single: int, n_double: int, gap: int) -> list[int]:
    """``gap`` received, then each loss run followed by ``gap`` received.

    n_single runs of one lost packet and n_double runs of two, interleaved.
    Every run is complete (received on both sides), so nothing is left out.
    """
    runs = [1] * n_single + [2] * n_double
    order = sorted(range(len(runs)), key=lambda i: (i * 7919) % len(runs))  # fixed shuffle
    out = [0] * gap
    for i in order:
        out += [1] * runs[i] + [0] * gap
    return out


def hand_z(complete_runs: list[int], n_lost: int, n_packets: int) -> float:
    """The burstiness z from its definition, computed here without the tools.

    q = lost / packets; independent loss gives geometric run lengths with mean
    1 / (1 - q) and variance q / (1 - q)^2, so the mean of R runs has standard
    error sqrt(q) / (1 - q) / sqrt(R).
    """
    q = n_lost / n_packets
    mean = sum(complete_runs) / len(complete_runs)
    se = math.sqrt(q) / (1.0 - q) / math.sqrt(len(complete_runs))
    return (mean - 1.0 / (1.0 - q)) / se


def pattern_z(n_single: int, n_double: int, gap: int) -> float:
    runs = [1] * n_single + [2] * n_double
    return hand_z(runs, sum(runs), gap * (len(runs) + 1) + sum(runs))


# (n_single, n_double, gap) chosen so the hand z lands near the value in the comment.
SWEEP = [
    (30, 27, 4),    # 1.20
    (104, 53, 5),   # 1.55
    (43, 42, 4),    # 1.62
    (110, 57, 5),   # 1.66
    (44, 44, 4),    # 1.70
    (28, 16, 6),    # 1.80
    (28, 36, 4),    # 1.95
    (31, 39, 4),    # 1.99
    (22, 32, 4),    # 2.01
    (52, 36, 5),    # 2.05
    (124, 36, 8),   # 2.20
    (42, 53, 4),    # 2.30
    (108, 49, 6),   # 2.31
    (62, 33, 6),    # 2.34
    (88, 43, 6),    # 2.40
    (40, 36, 5),    # 2.60
]


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------

def test_hand_z_matches_a_worked_example():
    """Truth: one pattern worked through by hand, so hand_z is not only the tools' formula typed again.

    30 single runs and 27 double runs, gap 4: R = 57 runs, 84 lost, 4 x 58 + 84 = 316 packets.
    q = 84 / 316 = 0.26582; observed mean run 84 / 57 = 1.47368; independent mean
    1 / (1 - q) = 316 / 232 = 1.36207; standard error sqrt(q) / (1 - q) / sqrt(57) = 0.093016;
    z = (1.47368 - 1.36207) / 0.093016 = 1.19996.
    """
    lost = pattern_log(30, 27, 4)
    assert (len(lost), sum(lost)) == (316, 84)
    assert pattern_z(30, 27, 4) == pytest.approx(1.19996, abs=1e-5)
    fit, res = run_both(doc_from_lost(lost))
    assert res["gilbert"]["burstiness"]["approx_z"] == pytest.approx(1.19996, abs=1e-5)
    assert fit["independence_check"]["approx_z"] == pytest.approx(1.19996, abs=1e-5)


def test_pattern_logs_give_the_hand_worked_z_in_both_tools():
    """Truth: z from the definition (hand_z).  Both tools must report exactly that value."""
    for n1, n2, gap in SWEEP:
        fit, res = run_both(doc_from_lost(pattern_log(n1, n2, gap)))
        want = pattern_z(n1, n2, gap)
        assert fit["independence_check"]["approx_z"] == pytest.approx(want, rel=1e-12, abs=1e-12)
        assert res["gilbert"]["burstiness"]["approx_z"] == pytest.approx(want, rel=1e-12, abs=1e-12)
        assert res["runs"]["n_complete"] == n1 + n2


@pytest.mark.parametrize("confidence", [0.95, 0.99])
def test_verdicts_disagree_exactly_between_the_two_cutoffs(confidence):
    """Sweep z from 1.2 to 2.6.  Truth: the band (lower cutoff, upper cutoff) from the two stated rules.

    At 0.95 the fit's cutoff 1.645 is below the bridge's 2.0: inside the band
    the fit picks the burst model and the bridge says independent.  At 0.99 the
    fit's cutoff 2.326 is above 2.0: the band flips and the bridge is the one
    saying bursty.
    """
    fit_cut = float(stats.norm.ppf(confidence))
    seen_disagree = 0
    for n1, n2, gap in SWEEP:
        z = pattern_z(n1, n2, gap)
        fit, res = run_both(doc_from_lost(pattern_log(n1, n2, gap)), confidence)
        f, b = fit_says(fit), bridge_says(res)
        assert f == ("bursty" if z > fit_cut else "independent"), (n1, n2, gap, z)
        assert b == ("bursty" if z >= BRIDGE_CUTOFF else "independent"), (n1, n2, gap, z)
        # The band from the two rules themselves: (1.645, 2.0) at 0.95, [2.0, 2.326] at 0.99.
        in_band = (z > fit_cut) != (z >= BRIDGE_CUTOFF)
        assert (f != b) == in_band, (n1, n2, gap, z, f, b)
        assert in_band == (min(fit_cut, BRIDGE_CUTOFF) < z < max(fit_cut, BRIDGE_CUTOFF))  # no sweep z on an edge
        seen_disagree += f != b
    assert seen_disagree >= 3  # the sweep really has points inside the band


def test_cutoffs_match_only_at_confidence_phi_of_two():
    """Truth: the one-sided cutoff equals 2.0 only at confidence Phi(2.0) = 0.97725.

    At that confidence the bursty side of the two rules is the same for every
    z in the sweep (the fit uses z > cutoff, the bridge z >= cutoff, which only
    differs at z exactly 2.0).
    """
    confidence = float(stats.norm.cdf(BRIDGE_CUTOFF))
    assert confidence == pytest.approx(0.97725, abs=1e-5)
    for n1, n2, gap in SWEEP:
        fit, res = run_both(doc_from_lost(pattern_log(n1, n2, gap)), confidence)
        assert fit_says(fit) == bridge_says(res), (n1, n2, gap)


def test_negative_z_is_anti_bursty_in_the_bridge_and_independent_in_the_fit():
    """Runs of exactly one lost packet: truth z = -sqrt(q R) (mean run 1, below 1 / (1 - q)).

    Not a bursty-versus-independent clash: the fit has no anti-bursty model, so
    it falls back to independent, and the bridge only names the pattern.
    """
    for n_runs, gap, want_bridge in ((36, 9, "independent"), (60, 10, "anti")):
        lost = pattern_log(n_runs, 0, gap)
        q = n_runs / len(lost)
        z_truth = -math.sqrt(q * n_runs)
        fit, res = run_both(doc_from_lost(lost))
        assert res["gilbert"]["burstiness"]["approx_z"] == pytest.approx(z_truth, rel=1e-12)
        assert fit_says(fit) == "independent"
        assert bridge_says(res) == want_bridge
        assert (abs(z_truth) >= BRIDGE_CUTOFF) == (want_bridge == "anti")


def test_both_tools_cut_a_messy_trace_into_the_same_pieces():
    """Three sessions, an unknown outcome, a seq gap, a duplicate record and edge runs at a piece end and start.

    Truth: the pieces and complete runs listed by hand below.  Both tools must
    count the same packets and give the same z as hand_z on those runs.
    """
    packets = []
    # session a, piece 1: seq 0..9, runs [2, 1] complete
    for seq, v in enumerate([0, 0, 1, 1, 0, 0, 1, 0, 0, 0]):
        packets.append(_packet("a", seq, "lost" if v else "received"))
    packets.append(_packet("a", 3, "received", duplicate=True))  # a duplicate: left out by both
    packets.append(_packet("a", 10, "unknown"))  # ends piece 1
    # session a, piece 2: seq 11..18, runs [3] complete
    for seq, v in zip(range(11, 19), [0, 1, 1, 1, 0, 0, 0, 0]):
        packets.append(_packet("a", seq, "lost" if v else "received"))
    # seq 19..21 never logged: a seq gap ends piece 2
    # session a, piece 3: seq 22..30, runs [1, 2] complete and an edge run of 1 at the end
    for seq, v in zip(range(22, 31), [0, 1, 0, 0, 1, 1, 0, 0, 1]):
        packets.append(_packet("a", seq, "lost" if v else "received"))
    # session b, one piece: seq 0..7, runs [1, 1, 2] complete
    for seq, v in enumerate([0, 1, 0, 1, 0, 1, 1, 0]):
        packets.append(_packet("b", seq, "lost" if v else "received"))
    # session c, one piece: seq 0..4, an edge run of 2 at the start, then run [1] complete
    for seq, v in enumerate([1, 1, 0, 1, 0]):
        packets.append(_packet("c", seq, "lost" if v else "received"))
    doc = trace_doc(packets)

    complete = [2, 1, 3, 1, 2, 1, 1, 2, 1]
    n_packets = 10 + 8 + 9 + 8 + 5
    n_lost = 3 + 3 + 4 + 4 + 3
    want = hand_z(complete, n_lost, n_packets)

    fit, res = run_both(doc)
    g = fit["gilbert_classic"]
    assert (g["n_sequences"], g["n_packets"], g["n_lost"]) == (5, n_packets, n_lost)
    assert (res["n_pieces"], res["n_packets"], res["n_lost"]) == (5, n_packets, n_lost)
    assert res["runs"]["n_complete"] == len(complete)
    assert res["runs"]["n_edge_excluded"] == 2
    assert fit["independence_check"]["approx_z"] == pytest.approx(want, rel=1e-12)
    assert res["gilbert"]["burstiness"]["approx_z"] == pytest.approx(want, rel=1e-12)


def _chain_log(n: int, p: float, r: float, rng: np.random.Generator) -> list[int]:
    """Two-state chain: P(lost | received) = p, P(received | lost) = r, started from its long-run split."""
    u = rng.random(n + 1)
    state = int(u[0] < p / (p + r))
    out = []
    for x in u[1:]:
        state = int(x < p) if state == 0 else int(x >= r)
        out.append(state)
    return out


def test_weakly_bursty_chain_logs_split_the_tools_only_inside_the_band():
    """Truth: a chain with loss 0.2 and lag-one memory 0.04 (mean loss run 1.30 packets,
    against 1.25 for independent loss at the same rate).

    Over 150 seeded logs of 2000 packets, the tools disagree on some logs, and
    every disagreement has 1.645 < z < 2.0.  The fit's "end below 1" condition
    never decides a log here: z above the cutoff always came with end below 1.
    """
    q, lam = 0.2, 0.04
    p, r = q * (1.0 - lam), (1.0 - q) * (1.0 - lam)
    fit_cut = float(stats.norm.ppf(0.95))
    rng = np.random.default_rng(20261027)
    disagree = 0
    for _ in range(150):
        fit, res = run_both(doc_from_lost(_chain_log(2000, p, r, rng)))
        z = res["gilbert"]["burstiness"]["approx_z"]
        assert fit["independence_check"]["approx_z"] == z
        assert fit_says(fit) == ("bursty" if z > fit_cut else "independent")
        f, b = fit_says(fit), bridge_says(res)
        if f != b:
            disagree += 1
            assert fit_cut < z < BRIDGE_CUTOFF and f == "bursty"
    assert disagree >= 5  # about 14 % of such logs in a 1000-seed scratch sweep


def test_independent_logs_the_fit_calls_bursty_get_a_near_independent_chain():
    """Truth: independent loss at 0.2, so the true burst end probability is 1 (every burst lasts one packet).

    When such a log lands above the fit's cutoff anyway, the fitted end stays
    close to 1, so the burst model it picks is nearly the independent one.
    This is why a false "bursty" from the lower cutoff costs little.
    """
    rng = np.random.default_rng(7)
    ends = []
    for _ in range(600):
        fit, _ = run_both(doc_from_lost((rng.random(2000) < 0.2).astype(int).tolist()))
        if fit["model_choice"]["loss_model"] == "burst":
            ends.append(fit["burst_chain"]["burst_end_probability"]["value"])
    assert len(ends) >= 6  # about 3 % of 600; a 4000-log scratch sweep gave 132
    assert min(ends) > 0.85
