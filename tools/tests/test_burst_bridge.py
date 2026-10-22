"""Tests for tools/burst_bridge.py.

Every expected value here comes from a known generator (GilbertElliott, numpy
Bernoulli draws, a loop written in this file) or from a hand count.  Nothing is
read back from the tool's own formulas except where a test says so.  All data
are synthetic; nothing here is a radio measurement.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sim.radio.channel import GilbertElliott
from tools.burst_bridge import (
    Piece,
    analyze,
    atlas_chain,
    bridge,
    iid_chain,
    loss_runs,
    main,
    match_cells,
    read_echo_sender,
    read_log,
    read_plain_csv,
    read_trace_json,
    slot_chain,
    tracker_cells,
)

SEEDS = range(10)


def piece_from_lost(lost, period_s: float = 0.1) -> Piece:
    lost = [bool(v) for v in lost]
    return Piece(tuple(float(i) * period_s for i in range(len(lost))), tuple(lost))


# Tolerance rule used for every fitted transition probability:
#   |p_hat - p| <= 4 * sqrt(p (1 - p) / n_from_received),  n_from_received = N (1 - loss)
#   |r_hat - r| <= 4 * sqrt(r (1 - r) / n_from_lost),      n_from_lost     = N * loss
# p_hat and r_hat are binomial proportions of Markov transitions out of a
# state, so the standard error is the binomial one; 4 SE keeps a false alarm
# over the whole file well below 1 in 1000.
def se_p(p: float, r: float, n: int) -> float:
    loss = p / (p + r)
    return math.sqrt(p * (1 - p) / (n * (1 - loss)))


def se_r(p: float, r: float, n: int) -> float:
    loss = p / (p + r)
    return math.sqrt(r * (1 - r) / (n * loss))


def assert_fit_close(res: dict, p: float, r: float, n: int) -> None:
    g = res["gilbert"]
    assert abs(g["p_good_to_bad"] - p) <= 4 * se_p(p, r, n)
    assert abs(g["p_bad_to_good"] - r) <= 4 * se_r(p, r, n)


# ----------------------------------------------------------------------------
# 1. parameter recovery
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("p,r", [(0.0789, 0.1842), (0.02, 0.5), (0.15, 0.3)])
def test_gilbert_parameters_recovered_across_seeds(p, r):
    n = 50_000
    ch = GilbertElliott(p, r)
    p_hats, r_hats = [], []
    for seed in SEEDS:
        _, lost = ch.sample(n, np.random.default_rng(seed))
        res = bridge([piece_from_lost(lost)])
        assert_fit_close(res, p, r, n)
        p_hats.append(res["gilbert"]["p_good_to_bad"])
        r_hats.append(res["gilbert"]["p_bad_to_good"])
    # Bias check: the mean of 10 independent estimates has SE / sqrt(10).
    assert abs(np.mean(p_hats) - p) <= 4 * se_p(p, r, n) / math.sqrt(10)
    assert abs(np.mean(r_hats) - r) <= 4 * se_r(p, r, n) / math.sqrt(10)


# ----------------------------------------------------------------------------
# 2. iid limiting case
# ----------------------------------------------------------------------------

def test_iid_loss_mean_run_and_verdict():
    # With independent loss q, a run that has reached length k continues with
    # chance q each packet, so P(length = k) = q^(k-1) (1 - q) and
    # mean = sum_k k q^(k-1) (1 - q) = 1 / (1 - q) = 1/0.7 = 1.428571.
    q = 0.3
    expected = 1.0 / (1.0 - q)
    assert expected == pytest.approx(1.428571, abs=1e-6)
    consistent = 0
    for seed in SEEDS:
        lost = np.random.default_rng(seed).random(200_000) < q
        res = analyze([piece_from_lost(lost, 0.1)])
        runs = res["runs"]
        assert abs(runs["mean_packets"] - expected) <= 0.02 * expected
        assert runs["mean_s"] == pytest.approx(runs["mean_packets"] * 0.1, abs=1e-9)
        consistent += res["verdict"] == "consistent with independent loss"
    assert consistent >= 8


# ----------------------------------------------------------------------------
# 3. hand-built sequence
# ----------------------------------------------------------------------------

def test_hand_built_sequence():
    lost = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    piece = piece_from_lost(lost, 0.1)
    packets, seconds, edge = loss_runs(piece)
    # Runs: indices 1-2 (length 2, next received at index 3: 0.3 - 0.1 = 0.2 s),
    # index 4 (length 1, next received at 5: 0.1 s), indices 7-9 touch the end.
    assert packets == [2, 1]
    assert edge == 1
    assert seconds == pytest.approx([0.2, 0.1], abs=1e-12)
    # Adjacent pairs (9 of them), by hand:
    #   (0,1) R->L  (1,1) L->L  (1,0) L->R  (0,1) R->L  (1,0) L->R
    #   (0,0) R->R  (0,1) R->L  (1,1) L->L  (1,1) L->L
    # R->R 1, R->L 3, L->R 2, L->L 3.
    res = analyze([piece])
    assert res["gilbert"]["transitions"] == {
        "received_to_received": 1, "received_to_lost": 3,
        "lost_to_received": 2, "lost_to_lost": 3,
    }
    assert res["gilbert"]["p_good_to_bad"] == pytest.approx(3 / 4)
    assert res["gilbert"]["p_bad_to_good"] == pytest.approx(2 / 5)
    assert res["loss_rate"] == pytest.approx(0.6)
    assert res["runs"]["n_complete"] == 2
    assert res["runs"]["n_edge_excluded"] == 1


# ----------------------------------------------------------------------------
# 4. converters, checked by brute-force simulation
# ----------------------------------------------------------------------------

def simulate_atlas_burst(loss: float, burst_mean_s: float, rate_hz: float, n: int, seed: int) -> np.ndarray:
    """The burst rule of sim/link.py SimulatedLink._loss_reason, written out here.

    If no burst is running, start one with chance s.  If a burst is running the
    packet is lost, and the burst then ends with chance e.  Here e = 1 / (mean
    burst packets) and s = loss e / (1 - loss (1 - e)) (steady-state solution
    from sim/s6_recovery._burst_start_probability, restated).
    """
    e = 1.0 / (burst_mean_s * rate_hz)
    s = loss * e / (1.0 - loss * (1.0 - e))
    rng = np.random.default_rng(seed)
    u_start = rng.random(n)
    u_end = rng.random(n)
    out = np.zeros(n, dtype=bool)
    active = False
    for i in range(n):
        if not active and u_start[i] < s:
            active = True
        if active:
            out[i] = True
            if u_end[i] < e:
                active = False
    return out


def simulate_slot_chain(loss: float, burst_mean_s: float, slot_s: float, every: int, n: int, seed: int) -> np.ndarray:
    """Slot-level Good/Bad chain, observed every ``every`` slots; Bad means lost."""
    r0 = slot_s / burst_mean_s
    q = loss * r0 / (1.0 - loss)
    rng = np.random.default_rng(seed)
    total = n * every
    u = rng.random(total)
    bad = rng.random() < loss  # stationary start
    states = np.empty(total, dtype=bool)
    for i in range(total):
        states[i] = bad
        bad = (u[i] >= r0) if bad else (u[i] < q)
    return states[::every]


def test_atlas_chain_matches_brute_force_burst_rule():
    n = 300_000
    chain = atlas_chain(0.3, 0.5, 10.0)
    lost = simulate_atlas_burst(0.3, 0.5, 10.0, n, seed=11)
    res = analyze([piece_from_lost(lost)])
    assert_fit_close(res, chain.p, chain.r, n)
    # and the long-run loss is the requested 0.3 (SE of a correlated rate is
    # larger than binomial, so a loose 0.01 band)
    assert abs(res["loss_rate"] - 0.3) < 0.01


@pytest.mark.parametrize("rate_hz,every", [(10.0, 10), (20.0, 5)])
def test_slot_chain_matches_brute_force_subsampling(rate_hz, every):
    n = 300_000
    chain = slot_chain(0.3, 0.5, rate_hz)
    lost = simulate_slot_chain(0.3, 0.5, 0.01, every, n, seed=5 + every)
    res = analyze([piece_from_lost(lost, 1.0 / rate_hz)])
    assert_fit_close(res, chain.p, chain.r, n)


def test_slot_chain_packet_period_equals_slot():
    # k = 1: the packet chain is the slot chain: p = q = 0.3 * 0.02 / 0.7, r = 0.02.
    chain = slot_chain(0.3, 0.5, 100.0)
    assert chain.p == pytest.approx(0.3 * 0.02 / 0.7, abs=1e-12)
    assert chain.r == pytest.approx(0.02, abs=1e-12)


def test_iid_chain_is_bernoulli():
    chain = iid_chain(0.3)
    assert (chain.p, chain.r) == (0.3, 0.7)
    assert chain.loss == pytest.approx(0.3, abs=1e-12)


# ----------------------------------------------------------------------------
# 5. tracker cells
# ----------------------------------------------------------------------------

def test_tracker_cells_contain_atlas_burst_and_match_finds_it():
    cells = tracker_cells(10.0)
    named = [c for c in cells if c["cell"] == "atlas burst 0.30 (0.5 s)"]
    assert len(named) == 1
    assert named[0]["loss"] == pytest.approx(0.3, abs=1e-12)
    top = match_cells(0.3, 0.543, 10.0)[0]
    assert top["cell"] == "atlas burst 0.30 (0.5 s)"


# ----------------------------------------------------------------------------
# 6. readers (synthetic files only)
# ----------------------------------------------------------------------------

def test_plain_csv_without_seq():
    text = "t_s,delivered\n0.0,1\n0.1,0\n0.2,0\n0.3,1\n"
    pieces, _ = read_plain_csv(text)
    assert len(pieces) == 1
    assert pieces[0].lost == (False, True, True, False)
    assert pieces[0].t_s == (0.0, 0.1, 0.2, 0.3)


def test_plain_csv_seq_gap_splits():
    text = "t_s,delivered,seq\n0.0,1,0\n0.1,0,1\n0.5,1,5\n0.6,0,6\n"
    pieces, info = read_plain_csv(text)
    assert len(pieces) == 2
    assert pieces[0].lost == (False, True)
    assert pieces[1].lost == (False, True)
    assert info["seq_gap_breaks"] == 1


def test_trace_json_unknown_duplicate_and_sessions(tmp_path):
    def pkt(seq, t, outcome, **kw):
        return {"sender_session_id": "s1", "seq": seq, "send_time_s": t, "outcome": outcome, **kw}

    raw = {"schema": "zephyr-espnow-trace-1", "packets": [
        pkt(0, 0.0, "received"),
        pkt(1, 0.1, "lost"),
        pkt(1, 0.1, "received", duplicate=True),  # ignored, would otherwise flip packet 1
        pkt(2, 0.2, "unknown"),                   # splits
        pkt(3, 0.3, "received"),
        pkt(4, 0.4, "lost"),
    ]}
    pieces, info = read_trace_json(raw)
    assert [p.lost for p in pieces] == [(False, True), (False, True)]
    assert info["unknown_breaks"] == 1
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    pieces2, info2 = read_log(path)
    assert info2["format"] == "trace"
    assert [p.lost for p in pieces2] == [(False, True), (False, True)]


ECHO_TEXT = """# zephyr-espnow-echo-log-1
# role=sender
# board=A
# peer=B
# session=s1
# firmware=test
seq,tx_us,echo_us
0,1000000,1004000
1,1100000,
2,1200000,1204100
"""


def test_echo_sender_reader_and_autodetect(tmp_path):
    pieces, info = read_echo_sender(ECHO_TEXT)
    assert len(pieces) == 1
    assert pieces[0].lost == (False, True, False)
    assert pieces[0].t_s == pytest.approx((1.0, 1.1, 1.2))
    assert info["loss_kind"] == "round_trip"
    path = tmp_path / "echo.csv"
    path.write_text(ECHO_TEXT, encoding="utf-8")
    pieces2, info2 = read_log(path)
    assert info2["format"] == "echo-sender"
    assert pieces2[0].lost == (False, True, False)


def test_read_log_autodetects_csv(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text("t_s,delivered\n0,1\n0.1,0\n", encoding="utf-8")
    _, info = read_log(path)
    assert info["format"] == "csv"


def test_bad_headers_raise(tmp_path):
    with pytest.raises(ValueError):
        read_plain_csv("time,ok\n0,1\n")
    with pytest.raises(ValueError):
        read_plain_csv("t_s,delivered\n0,2\n")
    with pytest.raises(ValueError):
        read_echo_sender("seq,tx_us,echo_us\n0,1,2\n")
    with pytest.raises(ValueError):
        read_echo_sender(ECHO_TEXT.replace("role=sender", "role=echoer"))
    with pytest.raises(ValueError):
        read_echo_sender(ECHO_TEXT.replace("seq,tx_us,echo_us", "seq,tx_us"))
    with pytest.raises(ValueError):
        read_trace_json({"schema": "wrong", "packets": []})
    path = tmp_path / "x.csv"
    path.write_text("t_s,delivered\n0,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_log(path, "nonsense")


# ----------------------------------------------------------------------------
# 7. CLI
# ----------------------------------------------------------------------------

def test_main_writes_json(tmp_path, capsys):
    _, lost = GilbertElliott(0.0789, 0.1842).sample(5_000, np.random.default_rng(3))
    lines = ["t_s,delivered"] + [f"{i * 0.1:.1f},{0 if v else 1}" for i, v in enumerate(lost)]
    log = tmp_path / "log.csv"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "out.json"
    assert main([str(log), "--json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    for key in ("loss_rate", "runs", "gilbert", "nearest_cells"):
        assert key in data
    assert data["nearest_cells"]
    assert "loss rate" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Critic repair: malformed input is rejected with the line, round-trip caution
# ---------------------------------------------------------------------------

_ECHO_HEAD = ("# zephyr-espnow-echo-log-1\n# role=sender\n# board=A\n# peer=B\n# session=s1\n# firmware=t\n"
              "seq,tx_us,echo_us\n")


@pytest.mark.parametrize(("body", "message"), [
    ("0,1000000,1004000\n0,1100000,\n", "line 9: seq 0 does not follow 0"),
    ("0,1000000,1004000\n2,1100000,\n", "line 9: seq 2 does not follow 0"),
    ("0,1000000,abc\n", "line 8: echo_us must be an integer"),
    ("0,1000000,999999\n", "line 8: echo_us is earlier than tx_us"),
    ("x,1000000,\n", "line 8: seq must be an integer"),
])
def test_echo_sender_rejects_malformed_rows(body, message):
    from tools.burst_bridge import read_echo_sender
    with pytest.raises(ValueError, match=message):
        read_echo_sender(_ECHO_HEAD + body)


def test_plain_csv_rejects_backwards_time_across_a_seq_gap():
    from tools.burst_bridge import read_plain_csv
    with pytest.raises(ValueError, match="line 4: send time goes backwards"):
        read_plain_csv("t_s,delivered,seq\n1.0,1,0\n2.0,1,1\n0.5,1,5\n")


def test_plain_csv_rejects_repeated_seq_and_bad_number():
    from tools.burst_bridge import read_plain_csv
    with pytest.raises(ValueError, match="line 3: seq 0 repeats"):
        read_plain_csv("t_s,delivered,seq\n1.0,1,0\n2.0,1,0\n")
    with pytest.raises(ValueError, match="line 2: t_s must be a number"):
        read_plain_csv("t_s,delivered\nabc,1\n")


def test_bom_is_accepted(tmp_path):
    from tools.burst_bridge import read_log
    path = tmp_path / "bom.csv"
    path.write_bytes("\ufefft_s,delivered\n0.0,1\n0.1,0\n0.2,1\n".encode())
    pieces, info = read_log(path)
    assert info["format"] == "csv" and pieces[0].lost == (False, True, False)


def test_round_trip_report_carries_caution():
    from tools.burst_bridge import bridge, format_report, read_echo_sender
    rows = "".join(f"{i},{1_000_000 + 100_000 * i},{'' if i % 3 == 1 else 1_000_000 + 100_000 * i + 4000}\n"
                   for i in range(200))
    pieces, info = read_echo_sender(_ECHO_HEAD + rows)
    info["format"] = "echo-sender"
    text = format_report(bridge(pieces, info))
    assert "CAUTION: this log is round-trip loss" in text


def test_independent_verdict_flags_few_runs():
    from tools.burst_bridge import Piece, analyze
    lost = [False, True, False, False, True, False, True, False, False, False] * 3
    res = analyze([Piece(tuple(0.1 * i for i in range(len(lost))), tuple(lost))])
    assert "weak test" in res["verdict"]


def test_radio_table_run_seed_is_deterministic(monkeypatch):
    import tools.burst_bridge_radio_table as table
    monkeypatch.setattr(table, "DURATION_S", 5.0)
    a = table.run_seed("unicast", "time_correlated", 0.5, 540.0, 123)
    b = table.run_seed("unicast", "time_correlated", 0.5, 540.0, 123)
    assert a == b and len(a.lost) == 50
