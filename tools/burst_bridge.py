#!/usr/bin/env python3
"""Compare a packet-loss log with the tracker's burst-loss settings.

Usage (repo root):

    PYTHONPATH=. ./.venv/bin/python tools/burst_bridge.py LOG [--format auto] [--json OUT]

LOG is one of:

- ``csv``: a plain CSV with a header row ``t_s,delivered`` (send time in
  seconds, 1 if the packet arrived, 0 if it was lost), one row per packet sent,
  in send order.  Optional third column ``seq``; a jump in ``seq`` splits the
  log into separate pieces.
- ``trace``: a ``zephyr-espnow-trace-1`` JSON file (sim/trace.py), including
  traces written by the ESP-NOW echo importer.  Only the packet fields the
  loss analysis needs are read.  Unknown outcomes and sequence gaps split the
  log; duplicate records are left out.
- ``echo-sender``: the sender CSV of the ESP-NOW echo log format
  (``# zephyr-espnow-echo-log-1``, columns ``seq,tx_us,echo_us``).  An empty
  ``echo_us`` means no echo came back, so this is ROUND-TRIP loss (either leg).
  Use the trace importer with the echoer log for one-way loss.

What it reports:

- loss rate;
- loss runs (a run is a stretch of lost packets with a received packet on
  each side; runs touching the start or end of a piece are left out and
  counted): mean and 95th percentile, in packets and in seconds (time from the
  first lost packet's send to the next received packet's send);
- a fitted classic Gilbert chain (no loss in Good, every packet lost in Bad)
  from transition counts, reusing ``fit_gilbert`` from
  tools/calibrate_radio_from_trace.py;
- the same quantities predicted by every tracker burst setting (S6 atlas and
  mission presets) at the log's packet rate, and the nearest one.

All tracker settings here are simulated.  A match says which simulated cell
the log's loss pattern most resembles; it says nothing about how the tracker
would do on the real link.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.calibrate_radio_from_trace import fit_gilbert

ECHO_MAGIC = "# zephyr-espnow-echo-log-1"
TRACE_SCHEMA = "zephyr-espnow-trace-1"
MISSION_SLOT_S = 0.01  # sim/missions/comms.py LossProcess slot (EXP-029, EXP-031)
# z cutoff for "looks independent": |approx_z| below this means the observed
# mean loss run is within about 2 standard errors of the independent-loss value.
IID_Z_CUTOFF = 2.0
VERDICT_IID = "consistent with independent loss"
VERDICT_BURSTY = "bursty (loss runs longer than independent loss would give)"
VERDICT_ANTI = "anti-bursty (loss runs shorter than independent loss would give)"
VERDICT_UNKNOWN = "not enough complete loss runs to tell"
MIN_RUNS_FOR_POWER = 30  # below this many complete runs, "independent" is flagged as a weak test


# ----------------------------------------------------------------------------
# loss sequences
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Piece:
    """One unbroken stretch of packets: send times (s) and lost flags, in send order."""

    t_s: tuple[float, ...]
    lost: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.t_s) != len(self.lost):
            raise ValueError("t_s and lost must have the same length")
        for a, b in zip(self.t_s[:-1], self.t_s[1:]):
            if not b >= a:
                raise ValueError("send times must be nondecreasing within a piece")


def _split_pieces(rows: Iterable[tuple[float, bool | None, int | None, str]]) -> tuple[list[Piece], dict]:
    """Rows of (send time, lost or None for unknown, seq or None, where) -> unbroken pieces.

    ``where`` names the row (file line or packet index) for error messages.
    An unknown outcome or a forward jump in seq ends a piece, so no transition
    is ever counted across a packet whose fate is not known.  A seq that
    repeats or goes backwards, or a send time that goes backwards anywhere in
    the log, is an error: the log is out of order and cannot be trusted.
    """
    pieces: list[Piece] = []
    t_cur: list[float] = []
    l_cur: list[bool] = []
    prev_seq: int | None = None
    prev_t: float | None = None
    breaks = {"unknown_breaks": 0, "seq_gap_breaks": 0}

    def flush() -> None:
        if t_cur:
            pieces.append(Piece(tuple(t_cur), tuple(l_cur)))
            t_cur.clear()
            l_cur.clear()

    for t, lost, seq, where in rows:
        if not math.isfinite(t) or t < 0:
            raise ValueError(f"{where}: send time must be finite and nonnegative")
        if prev_t is not None and t < prev_t:
            raise ValueError(f"{where}: send time goes backwards ({t} after {prev_t})")
        if seq is not None and prev_seq is not None and seq <= prev_seq:
            raise ValueError(f"{where}: seq {seq} repeats or goes backwards (previous {prev_seq})")
        prev_t = t
        if lost is None:
            flush()
            breaks["unknown_breaks"] += 1
            prev_seq = seq
            continue
        if seq is not None and prev_seq is not None and seq != prev_seq + 1:
            flush()
            breaks["seq_gap_breaks"] += 1
        t_cur.append(float(t))
        l_cur.append(bool(lost))
        prev_seq = seq
    flush()
    return pieces, breaks


def _int(text: str, where: str, field: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        raise ValueError(f"{where}: {field} must be an integer, got {text!r}") from None


def _float(text: str, where: str, field: str) -> float:
    try:
        return float(text.strip())
    except ValueError:
        raise ValueError(f"{where}: {field} must be a number, got {text!r}") from None


def read_plain_csv(text: str) -> tuple[list[Piece], dict]:
    """Plain CSV ``t_s,delivered[,seq]``.  A forward seq jump splits the log; see _split_pieces."""
    reader = csv.reader(io.StringIO(text.removeprefix("\ufeff")))
    header = [h.strip() for h in next(reader, [])]
    if header[:2] != ["t_s", "delivered"] or len(header) > 3 or (len(header) == 3 and header[2] != "seq"):
        raise ValueError("plain CSV header must be t_s,delivered or t_s,delivered,seq")
    rows = []
    for line_no, row in enumerate(reader, start=2):
        where = f"line {line_no}"
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"{where}: expected {len(header)} columns")
        flag = row[1].strip()
        if flag not in ("0", "1"):
            raise ValueError(f"{where}: delivered must be 0 or 1")
        seq = _int(row[2], where, "seq") if len(header) == 3 else None
        if seq is not None and seq < 0:
            raise ValueError(f"{where}: seq must be nonnegative")
        rows.append((_float(row[0], where, "t_s"), flag == "0", seq, where))
    return _split_pieces(rows)


def read_trace_json(raw: dict) -> tuple[list[Piece], dict]:
    """Loss pieces from a zephyr-espnow-trace-1 dict, one sender session at a time.

    Records flagged ``duplicate`` are skipped.  Within a session, packets are
    sorted by seq; a seq that repeats on a non-duplicate record, or a send time
    that goes backwards in seq order, raises ValueError (see _split_pieces).
    """
    if raw.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"trace schema must be {TRACE_SCHEMA}")
    packets = raw.get("packets")
    if not isinstance(packets, list):
        raise TypeError("trace has no packet list")
    by_session: dict[str, list[tuple[int, dict]]] = {}
    for i, pkt in enumerate(packets):
        if not isinstance(pkt, dict):
            raise TypeError(f"packet {i} is not an object")
        if pkt.get("duplicate") is True:
            continue
        for key in ("sender_session_id", "seq", "send_time_s", "outcome"):
            if key not in pkt:
                raise ValueError(f"packet {i} is missing {key}")
        if pkt["outcome"] not in ("received", "lost", "unknown"):
            raise ValueError(f"packet {i} has outcome {pkt['outcome']!r}")
        by_session.setdefault(str(pkt["sender_session_id"]), []).append((i, pkt))
    pieces: list[Piece] = []
    breaks = {"unknown_breaks": 0, "seq_gap_breaks": 0, "sessions": len(by_session)}
    for session in sorted(by_session):
        pkts = sorted(by_session[session], key=lambda ip: (int(ip[1]["seq"]), float(ip[1]["send_time_s"])))
        rows = [(float(p["send_time_s"]), None if p["outcome"] == "unknown" else p["outcome"] == "lost",
                 int(p["seq"]), f"packet {i} (session {session})") for i, p in pkts]
        got, b = _split_pieces(rows)
        pieces.extend(got)
        breaks["unknown_breaks"] += b["unknown_breaks"]
        breaks["seq_gap_breaks"] += b["seq_gap_breaks"]
    return pieces, breaks


def read_echo_sender(text: str) -> tuple[list[Piece], dict]:
    """Round-trip loss from an echo-log sender CSV.  Times are tx_us / 1e6 on the sender's clock.

    The format says seq rises by exactly 1 per row, so any other step is an
    error here (stricter than the plain CSV, which splits on a forward jump).
    """
    lines = text.removeprefix("\ufeff").splitlines()
    if not lines or lines[0].strip() != ECHO_MAGIC:
        raise ValueError(f"echo sender log must start with {ECHO_MAGIC}")
    header: dict[str, str] = {}
    body_start = None
    for i, line in enumerate(lines[1:], start=1):
        if line.startswith("#"):
            key, sep, value = line[1:].strip().partition("=")
            if sep:
                header[key.strip()] = value.strip()
            continue
        body_start = i
        break
    if header.get("role", "sender") != "sender":
        raise ValueError("this is not a sender log (role is not sender)")
    if body_start is None or lines[body_start].strip() != "seq,tx_us,echo_us":
        raise ValueError("sender log columns must be seq,tx_us,echo_us")
    rows = []
    prev_seq = None
    for line_no, line in enumerate(lines[body_start + 1:], start=body_start + 2):
        where = f"line {line_no}"
        if not line.strip():
            raise ValueError(f"{where}: blank line")
        parts = line.split(",")
        if len(parts) != 3:
            raise ValueError(f"{where}: expected 3 columns")
        seq = _int(parts[0], where, "seq")
        if prev_seq is not None and seq != prev_seq + 1:
            raise ValueError(f"{where}: seq {seq} does not follow {prev_seq} (the format requires +1 per row)")
        prev_seq = seq
        tx_us = _int(parts[1], where, "tx_us")
        echoed = parts[2].strip() != ""
        if echoed and _int(parts[2], where, "echo_us") < tx_us:
            raise ValueError(f"{where}: echo_us is earlier than tx_us")
        rows.append((tx_us / 1e6, not echoed, seq, where))
    pieces, breaks = _split_pieces(rows)
    breaks["loss_kind"] = "round_trip"
    return pieces, breaks


def read_log(path: str | Path, fmt: str = "auto") -> tuple[list[Piece], dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    if fmt == "auto":
        stripped = text.lstrip()
        if stripped.startswith("{"):
            fmt = "trace"
        elif text.startswith(ECHO_MAGIC):
            fmt = "echo-sender"
        else:
            fmt = "csv"
    if fmt == "trace":
        pieces, info = read_trace_json(json.loads(text))
        info.setdefault("loss_kind", "one_way")
    elif fmt == "echo-sender":
        pieces, info = read_echo_sender(text)
    elif fmt == "csv":
        pieces, info = read_plain_csv(text)
        info.setdefault("loss_kind", "as_logged")
    else:
        raise ValueError("format must be auto, csv, trace, or echo-sender")
    info["format"] = fmt
    return pieces, info


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------

def loss_runs(piece: Piece) -> tuple[list[int], list[float], int]:
    """Complete loss runs of one piece: lengths in packets, durations in seconds, and the edge-run count.

    A run's duration is the send time of the next received packet minus the
    send time of the run's first lost packet: how long the receiver heard
    nothing new.  At a steady rate that is (run length) x (packet period).
    """
    n = len(piece.lost)
    packets: list[int] = []
    seconds: list[float] = []
    edge = 0
    i = 0
    while i < n:
        if piece.lost[i]:
            j = i
            while j < n and piece.lost[j]:
                j += 1
            if i == 0 or j == n:
                edge += 1
            else:
                packets.append(j - i)
                seconds.append(piece.t_s[j] - piece.t_s[i])
            i = j
        else:
            i += 1
    return packets, seconds, edge


def packet_period_s(pieces: Sequence[Piece]) -> float | None:
    gaps = np.concatenate([np.diff(np.asarray(p.t_s)) for p in pieces if len(p.t_s) > 1] or [np.empty(0)])
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if gaps.size else None


def _f(x: Any) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else float(x)


def analyze(pieces: Sequence[Piece]) -> dict:
    """Loss rate, loss-run statistics and the fitted Gilbert chain for a list of pieces."""
    run_p: list[int] = []
    run_s: list[float] = []
    edge = 0
    for piece in pieces:
        p, s, e = loss_runs(piece)
        run_p.extend(p)
        run_s.extend(s)
        edge += e
    gilbert = fit_gilbert([[1 if v else 0 for v in piece.lost] for piece in pieces])
    period = packet_period_s(pieces)
    out = {
        "n_pieces": len(pieces),
        "n_packets": gilbert["n_packets"],
        "n_lost": gilbert["n_lost"],
        "loss_rate": gilbert["loss_rate"],
        "packet_period_s": period,
        "packet_rate_hz": _f(1.0 / period) if period else None,
        "runs": {
            "n_complete": len(run_p),
            "n_edge_excluded": edge,
            "mean_packets": _f(np.mean(run_p)) if run_p else None,
            "p95_packets": _f(np.percentile(run_p, 95)) if run_p else None,
            "max_packets": int(max(run_p)) if run_p else None,
            "mean_s": _f(np.mean(run_s)) if run_s else None,
            "p95_s": _f(np.percentile(run_s, 95)) if run_s else None,
            "max_s": _f(max(run_s)) if run_s else None,
        },
        "gilbert": {k: gilbert[k] for k in ("p_good_to_bad", "p_good_to_bad_ci", "p_bad_to_good",
                                             "p_bad_to_good_ci", "implied_mean_burst_packets", "transitions",
                                             "burstiness", "model")},
    }
    out["verdict"] = _verdict(gilbert["burstiness"]["approx_z"])
    if out["verdict"] == VERDICT_IID and len(run_p) < MIN_RUNS_FOR_POWER:
        out["verdict"] += f" (only {len(run_p)} complete runs: weak test)"
    if len(pieces) > 1:
        out["per_piece"] = _per_piece(pieces)
    return out


def _verdict(z: float | None) -> str:
    """Burstiness verdict from the approx z score of the mean complete loss run.

    "Consistent with independent loss" only means the test did not see
    clumping; with few complete runs it has little power, so check n_complete.
    """
    if z is None:
        return VERDICT_UNKNOWN
    if abs(z) < IID_Z_CUTOFF:
        return VERDICT_IID
    return VERDICT_BURSTY if z > 0 else VERDICT_ANTI


def _per_piece(pieces: Sequence[Piece]) -> dict:
    """Each piece on its own: loss spread and verdict counts.

    Pooling pieces with different loss rates makes the pooled loss runs look
    burstier than any single piece (a stretch at 90 % loss next to one at 0 %
    reads as clumped).  If the pooled verdict is bursty but most pieces say
    independent, the clumping is between pieces, not within them.
    """
    losses = []
    counts = {VERDICT_IID: 0, VERDICT_BURSTY: 0, VERDICT_ANTI: 0, VERDICT_UNKNOWN: 0}
    for piece in pieces:
        if not piece.lost:
            continue
        losses.append(sum(piece.lost) / len(piece.lost))
        g = fit_gilbert([[1 if v else 0 for v in piece.lost]])
        counts[_verdict(g["burstiness"]["approx_z"])] += 1
    return {"n": len(losses), "loss_min": _f(min(losses)) if losses else None,
            "loss_max": _f(max(losses)) if losses else None, "verdict_counts": counts}


# ----------------------------------------------------------------------------
# tracker units <-> packet-level Gilbert chain
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PacketChain:
    """Classic Gilbert chain on the lost/received sequence, one step per packet.

    p = P(lost | previous received), r = P(received | previous lost).
    Long-run loss p / (p + r); mean loss run 1 / r packets.
    """

    p: float
    r: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0 and 0.0 <= self.r <= 1.0) or self.p + self.r <= 0:
            raise ValueError("p and r must be probabilities, not both zero")

    @property
    def loss(self) -> float:
        return self.p / (self.p + self.r)

    @property
    def mean_run_packets(self) -> float:
        return math.inf if self.r == 0 else 1.0 / self.r

    def mean_run_s(self, rate_hz: float) -> float:
        return self.mean_run_packets / float(rate_hz)


def iid_chain(loss: float) -> PacketChain:
    """Independent loss: every packet lost with the same chance, so p = loss and r = 1 - loss."""
    if not 0.0 <= loss < 1.0:
        raise ValueError("loss must be in [0, 1)")
    return PacketChain(loss, 1.0 - loss)


def atlas_chain(loss: float, burst_mean_s: float, rate_hz: float) -> PacketChain:
    """The S6 atlas burst model (sim/link.py SimulatedLink) seen in a packet log.

    The atlas sets burst end e = 1 / (burst_mean_s x rate) and solves the start
    chance s so the long-run loss is ``loss`` (sim/s6_recovery.py
    _burst_start_probability).  A burst can end after a lost packet and a new
    one start on the very next packet, so the observed chain is
    P(lost | received) = s, P(received | lost) = e (1 - s).
    """
    from sim.s6_recovery import _burst_start_probability

    e = 1.0 / (float(burst_mean_s) * float(rate_hz))
    if not 0.0 < e <= 1.0:
        raise ValueError("burst_mean_s x rate_hz must be at least one packet")
    s = _burst_start_probability(float(loss), e)
    return PacketChain(s, e * (1.0 - s))


def slot_chain(loss: float, burst_mean_s: float, rate_hz: float, slot_s: float = MISSION_SLOT_S) -> PacketChain:
    """The mission LossProcess (EXP-029, EXP-031) seen by a stream sending at ``rate_hz``.

    The slot chain moves Good -> Bad with q = loss x r0 / (1 - loss) and Bad ->
    Good with r0 = slot_s / burst_mean_s every slot.  Looking only every
    k = (1 / rate) / slot_s slots, a two-state chain is again a two-state chain
    with the same long-run split; its 'memory' factor lambda = 1 - q - r0 is
    raised to the power k.  So P(received | lost) = (1 - loss)(1 - lambda^k)
    and P(lost | received) = loss (1 - lambda^k).
    """
    loss = float(loss)
    r0 = slot_s / float(burst_mean_s)
    q = loss * r0 / (1.0 - loss)
    if not (0.0 < r0 <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError("burst parameters give invalid slot probabilities")
    k = (1.0 / float(rate_hz)) / slot_s
    lam = 1.0 - q - r0
    if lam < 0.0 and abs(k - round(k)) > 1e-9:
        raise ValueError("slot chain flips state every slot; a non-integer slots-per-packet is undefined")
    lam_k = lam ** round(k) if lam < 0.0 else lam ** k
    return PacketChain(loss * (1.0 - lam_k), (1.0 - loss) * (1.0 - lam_k))


def tracker_cells(rate_hz: float) -> list[dict]:
    """Every tracker loss setting, as the packet chain it would show at ``rate_hz``."""
    from sim.missions.presets import PRESET_NAMES, get_preset
    from tools.run_s6_atlas import LOSSES

    atlas_burst_s = 0.5  # tools/run_s6_atlas.py: burst mean fixed at 0.5 s
    cells = []
    for model, loss in LOSSES:
        if loss == 0.0:
            continue
        name = f"atlas {model} {loss:.2f}" + (f" ({atlas_burst_s} s)" if model == "burst" else "")
        chain = iid_chain(loss) if model == "independent" else atlas_chain(loss, atlas_burst_s, rate_hz)
        cells.append({"cell": name, "family": "atlas", "chain": chain})
    for name in PRESET_NAMES:
        pre = get_preset(name)
        if pre.loss_model == "none":
            continue
        chain = (iid_chain(pre.loss_probability) if pre.loss_model == "independent"
                 else slot_chain(pre.loss_probability, pre.burst_mean_s, rate_hz))
        cells.append({"cell": f"mission {name}", "family": "mission", "chain": chain})
    for c in cells:
        ch = c.pop("chain")
        c.update(p=ch.p, r=ch.r, loss=ch.loss, mean_run_packets=ch.mean_run_packets,
                 mean_run_s=ch.mean_run_s(rate_hz))
    return cells


def match_cells(loss: float, mean_run_s: float, rate_hz: float) -> list[dict]:
    """Tracker cells sorted by distance to the log in (loss, log2 run length).

    Distance = sqrt(((loss - cell loss) / 0.1)^2 + log2(run / cell run)^2):
    10 points of loss count the same as a factor of 2 in run length.  It is a
    ranking rule for reading the grid, not a statistical test.
    """
    out = []
    for c in tracker_cells(rate_hz):
        d = math.hypot((loss - c["loss"]) / 0.1, math.log2(mean_run_s / c["mean_run_s"]))
        out.append(dict(c, distance=d))
    return sorted(out, key=lambda c: c["distance"])


def bridge(pieces: Sequence[Piece], info: dict | None = None) -> dict:
    res = analyze(pieces)
    res["input"] = dict(info or {})
    g = res["gilbert"]
    rate = res["packet_rate_hz"]
    if g["p_good_to_bad"] is not None and g["p_bad_to_good"] and rate:
        chain = PacketChain(g["p_good_to_bad"], g["p_bad_to_good"])
        res["fitted_tracker_units"] = {"loss": chain.loss, "mean_run_packets": chain.mean_run_packets,
                                       "mean_run_s": chain.mean_run_s(rate), "rate_hz": rate}
        res["nearest_cells"] = match_cells(chain.loss, chain.mean_run_s(rate), rate)[:3]
    else:
        res["fitted_tracker_units"] = None
        res["nearest_cells"] = []
    return res


def format_report(res: dict) -> str:
    r = res["runs"]

    def num(x, fmt=".3f"):
        return "n/a" if x is None else format(x, fmt)

    lines = [
        (f"input: {res['input'].get('format', '?')} ({res['input'].get('loss_kind', '?')} loss), "
         f"{res['n_packets']} packets in {res['n_pieces']} piece(s), rate {num(res['packet_rate_hz'], '.2f')} Hz"),
        f"loss rate: {num(res['loss_rate'])}",
        (f"loss runs: {r['n_complete']} complete ({r['n_edge_excluded']} at edges left out); "
         f"mean {num(r['mean_packets'], '.2f')} packets / {num(r['mean_s'])} s; "
         f"p95 {num(r['p95_packets'], '.1f')} packets / {num(r['p95_s'])} s"),
    ]
    b = res["gilbert"]["burstiness"]
    lines.append(f"independent loss would give mean run {num(b['independent_expected_mean_loss_run'], '.2f')} "
                 f"packets; approx z {num(b['approx_z'], '.1f')}: {res['verdict']}")
    pp = res.get("per_piece")
    if pp:
        c = pp["verdict_counts"]
        lines.append(f"per piece: loss {num(pp['loss_min'])} to {num(pp['loss_max'])}; "
                     f"{c[VERDICT_BURSTY]} bursty, {c[VERDICT_IID]} independent, {c[VERDICT_ANTI]} anti-bursty, "
                     f"{c[VERDICT_UNKNOWN]} too short (if pieces disagree with the pooled verdict, "
                     f"analyse them separately)")
    g = res["gilbert"]
    lines.append(f"Gilbert fit: p (good->bad) {num(g['p_good_to_bad'], '.4f')}, "
                 f"r (bad->good) {num(g['p_bad_to_good'], '.4f')}")
    if res["nearest_cells"]:
        lines.append("nearest tracker cells, compared on the FITTED chain (loss p/(p+r), mean run 1/r); "
                     "simulated settings; distance 1 = 10 points of loss or 2x run length:")
        if res["input"].get("loss_kind") == "round_trip":
            lines.append("  CAUTION: this log is round-trip loss (either leg), the tracker cells are one-way loss; "
                         "feed the importer's trace JSON for one-way loss")
        for c in res["nearest_cells"]:
            lines.append(f"  {c['cell']}: loss {c['loss']:.3f}, mean run {c['mean_run_packets']:.2f} packets "
                         f"/ {c['mean_run_s']:.3f} s, distance {c['distance']:.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log")
    ap.add_argument("--format", default="auto", choices=("auto", "csv", "trace", "echo-sender"))
    ap.add_argument("--json", help="also write the full result as JSON here")
    args = ap.parse_args(argv)
    pieces, info = read_log(args.log, args.format)
    res = bridge(pieces, info)
    print(format_report(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
