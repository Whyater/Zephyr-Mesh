#!/usr/bin/env python3
"""Fit radio-model parameters from ESP-NOW traces (EXP-018).

Input is a calibration manifest that lists one validated ``zephyr-espnow-trace-1``
trace per fixed position::

    {"entries": [{"trace": "p1.json", "distance_m": 5.0, "walls": 0, "bodies": 0,
                  "position_id": "p1"}], "notes": "..."}

What is fitted, and what is not:

- Path loss.  The mean RSSI of the received packets at each position is one
  data point.  Mean RSSI = A - n x with x = 10 log10(d / 1 m), fitted by
  ordinary least squares over positions.  Shadowing is one fixed draw per
  position, so the packets at one position are not independent evidence about
  n; only the number of positions is.  A is the RSSI at 1 m, which is transmit
  power plus antenna gains minus PL0.  PL0 alone cannot be separated from A
  without a known transmit power and antenna gain, so it is not reported.
  Only line-of-sight entries (walls = 0, bodies = 0) enter the fit; the other
  entries are listed with their mean RSSI and are not used.
- Delivery.  received / (received + lost) with a Wilson interval.  Packets
  with outcome "unknown" are left out of the denominator and counted and
  reported separately.  Nothing is imputed.
- Burst loss.  The classic Gilbert chain (no loss in Good, all lost in Bad) is
  fitted from transition counts on the lost/received sequence.  An unknown
  outcome or a gap in the sequence numbers ends a sequence, so no transition is
  ever counted across a packet whose fate is not known.

Limits to keep in mind: the mean RSSI of received packets is biased upward
where a large share of packets are lost near the receiver sensitivity
(censoring), and the residual spread around the fitted line is shadowing plus
any lack of fit (for example a dual-slope curve).  A result is called
"synthetic" if any trace is synthetic; nothing fitted from synthetic data is
measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from sim.radio.channel import GilbertElliott
from sim.s6_recovery import wilson_interval
from sim.trace import TRACE_SCHEMA, EspNowTrace

TOOL_NAME = "calibrate_radio_from_trace"
RESULT_SCHEMA = "zephyr-radio-calibration-1"
PROVENANCE_STATUS = "fitted from the traces listed; status of each trace copied"
PL0_NOTE = ("A is the RSSI at 1 m (transmit power + antenna gains - PL0). PL0 alone is not identifiable "
            "without a known transmit power and antenna gain.")
SIZING_PROBABILITIES = (0.5, 0.9, 0.95, 0.99)
SIZING_HALFWIDTHS = (0.01, 0.005)
CENSORING_WARN_DELIVERY = 0.9


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def _f(x: Any) -> float | None:
    """Plain float (or None) so the result is JSON-safe with allow_nan=False."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("non-finite number in calibration result")
    return x


def _interval(pair: tuple[float, float] | None) -> list[float | None] | None:
    return None if pair is None else [_f(pair[0]), _f(pair[1])]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------
# path loss
# ----------------------------------------------------------------------------

def fit_path_loss(distances_m: Sequence[float], mean_rssi_dbm: Sequence[float], confidence: float = 0.95) -> dict:
    """Ordinary least squares of per-position mean RSSI on x = 10 log10(d / 1 m).

    mean_rssi = A - n x.  One point per position.  Needs at least 3 positions
    with distinct distances; otherwise ``identifiable`` is False and the
    estimates are None.  ``n`` has a t interval with k - 2 degrees of freedom;
    ``shadowing_sigma_db`` is the residual standard deviation (ddof 2) with a
    chi-square interval.
    """
    d = np.asarray(distances_m, dtype=float)
    y = np.asarray(mean_rssi_dbm, dtype=float)
    if d.ndim != 1 or d.shape != y.shape:
        raise ValueError("distances_m and mean_rssi_dbm must be 1-D arrays of the same length")
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(y))):
        raise ValueError("distances and RSSI means must be finite")
    if np.any(d <= 0):
        raise ValueError("distances must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    k = int(d.size)
    n_distinct = int(np.unique(d).size)
    out: dict[str, Any] = {
        "identifiable": False, "reason": None, "k_positions": k, "n_distinct_distances": n_distinct,
        "df": max(k - 2, 0), "confidence": confidence, "model": "mean_rssi_dbm = A - n * 10*log10(d / 1 m)",
        "a_dbm_at_1m": None, "a_ci": None, "path_loss_exponent": None, "path_loss_exponent_ci": None,
        "path_loss_exponent_se": None, "shadowing_sigma_db": None, "shadowing_sigma_ci": None,
        "residuals_db": None, "note": PL0_NOTE,
    }
    if k < 3 or n_distinct < 3:
        out["reason"] = "path-loss exponent not identifiable: need at least 3 positions with distinct distances"
        return out
    x = 10.0 * np.log10(d)
    xm = float(x.mean())
    sxx = float(np.sum((x - xm) ** 2))
    slope = float(np.sum((x - xm) * (y - float(y.mean()))) / sxx)
    intercept = float(y.mean()) - slope * xm
    resid = y - (intercept + slope * x)
    df = k - 2
    ssr = float(np.sum(resid ** 2))
    s = math.sqrt(ssr / df)
    tcrit = float(stats.t.ppf(0.5 + confidence / 2.0, df))
    se_slope = s / math.sqrt(sxx)
    se_int = s * math.sqrt(1.0 / k + xm * xm / sxx)
    n_hat = -slope
    alpha = 1.0 - confidence
    sig_ci = (s * math.sqrt(df / float(stats.chi2.ppf(1.0 - alpha / 2.0, df))),
              s * math.sqrt(df / float(stats.chi2.ppf(alpha / 2.0, df))))
    out.update({
        "identifiable": True,
        "a_dbm_at_1m": _f(intercept), "a_ci": _interval((intercept - tcrit * se_int, intercept + tcrit * se_int)),
        "path_loss_exponent": _f(n_hat),
        "path_loss_exponent_ci": _interval((n_hat - tcrit * se_slope, n_hat + tcrit * se_slope)),
        "path_loss_exponent_se": _f(se_slope),
        "shadowing_sigma_db": _f(s), "shadowing_sigma_ci": _interval(sig_ci),
        "residuals_db": [_f(v) for v in resid],
    })
    return out


# ----------------------------------------------------------------------------
# Gilbert chain
# ----------------------------------------------------------------------------

def _runs_of_loss(seq: Sequence[int]) -> tuple[list[int], int]:
    """Lengths of complete loss runs (bounded by a received packet on both sides) and the count of edge runs left out."""
    complete: list[int] = []
    edge = 0
    n = len(seq)
    i = 0
    while i < n:
        if seq[i]:
            j = i
            while j < n and seq[j]:
                j += 1
            if i == 0 or j == n:
                edge += 1
            else:
                complete.append(j - i)
            i = j
        else:
            i += 1
    return complete, edge


def fit_gilbert(lost_sequences: Iterable[Sequence[int]], confidence: float = 0.95) -> dict:
    """Classic Gilbert chain from transition counts.

    ``lost_sequences`` is a list of sequences of 0 (received) and 1 (lost).
    Each sequence must be unbroken: no unknown outcome and no sequence-number
    gap inside it.  p = #(received -> lost) / #(received followed by anything);
    r = #(lost -> received) / #(lost followed by anything); each with a Wilson
    interval.  Also reports the observed mean complete loss-run length against
    the independent-loss expectation 1 / (1 - loss_rate), with an approximate z
    score (geometric run lengths under independence).
    """
    n_rr = n_rl = n_lr = n_ll = 0
    n_packets = n_lost = n_seq = 0
    runs: list[int] = []
    edge_runs = 0
    for raw in lost_sequences:
        seq = [1 if v else 0 for v in raw]
        if not seq:
            continue
        n_seq += 1
        n_packets += len(seq)
        n_lost += sum(seq)
        for a, b in zip(seq[:-1], seq[1:]):
            if a == 0 and b == 0:
                n_rr += 1
            elif a == 0:
                n_rl += 1
            elif b == 0:
                n_lr += 1
            else:
                n_ll += 1
        complete, edge = _runs_of_loss(seq)
        runs.extend(complete)
        edge_runs += edge
    from_rx = n_rr + n_rl
    from_lost = n_lr + n_ll
    p_hat = n_rl / from_rx if from_rx else None
    r_hat = n_lr / from_lost if from_lost else None
    out: dict[str, Any] = {
        "confidence": confidence, "n_sequences": n_seq, "n_packets": n_packets, "n_lost": n_lost,
        "loss_rate": _f(n_lost / n_packets) if n_packets else None,
        "transitions": {"received_to_received": n_rr, "received_to_lost": n_rl,
                        "lost_to_received": n_lr, "lost_to_lost": n_ll},
        "denominator_from_received": from_rx, "denominator_from_lost": from_lost,
        "p_good_to_bad": _f(p_hat), "p_good_to_bad_ci": _interval(wilson_interval(n_rl, from_rx, confidence)) if from_rx else None,
        "p_bad_to_good": _f(r_hat), "p_bad_to_good_ci": _interval(wilson_interval(n_lr, from_lost, confidence)) if from_lost else None,
        "implied_mean_burst_packets": _f(1.0 / r_hat) if r_hat else None,
        "burstiness": _burstiness(runs, edge_runs, n_lost, n_packets),
        "model": "classic Gilbert chain (no loss in Good, all lost in Bad); transition counts, Wilson intervals",
    }
    return out


def _burstiness(runs: list[int], edge_runs: int, n_lost: int, n_packets: int) -> dict:
    res: dict[str, Any] = {
        "n_complete_loss_runs": len(runs), "n_edge_loss_runs_excluded": edge_runs,
        "observed_mean_loss_run": None, "independent_expected_mean_loss_run": None,
        "ratio_observed_to_independent": None, "approx_z": None,
        "note": "complete runs only (bounded by a received packet on both sides); z is approximate (geometric run lengths under independent loss)",
    }
    if not runs or not n_packets:
        return res
    q = n_lost / n_packets
    obs = float(np.mean(runs))
    res["observed_mean_loss_run"] = _f(obs)
    if q < 1.0:
        exp = 1.0 / (1.0 - q)
        res["independent_expected_mean_loss_run"] = _f(exp)
        res["ratio_observed_to_independent"] = _f(obs / exp)
        se = math.sqrt(q) / (1.0 - q) / math.sqrt(len(runs))
        res["approx_z"] = _f((obs - exp) / se) if se > 0 else None
    return res


# ----------------------------------------------------------------------------
# one trace
# ----------------------------------------------------------------------------

def split_sequences(trace: EspNowTrace) -> tuple[list[list[int]], dict]:
    """Lost(1)/received(0) sequences per sender session.

    Packets are ordered by seq within each ``sender_session_id``.  Unknown
    outcomes and sequence-number gaps end a sequence.  Packets flagged
    ``duplicate`` are left out (they are not distinct logical packets).
    """
    by_session: dict[str, list] = {}
    for pkt in trace.packets:
        if pkt.duplicate is True:
            continue
        by_session.setdefault(pkt.sender_session_id, []).append(pkt)
    sequences: list[list[int]] = []
    breaks_unknown = 0
    breaks_gap = 0
    for session in sorted(by_session):
        pkts = sorted(by_session[session], key=lambda p: (p.seq, p.send_time_s, p.packet_id))
        cur: list[int] = []
        prev_seq: int | None = None
        for pkt in pkts:
            if pkt.outcome == "unknown":
                if cur:
                    sequences.append(cur)
                    cur = []
                breaks_unknown += 1
                prev_seq = pkt.seq
                continue
            if prev_seq is not None and pkt.seq != prev_seq + 1:
                if cur:
                    sequences.append(cur)
                    cur = []
                breaks_gap += 1
            cur.append(1 if pkt.outcome == "lost" else 0)
            prev_seq = pkt.seq
        if cur:
            sequences.append(cur)
    return sequences, {"unknown_breaks": breaks_unknown, "seq_gap_breaks": breaks_gap}


def analyze_trace(trace: EspNowTrace, confidence: float = 0.95) -> dict:
    """Counts, delivery, RSSI statistics and Gilbert fit for one trace (one fixed position)."""
    counted = [p for p in trace.packets if p.duplicate is not True]
    n_dup = len(trace.packets) - len(counted)
    received = [p for p in counted if p.outcome == "received"]
    n_rx = len(received)
    n_lost = sum(1 for p in counted if p.outcome == "lost")
    n_unknown = sum(1 for p in counted if p.outcome == "unknown")
    denom = n_rx + n_lost
    rssi = np.array([p.rssi_dbm for p in received if p.rssi_dbm is not None], dtype=float)
    rssi_missing = n_rx - int(rssi.size)
    delivery: dict[str, Any] = {
        "received": n_rx, "lost": n_lost, "unknown": n_unknown, "duplicates_excluded": n_dup,
        "denominator": denom, "n_packets_total": len(trace.packets),
        "delivery_ratio": None, "delivery_ci": None, "ci_method": "wilson",
        "denominator_rule": "received + lost; unknown outcomes excluded and reported, never imputed",
    }
    if denom:
        delivery["delivery_ratio"] = _f(n_rx / denom)
        delivery["delivery_ci"] = _interval(wilson_interval(n_rx, denom, confidence))
    rssi_stats: dict[str, Any] = {
        "n_rx_with_rssi": int(rssi.size), "n_rx_rssi_missing": rssi_missing,
        "mean_rssi_dbm": None, "std_rssi_dbm": None,
    }
    if rssi.size:
        rssi_stats["mean_rssi_dbm"] = _f(rssi.mean())
    if rssi.size >= 2:
        rssi_stats["std_rssi_dbm"] = _f(rssi.std(ddof=1))
    sequences, breaks = split_sequences(trace)
    gilbert = fit_gilbert(sequences, confidence)
    gilbert["sequence_breaks"] = breaks
    return {"delivery": delivery, "rssi": rssi_stats, "gilbert": gilbert, "_sequences": sequences}


# ----------------------------------------------------------------------------
# whole calibration
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionInput:
    position_id: str
    trace: EspNowTrace
    distance_m: float
    walls: int = 0
    bodies: int = 0


def _result_status(statuses: Iterable[str]) -> str:
    s = set(statuses)
    if "synthetic" in s:
        return "synthetic"
    if "fixture" in s:
        return "fixture"
    return "measured" if s == {"measured"} else "no traces"


def calibrate(positions: Sequence[PositionInput], confidence: float = 0.95, notes: str | None = None) -> dict:
    """Fit path loss, delivery and Gilbert parameters from validated traces.

    Pure computation; no file access.  See the module docstring for the rules.
    """
    ids = [p.position_id for p in positions]
    if len(ids) != len(set(ids)):
        raise ValueError("position_id values must be unique")
    entries_out = []
    fit_d: list[float] = []
    fit_y: list[float] = []
    fit_ids: list[str] = []
    excluded: list[dict] = []
    within_ss = 0.0
    within_df = 0
    all_sequences: list[list[int]] = []
    totals = {"received": 0, "lost": 0, "unknown": 0}
    warnings: list[str] = []
    for pos in positions:
        if not (math.isfinite(pos.distance_m) and pos.distance_m > 0):
            raise ValueError(f"distance_m must be positive and finite for {pos.position_id}")
        if pos.walls < 0 or pos.bodies < 0:
            raise ValueError(f"walls and bodies must be nonnegative for {pos.position_id}")
        a = analyze_trace(pos.trace, confidence)
        sequences = a.pop("_sequences")
        all_sequences.extend(sequences)
        for key in totals:
            totals[key] += a["delivery"][key]
        mean = a["rssi"]["mean_rssi_dbm"]
        reason = None
        if pos.walls or pos.bodies:
            reason = "not line of sight (walls or bodies > 0): wall and body losses are not separated from the path-loss exponent"
        elif mean is None:
            reason = "no received packets with an RSSI value"
        if reason is None:
            fit_d.append(float(pos.distance_m))
            fit_y.append(float(mean))
            fit_ids.append(pos.position_id)
            std = a["rssi"]["std_rssi_dbm"]
            m = a["rssi"]["n_rx_with_rssi"]
            if std is not None:
                within_ss += (m - 1) * std * std
                within_df += m - 1
            ratio = a["delivery"]["delivery_ratio"]
            if ratio is not None and ratio < CENSORING_WARN_DELIVERY:
                warnings.append(f"{pos.position_id}: delivery {ratio:.3f} is low; the mean RSSI of received packets may be biased "
                                "upward by censoring near the receiver sensitivity")
        else:
            excluded.append({"position_id": pos.position_id, "reason": reason})
        entries_out.append({
            "position_id": pos.position_id, "trace_id": pos.trace.trace_id, "trace_status": pos.trace.status,
            "distance_m": _f(pos.distance_m), "walls": pos.walls, "bodies": pos.bodies,
            "used_in_path_loss_fit": reason is None, **a,
        })
    path = fit_path_loss(fit_d, fit_y, confidence)
    path["positions_used"] = fit_ids
    path["positions_excluded"] = excluded
    path["unit_of_analysis"] = "one mean RSSI per position (shadowing is a fixed draw per position)"
    path["within_position_rssi_std_db"] = _f(math.sqrt(within_ss / within_df)) if within_df > 0 else None
    path["within_position_rssi_std_df"] = within_df
    path["within_position_note"] = ("pooled sample std of RSSI inside a position: fast variation plus measurement noise "
                                    "(and RSSI quantization); consecutive packets are not independent")
    pooled = fit_gilbert(all_sequences, confidence)
    result = {
        "schema": RESULT_SCHEMA,
        "status": _result_status(p.trace.status for p in positions),
        "status_rule": "synthetic if any trace is synthetic, fixture if any is a fixture, measured only if every trace is measured",
        "n_positions": len(positions),
        "confidence": confidence,
        "path_loss": path,
        "delivery_total": {**totals, "denominator": totals["received"] + totals["lost"],
                           "note": "unknown outcomes are reported, not in the denominator"},
        "gilbert_pooled": pooled,
        "entries": entries_out,
        "warnings": warnings,
        "notes": notes,
    }
    return result


# ----------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------

_MANIFEST_KEYS = {"entries", "notes"}
_ENTRY_KEYS = {"trace", "distance_m", "walls", "bodies", "position_id"}


def _nonneg_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def load_manifest(path: Path) -> tuple[dict, list[dict], bytes]:
    raw_bytes = Path(path).read_bytes()
    data = json.loads(raw_bytes)
    if not isinstance(data, dict):
        raise ValueError("manifest must be an object")
    extra = sorted(set(data) - _MANIFEST_KEYS)
    if extra:
        raise ValueError(f"manifest contains unknown field(s): {', '.join(extra)}")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest.entries must be a non-empty array")
    clean = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ValueError(f"entries[{i}] must be an object")
        extra = sorted(set(e) - _ENTRY_KEYS)
        if extra:
            raise ValueError(f"entries[{i}] contains unknown field(s): {', '.join(extra)}")
        if not isinstance(e.get("trace"), str) or not e["trace"].strip():
            raise ValueError(f"entries[{i}].trace must be a non-empty string")
        d = e.get("distance_m")
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or d <= 0:
            raise ValueError(f"entries[{i}].distance_m must be a positive finite number")
        pid = e.get("position_id")
        if pid is None:
            pid = Path(e["trace"]).stem
        if not isinstance(pid, str) or not pid.strip():
            raise ValueError(f"entries[{i}].position_id must be a non-empty string")
        clean.append({
            "trace": e["trace"], "distance_m": float(d), "position_id": pid,
            "walls": _nonneg_int(e.get("walls", 0), f"entries[{i}].walls"),
            "bodies": _nonneg_int(e.get("bodies", 0), f"entries[{i}].bodies"),
        })
    notes = data.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("manifest.notes must be a string")
    return data, clean, raw_bytes


def _trace_path(manifest_path: Path, trace: str) -> Path:
    p = Path(trace)
    return p if p.is_absolute() else Path(manifest_path).resolve().parent / p


def calibrate_manifest(manifest_path: str | Path, confidence: float = 0.95) -> dict:
    """Load a manifest, validate every trace with EspNowTrace, calibrate, and attach provenance."""
    manifest_path = Path(manifest_path)
    data, entries, manifest_bytes = load_manifest(manifest_path)
    positions = []
    trace_hashes = []
    for e in entries:
        tpath = _trace_path(manifest_path, e["trace"])
        raw = tpath.read_bytes()
        trace = EspNowTrace.from_dict(json.loads(raw))
        positions.append(PositionInput(e["position_id"], trace, e["distance_m"], e["walls"], e["bodies"]))
        trace_hashes.append({"position_id": e["position_id"], "trace": e["trace"], "sha256": _sha256_bytes(raw),
                             "trace_id": trace.trace_id, "status": trace.status})
    result = calibrate(positions, confidence, data.get("notes"))
    result["provenance"] = {
        "tool": TOOL_NAME, "status": PROVENANCE_STATUS,
        "manifest_sha256": _sha256_bytes(manifest_bytes), "trace_sha256": trace_hashes,
    }
    return result


# ----------------------------------------------------------------------------
# sizing
# ----------------------------------------------------------------------------

def _wilson_halfwidth(p: float, n: int, confidence: float) -> float:
    lo, hi = wilson_interval(p * n, n, confidence)
    return (hi - lo) / 2.0


def packets_for_wilson_halfwidth(p: float, halfwidth: float, confidence: float = 0.95) -> int:
    """Smallest n whose Wilson interval at observed proportion p has half-width <= halfwidth.

    Uses successes = p * n (not rounded), so the half-width falls smoothly
    with n and the answer is a clean search result.  Planning figure for how
    many packets a position needs; it ignores packet-to-packet dependence.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    if not halfwidth > 0:
        raise ValueError("halfwidth must be positive")
    lo, hi = 1, 2
    while _wilson_halfwidth(p, hi, confidence) > halfwidth:
        lo, hi = hi, hi * 2
        if hi > 10 ** 9:
            raise ValueError("halfwidth too small")
    while lo < hi:
        mid = (lo + hi) // 2
        if _wilson_halfwidth(p, mid, confidence) <= halfwidth:
            hi = mid
        else:
            lo = mid + 1
    return lo


def sizing_table(confidence: float = 0.95) -> list[dict]:
    return [{"p": p, "halfwidth": h, "packets": packets_for_wilson_halfwidth(p, h, confidence), "confidence": confidence}
            for p in SIZING_PROBABILITIES for h in SIZING_HALFWIDTHS]


def format_sizing_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["Packets needed for a Wilson interval of the given half-width (planning figure, packets treated as independent)",
             "  delivery p   half-width   packets"]
    for r in rows:
        lines.append(f"  {r['p']:<12g} {r['halfwidth']:<12g} {r['packets']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# synthetic generator (tests and EXP-018)
# ----------------------------------------------------------------------------

def build_trace_dict(trace_id: str, outcomes: Sequence[str], rssi: Sequence[float | None] | None = None, *,
                     session_id: str = "s0", status: str = "synthetic", send_interval_s: float = 0.01,
                     seq_start: int = 0) -> dict:
    """A zephyr-espnow-trace-1 document from outcome strings ('received', 'lost', 'unknown')."""
    packets = []
    for i, outcome in enumerate(outcomes):
        t = round(i * send_interval_s, 9)
        pkt: dict[str, Any] = {
            "packet_id": f"{trace_id}-{i}", "sender_id": "tx", "receiver_id": "rx", "sender_session_id": session_id,
            "seq": seq_start + i, "send_time_s": t, "outcome": outcome,
        }
        if outcome == "received":
            pkt["receive_time_s"] = round(t + 0.002, 9)
            pkt["rssi_dbm"] = None if rssi is None or rssi[i] is None else float(rssi[i])
        elif outcome == "lost":
            pkt["loss_reason"] = "synthetic_burst"
        else:
            pkt["unknown_reason"] = "synthetic_unknown"
        packets.append(pkt)
    return {
        "schema": TRACE_SCHEMA, "trace_id": trace_id, "status": status, "clock_domain": "shared_monotonic",
        "source": "calibrate_radio_from_trace synthetic generator",
        "notes": "synthetic data; not a radio measurement",
        "capture": {"transport": "ESP-NOW", "device_ids": ["tx", "rx"], "firmware": "simulated", "channel": None,
                    "phy": "simulated"},
        "packets": packets,
    }


def synthetic_trace_dicts(*, n: float, a_dbm: float, sigma_db: float, distances_m: Sequence[float],
                          packets_per_distance: int, ge: GilbertElliott, fast_sigma_db: float, seed: int
                          ) -> list[tuple[str, float, dict]]:
    """Synthetic trace documents, one per distance: (position_id, distance_m, trace dict).

    Per position a static shadow X ~ N(0, sigma_db) is drawn once; every packet
    carries RSSI = A - n x - X + N(0, fast_sigma_db) with x = 10 log10(d / 1 m).
    Loss comes from the given Gilbert-Elliott chain; lost packets carry no RSSI.
    Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    out = []
    count = int(packets_per_distance)
    for i, d in enumerate(distances_m):
        pid = f"d{i}_{d:g}m"
        shadow = float(rng.normal(0.0, sigma_db)) if sigma_db > 0 else 0.0
        mean = a_dbm - n * 10.0 * math.log10(d) - shadow
        rssi = mean + (rng.normal(0.0, fast_sigma_db, size=count) if fast_sigma_db > 0 else np.zeros(count))
        _, lost = ge.sample(count, rng)
        outcomes = ["lost" if v else "received" for v in lost.tolist()]
        doc = build_trace_dict(f"synthetic-{seed}-{pid}", outcomes, rssi.tolist(), session_id=f"session-{pid}")
        out.append((pid, float(d), doc))
    return out


def synthetic_manifest(tmpdir: str | Path, *, n: float, a_dbm: float, sigma_db: float, distances_m: Sequence[float],
                       packets_per_distance: int, ge: GilbertElliott, fast_sigma_db: float, seed: int) -> Path:
    """Write synthetic traces plus a calibration manifest into tmpdir; return the manifest path."""
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    entries = []
    for pid, d, doc in synthetic_trace_dicts(
            n=n, a_dbm=a_dbm, sigma_db=sigma_db, distances_m=distances_m, packets_per_distance=packets_per_distance,
            ge=ge, fast_sigma_db=fast_sigma_db, seed=seed):
        fname = f"{pid}.json"
        (tmpdir / fname).write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n", encoding="utf-8")
        entries.append({"trace": fname, "distance_m": d, "walls": 0, "bodies": 0, "position_id": pid})
    manifest = {"entries": entries,
                "notes": f"synthetic: n={n}, A={a_dbm} dBm, sigma={sigma_db} dB, seed={seed}; not a radio measurement"}
    path = tmpdir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifest", type=Path, help="calibration manifest JSON")
    parser.add_argument("--output", type=Path, help="write the result JSON here (default: stdout)")
    parser.add_argument("--sizing", action="store_true",
                        help="include the Wilson packet-count table in the result and print it to stderr")
    parser.add_argument("--confidence", type=float, default=0.95, help="interval confidence level (default 0.95)")
    args = parser.parse_args(argv)
    try:
        if not 0.0 < args.confidence < 1.0:
            raise ValueError("--confidence must be in (0, 1)")
        if args.output:
            protected = {args.manifest.resolve()}
            _, entries, _ = load_manifest(args.manifest)
            protected.update(_trace_path(args.manifest, e["trace"]).resolve() for e in entries)
            if args.output.resolve() in protected:
                raise ValueError("output must differ from the manifest and the trace files")
        result = calibrate_manifest(args.manifest, args.confidence)
        if args.sizing:
            rows = sizing_table(args.confidence)
            result["sizing_table"] = rows
            print(format_sizing_table(rows), file=sys.stderr)
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        print(f"{TOOL_NAME}: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
