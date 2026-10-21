"""Fit simulator link settings from one ESP-NOW echo trace, with uncertainty.

Input: one ``zephyr-espnow-trace-1`` trace with ``clock_domain =
"sender_round_trip"`` (made by tools/import_espnow_log.py from the two CSV
logs described in docs/ESPNOW_ECHO_LOG.md). Output: a JSON-safe dict with
schema ``zephyr-espnow-fit-1`` holding

* the measured round trip (what the echo really measures),
* a one-way delay and jitter ESTIMATE (rtt / 2 and std(rtt) / sqrt(2)), which
  rests on two stated assumptions about the two legs,
* outbound loss: the lost fraction, the burst pattern, and the matching
  ``sim.link.LinkConfig`` burst settings, all with intervals,
* return-leg loss (echoes that never came back),
* ready-to-use ``link_config`` (keyword arguments for ``LinkConfig``) and
  ``s6_deliver`` (keyword arguments for ``sim.s6_recovery.deliver``) blocks.

The numbers are only as real as the trace. A synthetic trace gives synthetic
settings.

How the burst settings are found
--------------------------------
The simulator's burst rule (``SimulatedLink._loss_reason``): when no burst is
running, a packet starts one with probability ``start``; every packet in a
burst is lost; after each lost packet the burst ends with probability
``end``, and the NEXT packet then rolls the start chance again. Seen from
outside (just lost / received), that is a two-state chain with

* P(next lost | this received) = start
* P(next received | this lost) = end * (1 - start)
  (the burst has to end AND the next packet must not start a new one)

so from the observed transition fractions p = P(received -> lost) and
r = P(lost -> received):

* start = p
* end = r / (1 - p)
* long-run lost fraction = p / (p + r) = start / (start + end * (1 - start))
* mean run of lost packets in a row = 1 / r = 1 / (end * (1 - start))
* mean length of ONE burst = 1 / end (a run can hold back-to-back bursts)

``fit_gilbert`` (tools/calibrate_radio_from_trace.py) reports the classic
chain's p and r; this module maps them. ``end`` can come out above 1 when
losses are less clumped than independent ones; it is then clipped to 1 and
flagged (end = 1 is exactly independent loss with probability start).

Uncertainty
-----------
* Delay: a moving block bootstrap over packets in send order (see
  ``block_length_rule``), percentile intervals. Packets close in time can
  share interference, so resampling single packets would understate the
  spread; resampling short runs of neighbours keeps that sharing.
* Outbound lost fraction: a Wilson interval with the packet count shrunk to
  an effective count n (1 - lam) / (1 + lam), where lam = 1 - p - r is how
  strongly one packet's fate predicts the next (zero for independent loss).
  For a two-state chain, the lost fraction of n packets has variance
  q (1 - q) / n * (1 + lam) / (1 - lam), so bursty loss is worth fewer
  independent packets. The plain Wilson interval is reported too.
* start, end and everything made from them: given how many times the link
  was in each state, the two transition counts are independent binomials.
  Each transition probability gets a Jeffreys Beta(count + 1/2,
  misses + 1/2) distribution; random draws of the pair are pushed through the
  formulas above and the percentile interval is reported. This handles zero
  counts without breaking.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from sim.link import LinkConfig
from sim.s6_recovery import wilson_interval
from sim.trace import EspNowTrace
from tools.calibrate_radio_from_trace import fit_gilbert, split_sequences

FIT_SCHEMA = "zephyr-espnow-fit-1"
MODEL_CHECK_DRAWS = 200_000
CHAIN_DRAWS = 20_000

DELAY_ASSUMPTIONS = [
    ("Symmetric on average: the sender-to-echoer leg and the echoer-to-sender leg take the same mean time, "
    "so one-way delay_s = mean(rtt) / 2."),
    ("Independent legs with equal spread: the two legs' delays vary independently of each other and by the "
    "same amount, so one-way delay_jitter_s = std(rtt) / sqrt(2)."),
]
DELAY_CONSEQUENCES = [
    ("An echo cannot see asymmetry. If the legs really take d_out and d_back, delay_s is (d_out + d_back) / 2, "
    "off from d_out by (d_back - d_out) / 2, and nothing in the log can detect it. Only a shared wired clock "
    "signal between the boards can check it."),
    ("If the two legs' delays are positively correlated (for example both slowed by the same interference), "
    "std(rtt) is larger than for independent legs, so delay_jitter_s comes out TOO HIGH; negatively "
    "correlated legs make it too low. If the legs are independent but spread by different amounts, "
    "delay_jitter_s is the root mean square of the two spreads."),
    ("Stamps are whole microseconds, which adds about 0.6 us of noise (standard deviation) to each round trip; "
    "this is negligible next to millisecond delays."),
]
WITHOUT_TURNAROUND_NOTE = ("These packets had an echo but no echoer hold time, so their round trip still "
                           "includes how long the echoer held the packet (too long by that hold).")
RETURN_LEG_LABEL = ("Return-leg loss: among packets the echoer logged, the fraction with no echo back at the "
                    "sender. This includes cases where the echoer chose not to echo, not only radio loss.")


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def _f(x: Any) -> float | None:
    """Plain finite float, or None when undefined."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _ci(lo: Any, hi: Any) -> list[float | None]:
    return [_f(lo), _f(hi)]


def _z_two_sided(confidence: float) -> float:
    return float(stats.norm.ppf(0.5 + confidence / 2.0))


# ----------------------------------------------------------------------------
# burst chain mapping (pure formulas, tested against SimulatedLink)
# ----------------------------------------------------------------------------

def burst_chain_from_gilbert(p_received_to_lost: float, p_lost_to_received: float) -> tuple[float, float | None, bool]:
    """Map classic-chain transition fractions to LinkConfig (start, end).

    start = p and end = r / (1 - p). Returns (start, end, clipped) where
    clipped says end came out above 1 and was set to 1. end is None when
    p = 1 (every received packet was followed by a loss, so the formula
    divides by zero).
    """
    p, r = float(p_received_to_lost), float(p_lost_to_received)
    if p >= 1.0:
        return p, None, False
    end = r / (1.0 - p)
    if end > 1.0:
        return p, 1.0, True
    return p, end, False


def gilbert_from_burst_chain(start: float, end: float) -> tuple[float, float]:
    """The other direction: (P(received -> lost), P(lost -> received)) = (start, end (1 - start))."""
    return float(start), float(end) * (1.0 - float(start))


def stationary_loss(start: float, end: float) -> float | None:
    """Long-run lost fraction of the burst rule: start / (start + end (1 - start))."""
    denom = start + end * (1.0 - start)
    return None if denom <= 0 else start / denom


def mean_loss_run(start: float, end: float) -> float | None:
    """Mean number of lost packets in a row: 1 / (end (1 - start))."""
    r = end * (1.0 - start)
    return None if r <= 0 else 1.0 / r


# ----------------------------------------------------------------------------
# block bootstrap for the delay
# ----------------------------------------------------------------------------

def block_length_rule(x: Sequence[float]) -> tuple[int, dict]:
    """Pick the moving-block length (in packets) for the delay bootstrap.

    Rule: b = max(ceil(n ** (1/3)), k), capped at n // 2, where k is the first
    lag at which the sample autocorrelation of the round trips is inside the
    +-2 / sqrt(n) band (that is, neighbours k packets apart no longer look
    related). n ** (1/3) is the textbook growth rate for block length when the
    target is a variance or a mean (Hall, Horowitz and Jing 1995); the
    autocorrelation guard makes sure one block is at least as long as the
    stretch over which packets visibly share their delay. It is a rule of
    thumb, not an optimised selector; pass ``block_length`` to override.
    """
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    b0 = max(1, math.ceil(n ** (1.0 / 3.0)))
    info: dict[str, Any] = {"n": n, "cube_root_rule": b0, "acf_cutoff_lag": None}
    if n < 4:
        return 1, info
    centred = arr - arr.mean()
    var = float(np.dot(centred, centred)) / n
    k_cut = 1
    if var > 0:
        max_lag = max(1, min(n // 4, 500))
        size = 1 << math.ceil(math.log2(2 * n))
        spec = np.fft.rfft(centred, size)
        acov = np.fft.irfft(spec * np.conj(spec), size)[: max_lag + 1] / n
        acf = acov / acov[0]
        band = 2.0 / math.sqrt(n)
        inside = np.nonzero(np.abs(acf[1:]) < band)[0]
        k_cut = int(inside[0]) + 1 if inside.size else max_lag
        info["acf_cutoff_lag"] = k_cut
    b = min(max(b0, k_cut), max(1, n // 2))
    return int(b), info


def moving_block_bootstrap(x: np.ndarray, n_boot: int, block: int, rng: np.random.Generator,
                           quantiles: Sequence[float]) -> dict[str, np.ndarray]:
    """Bootstrap replicates of mean, std and the given quantiles.

    Each replicate glues together randomly chosen runs of ``block``
    consecutive packets until it has n values. Done in chunks so memory stays
    near a few tens of megabytes.
    """
    n = int(x.size)
    k = math.ceil(n / block)
    chunk = max(1, min(n_boot, 4_000_000 // max(n, 1)))
    means, stds, qs = [], [], []
    offsets = np.arange(block)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        starts = rng.integers(0, n - block + 1, size=(m, k))
        idx = (starts[:, :, None] + offsets).reshape(m, k * block)[:, :n]
        sample = x[idx]
        means.append(sample.mean(axis=1))
        stds.append(sample.std(axis=1, ddof=1) if n > 1 else np.zeros(m))
        qs.append(np.quantile(sample, quantiles, axis=1).T)
        done += m
    return {"mean": np.concatenate(means), "std": np.concatenate(stds), "quantiles": np.concatenate(qs)}


def _lognormal_rtt_quantiles(delay: float, jitter: float, q: Sequence[float], rng: np.random.Generator) -> list[float]:
    """Quantiles of the sum of two independent one-way delays as the simulator draws them."""
    if jitter <= 0 or delay <= 0:
        return [2.0 * delay for _ in q]
    s2 = math.log1p((jitter / delay) ** 2)
    mu = math.log(delay) - s2 / 2.0
    draws = rng.lognormal(mu, math.sqrt(s2), size=(2, MODEL_CHECK_DRAWS)).sum(axis=0)
    return [float(v) for v in np.quantile(draws, q)]


def _ordered_packets(trace: EspNowTrace) -> list:
    return sorted((p for p in trace.packets if p.duplicate is not True),
                  key=lambda p: (p.sender_session_id, p.seq, p.send_time_s, p.packet_id))


def fit_delay(trace: EspNowTrace, *, confidence: float, n_boot: int, seed: int,
              block_length: int | None) -> tuple[dict | None, str | None]:
    """Round-trip statistics and the one-way estimate, or (None, reason)."""
    echoed = [p for p in _ordered_packets(trace) if p.echo_time_s is not None]
    if not echoed:
        return None, "no packet in the trace has an echo, so there is no round trip to measure"
    rtt = np.array([p.round_trip_s() for p in echoed], dtype=float)
    n = int(rtt.size)
    n_no_turn = sum(1 for p in echoed if p.echo_turnaround_s is None)
    alpha = 1.0 - confidence
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    qs = (0.5, 0.95, 0.99)
    rng = np.random.default_rng(seed)
    if block_length is None:
        block, block_info = block_length_rule(rtt)
        rule = ("b = max(ceil(n^(1/3)), first lag where |autocorrelation| < 2/sqrt(n)), capped at n // 2. "
                "n^(1/3) is the standard growth rate for a block bootstrap of a mean or variance; the "
                "autocorrelation guard keeps a block at least as long as the visibly correlated stretch.")
    else:
        if int(block_length) < 1:
            raise ValueError("block_length must be at least 1")
        block, block_info = min(int(block_length), n), {"n": n}
        rule = "block length given by the caller"
    point_q = np.quantile(rtt, qs)
    point = {
        "mean": float(rtt.mean()), "std": float(rtt.std(ddof=1)) if n > 1 else 0.0,
        "p50": float(point_q[0]), "p95": float(point_q[1]), "p99": float(point_q[2]),
    }
    if n >= 2:
        boot = moving_block_bootstrap(rtt, int(n_boot), block, rng, qs)
        def band(values: np.ndarray) -> list[float | None]:
            lo, hi = np.quantile(values, [lo_q, hi_q])
            return _ci(lo, hi)
        cis = {"mean": band(boot["mean"]), "std": band(boot["std"]),
               "p50": band(boot["quantiles"][:, 0]), "p95": band(boot["quantiles"][:, 1]),
               "p99": band(boot["quantiles"][:, 2])}
    else:
        cis = {key: None for key in point}
    rtt_stats: dict[str, Any] = {key: {"value": _f(point[key]), "ci": cis[key]} for key in point}
    extreme_reason = "a bootstrap interval is not valid for the smallest or largest value in a sample"
    rtt_stats["min"] = {"value": _f(rtt.min()), "ci": None, "ci_null_reason": extreme_reason}
    rtt_stats["max"] = {"value": _f(rtt.max()), "ci": None, "ci_null_reason": extreme_reason}

    root2 = math.sqrt(2.0)
    delay_s = point["mean"] / 2.0
    jitter_s = point["std"] / root2
    one_way = {
        "delay_s": {"value": _f(delay_s),
                    "ci": None if cis["mean"] is None else _ci(cis["mean"][0] / 2.0, cis["mean"][1] / 2.0)},
        "delay_jitter_s": {"value": _f(jitter_s),
                           "ci": None if cis["std"] is None else _ci(cis["std"][0] / root2, cis["std"][1] / root2)},
        "formula": "delay_s = mean(rtt) / 2; delay_jitter_s = std(rtt) / sqrt(2)",
    }
    model_q = _lognormal_rtt_quantiles(delay_s, jitter_s, qs, np.random.default_rng([seed, 1]))
    check_rows = {}
    for name, model_value in zip(("p50", "p95", "p99"), model_q):
        ci = cis[name]
        inside = None if ci is None else bool(ci[0] <= model_value <= ci[1])
        check_rows[name] = {"observed": _f(point[name]), "observed_ci": ci, "model": _f(model_value),
                            "model_inside_observed_band": inside}
    flags = [row["model_inside_observed_band"] for row in check_rows.values()]
    model_check = {
        "model": ("sim draws each one-way delay from a lognormal with mean delay_s and std delay_jitter_s; "
                  "the model round trip is the sum of two independent such draws "
                  f"({MODEL_CHECK_DRAWS} Monte Carlo draws)"),
        "quantiles": check_rows,
        "all_inside": None if any(f is None for f in flags) else all(flags),
        "note": ("The model is tuned to match the mean and spread, so this only checks the SHAPE (median and "
                 "tail). The band covers sampling noise in the observed quantile only, not uncertainty in the "
                 "fitted delay_s and delay_jitter_s, so a quantile just outside the band is a hint, not proof "
                 "that the lognormal shape is wrong. Inside the band does not prove the shape is right."),
    }
    section = {
        "n": n,
        "n_without_turnaround": int(n_no_turn),
        "without_turnaround_note": WITHOUT_TURNAROUND_NOTE,
        "rtt_s": rtt_stats,
        "one_way_estimate": one_way,
        "assumptions": list(DELAY_ASSUMPTIONS),
        "consequences": list(DELAY_CONSEQUENCES),
        "bootstrap": {
            "method": "moving block bootstrap over echoed packets in send order, percentile intervals",
            "block_length_packets": int(block), "block_rule": rule, "block_rule_inputs": block_info,
            "n_boot": int(n_boot), "seed": int(seed), "confidence": confidence,
            "note": ("Packets are put in send order; packets with no echo are skipped, so a block can span a "
                     "gap. Several sender sessions are joined end to end."),
        },
        "model_check": model_check,
    }
    return section, None


# ----------------------------------------------------------------------------
# loss
# ----------------------------------------------------------------------------

def _jeffreys_draws(successes: int, trials: int, size: int, rng: np.random.Generator) -> np.ndarray | None:
    if trials <= 0:
        return None
    return rng.beta(successes + 0.5, trials - successes + 0.5, size=size)


def _echoer_log_seen(packets: Sequence) -> bool:
    """True when the trace shows signs that the echoer's own log was used.

    Without it, a missing echo cannot be blamed on either leg, so both loss
    sections must refuse to give a rate.
    """
    return any(p.echo_turnaround_s is not None or p.outcome == "lost"
               or (p.outcome == "received" and p.echo_time_s is None) for p in packets)


def fit_loss(trace: EspNowTrace, *, confidence: float, seed: int) -> dict:
    """Outbound-leg loss: lost fraction, burst pattern, LinkConfig settings."""
    alpha = 1.0 - confidence
    non_dup = [p for p in trace.packets if p.duplicate is not True]
    n_dup = len(trace.packets) - len(non_dup)
    n_received = sum(1 for p in non_dup if p.outcome == "received")
    n_lost = sum(1 for p in non_dup if p.outcome == "lost")
    n_unknown = sum(1 for p in non_dup if p.outcome == "unknown")
    n = n_received + n_lost
    sequences, breaks = split_sequences(trace)
    gilbert = fit_gilbert(sequences, confidence)
    t = gilbert["transitions"]
    from_rx, from_lost = gilbert["denominator_from_received"], gilbert["denominator_from_lost"]
    out: dict[str, Any] = {
        "leg": "outbound (sender to echoer), from outcome received / lost",
        "n_records": len(trace.packets), "n_duplicates_excluded": n_dup, "n_unknown_excluded": n_unknown,
        "n_received": n_received, "n_lost": n_lost, "n": n,
        "loss_rate": None, "loss_rate_ci": None, "loss_rate_ci_method": None,
        "loss_rate_ci_wilson_independent": None, "zero_loss_upper_bound": None,
        "sequence_breaks": breaks, "gilbert_classic": gilbert,
    }
    if not _echoer_log_seen(non_dup):
        out["reason"] = ("no sign of an echoer log (no hold times, no lost packets, no received packet without an "
                         "echo). Then only packets that survived BOTH legs are known and every other packet is "
                         "unknown, so the known ones are a biased sample and no outbound loss rate is given.")
        return out
    if n == 0:
        out["reason"] = "no packet with a known outbound outcome"
        return out
    q = n_lost / n
    out["loss_rate"] = _f(q)
    out["loss_rate_ci_wilson_independent"] = _ci(*wilson_interval(n_lost, n, confidence))
    if n_lost == 0:
        bound = 1.0 - alpha ** (1.0 / n)
        out["zero_loss_upper_bound"] = {
            "value": _f(bound), "confidence": confidence,
            "rule": ("exact one-sided bound 1 - (1 - confidence)^(1/n): the largest loss rate that would still "
                     "give zero losses in n packets with chance at least 1 - confidence. At 95% this is about "
                     "3 / n (the rule of three)."),
        }

    p_hat = gilbert["p_good_to_bad"]
    r_hat = gilbert["p_bad_to_good"]
    lam = None
    if p_hat is not None and r_hat is not None:
        lam = 1.0 - p_hat - r_hat
    lam_used = 0.0 if lam is None else min(max(lam, 0.0), 0.999)
    n_eff = n * (1.0 - lam_used) / (1.0 + lam_used)
    out["lag1_correlation"] = _f(lam)
    out["effective_n"] = _f(n_eff)
    out["loss_rate_ci"] = _ci(*wilson_interval(q * n_eff, n_eff, confidence))
    out["loss_rate_ci_method"] = (
        "Wilson interval with the effective packet count n (1 - lam) / (1 + lam), lam = 1 - p - r clipped to "
        "[0, 0.999]. For a two-state chain the lost fraction has variance q (1 - q) / n * (1 + lam) / (1 - lam), "
        "so clumped losses carry less information than the raw count. The plain Wilson interval (which assumes "
        "independent packets) is in loss_rate_ci_wilson_independent.")

    rng = np.random.default_rng([seed, 2])
    p_draws = _jeffreys_draws(t["received_to_lost"], from_rx, CHAIN_DRAWS, rng)
    r_draws = _jeffreys_draws(t["lost_to_received"], from_lost, CHAIN_DRAWS, rng)
    chain: dict[str, Any] = {
        "burst_start_probability": None, "burst_end_probability": None, "end_clipped_to_one": False,
        "mean_burst_length_packets": None, "mean_loss_run_packets": None, "stationary_loss_rate": None,
        "formulas": ("start = p, end = r / (1 - p), stationary = p / (p + r), mean loss run = 1 / r, "
                     "mean single-burst length = 1 / end; p = P(received -> lost), r = P(lost -> received)"),
        "ci_method": ("Jeffreys Beta(count + 1/2, misses + 1/2) draws for p and r (independent given how often "
                      f"each state was visited), {CHAIN_DRAWS} draws pushed through the formulas, percentile "
                      "interval. end draws above 1 are clipped to 1."),
    }

    def pct(values: np.ndarray) -> list[float | None]:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return [None, None]
        return _ci(*np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0]))

    if p_draws is not None:
        chain["burst_start_probability"] = {"value": _f(p_hat), "ci": pct(p_draws)}
    if p_hat is not None and r_hat is not None and r_draws is not None:
        start, end, clipped = burst_chain_from_gilbert(p_hat, r_hat)
        if end is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                e_draws = np.minimum(r_draws / (1.0 - p_draws), 1.0)
                run_draws = 1.0 / r_draws
                burst_draws = 1.0 / e_draws
                stat_draws = p_draws / (p_draws + r_draws)
            chain["burst_end_probability"] = {"value": _f(end), "ci": pct(e_draws)}
            chain["end_clipped_to_one"] = bool(clipped)
            chain["mean_burst_length_packets"] = {"value": _f(1.0 / end) if end > 0 else None, "ci": pct(burst_draws)}
            chain["mean_loss_run_packets"] = {"value": _f(mean_loss_run(start, end)), "ci": pct(run_draws)}
            chain["stationary_loss_rate"] = {"value": _f(stationary_loss(start, end)), "ci": pct(stat_draws)}
    out["burst_chain"] = chain

    z_cut = float(stats.norm.ppf(confidence))
    approx_z = gilbert["burstiness"]["approx_z"]
    end_value = None if chain["burst_end_probability"] is None else chain["burst_end_probability"]["value"]
    bursty = approx_z is not None and approx_z > z_cut and end_value is not None and end_value < 1.0
    out["independence_check"] = {
        "approx_z": _f(approx_z), "threshold": _f(z_cut), "bursty": bool(bursty),
        "test": ("fit_gilbert's burstiness z: observed mean run of lost packets against the run length "
                 "independent loss would give, one-sided at the stated confidence."),
    }
    out["model_choice"] = {
        "loss_model": "burst" if bursty else "independent",
        "rule": ("burst if the burstiness z is above the one-sided normal cutoff for the confidence level AND the "
                 "fitted end probability is below 1; otherwise independent (losses not shown to clump)."),
    }
    return out


def fit_return_leg(trace: EspNowTrace, *, confidence: float) -> dict:
    """Echoes that did not come back, among packets the echoer logged."""
    non_dup = [p for p in trace.packets if p.duplicate is not True]
    received = [p for p in non_dup if p.outcome == "received"]
    n_echo = sum(1 for p in received if p.echo_time_s is not None)
    n_no_echo = len(received) - n_echo
    echoer_log_seen = _echoer_log_seen(non_dup)
    out: dict[str, Any] = {"label": RETURN_LEG_LABEL, "n_echoer_logged": len(received), "n_echoed": n_echo,
                           "n_no_echo": n_no_echo, "loss_rate": None, "loss_rate_ci": None}
    if not echoer_log_seen:
        out["reason"] = ("no sign of an echoer log in this trace (no hold times, no lost packets, no received "
                         "packet without an echo), so a missing echo cannot be blamed on the return leg")
        return out
    if not received:
        out["reason"] = "no packet was logged as received by the echoer"
        return out
    out["loss_rate"] = _f(n_no_echo / len(received))
    out["loss_rate_ci"] = _ci(*wilson_interval(n_no_echo, len(received), confidence))
    out["ci_method"] = "Wilson interval (assumes return losses are independent of each other)"
    return out


# ----------------------------------------------------------------------------
# top level
# ----------------------------------------------------------------------------

def fit_echo_trace(trace: EspNowTrace, *, confidence: float = 0.95, n_boot: int = 2000, seed: int = 0,
                   block_length: int | None = None) -> dict:
    """Fit link settings from one echo trace. See the module docstring."""
    if not isinstance(trace, EspNowTrace):
        raise TypeError("trace must be a validated EspNowTrace")
    if trace.clock_domain != "sender_round_trip":
        raise ValueError(f"fit_echo_trace needs a sender_round_trip (echo) trace, got clock_domain "
                         f"{trace.clock_domain!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1")
    delay, delay_reason = fit_delay(trace, confidence=confidence, n_boot=int(n_boot), seed=int(seed),
                                    block_length=block_length)
    loss = fit_loss(trace, confidence=confidence, seed=int(seed))
    ret = fit_return_leg(trace, confidence=confidence)

    link_config = s6 = None
    link_reason = None
    if delay is None:
        link_reason = "no delay estimate (no echoes), so no link settings can be given"
    elif loss["loss_rate"] is None:
        link_reason = "no packet with a known outbound outcome, so no loss setting can be given"
    else:
        delay_s = delay["one_way_estimate"]["delay_s"]["value"]
        jitter_s = delay["one_way_estimate"]["delay_jitter_s"]["value"]
        if delay_s == 0.0:
            jitter_s = 0.0
        chain = loss["burst_chain"]
        if loss["model_choice"]["loss_model"] == "burst":
            start = chain["burst_start_probability"]["value"]
            end = chain["burst_end_probability"]["value"]
            stationary = chain["stationary_loss_rate"]["value"]
            link_config = {"delay_s": delay_s, "delay_jitter_s": jitter_s, "loss_model": "burst",
                           "loss_probability": stationary, "burst_start_probability": start,
                           "burst_end_probability": end}
            s6 = {"delay_s": delay_s, "jitter_s": jitter_s, "loss_probability": stationary,
                  "loss_model": "burst", "burst_mean_length_packets": 1.0 / end}
        else:
            rate = loss["loss_rate"]
            link_config = {"delay_s": delay_s, "delay_jitter_s": jitter_s, "loss_model": "independent",
                           "loss_probability": rate, "burst_start_probability": 0.0, "burst_end_probability": 1.0}
            s6 = {"delay_s": delay_s, "jitter_s": jitter_s, "loss_probability": rate,
                  "loss_model": "independent", "burst_mean_length_packets": 1.0}
        LinkConfig(**link_config)  # fail here, not in the caller, if anything is out of range

    result = {
        "schema": FIT_SCHEMA,
        "status_note": (f"Trace status: {trace.status}. These numbers are only as real as the trace: a synthetic "
                        "or fixture trace gives synthetic settings, never a radio measurement."),
        "source_trace": {"trace_id": trace.trace_id, "status": trace.status, "n_packets": len(trace.packets)},
        "confidence": confidence,
        "delay": delay,
        "delay_null_reason": delay_reason,
        "loss": loss,
        "return_leg": ret,
        "link_config": link_config,
        "s6_deliver": s6,
        "link_config_null_reason": link_reason,
        "link_config_notes": (
            "link_config holds keyword arguments for sim.link.LinkConfig; s6_deliver holds the matching keyword "
            "arguments for sim.s6_recovery.deliver. In burst mode LinkConfig ignores loss_probability; it is "
            "filled with the long-run lost fraction implied by start and end, which is also what s6_deliver "
            "needs so that deliver() recomputes the same start probability. burst_mean_length_packets is "
            "1 / end, the mean length of ONE burst (a run of losses can hold back-to-back bursts)."),
    }
    json.dumps(result, allow_nan=False)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit simulator link settings from one ESP-NOW echo trace.")
    parser.add_argument("trace", nargs="?", type=Path, help="sender_round_trip trace JSON")
    parser.add_argument("--sender", type=Path, help="sender CSV log (instead of a trace JSON)")
    parser.add_argument("--echoer", type=Path, help="echoer CSV log (optional with --sender)")
    parser.add_argument("--status", choices=("measured", "synthetic", "fixture"),
                        help="trace status, required with --sender")
    parser.add_argument("--trace-id", default=None, help="trace id when reading CSV logs (default: sender file name)")
    parser.add_argument("--out", type=Path, help="write the fit JSON here (default: stdout)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    try:
        if (args.trace is None) == (args.sender is None):
            raise ValueError("give either a trace JSON path or --sender (with optional --echoer)")
        if args.sender is not None:
            if args.status is None:
                raise ValueError("--status is required with --sender")
            from tools.import_espnow_log import read_echo_logs
            trace = read_echo_logs(args.sender, args.echoer, trace_id=args.trace_id or args.sender.stem,
                                   status=args.status)
        else:
            if args.echoer is not None:
                raise ValueError("--echoer only goes with --sender")
            trace = EspNowTrace.from_path(args.trace)
        result = fit_echo_trace(trace, confidence=args.confidence, n_boot=args.n_boot, seed=args.seed)
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out:
        with args.out.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
