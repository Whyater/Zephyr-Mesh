"""Tests for tools/fit_espnow_log.py (fit link settings from one echo trace).

Every input here is synthetic, made by tools/make_espnow_echo_fixture.py, a
generator written separately from the fit (plain NumPy, no SimulatedLink).
Every tolerance is worked out in a comment from the TRUE settings, never read
from the fit's own intervals. Nothing in this file is a radio measurement.

Standard errors used below (n = number of packets that count):

* delay_s = mean(rtt) / 2 with rtt = out + back (independent legs), so
  SE = sqrt(var_out + var_back) / 2 / sqrt(n_echo).
* delay_jitter_s = std(rtt) / sqrt(2). For a sample standard deviation,
  SE(std) ~ sqrt(mu4 - sigma^4) / (2 sigma sqrt(n)), with mu4 the fourth
  central moment. For a sum of two independent legs,
  mu4 = mu4_out + mu4_back + 6 var_out var_back, and a lognormal leg with
  s2 = ln(1 + cv^2) has mu4 = sigma^4 (e^{4 s2} + 2 e^{3 s2} + 3 e^{2 s2} - 3).
* lost fraction of a two-state chain: var = q (1 - q) / n * (1 + lam) / (1 - lam),
  lam = 1 - p - r (how much one packet's fate predicts the next).
* start = p: var = p (1 - p) / (n (1 - q)) (one try per received packet).
* end = r / (1 - p): delta method, var = var_r / (1 - p)^2 + r^2 var_p / (1 - p)^4,
  var_r = r (1 - r) / (n q).

K_SE = 4: a single check fails by chance about 6 times in 100,000 for a
normal estimate, so a failure means a real error, not bad luck.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim.link import LinkConfig, SimulatedLink
from sim.s6_recovery import RecoveryConfig, _burst_start_probability, deliver
from sim.trace import EspNowTrace
from tools.fit_espnow_log import (
    FIT_SCHEMA,
    burst_chain_from_gilbert,
    fit_echo_trace,
    gilbert_from_burst_chain,
    main,
    mean_loss_run,
    stationary_loss,
)
from tools.make_espnow_echo_fixture import EchoParams, generate_echo_logs, truth_summary

K_SE = 4.0
FIXTURE_DIR = Path(__file__).parent / "data" / "espnow_echo"
ROUND_US = 1e-6  # stamps are whole microseconds


# ----------------------------------------------------------------------------
# helpers: CSV rows -> trace, following the labelling table in docs/ESPNOW_ECHO_LOG.md
# ----------------------------------------------------------------------------

def trace_from_rows(sender_rows, echoer_rows, *, trace_id="synthetic-echo", status="synthetic", with_echoer=True):
    """Build a sender_round_trip trace dict straight from the two logs' rows."""
    first: dict[int, tuple[int, int | None, int]] = {}
    extra: list[tuple[int, int, int]] = []
    if with_echoer:
        for seq, rx, etx, rssi in echoer_rows:
            if seq in first:
                extra.append((seq, rx, rssi))
            else:
                first[seq] = (rx, etx, rssi)
    lo = min(first) if first else None
    hi = max(first) if first else None
    tx_of = {}
    packets = []
    base = {"sender_id": "A", "receiver_id": "B", "sender_session_id": "a-boot"}
    for seq, tx, echo in sender_rows:
        tx_of[seq] = tx
        pkt = dict(base, packet_id=f"a-boot:{seq}", seq=seq, send_time_s=tx / 1e6)
        if echo is not None:
            pkt.update(outcome="received", echo_time_s=echo / 1e6)
            if seq in first and first[seq][1] is not None:
                rx, etx, rssi = first[seq]
                pkt.update(receive_time_s=rx / 1e6, echo_turnaround_s=(etx - rx) / 1e6, rssi_dbm=rssi)
        elif seq in first:
            rx, _, rssi = first[seq]
            pkt.update(outcome="received", receive_time_s=rx / 1e6, rssi_dbm=rssi)
        elif with_echoer and lo is not None and lo <= seq <= hi:
            pkt.update(outcome="lost", loss_reason="not_in_echoer_log")
        else:
            pkt.update(outcome="unknown", unknown_reason="outside_echoer_log" if with_echoer else "no_echoer_log")
        packets.append(pkt)
    for k, (seq, rx, rssi) in enumerate(extra):
        packets.append(dict(base, packet_id=f"a-boot:{seq}:dup{k}", seq=seq, send_time_s=tx_of[seq] / 1e6,
                            outcome="received", receive_time_s=rx / 1e6, rssi_dbm=rssi, duplicate=True))
    return {
        "schema": "zephyr-espnow-trace-1", "trace_id": trace_id, "status": status,
        "clock_domain": "sender_round_trip", "source": "tests/test_fit_espnow_log.py synthetic generator",
        "notes": "synthetic echo test; not a radio measurement",
        "capture": {"transport": "ESP-NOW", "device_ids": ["A", "B"], "firmware": "synthetic", "channel": 1,
                    "phy": "synthetic"},
        "packets": packets,
    }


def make_trace(params: EchoParams, seed: int, **kw):
    logs = generate_echo_logs(params, seed)
    return EspNowTrace.from_dict(trace_from_rows(logs.sender_rows, logs.echoer_rows, **kw)), logs


def parse_csv(path: Path, n_cols: int):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("#", "seq,")):
            continue
        parts = line.split(",")
        assert len(parts) == n_cols
        rows.append(tuple(None if v == "" else int(v) for v in parts))
    return rows


def lognormal_mu4(mean: float, std: float) -> float:
    """Fourth central moment of a lognormal with this mean and std (0 for a constant)."""
    if std == 0:
        return 0.0
    s2 = math.log1p((std / mean) ** 2)
    return std ** 4 * (math.exp(4 * s2) + 2 * math.exp(3 * s2) + 3 * math.exp(2 * s2) - 3)


def expected_ses(p: EchoParams, n_known: int, n_echo: int) -> dict[str, float]:
    """Standard errors from the TRUE settings (formulas in the module docstring)."""
    var_o, var_b = p.out_delay_std_s ** 2, p.back_delay_std_s ** 2
    var_rtt = var_o + var_b + ROUND_US ** 2 / 3.0  # plus stamp rounding (generator docstring)
    mu4 = lognormal_mu4(p.out_delay_mean_s, p.out_delay_std_s) + lognormal_mu4(p.back_delay_mean_s, p.back_delay_std_s) \
        + 6 * var_o * var_b
    se_std_rtt = math.sqrt(max(mu4 - var_rtt ** 2, 0.0)) / (2 * math.sqrt(var_rtt) * math.sqrt(n_echo))
    s, e = p.burst_start, p.burst_end
    pr, rr = s, e * (1 - s)
    q = s / (s + rr)
    lam = 1 - pr - rr
    var_p = pr * (1 - pr) / (n_known * (1 - q))
    var_r = rr * (1 - rr) / (n_known * q)
    return {
        "delay": math.sqrt(var_rtt) / 2 / math.sqrt(n_echo),
        "jitter": se_std_rtt / math.sqrt(2),
        "loss": math.sqrt(q * (1 - q) / n_known * (1 + lam) / (1 - lam)),
        "start": math.sqrt(var_p),
        "end": math.sqrt(var_r / (1 - pr) ** 2 + rr ** 2 * var_p / (1 - pr) ** 4),
    }


def assert_recovers(fit: dict, p: EchoParams, *, check_burst: bool = True) -> None:
    truth = truth_summary(p)
    ses = expected_ses(p, fit["loss"]["n"], fit["delay"]["n"])
    one = fit["delay"]["one_way_estimate"]
    jitter_true = math.sqrt(truth["echo_visible_jitter_s"] ** 2 + ROUND_US ** 2 / 6.0)
    assert abs(one["delay_s"]["value"] - truth["echo_visible_delay_s"]) <= K_SE * ses["delay"]
    assert abs(one["delay_jitter_s"]["value"] - jitter_true) <= K_SE * ses["jitter"]
    assert abs(fit["loss"]["loss_rate"] - truth["outbound_stationary_loss"]) <= K_SE * ses["loss"]
    if check_burst:
        chain = fit["loss"]["burst_chain"]
        assert abs(chain["burst_start_probability"]["value"] - p.burst_start) <= K_SE * ses["start"]
        assert abs(chain["burst_end_probability"]["value"] - p.burst_end) <= K_SE * ses["end"]


# ----------------------------------------------------------------------------
# 0. the burst-chain inversion, checked against the real simulator
# ----------------------------------------------------------------------------

def test_burst_chain_inversion_matches_simulated_link():
    s, e = 0.03, 0.35
    link = SimulatedLink(LinkConfig(loss_model="burst", burst_start_probability=s, burst_end_probability=e), seed=7)
    n = 200_000
    # send() decides loss by calling _loss_reason() once per packet; calling it directly skips the delay
    # bookkeeping, which is irrelevant here, and keeps the test fast.
    lost = np.array([link._loss_reason() is not None for _ in range(n)], dtype=int)
    a, b = lost[:-1], lost[1:]
    p_obs = float(((a == 0) & (b == 1)).sum() / (a == 0).sum())
    r_obs = float(((a == 1) & (b == 0)).sum() / (a == 1).sum())
    p_true, r_true = gilbert_from_burst_chain(s, e)  # (s, e (1 - s))
    q = stationary_loss(s, e)
    lam = 1 - p_true - r_true
    # Binomial SEs for transition fractions; burst-aware SE for the lost fraction.
    assert abs(p_obs - p_true) <= K_SE * math.sqrt(p_true * (1 - p_true) / (n * (1 - q)))
    assert abs(r_obs - r_true) <= K_SE * math.sqrt(r_true * (1 - r_true) / (n * q))
    assert abs(lost.mean() - q) <= K_SE * math.sqrt(q * (1 - q) / n * (1 + lam) / (1 - lam))
    # The inverse map gives back (s, e) exactly, and agrees with s6_recovery's own formula.
    start, end, clipped = burst_chain_from_gilbert(p_true, r_true)
    assert start == pytest.approx(s, abs=1e-15) and end == pytest.approx(e, abs=1e-15) and not clipped
    assert _burst_start_probability(q, e) == pytest.approx(s, rel=1e-12)
    assert mean_loss_run(s, e) == pytest.approx(1 / (e * (1 - s)))
    # end above one (losses less clumped than independent) is clipped and flagged.
    assert burst_chain_from_gilbert(0.2, 0.9) == (0.2, 1.0, True)


# ----------------------------------------------------------------------------
# 1. recovery on a large run
# ----------------------------------------------------------------------------

RECOVERY = EchoParams(n_packets=20_000, out_delay_mean_s=0.003, out_delay_std_s=0.001, back_delay_mean_s=0.003,
                      back_delay_std_s=0.001, burst_start=0.03, burst_end=0.35, return_loss=0.02)


@pytest.fixture(scope="module")
def recovery_fit():
    trace, logs = make_trace(RECOVERY, seed=11)
    return fit_echo_trace(trace, n_boot=200, seed=1), logs


def test_recovery_large_run(recovery_fit):
    fit, _ = recovery_fit
    assert fit["schema"] == FIT_SCHEMA
    assert_recovers(fit, RECOVERY)
    assert fit["loss"]["model_choice"]["loss_model"] == "burst"
    # Return leg: q_ret = 0.02 among the ~18,700 logged packets, binomial SE.
    ret = fit["return_leg"]
    se = math.sqrt(0.02 * 0.98 / ret["n_echoer_logged"])
    assert abs(ret["loss_rate"] - 0.02) <= K_SE * se
    # Output blocks are usable as-is.
    LinkConfig(**fit["link_config"])
    json.dumps(fit, allow_nan=False)
    s6 = fit["s6_deliver"]
    assert _burst_start_probability(s6["loss_probability"], 1 / s6["burst_mean_length_packets"]) == pytest.approx(
        fit["link_config"]["burst_start_probability"], rel=1e-9)
    out = deliver("constant_velocity", noise_std_m=0.05, seed=3, config=RecoveryConfig(duration_s=2.0, burn_in_s=0.5),
                  **s6)
    assert out  # the block is accepted by deliver()


# ----------------------------------------------------------------------------
# 2. coverage of the reported 95% intervals
# ----------------------------------------------------------------------------

def test_interval_coverage_over_seeds():
    """40 independent runs; count how often the 95% interval holds the truth.

    If the intervals are right, the count is Binomial(40, 0.95). P(count <= 33)
    = 0.0034, so we accept 34 to 40. The plain Wilson interval for the lost
    fraction ignores clumping; with lam = 1 - 0.02 - 0.4 * 0.98 = 0.588 its
    width is too small by sqrt((1 + lam) / (1 - lam)) = 1.96, so its true
    coverage is about P(|Z| < 1.0) = 0.68, and it must land below 34.
    """
    p = EchoParams(n_packets=1500, out_delay_mean_s=0.002, out_delay_std_s=0.0006, back_delay_mean_s=0.002,
                   back_delay_std_s=0.0006, burst_start=0.02, burst_end=0.4, return_loss=0.01)
    truth = truth_summary(p)
    hit_delay = hit_loss = hit_loss_naive = 0
    for seed in range(1000, 1040):
        trace, _ = make_trace(p, seed)
        fit = fit_echo_trace(trace, n_boot=300, seed=seed)
        lo, hi = fit["delay"]["one_way_estimate"]["delay_s"]["ci"]
        hit_delay += lo <= truth["echo_visible_delay_s"] <= hi
        lo, hi = fit["loss"]["loss_rate_ci"]
        hit_loss += lo <= truth["outbound_stationary_loss"] <= hi
        lo, hi = fit["loss"]["loss_rate_ci_wilson_independent"]
        hit_loss_naive += lo <= truth["outbound_stationary_loss"] <= hi
    assert 34 <= hit_delay <= 40, hit_delay
    assert 34 <= hit_loss <= 40, hit_loss
    assert hit_loss_naive < 34, hit_loss_naive


# ----------------------------------------------------------------------------
# 3. limiting cases
# ----------------------------------------------------------------------------

def test_zero_jitter_zero_loss():
    p = EchoParams(n_packets=2000, out_delay_mean_s=0.002, out_delay_std_s=0.0, back_delay_mean_s=0.002,
                   back_delay_std_s=0.0, burst_start=0.0, return_loss=0.0)
    trace, _ = make_trace(p, seed=5)
    fit = fit_echo_trace(trace, n_boot=200)
    one = fit["delay"]["one_way_estimate"]
    # Each round trip is off from 4 ms by less than 2 us (two differences of whole-us stamps),
    # so every rtt / 2, and therefore their mean, is within 1 us; their spread is at most 2 us / sqrt(2).
    assert abs(one["delay_s"]["value"] - 0.002) < 1e-6
    assert one["delay_jitter_s"]["value"] < 2e-6 / math.sqrt(2)
    loss = fit["loss"]
    assert loss["n_lost"] == 0 and loss["loss_rate"] == 0.0 and loss["n"] == 2000
    # Exact zero-loss bound 1 - 0.05^(1/n) = 1 - exp(-2.996 / n). That is below the rule of three 3 / n
    # by 0.13% (2.996 vs 3) plus about x / 2 = 0.075% (1 - exp(-x) is about x (1 - x / 2)): 0.2% in all.
    assert loss["zero_loss_upper_bound"]["value"] == pytest.approx(1 - 0.05 ** (1 / 2000), rel=1e-12)
    assert loss["zero_loss_upper_bound"]["value"] == pytest.approx(3 / 2000, rel=0.003)
    assert fit["link_config"]["loss_model"] == "independent" and fit["link_config"]["loss_probability"] == 0.0
    assert fit["return_leg"]["loss_rate"] == 0.0
    LinkConfig(**fit["link_config"])


def test_huge_hold_time_is_removed():
    base = EchoParams(n_packets=3000, burst_start=0.0, return_loss=0.0)
    slow = replace(base, hold_min_s=0.05, hold_max_s=0.10)
    fit_a = fit_echo_trace(make_trace(base, seed=9)[0], n_boot=50)
    trace_b, _ = make_trace(slow, seed=9)
    fit_b = fit_echo_trace(trace_b, n_boot=50)
    # Same seed, so the leg delays are the same draws; only the stamp rounding differs.
    # Each run's delay_s is within 1 us of the same true value, so they differ by under 2 us.
    da = fit_a["delay"]["one_way_estimate"]["delay_s"]["value"]
    db = fit_b["delay"]["one_way_estimate"]["delay_s"]["value"]
    assert abs(da - db) < 2e-6
    # Without the subtraction the estimate would be off by mean(hold) / 2 = 37.5 ms; the hold is
    # uniform with std 14.4 ms, so the SE of that naive offset is 14.4 / 2 / sqrt(3000) ms = 0.13 ms.
    naive = np.mean([(pk.echo_time_s - pk.send_time_s) / 2 for pk in trace_b.packets if pk.echo_time_s is not None])
    assert abs((naive - db) - 0.0375) < K_SE * 0.0144 / 2 / math.sqrt(3000)


def test_asymmetric_legs_show_documented_bias():
    p = EchoParams(n_packets=5000, out_delay_mean_s=0.002, out_delay_std_s=0.0005, back_delay_mean_s=0.006,
                   back_delay_std_s=0.0005, burst_start=0.0, return_loss=0.0)
    fit = fit_echo_trace(make_trace(p, seed=21)[0], n_boot=100)
    delay = fit["delay"]["one_way_estimate"]["delay_s"]["value"]
    se = math.sqrt(2 * 0.0005 ** 2) / 2 / math.sqrt(5000)  # 7 us
    # The fit sees (2 + 6) / 2 = 4 ms, not the true outbound 2 ms: the bias is (6 - 2) / 2 = 2 ms.
    assert abs(delay - 0.004) <= K_SE * se
    assert delay - p.out_delay_mean_s == pytest.approx(0.002, abs=K_SE * se)
    assert any("asymmetry" in text for text in fit["delay"]["consequences"])


def test_clock_drift_effect_is_negligible_and_as_computed():
    """Drift scales each board's differences by (1 + ppm 1e-6).

    measured rtt = (1 + dA)(d_out + d_back + h) - (1 + dB) h = (1 + dA)(d_out + d_back) + (dA - dB) h,
    so delay_s moves by dA mean(d_out + d_back) / 2 + (dA - dB) mean(h) / 2
    = 50e-6 * 4 ms / 2 + 100e-6 * 40 us / 2 = 0.100 us + 0.002 us = 0.102 us.
    Stamp rounding adds sqrt(1/3) / 2 us per packet to rtt / 2, so each run's mean carries
    0.29 / sqrt(5000) = 0.004 us of noise and the difference about 0.006 us; 0.03 us is 5 of those.
    """
    base = EchoParams(n_packets=5000, out_delay_mean_s=0.002, out_delay_std_s=0.0005, back_delay_mean_s=0.002,
                      back_delay_std_s=0.0005, burst_start=0.0, return_loss=0.0)
    drift = replace(base, sender_drift_ppm=50.0, echoer_drift_ppm=-50.0)
    t0, logs0 = make_trace(base, seed=33)
    t1, _ = make_trace(drift, seed=33)
    d0 = fit_echo_trace(t0, n_boot=50)["delay"]["one_way_estimate"]["delay_s"]["value"]
    d1 = fit_echo_trace(t1, n_boot=50)["delay"]["one_way_estimate"]["delay_s"]["value"]
    real = logs0.truth["realized"]
    expected = 50e-6 * (real["mean_out_delay_s"] + real["mean_back_delay_s"]) / 2 + 100e-6 * 40e-6 / 2
    assert expected == pytest.approx(1.02e-7, rel=0.01)
    assert abs((d1 - d0) - expected) < 3e-8


# ----------------------------------------------------------------------------
# 4. fitted settings fed back into the real simulator
# ----------------------------------------------------------------------------

def test_fitted_burst_settings_reproduce_observed_loss_in_simulator(recovery_fit):
    fit, _ = recovery_fit
    cfg = fit["link_config"]
    start, end = cfg["burst_start_probability"], cfg["burst_end_probability"]
    link = SimulatedLink(LinkConfig(loss_model="burst", burst_start_probability=start, burst_end_probability=end),
                         seed=99)
    n_sim = 200_000
    lost = np.array([link._loss_reason() is not None for _ in range(n_sim)], dtype=int)  # what send() uses
    q_sim = float(lost.mean())
    # Simulator's lost fraction vs the observed one: burst-aware SE of the 200k-packet sim run, plus the
    # observed run's own burst-aware SE (both are noisy estimates of the same long-run fraction).
    p_f, r_f = gilbert_from_burst_chain(start, end)
    q_f = stationary_loss(start, end)
    lam = 1 - p_f - r_f
    infl = (1 + lam) / (1 - lam)
    n_obs = fit["loss"]["n"]
    se = math.sqrt(q_f * (1 - q_f) * infl * (1 / n_sim + 1 / n_obs))
    assert abs(q_sim - fit["loss"]["loss_rate"]) <= K_SE * se
    # Mean run of lost packets (complete runs only, as fit_gilbert counts them). Runs are geometric
    # with end-of-run chance r, so each run length has std sqrt(1 - r) / r.
    padded = np.concatenate([[0], lost, [0]])
    edges = np.diff(padded)
    runs = np.nonzero(edges == -1)[0] - np.nonzero(edges == 1)[0]
    burst = fit["loss"]["gilbert_classic"]["burstiness"]
    obs_mean, obs_n = burst["observed_mean_loss_run"], burst["n_complete_loss_runs"]
    run_sd = math.sqrt(1 - r_f) / r_f
    se_run = run_sd * math.sqrt(1 / runs.size + 1 / obs_n)
    assert abs(runs.mean() - obs_mean) <= K_SE * se_run


# ----------------------------------------------------------------------------
# 5. the committed fixture
# ----------------------------------------------------------------------------

def _fixture_truth():
    truth = json.loads((FIXTURE_DIR / "truth.json").read_text(encoding="utf-8"))
    return truth, EchoParams(**truth["params"])


def test_committed_fixture_is_reproducible_and_small():
    truth, params = _fixture_truth()
    logs = generate_echo_logs(params, truth["seed"])
    assert (FIXTURE_DIR / "sender.csv").read_text(encoding="utf-8") == logs.sender_text
    assert (FIXTURE_DIR / "echoer.csv").read_text(encoding="utf-8") == logs.echoer_text
    for name in ("sender.csv", "echoer.csv"):
        assert (FIXTURE_DIR / name).stat().st_size < 150_000


def test_committed_fixture_recovery():
    truth, params = _fixture_truth()
    sender = parse_csv(FIXTURE_DIR / "sender.csv", 3)
    echoer = parse_csv(FIXTURE_DIR / "echoer.csv", 4)
    trace = EspNowTrace.from_dict(trace_from_rows(sender, echoer, trace_id="fixture-in-test"))
    fit = fit_echo_trace(trace, n_boot=500, seed=0)
    assert fit["source_trace"]["status"] == "synthetic" and "synthetic" in fit["status_note"]
    assert fit["delay"]["n"] == truth["realized"]["n_echoes"]
    assert fit["loss"]["n_duplicates_excluded"] == truth["realized"]["n_duplicate_arrivals"]
    # Drift here is +20 / -15 ppm: moves delay_s by about 20e-6 * 2.5 ms = 0.05 us, far inside the SE (~11 us).
    assert_recovers(fit, params)
    LinkConfig(**fit["link_config"])


def test_end_to_end_csv_import_then_fit():
    """CSV files -> tools/import_espnow_log.read_echo_logs -> fit. Needs the parallel importer."""
    from tools.import_espnow_log import read_echo_logs

    _, params = _fixture_truth()
    trace = read_echo_logs(FIXTURE_DIR / "sender.csv", FIXTURE_DIR / "echoer.csv", trace_id="fixture-e2e",
                           status="synthetic")
    fit = fit_echo_trace(trace, n_boot=500, seed=0)
    # The importer and this file's own row-to-trace helper follow the same labelling table,
    # so the counts and the point estimates must agree exactly.
    sender = parse_csv(FIXTURE_DIR / "sender.csv", 3)
    echoer = parse_csv(FIXTURE_DIR / "echoer.csv", 4)
    ref = fit_echo_trace(EspNowTrace.from_dict(trace_from_rows(sender, echoer)), n_boot=500, seed=0)
    for key in ("n_received", "n_lost", "n_unknown_excluded", "n_duplicates_excluded"):
        assert fit["loss"][key] == ref["loss"][key], key
    assert fit["delay"]["n"] == ref["delay"]["n"]
    assert fit["delay"]["n_without_turnaround"] == 0
    assert fit["delay"]["one_way_estimate"]["delay_s"]["value"] == pytest.approx(
        ref["delay"]["one_way_estimate"]["delay_s"]["value"], abs=1e-12)
    assert_recovers(fit, params)


# ----------------------------------------------------------------------------
# 6. errors and empty cases, plus the CLI
# ----------------------------------------------------------------------------

def test_non_echo_trace_rejected():
    doc = trace_from_rows([(0, 1000, None)], [(0, 500, None, -40)])
    doc["clock_domain"] = "shared_monotonic"
    doc["packets"] = [{"packet_id": "x", "sender_id": "A", "receiver_id": "B", "sender_session_id": "s", "seq": 0,
                       "send_time_s": 0.0, "outcome": "received", "receive_time_s": 0.001}]
    with pytest.raises(ValueError, match="sender_round_trip"):
        fit_echo_trace(EspNowTrace.from_dict(doc))


def test_no_echoes_gives_null_delay_not_crash():
    sender = [(i, 1000 + 20000 * i, None) for i in range(50)]
    echoer = [(i, 500 + 20000 * i, None, -45) for i in range(50) if i % 7]
    fit = fit_echo_trace(EspNowTrace.from_dict(trace_from_rows(sender, echoer)), n_boot=10)
    assert fit["delay"] is None and "no packet" in fit["delay_null_reason"]
    assert fit["link_config"] is None and fit["s6_deliver"] is None and fit["link_config_null_reason"]
    assert fit["loss"]["n_lost"] > 0
    assert fit["return_leg"]["loss_rate"] == 1.0  # the echoer never echoed
    json.dumps(fit, allow_nan=False)


def test_without_echoer_log_return_leg_is_not_identifiable():
    p = EchoParams(n_packets=300, burst_start=0.05, burst_end=0.5, return_loss=0.05)
    logs = generate_echo_logs(p, 4)
    # Drop the echoer's hold times to mimic a sender-only capture.
    doc = trace_from_rows(logs.sender_rows, [], with_echoer=False)
    fit = fit_echo_trace(EspNowTrace.from_dict(doc), n_boot=20)
    assert fit["return_leg"]["loss_rate"] is None and "echoer log" in fit["return_leg"]["reason"]
    assert fit["delay"]["n_without_turnaround"] == fit["delay"]["n"]
    # Only packets that survived both legs are known, so an outbound rate would be biased to zero.
    assert fit["loss"]["loss_rate"] is None and "biased" in fit["loss"]["reason"]
    assert fit["loss"]["n_unknown_excluded"] > 0 and fit["link_config"] is None


def test_cli_on_trace_json(tmp_path):
    sender = parse_csv(FIXTURE_DIR / "sender.csv", 3)
    echoer = parse_csv(FIXTURE_DIR / "echoer.csv", 4)
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace_from_rows(sender, echoer)), encoding="utf-8")
    out = tmp_path / "fit.json"
    assert main([str(trace_path), "--out", str(out), "--n-boot", "100", "--seed", "4"]) == 0
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    result = json.loads(raw)
    assert result["schema"] == FIT_SCHEMA
    LinkConfig(**result["link_config"])
    assert main(["--sender", str(FIXTURE_DIR / "sender.csv")]) == 1  # --status is required
    assert main([]) == 1
