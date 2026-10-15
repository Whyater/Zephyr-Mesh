"""Main estimators against the separately written reference (tests/reference_b).

The reference package was written independently from the model equations,
without looking at the main code. Every comparison below drives both sides with
the SAME inputs and requires agreement to 1e-9 absolute.

Shared inputs: a seeded measurement sequence of more than 50 updates with a
straight leg, a left turn, a climb on z, irregular time steps and one gap of
2.5 s, plus prediction queries at several horizons.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # makes tests/reference_b importable

from reference_b import ref_ca, ref_ct, ref_cv, ref_imm, ref_morris, ref_nis, ref_ntp, ref_quant_event, ref_retro  # noqa: E402

from sim.estimation.base import ArmSpec, EstimatorSettings  # noqa: E402
from sim.estimation.ca import CAEstimator  # noqa: E402
from sim.estimation.consistency import AdaptiveEstimator, chi2_gate_threshold, nis_band  # noqa: E402
from sim.estimation.ct import CTEstimator  # noqa: E402
from sim.estimation.cv import CVEstimator  # noqa: E402
from sim.estimation.engine import Bundle, CellSpec, Delivery, run_arm_detailed  # noqa: E402
from sim.estimation.imm import IMMEstimator  # noqa: E402
from sim.estimation import morris as main_morris  # noqa: E402
from sim.estimation import policies  # noqa: E402
from sim.link import LinkConfig, SimulatedLink  # noqa: E402
from sim import timesync  # noqa: E402

TOL = 1e-9
SETTINGS = EstimatorSettings()
R_STD = SETTINGS.measurement_std_m
HORIZONS = (0.0, 0.02, 0.1, 0.5, 1.0, 3.0)
START = np.zeros(3)


def truth(t):
    """Straight along +x for 3 s, a 1 m radius left turn at 1 rad/s for pi s,
    then straight along -x; z climbs at 0.1 m/s throughout."""
    t = float(t)
    t_turn = 3.0
    if t < t_turn:
        x, y = t, 0.0
    elif t < t_turn + np.pi:
        a = t - t_turn
        x, y = t_turn + np.sin(a), 1.0 - np.cos(a)
    else:
        x, y = t_turn - (t - t_turn - np.pi), 2.0
    return np.array([x, y, 0.1 * t])


def circle(t):
    """The exact constant-turn target (w = 1 rad/s, radius 1 m)."""
    return np.array([np.sin(t), 1.0 - np.cos(t), 0.0])


def make_sequence(seed=20261014, n_before_gap=30, n_after_gap=35, gap_s=2.5, path=truth):
    """Stamps start at 0 with irregular steps in [0.03, 0.2] s and one gap of gap_s."""
    rng = np.random.default_rng(seed)
    steps = list(rng.uniform(0.03, 0.2, size=n_before_gap - 1)) + [gap_s] + list(
        rng.uniform(0.03, 0.2, size=n_after_gap - 1))
    stamps = np.concatenate([[0.0], np.cumsum(steps)])
    zs = np.array([path(t) for t in stamps]) + rng.normal(0.0, R_STD, size=(stamps.size, 3))
    return stamps, zs


STAMPS, ZS = make_sequence()
CHECKPOINTS = (0, 1, 5, 28, 29, 30, 40, len(STAMPS) - 1)  # 29 is the first update after the 2.5 s gap


def test_shared_inputs_cover_the_required_cases():
    assert STAMPS.size >= 50
    gaps = np.diff(STAMPS)
    assert gaps.max() >= 2.0
    assert np.unique(np.round(gaps, 9)).size > 10  # irregular steps
    assert STAMPS[-1] > 3.0 + np.pi  # the turn is fully inside the run


def _assert_close(a, b, what):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, what
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert diff <= TOL, f"{what}: max abs difference {diff:.3e}"
    return diff


# ---------------------------------------------------------------- linear and EKF models

def _run_main(est, stamps, zs):
    """Yield (index, est) after each update."""
    for k, (t, z) in enumerate(zip(stamps, zs)):
        est.update(z, float(t))
        yield k, est


def _check_queries(est, ref_predict, x, P, t, label):
    for h in HORIZONS:
        pred = est.predict_at(t + h)
        xr, Pr = ref_predict(x, P, t, t + h)
        _assert_close(pred.position_m, xr[:3], f"{label} position at +{h}")
        _assert_close(pred.velocity_mps, xr[3:6], f"{label} velocity at +{h}")
        _assert_close(pred.position_cov, Pr[:3, :3], f"{label} position cov at +{h}")
    batch = est.predict_positions(np.array([t + h for h in HORIZONS]))
    ref_batch = np.array([ref_predict(x, P, t, t + h)[0][:3] for h in HORIZONS])
    _assert_close(batch, ref_batch, f"{label} predict_positions")


def test_cv_matches_reference():
    ref = ref_cv.run_cv(ZS, STAMPS, START, q_acc=SETTINGS.q_acc_psd, r_std=R_STD)
    est = CVEstimator(START, SETTINGS)
    for k, est in _run_main(est, STAMPS, ZS):
        x, P = ref[k]
        _assert_close(est.tracker.x, x, f"cv state {k}")
        _assert_close(est.tracker.P, P, f"cv cov {k}")
        if k in CHECKPOINTS:
            _check_queries(est, lambda x, P, a, b: ref_cv.predict(x, P, a, b, SETTINGS.q_acc_psd),
                           x, P, float(STAMPS[k]), f"cv {k}")


def test_ca_matches_reference():
    ref = ref_ca.run_ca(ZS, STAMPS, START, q_jerk=SETTINGS.q_jerk_psd, r_std=R_STD)
    est = CAEstimator(START, SETTINGS)
    for k, est in _run_main(est, STAMPS, ZS):
        x, P = ref[k]
        _assert_close(est.x, x, f"ca state {k}")
        _assert_close(est.P, P, f"ca cov {k}")
        if k in CHECKPOINTS:
            _check_queries(est, lambda x, P, a, b: ref_ca.predict(x, P, a, b, SETTINGS.q_jerk_psd),
                           x, P, float(STAMPS[k]), f"ca {k}")


@pytest.mark.parametrize("known_turn_rate", [None, 1.0])
def test_ct_matches_reference(known_turn_rate):
    path = truth if known_turn_rate is None else circle
    stamps, zs = make_sequence(seed=7 if known_turn_rate else 8, path=path)
    q_turn = 0.0 if known_turn_rate is not None else SETTINGS.q_turn_psd
    ref = ref_ct.run_ct(zs, stamps, START, q_acc=SETTINGS.q_acc_psd, r_std=R_STD,
                        known_turn_rate=known_turn_rate, q_turn=SETTINGS.q_turn_psd)
    options = {} if known_turn_rate is None else {"known_turn_rate": known_turn_rate}
    est = CTEstimator(START, SETTINGS, **options)
    used_series = used_closed_form = False
    for k, est in _run_main(est, stamps, zs):
        x, P = ref[k]
        _assert_close(est.x, x, f"ct state {k}")
        _assert_close(est.P, P, f"ct cov {k}")
        if k + 1 < stamps.size:
            u = abs(est.x[6] * (stamps[k + 1] - stamps[k]))
            used_series |= u < 1e-4
            used_closed_form |= u >= 1e-4
        if k in CHECKPOINTS:
            _check_queries(est, lambda x, P, a, b: ref_ct.predict(x, P, a, b, SETTINGS.q_acc_psd, q_turn),
                           x, P, float(stamps[k]), f"ct {k}")
    assert used_closed_form
    if known_turn_rate is None:
        assert abs(ref[-1][0][6]) > 0.1  # the turn rate was actually estimated
    # The small-angle series is exercised at w = 0 (first step) in the estimated case.
    assert used_series or known_turn_rate is not None


def _main_combined(est):
    return IMMEstimator._combine(est.mode_states, est.mode_probabilities)


def test_imm_matches_reference():
    ref = ref_imm.run_imm(ZS, STAMPS, START, q_acc=SETTINGS.q_acc_psd, r_std=R_STD, q_turn=SETTINGS.q_turn_psd)
    est = IMMEstimator(START, SETTINGS)
    _assert_close(est.mode_probabilities, [0.5, 0.5], "imm initial mu")
    max_ct = 0.0
    for k, est in _run_main(est, STAMPS, ZS):
        xr, Pr, mur = ref[k]
        x, P = _main_combined(est)
        _assert_close(est.mode_probabilities, mur, f"imm mu {k}")
        _assert_close(x, xr, f"imm combined state {k}")
        _assert_close(P, Pr, f"imm combined cov {k}")
        max_ct = max(max_ct, float(mur[1]))
        if k in CHECKPOINTS:
            t = float(STAMPS[k])
            queries = [t + h for h in HORIZONS]
            ref_q = ref_imm.run_imm(ZS[:k + 1], STAMPS[:k + 1], START, q_acc=SETTINGS.q_acc_psd, r_std=R_STD,
                                    q_turn=SETTINGS.q_turn_psd, query=queries)
            for h, (xq, Pq, _) in zip(HORIZONS, ref_q):
                pred = est.predict_at(t + h)
                _assert_close(pred.position_m, xq[:3], f"imm position {k} +{h}")
                _assert_close(pred.velocity_mps, xq[3:6], f"imm velocity {k} +{h}")
                _assert_close(pred.position_cov, Pq[:3, :3], f"imm position cov {k} +{h}")
            _assert_close(est.predict_positions(np.array(queries)), np.array([q[0][:3] for q in ref_q]),
                          f"imm predict_positions {k}")
    assert max_ct > 0.6  # the turn moved the mode probabilities, so mixing was really tested


def test_imm_switch_probability_is_the_d093_formula():
    est = IMMEstimator(START, SETTINGS)
    for dt in (0.0, 0.1, 1.0, 2.0 * np.log(2.0), 5.0, 50.0):
        p = (1.0 - np.exp(-2.0 * dt / 2.0)) / 2.0
        _assert_close(est._transition(dt), [[1 - p, p], [p, 1 - p]], f"Pi({dt})")
        assert p <= 0.5


# ---------------------------------------------------------------- retrodiction

class _FixedTarget:
    start_position = START

    @staticmethod
    def position(times):
        return np.array([truth(t) for t in np.asarray(times, dtype=float)])


def _retro_packets(seed=99):
    """10 Hz captures to 8 s with random delays (some packets very late), the
    t = 0 packet arriving after the t = 0.1 one, captures 3.0 to 5.5 s missing
    (a 2.6 s gap), and one duplicate."""
    rng = np.random.default_rng(seed)
    captures = np.round(np.arange(0, 81) * 0.1, 10)
    captures = captures[(captures < 3.0) | (captures > 5.55)]
    delays = rng.lognormal(np.log(0.08), 0.8, size=captures.size)
    delays[0], delays[1] = 0.25, 0.02  # t = 0 arrives late, into the fresh filter
    zs = np.array([truth(t) for t in captures]) + rng.normal(0.0, R_STD, size=(captures.size, 3))
    packets = [(float(c), z, float(c + d)) for c, z, d in zip(captures, zs, delays)]
    packets.append((float(captures[20]), zs[20] + 0.01, float(captures[20] + 0.3)))  # duplicate stamp
    packets.sort(key=lambda p: p[2])
    return packets


def _bundle(deliveries):
    """A Bundle holding hand-made deliveries. Byte counts do not affect the
    filter, so every required field the run does not use is filled with 0."""
    given = {"deliveries": deliveries, "target": _FixedTarget(), "capture_times_s": np.array([])}
    required = [f.name for f in dataclasses.fields(Bundle)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    return Bundle(**{name: given.get(name, 0) for name in required})


@pytest.mark.parametrize("window", [0.15, 0.4])
def test_retrodiction_matches_brute_force_replay(window):
    packets = _retro_packets()
    deliveries = [Delivery(seq=i, capture_time_s=s, stamp_s=s, receive_time_s=r, measurement_m=tuple(z))
                  for i, (s, z, r) in enumerate(packets)]
    assert max(d.receive_time_s for d in deliveries) < 11.0
    arm = ArmSpec("retro", model="cv", stamping="capture", late_policy="retrodict", retro_window_s=window)
    ref = ref_retro.replay_packets(packets, window, q_acc=SETTINGS.q_acc_psd, r_std=R_STD,
                                   start_time=0.0, start_position=START)
    refiltered = dropped = 0
    for n in range(1, len(deliveries) + 1):
        run = run_arm_detailed(_bundle(deliveries[:n]), CellSpec(), arm)
        x, P = ref[n - 1]
        _assert_close(run.estimator.tracker.x, x, f"retro window {window} state after arrival {n}")
        _assert_close(run.estimator.tracker.P, P, f"retro window {window} cov after arrival {n}")
        refiltered, dropped = run.outcome.refiltered_count, run.outcome.dropped_count
    assert refiltered >= 5  # late packets were really put back in order
    # The duplicate is always dropped; the narrow window also drops packets older than it.
    assert dropped >= (2 if window < 0.2 else 1)
    # The t = 0 packet arrived after the t = 0.1 one and went into the fresh filter.
    assert any(d.stamp_s == 0.0 for d in run.fused)


# ---------------------------------------------------------------- NIS, band, gate, adaptive

def _ref_nis_cv(x, P, t_from, t, z, q_acc=SETTINGS.q_acc_psd, r_var=R_STD**2):
    xp, Pp = ref_cv.predict(x, P, t_from, t, q_acc)
    return ref_nis.nis(np.asarray(z) - xp[:3], Pp[:3, :3] + r_var * np.eye(3))


def test_nis_matches_reference_for_every_model():
    R = R_STD**2 * np.eye(3)
    # CV, CA, CT: reference innovation from the reference filter state before each update.
    for name, est, run, predict in (
        ("cv", CVEstimator(START, SETTINGS), ref_cv.run_cv(ZS, STAMPS, START),
         lambda x, P, a, b: ref_cv.predict(x, P, a, b, 0.1)),
        ("ca", CAEstimator(START, SETTINGS), ref_ca.run_ca(ZS, STAMPS, START),
         lambda x, P, a, b: ref_ca.predict(x, P, a, b, 0.1)),
        ("ct", CTEstimator(START, SETTINGS), ref_ct.run_ct(ZS, STAMPS, START),
         lambda x, P, a, b: ref_ct.predict(x, P, a, b, 0.1, 0.1)),
    ):
        n = {"cv": 6, "ca": 9, "ct": 7}[name]
        x0 = np.zeros(n)
        prev = (x0, np.eye(n), 0.0)
        for k, (t, z) in enumerate(zip(STAMPS, ZS)):
            xp, Pp = predict(prev[0], prev[1], prev[2], t)
            _assert_close(est.nis(z, t), ref_nis.nis(z - xp[:3], Pp[:3, :3] + R), f"{name} nis {k}")
            est.update(z, t)
            prev = (run[k][0], run[k][1], t)
    # IMM: reference combined prediction from the reference run on the prefix.
    est = IMMEstimator(START, SETTINGS)
    for k, (t, z) in enumerate(zip(STAMPS, ZS)):
        if k % 4 == 0 or k in CHECKPOINTS:
            ((xq, Pq, _),) = ref_imm.run_imm(ZS[:k], STAMPS[:k], START, query=[t])
            _assert_close(est.nis(z, t), ref_nis.nis(z - xq[:3], Pq[:3, :3] + R), f"imm nis {k}")
        est.update(z, t)


def test_band_and_gate_threshold_match_reference():
    for W in (1, 2, 5, 10, 20, 50):
        _assert_close(nis_band(W), ref_nis.chi_square_window_band(W), f"band W={W}")
    for g in (0.9, 0.95, 0.99, 0.999):
        _assert_close(chi2_gate_threshold(g), ref_nis.gate_threshold(g), f"gate g={g}")


@pytest.mark.parametrize("mode,noise_std,path", [("r", 0.15, truth), ("q", 0.05, truth), ("r", 0.01, circle)])
def test_adaptive_wrapper_matches_reference(mode, noise_std, path):
    rng = np.random.default_rng(5)
    stamps = np.concatenate([[0.0], np.cumsum(rng.uniform(0.03, 0.2, size=79))])
    zs = np.array([path(t) for t in stamps]) + rng.normal(0.0, noise_std, size=(stamps.size, 3))
    est = AdaptiveEstimator(CVEstimator(START, SETTINGS), mode=mode, window=10)
    x, P, t_prev = np.r_[START, 0.0, 0.0, 0.0], np.eye(6), 0.0
    nis_values, scale = [], 1.0
    for k, (t, z) in enumerate(zip(stamps, zs)):
        q = SETTINGS.q_acc_psd * (scale if mode == "q" else 1.0)
        r_std = R_STD * np.sqrt(scale if mode == "r" else 1.0)
        nis_values.append(_ref_nis_cv(x, P, t_prev, t, z, q_acc=q, r_var=r_std**2))
        x, P = ref_cv.predict(x, P, t_prev, t, q)
        x, P = ref_cv._update(x, P, z, r_std)
        t_prev = t
        scale = ref_nis.adaptive_scale(nis_values, mode=mode, W=10)[-1]
        est.update(z, t)
        _assert_close(est.scale, scale, f"adaptive {mode} scale {k}")
        _assert_close(est.base.tracker.x, x, f"adaptive {mode} state {k}")
        _assert_close(est.base.tracker.P, P, f"adaptive {mode} cov {k}")
    assert abs(np.log(scale)) > 0.1  # the scale really moved


# ---------------------------------------------------------------- clocks and NTP

def test_ntp_offset_formula_matches_reference():
    rng = np.random.default_rng(3)
    for _ in range(200):
        t1, t2, t3, t4 = rng.uniform(-5.0, 50.0, size=4)
        theta, delta = ref_ntp.offset_round_trip(t1, t2, t3, t4)
        _assert_close(timesync.offset_from_timestamps(t1, t2, t3, t4), theta, "theta")
        _assert_close(timesync.round_trip(t1, t2, t3, t4), delta, "delta")


@pytest.mark.parametrize("e0,drift_ppm,d_up,d_down,processing,t_send", [
    (0.0, 0.0, 0.005, 0.005, 0.001, 0.0),
    (0.25, 0.0, 0.004, 0.012, 0.001, 1.0),
    (-0.4, 50.0, 0.020, 0.003, 0.002, 7.5),
    (1.5, -200.0, 0.010, 0.010, 0.0, 11.0),
])
def test_ntp_exchange_with_fixed_delays_matches_reference(e0, drift_ppm, d_up, d_down, processing, t_send):
    up = SimulatedLink(LinkConfig(delay_s=d_up, loss_model="none"), seed=1)
    down = SimulatedLink(LinkConfig(delay_s=d_down, loss_model="none"), seed=2)
    clock = timesync.ClockModel(offset_s=e0, drift_ppm=drift_ppm)
    sample = timesync.ntp_exchange(t_send, clock, up, down, 0, processing_s=processing)
    ref = ref_ntp.simulate_exchange(e0, drift_ppm, d_up, d_down, processing, t_send)
    _assert_close([sample.t1, sample.t2, sample.t3, sample.t4], [ref.t1, ref.t2, ref.t3, ref.t4], "t1..t4")
    _assert_close(sample.theta_s, ref.theta, "theta")
    _assert_close(sample.delta_s, ref_ntp.offset_round_trip(ref.t1, ref.t2, ref.t3, ref.t4)[1], "delta")
    if drift_ppm == 0.0:  # hand result: theta = e0 + (d_up - d_down) / 2
        _assert_close(sample.theta_s, e0 + (d_up - d_down) / 2.0, "theta hand value")


@pytest.mark.parametrize("mode", ["unsynced", "ntp"])
def test_stamp_errors_match_reference_composition(mode):
    params = {"mode": mode, "offset_s": 0.3, "drift_ppm": 80.0, "sync_interval_s": 1.0,
              "sync_delay_s": 0.006, "sync_asymmetry_s": 0.004, "processing_s": 0.001}
    captures = np.arange(0, 121) * 0.1
    main = timesync.stamp_errors(captures, params, {"clock": 1, "sync_up": 2, "sync_down": 3})
    r = params["drift_ppm"] * 1e-6
    reading = captures + params["offset_s"] + r * captures  # L(t) of the clock model
    theta = np.zeros_like(captures)
    if mode == "ntp":
        d_up = params["sync_delay_s"] + params["sync_asymmetry_s"] / 2.0
        d_down = params["sync_delay_s"] - params["sync_asymmetry_s"] / 2.0
        exchanges = [ref_ntp.simulate_exchange(params["offset_s"], params["drift_ppm"], d_up, d_down,
                                               params["processing_s"], k * 1.0) for k in range(13)]
        for i, t in enumerate(captures):
            arrived = [ex for ex in exchanges if ex.t4 <= t]  # tracker clock is true time
            theta[i] = arrived[-1].theta if arrived else 0.0
    _assert_close(main, reading - theta - captures, f"stamp errors {mode}")


# ---------------------------------------------------------------- quantization and event trigger

def test_quantizer_matches_reference():
    rng = np.random.default_rng(4)
    values = np.concatenate([rng.uniform(-5, 5, size=3000), [0.05, -0.05, 0.15, 0.25, -0.25, 0.35],
                             [1e6, -1e6, 3276.75, -3276.85]])
    for q in (0.001, 0.01, 0.1):
        codes, z_hat = policies.quantize(values, q)
        _assert_close(z_hat, ref_quant_event.quantize_int16(values, q), f"quantize q={q}")
        assert codes.dtype == np.int16


def _event_inputs(seed=12):
    rng = np.random.default_rng(seed)
    times = np.round(np.arange(0, 81) * 0.1, 10)  # 8 s: straight, the full turn, straight again
    zs = np.array([truth(t) for t in times]) + rng.normal(0.0, R_STD, size=(times.size, 3))
    return times, zs


@pytest.mark.parametrize("delta", [0.0, 0.02, 0.1, 0.3])
@pytest.mark.parametrize("quantization", [None, 0.01])
@pytest.mark.parametrize("heartbeat", [None, 0.5])
def test_event_trigger_matches_reference(delta, quantization, heartbeat):
    times, zs = _event_inputs()
    params = {"kind": "event", "delta_m": delta}
    if quantization is not None:
        params["quantization_m"] = quantization
    if heartbeat is not None:
        params["heartbeat_s"] = heartbeat
    main = policies.apply_sender(times, zs, params, SETTINGS).send_indices
    ref = ref_quant_event.event_trigger(times, zs, delta, q_acc=SETTINGS.q_acc_psd, r_std=R_STD,
                                        heartbeat=heartbeat, quantization=quantization)
    assert list(main) == list(ref)
    if delta == 0.0:
        assert len(ref) == times.size


# ---------------------------------------------------------------- Morris elementary effects

def test_morris_matches_reference_on_a_fixed_design():
    k, r, levels = 5, 8, 4
    design, orders = main_morris.morris_trajectories(k, r, levels=levels, seed=11)
    delta = levels / (2.0 * (levels - 1))
    # The design itself follows the standard Morris design: grid base point <= 1 - Delta, one +Delta move per input.
    grid = np.arange(levels) / (levels - 1)
    for traj, order in zip(design, orders):
        assert np.all(np.min(np.abs(traj[0][:, None] - grid[None, :]), axis=1) < 1e-12)
        assert np.all(traj[0] <= 1.0 - delta + 1e-12)
        _assert_close(np.diff(traj, axis=0), np.eye(k)[order] * delta, "morris steps")
    coeffs = np.array([3.0, -1.0, 0.2, 0.0, 2.0])
    y = np.array([[coeffs @ p + 4.0 * p[0] * p[4] + np.sin(3 * p[1]) for p in traj] for traj in design])
    main = main_morris.elementary_effects(design, orders, y)
    mu_star, sigma, _ = ref_morris.morris_statistics(design.reshape(-1, k), y.reshape(-1), p=levels)
    _assert_close(main["mu_star"], mu_star, "mu_star")
    _assert_close(main["sigma"], sigma, "sigma")
    assert list(np.argsort(-main["mu_star"], kind="stable")) == list(np.argsort(-mu_star, kind="stable"))
