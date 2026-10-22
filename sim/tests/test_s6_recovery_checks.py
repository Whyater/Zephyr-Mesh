"""Independent physics and bookkeeping checks for the S6 recovery sweep."""

import math

import numpy as np
import pytest

from sim.tracker import ConstantVelocityTracker
from sim.s6_recovery import (
    ArmResult,
    Delivery,
    RecoveryConfig,
    deliver,
    median_interval,
    run_arms,
    seed_list,
    simulate_run,
    summarize_arm,
    truth_position,
    wilson_interval,
)


CFG = RecoveryConfig()
DELAYS = (0.025, 0.05, 0.10, 0.20, 0.30)


def _run(motion, delay, noise=0.0, loss=0.0, seed=20261013, **kwargs):
    return simulate_run(motion, delay_s=delay, loss_probability=loss,
                         noise_std_m=noise, seed=seed, config=CFG, **kwargs)


def test_zero_delay_loss_noise_arms_agree():
    """With d=p=sigma=0, both arms see z(t)=truth(t), so errors are identical."""
    for motion in ("constant_velocity", "constant_turn"):
        for seed in seed_list(3):
            arms = _run(motion, 0.0, seed=seed)
            np.testing.assert_allclose(arms["naive"].errors_m,
                                       arms["recovery"].errors_m, atol=1e-9, rtol=0)


def test_naive_cv_error_is_speed_times_delay():
    """A receive-time CV estimate lags a straight target by d: |e|=v*d, v=1."""
    for d in DELAYS:
        result = _run("constant_velocity", d)
        assert result["naive"].p95_error_m == pytest.approx(d, rel=0.03, abs=1e-6)
        assert result["naive"].max_error_m == pytest.approx(d, rel=0.05, abs=1e-6)


def test_recovery_cv_error_near_zero_for_every_delay():
    """CV extrapolation exactly removes a constant-velocity lag after convergence."""
    for d in DELAYS:
        assert _run("constant_velocity", d)["recovery"].p95_error_m < 1e-3


def test_recovery_turn_error_grows_with_delay_and_stays_below_naive():
    values = {d: _run("constant_turn", d) for d in (0.0, *DELAYS[1:])}
    p95 = [values[d]["recovery"].p95_error_m for d in (0.0, *DELAYS[1:])]
    assert all(p95[i + 1] >= p95[i] - 1e-4 for i in range(len(p95) - 1))
    for d in DELAYS[1:]:
        assert values[d]["recovery"].p95_error_m < values[d]["naive"].p95_error_m


@pytest.mark.xfail(strict=True, reason=(
    "An arc-only reading of the pre-registered turn check (which named the filter's lag on a curve but gave it no number) "
    "FAILED: the arc gap alone underestimates the growth by 3 to 4.5x because the constant-velocity filter's "
    "velocity estimate lags on a turn. Kept as a record; the alpha-beta model in "
    "test_recovery_turn_error_matches_alpha_beta_lag_plus_arc was added after seeing these results."))
def test_recovery_turn_error_increase_matches_arc_gap_alone():
    """The straight-vs-arc gap is v^2*d^2/(2r), 0.02 m at .2 s and .045 m at .3 s."""
    values = {d: _run("constant_turn", d) for d in (0.0, 0.20, 0.30)}
    for d, gap in ((0.20, 0.02), (0.30, 0.045)):
        increase = values[d]["recovery"].p95_error_m - values[0.0]["recovery"].p95_error_m
        assert gap / 3 <= increase <= 3 * gap


def test_recovery_turn_error_matches_alpha_beta_lag_plus_arc():
    """Refined hand model, added after the arc-only check above failed.

    A Kalman filter with fixed tuning settles to fixed gains alpha (position)
    and beta (velocity x T). Fed a constant acceleration a, such a filter
    settles with a constant lag (derived by hand from the alpha-beta update):
        position lag  b_x = (1 - alpha) a T^2 / beta
        velocity lag  b_v = a T (alpha / beta - 1/2)
    Predicting forward a horizon h from the last update then misses by about
        b_x + b_v h + a h^2 / 2
    (all three point the same way, toward the inside of the turn). The gains
    come from iterating the 1D Riccati equation here, not from sim.tracker.
    On the turn a = v^2 / r = 1 m/s^2. Errors grow through each 0.1 s gap
    between updates, so the p95 sits at the last query before the next packet
    lands: the largest multiple of the 0.02 s query step below T + d.
    """
    T, q, r_var = 0.1, 0.1, 0.05 ** 2
    F = np.array([[1.0, T], [0.0, 1.0]])
    Q = q * np.array([[T ** 3 / 3, T ** 2 / 2], [T ** 2 / 2, T]])
    P = np.eye(2)
    for _ in range(2000):
        Pm = F @ P @ F.T + Q
        K = Pm[:, 0] / (Pm[0, 0] + r_var)
        P = (np.eye(2) - np.outer(K, [1.0, 0.0])) @ Pm
    alpha, beta = K[0], K[1] * T
    a = 1.0
    b_x = (1 - alpha) * a * T ** 2 / beta
    b_v = a * T * (alpha / beta - 0.5)
    assert b_x == pytest.approx(0.0364, abs=0.001)
    assert b_v == pytest.approx(0.272, abs=0.002)
    for d in (0.0, 0.05, 0.10, 0.20, 0.30):
        h = (math.ceil(round((T + d) / 0.02, 9)) - 1) * 0.02
        model = b_x + b_v * h + a * h * h / 2
        measured = _run("constant_turn", d)["recovery"].p95_error_m
        assert measured == pytest.approx(model, rel=0.02)


def test_loss_effect_is_the_gap_not_the_loss():
    """What hurts is the longest gap between delivered packets, not the loss fraction.

    Turn, no delay, no noise, packet lists built by hand:
    (a) every packet; (b) every other packet (50 % loss, all gaps 0.2 s);
    (c) 50 % loss with one 1.1 s gap (and a shorter 1.0 s gap at the very end);
    (d) about 8 % loss with the same 1.1 s gap and no other gaps.
    Same loss, different longest gap: (b) and (c) differ a lot.
    Same longest gap, different loss: (c) and (d) have the same max error,
    because the state entering the gap is the same and the max is reached
    at the end of the gap.
    Hand bounds for the worst miss over a gap g = 1.1 s on the 1 m circle:
    the straight line drifts off the arc by |p(t+g) - p(t) - v g| =
    sqrt((g - sin g)^2 + (1 - cos g)^2) = 0.585 m. The filter adds its own
    lag, at most b_x + b_v g = 0.036 + 0.272 x 1.1 = 0.335 m (alpha-beta lag,
    see the turn test below), so by the triangle inequality the max error
    lies between 0.585 - 0.335 = 0.25 m and 0.585 + 0.335 = 0.92 m.
    """
    times = np.arange(0.0, 12.0 + 1e-9, 0.1)
    truth = truth_position("constant_turn", times, CFG)
    every = [Delivery(i, float(t), float(t), tuple(truth[i])) for i, t in enumerate(times)]
    regular = [p for p in every if p.seq % 2 == 0]
    # 20 early packets, a 1.1 s gap (seq 20 to 30), then every other packet:
    # 61 kept, the same count as the regular 50 % list.
    long_gap_half = [p for p in every if p.seq < 20 or (30 <= p.seq <= 110 and p.seq % 2 == 0)]
    # Same 1.1 s gap, every packet kept outside it: only 10 of 121 lost.
    long_gap_full = [p for p in every if p.seq < 20 or p.seq >= 30]
    assert len(long_gap_half) == len(regular) == 61 and len(long_gap_full) == 111
    a = run_arms(every, "constant_turn", CFG)["recovery"]
    b = run_arms(regular, "constant_turn", CFG)["recovery"]
    c = run_arms(long_gap_half, "constant_turn", CFG)["recovery"]
    d = run_arms(long_gap_full, "constant_turn", CFG)["recovery"]
    assert c.max_error_m > 3 * b.max_error_m > 3 * a.max_error_m
    assert d.max_error_m == pytest.approx(c.max_error_m, rel=1e-9)
    g = 1.1
    arc = math.hypot(g - math.sin(g), 1 - math.cos(g))
    assert arc == pytest.approx(0.585, abs=0.001)
    assert arc - 0.335 <= c.max_error_m <= arc + 0.335


def test_naive_estimate_is_recovery_estimate_minus_velocity_times_delay():
    """Why recovery can lose. With a fixed delay d both arms fuse the same packets
    with the same spacing, so their filters hold the same state, only time
    shifted by d (once the different start has been forgotten). The naive arm
    then predicts d less far forward:
        naive estimate = recovery estimate - v_hat x d.
    So naive beats recovery exactly when recovery's own error points ahead
    along the motion by more than half of v_hat x d. This rebuilds the recovery
    estimate with sim.tracker directly and checks the naive error from run_arms.
    """
    for motion, d, loss, noise in (("constant_turn", 0.1, 0.5, 0.0), ("constant_velocity", 0.025, 0.0, 0.1)):
        deliveries = deliver(motion, delay_s=d, loss_probability=loss, noise_std_m=noise,
                             seed=seed_list(1)[0], config=CFG)
        naive = run_arms(deliveries, motion, CFG)["naive"]
        tracker = ConstantVelocityTracker(position_m=truth_position(motion, [0.0], CFG)[0],
                                          acceleration_spectral_density=0.1, measurement_std_m=0.05)
        cursor = 0
        for i, t in enumerate(naive.query_times_s):
            while cursor < len(deliveries) and deliveries[cursor].receive_time_s <= t + 1e-12:
                tracker.update(deliveries[cursor].measurement_m, deliveries[cursor].capture_time_s)
                cursor += 1
            if t < CFG.burn_in_s:
                continue
            state = tracker.predict_at(max(float(t), tracker.timestamp_s))
            shifted = np.asarray(state.position_m) - np.asarray(state.velocity_mps) * d
            truth_t = truth_position(motion, [t], CFG)[0]
            assert np.linalg.norm(shifted - truth_t) == pytest.approx(naive.errors_m[i], abs=1e-4)


def test_out_of_order_packets_are_counted_not_fused():
    """A capture timestamp no newer than an earlier delivery is dropped exactly once."""
    for seed in seed_list(5):
        deliveries = deliver("constant_velocity", delay_s=0.2, loss_probability=0,
                             noise_std_m=0.05, seed=seed, config=CFG, jitter_s=0.1)
        expected = sum(1 for i, p in enumerate(deliveries)
                       if i and p.capture_time_s <= max(q.capture_time_s for q in deliveries[:i]))
        arms = run_arms(deliveries, "constant_velocity", CFG)
        assert arms["recovery"].dropped_count == expected
        assert arms["recovery"].fused_count + arms["recovery"].dropped_count == len(deliveries)
        if expected:
            break
    else:
        pytest.fail("no out-of-order delivery in selected seeds")
    t = np.arange(4) * 0.1
    z = truth_position("constant_velocity", t, CFG)
    swapped = [Delivery(0, 0.0, 0.0, tuple(z[0])), Delivery(2, 0.2, 0.1, tuple(z[2])),
               Delivery(1, 0.1, 0.2, tuple(z[1]))]
    arms = run_arms(swapped, "constant_velocity", CFG)
    assert arms["recovery"].dropped_count == 1
    assert arms["naive"].fused_out_of_order_count == 1


def test_same_seed_is_deterministic():
    a = _run("constant_turn", 0.1, noise=0.05, loss=0.3, seed=17)
    b = _run("constant_turn", 0.1, noise=0.05, loss=0.3, seed=17)
    for arm in ("naive", "recovery"):
        for name in ("p95_error_m", "max_error_m", "rmse_m"):
            assert getattr(a[arm], name) == getattr(b[arm], name)
        assert np.array_equal(a[arm].errors_m, b[arm].errors_m)
    c = _run("constant_turn", 0.1, noise=0.05, loss=0.3, seed=18)
    assert not np.array_equal(a["recovery"].errors_m, c["recovery"].errors_m)


def test_common_random_numbers_between_cells():
    a = deliver("constant_velocity", delay_s=0.1, loss_probability=0.3, noise_std_m=1e-9, seed=8, config=CFG)
    b = deliver("constant_velocity", delay_s=0.1, loss_probability=0.3, noise_std_m=2e-9, seed=8, config=CFG)
    assert [p.seq for p in a] == [p.seq for p in b]
    c = deliver("constant_velocity", delay_s=0.1, loss_probability=0.0, noise_std_m=0.05, seed=8, config=CFG)
    d = deliver("constant_velocity", delay_s=0.1, loss_probability=1e-12, noise_std_m=0.05, seed=8, config=CFG)
    assert [(p.seq, p.measurement_m) for p in c] == [(p.seq, p.measurement_m) for p in d]


def test_burst_loss_long_run_fraction():
    from sim.link import LinkConfig, SimulatedLink
    p, mean = 0.3, 5.0
    e = 1 / mean
    s = p * e / (1 - p * (1 - e))
    lengths, lost = [], 0
    for seed in range(20):
        link = SimulatedLink(LinkConfig(loss_model="burst", loss_probability=p,
            burst_start_probability=s, burst_end_probability=e), seed=seed)
        for k in range(1000):
            link.send("a", "b", k, k * 0.1)
        events = link.events
        run = 0
        for ev in events:
            if ev.receive_time is None:
                lost += 1; run += 1
            elif run:
                lengths.append(run); run = 0
        if run: lengths.append(run)
    assert lost / (20 * 1000) == pytest.approx(p, abs=0.03)
    assert np.mean(lengths) == pytest.approx(mean, rel=0.15)


def test_statistics_helpers_independent():
    assert median_interval(range(1, 21)) == (10.5, 6, 15)
    assert median_interval(range(1, 51)) == (25.5, 18, 33)
    assert wilson_interval(25, 50) == pytest.approx((0.366, 0.634), abs=0.002)
    assert wilson_interval(0, 30)[1] == pytest.approx(0.114, abs=0.002)
    rng = np.random.default_rng(20261013)
    covered = 0
    for _ in range(2000):
        _, lo, hi = median_interval(rng.exponential(size=50))
        covered += lo <= math.log(2) <= hi
    # The order-statistic interval for n = 50 has exact coverage 96.7 %.
    assert 0.95 <= covered / 2000 <= 0.985


def test_failure_label_follows_criterion():
    def ar(p95, mx):
        return ArmResult(p95, mx, 0.0, 0, 0, 0, None, p95 > 0.25 or mx > 0.50,
                         np.array([0.0]), np.array([0.0]))
    assert summarize_arm([ar(0.3, 0), ar(0, 0.6), ar(0, 0)])["label"] == "fails"
    assert summarize_arm([ar(0.3, 0), ar(0, 0), ar(0, 0), ar(0, 0)])["label"] == "passes"
    # Exactly half failing counts as "fails" (pre-registered: at least half).
    assert summarize_arm([ar(0.3, 0), ar(0, 0.6), ar(0, 0), ar(0, 0)])["label"] == "fails"


def test_run_level_failure_flag_matches_hand_calculation():
    """Straight line, no noise or loss: naive p95 = max = v x d, so 0.20 m passes and 0.30 m fails."""
    assert _run("constant_velocity", 0.20)["naive"].failed is False
    assert _run("constant_velocity", 0.30)["naive"].failed is True
    assert _run("constant_velocity", 0.30)["recovery"].failed is False


def test_naive_skips_a_second_packet_at_the_same_receive_time():
    """Two packets landing within 1e-12 s of each other: the naive arm fuses the first and
    skips the second (counted), because it stamps both with the same arrival time."""
    times = np.arange(0.0, 12.0 + 1e-9, 0.1)
    truth = truth_position("constant_velocity", times, CFG)
    packets = [Delivery(i, float(t), float(t) + 0.2, tuple(truth[i])) for i, t in enumerate(times)]
    # Packet 51 arrives (within the tolerance) together with packet 50.
    packets[51] = Delivery(51, packets[51].capture_time_s, packets[50].receive_time_s + 5e-13, packets[51].measurement_m)
    packets.sort(key=lambda p: (p.receive_time_s, p.seq))
    arms = run_arms([p for p in packets if p.receive_time_s <= 12.0], "constant_velocity", CFG)
    assert arms["naive"].dropped_count == 1
    assert arms["recovery"].dropped_count == 0  # capture times are still distinct and in order


def test_nis_counts_fusions_from_the_end_of_warm_up():
    """Only fusions stamped at or after 2.0 s count toward the mean NIS.
    One packet at 1.98 s and one at exactly 2.0 s: the mean equals the NIS of the 2.0 s packet."""
    truth = truth_position("constant_velocity", [0.0, 1.98, 2.0], CFG)
    one = [Delivery(0, 0.0, 0.0, tuple(truth[0])), Delivery(1, 1.98, 1.98, tuple(truth[1] + 0.03))]
    two = one + [Delivery(2, 2.0, 2.0, tuple(truth[2] + 0.03))]
    assert run_arms(one, "constant_velocity", CFG)["recovery"].mean_nis is None
    # Hand rebuild: fuse the first two packets, then the NIS of the 2.0 s packet
    # is innovation^2 / innovation variance summed over the three axes.
    tracker = ConstantVelocityTracker(position_m=truth[0], acceleration_spectral_density=0.1, measurement_std_m=0.05)
    tracker.update(one[0].measurement_m, 0.0)
    tracker.update(one[1].measurement_m, 1.98)
    predicted = tracker.predict_at(2.0)
    innovation = np.asarray(two[2].measurement_m) - np.asarray(predicted.position_m)
    variance = np.diag(np.asarray(predicted.covariance))[:3] + 0.05 ** 2
    expected = float(np.sum(innovation ** 2 / variance))
    assert run_arms(two, "constant_velocity", CFG)["recovery"].mean_nis == pytest.approx(expected, rel=1e-9)

