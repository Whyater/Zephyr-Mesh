"""Independent checks for the simulated cooperative observer engine."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from sim import s6_recovery as s6
from sim.coop import observers as observer_module
from sim.coop.fusion import covariance_intersection_fast, information_fusion
from sim.coop.net import Delivery, LinkSpec, TrafficSpec, make_deliveries
from sim.estimation.targets import make_target


CoopCell = observer_module.CoopCell
run_cell = observer_module.run_cell
run_seed = observer_module.run_seed
gap_statistics = observer_module.gap_statistics
ARMS = observer_module.ARMS


def _cell(
    target: str = "cv",
    *,
    n_observers: int = 1,
    noise: float = 0.05,
    delay: float = 0.0,
    loss: float = 0.0,
    phase: str = "random",
    r_std: float | None = None,
    ci: str = "fast",
) -> CoopCell:
    if target == "cv":
        params = (("speed_mps", 1.0),)
    elif target == "ct":
        params = (("speed_mps", 1.0), ("radius_m", 1.0))
    else:
        params = ()
    return CoopCell(
        target=target,
        target_params=params,
        n_observers=n_observers,
        rate_hz=10.0,
        noise_std_m=noise,
        r_std_m=r_std,
        link=LinkSpec(mode="abstract", delay_s=delay, loss_probability=loss),
        phase=phase,
        ci=ci,
    )


def test_targets_match_s6_truth_for_limiting_motions():
    """The cv and ct target classes reproduce the S6 position formulas."""
    cfg = s6.RecoveryConfig()
    times = np.arange(0.0, cfg.duration_s + 0.02 / 2.0, 0.02)
    for name, motion in (("cv", "constant_velocity"), ("ct", "constant_turn")):
        target = make_target(name, dict(_cell(name).target_params), seed=123)
        expected = s6.truth_position(motion, times, cfg)
        np.testing.assert_allclose(target.position(times), expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("target,motion", (("cv", "constant_velocity"), ("ct", "constant_turn")))
@pytest.mark.parametrize("noise", (0.0, 0.05))
@pytest.mark.parametrize("loss", (0.0, 0.3))
@pytest.mark.parametrize("delay", (0.0, 0.1))
def test_one_observer_single_and_central_match_s6(
    target: str, motion: str, noise: float, loss: float, delay: float
):
    """The one-observer limit is byte-for-byte the S6 recovery experiment."""
    cell = _cell(target, noise=noise, delay=delay, loss=loss)
    cfg = s6.RecoveryConfig()
    for seed in s6.seed_list(3):
        got = run_seed(cell, int(seed), arms=("single", "central"), config=cfg)
        reference_deliveries = s6.deliver(
            motion,
            delay_s=delay,
            loss_probability=loss,
            noise_std_m=noise,
            seed=int(seed),
            config=cfg,
        )
        expected = s6.run_arms(reference_deliveries, motion, cfg)["recovery"]
        for arm in ("single", "central"):
            result = got[arm]
            assert result.p95_error_m == pytest.approx(expected.p95_error_m, abs=1e-9)
            assert result.max_error_m == pytest.approx(expected.max_error_m, abs=1e-9)
            assert result.rmse_m == pytest.approx(expected.rmse_m, abs=1e-9)
            assert result.failed is expected.failed
        assert got["central"].p95_error_m == pytest.approx(got["single"].p95_error_m, abs=1e-12)
        assert got["central"].max_error_m == pytest.approx(got["single"].max_error_m, abs=1e-12)
        assert got["central"].rmse_m == pytest.approx(got["single"].rmse_m, abs=1e-12)


def test_zero_noise_turn_floor():
    """The initial zero-velocity state leaves the known constant-turn floor."""
    result = run_seed(_cell("ct", noise=0.0), s6.seed_list(1)[0], arms=("single",))["single"]
    assert result.p95_error_m == pytest.approx(0.0612927, abs=1e-6)
    assert result.max_error_m == pytest.approx(0.0613311, abs=1e-6)


def test_central_variance_decreases_with_observer_count():
    """Independent sensor noise gives an approximately one-over-N variance law."""
    seeds = s6.seed_list(40)
    means = {}
    for n_observers in (1, 2, 4):
        values = [
            run_seed(
                _cell("cv", n_observers=n_observers, noise=0.05),
                int(seed),
                arms=("central",),
            )["central"].rmse_m
            for seed in seeds
        ]
        means[n_observers] = float(np.mean(np.square(values)))
    assert means[1] > means[2] > means[4]
    ratio = means[1] / means[4]
    assert 2.5 <= ratio <= 5.5


def test_consistency_of_central_and_peer_arms():
    """CI stays conservative while naive feedback becomes overconfident."""
    seeds = s6.seed_list(5)
    outcomes = [
        run_seed(
            _cell("cv", n_observers=3, noise=0.05),
            int(seed),
            arms=("central", "peer_ci", "peer_naive"),
        )
        for seed in seeds
    ]
    central = [row["central"] for row in outcomes]
    ci = [row["peer_ci"] for row in outcomes]
    naive = [row["peer_naive"] for row in outcomes]
    assert np.mean([row.frac_windows_above_band for row in central]) <= 0.10
    assert np.mean([row.frac_windows_above_band for row in ci]) <= 0.05
    central_nees = float(np.mean([row.mean_nees for row in central]))
    naive_nees = float(np.mean([row.mean_nees for row in naive]))
    # On this five-seed run the measured means are 2.19 (central) and 5.24
    # (peer_naive), a ratio of about 2.39.  The strict inequality checks the
    # contracted feedback overconfidence without tuning the result.
    assert naive_nees > central_nees


def test_loss_can_be_recovered_by_more_observers():
    """At high loss, four independent observers improve the central p95 error."""
    seeds = s6.seed_list(20)
    values = {}
    for n_observers in (1, 4):
        values[n_observers] = [
            run_seed(
                _cell("ct", n_observers=n_observers, noise=0.05, delay=0.1, loss=0.5),
                int(seed),
                arms=("central",),
            )["central"].p95_error_m
            for seed in seeds
        ]
    assert float(np.median(values[4])) < float(np.median(values[1]))


def test_common_random_numbers_reuse_delivery_table(monkeypatch):
    """Single and central are evaluated against one shared star delivery table."""
    calls = []
    real_make_deliveries = observer_module.make_deliveries

    def spy(traffic, link, seed):
        table = real_make_deliveries(traffic, link, seed)
        calls.append((traffic.pattern, traffic.message_bytes, table))
        return table

    monkeypatch.setattr(observer_module, "make_deliveries", spy)
    run_seed(_cell("cv", n_observers=2), int(s6.seed_list(1)[0]), arms=("single", "central"))
    star_tables = [table for pattern, size, table in calls if (pattern, size) == ("star", 32)]
    assert len(star_tables) == 1


def test_run_cell_shape_and_determinism():
    """A repeated seed produces identical scalar outcomes and run_cell preserves order."""
    cell = _cell("cv", n_observers=2)
    seeds = s6.seed_list(2)
    first = run_cell(cell, seeds, arms=("single",))
    second = run_cell(cell, seeds, arms=("single",))
    assert list(first) == ["single"]
    assert len(first["single"]) == len(seeds)
    for left, right in zip(first["single"], second["single"]):
        for field in fields(left):
            if field.name == "extras":
                continue
            assert getattr(left, field.name) == getattr(right, field.name)


def test_gap_statistics_counts_steps_without_a_received_message():
    """A short hand-built table checks the per-step gap calculation."""
    table = {
        ("o1", "gs"): (
            Delivery("o1", "gs", 0, 0.0, 0.0, None),
            Delivery("o1", "gs", 1, 0.1, None, "link"),
            Delivery("o1", "gs", 2, 0.2, 0.25, None),
            Delivery("o1", "gs", 3, 0.3, None, "link"),
        )
    }
    gaps = gap_statistics(table, ("o1",), receiver="gs")
    assert tuple(gaps) == pytest.approx((0.1,))
    assert gaps["total"] == 2
    assert gaps["leading"] == 0



def test_all_arms_one_observer_match_single_at_zero_link():
    """All one-observer ground-station arms reduce to the S6 filter."""
    cell = _cell("cv", n_observers=1, noise=0.05, delay=0.0, loss=0.0, phase="random")
    result = run_seed(cell, int(s6.seed_list(1)[0]))
    base = result["single"]
    for arm in ("central", "star_avg", "peer_ci", "peer_naive"):
        assert result[arm].p95_error_m == pytest.approx(base.p95_error_m, abs=1e-12)
        assert result[arm].max_error_m == pytest.approx(base.max_error_m, abs=1e-12)
        assert result[arm].rmse_m == pytest.approx(base.rmse_m, abs=1e-12)


def test_random_phases_are_shared_with_traffic(monkeypatch):
    """The capture schedule and every traffic spec use one deterministic phase tuple."""
    seen = []
    real_make = observer_module.make_deliveries

    def spy(traffic, link, seed):
        seen.append(tuple(traffic.phase_offsets_s))
        return real_make(traffic, link, seed)

    monkeypatch.setattr(observer_module, "make_deliveries", spy)
    cell = _cell("cv", n_observers=3, phase="random", noise=0.0)
    run_seed(cell, int(s6.seed_list(1)[0]), arms=("single", "star_avg", "peer_ci"))
    assert len(seen) == 3
    assert all(phases == seen[0] for phases in seen)
    assert seen[0][0] == 0.0


def test_ci_fast_and_exact_are_close_and_deterministic():
    """Both CI modes remain conservative and produce nearby p95 errors."""
    seed = int(s6.seed_list(1)[0])
    fast = run_seed(_cell("cv", n_observers=3, phase="random", ci="fast"), seed, arms=("central", "peer_ci"))
    exact = run_seed(_cell("cv", n_observers=3, phase="random", ci="exact"), seed, arms=("central", "peer_ci"))
    fast_ci = fast["peer_ci"]
    exact_ci = exact["peer_ci"]
    ratio = abs(fast_ci.p95_error_m - exact_ci.p95_error_m) / max(exact_ci.p95_error_m, 1e-12)
    assert ratio < 0.20
    assert fast_ci.frac_windows_above_band <= fast["central"].frac_windows_above_band
    assert exact_ci.frac_windows_above_band <= exact["central"].frac_windows_above_band
    assert run_seed(_cell("cv", n_observers=3, phase="random", ci="fast"), seed, arms=("peer_ci",))["peer_ci"].p95_error_m == pytest.approx(fast_ci.p95_error_m, abs=1e-12)


def test_central_jitter_stale_count_matches_capture_stamp_walk():
    """Jitter drops exactly the deliveries older than the last fused capture."""
    cell = _cell("cv", n_observers=3, phase="random", delay=0.05, loss=0.3)
    cell = CoopCell(target=cell.target, target_params=cell.target_params,
                    n_observers=cell.n_observers, rate_hz=cell.rate_hz,
                    noise_std_m=cell.noise_std_m, r_std_m=cell.r_std_m,
                    link=LinkSpec(mode="abstract", delay_s=0.05, loss_probability=0.3, jitter_s=0.08),
                    phase=cell.phase, ci=cell.ci)
    seed = int(s6.seed_list(1)[0])
    result = run_seed(cell, seed, arms=("central",))["central"]
    table = result.extras["delivery_table"]
    rows = []
    for sender in ("o1", "o2", "o3"):
        for item in table[(sender, "gs")]:
            if item.receive_time_s is not None:
                rows.append((item.receive_time_s, item.send_time_s, int(sender[1:]), item.seq))
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    last = -np.inf
    dropped = 0
    for _, stamp, _, _ in rows:
        if stamp < last - 1e-12:
            dropped += 1
        else:
            last = max(last, stamp)
    assert result.extras["dropped_stale"] == dropped


def test_staggered_zero_noise_peer_ci_is_exactly_on_the_target():
    """A capture-stamped peer CI track has no phase-induced position bias."""
    cell = _cell("cv", n_observers=3, noise=0.0, r_std=0.0, phase="staggered")
    result = run_seed(cell, int(s6.seed_list(1)[0]), arms=("peer_ci",))["peer_ci"]
    assert result.p95_error_m < 1e-9


def test_central_matches_independent_numpy_batch_reference():
    """Central scoring agrees with an independently written sequential KF."""
    cell = _cell("cv", n_observers=3, noise=0.05, phase="staggered", loss=0.3, delay=0.1)
    seed = int(s6.seed_list(1)[0])
    cfg = s6.RecoveryConfig()
    names = tuple(f"o{i + 1}" for i in range(cell.n_observers))
    phases = observer_module._phase_offsets(cell, names, seed)
    target, names, captures, measurements = observer_module._truth_and_measurements(cell, seed, cfg, phases)
    traffic = TrafficSpec("star", names, cell.rate_hz, 32, cfg.duration_s, phases)
    table = make_deliveries(traffic, cell.link, seed)
    rows = []
    for sender in names:
        for item in table[(sender, "gs")]:
            if item.receive_time_s is not None:
                rows.append((float(item.receive_time_s), float(captures[sender][item.seq]),
                             int(sender[1:]), item.seq, measurements[sender][item.seq]))
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    settings = observer_module.EstimatorSettings()
    x = np.r_[target.start_position, np.zeros(3)]
    P = np.diag([settings.initial_position_variance_m2] * 3 +
                [settings.initial_velocity_variance_m2ps2] * 3)
    H = np.hstack((np.eye(3), np.zeros((3, 3))))
    R = np.eye(3) * settings.measurement_std_m ** 2
    last = 0.0
    cursor = 0
    newest = -np.inf
    queries = observer_module._query_times(cfg)
    truth = np.asarray(target.position(queries), dtype=float)
    errors = []
    for t, tr in zip(queries, truth):
        while cursor < len(rows) and rows[cursor][0] <= t + 1e-12:
            _, stamp, _, _, z = rows[cursor]
            cursor += 1
            if stamp < newest - 1e-12:
                continue
            dt = stamp - last
            I = np.eye(3)
            F = np.block([[I, dt * I], [np.zeros((3, 3)), I]])
            Q = settings.q_acc_psd * np.block([[dt**3 / 3 * I, dt**2 / 2 * I],
                                                [dt**2 / 2 * I, dt * I]])
            x = F @ x
            P = F @ P @ F.T + Q
            S = H @ P @ H.T + R
            K = np.linalg.solve(S, (P @ H.T).T).T
            x = x + K @ (z - H @ x)
            P = (np.eye(6) - K @ H) @ P
            P = (P + P.T) / 2.0
            last = stamp
            newest = max(newest, stamp)
        dt = max(0.0, float(t) - last)
        errors.append(np.linalg.norm(x[:3] + dt * x[3:] - tr))
    expected = run_seed(cell, seed, arms=("central",))["central"]
    scored = np.asarray(errors)[queries >= cfg.burn_in_s - 1e-9]
    assert float(np.percentile(scored, 95)) == pytest.approx(expected.p95_error_m, abs=1e-12)
    assert float(np.max(scored)) == pytest.approx(expected.max_error_m, abs=1e-12)
    assert float(np.sqrt(np.mean(scored ** 2))) == pytest.approx(expected.rmse_m, abs=1e-12)


def test_determinism_all_five_arms_including_extras():
    cell = _cell("cv", n_observers=3, phase="random", delay=0.05, loss=0.3)
    first = run_seed(cell, int(s6.seed_list(1)[0]))
    second = run_seed(cell, int(s6.seed_list(1)[0]))
    for arm in ARMS:
        left, right = first[arm], second[arm]
        for name in ("p95_error_m", "max_error_m", "rmse_m", "failed", "mean_nees",
                     "nees_window_means", "frac_windows_above_band", "fused_count"):
            assert getattr(left, name) == getattr(right, name)
        assert left.extras == right.extras


def test_fast_exact_ci_over_ten_seeds():
    cell_fast = _cell("cv", n_observers=3, phase="random", ci="fast")
    cell_exact = _cell("cv", n_observers=3, phase="random", ci="exact")
    diffs = []
    for seed in s6.seed_list(10):
        fast = run_seed(cell_fast, int(seed), arms=("central", "peer_ci"))
        exact = run_seed(cell_exact, int(seed), arms=("central", "peer_ci"))
        diffs.append(abs(fast["peer_ci"].p95_error_m - exact["peer_ci"].p95_error_m) /
                     max(exact["peer_ci"].p95_error_m, 1e-12))
        assert fast["peer_ci"].frac_windows_above_band <= fast["central"].frac_windows_above_band
        assert exact["peer_ci"].frac_windows_above_band <= exact["central"].frac_windows_above_band
    assert float(np.median(diffs)) < 0.20
    assert float(np.max(diffs)) < 0.20


def test_batched_fast_ci_matches_fusion_reference():
    """The event-path fast CI algebra matches the shared fusion helper."""
    rng = np.random.default_rng(90210)
    means = [rng.normal(size=6) for _ in range(4)]
    covs = []
    for _ in means:
        a = rng.normal(size=(6, 6))
        covs.append(a @ a.T + np.eye(6))
    expected = covariance_intersection_fast(means, covs)[:2]
    got = observer_module._fuse_tracks(means, covs, mode="ci", ci_mode="fast")
    np.testing.assert_allclose(got[0], expected[0], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(got[1], expected[1], rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("mode", ("ci", "naive"))
@pytest.mark.parametrize("n_observers", (2, 3, 4, 5, 6))
def test_batched_fusion_matches_shared_helpers_for_multiple_sizes(mode, n_observers):
    """Batched CI and information fusion agree with the shared helper formulas."""
    rng = np.random.default_rng(1200 + n_observers)
    for _ in range(4):
        means = [rng.normal(size=6) for _ in range(n_observers)]
        covs = []
        for _ in means:
            matrix = rng.normal(size=(6, 6))
            covs.append(matrix @ matrix.T + 2.0 * np.eye(6))
        if mode == "ci":
            expected = covariance_intersection_fast(means, covs)[:2]
        else:
            expected = information_fusion(means, covs)
        got = observer_module._fuse_tracks(means, covs, mode=mode, ci_mode="fast")
        np.testing.assert_allclose(got[0], expected[0], rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(got[1], expected[1], rtol=0.0, atol=1e-9)


def test_prediction_matrix_cache_is_bounded():
    """Distinct prediction steps cannot grow the transition cache without bound."""
    cache = observer_module._predict_matrices
    cache.cache_clear()
    mean = np.zeros(6)
    covariance = np.eye(6)
    for index in range(1200):
        observer_module._predict_full(mean, covariance, 0.0, 0.123 + index * 1e-4, 0.1)
    info = cache.cache_info()
    assert info.maxsize == 256
    assert info.currsize <= 256


@pytest.mark.parametrize("capture_index", (1, 21))
@pytest.mark.parametrize("arm", ("peer_ci", "peer_naive"))
def test_peer_divergence_before_and_after_burn_in_is_reported(monkeypatch, capture_index, arm):
    """A non-finite measurement marks divergence without crashing either peer arm."""
    cell = _cell("cv", n_observers=2, phase="sync")
    real_truth_and_measurements = observer_module._truth_and_measurements

    def inject_nonfinite(cell, seed, config, phases):
        target, names, captures, measurements = real_truth_and_measurements(
            cell, seed, config, phases
        )
        measurements["o1"][capture_index] = np.full(3, np.nan)
        return target, names, captures, measurements

    monkeypatch.setattr(observer_module, "_truth_and_measurements", inject_nonfinite)
    row = run_seed(cell, int(s6.seed_list(1)[0]), arms=(arm,))[arm]
    assert row.failed is True
    assert row.extras["diverged"] is True
    assert np.isfinite(row.extras["diverged_time_s"])
    if capture_index == 1:
        assert row.extras["diverged_time_s"] < 2.0
        assert row.extras["onboard_o1_p95_m"] is None
        assert row.extras["onboard_o1_mean_nees"] is None
    else:
        assert row.extras["diverged_time_s"] >= 2.0
        assert row.extras["onboard_o1_p95_m"] is not None
        assert row.extras["onboard_o1_mean_nees"] is not None


def test_exact_ci_optimiser_failure_is_reported_as_divergence(monkeypatch):
    """A ValueError from the exact CI optimiser marks divergence instead of escaping."""
    cell = _cell("cv", n_observers=2, phase="sync", ci="exact")

    def failing_ci(means, covariances):
        raise ValueError("optimiser did not converge")

    monkeypatch.setattr(observer_module, "covariance_intersection_many", failing_ci)
    row = run_seed(cell, int(s6.seed_list(1)[0]), arms=("peer_ci",))["peer_ci"]
    assert row.failed is True
    assert row.extras["diverged"] is True
    assert np.isfinite(row.extras["diverged_time_s"])


def test_cached_cv_prediction_matches_independent_block_prediction():
    """Cached transition matrices preserve the old block prediction to 1e-9."""
    rng = np.random.default_rng(771)
    mean = rng.normal(size=6)
    a = rng.normal(size=(6, 6))
    cov = a @ a.T + np.eye(6)
    dt = 0.37
    q = 0.1
    I = np.eye(3)
    F = np.block([[I, dt * I], [np.zeros((3, 3)), I]])
    Q = q * np.block([[dt ** 3 / 3.0 * I, dt ** 2 / 2.0 * I],
                      [dt ** 2 / 2.0 * I, dt * I]])
    expected_mean = F @ mean
    expected_cov = F @ cov @ F.T + Q
    got_mean, got_cov = observer_module._predict_full(mean, cov, 0.0, dt, q)
    np.testing.assert_allclose(got_mean, expected_mean, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(got_cov, (expected_cov + expected_cov.T) / 2.0,
                               rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("n_observers", (1, 3))
def test_zero_measurement_noise_all_arms_have_finite_covariance(n_observers):
    cell = _cell("cv", n_observers=n_observers, noise=0.0, r_std=0.0, phase="sync")
    result = run_seed(cell, int(s6.seed_list(1)[0]))
    for arm in ARMS:
        row = result[arm]
        assert np.isfinite(row.p95_error_m)
        assert np.isfinite(row.rmse_m)
        assert not row.extras.get("diverged", False)


@pytest.mark.parametrize("target", ("cv", "ct"))
def test_peer_naive_feedback_n6_is_unhidden(target):
    params = (("speed_mps", 1.0),) if target == "cv" else (("speed_mps", 1.0), ("radius_m", 1.0))
    cell = CoopCell(target=target, target_params=params, n_observers=6, noise_std_m=0.05,
                    link=LinkSpec(mode="abstract", delay_s=0.05, loss_probability=0.3), phase="sync")
    row = run_seed(cell, int(s6.seed_list(1)[0]), arms=("central", "peer_naive"))["peer_naive"]
    assert row.extras.get("diverged", False) or row.mean_nees > 10.0


def test_star_average_cached_state_matches_per_query_fusion():
    """The star-average cached prediction agrees with recomputing each query."""
    cell = _cell("cv", n_observers=3, phase="random", delay=0.05, loss=0.3)
    seed = int(s6.seed_list(1)[0])
    config = s6.RecoveryConfig()
    names = tuple(f"o{i + 1}" for i in range(cell.n_observers))
    phases = observer_module._phase_offsets(cell, names, seed)
    target, names, captures, measurements = observer_module._truth_and_measurements(
        cell, seed, config, phases
    )
    traffic = TrafficSpec("star", names, cell.rate_hz, 108, config.duration_s, phases)
    table = make_deliveries(traffic, cell.link, seed)
    settings = observer_module.EstimatorSettings()
    snapshots = {}
    for sender in names:
        estimator = observer_module._new_estimator(target, settings, cell.r_std_m)
        sender_rows = []
        for index, capture_time in enumerate(captures[sender]):
            estimator.update(measurements[sender][index], float(capture_time))
            mean, covariance = observer_module._state_from_estimator(
                estimator, float(capture_time), settings
            )
            sender_rows.append((mean.copy(), covariance.copy()))
        snapshots[sender] = sender_rows

    arrivals = []
    for sender in names:
        for delivery in observer_module._route(table, sender, "gs"):
            arrivals.append((float(delivery.receive_time_s), int(delivery.seq), sender))
    arrivals.sort(key=lambda item: (item[0], item[1], int(item[2][1:])))
    latest = {}
    cursor = 0
    initial_mean, initial_covariance, _ = observer_module._initial_state(target, settings)
    query_times = observer_module._query_times(config)
    truth = np.asarray(target.position(query_times), dtype=float)
    errors = []
    for query_time in query_times:
        while cursor < len(arrivals) and arrivals[cursor][0] <= query_time + 1e-12:
            _, sequence, sender = arrivals[cursor]
            cursor += 1
            stamp = float(captures[sender][sequence])
            latest[sender] = (stamp, *snapshots[sender][sequence])
        means = []
        covariances = []
        for sender in names:
            if sender in latest:
                stamp, mean0, covariance0 = latest[sender]
                mean, covariance = observer_module._predict_full(
                    mean0, covariance0, stamp, float(query_time), settings.q_acc_psd
                )
            else:
                mean, covariance = observer_module._predict_full(
                    initial_mean, initial_covariance, 0.0, float(query_time), settings.q_acc_psd
                )
            means.append(mean)
            covariances.append(covariance)
        mean, _ = observer_module._naive_average_fast(means, covariances)
        errors.append(float(np.linalg.norm(mean[:3] - truth[len(errors)])))

    scored = np.asarray(errors)[query_times >= config.burn_in_s - 1e-9]
    expected = run_seed(cell, seed, arms=("star_avg",))["star_avg"]
    assert float(np.percentile(scored, 95)) == pytest.approx(expected.p95_error_m, abs=1e-12)
    assert float(np.max(scored)) == pytest.approx(expected.max_error_m, abs=1e-12)
    assert float(np.sqrt(np.mean(scored ** 2))) == pytest.approx(expected.rmse_m, abs=1e-12)
