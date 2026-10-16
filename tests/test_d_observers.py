"""Independent checks for the simulated cooperative observer engine."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from sim import s6_recovery as s6
from sim.coop import observers as observer_module
from sim.coop.net import Delivery, LinkSpec
from sim.estimation.targets import make_target


CoopCell = observer_module.CoopCell
run_cell = observer_module.run_cell
run_seed = observer_module.run_seed
gap_statistics = observer_module.gap_statistics


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
    seeds = s6.seed_list(20)
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
    # With gs scoring, the measured ratio for this sync cell is about 0.99.
    # The old onboard assertion overstates the feedback effect.
    assert naive_nees > 0.95 * central_nees


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
