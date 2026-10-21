"""Tests for the opt-in NEES divergence wiring in sim.coop.observers (proposal).

These were written before the wiring. The hand cases do not use the code under
test to make their expected numbers.
"""
import numpy as np
import pytest

from sim.coop.observers import (
    CoopCell,
    CoopOutcome,
    _nees_fast,
    apply_divergence_rule,
    run_seed,
)
from sim.estimation.divergence import DivergenceRule, nees_values


def _outcome(window_means, failed=False):
    return CoopOutcome(0.1, 0.2, 0.05, failed, 3.0, tuple(window_means), 0.0, 7,
                       {"dropped_stale": 0})


# 1. Hand-computable NEES.

def test_nees_by_hand_diagonal():
    # Error 0.2 m, 0.1 m, 0 m; claimed sigma 0.1 m, 0.05 m, 0.1 m.
    # Each axis in "claimed sigmas": 2, 2, 0. Squares add: 4 + 4 + 0 = 8.
    e = np.array([0.2, 0.1, 0.0])
    p = np.diag([0.1 ** 2, 0.05 ** 2, 0.1 ** 2])
    assert _nees_fast(e, p) == pytest.approx(8.0)
    assert nees_values([e], [p])[0] == pytest.approx(8.0)


def test_nees_by_hand_correlated():
    # P = [[2, 1, 0], [1, 2, 0], [0, 0, 1]], e = (1, 1, 0).
    # Solve [[2, 1], [1, 2]] x = (1, 1): x = (1/3, 1/3). NEES = e . x = 2/3.
    # The error points along the long axis of the error ellipse (the filter
    # expects x and y errors to move together), so it costs less than with
    # P = diag(2, 2, 1), where the same error gives 1/2 + 1/2 = 1.
    e = np.array([1.0, 1.0, 0.0])
    p = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    assert _nees_fast(e, p) == pytest.approx(2.0 / 3.0)


def test_nees_values_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        nees_values([np.zeros(3)] * 3, [np.eye(3)] * 2)


# 2. Hand-built window means through the outcome step.

def test_diverged_counts_as_failure_only_when_rule_given():
    # Window means chosen by hand: windows 2, 3, 4 are above 100, so divergence
    # starts at window 2; windows 1 to 4 are above 5.97, so the warning starts at 1.
    means = [2.0, 7.0, 150.0, 200.0, 300.0, 3.0]
    out = apply_divergence_rule(_outcome(means), DivergenceRule())
    check = out.extras["nees_check"]
    assert check["diverged"] is True and check["diverged_first_window"] == 2
    assert check["inconsistent"] is True and check["inconsistent_first_window"] == 1
    assert out.failed is True
    # Everything else is carried over unchanged.
    assert (out.p95_error_m, out.max_error_m, out.rmse_m, out.mean_nees,
            out.nees_window_means, out.fused_count) == (0.1, 0.2, 0.05, 3.0, tuple(means), 7)


def test_inconsistent_alone_is_only_a_warning():
    # Three windows in a row above 5.97 but none above 100.
    out = apply_divergence_rule(_outcome([7.0, 8.0, 9.0, 50.0]), DivergenceRule())
    assert out.extras["nees_check"]["inconsistent"] is True
    assert out.extras["nees_check"]["diverged"] is False
    assert out.failed is False


def test_two_high_windows_are_not_enough():
    out = apply_divergence_rule(_outcome([500.0, 500.0, 1.0, 500.0, 500.0]), DivergenceRule())
    assert out.extras["nees_check"]["diverged"] is False
    assert out.failed is False


def test_rule_never_clears_an_existing_failure():
    out = apply_divergence_rule(_outcome([1.0, 1.0, 1.0], failed=True), DivergenceRule())
    assert out.extras["nees_check"]["diverged"] is False
    assert out.failed is True


def test_input_outcome_is_not_mutated():
    original = _outcome([150.0, 150.0, 150.0])
    apply_divergence_rule(original, DivergenceRule())
    assert original.failed is False
    assert "nees_check" not in original.extras


# 3. Real simulator, default off.

def test_default_is_off_and_identical():
    cell = CoopCell(target="ct", n_observers=2, noise_std_m=0.05)
    plain = run_seed(cell, 11, arms=("single", "peer_ci"))
    explicit_none = run_seed(cell, 11, arms=("single", "peer_ci"), divergence_rule=None)
    for arm in ("single", "peer_ci"):
        a, b = plain[arm], explicit_none[arm]
        assert "nees_check" not in a.extras
        assert (a.p95_error_m, a.max_error_m, a.rmse_m, a.failed, a.mean_nees,
                a.nees_window_means, a.frac_windows_above_band, a.fused_count) == \
               (b.p95_error_m, b.max_error_m, b.rmse_m, b.failed, b.mean_nees,
                b.nees_window_means, b.frac_windows_above_band, b.fused_count)


# 4. Limiting case: the filter's model matches the world.

def test_matched_filter_is_not_flagged():
    # Straight-line target, sensor noise equal to what the filter assumes, one
    # observer, no loss. This is the textbook case where the filter is honest.
    cell = CoopCell(target="cv", n_observers=1, noise_std_m=0.05)
    rule = DivergenceRule()
    for seed in range(1, 11):
        out = run_seed(cell, seed, arms=("single",), divergence_rule=rule)["single"]
        assert out.extras["nees_check"]["diverged"] is False
        assert out.failed is False
        assert out.mean_nees < 5.97


# 5. Deliberately diverging (overconfident) filter.

def test_overconfident_filter_is_caught_while_the_error_rule_passes():
    # The filter is told its sensor is good to 2 mm; the real noise is 5 cm.
    # Its errors are small (old rule passes) but about 25 times bigger than it
    # claims, so window-mean NEES sits far above 100.
    cell = CoopCell(target="cv", n_observers=1, noise_std_m=0.05, r_std_m=0.002)
    off = run_seed(cell, 1, arms=("single",))["single"]
    on = run_seed(cell, 1, arms=("single",), divergence_rule=DivergenceRule())["single"]
    assert off.failed is False
    assert np.isfinite(on.p95_error_m) and on.p95_error_m < 0.25
    assert on.extras["nees_check"]["diverged"] is True
    assert on.failed is True
    assert on.p95_error_m == off.p95_error_m


def test_run_cell_passes_the_rule_through():
    from sim.coop.observers import run_cell
    cell = CoopCell(target="cv", n_observers=1, noise_std_m=0.05, r_std_m=0.002)
    res = run_cell(cell, [1, 2], arms=("single",), divergence_rule=DivergenceRule())
    assert all(o.extras["nees_check"]["diverged"] for o in res["single"])


# 6. EXP-023 runner flag (tools/d_run_fusion.py).

def _fusion_job(**extra):
    return {"n_observers": 1, "noise_std_m": 0.05, "target_label": "straight",
            "target": "cv", "target_params": (), "loss_probability": 0.0,
            "seed": 1, "ci": "fast", "arms": ("single",), **extra}


def test_runner_rows_unchanged_without_the_flag():
    from tools import d_run_fusion as runner
    row = runner._job(_fusion_job())["rows"][0]
    assert not set(runner.NEES_COLUMNS) & set(row)
    assert set(row) == set(runner.ROW_COLUMNS) | {"nonfinite_metric_count"}


def test_runner_flag_adds_columns_and_counts_divergence_as_failure(monkeypatch):
    from tools import d_run_fusion as runner
    # Swap in an overconfident filter through the cell builder: told 5 mm, real
    # noise 5 cm. The runner's 50 ms link delay adds lag error, so a 2 mm filter
    # would already fail the old error rule here; 5 mm keeps p95 near 0.22 m.
    original = runner._cell
    monkeypatch.setattr(runner, "_cell", lambda job: CoopCell(
        **{**original(job).__dict__, "r_std_m": 0.005}))
    off = runner._job(_fusion_job())["rows"][0]
    on = runner._job(_fusion_job(nees_divergence=True))["rows"][0]
    assert off["failed"] is False
    assert on["nees_diverged"] is True and on["failed"] is True
    assert on["p95_error_m"] == off["p95_error_m"]  # metrics are kept, not blanked
    summary = runner._summarize([on])[0]
    assert summary["nees_diverged_count"] == 1
    assert "nees_diverged_count" not in runner._summarize([off])[0]
