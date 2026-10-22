"""Tests for the opt-in windowed NEES divergence check (proposal)."""
import numpy as np
import pytest
from scipy.stats import binom, chi2

from sim.estimation.divergence import DivergenceRule, check_nees_divergence, nees_values


def _healthy_errors(rng, steps):
    """Errors drawn from exactly the covariance the filter claims."""
    a = rng.normal(size=(3, 3))
    p = a @ a.T + 0.5 * np.eye(3)
    errors = rng.multivariate_normal(np.zeros(3), p, size=steps)
    return errors, [p] * steps


def test_threshold_matches_chi_square_table():
    # chi-square(30) upper 0.1 % point is 59.703 (standard table), so 5.9703 per query.
    assert DivergenceRule().threshold == pytest.approx(5.9703, abs=1e-4)


def test_hand_built_linear_drift_is_caught_while_finite():
    # Error grows 1 cm per query, claimed sigma stays 5 cm: NEES_k = (0.01 k)^2 / 0.05^2 = 0.04 k^2.
    # Window w (k = 10w .. 10w+9) mean = 0.04 * (100 w^2 + 90 w + 28.5):
    # w=0: 1.14, w=1: 8.74, w=2: 24.34, w=4: 79.54, w=5: 119.14, w=6: 166.74, w=7: 222.34.
    k = np.arange(100)
    errors = np.stack([0.01 * k, np.zeros(100), np.zeros(100)], axis=1)
    covs = [np.eye(3) * 0.05 ** 2] * 100
    nees = nees_values(errors, covs)
    assert np.all(np.isfinite(nees))
    assert nees[10] == pytest.approx(4.0)
    result = check_nees_divergence(nees)
    for w, hand in [(0, 1.14), (1, 8.74), (2, 24.34), (4, 79.54), (5, 119.14), (7, 222.34)]:
        assert result.window_means[w] == pytest.approx(hand)
    # Warning from window 1 (first of 1, 2, 3 above 5.97); divergence from window 5 (5, 6, 7 above 100).
    assert result.inconsistent and result.inconsistent_first_window == 1
    assert result.diverged and result.first_window == 5


def test_overconfident_but_bounded_filter_warns_without_diverging():
    # Error steady at 3 claimed sigmas on every axis: NEES = 27 every query.
    nees = np.full(100, 27.0)
    result = check_nees_divergence(nees)
    assert result.inconsistent and not result.diverged


def test_healthy_filter_is_not_flagged():
    rng = np.random.default_rng(20261020)
    errors, covs = _healthy_errors(rng, 600)
    nees = nees_values(errors, covs)
    assert 2.6 < float(np.mean(nees)) < 3.4
    result = check_nees_divergence(nees)
    assert not result.inconsistent and not result.diverged


def test_single_spike_does_not_trigger():
    rng = np.random.default_rng(7)
    errors, covs = _healthy_errors(rng, 600)
    nees = nees_values(errors, covs)
    nees[305] = 1e6  # one wild query, two windows either side stay healthy
    result = check_nees_divergence(nees)
    assert result.window_means[30] > result.gross_nees
    assert not result.inconsistent and not result.diverged


def test_non_finite_counts_as_above_threshold():
    nees = np.full(40, np.inf)
    result = check_nees_divergence(nees)
    assert result.inconsistent and result.diverged


def test_false_alarm_rate_matches_chi_square_over_many_steps():
    # 2,000,000 healthy queries -> 200,000 windows. Each window should exceed
    # the threshold with probability 0.001 (chi-square(30) tail).
    rng = np.random.default_rng(11)
    rule = DivergenceRule()
    windows = 200_000
    errors = rng.standard_normal(size=(windows * rule.window, 3))
    nees = np.sum(errors ** 2, axis=1)  # identity covariance: NEES is the squared length
    means = nees.reshape(windows, rule.window).mean(axis=1)
    hits = int(np.sum(means > rule.threshold))
    lo, hi = binom.ppf([1e-4, 1 - 1e-4], windows, rule.window_false_alarm)
    assert lo <= hits <= hi
    # Two windows in a row at a looser 5 % level. Use disjoint pairs (0, 1), (2, 3), ...
    # so the pair indicators are independent and their count is binomial with p = 0.05^2.
    loose = DivergenceRule(window_false_alarm=0.05)
    above = means > loose.threshold
    assert float(np.mean(above)) == pytest.approx(0.05, abs=0.003)
    pairs = int(np.sum(above[0::2] & above[1::2]))
    lo2, hi2 = binom.ppf([1e-4, 1 - 1e-4], windows // 2, 0.05 ** 2)
    assert lo2 <= pairs <= hi2


def test_rule_validation():
    with pytest.raises(ValueError):
        DivergenceRule(window=0)
    with pytest.raises(ValueError):
        DivergenceRule(window_false_alarm=1.0)
    with pytest.raises(ValueError):
        DivergenceRule(gross_nees=5.0)
    # An honest window mean reaching 100 needs chi-square(30) above 1000.
    assert chi2.sf(1000, 30) < 1e-100
    assert chi2.isf(0.001, 30) == pytest.approx(59.703, abs=1e-3)


def test_registered_observer_path_does_not_use_the_new_check():
    # Only the opt-in wiring imports the module (plus the d_divergence_ diagnostic scripts):
    # observers.py inside apply_divergence_rule, and d_run_fusion.py behind --nees-divergence.
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    users = [f for d in ("sim", "tools") for f in (root / d).rglob("*.py")
             if not f.name.startswith("d_divergence_")
             and "sim.estimation.divergence" in f.read_text(encoding="utf-8")]
    assert sorted(f.relative_to(root).as_posix() for f in users) == [
        "sim/coop/observers.py", "tools/d_run_fusion.py"]
