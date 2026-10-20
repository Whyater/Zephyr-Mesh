"""Opt-in check for a filter that has quietly diverged (PROPOSAL, not adopted).

The existing check in ``sim.coop.observers`` only notices non-finite numbers.
A filter can also drift far from the truth while every number stays finite.
This module flags that case from the NEES series (normalized estimation error
squared: the squared error measured in the filter's own standard deviations).

How it works:

- Average the NEES values in non-overlapping windows of ``window`` queries.
- For an honest filter with ``dof`` axes and independent errors, a window mean
  is chi-square(dof * window) / window, so ``threshold`` below is crossed by
  chance with probability ``window_false_alarm``.
- Flag only when ``consecutive`` windows in a row are above the threshold, so
  one bad spike cannot trigger it.

Two levels come out:

- ``inconsistent``: the chi-square level above. The filter is more surprised
  than an honest filter would be. Overconfident but bounded filters (a turning
  target the motion model does not expect) also land here, so this is a
  warning, not a failure.
- ``diverged``: the same persistence test at ``gross_nees`` (default 100, an
  error that averages about 10 times the claimed standard deviation). An honest
  filter reaches it with essentially zero chance (chi-square(30) above 1000).
  The value 100 is a chosen line, not a chi-square quantile; it was fixed
  before looking at any run data.

Nothing in the registered runs calls this. It changes no existing result.
Errors at neighbouring queries are correlated in a real filter, so the
chi-square false-alarm figure is an idealized value; see the tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class DivergenceRule:
    """Settings for the windowed NEES check."""

    dof: int = 3
    window: int = 10
    window_false_alarm: float = 0.001
    consecutive: int = 3
    gross_nees: float = 100.0

    def __post_init__(self):
        for name in ("dof", "window", "consecutive"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.window_false_alarm < 1.0:
            raise ValueError("window_false_alarm must be in (0, 1)")
        if not np.isfinite(self.gross_nees) or self.gross_nees <= self.threshold:
            raise ValueError("gross_nees must be finite and above the chi-square threshold")

    @property
    def threshold(self) -> float:
        """Window-mean NEES above which a window counts as 'too surprised'."""
        return float(chi2.isf(self.window_false_alarm, self.dof * self.window) / self.window)


@dataclass(frozen=True)
class DivergenceResult:
    inconsistent: bool
    inconsistent_first_window: int | None
    diverged: bool
    first_window: int | None
    window_means: tuple[float, ...]
    threshold: float
    gross_nees: float


def _first_run(means, level: float, consecutive: int) -> int | None:
    run = 0
    for i, m in enumerate(means):
        if not np.isfinite(m) or m > level:
            run += 1
            if run >= consecutive:
                return i - consecutive + 1
        else:
            run = 0
    return None


def nees_values(errors, covariances) -> np.ndarray:
    """NEES e' P^-1 e for each (error vector, covariance) pair."""
    out = []
    for e, p in zip(errors, covariances):
        e = np.asarray(e, dtype=float)
        p = np.asarray(p, dtype=float)
        out.append(float(e @ np.linalg.solve(p, e)))
    return np.asarray(out, dtype=float)


def check_nees_divergence(nees, rule: DivergenceRule | None = None) -> DivergenceResult:
    """Flag a run whose windowed NEES stays above the chi-square threshold.

    A non-finite NEES value counts as an above-threshold window, so this check
    also covers what the old non-finite check catches.
    """
    rule = rule if rule is not None else DivergenceRule()
    vals = np.asarray(nees, dtype=float).ravel()
    nwin = vals.size // rule.window
    means = [float(np.mean(vals[i * rule.window:(i + 1) * rule.window])) for i in range(nwin)]
    return check_window_means(means, rule)


def check_window_means(window_means, rule: DivergenceRule | None = None) -> DivergenceResult:
    """Same check, starting from window means that were already computed.

    ``CoopOutcome.nees_window_means`` in ``sim.coop.observers`` holds exactly
    these (10 queries, 3 axes), so stored outcomes can be checked without a rerun.
    """
    rule = rule if rule is not None else DivergenceRule()
    means = tuple(float(m) for m in window_means)
    warn = _first_run(means, rule.threshold, rule.consecutive)
    gross = _first_run(means, rule.gross_nees, rule.consecutive)
    return DivergenceResult(warn is not None, warn, gross is not None, gross,
                            means, rule.threshold, rule.gross_nees)
