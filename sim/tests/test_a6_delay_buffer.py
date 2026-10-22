"""First-principles checks for the EXP-021 delay and loss channel."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim.physics.delay_buffer import DelayLossChannel


def _wilson_interval(successes: int, trials: int, confidence_z: float = 2.5758293035489004):
    proportion = successes / trials
    z2 = confidence_z * confidence_z
    denominator = 1.0 + z2 / trials
    centre = (proportion + z2 / (2.0 * trials)) / denominator
    radius = confidence_z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials)
    ) / denominator
    return centre - radius, centre + radius


def test_zero_delay_zero_loss_returns_current_sample():
    channel = DelayLossChannel(
        delay_s=0.0,
        loss_p=0.0,
        sample_period_s=0.02,
        rng=np.random.default_rng(11),
        initial_value="initial",
    )
    for index in range(100):
        t_now = index * 0.02
        channel.offer(t_now, index)
        assert channel.latest(t_now) == index


def test_pure_delay_returns_ramp_sample_from_exact_lag():
    sample_period = 0.01
    delay = 0.03
    channel = DelayLossChannel(delay, 0.0, sample_period, np.random.default_rng(12), initial_value=-1)
    for index in range(100):
        t_now = index * sample_period
        channel.offer(t_now, index)
        expected = index - 3 if index >= 3 else -1
        assert channel.latest(t_now) == expected


def test_drop_rate_is_inside_99_percent_wilson_interval():
    probability = 0.3
    trials = 100_000
    channel = DelayLossChannel(0.0, probability, 0.02, np.random.default_rng(12345), initial_value=0)
    for index in range(trials):
        channel.offer(index * 0.02, index)
    lower, upper = _wilson_interval(channel.dropped_count, trials)
    observed = channel.dropped_count / trials
    assert lower <= observed <= upper
    assert lower <= probability <= upper


def test_hold_last_extra_age_matches_geometric_loss_formula():
    probability = 0.3
    sample_period = 0.02
    trials = 100_000
    channel = DelayLossChannel(0.0, probability, sample_period, np.random.default_rng(54321), initial_value=0)
    ages = []
    for index in range(trials):
        t_now = index * sample_period
        channel.offer(t_now, index)
        held_index = channel.latest(t_now)
        if index > 100:
            ages.append((index - held_index) * sample_period)
    expected = sample_period * probability / (1.0 - probability)
    assert float(np.mean(ages)) == pytest.approx(expected, rel=0.05)


def test_common_random_numbers_make_higher_loss_a_superset_of_drops():
    sample_period = 0.02
    trials = 10_000
    channels = {
        probability: DelayLossChannel(
            0.0,
            probability,
            sample_period,
            np.random.default_rng(99),
            initial_value=0,
        )
        for probability in (0.1, 0.3)
    }
    for index in range(trials):
        t_send = index * sample_period
        for channel in channels.values():
            channel.offer(t_send, index)
    assert set(channels[0.1].drop_indices).issubset(set(channels[0.3].drop_indices))

