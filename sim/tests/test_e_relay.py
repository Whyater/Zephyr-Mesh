"""EXP-032 relay mission checks."""
import json

import pytest

from sim.missions.api import Scenario
from sim.missions.engine import run_mission
from sim.missions.log import validate_log
from sim.tests.reference_e import hand_checks

import sim.missions.relay  # noqa: F401


def _run(seed, *, distance=900.0, fraction=None, duration=10.0, log=False):
    return run_mission(
        Scenario(
            mission="relay",
            seed=seed,
            duration_s=duration,
            params={"distance_m": distance, "relay_fraction": fraction},
        ),
        log=log,
    )


def test_direct_sweep_brackets_reference_range(capsys):
    breakpoint_m = 4.0 * 1.5 * 1.5 / (299_792_458.0 / 2.437e9)
    reference_m = hand_checks.max_range_dual_slope_m(
        19.5 - 6.0, -98.0, 40.2, 2.0, breakpoint_m, 4.0, 0.0
    )
    distances = (300.0, 500.0, 700.0, 900.0, 1100.0)
    values = {
        distance: [_run(seed, distance=distance).metrics["delivery_ratio"] for seed in (1, 2, 3)]
        for distance in distances
    }
    medians = {distance: sum(rows) / len(rows) for distance, rows in values.items()}
    for distance in distances:
        print(f"D={distance:.0f} m deliveries={values[distance]} mean={medians[distance]:.3f}")
    crossing = next(distance for distance in distances if medians[distance] < 0.5)
    lower = max(distance for distance in distances if distance < crossing)
    print(f"reference_median_signal_range={reference_m:.3f} m bracket=({lower:.1f}, {crossing:.1f}) m")
    assert lower <= 1.5 * reference_m
    assert crossing >= reference_m / 1.5
    assert "D=300" in capsys.readouterr().out


def test_midpoint_relay_beats_direct_at_900m():
    direct = [_run(seed, fraction=None).metrics["delivery_ratio"] for seed in (1, 2, 3)]
    relay = [_run(seed, fraction=0.5).metrics["delivery_ratio"] for seed in (1, 2, 3)]
    assert sum(relay) / len(relay) > sum(direct) / len(direct)


def test_relay_latency_exceeds_direct_at_300m():
    direct = [_run(seed, distance=300.0, fraction=None).metrics["median_latency_s"] for seed in (1, 2, 3)]
    relay = [_run(seed, distance=300.0, fraction=0.5).metrics["median_latency_s"] for seed in (1, 2, 3)]
    assert sum(relay) / len(relay) > sum(direct) / len(direct)


def test_relay_is_deterministic_and_log_validates():
    first = _run(7, fraction=0.5, duration=3.0, log=True)
    second = _run(7, fraction=0.5, duration=3.0, log=True)
    validate_log(first.log)
    validate_log(second.log)
    assert first.metrics == second.metrics
    assert json.dumps(first.log, sort_keys=True) == json.dumps(second.log, sort_keys=True)


@pytest.mark.parametrize("fraction", [None, 0.25, 0.4, 0.5, 0.6, 0.75])
def test_relay_fraction_validation_and_metrics_shape(fraction):
    result = _run(2, fraction=fraction, duration=1.0)
    assert isinstance(result.metrics["delivery_ratio"], float)
    if fraction is None:
        assert result.metrics["hop1_delivery"] is None
        assert result.metrics["hop2_delivery"] is None
    else:
        assert result.metrics["hop1_delivery"] is not None
        assert result.metrics["hop2_delivery"] is not None
