"""Opt-in turn-aware filter and shared fading in the cooperative tracking simulation.

Checks for EXP-036 (proposal): ``CoopCell.estimator`` selects the filter used
by the single and central arms, and ``LinkSpec.loss_correlation="shared"``
makes every route lose packets in the same capture slots. The defaults must
leave every registered result unchanged.
"""
from __future__ import annotations

import csv
import gzip

import numpy as np
import pytest

from sim.coop.net import LinkSpec, TrafficSpec, make_deliveries
from sim.coop.observers import (
    CoopCell,
    _phase_offsets,
    _truth_and_measurements,
    run_seed,
)
from sim.estimation.base import EstimatorSettings
from sim.estimation.factory import make_estimator
from sim.s6_recovery import RecoveryConfig

BURST = {"mode": "abstract", "loss_probability": 0.3, "loss_model": "burst"}


def _lost(route):
    """Link losses only; a packet that would arrive after the 12 s end is not a fade."""
    return [row.loss_reason == "link" for row in route]


# ---------------------------------------------------------------- defaults


def test_defaults_are_the_registered_settings():
    assert LinkSpec("abstract", 0.0, 0.0).loss_correlation == "independent"
    assert CoopCell().estimator == "cv"


def test_default_tables_do_not_change_when_the_option_is_spelled_out():
    traffic = TrafficSpec("star", ("o1", "o2", "o3"), phase_offsets_s=(0.0, 0.03, 0.07))
    for seed in (3, 17):
        plain = make_deliveries(traffic, LinkSpec(delay_s=0.1, **BURST), seed)
        spelled = make_deliveries(traffic, LinkSpec(delay_s=0.1, loss_correlation="independent", **BURST), seed)
        assert dict(plain) == dict(spelled)


@pytest.mark.parametrize("arm,n", [("single", 1), ("central", 3)])
def test_default_cv_rows_still_match_the_registered_d3_run(arm, n):
    """Two registered EXP-024 rows (1 m turn, 30 % bursts) recomputed from scratch."""
    with gzip.open("sim/runs/d3-tolerance/per_seed.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["link"] == "abstract" and r["target"] == "ct_r1" and r["loss_model"] == "burst"
                and r["arm"] == arm and int(r["n"]) == n and float(r["delay"]) in (0.0, 0.2)]
    rows = rows[:2] + rows[-2:]
    assert len(rows) == 4
    for row in rows:
        cell = CoopCell(target="ct", n_observers=n,
                        link=LinkSpec(delay_s=float(row["delay"]), **BURST))
        got = run_seed(cell, int(row["seed"]), arms=(arm,))[arm]
        assert repr(float(got.p95_error_m)) == repr(float(row["p95"]))


# ---------------------------------------------------------------- shared fading


def test_option_values_are_checked():
    with pytest.raises(ValueError):
        LinkSpec("abstract", 0.0, 0.3, loss_correlation="partly")
    with pytest.raises(ValueError):
        LinkSpec("radio", 0.0, 0.3, positions={"gs": (0, 0, 0)}, loss_correlation="shared")
    with pytest.raises(ValueError):
        LinkSpec("abstract", 0.1, 0.3, jitter_s=0.01, loss_correlation="shared")


@pytest.mark.parametrize("pattern", ["star", "peer"])
def test_shared_fading_loses_the_same_capture_slots_on_every_route(pattern):
    names = ("o1", "o2", "o3")
    traffic = TrafficSpec(pattern, names, phase_offsets_s=(0.0, 0.04, 0.09))
    single = TrafficSpec("star", ("o1",))
    for seed in (5, 8, 13):
        link = LinkSpec(delay_s=0.05, loss_correlation="shared", **BURST)
        table = make_deliveries(traffic, link, seed)
        reference = _lost(make_deliveries(single, LinkSpec(delay_s=0.05, **BURST), seed)[("o1", "gs")])
        assert any(reference) and not all(reference)
        for route in table.values():
            # Observers with a phase offset miss the last capture at 12 s.
            assert len(route) in (len(reference), len(reference) - 1)
            assert _lost(route) == reference[:len(route)]
            for row in route:
                if row.receive_time_s is not None:
                    assert row.receive_time_s == row.send_time_s + 0.05


def test_shared_fading_with_one_observer_is_the_independent_table():
    traffic = TrafficSpec("star", ("o1",))
    for seed in (2, 9):
        for delay in (0.0, 0.15):
            a = make_deliveries(traffic, LinkSpec(delay_s=delay, **BURST), seed)
            b = make_deliveries(traffic, LinkSpec(delay_s=delay, loss_correlation="shared", **BURST), seed)
            assert dict(a) == dict(b)


def test_all_observers_lost_fraction_hand_values():
    """Hand values: shared bursts lose every observer together as often as one link
    loses a packet (0.3); independent bursts all fail together about 0.3 ** 3 = 0.027
    of the time. Pooled over 40 seeds of 121 slots."""
    names = ("o1", "o2", "o3")
    traffic = TrafficSpec("star", names, phase_offsets_s=(0.0, 0.03, 0.06))
    frac = {}
    for corr in ("independent", "shared"):
        lost_all = 0
        lost_one = 0
        total = 0
        for seed in range(40):
            table = make_deliveries(traffic, LinkSpec(delay_s=0.0, loss_correlation=corr, **BURST), seed)
            routes = [_lost(table[(n, "gs")]) for n in names]
            steps = min(len(r) for r in routes)
            lost_all += sum(all(r[k] for r in routes) for k in range(steps))
            lost_one += sum(routes[0][:steps])
            total += steps
        frac[corr] = (lost_all / total, lost_one / total)
    assert frac["shared"][0] == frac["shared"][1]
    assert abs(frac["shared"][0] - 0.3) < 0.06
    assert abs(frac["independent"][0] - 0.027) < 0.02


# ---------------------------------------------------------------- turn-aware filter


def test_estimator_option_is_checked_and_ct_is_limited_to_single_and_central():
    with pytest.raises(ValueError):
        CoopCell(estimator="imm")
    cell = CoopCell(target="ct", n_observers=2, estimator="ct")
    for arm in ("star_avg", "peer_ci", "peer_naive"):
        with pytest.raises(ValueError):
            run_seed(cell, 1, arms=(arm,))


@pytest.mark.parametrize("delay_key,delay", [("d0", 0.0), ("d0.2", 0.2)])
def test_one_observer_ct_reproduces_the_atlas_ct_arm(delay_key, delay):
    """Cross-implementation check: the atlas engine (sim/estimation/engine.py) and the
    cooperative single arm are separate code paths. On the atlas seeds they must give
    the same per-seed p95 for the turn-aware filter."""
    key = f"ct/rate10/{delay_key}/j0/l0.3-burst/gaussian0.05"
    with gzip.open("sim/runs/s6-atlas/per_seed.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        rows = {int(r["seed_index"]): r for r in csv.DictReader(handle)
                if r["cell_key"] == key and r["arm"] == "ct"}
    assert len(rows) == 20
    for index in (0, 1, 2, 7):
        row = rows[index]
        cell = CoopCell(target="ct", n_observers=1, estimator="ct", link=LinkSpec(delay_s=delay, **BURST))
        got = run_seed(cell, int(row["seed"]), arms=("single",))["single"]
        assert abs(got.p95_error_m - float(row["p95_error_m"])) <= 1e-9
        assert got.failed == (row["failed"] == "true")


def _averaged_reference(seed: int, delay: float) -> float:
    """Hand-built reference for the limiting case below, written without the
    cooperative code: one CT filter fed the mean of the three synchronous
    measurements with R = sigma^2 / 3, on the shared loss pattern, scored like
    the cooperative arms (every 0.02 s from 2 s to 12 s)."""
    config = RecoveryConfig()
    cell = CoopCell(target="ct", n_observers=3, phase="sync", estimator="ct")
    names = ("o1", "o2", "o3")
    phases = _phase_offsets(cell, names, seed)
    target, names, captures, measurements = _truth_and_measurements(cell, seed, config, phases)
    lost = _lost(make_deliveries(TrafficSpec("star", ("o1",)), LinkSpec(delay_s=delay, **BURST), seed)[("o1", "gs")])
    est = make_estimator("ct", EstimatorSettings(), np.asarray(target.start_position, dtype=float),
                         r_std_m=0.05 / np.sqrt(3.0))
    mean_z = (measurements["o1"] + measurements["o2"] + measurements["o3"]) / 3.0
    arrivals = [(captures["o1"][k] + delay, k) for k in range(len(lost)) if not lost[k]]
    queries = np.arange(0.0, config.duration_s + config.query_dt_s / 2.0, config.query_dt_s)
    errors = []
    cursor = 0
    for t in queries:
        while cursor < len(arrivals) and arrivals[cursor][0] <= t + 1e-12:
            k = arrivals[cursor][1]
            est.update(mean_z[k], float(captures["o1"][k]))
            cursor += 1
        pred = est.predict_at(max(float(t), est.timestamp_s))
        if t >= config.burn_in_s - 1e-9:
            errors.append(np.linalg.norm(pred.position_m - target.position(np.array([t]))[0]))
    return float(np.percentile(errors, 95))


@pytest.mark.parametrize("seed", [4, 21])
@pytest.mark.parametrize("delay", [0.0, 0.1])
def test_limiting_case_sync_shared_three_observers_equal_one_averaged_observer(seed, delay):
    """Three observers that capture at the same instants and fade together carry
    exactly the information of one observer whose measurement is their average with a
    third of the variance (sequential Kalman updates at one time with H = [I 0]). So
    under shared fading, cooperation can only help as much as noise averaging does."""
    cell = CoopCell(target="ct", n_observers=3, phase="sync", estimator="ct",
                    link=LinkSpec(delay_s=delay, loss_correlation="shared", **BURST))
    got = run_seed(cell, seed, arms=("central",))["central"]
    assert abs(got.p95_error_m - _averaged_reference(seed, delay)) <= 1e-9
