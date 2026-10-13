"""Independent cross-check for the S6 recovery implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest

import sim.s6_recovery as s6


Q_ACCELERATION_PSD = 0.1
DEFAULT_TRACKER_STD = 0.05


@dataclass
class _Axis:
    x: np.ndarray
    p: np.ndarray
    timestamp: float

    @classmethod
    def initial(cls, position: float) -> "_Axis":
        return cls(np.array([position, 0.0], dtype=float), np.eye(2, dtype=float), 0.0)

    def predicted(self, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
        dt = timestamp - self.timestamp
        if dt < -1e-12:
            raise ValueError("prediction time moved backwards")
        if dt <= 0.0:
            return self.x.copy(), self.p.copy()
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = Q_ACCELERATION_PSD * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]
        )
        return f @ self.x, f @ self.p @ f.T + q

    def predict_to(self, timestamp: float) -> None:
        self.x, self.p = self.predicted(timestamp)
        self.timestamp = timestamp

    def update(self, measurement: float, timestamp: float, measurement_std: float) -> tuple[float, float]:
        self.predict_to(timestamp)
        r = measurement_std**2
        innovation = measurement - self.x[0]
        s = self.p[0, 0] + r
        nis = innovation * innovation / s
        k = self.p[:, 0] / s
        self.x = self.x + k * innovation
        h = np.array([[1.0, 0.0]])
        identity = np.eye(2)
        ikh = identity - k[:, None] @ h
        self.p = ikh @ self.p @ ikh.T + np.outer(k, k) * r
        self.p = 0.5 * (self.p + self.p.T)
        return float(nis), float(s)


@dataclass
class _ReferenceArmResult:
    p95_error_m: float
    max_error_m: float
    rmse_m: float
    fused_count: int
    dropped_count: int
    fused_out_of_order_count: int
    mean_nis: float | None
    failed: bool
    query_times_s: np.ndarray
    errors_m: np.ndarray


def _truth_position(motion: str, times: np.ndarray, config) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if motion == "constant_velocity":
        out = np.zeros((times.size, 3), dtype=float)
        out[:, 0] = config.target_speed_mps * times
        return out
    if motion == "constant_turn":
        wt = config.target_speed_mps / config.turn_radius_m * times
        out = np.zeros((times.size, 3), dtype=float)
        out[:, 0] = config.turn_radius_m * np.sin(wt)
        out[:, 1] = config.turn_radius_m * (1.0 - np.cos(wt))
        return out
    raise ValueError(motion)


def _reference_arm(deliveries, motion: str, config, *, recovery: bool, tracker_std: float):
    query_times = np.arange(0.0, config.duration_s + config.query_dt_s / 2.0, config.query_dt_s)
    truth_queries = _truth_position(motion, query_times, config)
    truth_initial = truth_queries[0]
    axes = [_Axis.initial(float(v)) for v in truth_initial]
    next_delivery = 0
    last_capture = -math.inf
    newest_capture = -math.inf
    has_fused = False
    fused_count = dropped_count = out_of_order_count = 0
    nis_values: list[float] = []
    errors: list[float] = []

    for t, truth in zip(query_times, truth_queries):
        while next_delivery < len(deliveries) and deliveries[next_delivery].receive_time_s <= t + 1e-12:
            delivery = deliveries[next_delivery]
            next_delivery += 1
            stamp = delivery.capture_time_s if recovery else delivery.receive_time_s
            if recovery:
                if delivery.capture_time_s <= last_capture:
                    dropped_count += 1
                    continue
            elif has_fused and abs(stamp - axes[0].timestamp) <= 1e-12:
                dropped_count += 1
                continue
            if not recovery and delivery.capture_time_s < newest_capture:
                out_of_order_count += 1
            predicted_nis = 0.0
            for axis, measurement in zip(axes, delivery.measurement_m):
                nis, _ = axis.update(float(measurement), float(stamp), tracker_std)
                predicted_nis += nis
            if stamp >= config.burn_in_s - 1e-12:
                nis_values.append(predicted_nis)
            fused_count += 1
            has_fused = True
            newest_capture = max(newest_capture, delivery.capture_time_s)
            if recovery:
                last_capture = delivery.capture_time_s

        estimate = np.array([axis.predicted(float(t))[0][0] for axis in axes])
        errors.append(float(np.linalg.norm(estimate - truth)))

    errors_array = np.asarray(errors, dtype=float)
    scored = errors_array[query_times >= config.burn_in_s - 1e-9]
    p95 = float(np.percentile(scored, 95))
    max_error = float(np.max(scored))
    rmse = float(np.sqrt(np.mean(scored**2)))
    mean_nis = float(np.mean(nis_values)) if nis_values else None
    failed = bool(p95 > config.failure_p95_m or max_error > config.failure_max_m)
    return _ReferenceArmResult(
        p95, max_error, rmse, fused_count, dropped_count, out_of_order_count,
        mean_nis, failed, query_times, errors_array
    )


def _cells():
    cells = []
    for motion in s6.MOTIONS:
        cells.extend([
            ("zero", motion, 0.0, 0.0, 0.0, "independent", 0.0, 0.05),
            ("delay025", motion, 0.025, 0.0, 0.02, "independent", 0.0, 0.05),
            ("delay100", motion, 0.1, 0.3, 0.1, "independent", 0.0, 0.05),
            ("delay300", motion, 0.3, 0.5, 0.1, "independent", 0.0, 0.05),
            ("burst", motion, 0.1, 0.3, 0.05, "burst", 0.0, 0.05),
            ("jitter", motion, 0.2, 0.3, 0.05, "independent", 0.1, 0.05),
            ("oracle", motion, 0.1, 0.0, 0.1, "independent", 0.0, 0.1),
            ("oracle_loss", motion, 0.3, 0.3, 0.1, "independent", 0.0, 0.1),
        ])
    return cells


@pytest.mark.parametrize("cell", _cells(), ids=lambda c: f"{c[0]}-{c[1]}")
def test_reference_arms_match_production(cell):
    label, motion, delay, loss, noise, loss_model, jitter, tracker_std = cell
    config = s6.RecoveryConfig(tracker_measurement_std_m=tracker_std)
    seeds = s6.seed_list(3)
    for seed in seeds:
        deliveries = s6.deliver(
            motion,
            delay_s=delay,
            loss_probability=loss,
            noise_std_m=noise,
            seed=seed,
            config=config,
            loss_model=loss_model,
            jitter_s=jitter,
        )
        expected = {
            "naive": _reference_arm(deliveries, motion, config, recovery=False, tracker_std=tracker_std),
            "recovery": _reference_arm(deliveries, motion, config, recovery=True, tracker_std=tracker_std),
        }
        actual = s6.run_arms(deliveries, motion, config, tracker_measurement_std_m=tracker_std)
        for arm in ("naive", "recovery"):
            got = actual[arm]
            want = expected[arm]
            assert got.p95_error_m == pytest.approx(want.p95_error_m, abs=1e-9), (label, motion, seed, arm, "p95")
            assert got.max_error_m == pytest.approx(want.max_error_m, abs=1e-9), (label, motion, seed, arm, "max")
            assert got.rmse_m == pytest.approx(want.rmse_m, abs=1e-9), (label, motion, seed, arm, "rmse")
            np.testing.assert_allclose(got.errors_m, want.errors_m, atol=1e-9, rtol=0.0)
            assert got.dropped_count == want.dropped_count
            assert got.fused_count == want.fused_count
            assert got.fused_out_of_order_count == want.fused_out_of_order_count
            if got.mean_nis is None or want.mean_nis is None:
                assert got.mean_nis is want.mean_nis
            else:
                assert got.mean_nis == pytest.approx(want.mean_nis, abs=1e-6)


def test_simulate_run_matches_run_arms_for_one_cell():
    config = s6.RecoveryConfig()
    kwargs = dict(
        delay_s=0.1,
        loss_probability=0.3,
        noise_std_m=0.1,
        seed=s6.seed_list(3)[0],
        config=config,
        loss_model="independent",
        jitter_s=0.0,
    )
    deliveries = s6.deliver("constant_turn", **kwargs)
    direct = s6.run_arms(deliveries, "constant_turn", config, tracker_measurement_std_m=0.05)
    simulated = s6.simulate_run("constant_turn", **kwargs)
    for arm in ("naive", "recovery"):
        assert simulated[arm].p95_error_m == pytest.approx(direct[arm].p95_error_m, abs=1e-9)
        assert simulated[arm].max_error_m == pytest.approx(direct[arm].max_error_m, abs=1e-9)
        assert simulated[arm].rmse_m == pytest.approx(direct[arm].rmse_m, abs=1e-9)
        np.testing.assert_allclose(simulated[arm].errors_m, direct[arm].errors_m, atol=1e-9, rtol=0.0)
        assert simulated[arm].fused_count == direct[arm].fused_count
        assert simulated[arm].dropped_count == direct[arm].dropped_count
        assert simulated[arm].fused_out_of_order_count == direct[arm].fused_out_of_order_count
        assert simulated[arm].mean_nis == pytest.approx(direct[arm].mean_nis, abs=1e-6)
