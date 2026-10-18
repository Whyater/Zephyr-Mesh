"""Simulated cooperative multi-observer tracking experiments.

The five arms are a single local filter, a central sequential filter, a
ground-station average, and two peer-feedback filters.  ``peer_ci`` combines
the local track and the latest received peer tracks with covariance
intersection.  ``peer_naive`` uses information fusion for the same inputs and
feeds that fused track back into the next broadcast.  The feedback loop
therefore double counts information and can diverge; divergence is reported
as a failed run rather than hidden with damping or clipping.

``CoopCell.phase`` selects random, synchronous, or staggered capture phases.
``CoopCell.ci`` selects fast (trace-weighted) or exact simplex CI for peer
observers.  The ground station uses fast CI for the CI arm because it is the
stable, deterministic scoring path; the naive arm remains information fusion
so its overconfidence is observable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import numpy as np

from sim.estimation.base import EstimatorSettings
from sim.estimation.consistency import nis_band
from sim.estimation.factory import make_estimator
from sim.estimation.targets import make_target
from sim.s6_recovery import RecoveryConfig
from sim.sensor import SensorConfig, SensorModel

from .fusion import covariance_intersection_many
from .net import LinkSpec, TrafficSpec, default_phases, make_deliveries

TIME_TOLERANCE_S = 1e-12
_I3 = np.eye(3)
_I6 = np.eye(6)
BURN_IN_TOLERANCE_S = 1e-9
ARMS = ("single", "central", "star_avg", "peer_ci", "peer_naive")
_COVARIANCE_FLOOR = 1e-16
@lru_cache(maxsize=256)
def _predict_matrices(dt: float, q: float):
    """Return bounded-cache CV transition matrices for one time step."""
    F = np.block([[_I3, dt * _I3], [np.zeros((3, 3)), _I3]])
    Q = q * np.block([[_I3 * (dt ** 3 / 3.0), _I3 * (dt ** 2 / 2.0)],
                      [_I3 * (dt ** 2 / 2.0), _I3 * dt]])
    return F, Q


@dataclass(frozen=True)
class CoopCell:
    """One simulated cooperative tracking cell.

    ``noise_std_m`` controls simulated sensor samples. ``r_std_m`` controls the
    filter's measurement covariance. When it is ``None``, the filter keeps the
    S6 and atlas setting of 0.05 m even when the simulated sensor noise differs.
    ``phase`` is ``random`` by default, or ``sync``/``staggered`` for explicit
    schedules. ``ci`` selects fast trace-weighted CI or exact optimized CI for
    peer observers. Ground-station CI always uses the fast implementation for
    deterministic, low-cost scoring; the peer-naive ground-station arm uses
    information fusion to expose its feedback overconfidence. The ground
    station averages the uninformed initial state until each observer's first
    arrival in ``star_avg``. In ``peer_naive``, it predicts the last fused gs
    track between arrivals.
    """

    target: str = "cv"
    target_params: tuple = ()
    n_observers: int = 1
    rate_hz: float = 10.0
    noise_std_m: float = 0.05
    r_std_m: float | None = None
    link: LinkSpec = field(default_factory=lambda: LinkSpec(mode="abstract", delay_s=0.0,
                                                            loss_probability=0.0))
    phase: str = "random"
    ci: str = "fast"

    def __post_init__(self) -> None:
        if self.target not in ("cv", "ct", "jink"):
            raise ValueError("target must be cv, ct, or jink")
        if int(self.n_observers) != self.n_observers or self.n_observers < 1:
            raise ValueError("n_observers must be a positive integer")
        if not np.isfinite(self.rate_hz) or self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        if not np.isfinite(self.noise_std_m) or self.noise_std_m < 0:
            raise ValueError("noise_std_m must be nonnegative")
        if self.r_std_m is not None and (not np.isfinite(self.r_std_m) or self.r_std_m < 0):
            raise ValueError("r_std_m must be finite and nonnegative")
        if self.phase not in ("random", "sync", "staggered"):
            raise ValueError("phase must be random, sync, or staggered")
        if self.ci not in ("fast", "exact"):
            raise ValueError("ci must be fast or exact")
        params = self.target_params.items() if isinstance(self.target_params, dict) else self.target_params
        try:
            params = tuple(sorted((str(k), v) for k, v in params))
        except (TypeError, ValueError) as exc:
            raise ValueError("target_params must be a mapping or pairs") from exc
        object.__setattr__(self, "target_params", params)


@dataclass(frozen=True)
class CoopOutcome:
    p95_error_m: float
    max_error_m: float
    rmse_m: float
    failed: bool
    mean_nees: float | None
    nees_window_means: tuple
    frac_windows_above_band: float
    fused_count: int
    extras: dict = field(default_factory=dict)


class GapStatistics(tuple):
    """Backward-compatible gap durations with count summaries.

    Iteration yields durations as in the original helper. String indexing and
    attributes expose counts for all missing capture indices and the leading
    run before the first arrival.
    """

    def __new__(cls, gaps=(), *, total=0, leading=0):
        obj = super().__new__(cls, tuple(float(g) for g in gaps))
        obj.total = int(total)
        obj.leading = int(leading)
        obj.leading_loss = int(leading)
        return obj

    def __getitem__(self, item):
        if item == "total":
            return self.total
        if item in ("leading", "leading_loss"):
            return self.leading
        if item == "gaps":
            return tuple(self)
        return super().__getitem__(item)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default

    def keys(self):
        return ("gaps", "total", "leading", "leading_loss")

    def as_dict(self):
        return {"gaps": tuple(self), "total": self.total, "leading": self.leading, "leading_loss": self.leading_loss}


def _params(cell: CoopCell) -> dict:
    out = dict(cell.target_params)
    if "speed" in out and "speed_mps" not in out:
        out["speed_mps"] = out.pop("speed")
    if "radius" in out and "radius_m" not in out:
        out["radius_m"] = out.pop("radius")
    return out


def _capture_times(cell: CoopCell, phases: tuple[float, ...], observer_index: int,
                   config: RecoveryConfig) -> np.ndarray:
    period = 1.0 / float(cell.rate_hz)
    offset = float(phases[observer_index])
    count = int(round(float(config.duration_s) / period))
    raw = np.arange(count + 1, dtype=float) * period + offset
    return raw[raw <= config.duration_s + TIME_TOLERANCE_S]


def _phase_offsets(cell: CoopCell, names: tuple[str, ...], seed: int) -> tuple[float, ...]:
    if cell.phase == "sync":
        return (0.0,) * len(names)
    if cell.phase == "staggered":
        return tuple(float(i / (len(names) * cell.rate_hz)) for i in range(len(names)))
    return tuple(float(v) for v in default_phases(names, cell.rate_hz, int(seed)))


def _truth_and_measurements(cell: CoopCell, seed: int, config: RecoveryConfig,
                            phases: tuple[float, ...]):
    target_seed = int(np.random.SeedSequence([int(seed), 1]).generate_state(1)[0])
    target = make_target(cell.target, _params(cell), seed=target_seed)
    names = tuple(f"o{i + 1}" for i in range(cell.n_observers))
    captures = {}
    measurements = {}
    for i, name in enumerate(names):
        times = _capture_times(cell, phases, i, config)
        if i == 0:
            sensor_seed = int(np.random.SeedSequence(int(seed)).generate_state(2)[1])
        else:
            observer_number = i + 1
            # Keep observer o2 on SeedSequence([seed, 102]) for S6-compatible streams.
            sensor_seed = int(np.random.SeedSequence([int(seed), 100 + observer_number]).generate_state(1)[0])
        sensor = SensorModel(SensorConfig(noise_std_m=float(cell.noise_std_m)), seed=sensor_seed)
        pos = np.asarray(target.position(times), dtype=float)
        captures[name] = times
        measurements[name] = np.asarray(
            [sensor.sample(p, float(t)).measurement_m for p, t in zip(pos, times)], dtype=float
        )
    return target, names, captures, measurements


def _new_estimator(target, settings: EstimatorSettings, r_std_m: float | None = None):
    return make_estimator("cv", settings, np.asarray(target.start_position, dtype=float), r_std_m=r_std_m)


def _initial_state(target, settings: EstimatorSettings):
    x = np.r_[np.asarray(target.start_position, dtype=float), np.zeros(3)]
    P = np.diag([settings.initial_position_variance_m2] * 3 +
                [settings.initial_velocity_variance_m2ps2] * 3)
    return x, P, 0.0


def _predict_full(mean, cov, from_time: float, to_time: float, q: float):
    dt = max(0.0, float(to_time) - float(from_time))
    x = np.asarray(mean, dtype=float)
    p = np.asarray(cov, dtype=float)
    F, Q = _predict_matrices(dt, float(q))
    predicted_cov = F @ p @ F.T + Q
    return F @ x, (predicted_cov + predicted_cov.T) / 2.0


def _stabilize_cov(cov):
    """Apply the shared tiny symmetric floor required by zero-noise runs."""
    p = np.asarray(cov, dtype=float)
    p = (p + p.T) / 2.0
    return _floor_covariance(p)


def _floor_covariance(cov):
    """Add the shared invertibility floor without changing the mean."""
    p = np.asarray(cov, dtype=float)
    eye = _I6 if p.shape == (6, 6) else _I3
    return p + eye * _COVARIANCE_FLOOR


def _state_from_estimator(est, t: float, settings: EstimatorSettings):
    query = max(float(t), float(est.timestamp_s))
    tr = est.tracker
    q = float(settings.q_acc_psd) * float(est.q_factor) * float(est.q_scale)
    return _predict_full(tr.x, tr.P, float(tr.timestamp_s), query, q)


def _direct_equal_update(est, z: np.ndarray, stamp: float) -> None:
    """Apply a second measurement at an already predicted CV timestamp."""
    tr = est.tracker
    tr.q_acc = est._q()
    tr.P = _stabilize_cov(tr.P)
    tr.R = _stabilize_cov(est.measurement_cov())
    H = tr.H
    innovation = np.asarray(z, dtype=float) - H @ tr.x
    S = H @ tr.P @ H.T + tr.R
    K = np.linalg.solve(S, H @ tr.P).T
    tr.x = tr.x + K @ innovation
    I = np.eye(6)
    tr.P = (I - K @ H) @ tr.P @ (I - K @ H).T + K @ tr.R @ K.T
    tr.P = (tr.P + tr.P.T) / 2.0
    tr.update_count += 1
    tr.timestamp_s = float(stamp)


def _outcome(errors, covariances, config: RecoveryConfig, fused_count: int, extras: dict):
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return CoopOutcome(float("nan"), float("nan"), float("nan"), True,
                           None, (), 0.0, int(fused_count), extras)
    p95 = float(np.percentile(errors, 95))
    maximum = float(np.max(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    vals = np.asarray([_nees_fast(e, c) for e, c in covariances], dtype=float)
    mean = float(np.mean(vals)) if vals.size else None
    nwin = vals.size // 10
    windows = tuple(float(np.mean(vals[i * 10:(i + 1) * 10])) for i in range(nwin))
    high = nis_band(10, dof=3, confidence=0.95)[1]
    frac = float(np.mean(np.asarray(windows) > high)) if windows else 0.0
    return CoopOutcome(p95, maximum, rmse,
                       bool(p95 > config.failure_p95_m or maximum > config.failure_max_m),
                       mean, windows, frac, int(fused_count), extras)


def _query_times(config):
    return np.arange(0.0, config.duration_s + config.query_dt_s / 2.0, config.query_dt_s)


def _route(table, sender, receiver):
    return tuple(d for d in table.get((sender, receiver), ()) if d.receive_time_s is not None)


def _fuse_tracks(means, covariances, *, mode: str, ci_mode: str):
    if len(means) == 1:
        out = (np.asarray(means[0], dtype=float).copy(), _stabilize_cov(covariances[0]))
        if not np.all(np.isfinite(out[0])) or not np.all(np.isfinite(out[1])):
            raise FloatingPointError("non-finite one-track fusion")
        return out
    covariances = [_floor_covariance(p)
                   for p in covariances]
    if mode == "ci":
        if ci_mode == "exact":
            try:
                out = covariance_intersection_many(means, covariances)[:2]
            except ValueError as exc:
                # The exact weight optimiser can fail on degenerate inputs
                # (seen with zero measurement noise). Treat it as divergence.
                raise FloatingPointError(f"exact covariance intersection failed: {exc}") from exc
            if not np.all(np.isfinite(out[0])) or not np.all(np.isfinite(out[1])):
                raise FloatingPointError("non-finite covariance intersection")
            return out
        # Fast covariance intersection uses trace weights. CI remains consistent
        # for every convex weight, trading optimality for speed (Niehsen 2002;
        # Franken and Hupper 2005, fast covariance intersection). The batched
        # form below is algebraically the same as fusion.covariance_intersection_fast
        # but avoids repeated Python validation and one solve per track in the
        # peer event loop.
        xs = np.asarray(means, dtype=float)
        ps = np.asarray(covariances, dtype=float)
        invs = np.linalg.inv(ps)
        traces = np.trace(ps, axis1=1, axis2=2)
        weights = (1.0 / traces)
        weights /= float(np.sum(weights))
        info = np.einsum("i,ijk->jk", weights, invs)
        cov = np.linalg.inv(info)
        cov = (cov + cov.T) * 0.5
        mean = cov @ np.einsum("i,ijk,ik->j", weights, invs, xs)
        out = (mean, cov)
        if not np.all(np.isfinite(out[0])) or not np.all(np.isfinite(out[1])):
            raise FloatingPointError("non-finite covariance intersection")
        return out
    if mode == "naive":
        xs = np.asarray(means, dtype=float)
        ps = np.asarray(covariances, dtype=float)
        invs = np.linalg.inv(ps)
        info = np.sum(invs, axis=0)
        cov = np.linalg.inv(info)
        cov = (cov + cov.T) * 0.5
        out = (cov @ np.einsum("ijk,ik->j", invs, xs), cov)
        if not np.all(np.isfinite(out[0])) or not np.all(np.isfinite(out[1])):
            raise FloatingPointError("non-finite information fusion")
        return out
    raise ValueError(f"unknown fusion mode: {mode}")


def _naive_average_fast(means, covariances):
    xs = np.asarray(means, dtype=float)
    ps = np.asarray([_floor_covariance(p)
                     for p in covariances], dtype=float)
    n = float(len(xs))
    return np.mean(xs, axis=0), np.sum(ps, axis=0) / (n * n)


def _nees_fast(error, covariance):
    e = np.asarray(error, dtype=float)
    return float(e @ np.linalg.solve(np.asarray(covariance, dtype=float), e))


def _score_from_states(query_times, truth, state_fn, config, fused_count, extras):
    errors = []
    covs = []
    for t, tr in zip(query_times, truth):
        m, p = state_fn(float(t))
        err = np.asarray(m[:3]) - tr
        errors.append(float(np.linalg.norm(err)))
        if t >= config.burn_in_s - BURN_IN_TOLERANCE_S:
            covs.append((err, np.asarray(p[:3, :3])))
    mask = query_times >= config.burn_in_s - BURN_IN_TOLERANCE_S
    return _outcome(np.asarray(errors)[mask], covs, config, fused_count, extras)


def _single_or_central(cell, config, target, names, captures, measurements, table, central=False):
    settings = EstimatorSettings()
    est = _new_estimator(target, settings, cell.r_std_m)
    queries = _query_times(config)
    truth = np.asarray(target.position(queries), dtype=float)
    deliveries = []
    senders = names if central else names[:1]
    for sender in senders:
        for d in _route(table, sender, "gs"):
            deliveries.append((float(d.receive_time_s), int(d.seq), sender, d))
    deliveries.sort(key=lambda x: (x[0], float(captures[x[2]][x[1]]), int(x[2][1:]), x[1]))
    cursor = 0
    last_stamp = -np.inf
    fused = 0
    dropped = 0
    errors = []
    covs = []
    for t, tr in zip(queries, truth):
        while cursor < len(deliveries) and deliveries[cursor][0] <= t + TIME_TOLERANCE_S:
            _, _, sender, d = deliveries[cursor]
            cursor += 1
            stamp = float(captures[sender][d.seq])
            if central:
                # Central fuses in receive-time order, then uses capture stamps.
                # This preserves arrival causality without retrodicting late packets.
                # Equal capture stamps remain valid and are ordered by observer id.
                if stamp < last_stamp - TIME_TOLERANCE_S:
                    dropped += 1
                    continue
            elif stamp <= last_stamp + TIME_TOLERANCE_S:
                dropped += 1
                continue
            z = measurements[sender][d.seq]
            if est.update_count == 0:
                est.update(z, stamp)
            elif abs(stamp - est.timestamp_s) <= TIME_TOLERANCE_S:
                _direct_equal_update(est, z, stamp)
            else:
                est.update(z, stamp)
            last_stamp = max(last_stamp, stamp)
            fused += 1
        pred = est.predict_at(max(float(t), float(est.timestamp_s)))
        err = np.asarray(pred.position_m) - tr
        errors.append(float(np.linalg.norm(err)))
        if t >= config.burn_in_s - BURN_IN_TOLERANCE_S:
            covs.append((err, _stabilize_cov(pred.position_cov)))
    mask = queries >= config.burn_in_s - BURN_IN_TOLERANCE_S
    return _outcome(np.asarray(errors)[mask], covs, config, fused,
                    {"dropped_stale": int(dropped), "delivery_table": table})


def _star_average(cell, config, target, names, captures, measurements, table):
    settings = EstimatorSettings()
    local_snapshots = {}
    for sender in names:
        est = _new_estimator(target, settings, cell.r_std_m)
        rows = []
        for k, t in enumerate(captures[sender]):
            est.update(measurements[sender][k], float(t))
            m, p = _state_from_estimator(est, float(t), settings)
            rows.append((m, p))
        local_snapshots[sender] = rows

    latest = {}
    deliveries = []
    for sender in names:
        deliveries.extend((float(d.receive_time_s), int(d.seq), sender, d)
                          for d in _route(table, sender, "gs"))
    deliveries.sort(key=lambda x: (x[0], x[1], int(x[2][1:])))
    cursor = 0
    dropped = 0
    queries = _query_times(config)
    truth = np.asarray(target.position(queries), dtype=float)
    settings_q = settings.q_acc_psd
    initial = _initial_state(target, settings)
    errors = []
    covs = []
    fused_count = 0
    fused_state = None
    fused_dirty = True
    fused_time = 0.0
    for t, tr in zip(queries, truth):
        while cursor < len(deliveries) and deliveries[cursor][0] <= t + TIME_TOLERANCE_S:
            _, _, sender, d = deliveries[cursor]
            cursor += 1
            stamp = float(captures[sender][d.seq])
            old = latest.get(sender)
            if old is not None and stamp <= old[0] + TIME_TOLERANCE_S:
                dropped += 1
                continue
            m, p = local_snapshots[sender][d.seq]
            latest[sender] = (stamp, m.copy(), p.copy())
            fused_count += 1
            fused_dirty = True
        if fused_dirty:
            means = []
            covariances = []
            for sender in names:
                if sender in latest:
                    stamp, m0, p0 = latest[sender]
                    m, p = _predict_full(m0, p0, stamp, float(t), settings_q)
                else:
                    m, p = _predict_full(initial[0], initial[1], 0.0, float(t), settings_q)
                means.append(m)
                covariances.append(p)
            m, p = _naive_average_fast(means, covariances)
            fused_state = (m, p)
            fused_time = float(t)
            fused_dirty = False
        else:
            m, p = _predict_full(fused_state[0], fused_state[1], fused_time, float(t),
                                 settings_q / float(len(names)))
        err = m[:3] - tr
        errors.append(float(np.linalg.norm(err)))
        if t >= config.burn_in_s - BURN_IN_TOLERANCE_S:
            covs.append((err, _stabilize_cov(p[:3, :3])))
    mask = queries >= config.burn_in_s - BURN_IN_TOLERANCE_S
    return _outcome(np.asarray(errors)[mask], covs, config, fused_count,
                    {"delivery_table": table, "dropped_stale": int(dropped)})


def _peer(cell, config, target, names, captures, measurements, table, mode):
    settings = EstimatorSettings()
    q = settings.q_acc_psd
    local_est = {s: _new_estimator(target, settings, cell.r_std_m) for s in names}
    fused_state = {s: _initial_state(target, settings) for s in names}
    latest_neighbor = {s: {} for s in names}
    latest_gs = {}
    gs_cached_state = None
    gs_cached_time = 0.0
    gs_dirty = True
    broadcasts = {s: {} for s in names}
    events = []
    for s in names:
        for k, capture in enumerate(captures[s]):
            events.append((float(capture), 0, int(s[1:]), s, k, None))
    for sender in names:
        for receiver in (*names, "gs"):
            if sender == receiver:
                continue
            for d in _route(table, sender, receiver):
                events.append((float(d.receive_time_s), 1, int(sender[1:]), sender, int(d.seq), receiver))
    events.sort(key=lambda x: (x[0], x[1], x[2], int(x[4]), str(x[5] or "")))

    def recompute_observer(receiver: str, at_time: float, include_senders=None) -> None:
        local_m, local_p = _state_from_estimator(local_est[receiver], at_time, settings)
        means = [local_m]
        covariances = [local_p]
        sender_names = latest_neighbor[receiver] if include_senders is None else {s for s in include_senders if s in latest_neighbor[receiver]}
        for sender in sorted(sender_names, key=lambda name: int(name[1:])):
            stamp, m0, p0 = latest_neighbor[receiver][sender]
            pm, pp = _predict_full(m0, p0, stamp, at_time, q)
            means.append(pm)
            covariances.append(pp)
        fm, fp = _fuse_tracks(means, covariances, mode=mode, ci_mode=cell.ci)
        fused_state[receiver] = (fm, fp, float(at_time))

    def recompute_gs(at_time: float):
        nonlocal gs_cached_state, gs_cached_time, gs_dirty
        if mode == "naive" and not gs_dirty:
            return _predict_full(gs_cached_state[0], gs_cached_state[1], gs_cached_time, at_time, q)
        if not latest_gs:
            initial = _initial_state(target, settings)
            result = _predict_full(initial[0], initial[1], initial[2], at_time, q)
            if mode == "naive":
                gs_cached_state = result
                gs_cached_time = float(at_time)
                gs_dirty = False
            return result
        means = []
        covariances = []
        for sender in sorted(latest_gs, key=lambda name: int(name[1:])):
            stamp, m0, p0 = latest_gs[sender]
            pm, pp = _predict_full(m0, p0, stamp, at_time, q)
            means.append(pm)
            covariances.append(pp)
        # Ground-station CI uses the closed-form trace weights for a stable,
        # reproducible receive-time estimate. Observer broadcasts honor cell.ci.
        result = _fuse_tracks(means, covariances, mode=mode, ci_mode="fast")
        if mode == "naive":
            gs_cached_state = result
            gs_cached_time = float(at_time)
            gs_dirty = False
        return result

    queries = _query_times(config)
    truths = np.asarray(target.position(queries), dtype=float)
    errors = []
    covs = []
    onboard_errors = []
    onboard_covs = []
    qcursor = 0
    ecursor = 0
    fused_count = 0
    dropped = 0
    diverged = False
    diverged_time = None
    while qcursor < queries.size:
        t = float(queries[qcursor])
        try:
            while ecursor < len(events) and events[ecursor][0] <= t + TIME_TOLERANCE_S:
                when, kind, _, sender, seq, receiver = events[ecursor]
                ecursor += 1
                if kind == 0:
                    measurement = np.asarray(measurements[sender][seq], dtype=float)
                    if measurement.shape != (3,) or not np.all(np.isfinite(measurement)):
                        raise FloatingPointError("non-finite measurement")
                    local_est[sender].update(measurement, when)
                    # Re-fuse local and every latest neighbor track.  In the
                    # naive arm this feeds the result back into the loop.
                    recompute_observer(sender, when, include_senders=None)
                    fm, fp, stamp = fused_state[sender]
                    broadcasts[sender][seq] = (float(stamp), fm.copy(), fp.copy())
                    continue
                incoming = broadcasts[sender].get(seq)
                if incoming is None:
                    dropped += 1
                    continue
                if receiver == "gs":
                    old = latest_gs.get(sender)
                    if old is not None and incoming[0] <= old[0] + TIME_TOLERANCE_S:
                        dropped += 1
                    else:
                        latest_gs[sender] = incoming
                        fused_count += 1
                        gs_dirty = True
                else:
                    old = latest_neighbor[receiver].get(sender)
                    if old is not None and incoming[0] <= old[0] + TIME_TOLERANCE_S:
                        dropped += 1
                    else:
                        latest_neighbor[receiver][sender] = incoming
                        recompute_observer(receiver, when)
                        fused_count += 1
            gm, gp = recompute_gs(t)
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError):
            diverged = True
            diverged_time = float(t)
            break
        gerr = gm[:3] - truths[qcursor]
        o1m, o1p, o1stamp = fused_state["o1"]
        om, op = _predict_full(o1m, o1p, o1stamp, t, q)
        oerr = om[:3] - truths[qcursor]
        errors.append(float(np.linalg.norm(gerr)))
        onboard_errors.append(float(np.linalg.norm(oerr)))
        if t >= config.burn_in_s - BURN_IN_TOLERANCE_S:
            covs.append((gerr, _stabilize_cov(gp[:3, :3])))
            onboard_covs.append((oerr, _stabilize_cov(op[:3, :3])))
        qcursor += 1
    mask = queries[:len(errors)] >= config.burn_in_s - BURN_IN_TOLERANCE_S
    extras = {"delivery_table": table, "dropped_stale": int(dropped)}
    if diverged:
        extras.update({"diverged": True, "diverged_time_s": diverged_time})
    result = _outcome(np.asarray(errors)[mask], covs, config, fused_count, extras)
    if diverged:
        result = CoopOutcome(result.p95_error_m, result.max_error_m, result.rmse_m,
                             True, result.mean_nees, result.nees_window_means,
                             result.frac_windows_above_band, result.fused_count,
                             result.extras)
    oe = np.asarray(onboard_errors, dtype=float)[mask] if onboard_errors else np.empty(0)
    result.extras["onboard_o1_p95_m"] = float(np.percentile(oe, 95)) if oe.size else None
    result.extras["onboard_o1_mean_nees"] = (
        float(np.mean([_nees_fast(e, c) for e, c in onboard_covs]))
        if oe.size and onboard_covs else None
    )
    return result


def run_seed(cell: CoopCell, seed: int, arms=ARMS, config: RecoveryConfig = RecoveryConfig()):
    """Run selected simulated cooperative tracking arms for one seed."""
    if isinstance(arms, (set, frozenset)):
        arms = tuple(sorted(arms))
    else:
        arms = tuple(arms)
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arm(s): {unknown}")
    names = tuple(f"o{i + 1}" for i in range(cell.n_observers))
    phases = _phase_offsets(cell, names, int(seed))
    target, names, captures, measurements = _truth_and_measurements(cell, int(seed), config, phases)
    specs = {
        "single": TrafficSpec("star", names, cell.rate_hz, 32, config.duration_s, phases),
        "central": TrafficSpec("star", names, cell.rate_hz, 32, config.duration_s, phases),
        "star_avg": TrafficSpec("star", names, cell.rate_hz, 108, config.duration_s, phases),
        "peer_ci": TrafficSpec("peer", names, cell.rate_hz, 216, config.duration_s, phases),
        "peer_naive": TrafficSpec("peer", names, cell.rate_hz, 216, config.duration_s, phases),
    }
    tables = {}
    out = {}
    for arm in arms:
        spec = specs[arm]
        key = (spec.pattern, spec.message_bytes)
        if key not in tables:
            tables[key] = make_deliveries(spec, cell.link, int(seed))
        table = tables[key]
        if arm == "single":
            out[arm] = _single_or_central(cell, config, target, names, captures, measurements, table)
        elif arm == "central":
            out[arm] = _single_or_central(cell, config, target, names, captures, measurements, table, True)
        elif arm == "star_avg":
            out[arm] = _star_average(cell, config, target, names, captures, measurements, table)
        elif arm == "peer_ci":
            out[arm] = _peer(cell, config, target, names, captures, measurements, table, "ci")
        else:
            out[arm] = _peer(cell, config, target, names, captures, measurements, table, "naive")
    return out


def run_cell(cell: CoopCell, seeds: Iterable[int], arms=ARMS, config: RecoveryConfig = RecoveryConfig()):
    if isinstance(arms, (set, frozenset)):
        arms = tuple(sorted(arms))
    else:
        arms = tuple(arms)
    seed_values = tuple(int(seed) for seed in seeds)
    results = {arm: [] for arm in arms}
    for seed in seed_values:
        outcomes = run_seed(cell, seed, arms=arms, config=config)
        for arm in arms:
            results[arm].append(outcomes[arm])
    return results


def gap_statistics(table, observers, receiver="gs"):
    """Count missing capture indices and return inter-arrival gap durations.

    Capture index ``k`` is the shared schedule index for each observer. The
    returned object exposes ``total`` missing indices and ``leading`` missing
    indices before the first successful arrival, while tuple iteration keeps
    the historical duration view.
    """
    observers = tuple(sorted((str(o) for o in observers), key=lambda name: int(name[1:]) if name[1:].isdigit() else name))
    routes = [table.get((sender, receiver), ()) for sender in observers]
    if not routes:
        return GapStatistics()
    n_steps = max((len(route) for route in routes), default=0)
    arrived = np.zeros(n_steps, dtype=bool)
    send_times = np.full(n_steps, np.nan)
    for route in routes:
        for d in route:
            k = int(d.seq)
            if k < 0 or k >= n_steps:
                continue
            send_times[k] = float(d.send_time_s)
            if d.receive_time_s is not None:
                arrived[k] = True
    missing = ~arrived
    total = int(np.sum(missing))
    leading = 0
    while leading < n_steps and missing[leading]:
        leading += 1
    good = np.flatnonzero(arrived)
    gaps = []
    if good.size >= 2:
        for left, right in zip(good[:-1], good[1:]):
            if right - left > 1:
                all_steps = np.diff(send_times[np.isfinite(send_times)])
                positive = all_steps[all_steps > TIME_TOLERANCE_S]
                step = float(np.min(positive)) if positive.size else 0.0
                if step > 0.0:
                    gaps.append(float((right - left - 1) * step))
    return GapStatistics(gaps, total=total, leading=leading)
