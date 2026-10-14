"""Experiment engine: one cell, many seeds, every recovery arm on the same packets.

A cell is one set of conditions: which target moves how, how often the sensor
samples, what the link does to packets (delay, jitter, loss), what the sensor
noise looks like, how wrong the sensor clock is, and which sender policy
decides what gets transmitted. ``deliver_cell`` turns a cell and a seed into
the packets that reach the tracker. ``run_arm`` feeds those packets to one
recovery method (an ``ArmSpec``) and scores it against the truth.
``run_cell`` repeats that over seeds, with every arm seeing the same packets
for a given seed (common random numbers), so differences between arms come
from the arms alone.

The naive and recovery-v1 arms reproduce the S6 recovery sweep (``sim/s6_recovery.py``)
exactly: same link and sensor draws, same order of operations, same counts.
On top of them this engine adds:

- retrodiction: a late packet inside a time window is put back in
  capture order by rewinding to a saved copy of the filter and re-fusing
  everything after it, instead of being dropped;
- a chi-square gate that rejects measurements whose normalized
  innovation is too large to be believable;
- the adaptive wrapper and a NIS consistency monitor;
- sensor clock errors (sim/timesync.py) and sender policies
  (sim/estimation/policies.py).

Everything is synthetic: simulated link, simulated sensor, nothing measured.

Speed: between two fusions the filter does not change, so every query time
in that stretch is answered with one vectorized ``predict_positions`` call
instead of one prediction per query.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, fields
from math import exp, log, sqrt

import numpy as np
from scipy.stats import chi2, norm

from sim import s6_recovery
from sim.link import LinkConfig, SimulatedLink
from sim.s6_recovery import RecoveryConfig, paired_summary, summarize_arm
from sim.sensor import SensorConfig, SensorModel

from .base import ArmOutcome, ArmSpec, Estimator, EstimatorSettings
from .factory import make_estimator

QUERY_DT_S = 0.02
TIME_TOLERANCE_S = s6_recovery.TIME_TOLERANCE_S
BURN_IN_TOLERANCE_S = s6_recovery.BURN_IN_TOLERANCE_S
LOSS_MODELS = ("independent", "burst")
NOISE_MODELS = ("gaussian", "student_t", "gaussian_outliers")
# Quantile of the one-way delay used as the retrodiction window.
RETRO_WINDOW_QUANTILE = 0.999
MONITOR_WINDOW = 10
ADAPTIVE_WINDOW = 10
# Extra random streams beside the S6 recovery sweep's link and sensor seeds. Each one has
# its own SeedSequence([seed, k]) so turning a feature on never shifts the
# draws of another.
EXTRA_STREAMS = {"target": 1, "clock": 2, "sync_up": 3, "sync_down": 4, "outliers": 5, "sender": 6}
# Bytes per packet when every sample is sent as float32: seq 2 B + timestamp
# 4 B + position 12 B. The 43 B ESP-NOW overhead is literature-derived,
# not measured. Must equal policies.packet_bytes(False).
DEFAULT_DATA_BYTES = 18
DEFAULT_ON_AIR_BYTES = 18 + 43
MOTION_TARGETS = {"constant_velocity": "cv", "constant_turn": "ct"}


def _pairs(value, name: str) -> tuple:
    """Normalize a dict or a sequence of (key, value) pairs to sorted pairs."""
    items = value.items() if isinstance(value, dict) else value
    try:
        pairs = tuple(sorted((str(k), v) for k, v in items))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a dict or (key, value) pairs") from exc
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError(f"{name} has a repeated key")
    return pairs


def _fmt(value) -> str:
    """Shortest exact text for a number, so different values never share a key."""
    if isinstance(value, (bool, np.bool_)) or value is None or isinstance(value, str):
        return str(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.format_float_positional(float(value), trim="-")
    return str(value)


def _fmt_pairs(pairs: tuple) -> str:
    return ",".join(f"{k}={_fmt(v)}" for k, v in pairs)


@dataclass(frozen=True)
class CellSpec:
    """One set of simulated conditions. Tuple fields hold sorted (key, value) pairs."""

    target: str = "cv"
    target_params: tuple = ()
    rate_hz: float = 10.0
    delay_s: float = 0.0
    jitter_s: float = 0.0
    loss_probability: float = 0.0
    loss_model: str = "independent"
    burst_mean_length_packets: float = 5.0
    noise_model: str = "gaussian"
    noise_std_m: float = 0.05
    noise_dof: float = 3.0
    outlier_probability: float = 0.0
    outlier_m: float = 1.0
    clock: tuple = ()
    sender: tuple = ()

    def __post_init__(self) -> None:
        for name in ("target_params", "clock", "sender"):
            object.__setattr__(self, name, _pairs(getattr(self, name), name))
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("target must be a non-empty name")
        for name in ("rate_hz", "delay_s", "jitter_s", "loss_probability", "burst_mean_length_packets",
                     "noise_std_m", "noise_dof", "outlier_probability", "outlier_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        if self.jitter_s > 0 and self.delay_s <= 0:
            raise ValueError("jitter_s needs a positive delay_s (the delay is lognormal around it)")
        if self.loss_probability > 1 or self.outlier_probability > 1:
            raise ValueError("probabilities must be at most one")
        if self.loss_model not in LOSS_MODELS:
            raise ValueError(f"loss_model must be one of {LOSS_MODELS}")
        if self.burst_mean_length_packets < 1:
            raise ValueError("burst_mean_length_packets must be at least one")
        if self.noise_model not in NOISE_MODELS:
            raise ValueError(f"noise_model must be one of {NOISE_MODELS}")
        if self.noise_model == "student_t" and self.noise_dof <= 2:
            raise ValueError("student_t noise needs noise_dof > 2 so its variance is finite")

    def key(self) -> str:
        """Stable, readable, unique name. Optional fields appear only when set."""
        target = self.target + (f"({_fmt_pairs(self.target_params)})" if self.target_params else "")
        parts = [target, f"rate{_fmt(self.rate_hz)}", f"d{_fmt(self.delay_s)}", f"j{_fmt(self.jitter_s)}",
                 f"l{_fmt(self.loss_probability)}-{self.loss_model}"]
        defaults = CellSpec.__dataclass_fields__
        if self.burst_mean_length_packets != defaults["burst_mean_length_packets"].default:
            parts.append(f"burst{_fmt(self.burst_mean_length_packets)}")
        parts.append(f"{self.noise_model}{_fmt(self.noise_std_m)}")
        if self.noise_dof != defaults["noise_dof"].default:
            parts.append(f"dof{_fmt(self.noise_dof)}")
        if self.outlier_probability != defaults["outlier_probability"].default:
            parts.append(f"outp{_fmt(self.outlier_probability)}")
        if self.outlier_m != defaults["outlier_m"].default:
            parts.append(f"outm{_fmt(self.outlier_m)}")
        if self.clock:
            parts.append(f"clock({_fmt_pairs(self.clock)})")
        if self.sender:
            parts.append(f"sender({_fmt_pairs(self.sender)})")
        return "/".join(parts)

    def as_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        for name in ("target_params", "clock", "sender"):
            out[name] = dict(out[name])
        return out


def s6_cell(motion: str, delay_s: float, loss_probability: float, noise_std_m: float,
                loss_model: str = "independent", jitter_s: float = 0.0) -> CellSpec:
    """The CellSpec for one S6 recovery cell. The oracle slice's R = 0.10 m is an
    arm setting (ArmSpec.r_std_m), not a cell setting."""
    if motion not in MOTION_TARGETS:
        raise ValueError(f"motion must be one of {tuple(MOTION_TARGETS)}")
    return CellSpec(target=MOTION_TARGETS[motion], delay_s=float(delay_s), jitter_s=float(jitter_s),
                    loss_probability=float(loss_probability), loss_model=loss_model,
                    noise_std_m=float(noise_std_m))


def retro_window_for(cell: CellSpec) -> float:
    """Retrodiction window: 0 with a fixed delay, otherwise the 0.999
    quantile of the lognormal one-way delay with mean d and std j."""
    if cell.jitter_s == 0:
        return 0.0
    d, j = float(cell.delay_s), float(cell.jitter_s)
    sigma2 = log(1.0 + j * j / (d * d))
    mu = log(d) - sigma2 / 2.0
    return float(exp(mu + sqrt(sigma2) * norm.ppf(RETRO_WINDOW_QUANTILE)))


def seed_streams(seed: int) -> dict:
    """All per-run seeds. link_seed and sensor_seed are EXACTLY the S6 recovery sweep's."""
    link_seed, sensor_seed = (int(v) for v in np.random.SeedSequence(seed).generate_state(2))
    streams = {"link_seed": link_seed, "sensor_seed": sensor_seed}
    for name, k in EXTRA_STREAMS.items():
        streams[name] = int(np.random.SeedSequence([int(seed), k]).generate_state(1)[0])
    return streams


@dataclass(frozen=True)
class Delivery:
    """One packet that reached the tracker. ``stamp_s`` is the capture time as
    the tracker reads it (after any clock error)."""

    seq: int
    capture_time_s: float
    stamp_s: float
    receive_time_s: float
    measurement_m: tuple


@dataclass(frozen=True)
class Bundle:
    """Everything one seed of one cell produced before any arm runs."""

    deliveries: list
    target: object
    capture_times_s: np.ndarray
    capture_count: int
    sent_count: int
    data_bytes_per_packet: int
    on_air_bytes_per_packet: int
    data_bytes: int
    outlier_seqs: frozenset | None = None
    stamp_errors_s: np.ndarray | None = None
    sync_records: object = None


def _link_config(cell: CellSpec) -> LinkConfig:
    if cell.loss_model == "burst":
        end = 1.0 / float(cell.burst_mean_length_packets)
        start = s6_recovery._burst_start_probability(float(cell.loss_probability), end)
        return LinkConfig(delay_s=float(cell.delay_s), delay_jitter_s=float(cell.jitter_s), loss_model="burst",
                          burst_start_probability=start, burst_end_probability=end)
    return LinkConfig(delay_s=float(cell.delay_s), delay_jitter_s=float(cell.jitter_s), loss_model="independent",
                      loss_probability=float(cell.loss_probability))


def _measure(cell: CellSpec, truth: np.ndarray, capture_times: np.ndarray,
             streams: dict) -> tuple[np.ndarray, frozenset | None]:
    """Sensor measurements, one row per capture, plus the outlier seqs if any."""
    n = truth.shape[0]
    out = np.empty((n, 3))
    if cell.noise_model == "student_t":
        # Scaled so the per-axis std is noise_std_m: a t variable has variance dof / (dof - 2).
        rng = np.random.default_rng(streams["sensor_seed"])
        scale = cell.noise_std_m * sqrt((cell.noise_dof - 2.0) / cell.noise_dof)
        for k in range(n):
            out[k] = truth[k] + scale * rng.standard_t(cell.noise_dof, size=3)
        return out, None
    sensor = SensorModel(SensorConfig(noise_std_m=float(cell.noise_std_m)), seed=streams["sensor_seed"])
    for k in range(n):
        out[k] = sensor.sample(truth[k], float(capture_times[k])).measurement_m
    if cell.noise_model == "gaussian":
        return out, None
    # Every capture draws its coin and its direction whether or not it becomes
    # an outlier, so changing the outlier probability never shifts the stream.
    rng = np.random.default_rng(streams["outliers"])
    outliers = set()
    for k in range(n):
        u = rng.random()
        direction = rng.normal(size=3)
        if u < cell.outlier_probability:
            out[k] = out[k] + cell.outlier_m * direction / np.linalg.norm(direction)
            outliers.add(k)
    return out, frozenset(outliers)


def deliver_cell(cell: CellSpec, seed: int, config: RecoveryConfig = RecoveryConfig(),
                 settings: EstimatorSettings = EstimatorSettings()) -> Bundle:
    """Capture, (optionally) select and send, and deliver one seed of one cell.

    The Gaussian path is sim.s6_recovery.deliver step for step: the same
    link and SensorModel seeds and the same draws in the same order. Captures
    are at k / rate_hz for k = 0..round(duration x rate_hz). Packets arriving
    after the end of the run are not delivered. Deliveries are sorted by
    (receive time, seq).
    """
    from .targets import make_target

    streams = seed_streams(seed)
    target = make_target(cell.target, dict(cell.target_params), seed=streams["target"])
    period = 1.0 / float(cell.rate_hz)
    n_captures = int(round(config.duration_s * cell.rate_hz))
    capture_times = np.arange(n_captures + 1) * period
    truth = np.asarray(target.position(capture_times), dtype=float)
    measurements, outlier_seqs = _measure(cell, truth, capture_times, streams)

    stamp_errors = None
    stamps = capture_times
    if cell.clock:
        from sim.timesync import stamp_errors as clock_stamp_errors
        stamp_errors = np.asarray(clock_stamp_errors(capture_times, dict(cell.clock), streams), dtype=float)
        stamps = capture_times + stamp_errors

    if cell.sender:
        from .policies import apply_sender
        sent = apply_sender(capture_times, measurements, dict(cell.sender), settings, streams["sender"])
        send_indices = [int(i) for i in sent.send_indices]
        sent_measurements = np.asarray(sent.sent_measurements, dtype=float)
        data_per_packet = int(sent.data_bytes_per_packet)
        on_air_per_packet = int(sent.on_air_bytes_per_packet)
    else:
        send_indices = list(range(capture_times.size))
        sent_measurements = measurements
        data_per_packet, on_air_per_packet = DEFAULT_DATA_BYTES, DEFAULT_ON_AIR_BYTES

    link = SimulatedLink(_link_config(cell), seed=streams["link_seed"])
    sent_value = {}
    for row, seq in enumerate(send_indices):
        link.send("target", "tracker", seq, float(capture_times[seq]))
        sent_value[seq] = tuple(float(v) for v in sent_measurements[row])
    deliveries = [
        Delivery(event.seq, float(event.send_time), float(stamps[event.seq]), float(event.receive_time),
                 sent_value[event.seq])
        for event in link.deliver_all()
        if event.receive_time <= config.duration_s + TIME_TOLERANCE_S
    ]
    deliveries.sort(key=lambda d: (d.receive_time_s, d.seq))
    return Bundle(
        deliveries=deliveries, target=target, capture_times_s=capture_times, capture_count=int(capture_times.size),
        sent_count=len(send_indices), data_bytes_per_packet=data_per_packet,
        on_air_bytes_per_packet=on_air_per_packet, data_bytes=data_per_packet * len(send_indices),
        outlier_seqs=outlier_seqs, stamp_errors_s=stamp_errors,
    )


@dataclass
class ArmRun:
    """run_arm's outcome plus the final filter and every packet in it (for checks)."""

    outcome: ArmOutcome
    estimator: Estimator
    fused: list = field(default_factory=list)


def _inner(est: Estimator, adaptive: bool) -> Estimator:
    """The model estimator under an adaptive wrapper (or the estimator itself)."""
    return est.base if adaptive else est


class _ArmState:
    """Mutable bookkeeping for one arm on one bundle."""

    def __init__(self, est, query_times, scored, is_imm, adaptive):
        self.est = est
        self.adaptive = adaptive
        self.query_times = query_times
        self.scored = scored
        self.positions = np.empty((query_times.size, 3))
        self.pending = 0
        self.is_imm = is_imm
        self.ct_prob_sum = 0.0
        self.nis_values: list[float] = []
        self.nis_stamps: list[float] = []
        self.fused: list[Delivery] = []
        self.fused_count = self.dropped = self.out_of_order = self.gated = self.refiltered = 0
        self.last_capture = -np.inf
        self.newest_stamp = -np.inf  # newest stamp fused so far (capture stamping)
        self.gate_counts = {"gated_outliers": 0, "gated_clean": 0,
                            "gate_checked_outliers": 0, "gate_checked_clean": 0}
        self.gated_in_a_row = 0
        self.gate_forced = 0

    def flush(self, upto: int) -> None:
        """Answer queries pending..upto-1 with the current filter (it is about to change)."""
        if upto <= self.pending:
            return
        times = self.query_times[self.pending:upto]
        est = self.est
        self.positions[self.pending:upto] = est.predict_positions(np.maximum(times, est.timestamp_s))
        if self.is_imm:
            n_scored = int(np.count_nonzero(self.scored[self.pending:upto]))
            self.ct_prob_sum += n_scored * float(_inner(est, self.adaptive).mode_probabilities[1])
        self.pending = upto


def run_arm_detailed(bundle: Bundle, cell: CellSpec, arm: ArmSpec, config: RecoveryConfig = RecoveryConfig(),
                     settings: EstimatorSettings = EstimatorSettings()) -> ArmRun:
    """run_arm, also returning the final estimator and the packets it holds."""
    retro = arm.late_policy == "retrodict"
    if retro and arm.stamping != "capture":
        raise ValueError("retrodiction needs capture stamping (receive stamps never arrive late)")
    window = None
    if retro:
        window = retro_window_for(cell) if arm.retro_window_s is None else float(arm.retro_window_s)
        if not np.isfinite(window) or window < 0:
            raise ValueError("retro_window_s must be finite and nonnegative")
    gate = None if arm.gate_probability is None else float(chi2.ppf(arm.gate_probability, 3))
    outliers = bundle.outlier_seqs
    burn_in = config.burn_in_s - BURN_IN_TOLERANCE_S

    est = make_estimator(arm.model, settings, bundle.target.start_position, q_factor=arm.q_factor,
                         r_std_m=arm.r_std_m, model_options=arm.model_options)
    if arm.adaptive is not None:
        from .consistency import AdaptiveEstimator
        est = AdaptiveEstimator(est, mode=arm.adaptive, window=ADAPTIVE_WINDOW)

    query_times = np.arange(0.0, config.duration_s + QUERY_DT_S / 2, QUERY_DT_S)
    truth = np.asarray(bundle.target.position(query_times), dtype=float)
    scored = query_times >= burn_in
    state = _ArmState(est, query_times, scored, arm.model == "imm", arm.adaptive is not None)
    deliveries = bundle.deliveries
    # A packet is handled at the first query time t with receive <= t + tol,
    # the exact comparison sim.s6_recovery.run_arms makes.
    ready_at = np.searchsorted(query_times + TIME_TOLERANCE_S,
                               np.array([d.receive_time_s for d in deliveries], dtype=float), side="left")
    history: list[list] = []  # retrodiction entries: [stamp, z, filter_before, delivery]

    def screen(filt: Estimator, z: np.ndarray, stamp: float, packet: Delivery) -> bool:
        """Gate and NIS bookkeeping before a fusion. False means the gate rejected it."""
        nis = filt.nis(z, stamp) if (gate is not None or stamp >= burn_in) else None
        if gate is not None:
            is_outlier = outliers is not None and packet.seq in outliers
            if outliers is not None:
                state.gate_counts["gate_checked_outliers" if is_outlier else "gate_checked_clean"] += 1
            if nis > gate:
                if arm.gate_reset_after and state.gated_in_a_row >= arm.gate_reset_after:
                    # Lockout guard: too many rejections in a row means the
                    # filter, not the packets, is probably wrong. Fuse this one.
                    state.gate_forced += 1
                else:
                    state.gated += 1
                    state.gated_in_a_row += 1
                    if outliers is not None:
                        state.gate_counts["gated_outliers" if is_outlier else "gated_clean"] += 1
                    return False
            state.gated_in_a_row = 0
        if stamp >= burn_in:
            state.nis_values.append(nis)
            state.nis_stamps.append(stamp)
        return True

    def fuse_newest(z: np.ndarray, stamp: float, packet: Delivery, qi: int) -> None:
        if not screen(state.est, z, stamp, packet):
            return
        if packet.capture_time_s < state.last_capture:
            state.out_of_order += 1
        state.flush(qi)
        if retro:
            history.append([stamp, z, state.est.copy(), packet])
        state.est.update(z, stamp)
        state.last_capture = max(state.last_capture, packet.capture_time_s)
        state.fused_count += 1
        state.fused.append(packet)

    def retrodict(z: np.ndarray, stamp: float, packet: Delivery, qi: int) -> None:
        stamps = [entry[0] for entry in history]
        i = bisect_right(stamps, stamp)
        if i > 0 and stamps[i - 1] == stamp:
            state.dropped += 1  # duplicate of a fused stamp
            return
        snapshot = history[i][2]
        # A fresh filter may fuse at its own start time (the packet captured at
        # t = 0); after that only strictly newer stamps can go in.
        fresh = snapshot.update_count == 0 and stamp == snapshot.timestamp_s
        if not (stamp > snapshot.timestamp_s or fresh):
            state.dropped += 1  # older than the oldest filter copy still kept
            return
        if not screen(snapshot, z, stamp, packet):
            return
        state.flush(qi)
        filt = snapshot.copy()
        filt.update(z, stamp)
        rebuilt = [[stamp, z, snapshot, packet]]
        for entry in history[i:]:
            before = filt.copy()
            filt.update(entry[1], entry[0])
            rebuilt.append([entry[0], entry[1], before, entry[3]])
        history[i:] = rebuilt
        state.est = filt
        state.last_capture = max(state.last_capture, packet.capture_time_s)
        state.fused_count += 1
        state.refiltered += 1
        state.fused.append(packet)

    def prune() -> None:
        newest = history[-1][0]
        keep_from = 0
        while keep_from < len(history) - 1 and history[keep_from][0] < newest - window:
            keep_from += 1
        if keep_from:
            del history[:keep_from]

    for packet, qi in zip(deliveries, ready_at):
        qi = int(qi)
        if qi >= query_times.size:
            break  # arrives after the last query, never used (as in the S6 recovery sweep)
        z = np.asarray(packet.measurement_m, dtype=float)
        if arm.stamping == "receive":
            # The naive arm treats the packet as current when it arrives.
            stamp = packet.receive_time_s
            est = state.est
            if est.update_count > 0 and abs(stamp - est.timestamp_s) <= TIME_TOLERANCE_S:
                state.dropped += 1
                continue
            fuse_newest(z, stamp, packet, qi)
            continue
        stamp = packet.stamp_s
        newest = state.newest_stamp
        if stamp > newest:
            if stamp < state.est.timestamp_s:
                state.dropped += 1  # stamped before the filter start (clock error)
                continue
            before = state.fused_count
            fuse_newest(z, stamp, packet, qi)
            if state.fused_count > before:
                state.newest_stamp = stamp
                if retro:
                    prune()
        elif retro and stamp >= newest - window:
            before = state.fused_count
            retrodict(z, stamp, packet, qi)
            if state.fused_count > before:
                prune()
        else:
            state.dropped += 1  # late (or a duplicate) and not recoverable
    state.flush(query_times.size)

    errors = np.linalg.norm(state.positions - truth, axis=1)
    window_errors = errors[scored]
    p95 = float(np.percentile(window_errors, 95))
    maximum = float(np.max(window_errors))
    extras = _extras(state, arm, outliers is not None and gate is not None, int(np.count_nonzero(scored)))
    outcome = ArmOutcome(
        p95_error_m=p95,
        max_error_m=maximum,
        rmse_m=float(np.sqrt(np.mean(window_errors**2))),
        fused_count=state.fused_count,
        dropped_count=state.dropped,
        fused_out_of_order_count=state.out_of_order,
        mean_nis=float(np.mean(state.nis_values)) if state.nis_values else None,
        failed=bool(p95 > config.failure_p95_m or maximum > config.failure_max_m),
        gated_count=state.gated,
        refiltered_count=state.refiltered,
        retro_window_s=window,
        extras=extras,
        query_times_s=query_times,
        errors_m=errors,
    )
    return ArmRun(outcome, state.est, state.fused)


def _extras(state: _ArmState, arm: ArmSpec, gate_knows_outliers: bool, n_scored: int) -> dict:
    from .consistency import ConsistencyMonitor

    monitor = ConsistencyMonitor(window=MONITOR_WINDOW)
    for value in state.nis_values:
        monitor.add(value)
    n_windows = len(monitor.window_means)
    inner = _inner(state.est, state.adaptive)
    extras = {
        "monitor_warn_fraction": len(monitor.warnings) / n_windows if n_windows else None,
        # Capture stamp of the last NIS in the first flagged window: when an operator
        # watching the monitor would first have been warned.
        "monitor_first_warning_s": (state.nis_stamps[(monitor.warnings[0][0] + 1) * MONITOR_WINDOW - 1]
                                    if monitor.warnings else None),
        "final_q_scale": float(inner.q_scale),
        "final_r_scale": float(inner.r_scale),
    }
    if arm.model == "imm":
        extras["imm_ct_prob_mean"] = state.ct_prob_sum / n_scored if n_scored else None
    if arm.gate_probability is not None:
        extras["gate_forced_accepts"] = state.gate_forced
    if gate_knows_outliers:
        extras.update(state.gate_counts)
    return extras


def run_arm(bundle: Bundle, cell: CellSpec, arm: ArmSpec, config: RecoveryConfig = RecoveryConfig(),
            settings: EstimatorSettings = EstimatorSettings()) -> ArmOutcome:
    """Score one arm on one delivered bundle.

    Queries every 0.02 s from 0 to the end; the scored window starts at the
    burn-in. Each packet is handled at the first query time at or after its
    arrival. naive and recovery-v1 match sim.s6_recovery.run_arms exactly.
    fused_count includes refiltered packets; dropped_count is every packet
    not fused except the gated ones (gated_count).
    """
    return run_arm_detailed(bundle, cell, arm, config, settings).outcome


PER_SEED_FIELDS = ("p95_error_m", "max_error_m", "rmse_m", "mean_nis", "fused_count", "dropped_count",
                   "fused_out_of_order_count", "gated_count", "refiltered_count", "failed")


def run_cell(cell: CellSpec, arms, seeds, config: RecoveryConfig = RecoveryConfig(),
             settings: EstimatorSettings = EstimatorSettings()) -> dict:
    """Every arm on every seed of one cell, with one delivery per seed shared by all arms.

    paired_vs_naive[name] and paired_vs_recovery[name] use
    sim.s6_recovery.paired_summary(reference, arm): a positive difference
    (counted in recovery_better_count) means the arm beat the reference.
    """
    arms = list(arms)
    names = [arm.name for arm in arms]
    if not arms or len(set(names)) != len(names):
        raise ValueError("arms must be a non-empty list with unique names")
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("seeds must not be empty")
    results: dict[str, list[ArmOutcome]] = {name: [] for name in names}
    key = cell.key()
    rows = []
    for index, seed in enumerate(seeds):
        bundle = deliver_cell(cell, seed, config, settings)
        for arm in arms:
            outcome = run_arm(bundle, cell, arm, config, settings)
            outcome.query_times_s = None
            outcome.errors_m = None
            results[arm.name].append(outcome)
            row = {"cell_key": key, "arm": arm.name, "seed_index": index, "seed": seed}
            for name in PER_SEED_FIELDS:
                row[name] = getattr(outcome, name)
            for extra, value in sorted(outcome.extras.items()):
                row[f"x_{extra}"] = value
            rows.append(row)
    out = {
        "cell": cell.as_dict(),
        "key": key,
        "arms": {name: summarize_arm(results[name]) for name in names},
        "paired_vs_naive": {},
        "paired_vs_recovery": {},
        "per_seed": rows,
    }
    for reference, slot in (("naive", "paired_vs_naive"), ("recovery", "paired_vs_recovery")):
        if reference in results:
            out[slot] = {name: paired_summary(results[reference], results[name])
                         for name in names if name != reference}
    return out
