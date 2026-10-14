"""Shared interface for every estimator and arm.

An estimator holds a filter whose clock only moves forward. ``update`` fuses
one 3D position measurement at a time stamp at or after the filter's own time.
``predict_at`` answers "where is the target at time t" without changing the
filter, which is the predict-forward step from S6. ``innovation`` gives the
gap between a measurement and the prediction, plus how big that gap is
expected to be, for the NIS checks and the chi-square gate.

``q_scale`` and ``r_scale`` multiply the process noise and the measurement
noise used from the next step on. They start at 1 and are only moved by the
mis-tuning experiments and the adaptive wrapper.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

MODELS = ("cv", "ca", "ct", "imm")
STAMPINGS = ("receive", "capture")
LATE_POLICIES = ("drop", "retrodict")
ADAPTIVE_MODES = (None, "q", "r")


@dataclass(frozen=True)
class Prediction:
    """Predicted position, velocity and position covariance at one time."""

    timestamp_s: float
    position_m: np.ndarray
    velocity_mps: np.ndarray
    position_cov: np.ndarray


class Estimator(ABC):
    """Common interface. Subclasses keep ``timestamp_s`` and ``update_count`` current."""

    timestamp_s: float
    update_count: int
    q_scale: float = 1.0
    r_scale: float = 1.0

    @abstractmethod
    def update(self, measurement_m, timestamp_s: float) -> None:
        """Fuse one measurement. Raise ValueError if the time is earlier than
        ``timestamp_s``, or equal to it after the first update (a duplicate)."""

    @abstractmethod
    def predict_at(self, timestamp_s: float) -> Prediction:
        """Prediction at a time at or after ``timestamp_s``. Never changes the filter."""

    @abstractmethod
    def measurement_cov(self) -> np.ndarray:
        """The 3x3 measurement covariance used by the next update (already scaled)."""

    def innovation(self, measurement_m, timestamp_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Innovation nu = z - predicted position and its covariance S = P_pos + R.

        The same definition is used for every model, the IMM included, so the
        NIS and the gate mean the same thing in every arm.
        """
        pred = self.predict_at(timestamp_s)
        nu = np.asarray(measurement_m, dtype=float) - pred.position_m
        return nu, pred.position_cov + self.measurement_cov()

    def nis(self, measurement_m, timestamp_s: float) -> float:
        """Normalized innovation squared nu' S^-1 nu (about 3 on average when well tuned)."""
        nu, S = self.innovation(measurement_m, timestamp_s)
        return float(nu @ np.linalg.solve(S, nu))

    def predict_positions(self, times_s) -> np.ndarray:
        """Predicted positions at several times, shape (N, 3). Subclasses may vectorize,
        but must match ``predict_at`` to 1e-12."""
        times = np.asarray(times_s, dtype=float)
        out = np.empty((times.size, 3))
        for i, t in enumerate(times):
            out[i] = self.predict_at(float(t)).position_m
        return out

    def copy(self) -> "Estimator":
        return copy.deepcopy(self)


@dataclass(frozen=True)
class EstimatorSettings:
    """Tuning shared by the model factories. Defaults are the S6 recovery values.

    q_acc_psd: white-acceleration intensity (m^2/s^3) for CV, CT and the IMM.
    q_jerk_psd: white-jerk intensity (m^2/s^5) for CA.
    q_turn_psd: turn-rate random-walk intensity ((rad/s)^2/s) for CT and the IMM.
    """

    measurement_std_m: float = 0.05
    q_acc_psd: float = 0.1
    q_jerk_psd: float = 0.1
    q_turn_psd: float = 0.1
    initial_position_variance_m2: float = 1.0
    initial_velocity_variance_m2ps2: float = 1.0
    initial_acceleration_variance: float = 1.0
    initial_turn_rate_variance: float = 1.0
    imm_mean_sojourn_s: float = 2.0
    imm_initial_ct_probability: float = 0.5


@dataclass(frozen=True)
class ArmSpec:
    """One recovery method. naive and recovery-v1 are the S6 recovery arms.

    stamping: "receive" fuses a packet stamped with its arrival time (naive);
      "capture" uses the capture stamp as the tracker sees it (after any
      clock error) and predicts forward at query time.
    late_policy: "drop" discards a packet that is not newer than the newest
      fused one (as in the S6 recovery sweep); "retrodict" re-runs the filter from the late
      packet's capture time inside a window of ``retro_window_s`` seconds
      (None = chosen per cell from the delay tail, see engine).
    gate_probability: if set, a measurement whose NIS exceeds the chi-square
      quantile with 3 degrees of freedom at this probability is rejected.
    gate_reset_after: lockout guard for the gate. After this many rejections
      in a row the next measurement is fused anyway (counted as a forced
      accept), so one bad fusion early on cannot blind the filter for the
      rest of the run. 0 turns the guard off (a pure gate).
    adaptive: None, "q" or "r" (see sim/estimation/consistency.py).
    r_std_m: the tracker's measurement std (None = settings default 0.05 m).
    q_factor: multiplies every process-noise intensity (mis-tuning sweeps).
    model_options: extra per-model options as sorted (key, value) pairs, for
      example (("known_turn_rate", 1.0),) for the CT exact-model check.
    """

    name: str
    model: str = "cv"
    stamping: str = "capture"
    late_policy: str = "drop"
    retro_window_s: float | None = None
    gate_probability: float | None = None
    gate_reset_after: int = 3
    adaptive: str | None = None
    r_std_m: float | None = None
    q_factor: float = 1.0
    model_options: tuple = ()

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if self.stamping not in STAMPINGS:
            raise ValueError(f"stamping must be one of {STAMPINGS}")
        if self.late_policy not in LATE_POLICIES:
            raise ValueError(f"late_policy must be one of {LATE_POLICIES}")
        if self.adaptive not in ADAPTIVE_MODES:
            raise ValueError(f"adaptive must be one of {ADAPTIVE_MODES}")
        if self.gate_probability is not None and not 0.0 < self.gate_probability < 1.0:
            raise ValueError("gate_probability must be in (0, 1)")
        if int(self.gate_reset_after) != self.gate_reset_after or self.gate_reset_after < 0:
            raise ValueError("gate_reset_after must be a nonnegative integer")
        if not np.isfinite(self.q_factor) or self.q_factor <= 0:
            raise ValueError("q_factor must be positive")

    def as_dict(self) -> dict:
        return {
            "name": self.name, "model": self.model, "stamping": self.stamping,
            "late_policy": self.late_policy, "retro_window_s": self.retro_window_s,
            "gate_probability": self.gate_probability,
            "gate_reset_after": self.gate_reset_after, "adaptive": self.adaptive,
            "r_std_m": self.r_std_m, "q_factor": self.q_factor,
            "model_options": [list(kv) for kv in self.model_options],
        }


NAIVE = ArmSpec("naive", model="cv", stamping="receive", late_policy="drop")
RECOVERY_V1 = ArmSpec("recovery", model="cv", stamping="capture", late_policy="drop")


@dataclass
class ArmOutcome:
    """Per-seed result of one arm. Field names match s6_recovery.ArmResult so
    ``summarize_arm`` and ``paired_summary`` from sim/s6_recovery.py work on it."""

    p95_error_m: float
    max_error_m: float
    rmse_m: float
    fused_count: int
    dropped_count: int
    fused_out_of_order_count: int
    mean_nis: float | None
    failed: bool
    gated_count: int = 0
    refiltered_count: int = 0
    retro_window_s: float | None = None
    extras: dict = field(default_factory=dict)
    query_times_s: np.ndarray | None = None
    errors_m: np.ndarray | None = None
