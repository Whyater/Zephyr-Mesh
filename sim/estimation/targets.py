"""Truth motions for the estimation study (simulation only).

A target is a known path the tracked object follows. Each one answers two
questions for any list of times: where is it (``position``) and how fast is
it moving (``velocity``). They all start at the origin.

- ``cv`` and ``ct`` are the two S6 motions (straight line, circle) and match
  ``sim.s6_recovery.truth_position`` bit for bit.
- ``jink`` is a constant-speed path whose turn rate wanders randomly (an
  Ornstein-Uhlenbeck process), so the object keeps changing how hard it turns.
- ``figure8``, ``start_stop`` and ``mixed`` are closed-form paths with smooth
  or piecewise-smooth acceleration.
- ``ca_line`` and ``cwna`` exist only for checking estimators: a perfectly
  constant acceleration, and a path drawn from the exact constant-velocity
  model with white acceleration noise.

Random targets (``jink``, ``cwna``) are built once on a fixed grid from a seed
and cached, so the positions never depend on which times are asked for.
"""
from __future__ import annotations

import math

import numpy as np

from sim.estimation.kinematics import SMALL_TURN

# Random targets are generated up to at least this long, so a short request
# and a long request see the same path.
MIN_GENERATED_S = 12.0
JINK_GRID_S = 0.001

DEFAULTS: dict[str, dict[str, float]] = {
    "cv": {"speed_mps": 1.0},
    "ct": {"speed_mps": 1.0, "radius_m": 1.0},
    "jink": {"speed_mps": 1.0, "turn_rate_std_radps": 1.0, "correlation_time_s": 1.0},
    "figure8": {"amplitude_x_m": 1.5, "amplitude_y_m": 0.75, "omega_radps": 0.5},
    "start_stop": {"cruise_speed_mps": 1.0, "accel_mps2": 2.0, "cruise_s": 1.5, "stop_s": 1.0},
    "mixed": {"speed_mps": 1.0, "radius_m": 1.0, "straight1_s": 4.0, "turn_angle_rad": math.pi},
    "ca_line": {"accel_mps2": 1.0},
    "cwna": {"q_acc_psd": 0.1, "sample_period_s": 0.1},
}
# Parameters that may be zero; every other parameter must be strictly positive.
NONNEGATIVE = {"stop_s", "straight1_s", "turn_angle_rad", "cruise_s"}
TARGET_NAMES = tuple(DEFAULTS)


def _times(times_s) -> np.ndarray:
    t = np.asarray(times_s, dtype=float)
    if t.ndim == 0:
        t = t.reshape(1)
    if t.ndim != 1:
        raise ValueError("times must be a scalar or a 1D sequence")
    if t.size and (not np.all(np.isfinite(t)) or np.any(t < 0.0)):
        raise ValueError("times must be finite and nonnegative")
    return t


def _cv_position(t, p):
    out = np.zeros((t.size, 3))
    out[:, 0] = p["speed_mps"] * t
    return out


def _cv_velocity(t, p):
    out = np.zeros((t.size, 3))
    out[:, 0] = p["speed_mps"]
    return out


def _ct_position(t, p):
    r = p["radius_m"]
    w = p["speed_mps"] / r
    out = np.zeros((t.size, 3))
    out[:, 0] = r * np.sin(w * t)
    out[:, 1] = r * (1.0 - np.cos(w * t))
    return out


def _ct_velocity(t, p):
    v = p["speed_mps"]
    w = v / p["radius_m"]
    out = np.zeros((t.size, 3))
    out[:, 0] = v * np.cos(w * t)
    out[:, 1] = v * np.sin(w * t)
    return out


def _figure8_position(t, p):
    W = p["omega_radps"]
    out = np.zeros((t.size, 3))
    out[:, 0] = p["amplitude_x_m"] * np.sin(W * t)
    out[:, 1] = p["amplitude_y_m"] * np.sin(2.0 * W * t)
    return out


def _figure8_velocity(t, p):
    W = p["omega_radps"]
    out = np.zeros((t.size, 3))
    out[:, 0] = p["amplitude_x_m"] * W * np.cos(W * t)
    out[:, 1] = 2.0 * p["amplitude_y_m"] * W * np.cos(2.0 * W * t)
    return out


def _ca_position(t, p):
    out = np.zeros((t.size, 3))
    out[:, 0] = 0.5 * p["accel_mps2"] * t**2
    return out


def _ca_velocity(t, p):
    out = np.zeros((t.size, 3))
    out[:, 0] = p["accel_mps2"] * t
    return out


def _start_stop_along_track(t, p):
    """Distance and speed along +x for the repeating accelerate, cruise, brake, stand cycle."""
    v, a = p["cruise_speed_mps"], p["accel_mps2"]
    t_acc = v / a
    d_acc = v * v / (2.0 * a)
    period = 2.0 * t_acc + p["cruise_s"] + p["stop_s"]
    cycles = np.floor(t / period)
    tau = t - cycles * period
    base = cycles * (2.0 * d_acc + v * p["cruise_s"])
    t1 = t_acc
    t2 = t1 + p["cruise_s"]
    t3 = t2 + t_acc
    in_acc = tau < t1
    in_cruise = (tau >= t1) & (tau < t2)
    in_brake = (tau >= t2) & (tau < t3)
    s_acc = base + 0.5 * a * tau**2
    s_cruise = base + d_acc + v * (tau - t1)
    u = tau - t2
    s_brake = base + d_acc + v * p["cruise_s"] + v * u - 0.5 * a * u**2
    s_stand = base + 2.0 * d_acc + v * p["cruise_s"]
    s = np.select([in_acc, in_cruise, in_brake], [s_acc, s_cruise, s_brake], s_stand)
    speed = np.select([in_acc, in_cruise, in_brake], [a * tau, v, v - a * u], 0.0)
    return s, speed


def _start_stop_position(t, p):
    out = np.zeros((t.size, 3))
    out[:, 0] = _start_stop_along_track(t, p)[0]
    return out


def _start_stop_velocity(t, p):
    out = np.zeros((t.size, 3))
    out[:, 0] = _start_stop_along_track(t, p)[1]
    return out


def _mixed_parts(t, p):
    """Position and heading for straight, left arc, straight again."""
    v, r = p["speed_mps"], p["radius_m"]
    T1 = p["straight1_s"]
    theta = p["turn_angle_rad"]
    w = v / r
    T2 = theta / w
    s = np.clip(t - T1, 0.0, T2)  # time spent on the arc so far
    after = np.maximum(t - T1 - T2, 0.0)
    straight = np.minimum(t, T1)
    heading = w * s
    x = v * straight + r * np.sin(heading) + v * after * np.cos(theta)
    y = r * (1.0 - np.cos(heading)) + v * after * np.sin(theta)
    return x, y, heading


def _mixed_position(t, p):
    out = np.zeros((t.size, 3))
    out[:, 0], out[:, 1], _ = _mixed_parts(t, p)
    return out


def _mixed_velocity(t, p):
    _, _, heading = _mixed_parts(t, p)
    v = p["speed_mps"]
    out = np.zeros((t.size, 3))
    out[:, 0] = v * np.cos(heading)
    out[:, 1] = v * np.sin(heading)
    return out


_CLOSED_FORM = {
    "cv": (_cv_position, _cv_velocity),
    "ct": (_ct_position, _ct_velocity),
    "figure8": (_figure8_position, _figure8_velocity),
    "start_stop": (_start_stop_position, _start_stop_velocity),
    "mixed": (_mixed_position, _mixed_velocity),
    "ca_line": (_ca_position, _ca_velocity),
}


def _arc_coefficients(w: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Vector form of kinematics.turn_coefficients: A = sin(w dt)/w, B = (1 - cos(w dt))/w."""
    u = w * dt
    small = np.abs(u) < SMALL_TURN
    safe_w = np.where(small, 1.0, w)
    A = np.where(small, dt * (1.0 - u**2 / 6.0 + u**4 / 120.0), np.sin(u) / safe_w)
    B = np.where(small, dt * (u / 2.0 - u**3 / 24.0 + u**5 / 720.0), (1.0 - np.cos(u)) / safe_w)
    return A, B


class Target:
    """A truth motion. Use ``make_target`` to build one."""

    def __init__(self, name: str, params: dict, seed: int):
        self.name = name
        self.params = dict(params)
        self.seed = int(seed)
        self._grid_end_s = 0.0
        self._grid: dict[str, np.ndarray] = {}

    @property
    def start_position(self) -> np.ndarray:
        return np.zeros(3)

    # -- random targets: generate on a fixed grid, cached ------------------
    def _ensure_grid(self, t_max: float) -> None:
        if self.name not in ("jink", "cwna"):
            return
        if self._grid and t_max <= self._grid_end_s:
            return
        end = max(MIN_GENERATED_S, float(t_max))
        # Regenerating from the same seed keeps the earlier part identical
        # (numpy draws are a prefix of a longer draw, and the recursions are sequential).
        self._grid = self._build_jink(end) if self.name == "jink" else self._build_cwna(end)
        self._grid_end_s = end

    def _build_jink(self, end: float) -> dict[str, np.ndarray]:
        p = self.params
        v, sigma, tau = p["speed_mps"], p["turn_rate_std_radps"], p["correlation_time_s"]
        dt = JINK_GRID_S
        n = int(math.ceil(end / dt - 1e-9))
        rng = np.random.default_rng(self.seed)
        w0 = sigma * rng.standard_normal()
        noise = rng.standard_normal(n)
        a = math.exp(-dt / tau)
        b = sigma * math.sqrt(1.0 - math.exp(-2.0 * dt / tau))
        # w[k+1] = a w[k] + b noise[k], with w[0] a stationary draw.
        from scipy.signal import lfilter  # imported here: scipy.signal takes seconds to load

        tail, _ = lfilter([b], [1.0, -a], noise, zi=[a * w0])
        w = np.concatenate(([w0], tail))[:n]  # turn rate used for step k
        heading = np.concatenate(([0.0], np.cumsum(w * dt)))
        h = heading[:-1]
        A, B = _arc_coefficients(w, dt)
        dx = v * (A * np.cos(h) - B * np.sin(h))
        dy = v * (B * np.cos(h) + A * np.sin(h))
        pos = np.zeros((n + 1, 3))
        pos[1:, 0] = np.cumsum(dx)
        pos[1:, 1] = np.cumsum(dy)
        return {"t": np.arange(n + 1) * dt, "pos": pos, "heading": heading, "w": w}

    def _build_cwna(self, end: float) -> dict[str, np.ndarray]:
        q, T = self.params["q_acc_psd"], self.params["sample_period_s"]
        n = int(math.ceil(end / T - 1e-9))
        rng = np.random.default_rng(self.seed)
        v0 = rng.standard_normal(3)
        Q = q * np.array([[T**3 / 3.0, T**2 / 2.0], [T**2 / 2.0, T]])
        noise = rng.multivariate_normal(np.zeros(2), Q, size=(n, 3))  # (n, axis, [pos, vel])
        pos = np.zeros((n + 1, 3))
        vel = np.zeros((n + 1, 3))
        vel[0] = v0
        for k in range(n):
            pos[k + 1] = pos[k] + T * vel[k] + noise[k, :, 0]
            vel[k + 1] = vel[k] + noise[k, :, 1]
        return {"t": np.arange(n + 1) * T, "pos": pos, "vel": vel}

    # -- public queries ----------------------------------------------------
    def position(self, times_s) -> np.ndarray:
        """Positions, shape (N, 3)."""
        t = _times(times_s)
        if self.name in _CLOSED_FORM:
            return _CLOSED_FORM[self.name][0](t, self.params)
        self._ensure_grid(float(t.max()) if t.size else 0.0)
        g = self._grid
        return np.column_stack([np.interp(t, g["t"], g["pos"][:, i]) for i in range(3)])

    def velocity(self, times_s) -> np.ndarray:
        """Velocities, shape (N, 3)."""
        t = _times(times_s)
        if self.name in _CLOSED_FORM:
            return _CLOSED_FORM[self.name][1](t, self.params)
        self._ensure_grid(float(t.max()) if t.size else 0.0)
        g = self._grid
        if self.name == "jink":
            # Interpolate the heading, not the velocity, so the speed stays exact.
            h = np.interp(t, g["t"], g["heading"])
            v = self.params["speed_mps"]
            out = np.zeros((t.size, 3))
            out[:, 0] = v * np.cos(h)
            out[:, 1] = v * np.sin(h)
            return out
        return np.column_stack([np.interp(t, g["t"], g["vel"][:, i]) for i in range(3)])

    def turn_rate(self, times_s) -> np.ndarray:
        """Jink only: the turn rate (rad/s) in force at each time (held over each grid step)."""
        if self.name != "jink":
            raise ValueError("turn_rate is only defined for the jink target")
        t = _times(times_s)
        self._ensure_grid(float(t.max()) if t.size else 0.0)
        idx = np.minimum((t / JINK_GRID_S + 1e-9).astype(int), self._grid["w"].size - 1)
        return self._grid["w"][idx]


def make_target(name: str, params: dict | None = None, seed: int | None = None) -> Target:
    """Build a target by name. Unknown names or parameters raise ValueError.

    ``seed`` only matters for ``jink`` and ``cwna``; None means seed 0, so a
    target is always reproducible.
    """
    if name not in DEFAULTS:
        raise ValueError(f"target must be one of {TARGET_NAMES}")
    merged = dict(DEFAULTS[name])
    for key, value in (params or {}).items():
        if key not in merged:
            raise ValueError(f"unknown parameter {key!r} for target {name!r}; allowed: {sorted(merged)}")
        merged[key] = value
    for key, value in merged.items():
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or (value == 0.0 and key not in NONNEGATIVE):
            raise ValueError(f"{key} must be finite and {'nonnegative' if key in NONNEGATIVE else 'positive'}")
        merged[key] = value
    return Target(name, merged, 0 if seed is None else int(seed))
