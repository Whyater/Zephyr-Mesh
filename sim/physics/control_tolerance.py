"""Control-side delay and loss tolerance harness for EXP-021.

All trajectories in this module are synthetic simulator runs.  The two
profiles consumed here are literature-derived inputs, not flight measurements.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.controller import GeometricFlightController
from sim.hardware import RotorcraftProfile, load_hardware_profile
from sim.metrics import step_response
from sim.vehicle import ActuatedQuadrotor
from sim.physics.delay_buffer import DelayLossChannel

CONTROL_DT_S = 0.01
SETPOINT_SAMPLE_S = 0.02
WHOLE_STATE_SAMPLE_S = 0.01

# Rule sets for calling a run a failure.  EXP021_RULES are the pre-registered
# EXP-021 rules and stay the default, so EXP-021 reproduces unchanged.
# FAILED_TO_FLY_RULES is the stricter D-088 rule for future experiments: it
# is only used when a caller opts in with ``failed_to_fly_rule=True``.
EXP021_RULES = "exp021_preregistered"
FAILED_TO_FLY_RULES = "d088_failed_to_fly"
_MAX_ERROR_M = 5.0
_GROWTH_RATIO = 1.2
_GROWTH_FLOOR_M = 0.05
_RULE_LIMITS = {
    # rule set: (tilt limit in degrees, lowest allowed height in m or None, axes checked for growth)
    EXP021_RULES: (80.0, None, (0,)),
    FAILED_TO_FLY_RULES: (45.0, 0.0, (0, 1, 2)),
}


def _configured_scoring_windows() -> Mapping[str, Any]:
    """Read scoring windows from the experiment configuration."""
    config_path = Path(__file__).resolve().parents[2] / "tools" / "a_control_tolerance_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        windows = config["scoring_windows"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load scoring_windows from {config_path}") from exc
    if not isinstance(windows, Mapping):
        raise RuntimeError("scoring_windows must be a mapping")
    return windows


def reference_position(reference: str, t_s: float) -> np.ndarray:
    """Return the 3-D reference at time ``t_s`` for a named experiment arm."""
    t = float(t_s)
    if not math.isfinite(t) or t < 0:
        raise ValueError("time must be finite and nonnegative")
    if reference == "step":
        x = 0.0 if t < 1.0 else 1.0
    elif reference == "ramp":
        x = 0.0 if t < 1.0 else 0.5 * (t - 1.0)
    elif reference == "circle":
        tp = max(0.0, t - 1.0)
        # R=1 m and w=0.5 rad/s.  At t=1 this is (0, 0, 1).
        x = math.cos(0.5 * tp) - 1.0
        y = math.sin(0.5 * tp)
        return np.array([x, y, 1.0], dtype=float)
    else:
        raise ValueError("reference must be step, ramp, or circle")
    return np.array([x, 0.0, 1.0], dtype=float)


class _ProxyDrone:
    """Duck-typed state object accepted by GeometricFlightController."""

    def __init__(self, drone, state=None):
        self._fresh = drone
        if state is None:
            state = (drone.position.copy(), drone.velocity.copy(), drone.quaternion.copy(), drone.angular_velocity.copy())
        self.position = np.asarray(state[0], dtype=float).copy()
        self.velocity = np.asarray(state[1], dtype=float).copy()
        self.quaternion = np.asarray(state[2], dtype=float).copy()
        self.angular_velocity = np.asarray(state[3], dtype=float).copy()
        self.mass = float(drone.mass)
        self.I_x = float(drone.I_x)
        self.I_y = float(drone.I_y)
        self.I_z = float(drone.I_z)


class _DelayControllerBase(GeometricFlightController):
    def __init__(self, inner: GeometricFlightController | None = None):
        # Do not call the base constructor when an inner controller is supplied:
        # the wrapper delegates all controller math to that exact object.
        self.inner = inner or GeometricFlightController()
        self._time_s = 0.0
        self._step_index = 0

    def _advance(self, dt: float) -> None:
        self._time_s += float(dt)
        self._step_index += 1


class SetpointDelayController(_DelayControllerBase):
    """Arm (a): delay and loss apply to the reference outside the loop."""

    def __init__(self, delay_s: float, loss_p: float, rng: np.random.Generator,
                 *, initial_reference=(0.0, 0.0, 1.0), inner=None):
        super().__init__(inner)
        self.delay_s = float(delay_s)
        self.loss_p = float(loss_p)
        self.channel = DelayLossChannel(self.delay_s, self.loss_p, SETPOINT_SAMPLE_S, rng, initial_value=np.asarray(initial_reference, dtype=float))

    def update_control(self, drone, target_position, dt, **kwargs):
        t = self._time_s
        # References are sampled at 50 Hz, including the t=0 sample.
        if self._step_index % 2 == 0:
            self.channel.offer(t, np.asarray(target_position, dtype=float).copy())
        delayed = (np.asarray(target_position, dtype=float).copy()
                   if self.delay_s == 0.0 and self.loss_p == 0.0
                   else np.asarray(self.channel.latest(t), dtype=float))
        out = self.inner.update_control(drone, delayed, dt, **kwargs)
        self._advance(dt)
        return out


class FeedbackDelayController(_DelayControllerBase):
    """Arm (b): delay and loss apply to position and velocity feedback only."""

    def __init__(self, delay_s: float, loss_p: float, rng: np.random.Generator,
                 *, initial_state=None, inner=None):
        super().__init__(inner)
        self.delay_s = float(delay_s)
        self.loss_p = float(loss_p)
        self.channel = DelayLossChannel(self.delay_s, self.loss_p, SETPOINT_SAMPLE_S, rng, initial_value=initial_state)

    def update_control(self, drone, target_position, dt, **kwargs):
        t = self._time_s
        if self._step_index % 2 == 0:
            self.channel.offer(t, (drone.position.copy(), drone.velocity.copy()))
        if self.delay_s == 0.0 and self.loss_p == 0.0:
            delayed_pos, delayed_vel = drone.position, drone.velocity
        else:
            delayed_pos, delayed_vel = self.channel.latest(t)
        proxy = _ProxyDrone(drone, (delayed_pos, delayed_vel, drone.quaternion, drone.angular_velocity))
        out = self.inner.update_control(proxy, target_position, dt, **kwargs)
        self._advance(dt)
        return out


class WholeStateDelayController(_DelayControllerBase):
    """Slice (c): delay the complete state, while retaining a fresh reference."""

    def __init__(self, delay_s: float, rng: np.random.Generator, *, initial_state=None, inner=None):
        super().__init__(inner)
        self.delay_s = float(delay_s)
        self.channel = DelayLossChannel(self.delay_s, 0.0, WHOLE_STATE_SAMPLE_S, rng, initial_value=initial_state)

    def update_control(self, drone, target_position, dt, **kwargs):
        t = self._time_s
        self.channel.offer(t, (drone.position.copy(), drone.velocity.copy(), drone.quaternion.copy(), drone.angular_velocity.copy()))
        delayed = self.channel.latest(t)
        proxy = _ProxyDrone(drone, delayed)
        out = self.inner.update_control(proxy, target_position, dt, **kwargs)
        self._advance(dt)
        return out


def _profile_obj(profile: RotorcraftProfile | str | Path) -> RotorcraftProfile:
    if isinstance(profile, (str, Path)):
        loaded = load_hardware_profile(profile)
    else:
        loaded = profile
    if not isinstance(loaded, RotorcraftProfile):
        raise TypeError("EXP-021 requires a v2 RotorcraftProfile")
    return loaded


def _tilt_angle(drone) -> float:
    qw, qx, qy, qz = drone.quaternion
    body_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.acos(float(np.clip(body_z_z, -1.0, 1.0)))


def _controller_for(arm: str, delay_s: float, loss_p: float, rng: np.random.Generator, drone):
    hover = np.array([0.0, 0.0, 1.0], dtype=float)
    state = (drone.position.copy(), drone.velocity.copy(), drone.quaternion.copy(), drone.angular_velocity.copy())
    if arm == "a":
        return SetpointDelayController(delay_s, loss_p, rng, initial_reference=hover)
    if arm == "b":
        return FeedbackDelayController(delay_s, loss_p, rng, initial_state=(state[0], state[1]))
    if arm == "c":
        return WholeStateDelayController(delay_s, rng, initial_state=state)
    raise ValueError("arm must be a, b, or c")


def _state_instability(state, reference: str, t_s: float) -> tuple[str, float] | None:
    """Return a pre-registered instability rule triggered by ``state``."""
    arrays = [np.asarray(getattr(state, name), dtype=float)
              for name in ("position", "velocity", "quaternion", "angular_velocity")]
    if not all(np.all(np.isfinite(values)) for values in arrays):
        return "non_finite_state", float(t_s)
    if _tilt_angle(state) > math.radians(80.0):
        return "tilt_above_80_deg", float(t_s)
    target = reference_position(reference, t_s)
    if float(np.linalg.norm(state.position - target)) > 5.0:
        return "max_error_above_5_m", float(t_s)
    return None


def _growing_error(pos_arr: np.ndarray, ref_arr: np.ndarray, axes: tuple[int, ...]) -> bool:
    """True if the error on any of ``axes`` peaks more than 1.2 times higher in
    the last third of the run than in the middle third, and above 0.05 m."""
    third = max(1, len(pos_arr) // 3)
    for axis in axes:
        middle_peak = float(np.max(np.abs(pos_arr[third:2 * third, axis] - ref_arr[third:2 * third, axis])))
        last_peak = float(np.max(np.abs(pos_arr[2 * third:, axis] - ref_arr[2 * third:, axis])))
        # The 0.05 m floor keeps tiny converged wiggles from counting as growth.
        if last_peak > _GROWTH_RATIO * middle_peak and last_peak > _GROWTH_FLOOR_M:
            return True
    return False


def classify_trajectory(times_s, positions_m, references_m, tilts_deg, *,
                        rule_set: str) -> dict[str, Any]:
    """Classify a recorded run under one rule set.

    ``rule_set`` is required so a caller always names the rules it uses.
    Inputs are one row per control step: time (s), 3-D position and reference
    (m), and tilt from upright (degrees, NaN if the attitude was not finite).
    The first step that breaks a per-step rule decides the verdict:
    non-finite sample, tilt above the limit, height below the limit (D-088
    only), or 3-D error above 5 m.  If no step does, the growth rule runs on
    the whole record (x only for EXP-021, x, y and z for D-088).

    With EXP021_RULES this mirrors the pre-registered rules on a recorded
    trajectory (it sees position and tilt only, not velocities).  Every run
    those rules fail also fails FAILED_TO_FLY_RULES, no later in time.
    """
    if rule_set not in _RULE_LIMITS:
        raise ValueError(f"rule_set must be {EXP021_RULES!r} or {FAILED_TO_FLY_RULES!r}")
    t = np.asarray(times_s, dtype=float)
    pos = np.asarray(positions_m, dtype=float)
    ref = np.asarray(references_m, dtype=float)
    tilt = np.asarray(tilts_deg, dtype=float)
    n = len(t)
    if pos.shape != (n, 3) or ref.shape != (n, 3) or tilt.shape != (n,):
        raise ValueError("expected n times, n x 3 positions and references, and n tilts")
    if n == 0 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("times must be non-empty, finite and increasing")
    tilt_limit_deg, min_height_m, growth_axes = _RULE_LIMITS[rule_set]
    errors = np.linalg.norm(pos - ref, axis=1)
    with np.errstate(invalid="ignore"):
        checks = [
            ("non_finite_state", ~(np.all(np.isfinite(pos), axis=1) & np.isfinite(tilt)
                                   & np.all(np.isfinite(ref), axis=1))),
            (f"tilt_above_{tilt_limit_deg:g}_deg", tilt > tilt_limit_deg),
        ]
        if min_height_m is not None:
            checks.append((f"height_below_{min_height_m:g}_m", pos[:, 2] < min_height_m))
        checks.append(("max_error_above_5_m", errors > _MAX_ERROR_M))
    first_step, first_reason = n, None
    for reason, hits in checks:
        if np.any(hits):
            step = int(np.argmax(hits))
            # Earlier steps win; on the same step the order above wins.
            if step < first_step:
                first_step, first_reason = step, reason
    if first_reason is not None:
        return {"failed": True, "reason": first_reason, "time_s": float(t[first_step])}
    if n >= 6 and _growing_error(pos, ref, growth_axes):
        reason = "growing_oscillation" if rule_set == EXP021_RULES else "growing_error"
        return {"failed": True, "reason": reason, "time_s": float(t[-1])}
    return {"failed": False, "reason": None, "time_s": None}


def run_episode(profile, arm: str, reference: str, delay_s: float, loss_p: float, seed: int,
                axis_note: str | None = None, *, duration_s: float | None = None,
                scoring_windows: Mapping[str, Any] | None = None,
                failed_to_fly_rule: bool = False) -> dict[str, Any]:
    """Run one deterministic EXP-021 episode and return JSON-safe metrics.

    ``unstable`` always uses the pre-registered EXP-021 rules.  With
    ``failed_to_fly_rule=True`` the result also carries a D-088 verdict
    (``failed_to_fly`` and its reason and time) for the same flight.  The
    flight itself is unchanged: it still stops only on the EXP-021 rules.
    """
    p = _profile_obj(profile)
    delay = float(delay_s)
    loss = float(loss_p)
    if delay < 0 or not math.isfinite(delay) or loss < 0 or loss >= 1 or not math.isfinite(loss):
        raise ValueError("delay must be finite and nonnegative and loss_p in [0, 1)")
    if arm == "c" and loss != 0.0:
        raise ValueError("whole-state slice has zero loss")
    if arm == "c" and reference != "step":
        raise ValueError("whole-state slice uses the step reference")
    default_duration = 30.0 if reference == "circle" else 20.0
    duration = default_duration if duration_s is None else float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    dt = CONTROL_DT_S
    rng = np.random.default_rng(int(seed))
    # Construct first so the initial state can seed feedback channels, then trim rotors.
    vehicle = ActuatedQuadrotor(p, initial_position=(0.0, 0.0, 1.0), controller=GeometricFlightController(), substeps=1)
    vehicle.actuator.speed_rad_s[:] = p.hover_speed_rad_s()
    vehicle.controller = _controller_for(arm, delay, loss, rng, vehicle.drone)

    times, positions, references, tilts = [], [], [], []
    unstable = False
    reason = None
    unstable_time = None
    n_steps = int(round(duration / dt))
    for k in range(n_steps):
        t = (k + 1) * dt
        target = reference_position(reference, k * dt)
        try:
            vehicle.step(target, dt)
        except (FloatingPointError, OverflowError, ValueError):
            # A numerical exception is only a consequence of a registered
            # state rule when that rule is already true.  Check the state
            # before deciding whether to record instability; a finite,
            # bounded state must re-raise the simulator error.
            violation = _state_instability(vehicle.drone, reference, t)
            if violation is None:
                raise
            unstable, reason, unstable_time = True, violation[0], violation[1]
        else:
            violation = _state_instability(vehicle.drone, reference, t)
            if violation is not None:
                unstable, reason, unstable_time = True, violation[0], violation[1]
        times.append(t)
        positions.append(vehicle.drone.position.copy())
        references.append(reference_position(reference, t))
        tilts.append(_tilt_angle(vehicle.drone) if np.all(np.isfinite(vehicle.drone.quaternion)) else math.nan)
        if unstable:
            break

    t_arr = np.asarray(times, dtype=float)
    pos_arr = np.asarray(positions, dtype=float)
    ref_arr = np.asarray(references, dtype=float)
    errors = np.linalg.norm(pos_arr - ref_arr, axis=1) if len(pos_arr) else np.empty(0)
    if len(errors):
        # Error criteria use the complete run history, including the early transient.
        if not unstable and (not np.all(np.isfinite(errors)) or float(np.max(errors)) > 5.0):
            unstable, reason, unstable_time = True, "max_error_above_5_m", float(t_arr[int(np.argmax(errors))])
        if not unstable and len(errors) >= 6 and _growing_error(pos_arr, ref_arr, (0,)):
            unstable, reason, unstable_time = True, "growing_oscillation", float(t_arr[-1])
    # Production scoring windows are supplied by the run configuration.  Load
    # that same configuration for direct callers that omit the argument.
    windows = _configured_scoring_windows() if scoring_windows is None else scoring_windows
    try:
        raw_window = windows[reference]
        if isinstance(raw_window, Mapping):
            window_start = float(raw_window["start_s"])
            window_end = float(raw_window["end_s"])
        else:
            window_start, window_end = (float(value) for value in raw_window)
    except (KeyError, TypeError, ValueError):
        raise ValueError("scoring_windows must define [start_s, end_s] for each reference")
    if not (math.isfinite(window_start) and math.isfinite(window_end)
            and 0.0 <= window_start <= window_end):
        raise ValueError("scoring window bounds must be finite and ordered")
    if duration < window_end:
        # Keep shortened-duration behavior from the original harness: ramp
        # and circle score their latter half, while step retains its t=1 start.
        if reference in {"ramp", "circle"}:
            window_start = max(1.0, duration * 0.5)
        window_end = duration
    score_mask = (t_arr >= window_start - 1e-12) & (t_arr <= window_end + 1e-12)
    if not np.any(score_mask):
        score_mask = np.ones(len(t_arr), dtype=bool)
    score_errors = errors[score_mask]
    rms = float(np.sqrt(np.mean(score_errors ** 2))) if len(score_errors) else float("nan")
    max_error = float(np.max(score_errors)) if len(score_errors) else float("nan")
    response = None
    if reference == "step" and len(t_arr):
        mask = t_arr >= 1.0 - 1e-12
        response = step_response(t_arr[mask], pos_arr[mask, 0], 1.0, tolerance=0.02)
    result = {
        "profile": p.profile_id,
        "arm": arm,
        "reference": reference,
        "delay_s": delay,
        "loss_p": loss,
        "seed": int(seed),
        "axis_note": axis_note,
        "duration_s": duration,
        "rms_error_m": rms,
        "max_error_m": max_error,
        "unstable": bool(unstable),
        "unstable_reason": reason,
        "unstable_time_s": None if unstable_time is None else float(unstable_time),
        "n_steps": int(len(t_arr)),
        # Diagnostics over the whole run. They do not change the pre-registered
        # instability rules; they show runs that pass those rules while flying badly.
        "whole_run_max_error_m": float(np.max(errors)) if len(errors) else float("nan"),
        "whole_run_max_tilt_deg": float(np.degrees(np.nanmax(tilts))) if len(tilts) else float("nan"),
        "whole_run_min_z_m": float(np.min(pos_arr[:, 2])) if len(pos_arr) else float("nan"),
    }
    if response is not None:
        result.update({"step_overshoot_pct": response["overshoot_pct"], "settling_time_s": response["settling_time_s"], "step_response": response})
    if failed_to_fly_rule:
        verdict = classify_trajectory(t_arr, pos_arr.reshape(-1, 3), ref_arr.reshape(-1, 3),
                                      np.degrees(np.asarray(tilts, dtype=float)),
                                      rule_set=FAILED_TO_FLY_RULES)
        if not verdict["failed"] and unstable:
            # The state checks also see velocity and angular rate, which the
            # recorded trajectory lacks.  Keep D-088 at least as strict.
            verdict = {"failed": True, "reason": reason, "time_s": unstable_time}
        result.update({
            "failed_to_fly": bool(verdict["failed"]),
            "failed_to_fly_reason": verdict["reason"],
            "failed_to_fly_time_s": verdict["time_s"],
            "failed_to_fly_rule_set": FAILED_TO_FLY_RULES,
        })
    return result


def find_d_crit(profile, *, arm: str = "b", resolution_s: float = CONTROL_DT_S,
                bracket: tuple[float, float] = (0.0, 0.5), seed: int = 0,
                duration_s: float | None = None,
                scoring_windows: Mapping[str, Any] | None = None,
                failed_to_fly_rule: bool = False) -> dict[str, Any]:
    """Bisect the stable/unstable boundary on the 10 ms control grid.

    By default a run counts as unstable under the EXP-021 rules.  With
    ``failed_to_fly_rule=True`` it bisects on the D-088 ``failed_to_fly``
    verdict instead and records ``rule_set`` in the answer.
    """
    p = _profile_obj(profile)
    lo, hi = map(float, bracket)
    if lo < 0 or hi <= lo or resolution_s <= 0:
        raise ValueError("invalid d_crit bracket")
    # Delay transport is observed at each 10 ms control tick.  Searching on
    # integer ticks makes the reported bracket match the effective delay.
    low_step = int(math.floor(lo / CONTROL_DT_S + 1e-9))
    high_step = int(math.ceil(hi / CONTROL_DT_S - 1e-9))
    if high_step <= low_step:
        raise ValueError("d_crit bracket must span at least one control tick")

    run_kwargs: dict[str, Any] = {}
    if duration_s is not None:
        run_kwargs["duration_s"] = float(duration_s)
    if scoring_windows is not None:
        run_kwargs["scoring_windows"] = scoring_windows
    verdict_key = "unstable"
    if failed_to_fly_rule:
        run_kwargs["failed_to_fly_rule"] = True
        verdict_key = "failed_to_fly"

    results: dict[int, dict[str, Any]] = {}

    def _run(step: int) -> dict[str, Any]:
        results[step] = run_episode(p, arm, "step", step * CONTROL_DT_S, 0.0, seed, **run_kwargs)
        return results[step]

    low_result = _run(low_step)
    high_result = _run(high_step)
    if low_result[verdict_key]:
        raise ValueError("lower bracket must be stable")
    if not high_result[verdict_key]:
        raise ValueError("upper bracket must be unstable")
    while high_step - low_step > 1:
        mid_step = (low_step + high_step) // 2
        result = _run(mid_step)
        if result[verdict_key]:
            high_step = mid_step
        else:
            low_step = mid_step
    stable_delay = round(low_step * CONTROL_DT_S, 10)
    unstable_delay = round(high_step * CONTROL_DT_S, 10)
    answer = {
        "profile": p.profile_id,
        "arm": arm,
        "d_crit_s": float(unstable_delay),
        "stable_delay_s": float(stable_delay),
        "unstable_delay_s": float(unstable_delay),
        "resolution_s": float(CONTROL_DT_S),
        "effective_delay_grid_s": float(CONTROL_DT_S),
        "stable_side": _diagnostics(results[low_step]),
        "unstable_side": _diagnostics(results[high_step]),
    }
    if failed_to_fly_rule:
        answer["rule_set"] = FAILED_TO_FLY_RULES
    return answer


def _diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("rms_error_m", "max_error_m", "whole_run_max_error_m", "whole_run_max_tilt_deg",
            "whole_run_min_z_m", "unstable_reason", "unstable_time_s")
    out = {key: result.get(key) for key in keys}
    if "failed_to_fly" in result:
        out.update({key: result[key] for key in ("failed_to_fly", "failed_to_fly_reason", "failed_to_fly_time_s")})
    return out

# Alternate descriptive name used by sweep code.
d_crit_search = find_d_crit

__all__ = [
    "EXP021_RULES",
    "FAILED_TO_FLY_RULES",
    "FeedbackDelayController",
    "SetpointDelayController",
    "WholeStateDelayController",
    "classify_trajectory",
    "d_crit_search",
    "find_d_crit",
    "reference_position",
    "run_episode",
]
