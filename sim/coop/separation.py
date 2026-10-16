"""Simulated separation-buffer encounters for cooperative observers.

The model is deliberately small: agents are kinematic point masses, links are
abstract delayed/lossy links, and position and velocity noise are simulated
Gaussian draws.  Velocity noise is interpreted as a filtered velocity estimate
in metres per second.  It is not a sensor model and does not represent flight
dynamics or measured communication performance.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from sim.coop.avoidance import AvoidanceConfig, safe_velocity
from sim.interaction import ellipsoid_separation
from sim.link import LinkConfig, SimulatedLink


_EPS = 1.0e-12


def _finite(name: str, value: float, *, minimum: float | None = None,
            maximum: float | None = None, strict: bool = False) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (out < minimum or (strict and out == minimum)):
        op = ">" if strict else ">="
        raise ValueError(f"{name} must be {op} {minimum}")
    if maximum is not None and out > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return out


@dataclass(frozen=True)
class EncounterSpec:
    """One simulated encounter configuration.

    ``noise_std_m`` is applied independently to each position axis and to each
    velocity axis.  Velocity noise stands for a filtered velocity estimate.
    This abstraction omits estimator dynamics and real vehicle dynamics.
    ``avoid`` is retained as a diagnostic switch for geometry tests.  Even in
    that mode each agent calls :func:`safe_velocity`; the diagnostic simply
    supplies an empty neighbor list.
    """

    kind: str = "head_on"
    speed_mps: float = 1.0
    delay_s: float = 0.0
    loss_probability: float = 0.0
    noise_std_m: float = 0.0
    dead_reckoning: bool = False
    avoidance_radius_m: float = 0.35
    min_separation_m: float = 0.35
    share_rate_hz: float = 10.0
    control_dt_s: float = 0.05
    max_accel_mps2: float = 4.0
    max_speed_mps: float = 2.0
    time_horizon_s: float = 2.0
    n_ring: int = 6
    ring_radius_m: float = 3.0
    lateral_offset_m: float = 0.1
    altitude_m: float = 1.5
    avoid: bool = True

    def __post_init__(self) -> None:
        if self.kind not in {"head_on", "crossing", "ring_swap"}:
            raise ValueError("kind must be head_on, crossing, or ring_swap")
        _finite("speed_mps", self.speed_mps, minimum=0.0, strict=True)
        _finite("delay_s", self.delay_s, minimum=0.0)
        _finite("loss_probability", self.loss_probability, minimum=0.0, maximum=1.0)
        if self.loss_probability >= 1.0:
            raise ValueError("loss_probability must be below one")
        _finite("noise_std_m", self.noise_std_m, minimum=0.0)
        _finite("avoidance_radius_m", self.avoidance_radius_m, minimum=0.0, strict=True)
        _finite("min_separation_m", self.min_separation_m, minimum=0.0)
        _finite("share_rate_hz", self.share_rate_hz, minimum=0.0, strict=True)
        _finite("control_dt_s", self.control_dt_s, minimum=0.0, strict=True)
        _finite("max_accel_mps2", self.max_accel_mps2, minimum=0.0, strict=True)
        _finite("max_speed_mps", self.max_speed_mps, minimum=0.0, strict=True)
        _finite("time_horizon_s", self.time_horizon_s, minimum=0.0, strict=True)
        if isinstance(self.n_ring, (bool, np.bool_)) or int(self.n_ring) != self.n_ring or self.n_ring < 2:
            raise ValueError("n_ring must be an integer >= 2")
        _finite("ring_radius_m", self.ring_radius_m, minimum=0.0, strict=True)
        _finite("lateral_offset_m", self.lateral_offset_m, minimum=0.0)
        _finite("altitude_m", self.altitude_m, minimum=0.0)
        if not isinstance(self.dead_reckoning, (bool, np.bool_)):
            raise ValueError("dead_reckoning must be boolean")
        if not isinstance(self.avoid, (bool, np.bool_)):
            raise ValueError("avoid must be boolean")


def _geometry(spec: EncounterSpec) -> tuple[np.ndarray, np.ndarray]:
    z = float(spec.altitude_m)
    if spec.kind == "head_on":
        offset = float(spec.lateral_offset_m)
        positions = np.array(((-4.0, offset / 2.0, z), (4.0, -offset / 2.0, z)), dtype=float)
        goals = np.array(((4.0, offset / 2.0, z), (-4.0, -offset / 2.0, z)), dtype=float)
        return positions, goals
    if spec.kind == "crossing":
        positions = np.array(((-4.0, 0.0, z), (0.0, -4.0, z)), dtype=float)
        goals = np.array(((4.0, 0.0, z), (0.0, 4.0, z)), dtype=float)
        return positions, goals
    n = int(spec.n_ring)
    angles = np.arange(n, dtype=float) * (2.0 * np.pi / n)
    positions = np.column_stack((spec.ring_radius_m * np.cos(angles),
                                 spec.ring_radius_m * np.sin(angles),
                                 np.full(n, z)))
    goals = positions[(np.arange(n) + n // 2) % n].copy()
    return positions, goals


def _duration(spec: EncounterSpec, positions: np.ndarray, goals: np.ndarray) -> float:
    path_length = float(np.max(np.linalg.norm(goals - positions, axis=1)))
    return path_length / float(spec.speed_mps) + 6.0


def _unit_preferred(position: np.ndarray, goal: np.ndarray, speed: float) -> np.ndarray:
    delta = goal - position
    distance = float(np.linalg.norm(delta))
    if distance <= _EPS:
        return np.zeros(3)
    return speed * delta / distance * min(1.0, distance / 0.5)


def _limited_velocity(current: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    delta = target - current
    norm = float(np.linalg.norm(delta))
    if norm <= max_delta or norm <= _EPS:
        return target.copy()
    return current + delta * (max_delta / norm)


def _make_links(spec: EncounterSpec, seed: int, n: int) -> dict[tuple[int, int], SimulatedLink]:
    links: dict[tuple[int, int], SimulatedLink] = {}
    route_index = 0
    for sender in range(n):
        for receiver in range(n):
            if sender == receiver:
                continue
            config = LinkConfig(delay_s=spec.delay_s, loss_probability=spec.loss_probability,
                                loss_model="independent")
            # Use the route index in the SeedSequence exactly as specified.
            route_seed = [int(seed), 1000 + route_index]
            links[(sender, receiver)] = SimulatedLink(config, seed=route_seed)
            route_index += 1
    return links


def _message_noise(spec: EncounterSpec, seed: int, n: int) -> list[np.random.Generator]:
    return [np.random.default_rng(np.random.SeedSequence([int(seed), 2000 + agent]))
            for agent in range(n)]


def _spec_dict(spec: EncounterSpec) -> dict[str, Any]:
    return {field.name: getattr(spec, field.name) for field in fields(spec)}


def run_encounter(spec: EncounterSpec, seed: int) -> dict[str, Any]:
    """Run one deterministic simulated encounter and return its measurements.

    The controller is kinematic and changes velocity by at most
    ``max_accel_mps2 * control_dt_s`` per step.  ORCA's controller assumes an
    instantaneous velocity change, so the acceleration limit is an explicit
    model limitation.  A directed :class:`~sim.link.SimulatedLink` carries each
    noisy broadcast.  Before the first delivery, the neighbor is unknown and
    omitted from avoidance.  With dead reckoning disabled, a received position
    remains at its send-time value; with it enabled, the received velocity is
    used to extrapolate from send time to the current time.
    """
    if not isinstance(spec, EncounterSpec):
        raise TypeError("spec must be EncounterSpec")
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    seed_int = int(seed)
    if seed_int != seed:
        raise ValueError("seed must be an integer")

    initial, goals = _geometry(spec)
    n = initial.shape[0]
    duration = _duration(spec, initial, goals)
    dt = float(spec.control_dt_s)
    n_steps = int(np.ceil(duration / dt - 1.0e-12))
    links = _make_links(spec, seed_int, n)
    noise = _message_noise(spec, seed_int, n)
    positions = initial.copy()
    velocities = np.zeros_like(positions)
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    sequence = [0] * n
    next_broadcast = [0.0] * n
    flags = {"orca": 0, "infeasible": 0, "emergency": 0, "speed_cap": 0}
    min_center = float("inf")
    min_ellipsoid = float("inf")
    age_samples: list[float] = []
    perceived_trace: list[dict[str, Any]] = []
    max_delta = float(spec.max_accel_mps2) * dt

    def update_minima() -> None:
        nonlocal min_center, min_ellipsoid
        for i in range(n):
            for j in range(i + 1, n):
                distance = float(np.linalg.norm(positions[i] - positions[j]))
                min_center = min(min_center, distance)
                ell = ellipsoid_separation(positions[i], positions[j], (0.12, 0.12, 0.30))
                min_ellipsoid = min(min_ellipsoid, ell)

    update_minima()
    for step in range(n_steps):
        now = min(step * dt, duration)
        # Capture and send all broadcasts due at this control instant.
        for sender in range(n):
            if now + 1.0e-10 < next_broadcast[sender]:
                continue
            while next_broadcast[sender] <= now + 1.0e-10:
                noisy_position = positions[sender] + noise[sender].normal(0.0, spec.noise_std_m, size=3)
                noisy_velocity = velocities[sender] + noise[sender].normal(0.0, spec.noise_std_m, size=3)
                seq = sequence[sender]
                sequence[sender] += 1
                for receiver in range(n):
                    if receiver == sender:
                        continue
                    links[(sender, receiver)].send(sender=str(sender), receiver=str(receiver), seq=seq,
                                                   send_time=float(next_broadcast[sender]))
                    # SimulatedLink intentionally does not retain a message data.
                    latest[(sender, receiver, seq)] = {
                        "position": noisy_position.copy(), "velocity": noisy_velocity.copy(),
                        "send_time": float(next_broadcast[sender]),
                    }
                next_broadcast[sender] += 1.0 / float(spec.share_rate_hz)

        # Deliver packets available at this instant and install the latest one.
        for (sender, receiver), link in links.items():
            for event in link.advance(now):
                message = latest.get((sender, receiver, event.seq))
                if message is not None:
                    prior = latest.get((sender, receiver))
                    if prior is None or event.seq >= prior["seq"]:
                        latest[(sender, receiver)] = {**message, "seq": event.seq}

        perceived: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(n)]
        for receiver in range(n):
            for sender in range(n):
                if sender == receiver:
                    continue
                message = latest.get((sender, receiver))
                if message is None or "seq" not in message:
                    continue
                age = max(0.0, now - float(message["send_time"]))
                p = np.asarray(message["position"], dtype=float)
                v = np.asarray(message["velocity"], dtype=float)
                perceived_position = p + v * age if spec.dead_reckoning else p.copy()
                perceived[receiver].append((perceived_position, v.copy()))
                age_samples.append(age)
                if (not spec.avoid and age > 0.0
                        and float(np.linalg.norm(v)) >= float(spec.speed_mps) - 1.0e-9
                        and float(np.linalg.norm(velocities[sender])) >= float(spec.speed_mps) - 1.0e-9):
                    perceived_trace.append({
                        "sender": int(sender), "receiver": int(receiver), "now_s": float(now),
                        "send_time_s": float(message["send_time"]), "age_s": float(age),
                        "true_position": [float(x) for x in positions[sender]],
                        "perceived_position": [float(x) for x in perceived_position],
                        "true_position_m": [float(x) for x in positions[sender]],
                        "perceived_position_m": [float(x) for x in perceived_position],
                        "perceived_velocity_mps": [float(x) for x in v],
                    })

        chosen = np.zeros_like(positions)
        for agent in range(n):
            at_goal = float(np.linalg.norm(goals[agent] - positions[agent])) <= 0.3
            preferred = np.zeros(3) if at_goal else _unit_preferred(
                positions[agent], goals[agent], float(spec.speed_mps)
            )
            config = AvoidanceConfig(radius_m=spec.avoidance_radius_m,
                                     time_horizon_s=spec.time_horizon_s,
                                     max_speed_mps=spec.max_speed_mps,
                                     safety_margin_m=0.0)
            neighbors = perceived[agent] if spec.avoid else []
            value, result_flags = safe_velocity(
                positions[agent], velocities[agent], preferred, neighbors, config,
                solver="active_set",
            )
            chosen[agent] = value
            for flag in flags:
                if flag in result_flags:
                    flags[flag] += 1

        for agent in range(n):
            velocities[agent] = _limited_velocity(velocities[agent], chosen[agent], max_delta)
            speed_norm = float(np.linalg.norm(velocities[agent]))
            if speed_norm > spec.max_speed_mps:
                velocities[agent] *= float(spec.max_speed_mps) / speed_norm
            positions[agent] += velocities[agent] * dt
        update_minima()
        # Once every agent is within the goal tolerance, preferred velocity is
        # zero and the remaining configured horizon would leave the state
        # unchanged.  Stop integrating while retaining the configured duration
        # in the result below.
        if bool(np.all(np.linalg.norm(positions - goals, axis=1) <= 0.3)):
            break

    # A final delivery pass is not used for dynamics, but minima include the
    # complete simulated duration through the last integrated state.
    min_center = float(min_center) if np.isfinite(min_center) else 0.0
    min_ellipsoid = float(min_ellipsoid) if np.isfinite(min_ellipsoid) else 0.0
    penetration = max(0.0, float(spec.min_separation_m) - min_center)
    return {
        "seed": seed_int,
        "spec": _spec_dict(spec),
        "duration_s": float(duration),
        "min_center_distance_m": min_center,
        "violated": bool(min_center < spec.min_separation_m),
        "near_miss": bool(min_center < spec.min_separation_m + 0.10),
        "min_ellipsoid_separation": min_ellipsoid,
        "ellipsoid_violated": bool(min_ellipsoid < 2.0),
        "penetration_m": float(penetration),
        "reached_goals": bool(np.all(np.linalg.norm(positions - goals, axis=1) <= 0.3)),
        "flags": flags,
        "mean_message_age_s": float(np.mean(age_samples)) if age_samples else 0.0,
        "perceived_trace": perceived_trace,
    }


def hand_buffer_m(spec: EncounterSpec) -> float:
    """Return the hand-estimated extra buffer for this simulated condition."""
    if not isinstance(spec, EncounterSpec):
        raise TypeError("spec must be EncounterSpec")
    if spec.kind in {"head_on", "ring_swap"}:
        closing_speed = 2.0 * spec.speed_mps
    else:
        closing_speed = np.sqrt(2.0) * spec.speed_mps
    mean_loss_gap = (1.0 / spec.share_rate_hz) * spec.loss_probability / (1.0 - spec.loss_probability)
    return float(closing_speed * (spec.delay_s + mean_loss_gap + spec.control_dt_s)
                 + 2.0 * np.sqrt(2.0) * spec.noise_std_m)
