"""Shared loop for simulated missions.

A mission is a small class with four hooks (``setup``, ``tick``, ``done``,
``metrics``).  The engine builds the adapters, the synthetic radio
environment and the links, steps time on a fixed 10 ms grid, collects events
and, when asked, writes a mission log.  Ground-station logic inside a mission
must only use what has arrived over its links; the true state is for scoring.

Everything here is simulated.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from sim.adapters import Capabilities, DroneAdapter, Geofence, OffboardPointMassAdapter, SimAdapter
from sim.estimation.targets import make_target
from sim.failsafe import FailsafeConfig
from sim.missions.api import Scenario
from sim.missions.comms import LossProcess, MissionLink, OutageWindow
from sim.missions.log import MissionLogRecorder, build_log
from sim.missions.physics_adapter import PhysicsAdapter
from sim.missions.stack import LinkedDrone

DT = 0.01
LOG_EVERY = 5
COMMON_LIMITS = (
    "Simulated only: no drone was flown and no radio was measured.",
    "Link loss patterns are synthetic (independent or two-state bursts on a 10 ms grid) and blackouts are scripted.",
    "Vehicle and radio parameters are literature-derived, not measured.",
)


def make_adapter(kind: str, agent_id: str, *, initial_position_m, failsafe_config: FailsafeConfig,
                 geofence: Geofence | None, max_speed_mps: float = 1.5) -> DroneAdapter:
    """Build and connect one adapter of the requested kind."""
    capabilities = Capabilities(max_speed_mps=max_speed_mps)
    common = dict(
        capabilities=capabilities,
        initial_position_m=tuple(float(v) for v in initial_position_m),
        failsafe_config=failsafe_config,
        geofence=geofence,
    )
    if kind == "point_mass":
        adapter = SimAdapter(agent_id, dt_s=DT, response_time_s=0.25, max_acceleration_mps2=4.0, **common)
    elif kind == "point_mass_offboard":
        adapter = OffboardPointMassAdapter(agent_id, dt_s=DT, response_time_s=0.25,
                                           max_acceleration_mps2=4.0, **common)
    elif kind == "physics":
        adapter = PhysicsAdapter(agent_id, dt_s=DT, **common)
    elif kind == "physics_offboard":
        adapter = PhysicsAdapter(agent_id, dt_s=DT, feedback="offboard", **common)
    else:
        raise ValueError(f"unknown adapter kind: {kind!r}")
    adapter.connect()
    return adapter


def stopping_buffer_m(speed_mps: float, age_s: float, reaction_s: float = 0.05,
                      max_decel_mps2: float = 4.0) -> float:
    """Distance covered while acting on old data, then braking to a stop.

    ``v * (age + reaction) + v^2 / (2 a)``.  Run E's stand-in for Run D's
    separation buffer (not available on this branch).
    """
    v = abs(float(speed_mps))
    age = 0.0 if not math.isfinite(age_s) else max(0.0, float(age_s))
    return v * (age + reaction_s) + v * v / (2.0 * max_decel_mps2)


class PlacedTarget:
    """A Run B target path rotated, moved and lifted into a mission arena."""

    def __init__(self, base, *, origin, heading_rad: float, altitude_m: float) -> None:
        self.base = base
        self.origin = np.asarray(origin, dtype=float)[:2]
        self.altitude_m = float(altitude_m)
        c, s = math.cos(heading_rad), math.sin(heading_rad)
        self._rot = np.array([[c, -s], [s, c]])

    def position(self, times_s) -> np.ndarray:
        xy = np.asarray(self.base.position(times_s), dtype=float)[:, :2] @ self._rot.T + self.origin
        return np.column_stack([xy, np.full(len(xy), self.altitude_m)])

    def velocity(self, times_s) -> np.ndarray:
        vxy = np.asarray(self.base.velocity(times_s), dtype=float)[:, :2] @ self._rot.T
        return np.column_stack([vxy, np.zeros(len(vxy))])

    @property
    def start_position(self) -> np.ndarray:
        return self.position([0.0])[0]


TARGET_CLASSES = {
    "straight": ("cv", lambda speed: {"speed_mps": speed}),
    "turn": ("ct", lambda speed: {"speed_mps": speed, "radius_m": 1.5}),
    "figure8": ("figure8", lambda speed: {"amplitude_x_m": 3.0, "amplitude_y_m": 1.5, "omega_radps": 0.15}),
    "jink": ("jink", lambda speed: {"speed_mps": speed, "turn_rate_std_radps": 1.0, "correlation_time_s": 1.0}),
}


def target_path(class_name: str, seed: int, *, origin, heading_rad: float = 0.0, altitude_m: float = 1.5,
                speed_mps: float = 0.5) -> PlacedTarget:
    """Build a Run B motion class (seeded) placed in the arena."""
    if class_name not in TARGET_CLASSES:
        raise ValueError(f"unknown target class: {class_name!r}")
    name, params = TARGET_CLASSES[class_name]
    base = make_target(name, params(float(speed_mps)), seed=int(seed))
    return PlacedTarget(base, origin=origin, heading_rad=heading_rad, altitude_m=altitude_m)


@dataclass
class MissionResult:
    metrics: dict
    log: dict | None
    end_reason: str
    steps: int


class MissionContext:
    """Everything a mission needs during one run."""

    def __init__(self, scenario: Scenario, params: dict, *, log: bool, rng_names: tuple[str, ...]) -> None:
        self.scenario = scenario
        self.params = params
        self.dt = DT
        self.n_steps = int(round(scenario.duration_s / DT))
        self.failsafe_config = scenario.failsafe_config
        self.loss = LossProcess(scenario.preset, seed=scenario.seed, dt_s=DT,
                                duration_s=scenario.duration_s + 5.0)
        end_of_time = scenario.duration_s + 10.0
        self.outages = tuple(
            OutageWindow(item["start_s"], end_of_time if item["end_s"] is None else item["end_s"],
                         None if item["streams"] is None else tuple(item["streams"]))
            for item in scenario.outages
        )
        self.adapters: dict[str, DroneAdapter] = {}
        self.drones: dict[str, LinkedDrone] = {}
        self.links: dict[str, Any] = {}
        self.events: list[dict] = []
        self._rng_index = {name: k for k, name in enumerate(sorted(rng_names))}
        self._rngs: dict[str, np.random.Generator] = {}
        self.recorder: MissionLogRecorder | None = None
        self._log = log
        self._last_clip_time: dict[str, float] = {}
        self.state: dict[str, Any] = {}  # free space for the mission

    # randomness -------------------------------------------------------
    def rng(self, name: str) -> np.random.Generator:
        if name not in self._rng_index:
            raise KeyError(f"rng stream {name!r} was not declared by the mission")
        if name not in self._rngs:
            self._rngs[name] = np.random.default_rng(
                np.random.SeedSequence([self.scenario.seed, 9000 + self._rng_index[name]]))
        return self._rngs[name]

    def seed_for(self, name: str) -> int:
        """A plain integer seed for components that take an int (targets, sensors)."""
        if name not in self._rng_index:
            raise KeyError(f"rng stream {name!r} was not declared by the mission")
        return int(np.random.SeedSequence([self.scenario.seed, 9000 + self._rng_index[name]]).generate_state(1)[0])

    # links and drones ---------------------------------------------------
    def link(self, stream_id: str, *, delay_s: float | None = None, jitter_s: float = 0.0) -> MissionLink:
        """A one-way link in the mission's shared synthetic radio environment."""
        if stream_id in self.links:
            raise ValueError(f"link {stream_id!r} already exists")
        link = MissionLink(stream_id, delay_s=self.scenario.command_delay_s if delay_s is None else delay_s,
                           jitter_s=jitter_s, loss=self.loss, outages=self.outages, seed=self.scenario.seed)
        self.links[stream_id] = link
        return link

    def add_drone(self, agent_id: str, *, initial_position_m, geofence: Geofence | None,
                  max_speed_mps: float = 1.5, kind: str | None = None,
                  failsafe_config: FailsafeConfig | None = None) -> LinkedDrone:
        adapter = make_adapter(kind or self.scenario.adapter, agent_id, initial_position_m=initial_position_m,
                               failsafe_config=failsafe_config or self.failsafe_config,
                               geofence=geofence, max_speed_mps=max_speed_mps)
        drone = LinkedDrone(adapter, cmd_link=self.link(f"cmd:{agent_id}"), tel_link=self.link(f"tel:{agent_id}"))
        self.adapters[agent_id] = adapter
        self.drones[agent_id] = drone
        return drone

    # events and logging -------------------------------------------------
    def event(self, time_s: float, kind: str, agent: str | None = None, detail: dict | None = None) -> None:
        self.events.append({"time_s": float(time_s), "kind": kind, "agent": agent, "detail": dict(detail or {})})

    def collect_adapter_events(self) -> None:
        for agent_id in sorted(self.adapters):
            for event in self.adapters[agent_id].drain_events():
                if event["kind"] == "geofence_clip":
                    # Setpoints arrive at 20 Hz; keep at most one clip event
                    # per drone per second so logs stay readable.
                    last = self._last_clip_time.get(agent_id)
                    if last is not None and event["time_s"] - last < 1.0:
                        continue
                    self._last_clip_time[agent_id] = event["time_s"]
                self.events.append(event)

    def start_log(self, agents) -> None:
        if self._log:
            self.recorder = MissionLogRecorder(list(agents), DT, LOG_EVERY)

    def frame(self, i: int, t: float, target_position, per_agent: dict) -> None:
        if self.recorder is not None and i % LOG_EVERY == 0:
            self.recorder.frame(i, t, None if target_position is None else [float(v) for v in target_position],
                                per_agent)


def agent_frame(state, *, estimate_position_m=None, estimate_age_s=None) -> dict:
    """One logged frame for one drone, from its true state (for replay and scoring)."""
    return {
        "position_m": [float(v) for v in state.position_m],
        "velocity_mps": [float(v) for v in state.velocity_mps],
        "failsafe_state": state.failsafe_state,
        "authority": state.command_authority or "none",
        "link_age_s": None if state.link_age_s is None else float(state.link_age_s),
        "estimate_position_m": None if estimate_position_m is None else [float(v) for v in estimate_position_m],
        "estimate_age_s": None if estimate_age_s is None or not math.isfinite(estimate_age_s) else float(estimate_age_s),
    }


class Mission(ABC):
    """Base class for a simulated mission.  See the module docstring."""

    name: str = ""
    rng_names: tuple[str, ...] = ()
    limits: tuple[str, ...] = ()
    defaults: dict = {}

    def validate_params(self, params: dict) -> dict:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"unknown {self.name} params: {sorted(unknown)}")
        merged = dict(self.defaults)
        merged.update(params)
        return merged

    @abstractmethod
    def setup(self, ctx: MissionContext) -> None: ...

    @abstractmethod
    def tick(self, ctx: MissionContext, i: int, t: float) -> None: ...

    def done(self, ctx: MissionContext) -> str | None:
        return None

    @abstractmethod
    def metrics(self, ctx: MissionContext) -> dict: ...


MISSIONS: dict[str, Callable[[], Mission]] = {}


def register_mission(cls):
    if not cls.name:
        raise ValueError("a mission class needs a name")
    MISSIONS[cls.name] = cls
    return cls


def _load_builtin_missions() -> None:
    # Importing a mission module registers it.  Missing modules are skipped so
    # the engine works while missions are still being added.
    import importlib
    for module in ("hover", "tag", "sweep", "relay", "monitor", "formation"):
        try:
            importlib.import_module(f"sim.missions.{module}")
        except ModuleNotFoundError as exc:
            if exc.name != f"sim.missions.{module}":
                raise


def run_mission(scenario: Scenario, *, log: bool = False, code_revision: str = "unrecorded") -> MissionResult:
    """Run one simulated mission and return its metrics (and log if asked)."""
    if scenario.mission not in MISSIONS:
        _load_builtin_missions()
    if scenario.mission not in MISSIONS:
        raise ValueError(f"unknown mission: {scenario.mission!r}")
    mission = MISSIONS[scenario.mission]()
    params = mission.validate_params(scenario.params)
    ctx = MissionContext(scenario, params, log=log, rng_names=mission.rng_names)
    mission.setup(ctx)
    for item in scenario.outages:
        streams = item["streams"]
        ctx.event(item["start_s"], "outage_start", None, {"streams": streams, "end_s": item["end_s"]})
        if item["end_s"] is not None and item["end_s"] <= scenario.duration_s:
            ctx.event(item["end_s"], "outage_end", None, {"streams": streams})
    end_reason = "timeout"
    steps = 0
    for i in range(ctx.n_steps + 1):
        t = i * DT
        mission.tick(ctx, i, t)
        ctx.collect_adapter_events()
        steps = i + 1
        reason = mission.done(ctx)
        if reason is not None:
            end_reason = reason
            break
    end_time = (steps - 1) * DT
    # Adapter events are stamped with the adapter clock, one step ahead of the
    # mission's t (time advances before the failsafe is evaluated), so the last
    # step's events sit at end_time + DT; the tolerance keeps them in.
    ctx.events = [e for e in ctx.events if e["time_s"] <= end_time + DT + 1e-9]
    ctx.event(end_time, "mission_end", None, {"reason": end_reason})
    metrics = dict(mission.metrics(ctx))
    metrics["end_reason"] = end_reason
    metrics["duration_s"] = round(end_time, 6)
    log_doc = None
    if ctx.recorder is not None:
        events = sorted(ctx.events, key=lambda e: (e["time_s"], e["kind"], e["agent"] or ""))
        link_stats = {}
        for stream_id in sorted(ctx.links):
            stats = ctx.links[stream_id].stats()
            lost = stats.get("lost_by_reason", {})
            link_stats[stream_id] = {"sent": stats["sent"], "delivered": stats["delivered"],
                                     "lost_loss": lost.get("loss", 0), "lost_outage": lost.get("outage", 0)}
        log_doc = ctx.recorder.to_log(
            mission=scenario.mission, scenario=scenario.as_dict(), code_revision=code_revision,
            adapter=scenario.adapter, preset=scenario.preset, seed=scenario.seed,
            policy=ctx.failsafe_config.as_dict(), limits=list(COMMON_LIMITS + tuple(mission.limits)),
            link_stats=link_stats, metrics=metrics,
        )
        log_doc["events"] = build_log(events=events)["events"]
    return MissionResult(metrics=metrics, log=log_doc, end_reason=end_reason, steps=steps)
