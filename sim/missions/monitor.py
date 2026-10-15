"""Simulated scenario of a planned use case: event safety monitoring."""
from __future__ import annotations

import math

import numpy as np

from sim.adapters import Geofence
from sim.estimation.base import EstimatorSettings
from sim.estimation.factory import make_estimator
from sim.sensor import SensorConfig, SensorModel
from sim.missions.engine import Mission, agent_frame, register_mission


VENUE_MIN = np.array([0.0, 0.0], dtype=float)
VENUE_MAX = np.array([60.0, 40.0], dtype=float)
ZONE_MIN = np.array([20.0, 12.0], dtype=float)
ZONE_MAX = np.array([40.0, 28.0], dtype=float)
POSTS = {
    "o1": np.array([5.0, 5.0, 12.0], dtype=float),
    "o2": np.array([55.0, 35.0, 12.0], dtype=float),
}


def _path_position(path: str, t: float) -> np.ndarray:
    """Return the scripted unknown-drone position at time ``t``."""
    time = max(0.0, float(t))
    if path == "crossing":
        return np.array([min(60.0, 3.0 * time), 20.0, 15.0], dtype=float)
    if path == "skimming":
        return np.array([min(60.0, 3.0 * time), 10.0, 15.0], dtype=float)
    if path == "loiter":
        if time <= 10.0:
            return np.array([3.0 * time, 20.0, 15.0], dtype=float)
        radial_time = 4.0 / 3.0
        if time <= 10.0 + radial_time:
            return np.array([30.0 + 3.0 * (time - 10.0), 20.0, 15.0], dtype=float)
        theta = 0.75 * (time - 10.0 - radial_time)
        return np.array([30.0 + 4.0 * math.cos(theta), 20.0 + 4.0 * math.sin(theta), 15.0], dtype=float)
    raise ValueError(f"unknown monitor path: {path!r}")


def _distance_to_zone(position: np.ndarray) -> float:
    """Horizontal distance from a point to the alert rectangle."""
    xy = np.asarray(position, dtype=float)[:2]
    delta = np.maximum(np.maximum(ZONE_MIN - xy, 0.0), xy - ZONE_MAX)
    return float(np.linalg.norm(delta))


def _inside_zone(position: np.ndarray) -> bool:
    xy = np.asarray(position, dtype=float)[:2]
    return bool(np.all(xy >= ZONE_MIN) and np.all(xy <= ZONE_MAX))


def _inside_venue(position: np.ndarray) -> bool:
    xy = np.asarray(position, dtype=float)[:2]
    return bool(np.all(xy >= VENUE_MIN) and np.all(xy <= VENUE_MAX))


@register_mission
class MonitorMission(Mission):
    """Simulated scenario of a planned use case for monitoring and alerts."""

    name = "monitor"
    rng_names = ("sensor_o1", "sensor_o2")
    limits = (
        "The unknown aircraft follows a scripted path at fixed altitude and speed.",
        "Observers hold their posts; this simulation does not model human movement.",
        "Sensor noise, link delay, jitter, and loss are synthetic assumptions.",
    )
    defaults = {"path": "crossing"}

    def validate_params(self, params: dict) -> dict:
        merged = super().validate_params(params)
        if merged["path"] not in {"crossing", "skimming", "loiter"}:
            raise ValueError("monitor path must be crossing, skimming, or loiter")
        return merged

    def _post_goal(self, observer):
        """Return an observer's fixed post and reject any changed goal."""
        observer_id = observer if isinstance(observer, str) else observer.adapter.adapter_id
        if observer_id not in POSTS:
            raise KeyError(f"unknown observer {observer_id!r}")
        expected = POSTS[observer_id]
        drone = self._ctx.drones[observer_id]
        current = getattr(drone, "_goal", None)
        if current is not None and not np.array_equal(np.asarray(current, dtype=float), expected):
            raise RuntimeError(f"observer {observer_id} goal changed from its post")
        return expected.copy()

    def setup(self, ctx):
        self._ctx = ctx
        self._path = ctx.params["path"]
        self._post_positions = {name: value.copy() for name, value in POSTS.items()}
        geofence = Geofence((0.0, 0.0, 0.0), (60.0, 40.0, 30.0), margin_m=0.5)
        for observer_id, post in POSTS.items():
            drone = ctx.add_drone(observer_id, initial_position_m=post, geofence=geofence, max_speed_mps=1.5)
            drone.set_goal(self._post_goal(observer_id))

        self._links = {
            observer_id: ctx.link(
                f"meas:{observer_id}",
                delay_s=ctx.scenario.measurement_delay_s,
                jitter_s=ctx.scenario.measurement_jitter_s,
            )
            for observer_id in POSTS
        }
        self._sensors = {
            observer_id: SensorModel(SensorConfig(noise_std_m=0.5), seed=ctx.seed_for(f"sensor_{observer_id}"))
            for observer_id in POSTS
        }
        sequence_by_observer = {observer_id: 0 for observer_id in POSTS}
        n_captures = int(math.floor(ctx.scenario.duration_s * 10.0 + 1e-9))
        for index in range(n_captures + 1):
            capture_time = index * 0.1
            truth = _path_position(self._path, capture_time)
            for observer_id in POSTS:
                sample = self._sensors[observer_id].sample(truth, capture_time)
                if sample.measurement_m is not None:
                    self._links[observer_id].send(
                        sequence_by_observer[observer_id],
                        capture_time,
                        tuple(float(value) for value in sample.measurement_m),
                    )
                    sequence_by_observer[observer_id] += 1

        self._estimator = None
        self._newest_capture = -math.inf
        self._estimate_position = None
        self._estimate_age = math.inf
        self._last_check_time = -math.inf
        self._inside_count = 0
        self._outside_count = 0
        self._alert_active = False
        self._alert_times = []
        self._alert_detection_times = []
        self._check_rows = []
        self._max_age = 0.0
        self._observer_max_displacement = 0.0
        self._true_entry_time = 20.0 / 3.0 if self._path in {"crossing", "loiter"} else None
        ctx.start_log(sorted(POSTS))

    def _consume_fixes(self, ctx, now: float) -> None:
        packets = []
        for observer_id, link in self._links.items():
            for packet in link.deliveries_until(now):
                packets.append((float(packet.send_time_s), float(packet.receive_time_s), observer_id, packet))
        packets.sort(key=lambda item: (item[0], item[1], item[2], int(item[3].seq)))
        for capture_time, _receive_time, _observer_id, packet in packets:
            measurement = np.asarray(packet.message, dtype=float)
            if self._estimator is None:
                self._estimator = make_estimator(
                    "cv", EstimatorSettings(measurement_std_m=0.5), measurement
                )
                self._newest_capture = capture_time
                self._estimate_position = measurement.copy()
                continue
            if capture_time <= self._newest_capture + 1e-12:
                continue
            try:
                self._estimator.update(measurement, capture_time)
            except ValueError:
                continue
            self._newest_capture = capture_time

    def _check_alert(self, ctx, now: float) -> None:
        if now - self._last_check_time < 0.099999:
            return
        self._last_check_time = now
        if self._estimator is None:
            self._estimate_position = None
            self._estimate_age = math.inf
        else:
            prediction = self._estimator.predict_at(max(now, self._estimator.timestamp_s))
            self._estimate_position = np.asarray(prediction.position_m, dtype=float)
            self._estimate_age = max(0.0, now - self._newest_capture)
        truth = _path_position(self._path, now)
        in_venue = _inside_venue(truth)
        self._check_rows.append((now, in_venue, self._estimate_age))
        if math.isfinite(self._estimate_age):
            self._max_age = max(self._max_age, self._estimate_age)
        estimate_inside = self._estimate_position is not None and _inside_zone(self._estimate_position)
        if estimate_inside:
            self._inside_count += 1
            self._outside_count = 0
        else:
            self._outside_count += 1
            self._inside_count = 0
        if not self._alert_active and self._inside_count >= 3:
            self._alert_active = True
            self._alert_detection_times.append(now)
            self._emit_alert(ctx, now, "alert_raised", self._estimate_position)
        elif self._alert_active and self._outside_count >= 3:
            self._alert_active = False
            self._emit_alert(ctx, now, "alert_cleared", self._estimate_position)

    def _emit_alert(self, ctx, now: float, kind: str, estimate_position) -> None:
        # The ground station raises the alert the moment its own check decides;
        # the link delay is already inside the estimate it decided on.
        truth = _path_position(self._path, now)
        ctx.event(now, kind, None, {
            "distance_to_zone_m": _distance_to_zone(truth),
            "estimate_position_m": [float(value) for value in estimate_position],
        })
        if kind == "alert_raised":
            self._alert_times.append(now)

    def tick(self, ctx, i: int, t: float) -> None:
        self._consume_fixes(ctx, t)
        states = {}
        for observer_id in POSTS:
            goal = self._post_goal(observer_id)
            if not np.array_equal(goal, self._post_positions[observer_id]):
                raise RuntimeError(f"observer {observer_id} goal changed from its post")
            state = ctx.drones[observer_id].tick(i)
            states[observer_id] = state
            displacement = float(np.linalg.norm(np.asarray(state.position_m, dtype=float) - POSTS[observer_id]))
            self._observer_max_displacement = max(self._observer_max_displacement, displacement)

        if abs((t * 10.0) - round(t * 10.0)) < 1e-9:
            self._check_alert(ctx, t)
        estimate = self._estimate_position
        age = self._estimate_age
        ctx.frame(
            i,
            t,
            _path_position(self._path, t),
            {observer_id: agent_frame(state, estimate_position_m=estimate, estimate_age_s=age)
             for observer_id, state in states.items()},
        )

    def metrics(self, ctx):
        inside_checks = [row for row in self._check_rows if row[1]]
        good_checks = sum(row[2] < 0.5 for row in inside_checks)
        continuity = 0.0 if not inside_checks else good_checks / len(inside_checks)
        first_alert = self._alert_times[0] if self._alert_times else None
        latency = None if first_alert is None or self._true_entry_time is None else first_alert - self._true_entry_time
        path = self._path
        false_alert = bool(
            (path == "skimming" and self._alert_detection_times)
            or any(_distance_to_zone(_path_position(path, event_time)) > 1.0
                   for event_time in self._alert_detection_times)
        )
        inside_duration = 0.0
        if self._true_entry_time is not None:
            end = min(ctx.scenario.duration_s, 40.0 / 3.0 if path == "crossing" else ctx.scenario.duration_s)
            inside_duration = max(0.0, end - self._true_entry_time)
        missed_alert = bool(inside_duration >= 2.0 and (first_alert is None or first_alert > self._true_entry_time + 5.0))
        max_age = self._max_age
        return {
            "continuity": float(continuity),
            "alert_latency_s": None if latency is None else float(latency),
            "alerts_raised": int(len(self._alert_times)),
            "false_alert": bool(false_alert),
            "missed_alert": bool(missed_alert),
            "max_estimate_age_s": float(max_age),
            "observer_max_displacement_m": float(self._observer_max_displacement),
        }
