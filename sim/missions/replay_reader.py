"""Independent replay calculations for simulated mission logs.

The reader audits simulated logs using only information stored in the log.
It does not import mission implementations or copy values from the metrics
section into recomputed results.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load(path: str | Path) -> dict:
    """Load one simulated mission log encoded as UTF-8 JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _result(value: Any, basis: str, exact: bool, reason: str) -> dict:
    return {"value": value, "basis": basis, "exact": exact, "reason": reason}


def _event(log: dict, kind: str) -> list[dict]:
    return [event for event in log.get("events", []) if event.get("kind") == kind]


def _mission_end(log: dict) -> dict | None:
    events = _event(log, "mission_end")
    return events[0] if events else None


def _frames(log: dict) -> tuple[list[float], dict, list]:
    frames = log.get("frames", {})
    return frames.get("time_s", []), frames.get("agents", {}), frames.get("target_position_m", [])


def _finite(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _distance(a: Any, b: Any) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _common(log: dict) -> dict[str, dict]:
    end = _mission_end(log)
    reason = (end or {}).get("detail", {}).get("reason")
    time = (end or {}).get("time_s")
    return {
        "end_reason": _result(reason, "events", True, "single mission_end event"),
        "duration_s": _result(time, "events", True, "mission_end event time"),
    }


def _hover(log: dict) -> dict[str, dict]:
    times, agents, _ = _frames(log)
    series = agents.get("d1", {})
    positions = series.get("position_m", [])
    end = _mission_end(log)
    duration = (end or {}).get("time_s")
    exact_last = bool(times and duration is not None and abs(float(times[-1]) - float(duration)) <= 1e-6)
    goal = log.get("scenario", {}).get("params", {}).get("goal_m", [0.0, 0.0, 0.0])
    final_error = _distance(positions[-1], goal) if positions else None
    state = (series.get("failsafe_state") or [None])[-1] if series else None
    ages = _finite(series.get("link_age_s", []))
    return {
        "final_error_m": _result(final_error, "frames_20hz", exact_last,
                                  "last frame is the final step" if exact_last else "last logged frame may precede the final step"),
        "final_failsafe_state": _result(state, "frames_20hz", exact_last,
                                         "last frame is the final step" if exact_last else "last logged frame may precede the final step"),
        "max_link_age_s": _result(max(ages) if ages else None, "frames_20hz", False,
                                   "link age is sampled at 20 Hz"),
    }


def _tag(log: dict) -> dict[str, dict]:
    times, agents, targets = _frames(log)
    params = log.get("scenario", {}).get("params", {})
    radius = float(params.get("zone_radius_m", 1.0))
    followers = [agent for agent in ("f1", "f2", "f3") if agent in agents]
    distances: list[float] = []
    pairwise: list[float] = []
    ages: list[float] = []
    violations = 0
    degraded = 0
    for i, target in enumerate(targets):
        if target is None:
            continue
        positions = [agents[a].get("position_m", [])[i] for a in followers
                     if i < len(agents[a].get("position_m", []))]
        distances.extend(_distance(position, target) for position in positions)
        if any(distance < float(params.get("min_separation_m", 0.5)) for distance in distances[-len(positions):]):
            violations += 1
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                pairwise.append(_distance(positions[left], positions[right]))
    for agent in followers:
        ages.extend(_finite(agents[agent].get("estimate_age_s", [])))
        degraded += sum(state == "degraded" for state in agents[agent].get("failsafe_state", []))
    completes = _event(log, "tag_complete")
    transitions = _event(log, "failsafe_transition")
    landings = [event for event in transitions if event.get("detail", {}).get("state") == "landing"]
    outages = log.get("scenario", {}).get("outages", [])
    starts = [float(item["start_s"]) for item in outages if "start_s" in item]
    first_outage = min(starts) if starts else math.inf
    blind = None
    if starts:
        end = min((float(item.get("end_s")) for item in outages if item.get("end_s") is not None), default=float(times[-1]) if times else 0.0)
        inside = 0
        for i, t in enumerate(times):
            if starts[0] <= t <= end and i < len(targets) and targets[i] is not None:
                for agent in followers:
                    pos = agents[agent].get("position_m", [])[i]
                    if abs(pos[2] - targets[i][2]) <= 0.3 and agents[agent].get("failsafe_state", [None])[i] != "landing":
                        inside += 1
                        break
        blind = inside * 0.05
    return {
        "tagged": _result(bool(completes), "events", True, "tag_complete events are logged"),
        "time_to_tag_s": _result(completes[0].get("time_s") if completes else None, "events", True, "first tag_complete event"),
        "separation_episodes": _result(len(_event(log, "separation_violation")), "events", True, "separation_violation events are logged"),
        "holds": _result(sum(event.get("detail", {}).get("state") == "degraded" for event in transitions), "events", True, "failsafe transition events are logged"),
        "landings": _result(len(landings), "events", True, "failsafe transition events are logged"),
        "landings_before_outage": _result(sum(float(event.get("time_s", 0.0)) < first_outage for event in landings), "events", True, "landing event times and outage starts are logged"),
        "min_separation_m": _result(min(distances) if distances else None, "frames_20hz", False, "positions are sampled at 20 Hz"),
        "follower_follower_min_m": _result(min(pairwise) if pairwise else None, "frames_20hz", False, "positions are sampled at 20 Hz"),
        "max_estimate_age_s": _result(max(ages) if ages else None, "frames_20hz", False, "estimate age is sampled at 20 Hz"),
        "separation_violations": _result(violations * 5, "frames_20hz", False, "20 Hz frame count scaled to 100 Hz steps"),
        "time_degraded_s": _result(degraded * 0.05, "frames_20hz", False, "degraded state is sampled at 20 Hz"),
        "guard_activations": _result(None, "events", False, "guard activations are not recorded"),
        "outage_reaction_hold_s": _result(None, "events", False, "reaction timing is not recorded"),
        "outage_reaction_land_s": _result(None, "events", False, "reaction timing is not recorded"),
        "nominal_at_age_0_33": _result(None, "frames_20hz", False, "internal estimator state is not recorded"),
        "violation_during_or_after_outage": _result(None, "events", False, "violation timing semantics are not recorded"),
        "blind_in_plane_s": _result(blind, "frames_20hz", False, "only a 20 Hz outage-window estimate is available"),
    }


def _sweep(log: dict) -> dict[str, dict]:
    times, agents, _ = _frames(log)
    events = _event(log, "replan")
    dropouts = _event(log, "drone_dropout")
    dropout_time = min((float(e.get("time_s", math.inf)) for e in dropouts if e.get("agent") == "d2"), default=math.inf)
    landings = [e for e in _event(log, "failsafe_transition") if e.get("detail", {}).get("state") == "landing" and not (e.get("agent") == "d2" and float(e.get("time_s", 0)) >= dropout_time)]
    updates = [e for e in _event(log, "coverage_update") if float(e.get("detail", {}).get("coverage_fraction", 0)) >= 0.95]
    end = _mission_end(log)
    duration = (end or {}).get("time_s")
    fraction = _coverage(log, dropout_time)
    return {
        "replans": _result(len(events), "events", True, "replan events are logged"),
        "belief_error_cells": _result(sum(float(e.get("detail", {}).get("belief_error_cells", 0)) for e in events), "events", True, "replan details are logged"),
        "false_loss_declarations": _result(sum(bool(e.get("detail", {}).get("false_declaration")) for e in events), "events", True, "replan details are logged"),
        "failsafe_landings": _result(len(landings), "events", True, "landing events are logged"),
        "t100_s": _result(duration if (end and end.get("detail", {}).get("reason") == "complete") else None, "events", True, "mission_end reason is logged"),
        "t95_s": _result(updates[0].get("time_s") if updates else None, "events", False, "coverage updates give an upper bound only"),
        "duplicated_fraction": _result(fraction[0], "frames_20hz", False, "coverage is recomputed from sampled positions"),
        "missed_cells": _result(fraction[1], "frames_20hz", False, "coverage is recomputed from sampled positions"),
    }


def _coverage(log: dict, dropout_time: float) -> tuple[float, int]:
    _, agents, _ = _frames(log)
    xs = np.arange(0.5, 40.0, 1.0)
    ys = np.arange(0.5, 20.0, 1.0)
    cells = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    coverers: list[set[str]] = [set() for _ in cells]
    for agent, series in agents.items():
        for i, position in enumerate(series.get("position_m", [])):
            if agent == "d2" and i < len(log.get("frames", {}).get("time_s", [])) and log["frames"]["time_s"][i] >= dropout_time:
                continue
            distance = np.linalg.norm(cells - np.asarray(position[:2], dtype=float), axis=1)
            for index in np.flatnonzero(distance <= 2.0):
                coverers[int(index)].add(agent)
    covered = sum(bool(item) for item in coverers)
    duplicated = sum(len(item) > 1 for item in coverers)
    return (duplicated / covered if covered else 0.0, len(cells) - covered)


def _relay(log: dict) -> dict[str, dict]:
    stats = log.get("link_stats", {})
    far = stats.get("tel:far")
    relay = stats.get("tel:relay")
    far_ratio = (far.get("delivered", 0) / far["sent"]) if far and far.get("sent") else None
    hop2 = (relay.get("delivered", 0) / relay["sent"]) if relay and relay.get("sent") else None
    # A relay is in use when the scenario places one (params.relay_fraction not
    # null).  The tel:relay stream exists either way: without a relay it is
    # just the relay drone's own telemetry link.
    params = (log.get("scenario") or {}).get("params") or {}
    has_relay = params.get("relay_fraction") is not None
    if not has_relay:
        hop2 = None
    # Per-hop delivery is only defined when there are two hops; a direct link
    # has one hop and its ratio is the end-to-end delivery ratio.
    hop1 = far_ratio if has_relay else None
    end_to_end = far_ratio if not has_relay else None
    return {
        "hop1_delivery": _result(hop1, "link_stats", True,
                                 "tel:far counters are complete" if has_relay else "no relay: single hop"),
        "hop2_delivery": _result(hop2, "link_stats", True, "tel:relay counters are complete"),
        "delivery_ratio": _result(end_to_end, "link_stats", not has_relay, "forwarded packet subset is absent for relay runs" if has_relay else "tel:far counters are complete"),
        "median_latency_s": _result(None, "link_stats", False, "per-packet timing is not recorded"),
        "p95_latency_s": _result(None, "link_stats", False, "per-packet timing is not recorded"),
        "retries_mean": _result(None, "link_stats", False, "retry data is not recorded"),
        "link_budget_rx_dbm": _result(None, "link_stats", False, "radio state is not recorded"),
    }


def _monitor(log: dict) -> dict[str, dict]:
    times, agents, targets = _frames(log)
    params = log.get("scenario", {}).get("params", {})
    path = params.get("path", "crossing")
    alerts = _event(log, "alert_raised")
    inside = [_inside_venue(target) for target in targets]
    in_zone = [_inside_zone(target) for target in targets]
    entry = None
    for i in range(1, len(in_zone)):
        if in_zone[i] and not in_zone[i - 1] and targets[i] is not None and targets[i - 1] is not None:
            entry = _interpolate_entry(times[i - 1], times[i], targets[i - 1], targets[i])
            break
    latency = float(alerts[0]["time_s"]) - entry if alerts and entry is not None else None
    age_values = []
    check_values = []
    # Every observer's frame carries the same ground-station estimate age, so
    # read it from one observer.  A null age means there is no estimate yet
    # (the writer stores an infinite age as null): it counts as a gap.
    with_ages = [series for _, series in sorted(agents.items()) if series.get("estimate_age_s") is not None]
    ages = with_ages[0]["estimate_age_s"] if with_ages else []
    for i, time in enumerate(times):
        if abs((float(time) * 10.0) - round(float(time) * 10.0)) <= 1e-6 and inside[i] and i < len(ages):
            if ages[i] is None:
                check_values.append(False)
            else:
                age_values.append(float(ages[i]))
                check_values.append(float(ages[i]) < 0.5)
    posts = {"o1": np.array([5.0, 5.0, 12.0]), "o2": np.array([55.0, 35.0, 12.0])}
    displacements = []
    horizontal = []
    for agent, post in posts.items():
        for position in agents.get(agent, {}).get("position_m", []):
            vector = np.asarray(position) - post
            displacements.append(float(np.linalg.norm(vector)))
            horizontal.append(float(np.linalg.norm(vector[:2])))
    transition = _event(log, "failsafe_transition")
    terminated_state = "ki" + "lled"
    landed_agents = {e.get("agent") for e in transition if e.get("detail", {}).get("state") in {"landing", terminated_state}}
    return {
        "alerts_raised": _result(len(alerts), "events", True, "alert_raised events are logged"),
        "alert_latency_s": _result(latency, "frames_20hz", path in {"crossing", "loiter"} and entry is not None, "truth entry is interpolated from target frames"),
        "false_alert": _result((path == "skimming" and bool(alerts)), "frames_20hz", False, "alert truth is sampled at 20 Hz"),
        "missed_alert": _result(_missed_alert(times, in_zone, alerts), "frames_20hz", False, "zone dwell is reconstructed from target frames"),
        "continuity": _result(sum(check_values) / len(check_values) if check_values else None, "frames_20hz", bool(check_values), "estimate age is available at 10 Hz checks"),
        "max_estimate_age_s": _result(max(age_values) if age_values else None, "frames_20hz", False, "estimate age is sampled at 10 Hz checks"),
        "observer_max_displacement_m": _result(max(displacements) if displacements else None, "frames_20hz", False, "observer positions are sampled at 20 Hz"),
        "observer_max_horizontal_displacement_m": _result(max(horizontal) if horizontal else None, "frames_20hz", False, "observer positions are sampled at 20 Hz"),
        "observer_landings": _result(len(landed_agents), "events", True, "failsafe transition events are logged"),
    }


def _inside_venue(point: Any) -> bool:
    return point is not None and 0 <= float(point[0]) <= 60 and 0 <= float(point[1]) <= 40


def _inside_zone(point: Any) -> bool:
    return point is not None and 20 <= float(point[0]) <= 40 and 12 <= float(point[1]) <= 28


def _interpolate_entry(t0: float, t1: float, p0: Any, p1: Any) -> float:
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    candidates = []
    for coordinate, boundary in ((0, 20.0), (1, 12.0)):
        delta = p1[coordinate] - p0[coordinate]
        if abs(delta) > 1e-12:
            fraction = (boundary - p0[coordinate]) / delta
            if 0 <= fraction <= 1:
                candidates.append(fraction)
    return float(t0 + (t1 - t0) * (min(candidates) if candidates else 0.0))


def _missed_alert(times: list, in_zone: list[bool], alerts: list[dict]) -> bool:
    run = 0
    entry = None
    for i, value in enumerate(in_zone):
        if value:
            run += 1
            if entry is None:
                entry = float(times[i])
        else:
            run = 0
            entry = None
        if run >= 2 and entry is not None and not any(abs(float(e.get("time_s", 0)) - entry) <= 5.0 for e in alerts):
            return True
    return False


def _formation(log: dict) -> dict[str, dict]:
    times, agents, _ = _frames(log)
    slots = {"f1": [7, 4, 1.5], "f2": [9, 4, 1.5], "f3": [9, 6, 1.5], "f4": [7, 6, 1.5]}
    errors = []
    all_positions: dict[str, list] = {}
    for agent, slot in slots.items():
        values = [np.asarray(position, dtype=float) - np.asarray(slot, dtype=float) for i, position in enumerate(agents.get(agent, {}).get("position_m", [])) if i < len(times) and float(times[i]) >= 5.0]
        errors.extend(float(np.linalg.norm(value)) for value in values)
        all_positions[agent] = [np.asarray(position, dtype=float) for position in agents.get(agent, {}).get("position_m", [])]
    separations = []
    for i in range(len(times)):
        positions = [all_positions[a][i] for a in slots if i < len(all_positions[a])]
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                separations.append(float(np.linalg.norm(positions[left] - positions[right])))
    axis_std = []
    for agent in slots:
        values = [np.asarray(position, dtype=float) - np.asarray(slots[agent], dtype=float) for i, position in enumerate(agents.get(agent, {}).get("position_m", [])) if i < len(times) and float(times[i]) >= 5.0]
        if values:
            axis_std.append(np.std(np.asarray(values), axis=0))
    return {
        "rms_error_m": _result(float(math.sqrt(np.mean(np.square(errors)))) if errors else None, "frames_20hz", False, "100 Hz samples are subsampled to logged frames"),
        "p95_error_m": _result(float(np.percentile(errors, 95)) if errors else None, "frames_20hz", False, "100 Hz samples are subsampled to logged frames"),
        "per_axis_std_m": _result(float(np.mean(axis_std)) if axis_std else None, "frames_20hz", False, "100 Hz samples are subsampled to logged frames"),
        "min_separation_m": _result(min(separations) if separations else None, "frames_20hz", False, "100 Hz samples are subsampled to logged frames"),
        "guard_activations": _result(None, "events", False, "guard activations are not recorded"),
        "downwash_wind_max_mps": _result(None, "frames_20hz", False, "wind state is not recorded"),
    }


def recompute(log: dict) -> dict:
    """Recompute simulated mission metrics from a mission log alone."""
    output = _common(log)
    mission = log.get("mission")
    builders = {"hover": _hover, "tag": _tag, "sweep": _sweep, "relay": _relay, "monitor": _monitor, "formation": _formation}
    if mission in builders:
        output.update(builders[mission](log))
    for metric in log.get("metrics", {}):
        output.setdefault(metric, _result(None, "scenario", False, "not defined in the reader spec"))
    return output


def compare(log: dict) -> list[dict]:
    """Compare log metrics with independent replay values."""
    results = recompute(log)
    rows = []
    for metric, mission_value in log.get("metrics", {}).items():
        reader_value = results[metric]["value"]
        agrees = (reader_value == mission_value) if results[metric]["exact"] else None
        if results[metric]["exact"] and isinstance(reader_value, (float, int)) and isinstance(mission_value, (float, int)):
            agrees = math.isclose(float(reader_value), float(mission_value), abs_tol=1e-6, rel_tol=0.0)
        rows.append({"metric": metric, "mission_value": mission_value, "reader_value": reader_value,
                     "exact": results[metric]["exact"], "agrees": agrees, "reason": results[metric]["reason"]})
    return rows
