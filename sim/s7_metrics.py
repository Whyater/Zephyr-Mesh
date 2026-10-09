"""Evidence-bounded metrics for the deterministic S7 swarm event log."""
from __future__ import annotations

from collections import Counter, defaultdict
from math import floor, fsum, isfinite
from typing import Any, Mapping

SUMMARY_SCHEMA = "zephyr-s7-swarm-summary-1"


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return (1.0 - fraction) * ordered[lower] + fraction * ordered[upper]


def _distribution(values: list[float], unit: str) -> dict[str, Any]:
    if not values:
        return {"available": False, "unit": unit, "count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    return {"available": True, "unit": unit, "count": len(values), "min": min(values), "mean": fsum(values) / len(values), "p50": _quantile(values, .5), "p95": _quantile(values, .95), "max": max(values)}


def _outcome(event: Mapping[str, Any]) -> str:
    receive = event.get("receive_time")
    loss_reason = event.get("loss_reason")
    packet_age = event.get("packet_age")
    if receive is not None:
        if loss_reason is not None or packet_age is None:
            raise ValueError("received event must have packet_age and no loss_reason")
        return "received"
    if loss_reason:
        if packet_age is not None:
            raise ValueError("lost event must not have packet_age")
        return "lost"
    if packet_age is not None:
        raise ValueError("unknown event must not have packet_age")
    return "unknown"


def _reacquisition(availability: list[bool | None], dt_s: float) -> dict[str, Any]:
    intervals: list[float] = []
    censored = 0
    index = 0
    while index < len(availability):
        if availability[index] is None:
            censored += 1
            index += 1
            continue
        if availability[index]:
            index += 1
            continue
        start = index
        while index < len(availability) and availability[index] is False:
            index += 1
        end = index
        if start == 0 or end == len(availability):
            censored += 1
        else:
            intervals.append((end - start) * dt_s)
    return {"bracketed_intervals": _distribution(intervals, "s"), "censored_outages": censored, "definition": "false fused-target availability intervals on active observations bracketed by true observations; inactive or missing-agent observations and unbracketed start/end gaps are censored"}


def summarize_s7_run(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return descriptive metrics without inferring tolerance or safety."""
    if document.get("schema") != "zephyr-s7-swarm-run-1":
        raise ValueError("schema must be zephyr-s7-swarm-run-1")
    steps = document.get("steps")
    events = document.get("link_events")
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a non-empty list")
    if not isinstance(events, list):
        raise ValueError("link_events must be a list")
    dt_s = float(document.get("dt_s"))
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    outcomes = Counter(_outcome(event) for event in events)
    by_route: dict[str, dict[str, Any]] = {}
    route_events: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("each link event must be an object")
        route_events[f"{event.get('sender')}->{event.get('receiver')}"].append(event)
    delay_values: list[float] = []
    for route, route_rows in sorted(route_events.items()):
        route_delays = [float(row["packet_age"]) for row in route_rows if _outcome(row) == "received" and row.get("packet_age") is not None]
        delay_values.extend(route_delays)
        counts = Counter(_outcome(row) for row in route_rows)
        by_route[route] = {"attempted": len(route_rows), "received": counts["received"], "lost": counts["lost"], "unknown": counts["unknown"], "abstract_event_delay_s": _distribution(route_delays, "s")}
    all_agents = [agent for step in steps for agent in step.get("agents", [])]
    active_agents = [agent for agent in all_agents if agent.get("active")]
    estimate_count = sum(agent.get("target_estimate_m") is not None for agent in active_agents)
    fused_count = sum(agent.get("fused_target_m") is not None for agent in active_agents)
    age_values = [float(agent["target_age_s"]) for agent in active_agents if agent.get("target_age_s") is not None]
    flags = Counter(flag for agent in active_agents for flag in (agent.get("constraint_flags") or []))
    separations = [float(agent["min_neighbor_distance_m"]) for agent in active_agents if agent.get("min_neighbor_distance_m") is not None]
    agent_ids = sorted({agent.get("agent_id") for agent in all_agents})
    recovery = {}
    for agent_id in agent_ids:
        availability = []
        for step in steps:
            observed = next((agent for agent in step.get("agents", []) if agent.get("agent_id") == agent_id), None)
            availability.append(None if observed is None or not observed.get("active") else observed.get("fused_target_m") is not None)
        recovery[agent_id] = _reacquisition(availability, dt_s)
    return {
        "schema": SUMMARY_SCHEMA,
        "evidence_boundary": "descriptive synthetic S7 event-log summary; not radio tolerance, safety, or flight performance",
        "provenance": {"status": "synthetic", "source_schema": "zephyr-s7-swarm-run-1", "timestamp_basis": "abstract replay timestamps; packet_age is not a measured radio clock"},
        "run": {"scenario_id": document.get("scenario_id"), "seed": document.get("seed"), "frame_count": len(steps), "agent_count": len(agent_ids), "dt_s": dt_s, "time_span_s": float(steps[-1].get("time_s", 0.0)) - float(steps[0].get("time_s", 0.0))},
        "link_events": {"attempted": len(events), "received": outcomes["received"], "lost": outcomes["lost"], "unknown": outcomes["unknown"], "abstract_event_delay_s": _distribution(delay_values, "s"), "routes": by_route},
        "estimation": {"active_agent_observations": len(active_agents), "target_estimate_present": estimate_count, "fused_target_present": fused_count, "target_age_s": _distribution(age_values, "s"), "recovery": recovery},
        "constraints": {"flag_counts": dict(sorted(flags.items())), "observed_min_neighbor_distance_m": _distribution(separations, "m")},
    }
