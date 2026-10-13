"""Pure presentation calculations shared by the desktop replay surfaces."""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping

from .design_tokens import UI_TEXT


LEVEL_ORDER = {"alert": 0, "caution": 1, "unavailable": 2, "waiting": 3, "ok": 4}
SUMMARY_ORDER = ("ok", "caution", "alert", "waiting", "unavailable")
PRESENTATION_SCHEMA = "zephyr-s7-presentation-2"


def _agents(frame: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the agent records from either replay spelling."""
    return list(frame.get("agents", frame.get("drones", [])))


def _id(agent: Mapping[str, Any]) -> str:
    """Read the stable agent identifier."""
    return str(agent.get("agent_id", agent.get("id", "")))


def _age(agent: Mapping[str, Any]) -> float | None:
    """Read a target age in seconds."""
    raw = agent.get("target_age_s", agent.get("target_age"))
    return float(raw) if raw is not None else None


def _has_estimate(agent: Mapping[str, Any]) -> bool:
    """Return whether an agent has a fused target estimate."""
    return agent.get("fused_target_m", agent.get("fused_target")) is not None


def _status_spec(level: str) -> Mapping[str, Any]:
    """Return a status definition from the generated UI text."""
    return UI_TEXT["statusLevels"][level]


def _status(level: str, *, alternate: bool = False) -> dict[str, Any]:
    """Build the display status record from generated UI text."""
    spec = _status_spec(level)
    word = spec.get("alternateWord") if alternate else spec["word"]
    return {"level": level, "word": word or spec["word"], "shape": spec["shape"], "color": spec["color"]}


def status_for_agent(agent: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one agent using the display thresholds in the design tokens."""
    if not bool(agent.get("active", True)):
        return _status("alert", alternate=True)
    if not _has_estimate(agent):
        return _status("waiting")
    age = _age(agent)
    if age is None:
        return _status("waiting")
    late = float(UI_TEXT["displayThresholds"]["targetAgeLateS"]["value"])
    lost = float(UI_TEXT["displayThresholds"]["targetAgeLostS"]["value"])
    if age >= lost:
        return _status("alert")
    if age >= late:
        return _status("caution")
    return _status("ok")


def _observed(event: Mapping[str, Any], frame_time: float) -> bool:
    """Return whether an event contributes to cumulative frame statistics."""
    if event.get("outcome") == "received":
        receive_time = event.get("receive_time")
        return receive_time is not None and float(receive_time) <= frame_time
    send_time = event.get("send_time")
    return send_time is not None and float(send_time) <= frame_time


def _stats(events: Iterable[Mapping[str, Any]], frame_time: float) -> dict[str, Any]:
    """Calculate cumulative delay and loss through one frame time."""
    observed = [event for event in events if _observed(event, frame_time)]
    received = [event for event in observed if event.get("outcome") == "received"]
    if not received:
        return {"delay_ms": None, "loss_pct": None, "received": 0, "total": 0, "lost": 0}
    ages = [float(event["packet_age"]) for event in received if event.get("packet_age") is not None]
    lost = sum(event.get("outcome") != "received" for event in observed)
    total = len(received) + lost
    delay = sum(ages) / len(ages) * 1000 if ages else None
    loss = lost / total * 100 if total else None
    return {"delay_ms": delay, "loss_pct": loss, "received": len(received), "total": total, "lost": lost}


def _all_drone_ids(run: Mapping[str, Any]) -> list[str]:
    """Collect stable IDs from every frame."""
    ids = {_id(agent) for frame in run.get("steps", run.get("frames", [])) for agent in _agents(frame)}
    return sorted(ids)


def link_stats(run: Mapping[str, Any], frame_time: float | None = None) -> dict[str, Any]:
    """Calculate cumulative team and per-drone link stats at ``frame_time``."""
    frames = run.get("steps", run.get("frames", []))
    if frame_time is None:
        frame_time = max((float(frame.get("time_s", frame.get("time", 0))) for frame in frames), default=0.0)
    events = list(run.get("link_events", []))
    target = [event for event in events if event.get("sender") == "__target__"]
    neighbor = [event for event in events if event.get("sender") != "__target__"]
    result: dict[str, Any] = {
        "team": {"target": _stats(target, frame_time), "neighbor": _stats(neighbor, frame_time)},
        "drones": {},
    }
    for receiver in _all_drone_ids(run):
        result["drones"][receiver] = {
            "target": _stats((event for event in target if event.get("receiver") == receiver), frame_time),
            "neighbor": _stats((event for event in neighbor if event.get("receiver") == receiver), frame_time),
        }
    return result


def _p95(values: list[float]) -> float | None:
    """Return the linearly interpolated 95th percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 12)


def estimate_error_p95(frame: Mapping[str, Any]) -> float | None:
    """Calculate p95 fused-target error for one frame."""
    target = frame.get("target_position_m", frame.get("target_position"))
    if target is None:
        return None
    errors = []
    for agent in _agents(frame):
        fused = agent.get("fused_target_m", agent.get("fused_target"))
        if fused is not None:
            errors.append(math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(fused, target))))
    return _p95(errors)


def _round_metric(value: float | None, digits: int) -> float | None:
    """Round a metric while preserving an unavailable value."""
    if value is None:
        return None
    return round(float(value), digits)


def _frame_stats(run: Mapping[str, Any], frame: Mapping[str, Any]) -> dict[str, Any]:
    """Build the team strip for one frame."""
    frame_time = float(frame.get("time_s", frame.get("time", 0)))
    stats = link_stats(run, frame_time)
    ages = [_age(agent) for agent in _agents(frame) if _age(agent) is not None]
    target = stats["team"]["target"]
    neighbor = stats["team"]["neighbor"]
    return {
        "target_link_delay_ms": _round_metric(target["delay_ms"], 1),
        "target_packet_loss_pct": _round_metric(target["loss_pct"], 2),
        "neighbor_link_delay_ms": _round_metric(neighbor["delay_ms"], 1),
        "neighbor_packet_loss_pct": _round_metric(neighbor["loss_pct"], 2),
        "newest_target_age_s": round(min(ages), 2) if ages else None,
        "oldest_target_age_s": round(max(ages), 2) if ages else None,
    }


def tracking_headline(frame: Mapping[str, Any]) -> dict[str, str]:
    """Build the headline from current agent estimates."""
    agents = _agents(frame)
    active = [agent for agent in agents if bool(agent.get("active", True))]
    estimated = [agent for agent in active if _has_estimate(agent)]
    lost_age = float(UI_TEXT["displayThresholds"]["targetAgeLostS"]["value"])
    on_time = [agent for agent in active if _age(agent) is not None and _age(agent) < lost_age]
    if not estimated:
        return {"text": "Waiting for first target packet", "level": "waiting"}
    if len(on_time) == len(active):
        return {"text": "Tracking held", "level": "ok"}
    if all(_age(agent) is None or _age(agent) >= lost_age for agent in active):
        return {"text": "Tracking lost", "level": "alert"}
    return {"text": f"Tracking partial ({len(on_time)} of {len(active)})", "level": "caution"}


def _numbered_name(drone_id: str, heading: bool) -> str:
    """Format a compact or heading name from the last number in an ID."""
    match = re.search(r"(\d+)(?!.*\d)", drone_id)
    if match is None:
        return drone_id
    number = int(match.group(1))
    return f"Drone {number:02d}" if heading else f"D{number:02d}"


def compact_name(drone_id: str) -> str:
    """Return the compact display name for an agent ID."""
    return _numbered_name(drone_id, heading=False)


def heading_name(drone_id: str) -> str:
    """Return the heading display name for an agent ID."""
    return _numbered_name(drone_id, heading=True)


def _vector(agent: Mapping[str, Any], key: str) -> list[float]:
    """Read a numeric vector, defaulting to zeroes."""
    raw = agent.get(key) or [0, 0, 0]
    return [float(value) for value in raw]


def _drone_record(agent: Mapping[str, Any], stats: Mapping[str, Any], target: Any) -> dict[str, Any]:
    """Build the complete per-drone presentation record."""
    drone_id = _id(agent)
    status = status_for_agent(agent)
    target_stats = stats.get("target", {})
    neighbor_stats = stats.get("neighbor", {})
    position = _vector(agent, "position_m")
    velocity = _vector(agent, "velocity_mps")
    horizontal_speed = math.hypot(velocity[0], velocity[1])
    estimate = agent.get("fused_target_m", agent.get("fused_target"))
    error = None
    if estimate is not None and target is not None:
        error = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(estimate, target)))
    heading = math.degrees(math.atan2(velocity[1], velocity[0])) if horizontal_speed >= 0.01 else None
    return {
        "drone_id": drone_id,
        "compact_name": compact_name(drone_id),
        "heading_name": heading_name(drone_id),
        "level": status["level"],
        "word": status["word"],
        "shape": status["shape"],
        "color": status["color"],
        "target_age_s": round(_age(agent), 2) if _age(agent) is not None else None,
        "estimate_error_m": round(error, 3) if error is not None else None,
        "target_link_delay_ms": _round_metric(target_stats.get("delay_ms"), 1),
        "target_packet_loss_pct": _round_metric(target_stats.get("loss_pct"), 2),
        "neighbor_link_delay_ms": _round_metric(neighbor_stats.get("delay_ms"), 1),
        "neighbor_packet_loss_pct": _round_metric(neighbor_stats.get("loss_pct"), 2),
        "altitude_m": round(position[2], 3),
        "speed_mps": round(math.sqrt(sum(value * value for value in velocity)), 3),
        "heading_deg": round(heading, 1) if heading is not None else None,
        # A missing count is "not recorded", never zero neighbors.
        "neighbor_count": int(agent["neighbor_count"]) if agent.get("neighbor_count") is not None else None,
        "nearest_neighbor_m": (
            round(float(agent["min_neighbor_distance_m"]), 3)
            if agent.get("min_neighbor_distance_m") is not None
            else None
        ),
        "constraint_flags": list(agent.get("constraint_flags", [])),
        "profile_id": agent.get("profile_id"),
        "active": bool(agent.get("active", True)),
    }


def _range(values: list[Any], digits: int) -> Any:
    """Return one value or a JSON range for a group metric."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    low = round(float(min(present)), digits)
    high = round(float(max(present)), digits)
    return low if low == high and len(present) == len(values) else {"min": low, "max": high}


def _stalest_first(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Sort key: oldest target age first, unrecorded ages last, then drone ID."""
    age = record.get("target_age_s")
    return (age is None, -(age or 0.0), str(record["drone_id"]))


def group_rows(run: Mapping[str, Any], frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Group rows by status level and word only; ages, delays, and losses become ranges."""
    frame_time = float(frame.get("time_s", frame.get("time", 0)))
    stats = link_stats(run, frame_time)["drones"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    target = frame.get("target_position_m", frame.get("target_position"))
    for agent in _agents(frame):
        record = _drone_record(agent, stats.get(_id(agent), {}), target)
        key = (record["level"], record["word"])
        groups.setdefault(key, []).append(record)
    rows = []
    for (level, word), members in groups.items():
        members.sort(key=_stalest_first)
        status = members[0]
        rows.append({
            "drone_id": members[0]["drone_id"],
            "drone_ids": [item["drone_id"] for item in members],
            "count": len(members),
            "level": level,
            "word": word,
            "shape": status["shape"],
            "color": status["color"],
            "target_age_s": _range([item["target_age_s"] for item in members], 2),
            "target_link_delay_ms": _range(
                [item["target_link_delay_ms"] for item in members], 1
            ),
            "target_packet_loss_pct": _range(
                [item["target_packet_loss_pct"] for item in members], 2
            ),
        })
    rows.sort(key=lambda row: (LEVEL_ORDER.get(row["level"], 99), row["drone_ids"][0]))
    return rows


def pinned_levels() -> tuple[str, ...]:
    """Status levels that stay listed whatever the team filter says (late and lost)."""
    return tuple(UI_TEXT["teamFilter"]["pinnedLevels"])


def drone_matches(drone_id: str, query: str) -> bool:
    """Match a query against the raw ID, the compact name, and the heading name."""
    needle = query.strip().lower()
    if not needle:
        return True
    names = (drone_id, compact_name(drone_id), heading_name(drone_id))
    return any(needle in name.lower() for name in names)


def filter_group_rows(rows: Iterable[Mapping[str, Any]], query: str) -> list[dict[str, Any]]:
    """Apply the team filter to group rows.

    Pinned rows (late and lost) keep every member. Every other row keeps only
    the members whose names match, and disappears when none match. Each
    returned row keeps the unfiltered size in ``group_count``.
    """
    needle = query.strip()
    pinned = pinned_levels()
    result = []
    for row in rows:
        ids = [str(item) for item in row.get("drone_ids", [])]
        shown = ids if not needle or row.get("level") in pinned else [
            drone_id for drone_id in ids if drone_matches(drone_id, needle)
        ]
        if not shown:
            continue
        item = dict(row)
        item["group_count"] = int(row.get("count", len(ids)))
        item["drone_ids"] = shown
        item["drone_id"] = shown[0]
        item["count"] = len(shown)
        result.append(item)
    return result


def team_filter(rows: Iterable[Mapping[str, Any]], query: str) -> dict[str, Any]:
    """Filtered rows plus the counts and texts for the filter caption and empty state.

    ``shown`` counts every listed drone (pinned ones included), ``matched``
    counts drones whose names match the query, and ``total`` counts every
    drone in the frame. The caption and empty text are ``None`` with no query.
    """
    all_rows = list(rows)
    filtered = filter_group_rows(all_rows, query)
    needle = query.strip()
    total = sum(int(row.get("count", len(row.get("drone_ids", [])))) for row in all_rows)
    shown = sum(len(row["drone_ids"]) for row in filtered)
    matched = sum(
        drone_matches(str(drone_id), needle) for row in all_rows for drone_id in row.get("drone_ids", [])
    ) if needle else total
    texts = UI_TEXT["teamFilter"]
    return {
        "rows": filtered,
        "shown": shown,
        "matched": matched,
        "total": total,
        "caption": texts["showing"].format(shown=shown, total=total) if needle else None,
        "empty_text": texts["noMatches"].format(query=needle) if needle and matched == 0 else None,
    }


def team_summary(per_drone: Iterable[Mapping[str, Any]]) -> str:
    """The one-line team summary, for example ``50 drones: 47 OK, 3 late``."""
    records = list(per_drone)
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        level = str(record.get("level") or "unavailable")
        word = str(record.get("word") or _status_spec(level if level in UI_TEXT["statusLevels"] else "unavailable")["word"])
        counts[(level, word)] = counts.get((level, word), 0) + 1
    rank = {level: index for index, level in enumerate(SUMMARY_ORDER)}
    ordered = sorted(counts.items(), key=lambda item: (rank.get(item[0][0], len(rank)), item[0][1]))
    parts = [f"{count} {word if word.isupper() else word.lower()}" for (_level, word), count in ordered]
    text = UI_TEXT["teamSummary"]
    head = f"{len(records)} {text['drone'] if len(records) == 1 else text['drones']}"
    return text["template"].format(head=head, parts=", ".join(parts)) if parts else head


def run_name(run: Mapping[str, Any]) -> str:
    """Name the run in plain words, for example ``50-drone ring, seed 17``."""
    frames = run.get("steps", run.get("frames", []))
    first = frames[0] if frames else {}
    count = len(_agents(first)) if isinstance(first, Mapping) else 0
    identifier = str(run.get("scenario_id", run.get("run_id", ""))).lower()
    text = UI_TEXT["runName"]
    words = re.findall(r"[a-z0-9]+", identifier)
    shape = text["ringShape"] if "ring" in words else text["otherShape"]
    name = text["template"].format(count=count, shape=shape)
    seed = run.get("seed")
    return text["withSeed"].format(name=name, seed=seed) if seed is not None else name


def not_in_replay(run: Mapping[str, Any]) -> list[str]:
    """What this replay does not contain, checked against the data where the run could carry it."""
    agents = [agent for frame in run.get("steps", run.get("frames", [])) for agent in _agents(frame)]
    items = []
    for spec in UI_TEXT["notInReplay"]["items"]:
        field = spec.get("unlessRecorded")
        if field and any(agent.get(field) is not None for agent in agents):
            continue
        items.append(str(spec["text"]))
    return items


def _event(time_s: float, level: str, text: str, drone_ids: Iterable[str] = ()) -> dict[str, Any]:
    """Build one timeline event."""
    return {
        "time_s": round(time_s, 2),
        "level": level,
        "text": text,
        "drone_ids": sorted(set(drone_ids)),
    }


def _event_text(template: str, **values: Any) -> str:
    """Format one event from the shared JSON template table."""
    return str(UI_TEXT["eventTemplates"][template]).format(**values)


def _drone_list(drone_ids: list[str]) -> str:
    """Compact names for an event text, capped with "+N more"; the event keeps every ID."""
    spec = UI_TEXT["eventDroneList"]
    limit = int(spec["limit"])
    names = ", ".join(compact_name(drone_id) for drone_id in drone_ids[:limit])
    extra = len(drone_ids) - limit
    return f"{names} {spec['more'].format(count=extra)}" if extra > 0 else names


def _lost_receivers(run: Mapping[str, Any], lower: float, upper: float) -> list[str]:
    """Find receivers with a lost target packet in a frame window."""
    receivers = {
        str(event["receiver"])
        for event in run.get("link_events", [])
        if event.get("sender") == "__target__"
        and event.get("outcome") == "lost"
        and event.get("send_time") is not None
        and lower < float(event["send_time"]) <= upper
    }
    return sorted(receivers)


def frame_events(
    run: Mapping[str, Any],
    index: int,
    previous: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build events for state changes and packet losses in this frame window."""
    frames = run.get("steps", run.get("frames", []))
    frame = frames[index]
    agents = _agents(frame)
    current = {_id(agent): status_for_agent(agent) for agent in agents}
    time_s = float(frame.get("time_s", frame.get("time", 0)))
    events = []
    if index == 0:
        events.append(_event(time_s, "info", _event_text("replayLoaded", count=len(agents))))
    if previous is None:
        return events
    lower = float(previous.get("time_s", previous.get("time", 0)))
    lost_ids = _lost_receivers(run, lower, time_s)
    if lost_ids:
        events.append(
            _event(
                time_s,
                "info",
                _event_text("packetLost", count=len(lost_ids)),
                lost_ids,
            )
        )
    before = {_id(agent): status_for_agent(agent) for agent in _agents(previous)}
    old_estimated = sum(
        bool(agent.get("active", True)) and _has_estimate(agent)
        for agent in _agents(previous)
    )
    new_estimated = sum(
        bool(agent.get("active", True)) and _has_estimate(agent)
        for agent in agents
    )
    if new_estimated > old_estimated:
        active_count = sum(bool(agent.get("active", True)) for agent in agents)
        events.append(
            _event(
                time_s,
                "ok",
                _event_text("estimateReached", estimated=new_estimated, total=active_count),
            )
        )
    late = {_id(agent) for agent in agents if current[_id(agent)]["level"] == "caution"}
    old_late = {_id(agent) for agent in _agents(previous) if before[_id(agent)]["level"] == "caution"}
    newly_late = sorted(late - old_late)
    recovered = sorted(old_late - late)
    if newly_late:
        maximum_age = max(_age(agent) or 0 for agent in agents if _id(agent) in newly_late)
        compact_ids = _drone_list(newly_late)
        events.append(
            _event(
                time_s,
                "caution",
                _event_text(
                    "estimateLate",
                    count=len(newly_late),
                    drones=compact_ids,
                    age=maximum_age,
                ),
                newly_late,
            )
        )
    if recovered:
        events.append(
            _event(
                time_s,
                "ok",
                _event_text("backOnTime", count=len(recovered)),
                recovered,
            )
        )
    for drone_id in sorted(set(current) & set(before)):
        before_active = next(agent for agent in _agents(previous) if _id(agent) == drone_id).get("active", True)
        current_active = next(agent for agent in agents if _id(agent) == drone_id).get("active", True)
        if bool(before_active) and not bool(current_active):
            events.append(
                _event(
                    time_s,
                    "alert",
                    _event_text("inactive", drone=compact_name(drone_id)),
                    [drone_id],
                )
            )
    return events


def per_drone_records(run: Mapping[str, Any], frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one complete presentation record for every drone in a frame."""
    frame_time = float(frame.get("time_s", frame.get("time", 0)))
    stats = link_stats(run, frame_time)["drones"]
    target = frame.get("target_position_m", frame.get("target_position"))
    return [_drone_record(agent, stats.get(_id(agent), {}), target) for agent in _agents(frame)]


def about_content(run: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the run fields allowed in the about view."""
    fields = UI_TEXT["honesty"]["aboutBodyFields"]
    return {field: run[field] for field in fields if field in run}


def frame_presentation(
    run: Mapping[str, Any],
    index: int,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the presentation object for one frame."""
    frames = run.get("steps", run.get("frames", []))
    frame = frames[index]
    per_drone = per_drone_records(run, frame)
    return {
        "frame": index,
        "time_s": round(float(frame.get("time_s", frame.get("time", 0))), 2),
        "headline": tracking_headline(frame),
        "estimate_error_p95_m": (
            round(estimate_error_p95(frame), 3)
            if estimate_error_p95(frame) is not None
            else None
        ),
        "strip": _frame_stats(run, frame),
        "groups": group_rows(run, frame),
        "per_drone": per_drone,
        "team_summary": team_summary(per_drone),
        "events": frame_events(run, index, previous),
    }


def present_run(run: Mapping[str, Any]) -> dict[str, Any]:
    """Build the canonical presentation document."""
    frames = run.get("steps", run.get("frames", []))
    return {
        "schema": PRESENTATION_SCHEMA,
        "run_id": run.get("scenario_id", run.get("run_id")),
        "run_name": run_name(run),
        "not_in_replay": not_in_replay(run),
        "about": about_content(run),
        "frames": [
            frame_presentation(run, index, frames[index - 1] if index else None)
            for index in range(len(frames))
        ],
    }
