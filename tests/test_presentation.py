"""Independent presentation contract checks against the canonical event log."""
from __future__ import annotations

import json
from pathlib import Path

from desktop.presentation import estimate_error_p95, filter_group_rows, group_rows, present_run

ROOT = Path(__file__).parents[1]


def run():
    return json.loads((ROOT / "runs/s7-swarm/run.json").read_text())


def test_target_loss_is_hand_counted_from_raw_events_per_frame_window():
    document = run()
    presentation = present_run(document)
    times = [frame["time_s"] for frame in document["steps"]]
    target = [event for event in document["link_events"] if event["sender"] == "__target__"]
    for index, time_s in enumerate(times):
        lower = times[index - 1] if index else None
        expected = {
            event["receiver"]
            for event in target
            if lower is not None
            and event["outcome"] == "lost"
            and lower < event["send_time"] <= time_s
        }
        actual = {
            drone_id
            for event in presentation["frames"][index]["events"]
            if event["text"].startswith("Target packet lost")
            for drone_id in event["drone_ids"]
        }
        assert actual == expected


def test_cumulative_target_loss_percent_is_hand_computed_at_frames_two_and_five():
    """Cumulative loss percentages agree with a separate raw-event loop."""
    document = run()
    presentation = present_run(document)
    target_events = [
        event for event in document["link_events"] if event["sender"] == "__target__"
    ]
    frame_times = [frame["time_s"] for frame in document["steps"]]
    expected = {}
    for index in (2, 5):
        received = 0
        lost = 0
        for event in target_events:
            if event["outcome"] == "received":
                observed = event["receive_time"] <= frame_times[index]
            else:
                observed = event["send_time"] <= frame_times[index]
            if observed:
                received += event["outcome"] == "received"
                lost += event["outcome"] != "received"
        expected[index] = round(lost / (received + lost) * 100, 2)
    assert expected == {2: 25.81, 5: 14.83}
    assert presentation["frames"][2]["strip"]["target_packet_loss_pct"] == expected[2]
    assert presentation["frames"][5]["strip"]["target_packet_loss_pct"] == expected[5]


def test_all_six_headlines_are_explicitly_checked():
    """Each canonical frame has a stable headline contract."""
    headlines = [frame["headline"]["text"] for frame in present_run(run())["frames"]]
    assert headlines == [
        "Waiting for first target packet",
        "Waiting for first target packet",
        "Tracking partial (46 of 50)",
        "Tracking held",
        "Tracking held",
        "Tracking held",
    ]


def test_cumulative_link_stats_wait_for_first_arrival():
    presentation = present_run(run())
    assert presentation["frames"][0]["strip"]["target_link_delay_ms"] is None
    assert presentation["frames"][0]["strip"]["target_packet_loss_pct"] is None
    assert presentation["frames"][2]["strip"]["target_link_delay_ms"] == 80


def test_frame_three_age_counts_are_read_directly():
    document = run()
    frame = document["steps"][3]
    ages = [agent["target_age_s"] for agent in frame["agents"] if agent["target_age_s"] is not None]
    assert sum(age >= 0.15 for age in ages) == 3
    assert sum(age < 0.15 for age in ages) == 47
    groups = present_run(document)["frames"][3]["groups"]
    assert sum(group["count"] for group in groups) == 50
    assert sum(group["count"] for group in groups if group["level"] == "caution") == 3


def test_grouping_never_hides_non_ok_drones():
    document = run()
    for frame in document["steps"]:
        rows = group_rows(document, frame)
        ids = {drone for row in rows if row["level"] not in {"ok", "unavailable"} for drone in row["drone_ids"]}
        expected = {
            agent["agent_id"]
            for agent in frame["agents"]
            if agent["target_age_s"] is not None and agent["target_age_s"] >= 0.15
        }
        assert expected <= ids


def test_filter_keeps_non_ok_groups_when_query_misses():
    rows = [
        {"level": "alert", "drone_ids": ["drone-001"]},
        {"level": "waiting", "drone_ids": ["drone-002"]},
        {"level": "ok", "drone_ids": ["drone-003"]},
    ]
    result = filter_group_rows(rows, "missing")
    assert [row["level"] for row in result] == ["alert", "waiting"]


def test_filter_matches_compact_heading_and_raw_drone_names():
    """Only OK groups use the query, and all three displayed names match."""
    rows = [
        {"level": "ok", "drone_ids": ["drone-007"]},
        {"level": "ok", "drone_ids": ["drone-008"]},
        {"level": "unavailable", "drone_ids": ["drone-009"]},
    ]
    assert filter_group_rows(rows, "D07") == [rows[0], rows[2]]
    assert filter_group_rows(rows, "Drone 08") == [rows[1], rows[2]]
    assert filter_group_rows(rows, "drone-007") == [rows[0], rows[2]]


def test_group_range_has_hand_checked_minimum_and_maximum():
    """A group range is checked against its member values."""
    frame = present_run(run())["frames"][2]
    ok_group = next(group for group in frame["groups"] if group["level"] == "ok")
    assert ok_group["target_packet_loss_pct"] == {"min": 0.0, "max": 50.0}


def test_error_p95_on_tiny_document():
    frame = {
        "target_position_m": [0, 0, 0],
        "agents": [
            {"active": True, "fused_target_m": [0, 0, 0]},
            {"active": True, "fused_target_m": [1, 0, 0]},
            {"active": True, "fused_target_m": [2, 0, 0]},
            {"active": True, "fused_target_m": [3, 0, 0]},
        ],
    }
    assert estimate_error_p95(frame) == 2.85


def test_went_inactive_event_uses_shared_template_on_hand_document():
    """An active to inactive transition produces one identified event."""
    previous = {"time_s": 0.0, "agents": [{"agent_id": "drone-007", "active": True}]}
    current = {"time_s": 0.1, "agents": [{"agent_id": "drone-007", "active": False}]}
    run_document = {"frames": [previous, current], "link_events": []}
    events = present_run(run_document)["frames"][1]["events"]
    assert events == [
        {
            "time_s": 0.1,
            "level": "alert",
            "text": "D07 went inactive",
            "drone_ids": ["drone-007"],
        }
    ]


def test_drone_names_use_integer_suffixes():
    from desktop.presentation import compact_name, heading_name

    assert compact_name("drone-007") == "D07"
    assert heading_name("drone-007") == "Drone 07"
    assert compact_name("drone-050") == "D50"
    assert heading_name("drone-050") == "Drone 50"
    assert compact_name("unknown") == "unknown"
    assert heading_name("unknown") == "unknown"


def test_golden_is_current():
    expected = json.dumps(present_run(run()), indent=2, sort_keys=True) + "\n"
    assert expected == (ROOT / "design/golden/s7_presentation.json").read_text()
