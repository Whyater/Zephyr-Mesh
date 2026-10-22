"""Independent presentation contract checks against the canonical event log."""
from __future__ import annotations

import json
from pathlib import Path

from tools.desktop.presentation import estimate_error_p95, filter_group_rows, group_rows, present_run

ROOT = Path(__file__).parents[2]


def run():
    return json.loads((ROOT / "sim/runs/s7-swarm/run.json").read_text())


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


def test_filter_pins_only_late_and_lost_groups_when_query_misses():
    rows = [
        {"level": "alert", "drone_ids": ["drone-001"], "count": 1},
        {"level": "caution", "drone_ids": ["drone-004"], "count": 1},
        {"level": "waiting", "drone_ids": ["drone-002"], "count": 1},
        {"level": "unavailable", "drone_ids": ["drone-005"], "count": 1},
        {"level": "ok", "drone_ids": ["drone-003"], "count": 1},
    ]
    result = filter_group_rows(rows, "missing")
    assert [row["level"] for row in result] == ["alert", "caution"]


def test_filter_matches_compact_heading_and_raw_drone_names():
    """Unpinned groups use the query, and all three displayed names match."""
    rows = [
        {"level": "ok", "drone_id": "drone-007", "drone_ids": ["drone-007", "drone-017"], "count": 2},
        {"level": "ok", "drone_id": "drone-008", "drone_ids": ["drone-008"], "count": 1},
        {"level": "waiting", "drone_id": "drone-009", "drone_ids": ["drone-009"], "count": 1},
    ]
    only_seven = {"level": "ok", "drone_id": "drone-007", "drone_ids": ["drone-007"], "count": 1, "group_count": 2}
    assert filter_group_rows(rows, "D07") == [only_seven]
    assert [row["drone_ids"] for row in filter_group_rows(rows, "Drone 08")] == [["drone-008"]]
    assert filter_group_rows(rows, "drone-007") == [only_seven]
    assert [row["drone_ids"] for row in filter_group_rows(rows, "Drone 09")] == [["drone-009"]]
    unfiltered = filter_group_rows(rows, "  ")
    assert [row["drone_ids"] for row in unfiltered] == [row["drone_ids"] for row in rows]


def test_team_filter_counts_and_texts_for_the_caption_and_empty_state():
    from tools.desktop.presentation import team_filter

    groups = present_run(run())["frames"][3]["groups"]
    # Hand count: frame 3 has 3 late drones (always listed) and 47 OK drones.
    missing = team_filter(groups, "zzz")
    assert (missing["shown"], missing["matched"], missing["total"]) == (3, 0, 50)
    assert missing["caption"] == "Showing 3 of 50"
    assert missing["empty_text"] == "No drones match \u201czzz\u201d"
    one = team_filter(groups, "d07")
    assert (one["shown"], one["matched"]) == (4, 1)
    assert one["empty_text"] is None
    idle = team_filter(groups, "")
    assert (idle["shown"], idle["total"], idle["caption"]) == (50, 50, None)
    waiting = team_filter(present_run(run())["frames"][0]["groups"], "zzz")
    assert waiting["rows"] == [], "a waiting group is filtered like an OK group"


def test_one_group_per_status_with_age_range_and_stalest_first():
    """Frame 5 has six late drones at two ages; they form one Late group, not two."""
    document = run()
    frame = document["steps"][5]
    late_ages = sorted(
        (agent["target_age_s"] for agent in frame["agents"]
         if agent["target_age_s"] is not None and 0.15 <= agent["target_age_s"] < 0.3),
        reverse=True,
    )
    assert len(late_ages) == 6
    groups = present_run(document)["frames"][5]["groups"]
    late = [group for group in groups if group["level"] == "caution"]
    assert len(late) == 1
    assert late[0]["count"] == 6
    assert late[0]["target_age_s"] == {"min": round(late_ages[-1], 2), "max": round(late_ages[0], 2)}
    ages = {agent["agent_id"]: agent["target_age_s"] for agent in frame["agents"]}
    assert [round(ages[drone], 2) for drone in late[0]["drone_ids"]] == [round(age, 2) for age in late_ages]
    assert groups[0]["level"] != "ok", "non-OK groups still sort first"
    for presentation_frame in present_run(document)["frames"]:
        keys = [(group["level"], group["word"]) for group in presentation_frame["groups"]]
        assert len(keys) == len(set(keys))


def test_event_drone_list_is_capped_at_five_with_more_count():
    events = [event for frame in present_run(run())["frames"] for event in frame["events"]]
    late = next(event for event in events if event["text"].startswith("9 drones newly late"))
    assert late["text"] == "9 drones newly late: D06, D13, D20, D22, D23 +4 more (0.15 s)"
    assert len(late["drone_ids"]) == 9, "the event keeps every ID"
    short = next(event for event in events if event["text"].startswith("3 drones newly late"))
    assert "more" not in short["text"]


def test_run_level_presentation_fields():
    presentation = present_run(run())
    assert presentation["schema"] == "zephyr-s7-presentation-2"
    assert presentation["run_name"] == "50-drone ring, seed 17"
    assert presentation["not_in_replay"] == [
        "Live radio link",
        "Camera or vision input",
        "Battery state",
        "Flight controller commands",
        "Measured flight performance",
    ]
    summaries = [frame["team_summary"] for frame in presentation["frames"]]
    assert summaries[0] == "50 drones: 50 waiting"
    assert summaries[3] == "50 drones: 47 OK, 3 late"
    assert summaries[5] == "50 drones: 44 OK, 6 late"
    for frame in presentation["frames"]:
        assert frame["team_summary"].startswith(f"{len(frame['per_drone'])} drones")


def test_missing_neighbor_values_are_not_recorded_never_zero():
    from tools.desktop.presentation import per_drone_records
    from tools.desktop.replay import ReplayModel

    document = run()
    for agent in document["steps"][1]["agents"]:
        agent.pop("neighbor_count", None)
        agent.pop("min_neighbor_distance_m", None)
    record = per_drone_records(document, document["steps"][1])[0]
    assert record["neighbor_count"] is None
    assert record["nearest_neighbor_m"] is None
    snapshot = ReplayModel(document).frames[1].agents[0]
    assert snapshot.neighbor_count is None
    assert snapshot.min_neighbor_distance_m is None
    recorded = ReplayModel(run()).frames[1].agents[0]
    assert recorded.neighbor_count == run()["steps"][1]["agents"][0]["neighbor_count"]


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
    from tools.desktop.presentation import compact_name, heading_name

    assert compact_name("drone-007") == "D07"
    assert heading_name("drone-007") == "Drone 07"
    assert compact_name("drone-050") == "D50"
    assert heading_name("drone-050") == "Drone 50"
    assert compact_name("unknown") == "unknown"
    assert heading_name("unknown") == "unknown"


def test_golden_is_current():
    expected = json.dumps(present_run(run()), indent=2, sort_keys=True) + "\n"
    assert expected == (ROOT / "tools/design/golden/s7_presentation.json").read_text()


def _golden_events():
    golden = json.loads((ROOT / "tools/design/golden/s7_presentation.json").read_text())
    return [event for frame in golden["frames"] for event in frame["events"]]


def test_routine_packet_loss_is_never_an_alert_event():
    events = [e for e in _golden_events() if e["text"].startswith("Target packet lost")]
    assert events
    assert all(e["level"] == "info" for e in events)
    assert not any(e["level"] == "alert" for e in events)


def test_every_golden_event_level_is_a_complete_status_level():
    from tools.desktop.design_tokens import UI_TEXT

    levels = {e["level"] for e in _golden_events()}
    assert "info" in levels
    for level in levels:
        spec = UI_TEXT["statusLevels"][level]
        assert spec["word"] and spec["shape"] and spec["color"]
