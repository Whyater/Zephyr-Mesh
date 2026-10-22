"""Parity checks for the generated native cockpit projection."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tools/macos/ZephyrMeshApp/Sources/ZephyrMeshApp/Resources/demo_swarm.json"
GOLDEN = ROOT / "tools/design/golden/s7_presentation.json"


def test_native_fixture_regenerates_byte_identically(tmp_path):
    output = tmp_path / "fixture.json"
    result = subprocess.run(
        [sys.executable, "tools/build_native_fixture.py", "--output", str(output)],
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert output.read_bytes() == FIXTURE.read_bytes()


def test_native_fixture_embeds_the_golden_presentation():
    canonical = json.loads((ROOT / "runs/s7-swarm/run.json").read_text(encoding="utf-8"))
    native = json.loads(FIXTURE.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert native["schema"] == "zephyr-s7-native-fixture-4"
    assert native["run_id"] == canonical["scenario_id"]
    assert native["seed"] == canonical["seed"]
    embedded = {
        "schema": "zephyr-s7-presentation-2",
        "run_id": native["run_id"],
        "run_name": native["run_name"],
        "not_in_replay": native["not_in_replay"],
        "about": native["about"],
        "frames": [frame["presentation"] for frame in native["frames"]],
    }
    assert embedded == golden
    for frame in native["frames"]:
        for drone in frame["drones"]:
            assert "link_delay_ms" not in drone
            assert "link_loss_pct" not in drone


def test_native_fixture_per_drone_fields_match_source_or_presentation():
    """Every native per-drone field has a direct canonical source."""
    canonical = json.loads((ROOT / "runs/s7-swarm/run.json").read_text(encoding="utf-8"))
    native = json.loads(FIXTURE.read_text(encoding="utf-8"))
    from desktop.presentation import present_run

    presentation = present_run(canonical)
    source_fields = {
        "id": "agent_id",
        "position": "position_m",
        "velocity": "velocity_mps",
        "profile_id": "profile_id",
        "active": "active",
        "neighbor_count": "neighbor_count",
        "min_neighbor_distance_m": "min_neighbor_distance_m",
        "constraint_flags": "constraint_flags",
        "target_estimate_m": "target_estimate_m",
        "fused_target_m": "fused_target_m",
        "target_source_time_s": "target_source_time_s",
        "target_age_s": "target_age_s",
    }

    def assert_projection_matches(
        drone: dict[str, object], source_agents: dict[str, dict[str, object]]
    ) -> None:
        """Compare one native projection with its source agent."""
        source = source_agents[drone["id"]]
        for native_field, source_field in source_fields.items():
            assert drone[native_field] == source[source_field]

    first_agents = {
        agent["agent_id"]: agent for agent in canonical["steps"][0]["agents"]
    }
    for drone in native["drones"]:
        assert_projection_matches(drone, first_agents)

    for frame_index, frame in enumerate(native["frames"]):
        source_agents = {
            agent["agent_id"]: agent for agent in canonical["steps"][frame_index]["agents"]
        }
        expected_records = {
            record["drone_id"]: record
            for record in presentation["frames"][frame_index]["per_drone"]
        }
        assert frame["presentation"]["per_drone"] == list(expected_records.values())
        for drone in frame["drones"]:
            assert_projection_matches(drone, source_agents)
            assert expected_records[drone["id"]]["drone_id"] == drone["id"]
