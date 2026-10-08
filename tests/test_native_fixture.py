"""Parity checks for the native cockpit's checked-in S7 projection."""

import json
from pathlib import Path


def test_native_fixture_matches_canonical_s7_frames_and_provenance():
    root = Path(__file__).parents[1]
    canonical = json.loads((root / "runs/s7-swarm/run.json").read_text())
    native = json.loads((root / "macos/ZephyrMeshApp/Sources/ZephyrMeshApp/Resources/demo_swarm.json").read_text())
    assert native["schema"] == "zephyr-s7-native-fixture-1"
    assert native["run_id"] == canonical["scenario_id"]
    assert native["seed"] == canonical["seed"]
    assert native["command_authority"] == canonical["command_authority"] == "replay only"
    assert native["failsafe"] == canonical["failsafe"]
    assert native["source"] == "canonical Python S7 event-log projection"
    assert "flight performance" in native["evidence_boundary"].lower()
    assert {"live radio", "camera", "flight performance"}.issubset(set(native["unavailable"]))
    assert len(native["frames"]) == len(canonical["steps"])
    for native_frame, canonical_step in zip(native["frames"], canonical["steps"]):
        assert native_frame["time"] == canonical_step["time_s"]
        assert [drone["id"] for drone in native_frame["drones"]] == [agent["agent_id"] for agent in canonical_step["agents"]]
        for native_drone, canonical_agent in zip(native_frame["drones"], canonical_step["agents"]):
            assert native_drone["position"] == canonical_agent["position_m"]
            assert native_drone["velocity"] == canonical_agent["velocity_mps"]
            assert native_drone["profile_id"] == canonical_agent["profile_id"]
