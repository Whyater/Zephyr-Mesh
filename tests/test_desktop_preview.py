"""Pure-model checks for the cross-platform desktop replay preview."""
from pathlib import Path

import pytest

from desktop.replay import ReplayFormatError, ReplayModel, filter_agents


ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs/s7-swarm/run.json"


def test_replay_model_reads_canonical_s7_fixture():
    model = ReplayModel.from_path(RUN)
    assert model.schema == "zephyr-s7-swarm-run-1"
    assert model.scenario_id == "s7-ring-50-seed-17"
    assert model.frame_count == 6
    assert len(model.frame.agents) == 50
    assert model.frame.active_count == 50
    assert model.command_authority == "replay only"
    assert model.failsafe.startswith("synthetic constraints")


def test_replay_cursor_clamps_and_reset_is_deterministic():
    model = ReplayModel.from_path(RUN)
    first = model.frame
    assert model.step(-10) == first
    assert model.step(100).step_index == 5
    assert model.progress == 1.0
    assert model.reset() == first
    assert model.index == 0
    assert model.summary()["frame"] == "1/6"


def test_replay_model_rejects_missing_steps(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema":"zephyr-s7-swarm-run-1","scenario_id":"bad","seed":1,"dt_s":0.05,"status":"synthetic","command_authority":"replay only","failsafe":"synthetic","evidence_boundary":"fixture"}', encoding="utf-8")
    with pytest.raises(ReplayFormatError, match="steps"):
        ReplayModel.from_path(path)


def test_replay_preserves_optional_telemetry_without_inventing_defaults():
    model = ReplayModel.from_path(RUN)
    agent = model.frame.agents[0]
    assert agent.battery_pct is None
    assert agent.link_delay_ms is None


def test_replay_exposes_parts_and_observed_link_metrics():
    model = ReplayModel.from_path(RUN)
    profile = model.profile_for("drone-001")
    assert profile is not None
    assert profile.motor_count == 4
    assert profile.estimated_max_thrust_n > 0
    link = model.link_stats_for("drone-001")
    assert link is not None
    assert link.event_count == 300
    assert link.loss_count == 7
    assert link.mean_delay_ms is not None
    assert link.mean_delay_ms > 0
    assert model.summary()["profile_count"] == 50


def test_windows_mission_events_mirror_native_replay_vocabulary():
    from desktop.windows_preview import mission_event_lines

    model = ReplayModel.from_path(RUN)
    lines = mission_event_lines(model, "drone-001")

    assert lines[0].startswith("INFO  Replay manifest loaded")
    assert [
        line[6:].strip().split(" · ", 1)[0]
        for line in lines
    ] == [
        "Replay manifest loaded",
        "Command authority locked",
        "Failsafe boundary recorded",
        "Frame 0 committed",
        "Separation margin observed",
    ]


def test_windows_mission_events_cover_warning_branches_and_missing_selection(monkeypatch):
    from dataclasses import replace
    from desktop.replay import LinkStats
    from desktop.windows_preview import mission_event_lines

    model = ReplayModel.from_path(RUN)
    low_confidence = replace(model.frame.agents[0], confidence=0.9)
    model.frames = (replace(model.frame, agents=(low_confidence,) + model.frame.agents[1:]),) + model.frames[1:]
    monkeypatch.setattr(
        model,
        "link_stats_for",
        lambda _agent_id: LinkStats("receiver", 10, 9, 0.08, 0.08),
    )

    lines = mission_event_lines(model, "missing-agent")
    assert any("Estimator confidence changed" in line for line in lines)
    assert any("Link envelope visible" in line for line in lines)
    assert not any("Separation margin observed" in line for line in lines)


def test_fleet_filter_matches_ids_and_profile_ids_without_reordering():
    model = ReplayModel.from_path(RUN)
    agents = model.frame.agents[:3]
    assert [agent.agent_id for agent in filter_agents(agents, "DRONE-002")] == ["drone-002"]
    assert [agent.agent_id for agent in filter_agents(agents, "synthetic-profile")] == [
        "drone-001",
        "drone-002",
        "drone-003",
    ]
    assert filter_agents(agents, "   ") == agents


def test_windows_runtime_version_prefers_frozen_internal_resource(tmp_path, monkeypatch):
    import desktop.windows_preview as preview

    internal = tmp_path / "_internal"
    internal.mkdir()
    (internal / "VERSION.txt").write_text("v0.2.0", encoding="utf-8")
    monkeypatch.setattr(preview, "RESOURCE_ROOTS", (internal, tmp_path))

    assert preview._runtime_version() == "0.2.0"


def test_windows_build_writes_utf8_version_without_bom():
    script = (ROOT / "desktop" / "build_windows.ps1").read_text(encoding="utf-8")
    assert "UTF8Encoding" in script
    assert "WriteAllText" in script
    assert "hidden-import desktop.evidence" in script


def test_windows_requirements_pin_evidence_loader_dependencies():
    requirements = (ROOT / "requirements-windows.txt").read_text(encoding="utf-8")
    for dependency in ("attrs==", "jsonschema==", "jsonschema-specifications==", "referencing==", "rpds-py=="):
        assert dependency in requirements


def test_windows_smoke_packages_and_checks_investigation_report():
    build_script = (ROOT / "desktop" / "build_windows.ps1").read_text(encoding="utf-8")
    validator = (ROOT / "tools" / "validate_windows_bundle.py").read_text(encoding="utf-8")
    preview = (ROOT / "desktop" / "windows_preview.py").read_text(encoding="utf-8")
    assert "investigation_report_example.json;desktop" in build_script
    assert '"investigation_report_example.json", "investigation", "synthetic"' in validator
    assert 'action="append"' in preview
    assert '("s7_report_example.json", "swarm", "synthetic")' in validator
    assert "evidence_kind=espnow,investigation,swarm" in validator


def test_windows_release_job_runs_python_suite_before_packaging():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    windows_job = workflow.split("  windows:\n", 1)[1].split("  publish:\n", 1)[0]
    assert "pip install --requirement requirements.txt" in windows_job
    assert "python -m pytest -q" in windows_job
    assert "run: python tools/validate_windows_bundle.py dist/ZephyrMeshWindows --gui-smoke" in windows_job
    assert "run: py tools/validate_windows_bundle.py" not in windows_job
    assert windows_job.index("python -m pytest -q") < windows_job.index("Build and package the Windows preview")


def test_windows_glyph_rotation_keeps_quad_geometry_bounded():
    from desktop.windows_preview import WindowsReplayApp

    points = WindowsReplayApp._rotated_points(0.0, 0.20, 0.11)
    assert points[0] == (0.20, 0.0)
    assert points[2] == (-0.20, 0.0)
    rotated = WindowsReplayApp._rotated_points(1.5707963267948966, 0.20, 0.11)
    assert abs(rotated[0][0]) < 1e-9
    assert abs(rotated[0][1] - 0.20) < 1e-9


def test_windows_preview_constructs_when_tk_display_is_available():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no desktop display in this test environment")
    root.withdraw()
    from desktop.windows_preview import WindowsReplayApp
    app = WindowsReplayApp(root, ReplayModel.from_path(RUN))
    assert app.selected_id == "drone-001"
    root.destroy()
