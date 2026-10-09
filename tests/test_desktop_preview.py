"""Pure-model checks for the cross-platform desktop replay preview."""
from pathlib import Path
import sys
import types

import pytest

from desktop.replay import ReplayFormatError, ReplayModel, filter_agents
from desktop.scenario_run import ScenarioFrame, ScenarioRunCursor, clamp_scenario_rail_x, load_scenario_run, project_scenario_altitude, project_scenario_altitude_radius, project_scenario_point, scenario_frame_geometry, scenario_rail_endpoints, selected_frame_summary


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


def test_scenario_run_decoder_reads_summary_and_frames(tmp_path):
    from sim.scenario_runner import run_scenario, serialize_run
    from desktop.scenario_run import load_scenario_run

    config = {
        "agent_count": 2,
        "radius_m": 2.0,
        "altitude_m": 1.5,
        "steps": 3,
        "seed": 7,
        "target_link": {"delay_s": 0.0, "delay_jitter_s": 0.0, "loss_probability": 0.0},
        "neighbor_link": {"delay_s": 0.0, "delay_jitter_s": 0.0, "loss_probability": 0.0},
    }
    path = tmp_path / "scenario.json"
    path.write_bytes(serialize_run(run_scenario(config, code_revision="test")))
    run = load_scenario_run(path)
    assert run.summary()["schema"] == "zephyr-s7-scenario-run-1"
    assert run.summary()["scenario_hash"] == run.document["scenario_hash"]
    assert run.summary()["payload_sha256"] == run.document["payload_sha256"]
    assert run.summary()["dt_s"] == run.document["parameters"]["dt_s"]
    assert run.summary()["generator"] == run.document["provenance"]["generator"]
    assert run.summary()["python"] == run.document["provenance"]["python"]
    assert run.summary()["numpy"] == run.document["provenance"]["numpy"]
    assert run.summary()["code_revision_semantics"].startswith("base revision")
    assert run.summary()["dt_s"] == run.provenance["dt_s"]
    assert run.summary()["generator"] == "sim.scenario_runner"
    assert run.summary()["python"] == run.provenance["python"]
    assert run.summary()["numpy"] == run.provenance["numpy"]
    assert run.summary()["code_revision"] == run.provenance["code_revision"]
    assert run.summary()["frame_count"] == 3
    assert run.frames[0].agent_count == 2
    assert "not establish" in run.summary()["evidence_boundary"]


def test_scenario_run_decoder_rejects_duplicate_keys_and_hash_mismatch(tmp_path):
    import json
    from sim.scenario_runner import run_scenario
    from desktop.scenario_run import ScenarioRunFormatError, load_scenario_run

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"zephyr-s7-scenario-run-1","schema":"zephyr-s7-scenario-run-1"}', encoding="utf-8")
    with pytest.raises(ScenarioRunFormatError, match="duplicate"):
        load_scenario_run(duplicate)
    config = {"agent_count": 1, "radius_m": 2.0, "altitude_m": 1.5, "steps": 1, "seed": 7, "target_link": {"delay_s": 0.0, "delay_jitter_s": 0.0, "loss_probability": 0.0}, "neighbor_link": {"delay_s": 0.0, "delay_jitter_s": 0.0, "loss_probability": 0.0}}
    document = run_scenario(config, code_revision="test")
    document["scenario_hash"] = "0" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ScenarioRunFormatError, match="scenario_hash"):
        load_scenario_run(tampered)


def test_windows_cli_exposes_scenario_run_inspection_path():
    import desktop.windows_preview as preview

    source = (ROOT / "desktop" / "windows_preview.py").read_text(encoding="utf-8")
    assert "--scenario-run" in source
    assert "Open scenario run" in source
    assert "Integrity and provenance" in source
    assert "project_scenario_altitude" in source
    assert "altitude m" in source
    assert "selected_frame_summary(run, cursor)" in source
    assert "load_scenario_run(args.scenario_run).summary()" in source
    assert callable(preview.load_scenario_run)


def test_desktop_scenario_decoder_rejects_payload_and_link_mutations(tmp_path):
    import json
    from desktop.scenario_run import ScenarioRunFormatError, load_scenario_run

    source = ROOT / "macos/ZephyrMeshApp/Tests/ZephyrMeshAppTests/Fixtures/zephyr-s7-scenario-run-1.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    document["payload_sha256"] = "0" * 64
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ScenarioRunFormatError, match="payload_sha256"):
        load_scenario_run(payload_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    document["parameters"]["target_link"]["delay_jitter_s"] = 0.1
    link_path = tmp_path / "link.json"
    link_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ScenarioRunFormatError):
        load_scenario_run(link_path)


def test_desktop_decoder_rejects_rehashed_frame_time_sequence(tmp_path):
    import hashlib
    import json
    from desktop.scenario_run import ScenarioRunFormatError

    source = ROOT / "macos/ZephyrMeshApp/Tests/ZephyrMeshAppTests/Fixtures/zephyr-s7-moving-scenario-run-1.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    document["frames"][1]["time_s"] = 0.07
    payload = {key: document[key] for key in ("frames", "link_events", "profiles")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
    document["payload_canonical"] = canonical
    document["payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = tmp_path / "bad-time.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ScenarioRunFormatError, match="time_s sequence"):
        load_scenario_run(path)


def test_desktop_decoder_rejects_rehashed_step_reorder_and_epsilon_boundary(tmp_path):
    import hashlib
    import json
    from desktop.scenario_run import ScenarioRunFormatError

    source = ROOT / "macos/ZephyrMeshApp/Tests/ZephyrMeshAppTests/Fixtures/zephyr-s7-moving-scenario-run-1.json"
    base = json.loads(source.read_text(encoding="utf-8"))

    def write_rehashed(document, name):
        payload = {key: document[key] for key in ("frames", "link_events", "profiles")}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
        document["payload_canonical"] = canonical
        document["payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        path = tmp_path / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    step_mutation = json.loads(json.dumps(base))
    step_mutation["frames"][1]["step_index"] = 0
    with pytest.raises(ScenarioRunFormatError, match="step_index sequence"):
        load_scenario_run(write_rehashed(step_mutation, "bad-step.json"))

    reordered = json.loads(json.dumps(base))
    reordered["frames"] = [reordered["frames"][1], reordered["frames"][0], reordered["frames"][2]]
    with pytest.raises(ScenarioRunFormatError, match="step_index sequence"):
        load_scenario_run(write_rehashed(reordered, "bad-order.json"))

    just_inside = json.loads(json.dumps(base))
    just_inside["frames"][1]["time_s"] += 0.5e-9
    load_scenario_run(write_rehashed(just_inside, "inside-epsilon.json"))

    just_outside = json.loads(json.dumps(base))
    just_outside["frames"][1]["time_s"] += 1.5e-9
    with pytest.raises(ScenarioRunFormatError, match="time_s sequence"):
        load_scenario_run(write_rehashed(just_outside, "outside-epsilon.json"))


def test_replay_cursor_clamps_and_reset_is_deterministic():
    model = ReplayModel.from_path(RUN)
    first = model.frame
    assert model.step(-10) == first
    assert model.step(100).step_index == 5
    assert model.progress == 1.0
    assert model.reset() == first
    assert model.index == 0
    assert model.summary()["frame"] == "1/6"


def test_scenario_run_cursor_clamps_replaces_and_resets():
    cursor = ScenarioRunCursor(3)
    assert cursor.step(-10) == 0
    assert cursor.step(1) == 1
    assert cursor.seek(100) == 2
    assert cursor.step() == 2
    assert cursor.replace(1) == 0
    assert cursor.frame_count == 1
    assert cursor.reset() == 0


def test_selected_frame_summary_changes_after_cursor_selection(tmp_path):
    from sim.scenario_runner import load_json_config, run_scenario, serialize_run

    config = load_json_config(ROOT / "sim" / "scenario_example.json")
    path = tmp_path / "scenario.json"
    path.write_bytes(serialize_run(run_scenario(config, code_revision="test")))
    run = load_scenario_run(path)
    cursor = ScenarioRunCursor(len(run.frames))
    first = selected_frame_summary(run, cursor)
    cursor.step()
    second = selected_frame_summary(run, cursor)
    assert first != second
    cursor.seek(2)
    assert "Frame 3 of 3" in selected_frame_summary(run, cursor)


def test_desktop_decoder_preserves_moving_keep_out_snapshots():
    fixture = ROOT / "macos/ZephyrMeshApp/Tests/ZephyrMeshAppTests/Fixtures/zephyr-s7-moving-scenario-run-1.json"
    run = load_scenario_run(fixture)
    cursor = ScenarioRunCursor(len(run.frames))
    assert run.frames[0].keep_out_spheres[0]["center_m"] == [2.0, 0.0, 1.5]
    cursor.step()
    assert run.frames[1].keep_out_spheres[0]["center_m"] == [2.025, 0.0, 1.5]
    assert "keep-out 1" in selected_frame_summary(run, cursor)


def test_scenario_geometry_is_bounded_and_follows_selected_frame():
    fixture = ROOT / "macos/ZephyrMeshApp/Tests/ZephyrMeshAppTests/Fixtures/zephyr-s7-moving-scenario-run-1.json"
    run = load_scenario_run(fixture)
    first = scenario_frame_geometry(run.frames[0])
    second = scenario_frame_geometry(run.frames[1])
    assert len(first.agents) == 1
    assert first.keep_out_spheres[0][1:3] == (2.0, 0.0)
    assert second.keep_out_spheres[0][1:3] == (2.025, 0.0)
    assert first.bounds.min_x < 1.5 < first.bounds.max_x
    assert first.bounds.min_y < 0.0 < first.bounds.max_y
    projected = project_scenario_point(first.target, first.bounds, 300, 180)
    assert 0.0 <= projected[0] <= 300.0
    assert 0.0 <= projected[1] <= 180.0
    assert first.target_altitude == 1.5
    assert first.agent_altitudes == (1.5,)
    assert first.keep_out_altitudes == (1.5,)
    altitude_y = project_scenario_altitude(first.target_altitude, first.altitude_bounds, 180)
    assert 0.0 <= altitude_y <= 180.0


def test_scenario_geometry_preserves_varied_altitudes_and_tiny_profile_bounds():
    frame = ScenarioFrame(
        step_index=0,
        time_s=0.0,
        target_position_m=(0.0, 0.0, 3.0),
        active_count=1,
        agent_count=1,
        agents=({"agent_id": "agent-1", "position_m": [1.0, 2.0, -1.0], "active": True},),
        keep_out_spheres=({"label": "ceiling", "center_m": [0.0, 0.0, 2.0], "radius_m": 0.75},),
    )
    geometry = scenario_frame_geometry(frame)
    assert geometry.target_altitude == 3.0
    assert geometry.agent_altitudes == (-1.0,)
    assert geometry.keep_out_altitudes == (2.0,)
    assert geometry.altitude_bounds.min_z < 1.25
    assert geometry.altitude_bounds.max_z > 2.75
    assert 0.0 <= project_scenario_altitude(-1.0, geometry.altitude_bounds, 2.0, inset=30.0) <= 2.0
    assert 0.0 <= project_scenario_altitude(3.0, geometry.altitude_bounds, 2.0, inset=30.0) <= 2.0
    assert project_scenario_altitude_radius(0.75, geometry.altitude_bounds, 2.0, inset=30.0) <= 1.0
    assert clamp_scenario_rail_x(1.0, 12.0) == 1.0
    assert scenario_rail_endpoints(1.0) == (0.5, 0.5)
    assert geometry.altitude_bounds.min_z == -1.25
    assert geometry.altitude_bounds.max_z == 3.25
    assert project_scenario_altitude(3.0, geometry.altitude_bounds, 100.0, inset=18.0) == pytest.approx(21.555555555555557)
    assert project_scenario_altitude(-1.0, geometry.altitude_bounds, 100.0, inset=18.0) == pytest.approx(78.44444444444444)
    assert project_scenario_altitude(2.0, geometry.altitude_bounds, 100.0, inset=18.0) == pytest.approx(35.77777777777778)
    assert project_scenario_altitude_radius(0.75, geometry.altitude_bounds, 100.0, inset=18.0) == pytest.approx(10.666666666666666)
    run = types.SimpleNamespace(frames=(frame,))
    assert "synthetic altitude 3.000 m" in selected_frame_summary(run, ScenarioRunCursor(1))


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
    assert "hidden-import desktop.scenario_run" in script
    assert "README-Windows.txt" in script
    assert "Extract the complete ZIP" in script


def test_windows_startup_diagnostic_is_written_outside_install_dir(tmp_path, monkeypatch):
    import desktop.windows_preview as preview

    app_data = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(app_data))
    monkeypatch.setattr(preview, "CURRENT_VERSION", "0.2.test")
    error = RuntimeError("missing bundled replay")
    path = preview.write_startup_diagnostic(error)

    assert path == app_data / "ZephyrMesh" / "startup.log"
    text = path.read_text(encoding="utf-8")
    assert "version: 0.2.test" in text
    assert "missing bundled replay" in text
    assert "default replay exists:" in text


def test_windows_startup_failure_message_explains_zip_extraction(capsys, monkeypatch, tmp_path):
    import desktop.windows_preview as preview

    monkeypatch.setattr(preview.sys, "platform", "linux")
    preview.show_startup_failure(RuntimeError("Tk failed"), tmp_path / "startup.log")

    assert "extract the entire ZIP" in capsys.readouterr().err


def test_windows_startup_failure_is_non_throwing_without_stderr(monkeypatch, tmp_path):
    import desktop.windows_preview as preview

    class BrokenMessageBox:
        def MessageBoxW(self, *_args):
            raise OSError("message box unavailable")

    fake_ctypes = types.SimpleNamespace(windll=types.SimpleNamespace(user32=BrokenMessageBox()))
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setattr(preview.sys, "platform", "win32")
    monkeypatch.setattr(preview.sys, "stderr", None)

    preview.show_startup_failure(RuntimeError("Tk failed"), tmp_path / "startup.log")


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
    assert 'README-Windows.txt' in validator
    assert 'startup.log' in validator


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
