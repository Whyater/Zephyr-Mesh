"""Pure-model checks for the cross-platform desktop replay preview."""
import json
from pathlib import Path
import sys
import types

import pytest

from desktop.replay import ReplayFormatError, ReplayModel, filter_agents
from desktop.scenario_run import ScenarioFrame, ScenarioRunCursor, clamp_scenario_rail_x, load_scenario_run, project_scenario_altitude, project_scenario_altitude_radius, project_scenario_point, scenario_frame_geometry, scenario_rail_endpoints, selected_frame_summary


ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs/s7-swarm/run.json"
# Read the counts straight from the file so regenerating the S7 run does not
# break these tests. The JSON is parsed separately from desktop.replay.
_RAW = json.loads(RUN.read_text(encoding="utf-8"))
RAW_FRAMES = len(_RAW["steps"])
RAW_AGENTS = len(_RAW["steps"][0]["agents"])
RAW_SCENARIO = _RAW["scenario_id"]


def test_replay_model_reads_canonical_s7_fixture():
    model = ReplayModel.from_path(RUN)
    assert model.schema == "zephyr-s7-swarm-run-1"
    assert model.scenario_id == RAW_SCENARIO
    assert model.frame_count == RAW_FRAMES
    assert len(model.frame.agents) == RAW_AGENTS
    assert model.frame.active_count == _RAW["steps"][0]["active_count"]
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
    viewers = (ROOT / "desktop" / "ui_viewers.py").read_text(encoding="utf-8")
    assert "--scenario-run" in source
    assert 'UI_TEXT["menus"]["openScenarioRun"], "openScenarioRun"' in source
    assert any(label == "Open Scenario Run..." for _menu, items in preview.MENU_LAYOUT for label, _a, _k in items)
    assert "load_scenario_run(args.scenario_run).summary()" in source
    assert "load_scenario_run(path)" in source
    assert "project_scenario_altitude" in viewers
    assert '"Details"' in viewers
    assert "json.dumps(summary" not in source, "raw JSON stays under Details in the viewers"
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
    assert model.step(100).step_index == _RAW["steps"][-1]["step_index"]
    assert model.progress == 1.0
    assert model.reset() == first
    assert model.index == 0
    assert model.summary()["frame"] == f"1/{RAW_FRAMES}"


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
    events = [event for event in _RAW["link_events"] if event["receiver"] == "drone-001"]
    assert link.event_count == len(events)
    assert link.loss_count == sum(1 for event in events if event["loss_reason"] is not None)
    assert link.mean_delay_ms is not None
    assert link.mean_delay_ms > 0
    assert model.summary()["profile_count"] == len(_RAW["agent_profiles"])


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


def test_windows_runtime_version_reads_the_file_beside_the_frozen_exe(tmp_path, monkeypatch):
    # PyInstaller 6 layout: _MEIPASS is <app>/_internal, VERSION.txt sits beside the exe.
    import desktop.windows_preview as preview

    app = tmp_path / "ZephyrMeshWindows"
    internal = app / "_internal"
    internal.mkdir(parents=True)
    (app / "ZephyrMeshWindows.exe").write_bytes(b"")
    (app / "VERSION.txt").write_text("0.2.3", encoding="utf-8")
    monkeypatch.setattr(preview, "RESOURCE_ROOT", internal)
    monkeypatch.setattr(preview, "RESOURCE_ROOTS", (internal / "_internal", internal))
    monkeypatch.setattr(preview.sys, "frozen", True, raising=False)
    monkeypatch.setattr(preview.sys, "executable", str(app / "ZephyrMeshWindows.exe"))

    assert preview._runtime_version() == "0.2.3"


def test_windows_source_checkout_version_is_the_dev_version():
    import desktop.windows_preview as preview

    expected = (ROOT / "VERSION").read_text(encoding="utf-8").strip() + "-dev"
    assert preview._runtime_version() == expected


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
    from tools.validate_windows_bundle import expected_smoke_marker
    from desktop.evidence import load_evidence_report
    kinds = [load_evidence_report(ROOT / "desktop" / name)["kind"] for name in ("evidence_report_example.json", "investigation_report_example.json", "s7_report_example.json")]
    assert expected_smoke_marker("ready", RAW_FRAMES, kinds) == f"ready frame=2/{RAW_FRAMES} evidence_kind=espnow,investigation,swarm"
    assert expected_smoke_marker("healthy", 1, ["swarm", "espnow", "swarm"]) == "healthy frame=1/1 evidence_kind=espnow,swarm"
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


def test_windows_preview_constructs_when_tk_display_is_available():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no desktop display in this test environment")
    root.withdraw()
    from desktop.windows_preview import WindowsReplayApp
    app = WindowsReplayApp(root, ReplayModel.from_path(RUN), appearance="dark", persist_appearance=False)
    assert app.selected_id == "drone-001"
    root.update()
    renders = app.render_count
    for _ in range(20):
        root.update()
    assert app.render_count == renders, "an idle window must not keep rendering"
    root.destroy()




# ---- Windows UI logic (no Tk display needed) --------------------------------
from desktop.replay_session import ReplaySession  # noqa: E402


class _QueuedTree:
    """Models ttk.Treeview selection: selection_set queues <<TreeviewSelect>> for later.

    Real Tk delivers the virtual event from the event loop after the caller
    returns, so a boolean "updating" guard is already reset when it fires.
    This fake always queues, even when the selection did not change, which is
    the worst case across Tk versions.
    """

    def __init__(self, queue):
        self.queue = queue
        self.selected = ()
        self.open = {}
        self.focused = ""
        self.set_calls = 0
        self.tags = {}

    def selection(self):
        return self.selected

    def selection_set(self, value):
        self.set_calls += 1
        self.selected = tuple(value) if isinstance(value, (tuple, list)) else (value,)
        self.queue.append("<<TreeviewSelect>>")

    def focus(self, value=None):
        if value is None:
            return self.focused
        self.focused = value

    def exists(self, iid):
        return iid in self.open

    def item(self, iid, option=None, **options):
        if options:
            self.tags.update({iid: tuple(options["tags"])} if "tags" in options else {})
            return None
        if option == "tags":
            return self.tags.get(iid, ())
        assert option == "open"
        return self.open.get(iid, False)

    def see(self, _iid):
        pass


def _selection_app(queue):
    import desktop.windows_preview as preview

    app = object.__new__(preview.WindowsReplayApp)
    app.session = ReplaySession(RUN)
    app.session.seek_frame(3)
    app.fleet = _QueuedTree(queue)
    presentation = app.session.presentation
    app._team_plan = preview.team_rows(presentation["groups"], presentation["per_drone"])
    for row in app._team_plan:
        app.fleet.open[row["iid"]] = row["level"] != "ok"
    app.render_count = 0
    app.inspector_renders = 0
    app._render_inspector = lambda: setattr(app, "inspector_renders", app.inspector_renders + 1)
    app.scene = types.SimpleNamespace(set_selected=lambda _drone: None)
    return app


def _drain(app, queue, limit=50):
    """Run queued virtual events like the Tk event loop; return how many ran."""
    ran = 0
    while queue and ran < limit:
        queue.pop(0)
        app._on_select(None)
        ran += 1
    return ran


def test_queued_treeview_select_events_settle_without_rendering():
    queue = []
    app = _selection_app(queue)
    late = next(row for row in app._team_plan if row["level"] != "ok")
    app.session.selected_drone_id = late["members"][0]  # in an open group, so a row really gets selected
    app._sync_team_selection()  # what _render_fleet does on startup
    assert queue == ["<<TreeviewSelect>>"]
    assert _drain(app, queue) == 1
    assert queue == [] and app.inspector_renders == 0
    app._sync_team_selection()  # a later render with nothing changed
    assert app.fleet.set_calls == 1, "selection_set must not be called when the selection already matches"


def test_user_selection_renders_once_and_never_loops():
    queue = []
    app = _selection_app(queue)
    app._sync_team_selection()
    _drain(app, queue)
    late = next(row for row in app._team_plan if row["level"] == "caution")
    other = next(drone for drone in late["members"] if drone != app.session.selected_drone_id)
    app.fleet.selected = (other,)  # the user clicks a row: Tk queues one event
    queue.append("<<TreeviewSelect>>")
    assert _drain(app, queue) == 1
    assert queue == []
    assert app.session.selected_drone_id == other
    assert app.inspector_renders == 1


def test_landing_on_a_group_header_keeps_the_selected_drone_and_settles():
    queue = []
    app = _selection_app(queue)
    late = next(row for row in app._team_plan if row["level"] != "ok")
    app.session.selected_drone_id = before = late["members"][0]
    ok_group = next(row for row in app._team_plan if row["level"] == "ok")
    assert before not in ok_group["members"]
    app.fleet.selected = (ok_group["iid"],)  # arrow keys land on a header: Tk queues one event
    queue.append("<<TreeviewSelect>>")
    _drain(app, queue)
    assert app.session.selected_drone_id == before, "a header never picks a drone by itself"
    assert app.inspector_renders == 0 and queue == []
    app.session.selected_drone_id = ok_group["members"][0]
    app._sync_team_selection()  # drone in a closed group: no selection, the group gets the quiet "holds" tag
    assert app.fleet.selection() == ()
    assert "holds" in app.fleet.tags[ok_group["iid"]] and app.fleet.focused == ok_group["iid"]
    _drain(app, queue)
    app.fleet.open[ok_group["iid"]] = True  # expanded: the selection moves to the drone row, the tag goes away
    app._sync_team_selection()
    assert app.fleet.selection() == (ok_group["members"][0],)
    assert "holds" not in app.fleet.tags[ok_group["iid"]] and "group" in app.fleet.tags[ok_group["iid"]]
    renders = app.inspector_renders
    assert _drain(app, queue) == 1 and queue == []
    assert app.inspector_renders == renders


def test_queued_event_model_catches_the_old_feedback_loop():
    """The fake is strict enough: an unconditional render-on-select handler never settles."""
    queue = []
    app = _selection_app(queue)

    def naive_on_select(_event):
        selection = app.fleet.selection()
        if selection:
            app.fleet.selection_set(selection[0])  # render -> _render_fleet -> selection_set

    app._on_select = naive_on_select
    app.fleet.selection_set(app.session.selected_drone_id)
    assert _drain(app, queue, limit=50) == 50
    assert queue, "the old handler keeps queueing events forever"


def test_team_rows_group_drones_and_filter_never_hides_non_ok():
    from desktop.windows_preview import desired_team_selection, drone_for_team_row, team_rows

    session = ReplaySession(RUN)
    session.seek_frame(5)
    presentation = session.presentation
    rows = team_rows(presentation["groups"], presentation["per_drone"])
    assert sum(len(row["members"]) for row in rows) == RAW_AGENTS
    assert rows[0]["level"] != "ok", "non-OK groups sort first"
    assert rows[-1]["label"] == f"OK · {len(rows[-1]['members'])} drones", "status word first, like the Mac"
    filtered = team_rows(presentation["groups"], presentation["per_drone"], "D01")
    assert {row["level"] for row in rows if row["level"] != "ok"} <= {row["level"] for row in filtered}
    ok_row = next(row for row in filtered if row["level"] == "ok")
    assert [child["label"] for child in ok_row["children"]] == ["D01 · OK"], "drone rows repeat their word"
    assert all(child["level"] == "ok" for child in ok_row["children"]), "and their status shape"
    ok_full = next(row for row in rows if row["level"] == "ok")
    assert ok_row["label"] == f"OK · {len(ok_row['children'])} of {len(ok_full['members'])}"
    first = rows[0]
    assert desired_team_selection(rows, first["members"][0], set()) is None, "a closed group is never selected"
    assert desired_team_selection(rows, first["members"][0], {first["iid"]}) == first["members"][0]
    assert drone_for_team_row(rows, first["iid"], "nobody") is None, "a header never picks a drone"
    assert drone_for_team_row(rows, first["iid"], first["members"][-1]) == first["members"][-1]
    assert drone_for_team_row(rows, first["members"][-1], "nobody") == first["members"][-1]
    waiting = team_rows(ReplaySession(RUN).presentation["groups"], ReplaySession(RUN).presentation["per_drone"])
    assert waiting[0]["values"] == ("–", "–"), "missing values are never shown as 0"
    assert all(child["values"] == ("–", "–") for child in waiting[0]["children"])
    late = rows[0]["children"][0]["values"]
    assert late[0].count(".") == 1 and not late[0].endswith("%"), "loss: one decimal, the % lives in the heading"
    assert not late[1].endswith("s"), "age: the unit lives in the heading"


class _FakeRoot:
    """Records after()/after_cancel() so the test can count live timer chains."""

    def __init__(self):
        self.jobs = {}
        self.counter = 0

    def after(self, _delay_ms, callback, *args):
        self.counter += 1
        self.jobs[f"job{self.counter}"] = (callback, args)
        return f"job{self.counter}"

    def after_cancel(self, job):
        self.jobs.pop(job, None)


class _FakeWidget:
    def __init__(self):
        self.options = {}

    def configure(self, **options):
        self.options.update(options)


def _playback_app():
    from desktop.windows_preview import WindowsReplayApp

    app = object.__new__(WindowsReplayApp)  # skip __init__: it needs a Tk display
    app.root = _FakeRoot()
    app.model = ReplayModel.from_path(RUN)
    app.session = ReplaySession(app.model, schedule=app._schedule, cancel=app._cancel)
    app.renders = 0
    app._render = lambda: setattr(app, "renders", app.renders + 1)
    app.play_button = _FakeWidget()
    app.play_tip = types.SimpleNamespace(text="", set_text=lambda text: setattr(app.play_tip, "text", text))
    app.time_label = _FakeWidget()
    app.slider = types.SimpleNamespace(set_state=lambda *args: None)
    app.images = types.SimpleNamespace(icon=lambda name, *args: name)
    app.speed_var = types.SimpleNamespace(get=lambda: app.session.speed, set=lambda value: None)
    return app


def _run_next_job(app):
    (job, (callback, args)), = app.root.jobs.items()
    del app.root.jobs[job]
    callback(*args)


def test_play_button_icon_and_tooltip_follow_playback():
    app = _playback_app()
    app._render_timeline()
    assert app.play_button.options["image"] == "play" and app.play_tip.text == "Play (Space)"
    app._toggle()
    app._render_timeline()
    assert app.playing and app.play_button.options["image"] == "pause"
    assert app.play_tip.text == "Pause (Space)"
    app._step()  # stepping pauses
    app._render_timeline()
    assert not app.playing and app.play_button.options["image"] == "play" and app.play_tip.text.startswith("Play")
    assert app.time_label.options["text"] == "0.05 / 0.25 s"


def test_play_pause_play_keeps_exactly_one_timer_chain():
    app = _playback_app()
    app._toggle()
    app._toggle()
    app._toggle()
    assert len(app.root.jobs) == 1, "a second after() chain means double-speed playback"
    for expected in (1, 2, 3):
        _run_next_job(app)
        assert app.session.frame_index == expected
        assert len(app.root.jobs) == 1


def test_playback_stops_at_the_last_frame_and_play_restarts():
    app = _playback_app()
    app._toggle()
    while app.root.jobs:
        _run_next_job(app)
    assert not app.playing
    assert app.session.frame_index == app.session.frame_count - 1
    app._toggle()  # Play at the end starts again from the first frame
    assert app.playing and app.session.frame_index == 0


@pytest.mark.parametrize("widget_class,typing", [("Entry", True), ("TEntry", True), ("Text", True), ("Spinbox", True), ("TSpinbox", True), ("TCombobox", True), ("Canvas", False), ("TButton", False), ("Treeview", False), ("Tk", False)])
def test_text_input_widgets_are_recognised(widget_class, typing):
    from desktop.windows_preview import is_text_input_class

    assert is_text_input_class(widget_class) is typing


def test_single_key_shortcuts_yield_to_the_focused_widget():
    from desktop.windows_preview import WindowsReplayApp

    app = object.__new__(WindowsReplayApp)
    fired = []

    def press(widget_class, key):
        handler = app._shortcut(lambda: fired.append(key), key)
        return handler(types.SimpleNamespace(widget=types.SimpleNamespace(winfo_class=lambda: widget_class)))

    assert press("TEntry", "f") is None and fired == []
    assert press("TButton", "space") is None and fired == [], "a focused button owns Space"
    assert press("Treeview", "right") is None and fired == [], "the team list owns the arrow keys"
    assert press("Treeview", "space") is None and fired == [], "and Space, which opens or closes a group"
    assert press("Canvas", "f") == "break" and fired == ["f"]
    assert press("TButton", "f") == "break"
    assert press("Treeview", "home") == "break", "Home has no list binding, so it stays Restart"
    assert fired == ["f", "f", "home"]


def test_menu_layout_covers_every_shortcut_with_generated_accelerators():
    from desktop.design_tokens import SHORTCUTS
    from desktop.windows_preview import MENU_LAYOUT, SHORTCUT_HANDLERS, WindowsReplayApp, menu_accelerators

    assert set(SHORTCUT_HANDLERS) == {shortcut.action_id for shortcut in SHORTCUTS}
    assert all(callable(getattr(WindowsReplayApp, name)) for name in SHORTCUT_HANDLERS.values())
    accelerators = menu_accelerators()
    assert accelerators["openScenarioRun"] == "Ctrl+Shift+O"
    assert accelerators["playPause"] == "Space"
    assert accelerators["stepBack"] == "Left"
    titles = [title for title, _items in MENU_LAYOUT]
    assert titles == ["File", "View", "Playback", "Help"]
    labels = {label for _title, items in MENU_LAYOUT for label, _action, _kind in items}
    for label in ("Open Evidence Report...", "Open Scenario Run...", "Exit", "Show Keep-out Volume", "Show Inspector",
                  "Focus Selected Drone", "Reset Camera", "Appearance", "Play/Pause", "Step Forward", "Step Back",
                  "Restart", "Speed", "About This Replay", "Check for Updates"):
        assert label in labels


def test_events_through_is_cumulative_oldest_first():
    session = ReplaySession(RUN)
    assert [event["text"] for _frame, event in session.events_through()] == ["Replay loaded: 50 drones"]
    session.seek_frame(3)
    events = session.events_through()
    assert [frame for frame, _event in events] == sorted(frame for frame, _event in events)
    assert events[-1][0] == 3
    assert session.event_frames()[0] == 0


# ---- formatting ------------------------------------------------------------
def test_formatting_precision_and_missing_values():
    from desktop import ui_format as fmt

    assert fmt.format_value(None, "ms") is None
    assert fmt.display(None, "ms") == "Not recorded"
    assert fmt.format_value(80.0, "ms") == "80 ms"
    assert fmt.format_value(25.81, "pct") == "25.8%"
    assert fmt.format_value(50.0, "pct") == "50%"
    assert fmt.format_value({"min": 50.0, "max": 66.67}, "pct") == "50–67%"
    assert fmt.format_value({"min": 80.0, "max": 80.0}, "ms") == "80 ms"
    assert fmt.format_value(0.15000000000000002, "s") == "0.15 s"
    assert fmt.format_value(0.0, "m") == "0.00 m", "a recorded zero stays zero"
    assert fmt.format_age_span(0.1, 0.2) == "0.10–0.20 s"
    assert fmt.format_age_span(0.1, 0.1) == "0.10 s"
    assert fmt.format_age_span(None, None) is None
    assert fmt.time_label(0.1, 0.25) == "0.10 / 0.25 s"


def test_team_summary_and_run_name_come_from_the_presentation():
    from desktop import ui_format as fmt
    from desktop.presentation import run_name, team_summary

    records = [{"level": "ok", "word": "OK"}] * 45 + [{"level": "caution", "word": "Late"}] * 4 + [{"level": "waiting", "word": "Waiting"}]
    assert team_summary(records) == "50 drones: 45 OK, 4 late, 1 waiting"
    assert run_name(_RAW) == "50-drone ring, seed 17"
    assert RAW_SCENARIO not in run_name(_RAW)
    assert not hasattr(fmt, "team_summary") and not hasattr(fmt, "plain_run_name"), "one source only"
    session = ReplaySession(RUN)
    session.seek_frame(3)
    assert session.presentation["team_summary"] == "50 drones: 47 OK, 3 late"


def test_windows_inspector_shows_not_recorded_for_missing_neighbor_values():
    """A frame without neighbor_count or min_neighbor_distance_m reads "Not recorded", never 0."""
    from desktop import ui_format as fmt

    document = json.loads(json.dumps(_RAW))
    for agent in document["steps"][2]["agents"]:
        agent.pop("neighbor_count", None)
        agent.pop("min_neighbor_distance_m", None)
    session = ReplaySession(document)
    session.seek_frame(2)
    record = session.record_for(session.presentation["per_drone"][0]["drone_id"])
    assert record["neighbor_count"] is None
    assert record["nearest_neighbor_m"] is None
    # The exact calls _render_inspector makes for the Neighbors rows.
    assert fmt.format_scalar(record.get("neighbor_count")) == "Not recorded"
    assert fmt.display(record.get("nearest_neighbor_m"), "m") == "Not recorded"
    assert fmt.format_scalar(0) == "0", "a recorded zero stays zero"
    source = (ROOT / "desktop" / "windows_preview.py").read_text(encoding="utf-8")
    assert 'fmt.format_scalar(record.get("neighbor_count"))' in source
    assert 'fmt.format_value(record.get("nearest_neighbor_m"), "m")' in source


def test_about_not_in_replay_uses_the_presentation_list():
    from desktop.presentation import not_in_replay
    from desktop.ui_viewers import unavailable_items

    assert unavailable_items(_RAW) == (
        "Live radio link, camera or vision input, battery state, flight controller commands, measured flight performance"
    )
    with_battery = json.loads(json.dumps(_RAW))
    with_battery["steps"][0]["agents"][0]["battery_pct"] = 90.0
    assert "Battery state" not in not_in_replay(with_battery)


def test_labels_never_show_snake_case():
    from desktop import ui_format as fmt

    assert fmt.field_label("target_age_s") == "Target age (s)"
    assert fmt.humanize_key("p95_delay_ms") == "P95 delay (ms)"
    assert fmt.humanize_key("observed_rate_hz") == "Observed rate (Hz)"
    for key in ("unseen_key_name", "max_power_w", "abc_def_ghi_mps"):
        assert "_" not in fmt.field_label(key)
    sections = fmt.summary_sections({"provenance": {"status": "synthetic"}, "frame_count": 6, "nested": {"deep": {"a": 1}}})
    assert sections[0] == (None, [("Frame count", "6")])
    assert all("_" not in label for _title, rows in sections for label, _value in rows)
    assert fmt.format_scalar({"min": 1.0, "max": 2.0}) == "Max 2, Min 1"


def test_accelerator_text_matches_generated_shortcuts():
    from desktop import ui_format as fmt

    assert fmt.accelerator("openEvidenceReport") == "Ctrl+O"
    assert fmt.accelerator("focusFilter") == "Ctrl+L"
    assert fmt.accelerator("toggleKeepOut") == "K"
    assert fmt.accelerator("restart") == "Home"


# ---- icons -------------------------------------------------------------------
def test_status_glyph_png_is_valid_rgba_with_antialiased_edges():
    import struct
    import zlib
    from desktop.ui_icons import STATUS_SHAPES, coverage, status_png
    from desktop.design_tokens import UI_TEXT

    assert {spec["shape"] for spec in UI_TEXT["statusLevels"].values()} <= set(STATUS_SHAPES)
    png = status_png("circle", "#5FD39B", 12, 16)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, color_type = struct.unpack(">IIBB", png[16:26])
    assert (width, height, depth, color_type) == (16, 12, 8, 6)
    raw = zlib.decompress(png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8])
    assert len(raw) == height * (1 + width * 4)
    grid = coverage(STATUS_SHAPES["circle"], 12)
    assert grid[6][6] == 1.0 and grid[0][0] == 0.0
    assert any(0.0 < value < 1.0 for row in grid for value in row), "edges are antialiased"
    hollow = coverage(STATUS_SHAPES["hollow circle"], 12)
    assert hollow[6][6] == 0.0, "the waiting glyph is hollow, so it differs by shape and not only color"
    info = coverage(STATUS_SHAPES["info"], 24)
    assert info[12][1] > 0.5 and info[12][22] > 0.5, "info has a circle outline"
    assert info[17][12] > 0.5 and info[7][12] > 0.5, "with an i stem and dot inside"
    assert info[12][6] == 0.0 and info[11][17] == 0.0, "and open space between"


def test_every_golden_event_renders_a_word_and_known_glyph():
    """Runtime check over the canonical presentation: no bare shapes in the event list."""
    from desktop import ui_format as fmt
    from desktop.design_tokens import UI_TEXT
    from desktop.ui_icons import STATUS_SHAPES, status_png
    from desktop.ui_theme import Theme, status_shape

    golden = json.loads((ROOT / "design" / "golden" / "s7_presentation.json").read_text(encoding="utf-8"))
    events = [event for frame in golden["frames"] for event in frame["events"]]
    assert {event["level"] for event in events} >= {"info", "ok", "caution"}
    for theme in (Theme("dark"), Theme("light")):
        for event in events:
            level, word, time_text, text = fmt.event_row(event)
            spec = UI_TEXT["statusLevels"][level]
            assert level == event["level"] and word == spec["word"] and word.strip()
            assert status_shape(level) == spec["shape"] and spec["shape"] in STATUS_SHAPES
            assert theme.event_color(level) == getattr(theme.p, spec["color"])
            assert status_png(status_shape(level), theme.event_color(level), 11, 18)[:4] == b"\x89PNG"
            assert time_text.endswith(" s") and text
    assert fmt.event_row({"level": "mystery", "time_s": 0.1, "text": "x"})[:2] == ("unavailable", "Not recorded")


# ---- scene projection (renderer math) ----------------------------------------
def test_projection_is_not_mirrored():
    from desktop.scene_projection import OrbitCamera, Projector, project_point, top_pose

    looking_north = OrbitCamera(yaw=-90, pitch=35, distance=10, target=(0, 0, 0), swarm_radius=4)
    assert project_point(looking_north, (1, 0, 0), 800, 600).x > 400, "east is to the right when facing north"
    assert project_point(looking_north, (0, 1, 0), 800, 600).y < 300, "farther north is higher on screen"
    top = Projector(top_pose(looking_north), 800, 600)
    east, north = top.project((1, 0, 0)), top.project((0, 1, 0))
    assert east.x > 400 and abs(east.y - 300) < 1e-6
    assert north.y < 300 and abs(north.x - 400) < 1e-6


def test_segment_clipping_and_framing():
    from desktop.scene_projection import NEAR_PLANE, OrbitCamera, Projector, ViewPose, fit_distance, framing_camera

    pose = ViewPose(0.0, 10.0, 5.0, (0.0, 0.0, 0.0))
    projector = Projector(pose, 800, 600)
    behind = projector.segment((20.0, 0.0, 0.0), (30.0, 0.0, 0.0))
    assert behind is None
    crossing = projector.segment((20.0, 0.0, 0.0), (-5.0, 0.0, 0.0))
    assert crossing is not None and min(point.depth for point in crossing) >= NEAR_PLANE - 1e-9
    ring = [agent["position_m"] for agent in _RAW["steps"][0]["agents"]]
    camera = framing_camera(ring)
    distance = fit_distance(camera, ring, 800, 500, 30)
    assert distance <= camera.swarm_radius * 3 + 1e-9
    fitted = Projector(ViewPose(camera.yaw, camera.pitch, distance, camera.target), 800, 500)
    assert all(30 - 1e-6 <= fitted.project(point).x <= 770 + 1e-6 for point in ring)
    assert fit_distance(camera, ring, 800, 500, 30) < fit_distance(camera, ring, 400, 250, 30)


def test_xframe_glyph_shape_and_honest_heading():
    import math
    from desktop.scene_projection import Projector, ViewPose, xframe_glyph

    projector = Projector(ViewPose(-90.0, 90.0, 10.0, (0.0, 0.0, 0.0)), 800, 600)
    resting = xframe_glyph(projector, (0.0, 0.0, 0.0), None)
    assert resting is not None and resting.heading is None, "no heading marker without a heading in the data"
    assert len(resting.arm_tips) == 4 and len(resting.rotors) == 4
    dx, dy = resting.arm_tips[0][0] - resting.center.x, resting.arm_tips[0][1] - resting.center.y
    assert abs(abs(dx) - abs(dy)) < 1e-6, "X frame: arms sit at 45 degrees"
    assert all(abs(rx - ry) < 1e-6 for _x, _y, rx, ry in resting.rotors), "discs are circles from straight above"
    moving = xframe_glyph(projector, (0.0, 0.0, 0.0), 0.0)
    assert moving.heading is not None and moving.heading[0][0] > moving.center.x
    tilted = xframe_glyph(Projector(ViewPose(-90.0, 30.0, 10.0, (0.0, 0.0, 0.0)), 800, 600), (0.0, 0.0, 0.0), math.pi / 2)
    assert all(ry < rx for _x, _y, rx, ry in tilted.rotors), "discs flatten into ellipses at a low pitch"
    assert xframe_glyph(Projector(ViewPose(0.0, 10.0, 2.0, (0.0, 0.0, 0.0)), 800, 600), (50.0, 0.0, 0.0), None) is None


def test_camera_animator_eases_and_respects_reduced_motion():
    from desktop.scene_projection import CameraAnimator, ViewPose

    start = ViewPose(170.0, 30.0, 10.0, (0.0, 0.0, 0.0))
    goal = ViewPose(-170.0, 60.0, 5.0, (1.0, 0.0, 0.0))
    animator = CameraAnimator(start)
    animator.go(goal, 0, 180)
    half = animator.advance(90)
    assert 170.0 < half.yaw < 200.0, "yaw takes the short way across 180 degrees"
    assert 30.0 < half.pitch < 60.0 and animator.active
    assert animator.advance(180) == goal and not animator.active
    reduced = CameraAnimator(start)
    reduced.go(goal, 0, 180, reduced_motion=True)
    assert reduced.pose == goal and not reduced.active


def test_grid_has_one_meter_lines_and_stronger_five_meter_lines():
    from desktop.scene_projection import grid_lines

    lines = grid_lines((0.0, 0.0), 6.0)
    xs = sorted({start[0] for start, end, _major in lines if start[0] == end[0]})
    assert xs == [float(value) for value in range(-6, 7)]
    majors = sorted({start[0] for start, end, major in lines if major and start[0] == end[0]})
    assert majors == [-5.0, 0.0, 5.0]


def test_appearance_resolution_defaults_to_dark():
    from desktop.ui_settings import resolve_theme, system_reduced_motion

    assert resolve_theme("system", None) == "dark"
    assert resolve_theme("system", False) == "light"
    assert resolve_theme("system", True) == "dark"
    assert resolve_theme("light") == "light"
    assert resolve_theme("nonsense") == "dark"
    assert system_reduced_motion() in (True, False)


def test_windows_build_lists_every_ui_module():
    from tools.validate_windows_bundle import WINDOWS_UI_MODULES

    script = (ROOT / "desktop" / "build_windows.ps1").read_text(encoding="utf-8")
    for module in ("presentation", "replay_session", "scene_projection", "scene_view", "ui_format", "ui_icons",
                   "ui_settings", "ui_theme", "ui_viewers", "ui_widgets"):
        assert f"--hidden-import desktop.{module}" in script
        assert f"desktop.{module}" in WINDOWS_UI_MODULES


def test_scene_hides_stale_target_and_bounds_top_view():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no desktop display in this test environment")
    root.withdraw()
    from desktop.windows_preview import WindowsReplayApp
    app = WindowsReplayApp(root, ReplayModel.from_path(RUN), appearance="dark", persist_appearance=False)
    root.update()
    scene = app.scene
    frame = dict(app._raw_frame())
    scene._draw_now()
    assert all(scene.canvas.itemcget(item, "state") != "hidden" for item in scene.canvas.find_withtag("target"))
    frame.pop("target_position_m")
    scene.set_frame(frame, app.session.per_drone, app.session.selected_drone_id)
    scene._draw_now()
    assert all(scene.canvas.itemcget(item, "state") == "hidden" for item in scene.canvas.find_withtag("target"))
    scene.set_view_mode("top")
    assert scene.camera.distance <= scene.camera.swarm_radius * 6 + 1e-9, "orbit stays within 6x the swarm radius"
    assert scene._goal_pose().distance <= scene.camera.swarm_radius * 6 + 1e-9, "top view is bounded too"
    root.destroy()


def test_report_view_puts_ids_in_details_and_units_on_values():
    from desktop import ui_format as fmt
    from desktop.evidence import load_evidence_report

    report = load_evidence_report(ROOT / "desktop" / "evidence_report_example.json")
    view = fmt.report_view(report["summary"])
    rows = dict(row for _title, section in view["sections"] for row in section)
    assert rows["Delay"] == "Median 12 ms, p95 12 ms, max 12 ms (1 sample)", "seconds show as ms, no Unit row"
    assert not any(label in ("Schema", "Unit") or "ID" in label for label in rows)
    assert "Schema" in dict(view["details"]) and any("Trace ID" in label for label, _value in view["details"])
    assert [label for label, _stat in view["timings"]] == ["Delay"]
    assert all("_" not in label for label in rows)
    swarm = fmt.report_view(load_evidence_report(ROOT / "desktop" / "s7_report_example.json")["summary"])
    swarm_rows = dict(row for _title, section in swarm["sections"] for row in section)
    assert swarm_rows["Target age"].startswith("Median 100 ms")
    assert fmt.stat_line({"available": False, "count": 0, "unit": "s", "p50": None}) == "Not recorded"
    assert fmt.humanize_filename("evidence_report_example.json") == "Evidence report example"


def test_missing_link_and_part_values_never_read_zero():
    from desktop import ui_format as fmt

    assert fmt.link_text({}) == "Not recorded"
    assert fmt.link_text({"loss_model": "none"}) == "delay not recorded, no loss"
    assert fmt.link_text({"delay_s": 0.0, "delay_jitter_s": 0.0, "loss_model": "independent", "loss_probability": 0.1}) \
        == "0 ms delay, 10% independent loss", "a recorded zero is still shown"
    assert fmt.quantity(None, "mm", 1000.0, 1) == "Not recorded"
    assert fmt.part_text("prop-1", fmt.quantity(None, "mm")) == "prop-1"
    assert fmt.part_text(None, fmt.quantity(0.08, "mm", 1000.0, 1)) == "Not recorded, 80.0 mm"
