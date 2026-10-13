"""Checks for the Windows UI polish pass: each test pins one fixed defect.

Most tests drive the real logic without a display. The few that need Tk build
the real window and measure it (control heights, timing plot geometry),
which is an independent check rather than a restatement of the code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import types

import pytest

from desktop import ui_format as fmt
from desktop.design_tokens import UI_TEXT, WIN_LAYOUT
from desktop.replay import ReplayModel
from desktop.replay_session import ReplaySession
from desktop.ui_settings import load_appearance, settings_path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/s7-swarm/run.json"
WIN_TEXT = UI_TEXT["windowsText"]


# ---- events list: clicks always seek; keys go to the replay -------------------------------------
class _EventsTree:
    """Like ttk.Treeview for the Events list: clicking a row that is already selected fires no <<TreeviewSelect>>."""

    def __init__(self, rows_by_y):
        self.rows_by_y = rows_by_y
        self.selected = ()

    def identify_row(self, y):
        return self.rows_by_y.get(y, "")

    def selection(self):
        return self.selected


def _events_app():
    from desktop.windows_preview import WindowsReplayApp

    app = object.__new__(WindowsReplayApp)
    app.session = ReplaySession(RUN)
    app.seeks = []

    def seek(index):
        app.seeks.append(index)
        if index != app.session.frame_index:
            app.session.seek_frame(index)
    app._seek_frame = seek
    app._event_frames = {"event0": 0, "event3": 2}
    app.events_tree = _EventsTree({10: "event3"})
    return app


def test_clicking_the_same_event_row_twice_seeks_twice():
    app = _events_app()
    click = types.SimpleNamespace(y=10)
    app._on_event_click(click)
    assert app.session.frame_index == 2
    app.events_tree.selected = ("event3",)  # Tk keeps the row selected
    app.session.seek_frame(4)  # the user moves away with the slider
    app._on_event_click(click)  # same row again: no new <<TreeviewSelect>>, but the click still seeks
    assert app.seeks == [2, 2] and app.session.frame_index == 2


def test_return_and_keyboard_selection_in_the_events_list_seek():
    app = _events_app()
    app.events_tree.selected = ("event3",)
    app._on_event_select(force=True)
    assert app.session.frame_index == 2
    app._on_event_click(types.SimpleNamespace(y=999))  # a click on empty space does nothing
    assert app.seeks == [2]


def test_a_repeated_select_event_for_the_same_row_does_not_seek_back():
    """Right on the Events list re-sets the same row; that must not undo the step the key just made."""
    app = _events_app()
    app.events_tree.selected = ("event3",)
    app._on_event_select()  # the user moved the selection onto the row: seek
    assert app.session.frame_index == 2
    app.session.seek_frame(3)  # Right stepped the replay forward
    app._on_event_select()  # Tk fires <<TreeviewSelect>> again for the unchanged row
    assert app.session.frame_index == 3 and app.seeks == [2]


def test_left_right_home_and_space_from_the_events_list_reach_the_replay():
    from desktop.windows_preview import WindowsReplayApp

    app = object.__new__(WindowsReplayApp)
    events = types.SimpleNamespace(winfo_class=lambda: "Treeview", __str__=lambda self: ".events")
    events = type("Tree", (), {"winfo_class": lambda self: "Treeview", "__str__": lambda self: ".!frame.events"})()
    team = type("Tree", (), {"winfo_class": lambda self: "Treeview", "__str__": lambda self: ".!frame.team"})()
    app.events_tree = events
    fired = []
    for key in ("left", "right", "home", "space"):
        handler = app._shortcut(lambda key=key: fired.append(key), key)
        assert handler(types.SimpleNamespace(widget=events)) == "break"
    assert fired == ["left", "right", "home", "space"]
    for key in ("left", "right", "space"):
        assert app._shortcut(lambda: fired.append("team"), key)(types.SimpleNamespace(widget=team)) is None
    assert "team" not in fired, "the team list keeps its own open/close keys"


# ---- inspector header cache -----------------------------------------------------------------------
class _Label:
    def __init__(self, name):
        self.name = name
        self.options = {}
        self.configure_calls = 0

    def configure(self, **options):
        self.configure_calls += 1
        self.options.update(options)

    def __str__(self):
        return self.name


def test_inspector_header_refreshes_when_a_drone_comes_back():
    from desktop.windows_preview import WindowsReplayApp

    app = object.__new__(WindowsReplayApp)
    app.session = ReplaySession(RUN)
    app.session.seek_frame(3)
    app.inspector_title, app.inspector_word = _Label(".title"), _Label(".word")
    app.images = types.SimpleNamespace(status=lambda *args: "glyph")
    hidden = []
    app._show_inspector_sections = lambda visible: hidden.append(visible)
    drone = app.session.selected_drone_id
    record = app.session.record_for(drone)
    real_record_for = app.session.record_for
    app._set(app.inspector_title, text=record["heading_name"])  # what a normal render left on screen
    app.session.record_for = lambda _drone: None  # the drone is missing from this frame
    app.__class__._render_inspector(app)
    assert app.inspector_title.options["text"] == UI_TEXT["inspector"]["emptyTitle"]
    app.session.record_for = real_record_for
    calls = app.inspector_title.configure_calls
    app._set(app.inspector_title, text=record["heading_name"])  # the same drone is back
    assert app.inspector_title.configure_calls == calls + 1, "the cache must know the empty title replaced the name"
    assert app.inspector_title.options["text"] == record["heading_name"]
    assert hidden == [False]


# ---- settings file robustness -----------------------------------------------------------------------
@pytest.mark.parametrize("content", [
    b"[" * 200_000 + b"]" * 200_000,                      # deep nesting: RecursionError in json
    b'{"appearance": "light"' + b" " * 70_000 + b"}",     # larger than any real settings file
    b'{"appearance": "\xff\xfe"}',                        # not UTF-8
    b"",                                                   # empty
    b'["light"]',                                         # wrong JSON type
])
def test_corrupt_settings_files_fall_back_to_dark(tmp_path, content):
    path = settings_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    assert load_appearance(tmp_path) == "dark"


def test_a_settings_folder_in_place_of_the_file_falls_back(tmp_path):
    settings_path(tmp_path).mkdir(parents=True)
    assert load_appearance(tmp_path) == "dark"


# ---- projection and scene data guards -----------------------------------------------------------
def test_projection_guards_zero_and_non_finite_distance():
    from desktop.scene_projection import Projector, ViewPose, finite_point

    zero = Projector(ViewPose(-60.0, 30.0, 0.0, (0.0, 0.0, 0.0)), 400, 300)
    point = zero.project((1.0, 1.0, 1.0))
    assert all(math.isfinite(value) for value in point), "zero distance must not divide by zero"
    with pytest.raises(ValueError):
        Projector(ViewPose(-60.0, 30.0, float("nan"), (0.0, 0.0, 0.0)), 400, 300)
    with pytest.raises(ValueError):
        Projector(ViewPose(-60.0, 30.0, 5.0, (0.0, float("inf"), 0.0)), 400, 300)
    assert finite_point([1, 2, 3]) == (1.0, 2.0, 3.0)
    for bad in (None, [1, 2], [1, 2, float("nan")], ["a", 1, 2], "abc", {"x": 1}):
        assert finite_point(bad) is None


def test_wide_short_canvas_fits_the_whole_ring():
    """QA defect 7: at 1158x434 every drone must project inside the canvas (independent re-projection)."""
    from desktop.scene_projection import Projector, ViewPose, fit_distance, framing_camera

    run = json.loads(RUN.read_text())
    points = [tuple(agent["position_m"]) for step in run["steps"] for agent in step["agents"]]
    camera = framing_camera(points)
    width, height, margin = 1158, 434, 34
    distance = fit_distance(camera, points, width, height, margin)
    projector = Projector(ViewPose(camera.yaw, camera.pitch, distance, camera.target), width, height)
    for point in points:
        projected = projector.project(point)
        assert margin - 1e-6 <= projected.x <= width - margin + 1e-6
        assert margin - 1e-6 <= projected.y <= height - margin + 1e-6


def test_malformed_vectors_never_reach_the_scene():
    from desktop.scene_view import _agent_position, _heading

    assert _agent_position({"position_m": [0, float("nan"), 1]}) is None
    assert _agent_position({"position_m": [1, 2]}) is None
    assert _agent_position({"position_m": [1, 2, 3]}) == (1.0, 2.0, 3.0)
    assert _heading([float("inf"), 0, 0]) is None
    assert _heading(["x", 0, 0]) is None
    assert _heading([0, 1, 0]) == pytest.approx(math.pi / 2)


def test_replay_session_rejects_non_finite_frame_times():
    run = json.loads(RUN.read_text())
    for bad in (float("nan"), float("inf"), "0.1", True):
        broken = json.loads(json.dumps(run))
        broken["steps"][1]["time_s"] = bad
        with pytest.raises(ValueError, match="frame 2 time"):
            ReplaySession(broken)
    huge = json.loads(json.dumps(run))
    for index, step in enumerate(huge["steps"]):
        step["time_s"] = index * 1e12
    assert ReplaySession(huge)._delay_ms() <= 60_000, "a huge time step must not schedule an overflowing timer"


# ---- honest values ----------------------------------------------------------------------------------
def test_missing_velocity_is_not_recorded_and_never_at_rest():
    from desktop.presentation import per_drone_records

    run = json.loads(RUN.read_text())
    frame = run["steps"][3]
    agent = frame["agents"][0]
    agent.pop("velocity_mps")
    agent["position_m"] = None
    record = next(item for item in per_drone_records(run, frame) if item["drone_id"] == agent["agent_id"])
    assert record["speed_mps"] is None and record["heading_deg"] is None and record["altitude_m"] is None
    assert fmt.heading_text(record["heading_deg"], record["speed_mps"]) is None  # shows "Not recorded"
    assert fmt.heading_text(None, 0.0) == WIN_TEXT["atRest"], "a recorded zero speed is at rest"
    assert fmt.heading_text(90.0, 1.0) == "90°"


def test_non_numeric_statistics_are_not_recorded_and_not_plotted():
    from desktop.ui_viewers import timing_rows

    bad = {"unit": "s", "min": 0.01, "p50": "oops", "p95": float("nan"), "max": 0.08, "count": 3}
    assert fmt.stat_value(bad, "p50") == fmt.NOT_RECORDED
    assert fmt.stat_line(bad).startswith("Median Not recorded")
    unusable = {"unit": "s", "p50": "a", "max": None, "count": 3}
    view = fmt.report_view({"delay": bad, "age": unusable})
    assert [label for label, _stat in view["timings"]] == ["Delay"], "a stat with no numeric median or max is not plotted"
    rows = timing_rows(view["timings"])
    assert rows[0]["low"] == pytest.approx(10.0) and rows[0]["high"] == pytest.approx(80.0)
    assert rows[0]["median"] is None and rows[0]["p95"] is None, "bad values are skipped, never drawn as zero"
    assert fmt.format_value(float("nan"), "m") is None and fmt.format_vector([1, float("inf"), 2]) == fmt.NOT_RECORDED


def test_timing_rows_keep_a_p95_equal_to_the_median_and_mark_single_samples():
    from desktop.ui_viewers import timing_rows, timing_value_text

    same = {"unit": "s", "min": 0.02, "p50": 0.04, "p95": 0.04, "max": 0.08, "count": 300}
    single = {"unit": "s", "min": 0.012, "p50": 0.012, "p95": 0.012, "max": 0.012, "count": 1}
    rows = timing_rows([("Delay", same), ("Single", single)])
    assert rows[0]["p95"] == rows[0]["median"] == pytest.approx(40.0), "the ring is drawn even when it equals the median"
    assert not rows[0]["single"] and rows[1]["single"]
    assert timing_value_text(same) == "median 40 ms, p95 40 ms"
    assert timing_value_text(single).startswith(WIN_TEXT["timingSingle"])


def test_team_rows_mark_when_the_dash_legend_is_needed():
    from desktop.windows_preview import rows_have_missing, team_rows

    waiting = ReplaySession(RUN).presentation
    rows = team_rows(waiting["groups"], waiting["per_drone"])
    assert rows_have_missing(rows, set())
    later = ReplaySession(RUN)
    later.seek_frame(3)
    rows = team_rows(later.presentation["groups"], later.presentation["per_drone"])
    assert not rows_have_missing(rows, {row["iid"] for row in rows})
    loss = [child["values"][0] for row in rows for child in row["children"]]
    assert all(value.endswith("%") and "." in value for value in loss), "single loss values share one decimal"


@pytest.mark.parametrize("make,expected", [
    (lambda tmp: (tmp / "gone.json", None), "openMissing"),
    (lambda tmp: (tmp, None), "openNotFile"),
    (lambda tmp: (tmp / "x.json", PermissionError(13, "Permission denied")), "openPermission"),
    (lambda tmp: (tmp / "x.json", UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")), "openNotUtf8"),
])
def test_open_errors_are_plain_and_name_the_file(tmp_path, make, expected):
    from desktop.windows_preview import open_error_text

    path, exc = make(tmp_path)
    text = open_error_text(path, exc, WIN_TEXT["scenarioNoun"])
    template = WIN_TEXT[expected]
    assert text == template.format(name=path.name, noun=WIN_TEXT["scenarioNoun"])
    assert "Errno" not in text and "\\" not in text


def test_open_errors_unwrap_decoder_causes(tmp_path):
    from desktop.evidence import EvidenceError, load_evidence_report
    from desktop.windows_preview import open_error_text

    latin = tmp_path / "latin1.json"
    latin.write_bytes(b'{"name": "caf\xe9"}')
    broken = tmp_path / "broken.json"
    broken.write_text('{"kind": ', encoding="utf-8")
    for path, key in ((latin, "openNotUtf8"), (broken, "openNotJson")):
        with pytest.raises(EvidenceError) as caught:
            load_evidence_report(path)
        text = open_error_text(path, caught.value, WIN_TEXT["evidenceNoun"])
        assert text.startswith(WIN_TEXT[key].split("{name}")[0]) and path.name in text
        assert "Errno" not in text


# ---- real window measurements ---------------------------------------------------------------------
@pytest.fixture
def app():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no desktop display in this test environment")
    root.withdraw()
    from desktop.windows_preview import WindowsReplayApp

    application = WindowsReplayApp(root, ReplayModel.from_path(RUN), appearance="dark", persist_appearance=False)
    root.update_idletasks()
    yield application
    root.destroy()


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def test_every_title_and_timeline_control_is_one_height(app):
    height = app.px(WIN_LAYOUT["controlHeight"])
    controls = [widget for widget in _walk(app.root) if widget.winfo_class() in ("TButton", "TRadiobutton")
                and str(widget.cget("style")) in ("Tool.TButton", "Icon.TButton", "Primary.Icon.TButton", "Segment.Toolbutton")]
    assert len(controls) >= 10
    assert {widget.winfo_reqheight() for widget in controls} == {height}
    icons = [widget for widget in controls if "Icon" in str(widget.cget("style"))]
    assert {widget.winfo_reqwidth() for widget in icons} == {height}, "icon buttons are square"
    assert app.filter_entry.winfo_reqheight() == height


def test_segments_in_one_control_have_equal_widths(app):
    speeds = [widget for widget in _walk(app.root) if widget.winfo_class() == "TRadiobutton"
              and str(widget.cget("text")).endswith("x")]
    assert len(speeds) == 3 and len({widget.winfo_reqwidth() for widget in speeds}) == 1


def test_filter_shows_caption_empty_state_and_clear(app):
    app.session.seek_frame(3)
    app._render()
    app.filter_var.set("zzzz")
    app.root.update_idletasks()
    caption = str(app.filter_caption.cget("text"))
    assert caption.startswith(UI_TEXT["teamFilter"]["noMatches"].format(query="zzzz"))
    assert "Showing 3 of 50" in caption and UI_TEXT["teamFilter"]["pinnedNote"] in caption
    assert app.filter_caption.winfo_manager() == "grid" and app.clear_button.winfo_manager() == "place"
    app._clear_filter()
    assert app.filter_var.get() == "" and app.filter_caption.winfo_manager() == ""


def test_timing_plot_labels_sit_past_the_last_grid_line(app):
    from desktop.ui_viewers import timing_plot

    ctx = app._viewer_context()
    top = app.tk.Toplevel(app.root)
    stats = [("Target age", {"unit": "s", "min": 0.05, "p50": 0.1, "p95": 0.15, "max": 0.2, "count": 196}),
             ("Abstract event delay", {"unit": "s", "min": 0.02, "p50": 0.04, "p95": 0.04, "max": 0.08, "count": 300})]
    canvas = timing_plot(ctx, top, stats)
    canvas.configure(width=640)
    canvas.pack(fill="x")
    top.update()
    canvas.event_generate("<Configure>", width=640, height=int(canvas.cget("height")))
    top.update()
    lines = [canvas.coords(item) for item in canvas.find_all() if canvas.type(item) == "line"]
    grid_x = max(coords[0] for coords in lines if coords[0] == coords[2])
    labels = [item for item in canvas.find_all() if canvas.type(item) == "text" and "median" in canvas.itemcget(item, "text")]
    assert labels and all(canvas.bbox(item)[0] > grid_x for item in labels), "no median label crosses a tick line"
    rings = [item for item in canvas.find_all() if canvas.type(item) == "oval" and canvas.itemcget(item, "fill") == ""]
    dots = [item for item in canvas.find_all() if canvas.type(item) == "oval" and canvas.itemcget(item, "fill") != ""]
    assert len(rings) == 2 and len(dots) == 2
    ring, dot = canvas.coords(rings[1]), canvas.coords(dots[1])
    assert ring[0] < dot[0] and ring[2] > dot[2], "a p95 equal to the median shows as a halo around the dot"
    assert canvas.find_all().index(rings[1]) > canvas.find_all().index(dots[1]), "and is drawn on top of it"
