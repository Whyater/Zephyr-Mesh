"""Real-Tk checks for the Results window (skipped without a display, like test_windows_polish.py).

Each test drives the window the way a user would (shortcut handler, keys,
clicks, theme switch) and checks what is on screen against the raw fixture.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from desktop import results_data as rd
from desktop.design_tokens import UI_TEXT, palette
from desktop.replay import ReplayModel


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/s7-swarm/run.json"
RAW = json.loads((ROOT / "desktop" / "results_fixture.json").read_text(encoding="utf-8"))
TEXT, ATLAS = UI_TEXT["results"], UI_TEXT["atlas"]


@pytest.fixture
def app(monkeypatch):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no desktop display in this test environment")
    root.withdraw()
    from desktop import ui_results

    real_init = ui_results.ResultsWindow.__init__

    def quiet_init(self, *args, **kwargs):
        """The window is real and laid out, but fully transparent, so a test run never flashes over the desktop."""
        real_init(self, *args, **kwargs)
        try:
            self.window.attributes("-alpha", 0.0)
        except tk.TclError:
            pass

    monkeypatch.setattr(ui_results.ResultsWindow, "__init__", quiet_init)
    from desktop.windows_preview import WindowsReplayApp

    application = WindowsReplayApp(root, ReplayModel.from_path(RUN), appearance="dark", persist_appearance=False)
    root.update_idletasks()
    yield application
    root.destroy()


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def _texts(widget):
    found = []
    for child in _walk(widget):
        try:
            found.append(str(child.cget("text")))
        except Exception:
            continue
    return found


def _settle(results, rounds=3):
    for _ in range(rounds):
        results.window.update()


def _key(widget, key):
    """Run the widget's own binding for ``key`` as Tk would on a key press.

    A test process is not the active app, so Tk will not give it keyboard focus and a generated
    key event never arrives. Running the bound script directly exercises the same handler.
    Returns the handler's result ("break" when the key was used).
    """
    script = widget.bind(f"<{key}>")
    assert script, f"{widget} has no <{key}> binding"
    import tkinter as tk

    try:
        widget.tk.eval(re.sub(r"%.", "0", script))  # the event fields are unused by these handlers
        result = ""
    except tk.TclError as exc:  # a handler that returns "break" makes the bound script call Tcl's break
        if "break" not in str(exc):
            raise
        result = "break"
    if widget.winfo_exists():
        widget.update()
    return result


def test_shortcut_opens_one_window_and_reopening_focuses_it(app):
    from desktop.windows_preview import MENU_LAYOUT, SHORTCUT_HANDLERS, menu_accelerators

    assert SHORTCUT_HANDLERS["showResults"] == "_show_results"
    view_items = dict((action, label) for label, action, _kind in dict(MENU_LAYOUT)["View"] if action)
    assert view_items["showResults"] == UI_TEXT["menus"]["showResults"] and menu_accelerators()["showResults"] == "Ctrl+R"
    first = getattr(app, SHORTCUT_HANDLERS["showResults"])()
    _settle(first)
    assert first.window.title() == TEXT["windowTitle"]
    assert app._show_results() is first, "a second Ctrl+R focuses the open window"
    assert sum(isinstance(w, type(first.window)) and w.title() == TEXT["windowTitle"] for w in app.root.winfo_children()) == 1


def test_one_badge_and_the_empty_inspector_state(app):
    results = app._show_results("delay")
    _settle(results)
    texts = _texts(results.window)
    assert texts.count(TEXT["badge"]) == 1
    page = results.views["delay"].page
    assert not page.filled and TEXT["inspectorEmptyTitle"] in _texts(page.empty)
    assert TEXT["inspectorEmptyHintDelay"] in _texts(page.empty)


def test_arrow_keys_select_points_and_fill_the_inspector(app):
    results = app._show_results("delay")
    _settle(results)
    view = results.views["delay"]
    canvas = view.chart.canvas
    _key(canvas, "Right")  # first key selects the recovery point at 0 ms in the first panel
    assert view.chart.selected == rd.Point("constant_velocity", "independent", 0.0, "recovery", 0)
    _key(canvas, "Left")
    assert view.chart.selected.index == 0, "Left at the first delay stays put"
    for _ in range(9):
        _key(canvas, "Right")
    assert view.chart.selected.index == len(RAW["s6"]["delays_s"]) - 1
    _key(canvas, "Up")
    assert view.chart.selected.arm == "naive"
    _key(canvas, "bracketright")
    point = view.chart.selected
    assert (point.loss_model, point.loss_probability) == ("independent", 0.1)
    raw = next(s for s in RAW["s6"]["series"] if (s["motion"], s["loss_model"], s["loss_probability"], s["arm"])
               == ("constant_velocity", "independent", 0.1, "naive"))
    assert view.page.filled and str(view.page.title.cget("text")) == TEXT["armNaive"]
    assert view.tiles[0].text == f"{raw['p95_median_m'][6]:.3f} m"
    assert view.tiles[2].text == f"{raw['fail_count'][6]} of {raw['n_seeds'][6]}"
    assert view.tiles[3].text == (TEXT["cellFails"] if raw["label"][6] == "fails" else TEXT["cellPasses"])
    _key(canvas, "Escape")
    assert view.chart.selected is None and not view.page.filled
    assert results.exists(), "Escape with a selection clears it instead of closing the window"


def test_tab_walks_the_panels_then_leaves_the_chart(app):
    results = app._show_results("delay")
    _settle(results)
    chart = results.views["delay"].chart
    assert [chart.key("next") for _ in range(3)] == ["break"] * 3
    assert chart.cursor_panel == 3
    assert chart.key("next") is None, "Tab past the last panel moves focus on"


def test_click_on_a_drawn_point_selects_it(app):
    results = app._show_results("delay")
    _settle(results)
    chart = results.views["delay"].chart
    target = rd.Point("constant_velocity", "independent", 0.3, "naive", 4)
    x, y = chart.point_xy(target)
    chart.canvas.event_generate("<Button-1>", x=int(round(x)), y=int(round(y)))
    chart.canvas.update()
    assert chart.selected == target
    assert chart.canvas.find_withtag("overlay"), "the selection ring is drawn"


def test_switching_controls_keeps_the_scale_and_moves_the_selection(app):
    results = app._show_results("delay")
    _settle(results)
    view = results.views["delay"]
    view.chart.select(rd.Point("constant_velocity", "independent", 0.3, "recovery", 2))
    view.path_var.set("constant_turn")
    view.loss_var.set("burst")
    view._changed()
    _settle(results)
    assert view.chart.selected == rd.Point("constant_turn", "burst", 0.3, "recovery", 2)
    assert view.finding.cget("text") == RAW["s6"]["findings"][-1]["text"]
    labels = [view.chart.canvas.itemcget(item, "text") for item in view.chart.canvas.find_all()
              if view.chart.canvas.type(item) == "text"]
    assert {"0.0", "0.5", "1.0", "1.5"} <= set(labels), "the y axis never rescales"
    assert {"No loss", "10 % loss", "30 % loss", "50 % loss"} <= set(labels)


def test_atlas_keyboard_selection_and_inspector(app):
    results = app._show_results("atlas")
    _settle(results)
    view = results.views["atlas"]
    canvas = view.grid.canvas
    _key(canvas, "Down")  # first key selects the top-left cell
    stratum = RAW["atlas"]["strata"][0]
    assert view.grid.selected == (stratum["variants"][0], stratum["arms"][0])
    _key(canvas, "Down")
    _key(canvas, "Right")
    variant, arm = stratum["variants"][1], stratum["arms"][1]
    assert view.grid.selected == (variant, arm)
    entry = next(e for e in RAW["atlas"]["entries"] if (e["stratum"], e["variant"], e["loss_condition"], e["arm"])
                 == ("central", variant, "independent:0", arm))
    assert str(view.page.title.cget("text")) == RAW["atlas"]["arm_labels"][arm]
    assert view.tiles[0].text == rd.tolerated_text(entry)
    assert view.tiles[3].text == ATLAS["seedsPerCell"].format(n=RAW["atlas"]["n_seeds"])
    selected_rows = [row for row, _name, _value in view._row_widgets if str(row.cget("style")) == "Results.RowSelected.TFrame"]
    assert len(selected_rows) == 1
    _key(canvas, "End")
    assert view.grid.selected[1] == stratum["arms"][-1]


def test_atlas_conditions_change_keeps_a_selection_that_still_exists(app):
    results = app._show_results("atlas")
    _settle(results)
    view = results.views["atlas"]
    view.grid.select(("cv_1", "recovery"))
    view.stratum_var.set("noise10")
    view._changed()
    assert view.grid.selected == ("cv_1", "recovery")
    view.grid.select(("ct_r1", "adaptive"))
    view.stratum_var.set("central")
    view._changed()
    view.grid.select(("cv_0.5", "imm"))
    view.stratum_var.set("noise0")
    view._changed()
    assert view.grid.selected is None and not view.page.filled, "IMM is not run in that stratum"


def test_select_specs_populate_both_views(app):
    results = app._show_results("atlas", rd.parse_select_spec("stratum=central,loss=burst,level=0.3,variant=figure8,arm=ct"))
    _settle(results)
    atlas = results.views["atlas"]
    assert atlas.grid.selected == ("figure8", "ct") and atlas.loss_key() == "burst:0.3"
    assert atlas.tiles[0].text == ATLAS["failsAtZeroLong"]
    results = app._show_results("delay", rd.parse_select_spec("path=turn,loss=independent,level=0.3,arm=recovery,delay=0.2"))
    _settle(results)
    delay = results.views["delay"]
    assert delay.chart.selected == rd.Point("constant_turn", "independent", 0.3, "recovery", 5)
    assert str(delay.page.subtitle.cget("text")) == "1 m turn, 30 % random, 200 ms"


def test_limits_and_source_are_collapsed_and_verbatim(app):
    results = app._show_results("delay")
    _settle(results)
    view = results.views["delay"]
    assert not view.limits.expanded and not view.source.expanded
    texts = _texts(view.limits.body)
    for limit in RAW["s6"]["limits"]:
        assert limit in texts
    source = _texts(view.source.body)
    assert RAW["s6"]["source"]["sha256"][:12] in source and RAW["s6"]["source"]["sha256"] not in source
    assert RAW["s6"]["source"]["run_dir"] in source


def test_theme_switch_redraws_in_place_and_close_reopen_works(app):
    results = app._show_results("delay")
    _settle(results)
    chart = results.views["delay"].chart
    before = chart.draw_count
    app._set_appearance("light")
    _settle(results)
    assert chart.draw_count == before + 1
    assert chart.canvas.cget("background").lower() == palette("light").surface.lower()
    results.destroy()
    app._set_appearance("dark")  # the closed window's hook is dropped, not raised
    again = app._show_results("atlas")
    _settle(again)
    assert again is not results and again.exists() and again.view == "atlas"
    assert again.views["atlas"].grid.canvas.cget("background").lower() == palette("dark").surface.lower()


def test_hover_redraws_only_the_overlay(app):
    results = app._show_results("delay")
    _settle(results)
    chart = results.views["delay"].chart
    before = chart.draw_count
    point = rd.Point("constant_velocity", "independent", 0.0, "naive", 3)
    x, y = chart.point_xy(point)
    chart.canvas.event_generate("<Motion>", x=int(round(x)), y=int(round(y)))
    chart.canvas.update()
    assert chart.hover == point and chart.draw_count == before
    assert rd.hover_text(rd.load_fixture(), point) == chart._tip(int(round(x)), int(round(y)))


def test_a_broken_fixture_shows_a_plain_failure_state(app):
    from desktop.ui_results import open_results

    def broken():
        raise rd.FixtureError("results_fixture.json was not found")

    results = open_results(app._viewer_context(), "delay", loader=broken)
    _settle(results)
    texts = _texts(results.window)
    assert TEXT["loadFailed"] in texts and "results_fixture.json was not found" in texts
    assert not results.views and texts.count(TEXT["badge"]) == 1
    results.destroy()


def test_sidebar_selection_switches_views(app):
    results = app._show_results("delay")
    _settle(results)
    results.sidebar.selection_set("atlas")  # what Down does in the list
    _settle(results)
    assert results.view == "atlas" and str(results.heading.cget("text")) == TEXT["atlasTitle"]
    assert "atlas" in results.views and not results.views["delay"].frame.winfo_ismapped()
    results.sidebar.selection_set("delay")
    _settle(results)
    assert results.view == "delay"


def test_escape_clears_a_selection_first_then_closes_the_window(app):
    results = app._show_results("delay")
    _settle(results)
    chart = results.views["delay"].chart
    chart.select(rd.Point("constant_velocity", "independent", 0.0, "naive", 1))
    assert _key(chart.canvas, "Escape") == "break" and chart.selected is None
    assert _key(chart.canvas, "Escape") == "", "no selection: the key falls through to the window"
    _key(results.window, "Escape")
    assert not results.exists()
