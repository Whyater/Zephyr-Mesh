"""Real-Tk checks for the Results window (skipped without a display, like test_windows_polish.py).

Each test drives the window the way a user would (shortcut handler, keys,
clicks, theme switch) and checks what is on screen against the raw fixture.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools.desktop import results_data as rd
from tools.desktop.design_tokens import UI_TEXT, palette
from tools.desktop.replay import ReplayModel


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/s7-swarm/run.json"
RAW = json.loads((ROOT / "tools" / "desktop" / "results_fixture.json").read_text(encoding="utf-8"))
TEXT, ATLAS = UI_TEXT["results"], UI_TEXT["atlas"]


@pytest.fixture
def app(monkeypatch):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no desktop display in this test environment")
    root.withdraw()
    # On macOS an outside "quit" request (another process asking the Python app to quit) makes Tk exit the whole test run
    # quietly. Ignore it here; nothing in these tests needs the application quit handler.
    root.tk.eval("proc ::tk::mac::Quit {args} {}")
    from tools.desktop import ui_results

    real_init = ui_results.ResultsWindow.__init__

    def quiet_init(self, *args, **kwargs):
        """The window is real and laid out, but fully transparent, so a test run never flashes over the desktop."""
        real_init(self, *args, **kwargs)
        try:
            self.window.attributes("-alpha", 0.0)
        except tk.TclError:
            pass

    monkeypatch.setattr(ui_results.ResultsWindow, "__init__", quiet_init)
    from tools.desktop.windows_preview import WindowsReplayApp

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
    from tools.desktop.windows_preview import MENU_LAYOUT, SHORTCUT_HANDLERS, menu_accelerators

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


def _canvas_texts(canvas):
    return [canvas.itemcget(item, "text") for item in canvas.find_all() if canvas.type(item) == "text"]


def test_switching_controls_rescales_y_and_moves_the_selection(app):
    results = app._show_results("delay")
    _settle(results)
    view = results.views["delay"]
    assert {"0.0", "0.1", "0.2", "0.3", "0.4"} <= set(_canvas_texts(view.chart.canvas)), "straight path, random loss: 0 to 0.4 m"
    assert "0.5" not in _canvas_texts(view.chart.canvas) and "1.0" not in _canvas_texts(view.chart.canvas)
    view.chart.select(rd.Point("constant_velocity", "independent", 0.3, "recovery", 2))
    view.path_var.set("constant_turn")
    view.loss_var.set("burst")
    view._changed()
    _settle(results)
    assert view.chart.selected == rd.Point("constant_turn", "burst", 0.3, "recovery", 2)
    labels = _canvas_texts(view.chart.canvas)
    assert {"0.0", "0.5", "1.0", "1.5"} <= set(labels), "turn, bursty loss: 0 to 1.5 m"
    assert "0.1" not in labels
    assert {"No loss", "10 % loss", "30 % loss", "50 % loss"} <= set(labels)
    view.path_var.set("constant_velocity")
    view._changed()
    _settle(results)
    assert "1.5" not in _canvas_texts(view.chart.canvas) and "0.5" in _canvas_texts(view.chart.canvas), "straight, bursty: 0.5"


def test_failure_line_label_has_three_decimals(app):
    results = app._show_results("delay")
    _settle(results)
    labels = _canvas_texts(results.views["delay"].chart.canvas)
    assert labels.count("Failure line 0.250 m") == 4, "one caption per panel"


def test_lead_finding_hides_the_others_in_a_collapsed_section(app):
    results = app._show_results("delay")
    _settle(results)
    view = results.views["delay"]
    findings = RAW["s6"]["findings"]
    assert view.finding.text == findings[1]["text"].replace("`", "")
    assert "`" not in " ".join(_texts(view.finding.frame))
    disclosure = view.finding.disclosure
    assert disclosure is not None and not disclosure.expanded
    assert TEXT["moreFindings"] in " ".join(_texts(disclosure.frame)).replace("  ", " ") or any(
        TEXT["moreFindings"] in text for text in _texts(disclosure.frame))
    assert not disclosure.body.winfo_ismapped()
    disclosure.toggle()
    _settle(results)
    shown = _texts(disclosure.body)
    assert findings[0]["text"].replace("`", "") in shown and findings[2]["text"].replace("`", "") in shown
    assert findings[1]["text"].replace("`", "") not in shown
    assert TEXT["sourcePrefix"].format(source="README.md") in shown
    # The finding no longer changes with the path.
    view.path_var.set("constant_turn")
    view._changed()
    assert view.finding.text == findings[1]["text"].replace("`", "")


def test_atlas_has_a_single_finding_and_no_more_section(app):
    results = app._show_results("atlas")
    _settle(results)
    finding = results.views["atlas"].finding
    assert finding.text == RAW["atlas"]["findings"][0]["text"].replace("`", "") and finding.disclosure is None


def test_atlas_legend_sits_between_the_summary_and_the_grid(app):
    results = app._show_results("atlas")
    _settle(results)
    view = results.views["atlas"]
    summary_bottom = view.summary.winfo_y() + view.summary.winfo_height()
    assert summary_bottom <= view.legend.canvas.winfo_y()
    assert view.legend.canvas.winfo_y() + view.legend.canvas.winfo_height() <= view.grid.frame.winfo_y()
    assert ATLAS["clearWinner"] in _canvas_texts(view.legend.canvas)


def test_legend_sits_above_the_charts(app):
    results = app._show_results("delay")
    _settle(results)
    view = results.views["delay"]
    assert view.legend.canvas.winfo_y() + view.legend.canvas.winfo_height() <= view.chart.canvas.winfo_y()
    assert TEXT["armNaive"] in _canvas_texts(view.legend.canvas)


def test_limits_notes_match_the_macos_window(app):
    results = app._show_results("delay")
    _settle(results)
    texts = _texts(results.views["delay"].limits.body)
    s6 = RAW["s6"]
    for note in (s6["status"][:1].upper() + s6["status"][1:] + ".", s6["criterion"][:1].upper() + s6["criterion"][1:] + ".",
                 ATLAS["cellRule"].format(status=RAW["atlas"]["failure_rule"]["cell_rule_status"])):
        assert note in texts


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
    assert atlas.tiles[0].text == "Fails at 0 ms"
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
    from tools.desktop.ui_results import open_results

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


def test_scale_note_follows_the_controls(app):
    results = app._show_results("delay")
    _settle(results)
    view = results.views["delay"]
    assert str(view.scale_note.cget("text")) == "Error axis 0 to 0.400 m for this path and loss type"
    view.path_var.set("constant_turn")
    view.loss_var.set("burst")
    view._changed()
    assert str(view.scale_note.cget("text")) == "Error axis 0 to 1.500 m for this path and loss type"


def test_failure_caption_slots_are_valid_and_clear_of_the_data(app):
    results = app._show_results("delay")
    _settle(results)
    chart = results.views["delay"].chart
    layout = chart._layout(float(chart.canvas.winfo_width()))
    slots = chart.label_slots(layout, 120.0, 14.0)
    assert len(slots) == 4 and set(slots) <= {0, 1, 2, 3}
    assert chart.label_slots(layout, 1.0, 1.0) == [0, 0, 0, 0], "nothing in the way: right end, above the line, every panel"


def test_the_turn_bursty_no_loss_label_moves_off_the_curves(app):
    """The right end above the line runs into the naive curve there, so the label takes another spot. Which one depends on the
    panel width: below the line at the right end, or the left end."""
    results = app._show_results("delay", rd.parse_select_spec("path=turn,loss=burst"))
    _settle(results)
    chart = results.views["delay"].chart
    width, height = chart.ctx.fonts["caption"].measure("Failure line 0.250 m"), chart.ctx.fonts["caption"].metrics("linespace")
    layout = chart._layout(float(chart.canvas.winfo_width()))
    slot = chart.label_slots(layout, float(width), float(height))[0]
    assert slot != 0, slot


def test_atlas_winner_gets_a_bar_and_bold_text_and_cells_keep_an_8_pt_gap(app):
    from tools.desktop.design_tokens import LAYOUT, SPACE

    results = app._show_results("atlas")
    _settle(results)
    grid = results.views["atlas"].grid
    ctx = app._viewer_context()
    assert grid.gap == ctx.px(SPACE["sm"]) == ctx.px(8)
    canvas = grid.canvas
    accent = ctx.theme().p.accent
    bars = [item for item in canvas.find_all() if canvas.type(item) == "polygon" and canvas.itemcget(item, "fill") == accent
            and canvas.itemcget(item, "outline") == ""]
    winners = sum(1 for variant in grid.variants
                  if (c := grid.data.cell("central", variant, "independent:0")) and c["kind"] == "winner")
    assert len(bars) == winners > 0, "one accent bar per clear winner, no accent borders"
    for item in bars:
        x0, y0, x1, y1 = canvas.bbox(item)
        assert y1 - y0 <= ctx.px(LAYOUT["atlasWinnerBar"]) + 2
    fonts = {canvas.itemcget(item, "font") for item in canvas.find_all() if canvas.type(item) == "text"}
    assert any("bold" in str(canvas.tk.call("font", "actual", font)) for font in fonts)


def test_selection_ring_is_drawn_outside_the_cell(app):
    results = app._show_results("atlas")
    _settle(results)
    grid = results.views["atlas"].grid
    grid.select((grid.variants[1], grid.arms[1]))
    _settle(results)
    x0, y0, x1, y1 = grid._cells[(1, 1)]
    canvas = grid.canvas
    ring = [item for item in canvas.find_withtag("overlay") if canvas.itemcget(item, "outline") == app._viewer_context().theme().p.accent]
    assert ring
    bx0, by0, bx1, by1 = canvas.bbox(ring[0])
    assert bx0 < x0 - 2 and by0 < y0 - 2 and bx1 > x1 + 2 and by1 > y1 + 2


def test_cells_shrink_to_the_minimum_width_before_the_grid_scrolls(app):
    from tools.desktop.design_tokens import WIN_LAYOUT

    results = app._show_results("atlas")
    _settle(results)
    grid = results.views["atlas"].grid
    ctx = app._viewer_context()
    full, minimum = ctx.px(WIN_LAYOUT["atlasCellWidth"]), max(grid.min_cell, ctx.px(WIN_LAYOUT["atlasCellMinWidth"]))
    assert WIN_LAYOUT["atlasCellMinWidth"] == 84 and WIN_LAYOUT["resultsWindow"]["width"] == 1440
    assert grid.cell_width(5000) == full and grid.cell_width(10) == minimum
    widths = [grid.cell_width(width) for width in range(400, 1600, 20)]
    assert widths == sorted(widths) and all(minimum <= w <= full for w in widths)


def test_a_tie_of_three_trackers_wraps_inside_the_most_delay_column(app):
    results = app._show_results("atlas", {"stratum": "student_t"})
    _settle(results)
    grid = results.views["atlas"].grid
    canvas = grid.canvas
    texts = _canvas_texts(canvas)
    assert any(text.startswith("Tie: ") and "," in text for text in texts) or any(text.startswith("Tie: ") for text in texts)
    boxes = [canvas.bbox(item) for item in canvas.find_all() if canvas.type(item) == "text"]
    right = grid._total_width
    assert all(box[2] <= right + 1 for box in boxes), "no text runs past the grid"


def test_atlas_inspector_header_and_interval_caption(app):
    results = app._show_results("atlas", rd.parse_select_spec("stratum=central,loss=burst,level=0.3,variant=figure8,arm=ct"))
    _settle(results)
    view = results.views["atlas"]
    entry = next(e for e in RAW["atlas"]["entries"] if (e["stratum"], e["variant"], e["loss_condition"], e["arm"])
                 == ("central", "figure8", "burst:0.3", "ct"))
    assert str(view.page.subtitle.cget("text")) == "Figure eight, 30 % bursty"
    assert str(view.page.detail.cget("text")) == RAW["atlas"]["strata"][0]["title"]
    assert str(view.interval_note.cget("text")) == rd.interval_caption(entry, RAW["atlas"]["n_seeds"])
    assert view.tiles[1].text == rd.interval_text(entry["interval_s"])


def test_clicking_a_loss_row_shows_that_loss_in_the_atlas(app):
    results = app._show_results("atlas", rd.parse_select_spec("stratum=central,loss=independent,level=0,variant=cv_1,arm=recovery"))
    _settle(results)
    view = results.views["atlas"]
    conditions = RAW["atlas"]["strata"][0]["loss_conditions"]
    index = conditions.index("burst:0.5")
    view._row_widgets[index][1].event_generate("<Button-1>", x=2, y=2)
    _settle(results)
    assert view.loss_key() == "burst:0.5" and view.grid.loss == "burst:0.5"
    view._row_widgets[conditions.index("independent:0")][0].event_generate("<Button-1>", x=2, y=2)
    _settle(results)
    assert view.loss_key() == "independent:0"
    assert view.grid.selected == ("cv_1", "recovery"), "the selection stays put"


def test_the_window_takes_its_size_from_the_layout_tokens(app):
    from tools.desktop.design_tokens import WIN_LAYOUT

    results = app._show_results("delay")
    _settle(results)
    ctx = app._viewer_context()
    size = WIN_LAYOUT["resultsWindow"]
    assert size["width"] == 1440
    assert tuple(results.window.wm_minsize()) == (ctx.px(size["minWidth"]), ctx.px(size["minHeight"]))


def _pending(widget):
    """Ids of every pending ``after`` job in this interpreter (timers and idle callbacks)."""
    return set(widget.tk.splitlist(widget.tk.call("after", "info")))


def test_hover_leaves_no_after_job_pending_and_the_idle_window_does_not_redraw(app):
    import time

    results = app._show_results("delay")
    _settle(results)
    chart = results.views["delay"].chart
    atlas_draws = chart.draw_count
    time.sleep(0.2)
    _settle(results)
    baseline = _pending(app.root)
    point = rd.Point("constant_velocity", "independent", 0.0, "naive", 3)
    x, y = (int(round(v)) for v in chart.point_xy(point))
    chart.canvas.event_generate("<Motion>", x=x, y=y)
    chart.canvas.update()
    assert len(_pending(app.root) - baseline) == 1, "the hint waits on one one-shot timer"
    chart.canvas.event_generate("<Leave>")
    chart.canvas.update()
    assert _pending(app.root) == baseline, "leaving cancels it"
    # A hint that is allowed to fire runs its one-shot job and leaves nothing behind once the pointer goes.
    chart.canvas.event_generate("<Motion>", x=x, y=y)
    chart.canvas.update()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and chart.tooltip._window is None:
        chart.canvas.update()
        time.sleep(0.05)
    assert chart.tooltip._window is not None, "the hint appears after its delay"
    chart.canvas.event_generate("<Leave>")
    chart.canvas.update()
    assert chart.tooltip._window is None and _pending(app.root) == baseline
    # Idle: nothing in the Results window redraws or schedules anything by itself.
    results.sidebar.selection_set("atlas")
    _settle(results)
    grid = results.views["atlas"].grid
    before = (chart.draw_count, grid.draw_count, _pending(app.root))
    time.sleep(1.2)
    _settle(results)
    assert (chart.draw_count, grid.draw_count, _pending(app.root)) == before
    assert atlas_draws >= 1


def test_legend_swatches_have_square_corners_and_hairline_outlines(app):
    results = app._show_results("atlas")
    _settle(results)
    canvas = results.views["atlas"].legend.canvas
    p = app._viewer_context().theme().p
    kinds = [canvas.type(item) for item in canvas.find_all()]
    assert "polygon" not in kinds, "no smoothed shapes in the legend, which scalloped at small sizes"
    outlined = [item for item in canvas.find_all() if canvas.type(item) == "rectangle" and canvas.itemcget(item, "outline") == p.line]
    assert len(outlined) == 5, "four ramp swatches and the winner box have the `line` outline"
    assert any(canvas.itemcget(item, "fill") == p.accent for item in canvas.find_all() if canvas.type(item) == "rectangle")


def test_conditions_description_is_a_tooltip_not_a_label(app):
    results = app._show_results("atlas")
    _settle(results)
    view = results.views["atlas"]
    strata = {item["id"]: item for item in RAW["atlas"]["strata"]}
    assert not hasattr(view, "description")
    assert RAW["atlas"]["strata"][0]["description"] not in " ".join(_texts(view.frame))
    assert view.conditions_tip.text == strata["central"]["description"]
    view.stratum_var.set("noise10")
    view._changed()
    assert view.conditions_tip.text == strata["noise10"]["description"]


def test_a_unit_never_wraps_alone_in_a_tile(app):
    results = app._show_results("delay", rd.parse_select_spec("path=turn,loss=independent,level=0.3,arm=recovery,delay=0.2"))
    _settle(results)
    view = results.views["delay"]
    shown = str(view.tiles[1].value.cget("text"))
    assert shown.endswith(" m") and view.tiles[1].text.endswith(" m")
    assert shown.replace(" ", " ") == view.tiles[1].text


def test_winner_text_is_centred_like_the_other_cells(app):
    results = app._show_results("atlas")
    _settle(results)
    grid = results.views["atlas"].grid
    canvas = grid.canvas
    cell = next(c for c in RAW["atlas"]["cells"] if c["stratum"] == "central" and c["loss_condition"] == "independent:0"
                and c["kind"] == "winner")
    row, column = grid.variants.index(cell["variant"]), grid.arms.index(cell["arms"][0])
    x0, y0, x1, y1 = grid._cells[(row, column)]
    texts = [item for item in canvas.find_overlapping(x0, y0, x1, y1) if canvas.type(item) == "text"]
    assert texts
    bx0, by0, bx1, by1 = canvas.bbox(texts[0])
    assert abs((by0 + by1) / 2.0 - (y0 + y1) / 2.0) <= 1.5


def test_segments_share_a_minimum_width_and_butt_together(app):
    from tools.desktop.ui_results import SEGMENT_MIN_CHARS

    results = app._show_results("atlas")
    _settle(results)
    view = results.views["atlas"]
    buttons = [w for w in _walk(view.frame) if w.winfo_class() == "TRadiobutton"]
    assert buttons and all(int(str(b.cget("width"))) >= SEGMENT_MIN_CHARS for b in buttons)
    assert all(b.pack_info()["padx"] in (0, "0", (0, 0)) for b in buttons)


def test_delay_chart_has_vertical_gridlines_and_air_around_the_panel(app):
    from tools.desktop.design_tokens import SPACE

    results = app._show_results("delay")
    _settle(results)
    chart = results.views["delay"].chart
    canvas = chart.canvas
    p = app._viewer_context().theme().p
    layout = chart._layout(float(canvas.winfo_width()))
    box, scale = layout[0]
    vertical = [item for item in canvas.find_all() if canvas.type(item) == "line" and canvas.itemcget(item, "fill") == p.line
                and abs(canvas.coords(item)[0] - canvas.coords(item)[2]) < 0.5]
    xs = sorted(round(canvas.coords(item)[0]) for item in vertical)
    for ms in (100, 200, 300):
        assert round(scale.x(ms)) in xs
    pad = app._viewer_context().px(SPACE["sm"])
    title = next(i for i in canvas.find_all() if canvas.type(i) == "text" and canvas.itemcget(i, "text") == "No loss")
    assert canvas.coords(title)[0] - box[0] >= pad and canvas.coords(title)[1] - box[1] >= pad


def test_inspector_arm_help_has_a_gap_above_and_the_selected_loss_row_is_not_bold(app):
    from tools.desktop.design_tokens import SPACE

    results = app._show_results("atlas", rd.parse_select_spec("stratum=central,loss=burst,level=0.3,variant=figure8,arm=ct"))
    _settle(results)
    view = results.views["atlas"]
    top = view.help.pack_info()["pady"]
    top = top[0] if isinstance(top, (tuple, list)) else top
    assert int(str(top)) == app._viewer_context().px(SPACE["xs"])
    selected = next(name for row, name, _value in view._row_widgets if str(row.cget("style")) == "Results.RowSelected.TFrame")
    font = str(ttk_style_font(app, "Results.RowSelected.TLabel"))
    assert "-weight bold" not in font, font
    assert str(selected.cget("style")) == "Results.RowSelected.TLabel"


def ttk_style_font(app, style_name):
    from tkinter import ttk

    font = ttk.Style(app.root).lookup(style_name, "font")
    return app.root.tk.call("font", "actual", font)
