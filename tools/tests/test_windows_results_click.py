"""Click-everything coverage for the Windows Results window.

The window is real, laid out and mapped, but fully transparent. Keys are delivered to the focused canvas the way Tk
delivers them to a user, and spies on the chart, the grid and the wheel routing prove each handler really ran.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "sim/runs/s7-swarm/run.json"

PUSH_CLASSES = {"Button", "TButton", "Checkbutton", "TCheckbutton", "Radiobutton", "TRadiobutton"}
MENU_BUTTONS = {"Menubutton", "TMenubutton"}
KEYS = ("Left", "Right", "Up", "Down", "bracketleft", "bracketright", "Tab", "Shift-Tab")


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
    # A transient window is only viewable while its parent is, so the parent is shown too, but fully transparent.
    try:
        root.attributes("-alpha", 0.0)
    except tk.TclError:
        root.destroy()
        pytest.skip("this display cannot show a transparent window, so the test would flash over the desktop")
    root.deiconify()
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
    from tools.desktop.replay import ReplayModel
    from tools.desktop.windows_preview import WindowsReplayApp

    application = WindowsReplayApp(root, ReplayModel.from_path(RUN), appearance="dark", persist_appearance=False)
    root.update_idletasks()
    yield application
    root.destroy()


def _spy(monkeypatch, cls, name, tally, label):
    original = getattr(cls, name)

    def wrapper(self, *args, **kwargs):
        tally[label] += 1
        return original(self, *args, **kwargs)

    wrapper.__name__ = name
    monkeypatch.setattr(cls, name, wrapper)


@pytest.fixture
def tally(app, monkeypatch):
    """Counts of what really happened, installed before any Results window (and so any canvas binding) exists."""
    from tools.desktop import ui_results, ui_results_atlas, ui_results_chart, ui_widgets

    counts = Counter()
    _spy(monkeypatch, ui_results_chart.DelayChart, "key", counts, "chart_key")
    _spy(monkeypatch, ui_results_chart.DelayChart, "_on_click", counts, "chart_click")
    _spy(monkeypatch, ui_results_atlas.AtlasGrid, "key", counts, "grid_key")
    _spy(monkeypatch, ui_results_atlas.AtlasGrid, "_on_click", counts, "grid_click")
    _spy(monkeypatch, ui_results.AtlasView, "pick_loss", counts, "pick_loss")
    _spy(monkeypatch, ui_results.ResultsWindow, "_wheel", counts, "wheel")
    _spy(monkeypatch, ui_widgets.ScrollFrame, "scroll", counts, "scrolled")
    return counts


# ---- small helpers: only a widget that was destroyed mid-walk is tolerated ------------------------------------------
def _alive(widget):
    import tkinter as tk

    try:
        return bool(widget.winfo_exists())
    except tk.TclError:
        return False


def _walk(widget):
    """Yield a snapshot of a widget subtree; a widget rebuilt by a callback while walking is simply skipped."""
    import tkinter as tk

    if not _alive(widget):
        return
    yield widget
    try:
        children = tuple(widget.winfo_children())
    except tk.TclError:
        if _alive(widget):
            raise
        return
    for child in children:
        yield from _walk(child)


def _settle(root):
    root.update_idletasks()
    root.update()


def _enabled(widget):
    from tkinter import ttk

    if isinstance(widget, ttk.Widget):
        return "disabled" not in widget.state()
    return str(widget.cget("state")) != "disabled"


def _focus(root, widget):
    """Give the widget keyboard focus and confirm it, or skip loudly: a key sent without focus would reach nothing."""
    widget.focus_force()
    _settle(root)
    if root.focus_get() is not widget:
        pytest.skip(f"this display gives no keyboard focus to {widget}, so key handlers cannot be exercised")


def _owner_of(results, canvas):
    """The DelayChart or AtlasGrid that owns this canvas, if any."""
    for view in results.views.values():
        for attr in ("chart", "grid"):
            owner = getattr(view, attr, None)
            if owner is not None and owner.canvas is canvas:
                return owner
    return None


def _click_points(owner, canvas, limit=10):
    """Places on the canvas that really hold a data point or a cell, spread across the whole chart."""
    points = []
    if owner is not None and hasattr(owner, "_points"):
        points = [(x, y) for x, y, _point in owner._points]
    elif owner is not None and hasattr(owner, "_cells"):
        left, width = canvas.canvasx(0), canvas.winfo_width()
        for box in owner._cells.values():
            x = (box[0] + box[2]) / 2.0 - left
            if 0 <= x < width:
                points.append((x, (box[1] + box[3]) / 2.0))
    return points[::max(1, len(points) // limit)][:limit]


def _labels_with_text(window):
    import tkinter as tk

    values = []
    for widget in _walk(window):
        if widget.winfo_class() not in {"Label", "TLabel"}:
            continue
        try:
            text = str(widget.cget("text"))
        except tk.TclError:
            continue
        if text.strip():
            values.append(text)
    return values


# ---- the exercise ---------------------------------------------------------------------------------------------------
class _Driver:
    def __init__(self, results, root, tally, view):
        self.results, self.root, self.tally, self.view = results, root, tally, view
        self.done: set[str] = set()

    def act(self, kind, widget, callback):
        """Run one user-like action; count it only when it really ran."""
        import tkinter as tk

        if not _alive(widget):
            return False
        try:
            callback()
        except tk.TclError:
            if _alive(widget):
                raise
            return False
        self.tally[kind] += 1
        _settle(self.root)
        return True

    def sidebar_rows(self, tree):
        rows = []
        pending = list(tree.get_children(""))
        while pending:
            row = pending.pop(0)
            rows.append(row)
            pending.extend(tree.get_children(row))
        return rows

    def tree(self, tree):
        rows = self.sidebar_rows(tree)
        is_sidebar = tree is self.results.sidebar
        if is_sidebar:
            rows.sort(key=lambda row: row == self.view)  # finish on the view under test, so its canvas is the one on screen
        for row in rows:
            def select(row=row):
                tree.selection_set(row)
                tree.focus(row)
                tree.event_generate("<<TreeviewSelect>>")

            self.act("tree", tree, select)
            if is_sidebar:
                assert self.results.view == row, f"selecting {row!r} in the sidebar left {self.results.view!r} showing"
                self.tally["view_switched"] += 1

    def combo(self, box):
        for value in box.tk.splitlist(box.cget("values")):
            self.act("combo", box, lambda value=value: (box.set(value), box.event_generate("<<ComboboxSelected>>")))

    def menu(self, button):
        """Pick each entry of a menubutton's menu directly: invoke() posts no popup, so the run stays non-modal."""
        menu = button.nametowidget(str(button.cget("menu")))
        last = menu.index("end")
        for index in range(0 if last is None else last + 1):
            kind = menu.type(index)
            if kind not in {"command", "radiobutton", "checkbutton"}:
                continue
            label = menu.entrycget(index, "label")
            self.act("menu", button, lambda index=index: menu.invoke(index))
            if kind == "radiobutton":
                assert str(button.cget("text")).strip() == label, f"menu entry {label!r} did not reach the menubutton"
                self.tally["menu_effect"] += 1

    def canvas(self, canvas):
        owner = _owner_of(self.results, canvas)
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        spots = _click_points(owner, canvas)
        spots += [(int(width * (c + 0.5) / 6), int(height * (r + 0.5) / 4)) for c in range(6) for r in range(4)]
        for x, y in spots:
            before = None if owner is None else owner.selected
            self.act("canvas_event", canvas, lambda x=x, y=y: (canvas.event_generate("<Motion>", x=x, y=y),
                                                               canvas.event_generate("<Button-1>", x=x, y=y)))
            if owner is not None and owner.selected != before:
                self.tally["click_selects"] += 1
        if owner is None:
            return
        for key in KEYS:
            if not canvas.bind(f"<{key}>"):
                continue
            before = owner.selected
            _focus(self.root, canvas)
            self.act("key_event", canvas, lambda key=key: canvas.event_generate(f"<{key}>"))
            if owner.selected != before:
                self.tally["key_moves"] += 1

    def bound_click(self, widget):
        """Plain labels and frames that carry their own click binding (the loss-level rows in the atlas inspector)."""
        self.act("bound_click", widget, lambda: (widget.event_generate("<Button-1>", x=1, y=1),
                                                 widget.event_generate("<ButtonRelease-1>", x=1, y=1)))

    def one(self, widget):
        cls = widget.winfo_class()
        if cls in PUSH_CLASSES:
            if _enabled(widget):
                self.act("invoke", widget, widget.invoke)
        elif cls == "TCombobox":
            self.combo(widget)
        elif cls == "Treeview":
            self.tree(widget)
        elif cls in MENU_BUTTONS:
            self.menu(widget)
        elif cls == "Canvas":
            self.canvas(widget)
        elif cls != "Menu" and widget.bind("<Button-1>"):
            self.bound_click(widget)

    def sweep(self):
        """Repeat full passes over what is on screen until a pass finds nothing new to act on."""
        for _pass in range(8):
            found = False
            for widget in tuple(_walk(self.results.window)):
                if not _alive(widget) or str(widget) in self.done or not widget.winfo_viewable():
                    continue
                self.done.add(str(widget))
                found = True
                self.one(widget)
            if not found:
                return
        pytest.fail("the Results window kept producing new controls after eight passes")

    def wheel(self):
        """One wheel turn over the content and one over the inspector, as a mouse would send them."""
        window = self.results.window
        for scroll in (self.results.content_scroll, self.results.inspector_scroll):
            area = scroll.outer
            x, y = max(1, area.winfo_width() // 2), max(1, area.winfo_height() // 2)
            for delta in (-120, 120):
                self.act("wheel_event", window, lambda x=x, y=y, delta=delta, area=area: window.event_generate(
                    "<MouseWheel>", x=x, y=y, rootx=area.winfo_rootx() + x, rooty=area.winfo_rooty() + y, delta=delta))


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("view", ["delay", "atlas"])
def test_results_click_everything(app, tally, theme, view, capsys):
    app._set_appearance(theme)
    errors = []
    app.root.report_callback_exception = lambda exc, value, tb: errors.append((exc, value, tb))
    results = app._show_results(view)
    _settle(app.root)
    if not results.window.winfo_viewable():
        pytest.skip("the Results window never became viewable on this display, so clicks and keys cannot reach it")
    driver = _Driver(results, app.root, tally, view)
    driver.sweep()
    driver.wheel()
    with capsys.disabled():
        print(f"\n[{theme}/{view}] " + ", ".join(f"{name}={count}" for name, count in sorted(tally.items())))
    assert not errors, "Tk callback exceptions: " + "; ".join(f"{item[0].__name__}: {item[1]}" for item in errors)

    # Every kind of control was really used, and the handlers behind them really ran.
    assert tally["invoke"] >= 1, "no button or radio button was invoked"
    assert tally["tree"] >= 2 and tally["view_switched"] >= 2, "the sidebar tree did not switch views"
    assert tally["canvas_event"] >= 1 and tally["click_selects"] >= 1, "no canvas click changed the selection"
    assert tally["key_event"] >= 1 and tally["key_moves"] >= 1, "no key press moved the selection"
    assert tally["wheel"] >= 4 and tally["scrolled"] >= 1, "the mouse wheel never reached a scroll area"
    if view == "delay":
        assert tally["chart_click"] >= 1 and tally["chart_key"] >= 1, "the delay chart handlers never ran"
        owner = results.views["delay"].chart
    else:
        assert tally["grid_click"] >= 1 and tally["grid_key"] >= 1, "the atlas grid handlers never ran"
        assert tally["menu"] >= 2 and tally["menu_effect"] >= 2, "the Conditions menu entries were not all picked"
        assert tally["pick_loss"] >= 1, "no loss-level row in the atlas inspector was clicked"
        owner = results.views["atlas"].grid
    if any(widget.winfo_class() == "TCombobox" for widget in _walk(results.window)):
        assert tally["combo"] >= 1, "a combobox exists but none of its values was set"

    # A selection fills the inspector; Escape clears it; a second Escape on the empty chart closes the window.
    page = results.views[view].page
    if owner.selected is None:
        _focus(app.root, owner.canvas)
        owner.canvas.event_generate("<Down>")
        _settle(app.root)
    assert owner.selected is not None and page.filled, "a selection did not fill the inspector"
    assert _labels_with_text(results.inspector_body), "inspector labels are empty after selection"
    _focus(app.root, owner.canvas)
    owner.canvas.event_generate("<Escape>")
    _settle(app.root)
    assert owner.selected is None and not page.filled, "Escape did not clear the selection"
    assert results.exists(), "the first Escape closed the window instead of clearing the selection"
    _focus(app.root, owner.canvas)
    owner.canvas.event_generate("<Escape>")
    _settle(app.root)
    assert not results.exists(), "Escape on an empty chart did not close the window"
    assert not errors, "Tk callback exceptions: " + "; ".join(f"{item[0].__name__}: {item[1]}" for item in errors)


def test_results_open_close_three_times_with_theme_switch(app):
    errors = []
    app.root.report_callback_exception = lambda exc, value, tb: errors.append((exc, value, tb))
    for index, theme in enumerate(("dark", "light", "dark")):
        app._set_appearance(theme)
        results = app._show_results("delay" if index % 2 == 0 else "atlas")
        _settle(app.root)
        assert results.exists()
        results.destroy()
        _settle(app.root)
    assert not errors, "Tk callback exceptions while reopening: " + "; ".join(str(item[1]) for item in errors)
