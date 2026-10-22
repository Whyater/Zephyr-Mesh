"""Windows Tkinter replay window for the Zephyr Mesh S7 fixture.

The app is offline and read-only. It replays the canonical Python event log;
it is not a vehicle controller or a live telemetry client. Tkinter is
imported only when the window is built, so the pure helpers stay testable on
machines without Tk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from desktop.design_tokens import LAYOUT, SHORTCUTS, SPACE, TRANSPORT, UI_TEXT, WIN_LAYOUT
    from desktop.presentation import compact_name, filter_group_rows, pinned_levels, run_name, team_filter
    from desktop.replay import ReplayModel
    from desktop.replay_session import SPEEDS, ReplaySession
    from desktop.scenario_run import load_scenario_run
    from desktop.ui_icons import set_window_icon
    from desktop import ui_format as fmt
    from desktop.ui_settings import load_appearance, resolve_theme, save_appearance, system_prefers_dark, system_reduced_motion
else:
    from .design_tokens import LAYOUT, SHORTCUTS, SPACE, TRANSPORT, UI_TEXT, WIN_LAYOUT
    from .presentation import compact_name, filter_group_rows, pinned_levels, run_name, team_filter
    from .replay import ReplayModel
    from .replay_session import SPEEDS, ReplaySession
    from .scenario_run import load_scenario_run
    from .ui_icons import set_window_icon
    from . import ui_format as fmt
    from .ui_settings import load_appearance, resolve_theme, save_appearance, system_prefers_dark, system_reduced_motion


RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
RESOURCE_ROOTS = (RESOURCE_ROOT / "_internal", RESOURCE_ROOT)


def _resource_path(*parts: str) -> Path:
    for root in RESOURCE_ROOTS:
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return candidate
    return RESOURCE_ROOT.joinpath(*parts)


DEFAULT_RUN = _resource_path("runs", "s7-swarm", "run.json")


def _runtime_version() -> str:
    """Read the version embedded by the Windows release build.

    ``build_windows.ps1`` writes ``VERSION.txt`` beside the exe. In PyInstaller 6
    ``_MEIPASS`` is the ``_internal`` folder below it, so the exe folder is checked
    first. A source checkout uses the repository ``VERSION`` plus ``-dev``, the
    same rule as ``tools/macos/build_app.py``.
    """
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "VERSION.txt")
    candidates.append(_resource_path("VERSION.txt"))
    for version_file in candidates:
        try:
            version = version_file.read_text(encoding="utf-8").strip().removeprefix("v")
        except OSError:
            continue
        if version:
            return version
    try:
        checkout = (RESOURCE_ROOT / "VERSION").read_text(encoding="utf-8").strip().removeprefix("v")
    except OSError:
        checkout = ""
    return f"{checkout}-dev" if checkout else "0.2.4-dev"


CURRENT_VERSION = _runtime_version()


def startup_log_path() -> Path:
    """Return the per-user path used for failures before a Tk window exists.

    The release executable is built with PyInstaller's ``--windowed`` bootloader,
    so an exception during imports or Tk construction otherwise disappears with
    no console. Keep the diagnostic outside the extracted application folder
    because that folder may be read-only or replaced by the updater.
    """

    app_data = os.environ.get("LOCALAPPDATA")
    base = Path(app_data) / "ZephyrMesh" if app_data else Path.home() / ".zephyr-mesh"
    try:
        base.mkdir(parents=True, exist_ok=True)
        return base / "startup.log"
    except OSError:
        return Path(tempfile.gettempdir()) / "ZephyrMesh-startup.log"


def write_startup_diagnostic(exc: BaseException) -> Path:
    """Persist enough context to diagnose a windowed startup failure."""

    path = startup_log_path()
    details = (
        "Zephyr Mesh Windows startup failure\n"
        f"version: {CURRENT_VERSION}\n"
        f"executable: {Path(sys.executable).resolve()}\n"
        f"working directory: {Path.cwd()}\n"
        f"resource root: {RESOURCE_ROOT}\n"
        f"default replay: {DEFAULT_RUN}\n"
        f"default replay exists: {DEFAULT_RUN.is_file()}\n"
        f"platform: {sys.platform}\n"
        f"python: {sys.version}\n\n"
        f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
    )
    try:
        path.write_text(details, encoding="utf-8")
    except OSError:
        # The caller still reports the original exception when the diagnostic
        # directory is unavailable. Never replace a useful startup error with
        # a logging failure.
        pass
    return path


def show_startup_failure(exc: BaseException, log_path: Path) -> None:
    """Make a windowed startup failure visible instead of silently exiting."""

    message = (
        "Zephyr Mesh could not start.\n\n"
        f"{type(exc).__name__}: {exc}\n\n"
        f"A diagnostic was written to:\n{log_path}\n\n"
        "If this came from a ZIP download, extract the entire ZIP before "
        "opening ZephyrMeshWindows.exe and keep its _internal folder beside it."
    )
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, "Zephyr Mesh", 0x10)
            return
        except Exception:
            # A frozen windowed process may have no usable GUI or standard
            # streams. Keep this reporting path best effort.
            pass
    try:
        stream = getattr(sys, "stderr", None)
        if stream is not None:
            print(message, file=stream)
    except Exception:
        pass


def show_plain_startup_error(message: str) -> None:
    """Report a bad input file before any window exists: stderr, plus a message box on Windows (no console there)."""
    try:
        print(message, file=sys.stderr)
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, "Zephyr Mesh", 0x30)
        except Exception:
            pass


def _take_previous_update_failure() -> str | None:
    try:
        try:
            from tools.release_updater import take_update_result
        except ModuleNotFoundError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from tools.release_updater import take_update_result
        return take_update_result()
    except Exception:
        return None


def _show_update_failure(root: Any, message: str) -> None:
    from tkinter import messagebox

    messagebox.showwarning("Zephyr Mesh update", message, parent=root)


TEXT_INPUT_CLASSES = frozenset({"Entry", "TEntry", "Text", "Spinbox", "TSpinbox", "TCombobox"})
BUTTON_CLASSES = frozenset({"Button", "TButton", "TRadiobutton", "TCheckbutton", "Radiobutton", "Checkbutton", "TMenubutton"})
# Keys the team list handles itself (Tk's Treeview class bindings): Left collapses, Right
# expands, Space toggles the focused group. Home has no Treeview binding, so it stays Restart.
TREE_KEYS = frozenset({"left", "right", "space"})


def is_text_input_class(widget_class: str) -> bool:
    """True for Tk widgets where the user types text (so bare-key shortcuts must stand down)."""
    return widget_class in TEXT_INPUT_CLASSES


def shortcut_blocked(widget_class: str, key: str, *, tree_owns_keys: bool = True) -> bool:
    """Whether a single-key shortcut must yield to the focused widget's own key handling.

    Text fields own every key. A focused button owns Space (it presses the
    button). The team list owns Left, Right, and Space (collapse, expand,
    toggle), which keeps list navigation native. A flat list with no groups,
    like the Events list, passes ``tree_owns_keys=False`` so Left, Right,
    Home, and Space keep stepping and playing the replay.
    """
    if is_text_input_class(widget_class):
        return True
    if key == "space" and widget_class in BUTTON_CLASSES:
        return True
    return tree_owns_keys and widget_class == "Treeview" and key in TREE_KEYS


def updater_launch_path(updater: Path, install_dir: Path, *, pid: int) -> Path:
    """Return a runnable helper path that is outside the directory being swapped.

    Release ZIPs keep one root application directory, so the detached helper
    is embedded there for a simple download and extraction. Windows cannot
    replace a directory containing the helper process itself. Copying that
    helper to the temporary directory before launch keeps the swap atomic;
    local development builds can continue to keep the helper as a sibling.
    """
    updater = Path(updater).resolve()
    install_dir = Path(install_dir).resolve()
    if updater.parent != install_dir:
        return updater
    temporary = Path(tempfile.gettempdir()) / f"ZephyrMeshUpdater-{pid}.exe"
    shutil.copy2(updater, temporary)
    return temporary


# ---- menus and shortcuts ----------------------------------------------------
# Every generated shortcut maps to exactly one app method. The hygiene test
# asserts this table and the menu accelerators match UI_TEXT.
SHORTCUT_HANDLERS: dict[str, str] = {
    "playPause": "_toggle",
    "stepForward": "_step",
    "stepBack": "_step_back",
    "restart": "_restart",
    "focusSelected": "_focus_selected_drone",
    "resetCamera": "_reset_camera",
    "toggleTopView": "_toggle_view_mode",
    "toggleKeepOut": "_toggle_keepout",
    "toggleInspector": "_toggle_inspector",
    "openEvidenceReport": "_open_evidence_report",
    "openScenarioRun": "_open_scenario_run",
    "focusFilter": "_focus_filter",
    "aboutReplay": "_show_about",
    "showResults": "_show_results",
}

# (menu, label, action id or None, kind). Kinds: command, check, radio, separator, cascade.
MENU_LAYOUT: tuple[tuple[str, tuple[tuple[str, str | None, str], ...]], ...] = (
    ("File", (
        (UI_TEXT["menus"]["openEvidenceReport"], "openEvidenceReport", "command"),
        (UI_TEXT["menus"]["openScenarioRun"], "openScenarioRun", "command"),
        ("", None, "separator"),
        ("Exit", None, "exit"),
    )),
    ("View", (
        ("3D View", "toggleTopView", "view3d"),
        ("Top View", "toggleTopView", "viewtop"),
        ("", None, "separator"),
        (UI_TEXT["menus"]["showKeepOut"], "toggleKeepOut", "check-keepout"),
        (UI_TEXT["menus"]["showInspector"], "toggleInspector", "check-inspector"),
        ("", None, "separator"),
        (UI_TEXT["menus"]["focusSelected"], "focusSelected", "command"),
        (UI_TEXT["menus"]["resetCamera"], "resetCamera", "command"),
        ("", None, "separator"),
        (UI_TEXT["menus"]["showResults"], "showResults", "command"),
        ("", None, "separator"),
        ("Appearance", None, "appearance"),
    )),
    (UI_TEXT["menus"]["playback"], (
        ("Play/Pause", "playPause", "command"),
        (UI_TEXT["menus"]["stepForward"], "stepForward", "command"),
        (UI_TEXT["menus"]["stepBack"], "stepBack", "command"),
        (UI_TEXT["shortcuts"]["restart"]["label"], "restart", "command"),
        ("", None, "separator"),
        (UI_TEXT["menus"]["speed"], None, "speed"),
    )),
    ("Help", (
        (UI_TEXT["menus"]["aboutReplay"], "aboutReplay", "command"),
        ("Check for Updates", None, "updates"),
    )),
)


def menu_accelerators() -> dict[str, str]:
    """Accelerator text shown in the menus, keyed by shortcut action id."""
    return {action: fmt.accelerator(action) for _menu, items in MENU_LAYOUT for _label, action, _kind in items if action}


def speed_label(speed: float) -> str:
    return f"{speed:g}x"


# ---- pure team-list planning (tested without Tk) ------------------------------
WIN_TEXT: Mapping[str, str] = UI_TEXT["windowsText"]
TEAM_COLUMNS = ("loss", "age", "end")  # "end" is an empty gap column, not data


def group_key(row: Mapping[str, Any]) -> str:
    """Stable Treeview id for a group: one row per status level and word."""
    return f"group:{row['level']}:{row['word']}"


def _cell(value: Any, kind: str) -> str:
    """A narrow table cell: the formatted value, or an en dash (the list explains it as "Not recorded")."""
    return fmt.format_value(value, kind) or fmt.EN_DASH


def _loss_cell(value: Any) -> str:
    """Loss with one decimal, like everywhere else; the "%" lives in the column heading."""
    return _cell(value, "pct1_cell")


def team_rows(
    groups: Sequence[Mapping[str, Any]],
    per_drone: Sequence[Mapping[str, Any]],
    query: str = "",
) -> list[dict[str, Any]]:
    """Plan the team list: one row per group, children per drone, filter applied.

    The presentation filter (``filter_group_rows``) never removes a late or
    lost group. Any other group keeps only its matching drones while a filter
    is active, and the label counts them against the whole group. Every row,
    group or drone, carries its status shape (``level``) and word, so an
    expanded drone row reads on its own.
    """
    records = {str(record["drone_id"]): record for record in per_drone}
    rows = []
    for group in filter_group_rows(groups, query):
        members = [str(drone) for drone in group.get("drone_ids", [])]
        total = int(group.get("group_count", group.get("count", len(members))))
        word = fmt.status_word(str(group["level"]), group.get("word"))
        count = fmt.group_label(total)
        label = (WIN_TEXT["groupLabel"].format(word=word, count=count) if len(members) == total
                 else WIN_TEXT["groupLabelFiltered"].format(word=word, shown=len(members), total=total))
        children = []
        for drone in members:
            record = records.get(drone, {})
            level = str(record.get("level", group["level"]))
            children.append({
                "iid": drone,
                "level": level,
                "label": WIN_TEXT["droneLabel"].format(name=record.get("compact_name", drone),
                                                       word=fmt.status_word(level, record.get("word"))),
                "values": (_loss_cell(record.get("target_packet_loss_pct")), _cell(record.get("target_age_s"), "s_cell")),
            })
        rows.append({
            "iid": group_key(group),
            "level": str(group["level"]),
            "label": label,
            "members": members,
            "values": (_loss_cell(group.get("target_packet_loss_pct")), _cell(group.get("target_age_s"), "s_cell")),
            "children": children,
        })
    return rows


def row_region(tree: Any, x: int, y: int) -> bool:
    """True when (x, y) is over a Treeview row ("tree" or "cell"), never the headings or blank space."""
    try:
        return tree.identify_region(x, y) in ("tree", "cell")
    except Exception:
        return False


def rows_have_missing(rows: Sequence[Mapping[str, Any]], open_groups: set[str]) -> bool:
    """True when a visible row shows the en dash, so the list must say what the dash means."""
    for row in rows:
        if fmt.EN_DASH in row["values"]:
            return True
        if row["iid"] in open_groups and any(fmt.EN_DASH in child["values"] for child in row["children"]):
            return True
    return False


def open_error_text(path: Path, exc: BaseException | None, noun: str) -> str:
    """One plain sentence for a file that could not be opened, the same for every viewer.

    ``exc`` is None when the path failed the pre-check (missing or not a file).
    Format errors keep the decoder's reason, written as a sentence.
    """
    name = Path(path).name or str(path)
    cause = exc.__cause__ if exc is not None and exc.__cause__ is not None else exc
    if exc is None or isinstance(cause, FileNotFoundError):
        if Path(path).exists():
            return WIN_TEXT["openNotFile"].format(name=name)
        return WIN_TEXT["openMissing"].format(name=name)
    if isinstance(cause, IsADirectoryError):
        return WIN_TEXT["openNotFile"].format(name=name)
    if isinstance(cause, PermissionError):
        return WIN_TEXT["openPermission"].format(name=name)
    if isinstance(cause, UnicodeDecodeError):
        return WIN_TEXT["openNotUtf8"].format(name=name, noun=noun)
    if isinstance(cause, json.JSONDecodeError):
        return WIN_TEXT["openNotJson"].format(name=name, line=cause.lineno, column=cause.colno)
    if isinstance(cause, OSError):
        reason = cause.strerror or type(cause).__name__
        return WIN_TEXT["openUnreadable"].format(name=name, reason=_sentence(reason))
    return WIN_TEXT["openInvalid"].format(name=name, noun=noun, reason=_sentence(str(exc)))


def _sentence(text: str) -> str:
    text = " ".join(str(text).split()) or type(text).__name__
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def desired_team_selection(rows: Sequence[Mapping[str, Any]], selected: str, open_groups: set[str]) -> str | None:
    """The row to select: the drone when its group is open. A closed group is never selected for it."""
    for row in rows:
        if selected in row["members"]:
            return selected if row["iid"] in open_groups else None
    return None


def group_holding_selection(rows: Sequence[Mapping[str, Any]], selected: str, open_groups: set[str]) -> str | None:
    """The closed group that hides the selected drone; it gets a quiet "holds" mark instead of the selection."""
    for row in rows:
        if selected in row["members"] and row["iid"] not in open_groups:
            return str(row["iid"])
    return None


def drone_for_team_row(rows: Sequence[Mapping[str, Any]], iid: str, selected: str) -> str | None:
    """Map a selected Treeview row back to a drone id.

    A drone row is that drone. A group row never picks a drone by itself:
    it keeps the current drone when the group holds it, else ``None``, so
    landing on a header (arrow keys or a click) never moves the selection.
    """
    for row in rows:
        if iid == row["iid"]:
            return selected if selected in row["members"] else None
        if iid in row["members"]:
            return iid
    return None


class WindowsReplayApp:
    """Replay window: status strip, team list, 3D scene, inspector, timeline, events."""

    def __init__(self, root: Any, model: ReplayModel, *, appearance: str | None = None, persist_appearance: bool = True) -> None:
        import tkinter as tk
        from tkinter import ttk

        if __package__ in (None, ""):
            from desktop.scene_view import SceneView
            from desktop.ui_theme import Fonts, ImageBank, Theme, apply_title_bar, configure_styles, ui_scale
            from desktop import ui_widgets as widgets
        else:
            from .scene_view import SceneView
            from .ui_theme import Fonts, ImageBank, Theme, apply_title_bar, configure_styles, ui_scale
            from . import ui_widgets as widgets

        self.tk = tk
        self.ttk = ttk
        self.widgets = widgets
        self._Theme = Theme
        self._configure_styles = configure_styles
        self._apply_title_bar = apply_title_bar
        self.root = root
        self.model = model
        self.session = ReplaySession(model, schedule=self._schedule, cancel=self._cancel)
        self.run: Mapping[str, Any] = self.session.run
        self.render_count = 0
        self._persist_appearance = persist_appearance
        self.appearance = appearance or load_appearance()
        self.scale = ui_scale(root)
        self.fonts = Fonts(root, self.scale)
        self.images = ImageBank(root, self.scale)
        self.theme = Theme(resolve_theme(self.appearance, system_prefers_dark()))
        self.style = ttk.Style(root)
        self._theme_hooks: list[Callable[[Any], None]] = []
        self._team_plan: list[dict[str, Any]] = []
        self._open_groups: set[str] = set()
        self._seen_groups: set[str] = set()
        self._event_frames: dict[str, int] = {}
        self._event_count = -1
        self._event_signature: tuple[Any, ...] = ()
        self._evidence_report: dict[str, Any] | None = None
        self._evidence_path: Path | None = None
        self._evidence_window: Any | None = None
        self._scenario_window: Any | None = None
        self._about_window: Any | None = None
        self._results: Any | None = None  # the single Results window (ui_results.ResultsWindow)
        self.reduced_motion = system_reduced_motion()
        self._configure_styles(self.style, self.theme, self.fonts, self.scale)
        self.images.apply(self.theme)
        self._build(SceneView)
        self._apply_theme()
        self._render()

    # ---- compatibility -----------------------------------------------------
    @property
    def selected_id(self) -> str:
        return self.session.selected_drone_id

    @property
    def playing(self) -> bool:
        return self.session.playing

    def px(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    def _set(self, widget: Any, **options: Any) -> None:
        """Configure only what changed: Tk relayout and redraw are the expensive part of a frame."""
        cache = self.__dict__.setdefault("_configured", {})
        key = str(widget)
        previous = cache.setdefault(key, {})
        changed = {name: value for name, value in options.items() if previous.get(name, object()) != value}
        if changed:
            widget.configure(**changed)
            previous.update(changed)

    # ---- construction ------------------------------------------------------
    def _hairline(self, parent: Any, *, vertical: bool = False) -> Any:
        size = self.px(WIN_LAYOUT["hairline"])
        return self.ttk.Frame(parent, style="Line.TFrame", **({"width": size} if vertical else {"height": size}))

    def _baseline_pad(self, reference: Any, font: Any, *, inset: int = 0, height: int | None = None) -> tuple[int, int]:
        """pack ``pady`` that puts a smaller label's text baseline on ``reference``'s baseline.

        ``pack`` centers both widgets in the row, so the smaller one moves down by
        half of its top padding. ``inset`` is the space above the text inside the
        widget (padding and border) and ``height`` its full height.
        """
        own = height if height is not None else font.metrics("linespace")
        reference_baseline = reference.metrics("ascent") - reference.metrics("linespace") / 2.0
        own_baseline = inset + font.metrics("ascent") - own / 2.0
        shift = max(0, int(round(2 * (reference_baseline - own_baseline))))
        return shift, 0

    def _build(self, scene_class: Any) -> None:
        tk, ttk, root = self.tk, self.ttk, self.root
        root.title("Zephyr Mesh")
        self._team_layout = self._measure_team_list()
        # The token minimum, or wider if this machine's fonts need a wider team list to show every label whole.
        hairlines = 2 * self.px(WIN_LAYOUT["hairline"])
        needed = (self._team_layout["sidebar"] + self.px(LAYOUT["inspectorWidth"]) + self.px(LAYOUT["sceneMinWidth"])
                  + hairlines)
        root.minsize(max(self.px(LAYOUT["windowMinWidth"]), needed), self.px(LAYOUT["windowMinHeight"]))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)
        self.keepout_var = tk.BooleanVar(value=self.session.show_keepout)
        self.inspector_var = tk.BooleanVar(value=self.session.show_inspector)
        self.view_var = tk.StringVar(value=self.session.view_mode)
        self.speed_var = tk.DoubleVar(value=self.session.speed)
        self.appearance_var = tk.StringVar(value=self.appearance)
        self.filter_var = tk.StringVar()
        self._build_menu()
        self._build_title(root).grid(row=0, column=0, sticky="ew")
        self._hairline(root).grid(row=1, column=0, sticky="ew")
        self._build_strip(root).grid(row=2, column=0, sticky="ew")
        self._hairline(root).grid(row=3, column=0, sticky="ew")
        self._build_middle(root, scene_class).grid(row=4, column=0, sticky="nsew")
        self._hairline(root).grid(row=5, column=0, sticky="ew")
        self._build_timeline(root).grid(row=6, column=0, sticky="ew")
        self._hairline(root).grid(row=7, column=0, sticky="ew")
        self._build_events(root).grid(row=8, column=0, sticky="ew")
        self._bind_shortcuts()
        root.bind_all("<MouseWheel>", self._route_wheel, add="+")
        try:
            root.bind_all("<TouchpadScroll>", self._route_touchpad, add="+")
        except tk.TclError:
            pass

    def _build_menu(self) -> None:
        tk = self.tk
        menubar = tk.Menu(self.root)
        accelerators = menu_accelerators()
        self._menus: dict[str, Any] = {}
        for title, items in MENU_LAYOUT:
            menu = tk.Menu(menubar, tearoff=False)
            for label, action, kind in items:
                accel = accelerators.get(action, "") if action else ""
                if kind == "separator":
                    menu.add_separator()
                elif kind == "exit":
                    menu.add_command(label=label, command=self.root.destroy)
                elif kind == "view3d":
                    menu.add_radiobutton(label=label, value="3d", variable=self.view_var, accelerator=accel,
                                         command=lambda: self._set_view_mode("3d"))
                elif kind == "viewtop":
                    menu.add_radiobutton(label=label, value="top", variable=self.view_var, accelerator=accel,
                                         command=lambda: self._set_view_mode("top"))
                elif kind == "check-keepout":
                    menu.add_checkbutton(label=label, variable=self.keepout_var, accelerator=accel,
                                         command=lambda: self._set_keepout(self.keepout_var.get()))
                elif kind == "check-inspector":
                    menu.add_checkbutton(label=label, variable=self.inspector_var, accelerator=accel,
                                         command=lambda: self._set_inspector(self.inspector_var.get()))
                elif kind == "appearance":
                    sub = tk.Menu(menu, tearoff=False)
                    for value, text in (("system", "System"), ("dark", "Dark"), ("light", "Light")):
                        sub.add_radiobutton(label=text, value=value, variable=self.appearance_var,
                                            command=lambda value=value: self._set_appearance(value))
                    menu.add_cascade(label=label, menu=sub)
                elif kind == "speed":
                    sub = tk.Menu(menu, tearoff=False)
                    for speed in SPEEDS:
                        sub.add_radiobutton(label=speed_label(speed), value=speed, variable=self.speed_var,
                                            command=lambda speed=speed: self._set_speed(speed))
                    menu.add_cascade(label=label, menu=sub)
                elif kind == "updates":
                    menu.add_command(label=label, command=self._check_update)
                else:
                    handler = getattr(self, SHORTCUT_HANDLERS[action])
                    menu.add_command(label=label, accelerator=accel, command=handler)
            menubar.add_cascade(label=title, menu=menu)
            self._menus[title] = menu
        self.root.configure(menu=menubar)

    def _segments(self, parent: Any, choices: Sequence[tuple[Any, str]], variable: Any, command: Callable[[], None]) -> Any:
        """A segmented choice whose segments all have the same width, with SPACE["xxs"] between them."""
        frame = self.ttk.Frame(parent, style="Surface.TFrame")
        width = max(len(text) for _value, text in choices)
        for index, (value, text) in enumerate(choices):
            self.ttk.Radiobutton(frame, text=text, value=value, variable=variable, style="Segment.Toolbutton", width=width,
                                 command=command, takefocus=True).pack(
                side="left", padx=(0 if index == 0 else self.px(SPACE["xxs"]), 0))
        return frame

    def _build_title(self, parent: Any) -> Any:
        """One toolbar row, like the Mac window: name, run, badge on the left; view controls on the right."""
        ttk = self.ttk
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=(self.px(SPACE["lg"]), self.px(SPACE["sm"])))
        title_font = self.fonts["title"]
        ttk.Label(frame, text="Zephyr Mesh", style="Surface.Title.TLabel").pack(side="left")
        ttk.Label(frame, text=run_name(self.run), style="Surface.Secondary.TLabel").pack(
            side="left", padx=(self.px(SPACE["md"]), 0), pady=self._baseline_pad(title_font, self.fonts["body"]))
        badge_inset = self.px(WIN_LAYOUT["hairline"]) + self.px(SPACE["xxs"])
        badge_height = self.fonts["caption"].metrics("linespace") + 2 * badge_inset
        ttk.Label(frame, text=UI_TEXT["honesty"]["badge"], style="Badge.TLabel").pack(
            side="left", padx=(self.px(SPACE["md"]), 0),
            pady=self._baseline_pad(title_font, self.fonts["caption"], inset=badge_inset, height=badge_height))
        gap = self.px(SPACE["sm"])
        about = ttk.Button(frame, text=f" {UI_TEXT['honesty']['aboutButton']}", style="Tool.TButton", compound="left",
                           image=self.images.icon("info", "textSecondary", WIN_LAYOUT["iconSize"]), command=self._show_about)
        about.pack(side="right")
        self.view_modes = self._segments(frame, (("3d", "3D"), ("top", "Top")), self.view_var,
                                         lambda: self._set_view_mode(self.view_var.get()))
        self.view_modes.pack(side="right", padx=(0, gap))
        self.notice = ttk.Label(frame, text="", style="Surface.Caption.TLabel")
        self.notice.pack(side="right", padx=(0, self.px(SPACE["lg"])))
        return frame

    def _build_strip(self, parent: Any) -> Any:
        ttk = self.ttk
        # Cells carry SPACE["lg"] on the left, so the first value lines up with the title and the sidebar.
        frame = ttk.Frame(parent, style="Base.TFrame", padding=(0, self.px(SPACE["md"])))
        labels = UI_TEXT["statusStrip"]
        self.strip_values: dict[str, Any] = {}
        column = 0
        for index, label in enumerate(labels):
            if index:
                self._hairline(frame, vertical=True).grid(row=0, column=column, sticky="ns", pady=self.px(SPACE["xxs"]))
                column += 1
            cell = ttk.Frame(frame, style="Base.TFrame", padding=(self.px(SPACE["lg"]), 0, self.px(SPACE["md"]), 0))
            cell.grid(row=0, column=column, sticky="nsew")
            # The headline cell gets a little more room than each number, not twice as much (7:5).
            frame.columnconfigure(column, weight=7 if index == 0 else 5, uniform="strip")
            ttk.Label(cell, text=label, style="Base.Caption.TLabel").pack(anchor="w")
            row = ttk.Frame(cell, style="Base.TFrame", height=self.fonts["metric"].metrics("linespace"))
            row.pack(anchor="w", fill="x", pady=(self.px(SPACE["xxs"]), 0))
            row.pack_propagate(False)
            if index == 0:
                glyph = ttk.Label(row, style="Base.Glyph.TLabel")
                glyph.pack(side="left", padx=(0, self.px(SPACE["sm"])))
                self.strip_glyph = glyph
            value = ttk.Label(row, style="Base.Metric.TLabel", anchor="w")
            value.pack(side="left", fill="both", expand=True)
            self.strip_values[label] = value
            column += 1
        # A fixed-height strip keeps value changes from re-laying out the whole window.
        caption, metric = self.fonts["caption"], self.fonts["metric"]
        height = caption.metrics("linespace") + metric.metrics("linespace") + self.px(SPACE["xxs"]) + 2 * self.px(SPACE["md"])
        frame.configure(height=height)
        frame.grid_propagate(False)
        return frame

    def _build_middle(self, parent: Any, scene_class: Any) -> Any:
        ttk = self.ttk
        frame = ttk.Frame(parent, style="Base.TFrame")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(2, weight=1, minsize=self.px(LAYOUT["sceneMinWidth"]))
        self._build_team(frame).grid(row=0, column=0, sticky="nsw")
        self._hairline(frame, vertical=True).grid(row=0, column=1, sticky="ns")
        self._build_scene(frame, scene_class).grid(row=0, column=2, sticky="nsew")
        self.inspector_line = self._hairline(frame, vertical=True)
        self.inspector_line.grid(row=0, column=3, sticky="ns")
        self.inspector_panel = self._build_inspector(frame)
        self.inspector_panel.grid(row=0, column=4, sticky="nse")
        return frame

    def _header_row(self, parent: Any, row: int) -> None:
        """Panel headers share one fixed-height first row, so the sidebar and inspector titles line up."""
        parent.rowconfigure(row, minsize=self.px(LAYOUT["controlHeight"]))

    def _measure_team_list(self) -> dict[str, Any]:
        """Column widths from the widest real labels in the bold group font, and the sidebar width they need.

        The sidebar is the token width unless this run's longest group or drone
        label needs more; then it grows, so a label is never cut off.
        """
        strong, caption = self.fonts["bodyStrong"], self.fonts["caption"]
        pad = self.px(SPACE["sm"])

        def fit(*samples: str) -> int:
            return max(strong.measure(sample) for sample in samples) + pad

        dash = fmt.EN_DASH
        widths = {
            "loss": max(fit(f"00.0{dash}100.0"), caption.measure(WIN_TEXT["columnLoss"]) + pad),
            "age": max(fit(f"0.00{dash}0.00"), caption.measure(WIN_TEXT["columnAge"]) + pad),
            "end": self.px(SPACE["sm"]),  # keeps right-aligned numbers off the edge of the selection highlight
        }
        agents = self.run.get("steps", [{}])[0].get("agents", []) if self.run.get("steps") else []
        count = max(len(agents), 1)
        # The words a drone row can carry (status_for_agent): OK, Late, Lost or its alternate, Waiting.
        words = [str(word) for level in ("ok", "caution", "alert", "waiting")
                 for word in (fmt.LEVELS[level]["word"], fmt.LEVELS[level].get("alternateWord")) if word]
        names = [compact_name(str(agent.get("agent_id", ""))) for agent in agents] or ["D00"]
        group_text = max(strong.measure(WIN_TEXT["groupLabel"].format(word=word, count=fmt.group_label(count))) for word in words)
        drone_text = max(strong.measure(WIN_TEXT["droneLabel"].format(name=name, word=word)) for name in names[:200] for word in words)
        # Row geometry as ttk draws it (checked with identify_element): the disclosure slot (indicatorsize
        # plus a SPACE["xs"] margin), the glyph box, then the text; drone rows add one indent first.
        indicator = self.px(SPACE["sm"]) + self.px(SPACE["xs"])
        glyph = self.px(WIN_LAYOUT["glyphBox"])
        indent = self.px(SPACE["lg"])
        tree_column = max(indicator + glyph + group_text, indent + indicator + glyph + drone_text) + pad
        scroll = self.px(SPACE["sm"]) + self.px(SPACE["xxs"])
        sidebar = max(self.px(WIN_LAYOUT["sidebarWidth"]), tree_column + sum(widths.values()) + scroll + 2 * self.px(SPACE["lg"]))
        widths["#0"] = sidebar - 2 * self.px(SPACE["lg"]) - scroll - sum(widths.values())
        return {"widths": widths, "sidebar": sidebar, "scroll": scroll}

    def _build_team(self, parent: Any) -> Any:
        ttk = self.ttk
        width = self._team_layout["sidebar"]
        pad = self.px(SPACE["lg"])
        frame = ttk.Frame(parent, style="Surface.TFrame", width=width, padding=(pad, self.px(SPACE["md"]), pad, pad))
        frame.grid_propagate(False)
        frame.pack_propagate(False)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)
        self._header_row(frame, 0)
        self.team_summary = ttk.Label(frame, style="Surface.Strong.TLabel", wraplength=width - 2 * pad)
        self.team_summary.grid(row=0, column=0, columnspan=2, sticky="w")
        field = ttk.Frame(frame, style="Surface.TFrame")
        field.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(self.px(SPACE["sm"]), 0))
        field.columnconfigure(0, weight=1)
        self.filter_entry = ttk.Entry(field, textvariable=self.filter_var, style="Filter.TEntry", font=self.fonts["body"])
        self.filter_entry.grid(row=0, column=0, sticky="ew")
        glyph = self.px(WIN_LAYOUT["iconSize"])
        text_inset = self.px(SPACE["sm"]) + self.px(WIN_LAYOUT["hairline"])
        search = ttk.Label(field, image=self.images.glyph("search", "textSecondary", WIN_LAYOUT["iconSize"]),
                           style="Field.Placeholder.TLabel")
        search.place(in_=self.filter_entry, x=text_inset, rely=0.5, anchor="w")
        search.bind("<Button-1>", lambda _event: self.filter_entry.focus_set())
        self.widgets.Placeholder(self.filter_entry, self.filter_var, f"Filter drones ({fmt.accelerator('focusFilter')})",
                                 "Field.Placeholder.TLabel", inset=text_inset + glyph + self.px(SPACE["xs"]))
        self.clear_button = ttk.Button(field, style="FieldIcon.TButton", takefocus=False, command=self._clear_filter,
                                       image=self.images.glyph("clear", "textSecondary", WIN_LAYOUT["iconSize"]))
        self.widgets.Tooltip(self.clear_button, f"{UI_TEXT['teamFilter']['clearFilter']} (Esc)", lambda: self.theme,
                             self.fonts["caption"], scale=self.scale)
        self.filter_var.trace_add("write", lambda *_args: self._on_filter())
        self.filter_entry.bind("<Escape>", self._clear_filter)
        self.filter_caption = ttk.Label(frame, style="Surface.Caption.TLabel", wraplength=width - 2 * pad, justify="left")
        self.filter_caption.grid(row=2, column=0, columnspan=2, sticky="w", pady=(self.px(SPACE["xs"]), 0))
        self.filter_caption.grid_remove()
        columns = TEAM_COLUMNS
        tree = ttk.Treeview(frame, columns=columns, show="tree headings", selectmode="browse", style="Zephyr.Treeview", takefocus=True)
        # Always reserve the scrollbar column, so the columns never jump when it appears.
        frame.columnconfigure(1, minsize=self._team_layout["scroll"])
        widths = self._team_layout["widths"]
        headings = {"#0": WIN_TEXT["columnDrones"], "loss": WIN_TEXT["columnLoss"], "age": WIN_TEXT["columnAge"], "end": ""}
        for column in ("#0",) + columns:
            anchor = "w" if column == "#0" else "e"
            tree.heading(column, text=headings[column], anchor=anchor)
            tree.column(column, width=widths[column], minwidth=widths[column], stretch=column == "#0", anchor=anchor)
        scroll = ttk.Scrollbar(frame, orient="vertical", style="Thin.Vertical.TScrollbar", command=tree.yview)
        tree.configure(yscrollcommand=self._auto_scroll(scroll, row=3, column=1))
        tree.grid(row=3, column=0, sticky="nsew", pady=(self.px(SPACE["sm"]), 0))
        tree.bind("<<TreeviewSelect>>", self._on_select)
        tree.bind("<<TreeviewOpen>>", lambda _event: self._on_group_toggle(True))
        tree.bind("<<TreeviewClose>>", lambda _event: self._on_group_toggle(False))
        tree.bind("<Button-1>", self._on_team_click)
        tree.bind("<Double-Button-1>", self._on_team_double)
        tree.bind("<Return>", lambda _event: self._focus_selected_drone())
        for sequence in self._context_sequences():
            tree.bind(sequence, self._on_team_context)
        self.fleet = tree
        self.team_hover = self.widgets.TreeHover(tree)
        self.team_tip = self.widgets.Tooltip(tree, "", lambda: self.theme, self.fonts["caption"], scale=self.scale,
                                             text_for=self._team_cell_tip)
        self.team_empty = ttk.Label(tree, style="Surface.Secondary.TLabel", anchor="center", justify="center",
                                    wraplength=width - 2 * pad - 2 * self.px(SPACE["lg"]))
        self.missing_legend = ttk.Label(frame, text=WIN_TEXT["missingLegend"], style="Surface.Caption.TLabel")
        self.missing_legend.grid(row=4, column=0, columnspan=2, sticky="w", pady=(self.px(SPACE["xs"]), 0))
        self.missing_legend.grid_remove()
        return frame

    def _team_cell_tip(self, x: int, y: int) -> str | None:
        """Hover text for a dash cell: say plainly that the value was not recorded."""
        tree = self.fleet
        if not row_region(tree, x, y):
            return None
        row, column = tree.identify_row(y), tree.identify_column(x)
        if not row or column in ("", "#0"):
            return None
        try:
            values = tree.item(row, "values")
            index = int(column.lstrip("#")) - 1
        except (self.tk.TclError, ValueError):
            return None
        return fmt.NOT_RECORDED if 0 <= index < len(values) and str(values[index]) == fmt.EN_DASH else None

    def _build_scene(self, parent: Any, scene_class: Any) -> Any:
        ttk = self.ttk
        frame = ttk.Frame(parent, style="Base.TFrame")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        points = []
        for step in self.run.get("steps", []):
            points.extend(agent["position_m"] for agent in step.get("agents", []) if agent.get("position_m"))
            if step.get("target_position_m"):
                points.append(step["target_position_m"])
        self.scene = scene_class(
            frame, scale=self.scale, fonts=self.fonts, all_points=points,
            on_select=self._select_drone, on_focus=self._focus_drone, on_context=self._show_context_menu,
            reduced_motion=self.reduced_motion,
        )
        self.canvas = self.scene.canvas
        self.canvas.grid(row=0, column=0, sticky="nsew")
        return frame

    def _build_inspector(self, parent: Any) -> Any:
        ttk = self.ttk
        width = self.px(LAYOUT["inspectorWidth"])
        frame = ttk.Frame(parent, style="Surface.TFrame", width=width)
        frame.grid_propagate(False)
        frame.pack_propagate(False)
        scroll = self.widgets.ScrollFrame(frame, scale=self.scale)
        scroll.outer.pack(fill="both", expand=True)
        self.inspector_scroll = scroll
        self._theme_hooks.append(scroll.apply_theme)
        pad = self.px(SPACE["lg"])
        # The scroll area always reserves its scrollbar column, so the right padding is what is left of
        # SPACE["lg"] after that column; the right inset is the same with or without a scrollbar.
        right = max(0, pad - scroll.reserve)
        content = width - pad - scroll.reserve - right
        body = ttk.Frame(scroll.inner, style="Surface.TFrame", padding=(pad, self.px(SPACE["md"]), right, pad))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        self._header_row(body, 0)
        header = ttk.Frame(body, style="Surface.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        self.inspector_title = ttk.Label(header, style="Surface.Headline.TLabel")
        self.inspector_title.pack(side="left")
        self.inspector_word = ttk.Label(header, style="Surface.Body.TLabel", compound="left", anchor="e")
        self.inspector_word.pack(side="right")
        self.inspector_hint = ttk.Label(body, text=UI_TEXT["inspector"]["emptyHint"], style="Surface.Secondary.TLabel",
                                        wraplength=content, justify="left")
        self.inspector_hint.grid(row=1, column=0, sticky="w", pady=(self.px(SPACE["xs"]), 0))
        self.inspector_hint.grid_remove()
        tiles = ttk.Frame(body, style="Surface.TFrame")
        tiles.grid(row=2, column=0, sticky="ew", pady=(self.px(SPACE["sm"]), 0))
        tiles.columnconfigure(0, weight=1, uniform="tiles")
        tiles.columnconfigure(1, weight=1, uniform="tiles")
        self.tiles = {}
        # Same order as the status strip: estimate error, delay, loss, age.
        captions = (("estimate", "Estimate error"), ("delay", fmt.field_label("target_link_delay")),
                    ("loss", fmt.field_label("target_packet_loss")), ("target_age", "Target age"))
        gap = self.px(SPACE["sm"])
        for index, (key, caption) in enumerate(captions):
            tile = self.widgets.MetricTile(tiles, caption, self.scale)
            tile.frame.grid(row=index // 2, column=index % 2, sticky="nsew",
                            padx=(0, gap) if index % 2 == 0 else (0, 0), pady=(0, gap))
            self.tiles[key] = tile
        label_width = self.px(WIN_LAYOUT["inspectorLabelWidth"])
        wrap = content - label_width - self.px(SPACE["md"])
        section_pad = (self.px(SPACE["md"]), self.px(SPACE["xs"]))
        self.motion_heading = ttk.Label(body, text=UI_TEXT["inspector"]["positionAndMotion"], style="Surface.Strong.TLabel")
        self.motion_heading.grid(row=3, column=0, sticky="w", pady=section_pad)
        # Labels wrap inside their column, so every grid keeps the same value column and nothing is pushed off the edge.
        grid_options = {"label_width": label_width, "wrap": wrap, "scale": self.scale, "label_wrap": label_width - self.px(SPACE["md"])}
        self.motion_grid = self.widgets.KeyValueGrid(body, **grid_options)
        self.motion_grid.frame.grid(row=4, column=0, sticky="ew")
        self.neighbor_heading = ttk.Label(body, text=UI_TEXT["inspector"]["neighbors"], style="Surface.Strong.TLabel")
        self.neighbor_heading.grid(row=5, column=0, sticky="w", pady=section_pad)
        self.neighbor_grid = self.widgets.KeyValueGrid(body, **grid_options)
        self.neighbor_grid.frame.grid(row=6, column=0, sticky="ew")
        self.inspector_rule = self._hairline(body)
        self.inspector_rule.grid(row=7, column=0, sticky="ew", pady=(self.px(SPACE["md"]), self.px(SPACE["xs"])))
        self.details = self.widgets.Disclosure(
            body, UI_TEXT["inspector"]["details"], self.images, pad=self.px(SPACE["xs"]),
            on_toggle=lambda opened: opened and self.root.after_idle(lambda: self.inspector_scroll.reveal(self.details.frame)))
        self.details.frame.grid(row=8, column=0, sticky="ew")
        # Details holds long ids and part names: each label sits above its value, so a value gets the full
        # panel width and an id like "drone-009-synthetic-profile" stays on one line.
        self.details_grid = self.widgets.KeyValueGrid(self.details.body, wrap=content, scale=self.scale, stacked=True)
        self.details_grid.frame.pack(fill="x")
        self._inspector_sections = (tiles, self.motion_heading, self.motion_grid.frame, self.neighbor_heading,
                                    self.neighbor_grid.frame, self.inspector_rule, self.details.frame)
        return frame

    def _transport_button(self, parent: Any, icon: str, tip: str, command: Callable[[], None]) -> tuple[Any, Any]:
        button = self.ttk.Button(parent, style="Icon.TButton", command=command,
                                 image=self.images.icon(icon, "text", WIN_LAYOUT["iconSize"]))
        return button, self.widgets.Tooltip(button, tip, lambda: self.theme, self.fonts["caption"], scale=self.scale)

    def _build_timeline(self, parent: Any) -> Any:
        """Transport, scrubber, time, speed: one control height and SPACE["lg"] between groups."""
        ttk = self.ttk
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=(self.px(SPACE["lg"]), self.px(SPACE["sm"])))
        frame.columnconfigure(1, weight=1)
        group = self.px(SPACE["lg"])
        transport = ttk.Frame(frame, style="Surface.TFrame")
        transport.grid(row=0, column=0)
        items = (
            ("restart", "restart", self._restart),
            ("step_back", "stepBack", self._step_back),
            ("play", "play", self._toggle),
            ("step_forward", "stepForward", self._step),
        )
        for index, (icon, action, command) in enumerate(items):
            tip = self._transport_tip(action)
            button, tooltip = self._transport_button(transport, icon, tip, command)
            button.pack(side="left", padx=(0 if index == 0 else self.px(SPACE["xs"]), 0))
            if action == "play":
                self.play_button, self.play_tip = button, tooltip
                button.configure(style="Primary.Icon.TButton", image=self.images.icon("play", "onAccent", WIN_LAYOUT["iconSize"]))
        self.slider = self.widgets.TimelineSlider(frame, self._seek_frame, self.scale)
        # The slider keeps SPACE["sm"] inside each end for its thumb, so the visible gap is SPACE["lg"].
        inset = self.px(SPACE["sm"])
        self.slider.canvas.grid(row=0, column=1, sticky="ew", padx=(group - inset, group - inset))
        self._theme_hooks.append(self.slider.apply_theme)
        body = self.fonts["body"]
        longest = fmt.time_label(self.session.duration, self.session.duration)
        holder = ttk.Frame(frame, style="Surface.TFrame", width=body.measure(longest.replace("1", "0")) + self.px(SPACE["xxs"]),
                           height=body.metrics("linespace"))
        holder.grid(row=0, column=2)
        holder.pack_propagate(False)
        self.time_label = ttk.Label(holder, style="Surface.Body.TLabel", anchor="e")
        self.time_label.pack(fill="both", expand=True)
        speeds = ttk.Frame(frame, style="Surface.TFrame")
        speeds.grid(row=0, column=3, padx=(group, 0))
        ttk.Label(speeds, text="Speed", style="Surface.Caption.TLabel").pack(side="left", padx=(0, self.px(SPACE["sm"])))
        self._segments(speeds, [(speed, speed_label(speed)) for speed in SPEEDS], self.speed_var,
                       lambda: self._set_speed(self.speed_var.get())).pack(side="left")
        return frame

    @staticmethod
    def _transport_tip(action: str) -> str:
        spec = TRANSPORT[action]
        shortcut = "playPause" if action in ("play", "pause") else action
        return f"{spec['label']} ({fmt.accelerator(shortcut)})"

    def _build_events(self, parent: Any) -> Any:
        ttk = self.ttk
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=(self.px(SPACE["lg"]), self.px(SPACE["sm"])))
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=WIN_TEXT["eventsTitle"], style="Surface.Caption.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, self.px(SPACE["xs"])))
        ttk.Label(frame, text=WIN_TEXT["eventsHint"], style="Surface.Caption.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="e", pady=(0, self.px(SPACE["xs"])))
        # The list is only as wide as its longest event, so its scrollbar sits right next to the text.
        holder = ttk.Frame(frame, style="Surface.TFrame")
        holder.grid(row=1, column=0, columnspan=2, sticky="w")
        tree = ttk.Treeview(holder, columns=("word", "time", "text"), show="tree", selectmode="browse",
                            height=WIN_LAYOUT["eventRows"], style="Events.Treeview", takefocus=True)
        body = self.fonts["body"]
        pad = self.px(SPACE["md"])
        glyph = self.px(WIN_LAYOUT["glyphSize"]) + self.px(SPACE["md"])
        word = max(body.measure(fmt.status_word(level)) for level in fmt.LEVELS) + pad
        time_width = body.measure("00.00 s") + pad
        tree.column("#0", width=glyph, minwidth=glyph, stretch=False)
        tree.column("word", width=word, minwidth=word, stretch=False, anchor="w")
        tree.column("time", width=time_width, minwidth=time_width, stretch=False, anchor="w")
        tree.column("text", width=self.px(WIN_LAYOUT["eventTextMinWidth"]), stretch=False, anchor="w")
        self._events_fixed = glyph + word + time_width
        scroll = ttk.Scrollbar(holder, orient="vertical", style="Thin.Vertical.TScrollbar", command=tree.yview)
        tree.configure(yscrollcommand=self._auto_scroll(scroll, row=0, column=1))
        holder.columnconfigure(1, minsize=self.px(SPACE["sm"]) + self.px(SPACE["xxs"]))
        tree.grid(row=0, column=0, sticky="w")
        frame.bind("<Configure>", lambda _event: self._fit_events_width(), add="+")
        tree.bind("<<TreeviewSelect>>", self._on_event_select)
        # Act on the press, not the release: the seek re-renders the list, and by release time
        # a different row can sit under the pointer.
        tree.bind("<Button-1>", self._on_event_click)
        tree.bind("<Return>", lambda _event: self._on_event_select(force=True))
        self.events_tree = tree
        self.events_hover = self.widgets.TreeHover(tree)
        self.events_tip = self.widgets.Tooltip(tree, "", lambda: self.theme, self.fonts["caption"], scale=self.scale,
                                               text_for=self._event_tip)
        return frame

    def _fit_events_width(self) -> None:
        """Size the text column to the longest event, within the pane, so the scrollbar stays beside it."""
        tree = self.events_tree
        frame = tree.master.master
        try:
            available = int(frame.winfo_width()) - 2 * self.px(SPACE["lg"]) - self._events_fixed - 2 * self.px(SPACE["sm"])
        except self.tk.TclError:
            return
        body = self.fonts["body"]
        longest = max((body.measure(str(tree.item(iid, "values")[2])) for iid in tree.get_children("")), default=0)
        wanted = max(self.px(WIN_LAYOUT["eventTextMinWidth"]), longest + self.px(SPACE["xl"]))
        width = max(self.px(WIN_LAYOUT["eventTextMinWidth"]), min(wanted, available))
        if int(tree.column("text", "width")) != width:
            tree.column("text", width=width)
            # ttk does not recompute a Treeview's requested width when a column changes; re-setting the
            # displayed columns makes it, so the list (and the scrollbar beside it) actually grows.
            tree.configure(displaycolumns=tree.cget("displaycolumns"))

    def _event_tip(self, _x: int, y: int) -> str | None:
        """Full event text on hover, for a long row the column cuts off."""
        row = self.events_tree.identify_row(y)
        if not row:
            return None
        try:
            values = self.events_tree.item(row, "values")
        except self.tk.TclError:
            return None
        text = str(values[2]) if len(values) > 2 else ""
        column = self.events_tree.column("text", "width")
        return text if self.fonts["body"].measure(text) > int(column) else None

    def _auto_scroll(self, scrollbar: Any, *, row: int, column: int) -> Callable[[str, str], None]:
        state = {"shown": False}

        def update(first: str, last: str) -> None:
            scrollbar.set(first, last)
            needed = float(first) > 0.0 or float(last) < 1.0
            if needed != state["shown"]:
                state["shown"] = needed
                if needed:
                    scrollbar.grid(row=row, column=column, sticky="nse", padx=(self.px(SPACE["xxs"]), 0))
                else:
                    scrollbar.grid_remove()
        return update

    @staticmethod
    def _context_sequences() -> tuple[str, ...]:
        if sys.platform == "darwin":
            return ("<Button-2>", "<Button-3>", "<Control-Button-1>")
        return ("<Button-3>",)

    def _bind_shortcuts(self) -> None:
        for shortcut in SHORTCUTS:
            handler = getattr(self, SHORTCUT_HANDLERS[shortcut.action_id])
            if shortcut.single_key:
                callback = self._shortcut(handler, shortcut.key)
                self.root.bind(shortcut.tk_binding, callback)
                if len(shortcut.key) == 1:
                    self.root.bind(f"<{shortcut.key.upper()}>", callback)
            else:
                chord = lambda _event, handler=handler: (handler(), "break")[1]  # noqa: E731
                self.root.bind(shortcut.tk_binding, chord)
                if sys.platform == "darwin" and "Control-" in shortcut.tk_binding:
                    # On a Mac the same chords also answer to Command, as Mac users expect.
                    self.root.bind(shortcut.tk_binding.replace("Control-", "Command-"), chord)

    def _shortcut(self, action: Callable[[], Any], key: str = "") -> Callable[[Any], Any]:
        """Wrap a single-key shortcut so it stands down while a text field (or owning widget) has focus."""
        def handler(event: Any) -> Any:
            widget = getattr(event, "widget", None)
            try:
                widget_class = widget.winfo_class() if widget is not None else ""
            except Exception:
                widget_class = ""
            # Only the team list has groups to open and close; the Events list hands every key to the replay.
            events = getattr(self, "events_tree", None)
            owns = not (events is not None and widget is not None and str(widget) == str(events))
            if shortcut_blocked(widget_class, key, tree_owns_keys=owns):
                return None
            action()
            return "break"

        return handler

    # ---- theme -------------------------------------------------------------
    def _apply_theme(self) -> None:
        theme = self.theme
        p = theme.p
        self._configure_styles(self.style, theme, self.fonts, self.scale)
        self.style.configure("Field.Placeholder.TLabel", background=p.background, foreground=p.textSecondary, font=self.fonts["body"])
        self.images.apply(theme)
        self.root.configure(background=p.background)
        for tree in (self.fleet, self.events_tree):
            tree.tag_configure("hover", background=p.hoverFill)
        self.events_tree.tag_configure("past", foreground=p.textSecondary)
        self.events_tree.tag_configure("current", foreground=p.text)
        self.fleet.tag_configure("group", font=self.fonts["bodyStrong"], foreground=p.text)
        # A closed group holding the selected drone: accent-colored text, no fill, so it never reads as selected.
        self.fleet.tag_configure("holds", foreground=p.accent, font=self.fonts["bodyStrong"])
        self.scene.apply_theme(theme)
        alive = []
        for hook in self._theme_hooks:
            try:
                hook(theme)
                alive.append(hook)
            except self.tk.TclError:
                continue
        self._theme_hooks = alive
        results = self._results.window if self._results is not None and self._results.exists() else None
        for window in (self.root, self._about_window, self._evidence_window, self._scenario_window, results):
            if window is not None:
                try:
                    if window is not self.root:
                        window.configure(background=p.background)
                    self._apply_title_bar(window, theme.dark)
                except self.tk.TclError:
                    pass

    def _set_appearance(self, value: str) -> None:
        self.appearance = value
        self.appearance_var.set(value)
        if self._persist_appearance:
            try:
                save_appearance(value)
            except OSError:
                pass
        name = resolve_theme(value, system_prefers_dark())
        if name != self.theme.name:
            self.theme = self._Theme(name)
            self._apply_theme()
            self._render()

    # ---- rendering ---------------------------------------------------------
    def _render(self) -> None:
        """Refresh every region from the session. Never re-enters through Tk selection events."""
        self.render_count += 1
        presentation = self.session.presentation
        self._render_strip(presentation)
        self._render_fleet()
        self._render_inspector()
        self._render_timeline()
        self._render_events()
        self.scene.set_frame(self._raw_frame(), presentation["per_drone"], self.session.selected_drone_id)

    def _raw_frame(self) -> Mapping[str, Any]:
        return self.run.get("steps", self.run.get("frames", []))[self.session.frame_index]

    def _set_metric(self, label: str, text: str | None) -> None:
        widget = self.strip_values[label]
        if text is None:
            self._set(widget, text=fmt.NOT_RECORDED, style="Base.MetricMissing.TLabel")
        else:
            self._set(widget, text=text, style="Base.Metric.TLabel")

    def _render_strip(self, presentation: Mapping[str, Any]) -> None:
        labels = UI_TEXT["statusStrip"]
        headline = presentation["headline"]
        self._set(self.strip_glyph, image=self.images.status(headline["level"], WIN_LAYOUT["stripGlyphSize"],
                                                             WIN_LAYOUT["glyphBox"]))
        self._set(self.strip_values[labels[0]], text=headline["text"], style="Base.Metric.TLabel")
        strip = presentation["strip"]
        self._set_metric(labels[1], fmt.format_value(presentation.get("estimate_error_p95_m"), "m"))
        self._set_metric(labels[2], fmt.format_value(strip.get("target_link_delay_ms"), "ms"))
        self._set_metric(labels[3], fmt.format_value(strip.get("target_packet_loss_pct"), "pct1"))
        self._set_metric(labels[4], fmt.format_age_span(strip.get("newest_target_age_s"), strip.get("oldest_target_age_s")))

    def _render_fleet(self) -> None:
        tree = self.fleet
        presentation = self.session.presentation
        query = self.filter_var.get()
        self._set(self.team_summary, text=presentation["team_summary"])
        plan = team_rows(presentation["groups"], presentation["per_drone"], query)
        filtering = bool(query.strip())
        wanted_groups = [row["iid"] for row in plan]
        for row in plan:
            if row["iid"] not in self._seen_groups:
                self._seen_groups.add(row["iid"])
                if row["level"] != "ok":
                    self._open_groups.add(row["iid"])
        existing_groups = set(tree.get_children(""))
        wanted_children = {child["iid"] for row in plan for child in row["children"]}
        glyph, box = WIN_LAYOUT["glyphSize"], WIN_LAYOUT["glyphBox"]
        for row in plan:
            image = self.images.status(row["level"], glyph, box)
            if tree.exists(row["iid"]):
                self._tree_item(row["iid"], text=row["label"], values=row["values"], image=image)
            else:
                tree.insert("", "end", iid=row["iid"], text=row["label"], values=row["values"], image=image, tags=("group",))
        for index, row in enumerate(plan):
            if tree.parent(row["iid"]) != "" or tree.index(row["iid"]) != index:
                tree.move(row["iid"], "", index)
            for child_index, child in enumerate(row["children"]):
                image = self.images.status(child["level"], glyph, box)
                if tree.exists(child["iid"]):
                    self._tree_item(child["iid"], text=child["label"], values=child["values"], image=image)
                    if tree.parent(child["iid"]) != row["iid"] or tree.index(child["iid"]) != child_index:
                        tree.move(child["iid"], row["iid"], child_index)
                else:
                    tree.insert(row["iid"], child_index, iid=child["iid"], text=child["label"], values=child["values"],
                                image=image)
            should_open = row["iid"] in self._open_groups or (filtering and row["level"] not in pinned_levels())
            if bool(tree.item(row["iid"], "open")) != should_open:
                tree.item(row["iid"], open=should_open)
        for group in existing_groups:
            for child in tree.get_children(group):
                if child not in wanted_children:
                    self.team_hover.forget(child)
                    tree.delete(child)
        for group in existing_groups - set(wanted_groups):
            if tree.exists(group):
                self.team_hover.forget(group)
                tree.delete(group)
        self._team_plan = plan
        self._render_filter_state(presentation, query, plan)
        self._sync_team_selection()

    def _render_filter_state(self, presentation: Mapping[str, Any], query: str, plan: Sequence[Mapping[str, Any]]) -> None:
        """Caption ("Showing 3 of 50"), the no-match message, the clear button, and the dash legend."""
        state = team_filter(presentation["groups"], query)
        caption = state["caption"]
        if caption:
            pinned = any(row["level"] in pinned_levels() for row in plan)
            parts = [state["empty_text"], caption] if state["empty_text"] else [caption]
            if pinned:
                parts.append(UI_TEXT["teamFilter"]["pinnedNote"])
            text = " ".join(part if part.endswith(".") else f"{part}." for part in parts)
            self._set(self.filter_caption, text=text)
        self._show("caption", bool(caption), lambda: self.filter_caption.grid(),
                   self.filter_caption.grid_remove)
        self._show("clear", bool(caption),
                   lambda: self.clear_button.place(in_=self.filter_entry, relx=1.0, x=-self.px(SPACE["xs"]), rely=0.5, anchor="e"),
                   self.clear_button.place_forget)
        empty = not plan and bool(state["empty_text"])
        if empty:
            self._set(self.team_empty, text=state["empty_text"])
        self._show("empty", empty, lambda: self.team_empty.place(relx=0.5, rely=0.3, anchor="center"),
                   self.team_empty.place_forget)
        open_now = {row["iid"] for row in plan if self.fleet.exists(row["iid"]) and self.fleet.item(row["iid"], "open")}
        self._show("legend", rows_have_missing(plan, open_now), lambda: self.missing_legend.grid(),
                   self.missing_legend.grid_remove)

    def _show(self, key: str, visible: bool, show: Callable[[], Any], hide: Callable[[], Any]) -> None:
        """Show or hide a widget only when that changes, so a playback frame does no geometry work for it."""
        shown = self.__dict__.setdefault("_shown_state", {})
        if shown.get(key) is visible:
            return
        shown[key] = visible
        (show if visible else hide)()

    def _tree_item(self, iid: str, **options: Any) -> None:
        """Update a team row only when its text, values, or image changed."""
        current = self.fleet.item(iid)
        changed = {}
        for name, value in options.items():
            now = current.get(name)
            if name == "values":
                same = tuple(str(item) for item in (now or ())) == tuple(str(item) for item in value)
            elif name == "image":
                same = tuple(str(item) for item in (now or ())) == (str(value),)
            else:
                same = str(now) == str(value)
            if not same:
                changed[name] = value
        if changed:
            self.fleet.item(iid, **changed)

    def _open_now(self) -> set[str]:
        return {row["iid"] for row in self._team_plan if self.fleet.exists(row["iid"]) and self.fleet.item(row["iid"], "open")}

    def _sync_team_selection(self, reveal: bool = False) -> None:
        """Make the tree selection match the session. Only calls selection_set when it differs.

        A drone hidden inside a closed group leaves the selection empty and marks
        that group with the quieter "holds" tag, so the highlight never sits on a
        row the user did not pick.
        """
        open_now = self._open_now()
        selected = self.session.selected_drone_id
        desired = desired_team_selection(self._team_plan, selected, open_now)
        current = tuple(self.fleet.selection())
        if desired is None:
            if current:
                self.fleet.selection_set(())
        elif current != (desired,):
            self.fleet.selection_set(desired)
            self.fleet.focus(desired)
        self._mark_holding_group(group_holding_selection(self._team_plan, selected, open_now))
        if desired is not None and reveal:
            self.fleet.see(desired)

    def _mark_holding_group(self, iid: str | None) -> None:
        previous = self.__dict__.get("_holding_group")
        if previous == iid:
            return
        tree = self.fleet
        for row, add in ((previous, False), (iid, True)):
            if row and tree.exists(row):
                # "holds" replaces "group" (same bold font, accent color), so no two tags compete for the color.
                tags = [tag for tag in tree.item(row, "tags") if tag not in ("holds", "group")]
                tree.item(row, tags=tuple(tags + ["holds" if add else "group"]))
        self._holding_group = iid
        if iid and not tree.selection():
            tree.focus(iid)  # arrow keys continue from the row that holds the drone

    def _on_select(self, _event: Any = None) -> None:
        """Tree selection changed. Idempotent: a selection that already matches renders nothing.

        ``selection_set`` queues ``<<TreeviewSelect>>`` for later, after any
        boolean guard would already be reset, so the guard here is the state
        comparison itself.
        """
        selection = self.fleet.selection()
        if not selection:
            return
        drone = drone_for_team_row(self._team_plan, selection[0], self.session.selected_drone_id)
        if drone is None or drone == self.session.selected_drone_id:
            return
        self.session.selected_drone_id = drone
        self._render_selection(reveal=False)

    def _render_selection(self, reveal: bool) -> None:
        self.render_count += 1
        self._sync_team_selection(reveal=reveal)
        self._render_inspector()
        self.scene.set_selected(self.session.selected_drone_id)

    def _select_drone(self, drone_id: str, reveal: bool = True) -> None:
        if drone_id == self.session.selected_drone_id:
            return
        self.session.selected_drone_id = drone_id
        self._render_selection(reveal=reveal)

    def _focus_drone(self, drone_id: str) -> None:
        self._select_drone(drone_id)
        self.scene.focus_drone(drone_id)

    def _on_group_toggle(self, opened: bool) -> None:
        item = self.fleet.focus()
        if not item or item not in {row["iid"] for row in self._team_plan}:
            return
        if opened:
            self._open_groups.add(item)
        else:
            self._open_groups.discard(item)
        self.root.after_idle(self._sync_team_selection)
        self.root.after_idle(lambda: self._render_filter_state(self.session.presentation, self.filter_var.get(), self._team_plan))

    def _on_team_click(self, event: Any) -> str | None:
        """A click on a group header opens or closes it and leaves the selected drone alone."""
        if not self._in_rows(self.fleet, event):
            return None
        row = self.fleet.identify_row(event.y)
        if not row or row not in {plan["iid"] for plan in self._team_plan}:
            return None
        self.fleet.focus_set()
        self.fleet.focus(row)
        opened = not bool(self.fleet.item(row, "open"))
        self.fleet.item(row, open=opened)
        if opened:
            self._open_groups.add(row)
        else:
            self._open_groups.discard(row)
        self._sync_team_selection()
        self._render_filter_state(self.session.presentation, self.filter_var.get(), self._team_plan)
        return "break"

    @staticmethod
    def _in_rows(tree: Any, event: Any) -> bool:
        """True when the pointer is over a row, not the headings or empty space.

        Tk 8.6 ``identify_row`` reports the row scrolled above the view when the
        pointer is on the headings, so every row action checks the region first.
        """
        return row_region(tree, event.x, event.y)

    def _on_team_double(self, event: Any) -> str | None:
        if not self._in_rows(self.fleet, event):
            return None
        row = self.fleet.identify_row(event.y)
        if row in {plan["iid"] for plan in self._team_plan}:
            # The first click already opened or closed the group; Tk's own double-click
            # toggle would undo it, so the double-click stops here.
            return "break"
        if row and row in {child["iid"] for plan in self._team_plan for child in plan["children"]}:
            self._focus_drone(row)
            return "break"
        return None

    def _on_team_context(self, event: Any) -> str:
        row = self.fleet.identify_row(event.y) if self._in_rows(self.fleet, event) else ""
        drone = drone_for_team_row(self._team_plan, row, self.session.selected_drone_id) if row else None
        if drone:
            self._show_context_menu(drone, event.x_root, event.y_root)
        return "break"

    def _show_context_menu(self, drone_id: str, x: int, y: int) -> None:
        menu = self.tk.Menu(self.root, tearoff=False)
        menu.add_command(label="Focus Camera", command=lambda: self._focus_drone(drone_id))
        menu.add_command(label="Show in Inspector", command=lambda: self._show_in_inspector(drone_id))
        menu.add_command(label="Copy Drone ID", command=lambda: self._copy_text(drone_id))
        try:
            menu.tk_popup(x, y)
        except self.tk.TclError:
            return  # no popup, so there is no grab to release
        try:
            menu.grab_release()
        except self.tk.TclError:
            pass

    def _show_in_inspector(self, drone_id: str) -> None:
        self._select_drone(drone_id)
        if not self.session.show_inspector:
            self._set_inspector(True)

    def _copy_text(self, text: str) -> bool:
        """Copy to the clipboard and say so; a busy clipboard (Windows can lock it) is reported, not raised."""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except self.tk.TclError:
            self._flash_notice(WIN_TEXT["clipboardBusy"])
            return False
        self._flash_notice(WIN_TEXT["copied"])
        return True

    def _flash_notice(self, text: str, duration_ms: int = WIN_LAYOUT["noticeMs"]) -> None:
        """Show a short note in the title row, then clear it unless something else replaced it."""
        notice = getattr(self, "notice", None)
        if notice is None:
            return
        try:
            notice.configure(text=text)
        except self.tk.TclError:
            return

        def clear() -> None:
            try:
                if str(notice.cget("text")) == text:
                    notice.configure(text="")
            except self.tk.TclError:
                pass

        self.root.after(duration_ms, clear)

    def _profile(self, profile_id: str | None) -> Mapping[str, Any] | None:
        return next((item for item in self.run.get("agent_profiles", []) if item.get("profile_id") == profile_id), None)

    def _show_inspector_sections(self, visible: bool) -> None:
        """With no drone selected, show one hint line instead of empty headings and tiles."""
        if getattr(self, "_sections_shown", None) == visible:
            return
        self._sections_shown = visible
        for widget in self._inspector_sections:
            widget.grid() if visible else widget.grid_remove()
        self.inspector_hint.grid_remove() if visible else self.inspector_hint.grid()

    def _render_inspector(self) -> None:
        record = self.session.record_for(self.session.selected_drone_id)
        if record is None:
            # Through _set, like the normal path: a direct configure would leave the cache stale,
            # and the header would keep "No drone selected" when the same drone comes back.
            self._set(self.inspector_title, text=UI_TEXT["inspector"]["emptyTitle"])
            self._set(self.inspector_word, text="", image="")
            self._show_inspector_sections(False)
            return
        self._show_inspector_sections(True)
        glyph = WIN_LAYOUT["glyphSize"]
        self._set(self.inspector_title, text=record["heading_name"])
        self._set(self.inspector_word, text=f" {record['word']}", image=self.images.status(record["level"], glyph,
                                                                                            WIN_LAYOUT["glyphBox"]))
        self.tiles["target_age"].set(fmt.format_value(record.get("target_age_s"), "s"))
        self.tiles["estimate"].set(fmt.format_value(record.get("estimate_error_m"), "m"))
        self.tiles["delay"].set(fmt.format_value(record.get("target_link_delay_ms"), "ms"))
        self.tiles["loss"].set(fmt.format_value(record.get("target_packet_loss_pct"), "pct1"))
        self.motion_grid.set_rows((
            (UI_TEXT["inspector"]["altitude"], fmt.format_value(record.get("altitude_m"), "m")),
            (fmt.field_label("speed_mps"), fmt.format_value(record.get("speed_mps"), "mps")),
            (fmt.field_label("heading_deg"), fmt.heading_text(record.get("heading_deg"), record.get("speed_mps"))),
        ))
        self.neighbor_grid.set_rows((
            (fmt.field_label("neighbor_count"), fmt.format_scalar(record.get("neighbor_count"))
             if record.get("neighbor_count") is not None else None),
            (fmt.field_label("nearest_neighbor_m"), fmt.format_value(record.get("nearest_neighbor_m"), "m")),
            (fmt.field_label("neighbor_link_delay_ms"), fmt.format_value(record.get("neighbor_link_delay_ms"), "ms")),
            (fmt.field_label("neighbor_packet_loss_pct"), fmt.format_value(record.get("neighbor_packet_loss_pct"), "pct1")),
        ))
        agent = next((item for item in self._raw_frame().get("agents", []) if item.get("agent_id") == record["drone_id"]), {})
        profile = self._profile(record.get("profile_id")) or {}
        motor = profile.get("motor") or {}
        propeller = profile.get("propeller") or {}
        flags = record.get("constraint_flags") or []
        rows = [
            (fmt.field_label("agent_id"), record["drone_id"], "mono"),
            (fmt.field_label("position_m"), fmt.format_vector(agent.get("position_m"))),
            (fmt.field_label("velocity_mps"), fmt.format_vector(agent.get("velocity_mps"))),
            (fmt.field_label("profile_id"), record.get("profile_id"), "mono"),
        ]
        if profile:
            rows += [
                (fmt.field_label("motor"), fmt.part_text(motor.get("part_id"), fmt.quantity(motor.get("max_power_w"), "W max"))),
                (fmt.field_label("propeller"), fmt.part_text(propeller.get("part_id"),
                                                             fmt.quantity(propeller.get("diameter_m"), "mm", 1000.0, 1))),
                (fmt.field_label("motor_count"), fmt.format_scalar(profile.get("motor_count"))),
                (fmt.field_label("total_mass_kg"), fmt.format_scalar(profile.get("total_mass_kg"))),
                (fmt.field_label("estimated_max_thrust_n"), fmt.format_scalar(profile.get("estimated_max_thrust_n"))),
                (fmt.field_label("battery_capacity_wh"), fmt.format_scalar(profile.get("battery_capacity_wh"))),
                ("Profile source", fmt.format_scalar(profile.get("status"))),
            ]
        rows.append((fmt.field_label("constraint_flags"), ", ".join(fmt.humanize_key(flag) for flag in flags) if flags else "None"))
        self.details_grid.set_rows(rows)

    def _render_timeline(self) -> None:
        session = self.session
        self.slider.set_state(session.frame_index, session.frame_count, session.event_frames())
        self._set(self.time_label, text=fmt.time_label(session.frame_time, session.duration))
        playing = session.playing
        self._set(self.play_button, image=self.images.icon("pause" if playing else "play", "onAccent", WIN_LAYOUT["iconSize"]))
        tip = getattr(self, "play_tip", None)
        if tip is not None:
            tip.set_text(self._transport_tip("pause" if playing else "play"))
        if abs(self.speed_var.get() - session.speed) > 1e-9:
            self.speed_var.set(session.speed)

    def _render_events(self) -> None:
        tree = self.events_tree
        events = self.session.events_through()
        current = self.session.frame_index
        signature = tuple((frame, event["text"]) for frame, event in events)
        if signature != self._event_signature:
            keep = 0
            while keep < min(len(signature), len(self._event_signature)) and signature[keep] == self._event_signature[keep]:
                keep += 1
            for index in range(keep, len(self._event_signature)):
                iid = f"event{index}"
                if tree.exists(iid):
                    self.events_hover.forget(iid)
                    tree.delete(iid)
                    self._event_frames.pop(iid, None)
                    if self.__dict__.get("_event_selected") == iid:
                        # The row is gone; a later row with the same id is a different event.
                        self._event_selected = None
            # No padding box around the glyph: its ink starts at the row edge, in line with the "Events" header.
            box = WIN_LAYOUT["glyphSize"]
            for index in range(keep, len(events)):
                frame, event = events[index]
                level, word, time_text, text = fmt.event_row(event)
                iid = f"event{index}"
                # One glyph size and box for every level, so the row centers line up.
                tree.insert("", "end", iid=iid, image=self.images.status(level, WIN_LAYOUT["glyphSize"], box),
                            values=(word, time_text, text))
                self._event_frames[iid] = frame
            self._event_signature = signature
            self._fit_events_width()
            children = tree.get_children("")
            if children:
                tree.see(children[-1])
        # Events from this frame read in full text; earlier ones step back to the secondary color.
        if getattr(self, "_events_tagged_at", None) != (current, signature):
            self._events_tagged_at = (current, signature)
            for iid, frame in self._event_frames.items():
                if tree.exists(iid):
                    hover = ("hover",) if "hover" in tree.item(iid, "tags") else ()
                    tree.item(iid, tags=("current" if frame == current else "past",) + hover)

    def _on_event_select(self, _event: Any = None, force: bool = False) -> None:
        """Moving the selection with the arrow keys (or Return, ``force``) jumps to that event's frame.

        Tk also fires <<TreeviewSelect>> when a key re-sets the same row (Right does), so an
        unchanged selection is ignored; otherwise that key would seek back to the event.
        """
        selection = self.events_tree.selection()
        if not selection or (not force and selection[0] == self.__dict__.get("_event_selected")):
            return
        self._event_selected = selection[0]
        self._activate_event(selection[0])

    def _on_event_click(self, event: Any) -> str | None:
        """Every click jumps to the row under the pointer at press time, even if it is already selected.

        Returns "break" so Tk's own press handling does not select again after the
        seek has re-rendered (and possibly scrolled) the list.
        """
        tree = self.events_tree
        if not row_region(tree, event.x, event.y):
            return None
        row = tree.identify_row(event.y)
        if not row:
            return None
        tree.focus_set()
        self._event_selected = row  # the <<TreeviewSelect>> that selection_set queues now matches and is ignored
        tree.selection_set(row)
        tree.focus(row)
        self._activate_event(row)
        return "break"

    def _activate_event(self, iid: str) -> None:
        frame = self._event_frames.get(iid)
        if frame is not None:
            self._seek_frame(frame)

    # ---- actions -----------------------------------------------------------
    def _schedule(self, callback: Callable[[], None], delay: int) -> Any:
        return self.root.after(delay, self._run_tick, callback)

    def _cancel(self, job: Any) -> None:
        if job is not None:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass

    def _run_tick(self, callback: Callable[[], None]) -> None:
        callback()
        self._render()

    def _toggle(self) -> None:
        self.session.toggle_play()
        self._render()

    def _step(self) -> None:
        self.session.pause()
        self.session.step_forward()
        self._render()

    def _step_back(self) -> None:
        self.session.pause()
        self.session.step_back()
        self._render()

    def _restart(self) -> None:
        self.session.restart()
        self._render()

    def _reset(self) -> None:
        self._restart()

    def _seek_frame(self, index: int) -> None:
        if index == self.session.frame_index:
            return
        self.session.pause()
        self.session.seek_frame(int(index))
        self._render()

    def _set_speed(self, speed: float) -> None:
        self.session.set_speed(float(speed))
        self.speed_var.set(self.session.speed)

    def _focus_selected_drone(self) -> None:
        if self.session.record_for(self.session.selected_drone_id) is None:
            self._flash_notice(WIN_TEXT["selectDroneFirst"])
            return
        self.scene.focus_drone(self.session.selected_drone_id)

    def _reset_camera(self) -> None:
        self.scene.reset_camera()

    def _set_view_mode(self, mode: str) -> None:
        self.session.view_mode = mode
        self.view_var.set(mode)
        self.scene.set_view_mode(mode)

    def _toggle_view_mode(self) -> None:
        self._set_view_mode("3d" if self.session.view_mode == "top" else "top")

    def _set_keepout(self, visible: bool) -> None:
        self.session.show_keepout = visible
        self.keepout_var.set(visible)
        self.scene.set_keepout_visible(visible)

    def _toggle_keepout(self) -> None:
        self._set_keepout(not self.session.show_keepout)

    def _set_inspector(self, visible: bool) -> None:
        self.session.show_inspector = visible
        self.inspector_var.set(visible)
        if visible:
            self.inspector_line.grid()
            self.inspector_panel.grid()
        else:
            self.inspector_line.grid_remove()
            self.inspector_panel.grid_remove()
        # Refit against the new canvas size once Tk has laid it out, so no drone is cut off.
        self.root.after_idle(self.scene.refit)

    def _toggle_inspector(self) -> None:
        self._set_inspector(not self.session.show_inspector)

    def _focus_filter(self) -> None:
        self.filter_entry.focus_set()
        self.filter_entry.select_range(0, "end")

    def _clear_filter(self, _event: Any = None) -> str:
        """Escape clears the filter and hands the keys back to the list; the clear button keeps typing focus."""
        self.filter_var.set("")
        (self.fleet if _event is not None else self.filter_entry).focus_set()
        return "break"

    def _on_filter(self) -> None:
        self._render_fleet()

    def _route_wheel(self, event: Any) -> None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget is not None and self.inspector_scroll.contains(widget):
            self.inspector_scroll.scroll(event.delta)

    def _route_touchpad(self, event: Any) -> None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        try:
            delta = int(event.delta)
            dy = (delta & 0xFFFF) - (0x10000 if delta & 0x8000 else 0)
        except (TypeError, ValueError):
            return
        if widget is not None and self.inspector_scroll.contains(widget):
            self.inspector_scroll.scroll(dy * 40)
        elif widget is self.canvas:
            self.scene.wheel(dy * 40)

    # ---- windows -------------------------------------------------------------
    def _viewer_context(self) -> Any:
        if __package__ in (None, ""):
            from desktop.ui_viewers import ViewerContext
        else:
            from .ui_viewers import ViewerContext
        return ViewerContext(self.root, lambda: self.theme, self.fonts, self.images, self.scale, self._theme_hooks.append,
                             self._apply_title_bar)

    def _replace_window(self, attribute: str) -> None:
        window = getattr(self, attribute)
        if window is not None:
            try:
                window.destroy()
            except self.tk.TclError:
                pass
            setattr(self, attribute, None)

    def _show_about(self) -> None:
        if __package__ in (None, ""):
            from desktop.ui_viewers import about_window
        else:
            from .ui_viewers import about_window
        self._replace_window("_about_window")
        self._about_window = about_window(self._viewer_context(), self.run, self._about_details())

    def _show_results(self, view: str | None = None, select: Mapping[str, str] | None = None) -> Any:
        """Open the Results window, or bring the open one forward (there is only ever one)."""
        if __package__ in (None, ""):
            from desktop.ui_results import open_results
        else:
            from .ui_results import open_results
        if self._results is not None and self._results.exists():
            self._results.focus()
            if view:
                self._results.show_view(view)
        else:
            self._results = open_results(self._viewer_context(), view or "delay")
        if select:
            self._results.apply_select(select)
        return self._results

    def _replay_digest(self) -> str | None:
        """SHA-256 of the replay file, hashed once in chunks and then cached for later About windows."""
        if "_digest" in self.__dict__:
            return self._digest
        source = getattr(self.model, "source_path", None)
        digest = None
        if source is not None:
            try:
                hasher = hashlib.sha256()
                with Path(source).open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
            except OSError:
                digest = None
        self._digest = digest
        return digest

    def _about_details(self) -> list[tuple[str, Any, str]]:
        digest = self._replay_digest()
        frames = self.run.get("steps", [])
        return [
            (fmt.field_label("scenario_id"), self.run.get("scenario_id"), "mono"),
            (fmt.field_label("seed"), self.run.get("seed"), "body"),
            (fmt.field_label("schema"), self.run.get("schema"), "mono"),
            (fmt.field_label("dt_s"), self.run.get("dt_s"), "body"),
            (fmt.field_label("frame_count"), len(frames), "body"),
            (fmt.field_label("agent_count"), len(frames[0].get("agents", [])) if frames else None, "body"),
            (fmt.field_label("link_event_count"), len(self.run.get("link_events", [])), "body"),
            ("Replay file SHA-256", digest, "mono"),
            ("App version", CURRENT_VERSION, "body"),
            (fmt.field_label("python"), platform.python_version(), "body"),
            ("Tk version", str(self.tk.TkVersion), "body"),
        ]

    def _check_update(self) -> None:
        """Check GitHub in a worker thread and show the result on the Tk thread.

        The worker never touches Tk. With a Tcl built without threads (Tk 9 on
        macOS), ``root.after`` from another thread silently never runs, so the
        worker posts to a queue and the Tk thread polls it. A second click while
        a check is running does nothing.
        """
        import queue

        if getattr(self, "_update_in_flight", False):
            return
        self._update_in_flight = True
        update_text = UI_TEXT["update"]
        self.notice.configure(text=update_text["checking"])
        results: queue.Queue[tuple[str, Any]] = queue.Queue()

        def report(message: str, error: bool) -> None:
            from tkinter import messagebox

            self._update_in_flight = False
            self.notice.configure(text="")
            if error:
                messagebox.showwarning("Check for Updates", message, parent=self.root)
            else:
                messagebox.showinfo("Check for Updates", message, parent=self.root)

        def poll() -> None:
            try:
                while True:
                    kind, value = results.get_nowait()
                    if kind == "notice":
                        self.notice.configure(text=value)
                    elif kind == "quit":
                        self.notice.configure(text=update_text["installing"])
                        self.root.after(700, self.root.destroy)
                        return
                    else:
                        report(*value)
                        return
            except queue.Empty:
                pass
            except Exception:
                return  # the window is already gone
            try:
                self.root.after(100, poll)
            except Exception:
                pass  # the window was closed while the check ran

        def worker() -> None:
            error = False
            staged = None
            try:
                try:
                    from tools.release_updater import GitHubReleaseClient, describe_update_error
                except ModuleNotFoundError:
                    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                    from tools.release_updater import GitHubReleaseClient, describe_update_error
            except Exception as exc:
                results.put(("done", (update_text["failed"].format(reason=f"the updater is missing ({type(exc).__name__})"), True)))
                return
            try:
                client = GitHubReleaseClient()
                release, asset, newer = client.check(CURRENT_VERSION, platform=client.platform_name())
                if not newer:
                    message = update_text["upToDate"].format(tag=release.tag)
                elif not getattr(sys, "frozen", False):
                    message = update_text["available"].format(tag=release.tag)
                else:
                    # Everything that would make the install fail is checked before
                    # the download, so nobody waits for a file that cannot be used.
                    install_dir = Path(sys.executable).resolve().parent
                    updater_candidates = (
                        install_dir / "ZephyrMeshUpdater.exe",
                        install_dir.parent / "ZephyrMeshUpdater.exe",
                    )
                    updater = next((candidate for candidate in updater_candidates if candidate.is_file()), updater_candidates[0])
                    launch = install_dir / "ZephyrMeshWindows.exe"
                    try:
                        # A real write, because os.access ignores Windows folder permissions.
                        with tempfile.TemporaryFile(dir=install_dir.parent, prefix=".zephyr-write-check-"):
                            writable = True
                    except OSError:
                        writable = False
                    if not writable:
                        message = update_text["failed"].format(reason="the application folder is read-only")
                        error = True
                    elif not updater.is_file():
                        message = update_text["failed"].format(reason="the update helper is missing")
                        error = True
                    else:
                        results.put(("notice", update_text["downloading"]))
                        staged = client.download_and_stage(asset)
                        updater = updater_launch_path(updater, install_dir, pid=os.getpid())
                        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        subprocess.Popen(
                            [str(updater), "--pid", str(os.getpid()), "--staged", str(staged.payload), "--install-dir",
                                    str(install_dir), "--launch", str(launch)],
                            creationflags=flags,
                            close_fds=True,
                        )
                        results.put(("quit", None))
                        return
            except Exception as exc:
                if staged is not None:
                    shutil.rmtree(staged.root, ignore_errors=True)
                message = update_text["failed"].format(reason=describe_update_error(exc))
                error = True
            results.put(("done", (message, error)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, poll)

    @staticmethod
    def _evidence_loader() -> tuple[Any, type[ValueError]]:
        """Load the shared report decoder without making Tk import it early."""

        if __package__ in (None, ""):
            from desktop.evidence import EvidenceError, load_evidence_report
        else:
            from .evidence import EvidenceError, load_evidence_report
        return load_evidence_report, EvidenceError

    def _open_evidence_report(self) -> bool:
        """Ask for one portable report and show it in a read-only viewer."""

        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Open Evidence Report",
            filetypes=(("JSON evidence reports", "*.json"), ("All files", "*.*")),
        )
        if not selected:
            return False
        return self._show_evidence_report(Path(selected))

    def _open_scenario_run(self) -> bool:
        """Open one generated scenario run in a read-only viewer."""

        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Open Scenario Run",
            filetypes=(("JSON scenario runs", "*.json"), ("All files", "*.*")),
        )
        if not selected:
            return False
        return self._show_scenario_run(Path(selected))

    def _show_scenario_run(self, path: Path) -> bool:
        """Decode a scenario run strictly, then show it without touching replay state."""

        if __package__ in (None, ""):
            from desktop.scenario_run import ScenarioRunFormatError
            from desktop.ui_viewers import scenario_window
        else:
            from .scenario_run import ScenarioRunFormatError
            from .ui_viewers import scenario_window
        title, noun = WIN_TEXT["openScenarioTitle"], WIN_TEXT["scenarioNoun"]
        if not Path(path).is_file():
            self._report_open_failure(title, open_error_text(Path(path), None, noun))
            return False
        try:
            run = load_scenario_run(path)
        except (ScenarioRunFormatError, OSError, TypeError, ValueError, RecursionError) as exc:
            self._report_open_failure(title, open_error_text(Path(path), exc, noun))
            return False
        self._replace_window("_scenario_window")
        self._scenario_window = scenario_window(self._viewer_context(), run, Path(path))
        return True

    def _show_evidence_report(self, path: Path) -> bool:
        """Decode and display an evidence envelope without touching replay state."""

        loader, evidence_error = self._evidence_loader()
        title, noun = WIN_TEXT["openEvidenceTitle"], WIN_TEXT["evidenceNoun"]
        if not Path(path).is_file():
            self._report_open_failure(title, open_error_text(Path(path), None, noun))
            return False
        try:
            report = loader(Path(path))
        except (evidence_error, OSError, TypeError, ValueError, RecursionError) as exc:
            self._report_open_failure(title, open_error_text(Path(path), exc, noun))
            return False
        if not isinstance(report, dict):
            self._report_open_failure(title, open_error_text(
                Path(path), ValueError("The report decoder returned an invalid object"), noun))
            return False
        self._evidence_report = report
        self._evidence_path = Path(path)
        self._render_evidence_report(report, self._evidence_path)
        return True

    def _report_open_failure(self, title: str, message: str) -> None:
        """Show an open-file error, or print it during a --smoke-* run, where a dialog would hang."""
        if getattr(self, "_smoke_mode", False):
            print(f"{title}: {message}", file=sys.stderr)
            return
        from tkinter import messagebox

        messagebox.showerror(title, message, parent=self.root)

    def _render_evidence_report(self, report: dict[str, Any], path: Path) -> None:
        if __package__ in (None, ""):
            from desktop.ui_viewers import evidence_window
        else:
            from .ui_viewers import evidence_window
        self._replace_window("_evidence_window")
        self._evidence_window = evidence_window(self._viewer_context(), report, path)


def _initial_geometry(root: Any, scale: float) -> str:
    """The token default size, shrunk to leave room for the taskbar and title bar on small screens."""
    side, top = int(WIN_LAYOUT["screenMarginX"] * scale), int(WIN_LAYOUT["screenMarginY"] * scale)
    width = min(int(LAYOUT["windowDefaultWidth"] * scale),
                max(root.winfo_screenwidth() - side, int(LAYOUT["windowMinWidth"] * scale)))
    height = min(int(LAYOUT["windowDefaultHeight"] * scale),
                 max(root.winfo_screenheight() - top, int(LAYOUT["windowMinHeight"] * scale)))
    return f"{width}x{height}"


def run_self_test() -> int:
    """Validate the bundled replay and the Tcl/Tk runtime without creating a window."""
    try:
        model = ReplayModel.from_path(DEFAULT_RUN)
        if model.frame_count < 1 or not model.frame.agents:
            raise RuntimeError("bundled replay is empty")
        # A frozen build without its Tcl/Tk library files fails here, not at the
        # first window. tkinter.Tcl() starts an interpreter and opens nothing.
        import tkinter

        tkinter.Tcl().eval("info patchlevel")
    except Exception as exc:
        print(f"self-test failed: {exc}")
        return 1
    print("self-test ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="canonical S7 run.json")
    parser.add_argument("--scenario-run", type=Path, help="open a generated zephyr-s7-scenario-run-1 JSON in the read-only viewer")
    parser.add_argument("--headless", action="store_true", help="print the replay summary without opening Tk")
    parser.add_argument("--self-test", action="store_true", help="validate startup dependencies and exit")
    parser.add_argument("--appearance", choices=("system", "dark", "light"), help="use this appearance for this session only")
    parser.add_argument("--smoke-ready", type=Path, help="write a readiness marker after the GUI is constructed")
    parser.add_argument("--smoke-step", action="store_true", help="step one replay frame before writing the readiness marker")
    parser.add_argument("--smoke-health", type=Path, help="write a second marker after the GUI remains healthy in its event loop")
    parser.add_argument("--smoke-evidence", type=Path, action="append",
            help="open and validate an evidence report during frozen GUI smoke; repeat for parity checks")
    parser.add_argument("--smoke-idle", type=Path, help="run the real event loop, then write whether the idle window stopped rendering")
    parser.add_argument("--open-results", choices=("delay", "atlas"), help="open the Results window on this view (screenshots)")
    parser.add_argument("--results-select", metavar="SPEC",
            help="select a chart point or atlas cell, e.g. 'path=turn,loss=independent,level=0.3,arm=recovery,delay=0.2'")
    args = parser.parse_args(argv)
    results_select = None
    if args.results_select:
        if __package__ in (None, ""):
            from desktop.results_data import parse_select_spec
        else:
            from .results_data import parse_select_spec
        try:
            results_select = parse_select_spec(args.results_select)
        except ValueError as exc:
            parser.error(f"--results-select: {exc}")
    if args.self_test:
        return run_self_test()
    smoke_mode = any((args.smoke_ready, args.smoke_step, args.smoke_health, args.smoke_evidence, args.smoke_idle))
    if args.headless:
        try:
            if args.scenario_run:
                print(json.dumps(load_scenario_run(args.scenario_run).summary(), indent=2, sort_keys=True))
            else:
                print(json.dumps(ReplayModel.from_path(args.run).summary(), indent=2, sort_keys=True))
        except Exception as exc:
            target = Path(args.scenario_run or args.run)
            noun = WIN_TEXT["scenarioNoun"] if args.scenario_run else "replay"
            print(open_error_text(target, exc if target.is_file() else None, noun), file=sys.stderr)
            return 2
        return 0
    try:
        try:
            model = ReplayModel.from_path(args.run)
        except (ValueError, OSError, RecursionError) as exc:
            # Checked before Tk loads. A missing or broken --run file is the user's input, not a crash: one plain sentence, exit 2.
            run_path = Path(args.run)
            show_plain_startup_error(open_error_text(run_path, exc if run_path.is_file() else None, "replay"))
            return 2
        try:
            import tkinter as tk
        except ImportError as exc:
            raise RuntimeError("Tkinter is required. Install Python with Tcl/Tk support on Windows.") from exc
        if sys.platform.startswith("win"):
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except (AttributeError, OSError):
                pass
        root = tk.Tk()
        set_window_icon(root)
        root.withdraw()
        app = WindowsReplayApp(root, model, appearance=args.appearance, persist_appearance=args.appearance is None)
        # In a --smoke-* run nobody can click a dialog, so errors go to stderr.
        app._smoke_mode = smoke_mode
        root.geometry(_initial_geometry(root, app.scale))
        root.deiconify()
        app._apply_title_bar(root, app.theme.dark)
        if args.scenario_run and not app._show_scenario_run(args.scenario_run):
            root.destroy()
            return 2
        if args.open_results or results_select:
            try:
                app._show_results(args.open_results or "delay", results_select)
                if args.open_results and app._results is not None:
                    app._results.stand_alone()
            except ValueError as exc:
                print(f"--results-select: {exc}", file=sys.stderr)
        if args.smoke_step:
            app._step()
        if args.smoke_evidence:
            evidence_kinds = []
            for evidence_path in args.smoke_evidence:
                if not app._show_evidence_report(evidence_path):
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass
                    return 2
                if app._evidence_report and app._evidence_report.get("kind"):
                    evidence_kinds.append(str(app._evidence_report["kind"]))
            evidence_kind = ",".join(sorted(set(evidence_kinds))) or None
        else:
            evidence_kind = None
        frame_text = f"frame={app.session.frame_index + 1}/{app.session.frame_count}"
        if args.smoke_ready:
            suffix = f" evidence_kind={evidence_kind}" if evidence_kind else ""
            args.smoke_ready.write_text(f"ready {frame_text}{suffix}\n", encoding="utf-8")
        if args.smoke_idle:
            # Real event loop check: after start-up settles, an idle window must
            # stop rendering. A selection feedback loop would keep counting.
            def check_idle() -> None:
                baseline = app.render_count + app.scene.draw_count

                def finish() -> None:
                    renders = app.render_count + app.scene.draw_count - baseline
                    word = "idle" if renders == 0 else "busy"
                    args.smoke_idle.write_text(f"{word} renders={renders} {frame_text}\n", encoding="utf-8")
                    root.destroy()

                root.after(1000, finish)

            root.after(1500, check_idle)
            root.mainloop()
            return 0
        if args.smoke_ready or args.smoke_health:
            # Hosted Windows runners can terminate a Tk event loop immediately,
            # and calling ``update`` in that state can make the frozen process
            # exit with code 1 without a useful stderr stream. The smoke path
            # therefore uses the successful Tk construction above as its GUI
            # check, then stays alive with a bounded sleep so the validator can
            # observe both markers. A guarded idle flush exercises pending Tk
            # layout work without entering the event loop.
            try:
                root.update_idletasks()
            except tk.TclError:
                pass
            smoke_started = time.monotonic()
            smoke_deadline = smoke_started + 3.0
            health_written = False
            while time.monotonic() < smoke_deadline:
                elapsed = time.monotonic() - smoke_started
                if args.smoke_health and not health_written and elapsed >= 0.25:
                    suffix = f" evidence_kind={evidence_kind}" if evidence_kind else ""
                    args.smoke_health.write_text(f"healthy {frame_text}{suffix}\n", encoding="utf-8")
                    health_written = True
                time.sleep(0.02)
            try:
                root.destroy()
            except tk.TclError:
                pass
        else:
            # An update helper cannot show anything itself (no console, and this
            # app had already exited), so it leaves a note. Show it once.
            previous_failure = _take_previous_update_failure()
            if previous_failure:
                root.after(300, lambda: _show_update_failure(root, previous_failure))
            root.mainloop()
        return 0
    except Exception as exc:
        log_path = write_startup_diagnostic(exc)
        if smoke_mode:
            print(f"startup failed: {type(exc).__name__}: {exc} (log: {log_path})", file=sys.stderr)
        else:
            show_startup_failure(exc, log_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
