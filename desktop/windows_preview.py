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
    from desktop.design_tokens import LAYOUT, SHORTCUTS, SPACE, UI_TEXT
    from desktop.presentation import filter_group_rows, pinned_levels, run_name
    from desktop.replay import ReplayModel
    from desktop.replay_session import SPEEDS, ReplaySession
    from desktop.scenario_run import load_scenario_run
    from desktop import ui_format as fmt
    from desktop.ui_settings import load_appearance, resolve_theme, save_appearance, system_prefers_dark, system_reduced_motion
else:
    from .design_tokens import LAYOUT, SHORTCUTS, SPACE, UI_TEXT
    from .presentation import filter_group_rows, pinned_levels, run_name
    from .replay import ReplayModel
    from .replay_session import SPEEDS, ReplaySession
    from .scenario_run import load_scenario_run
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
    """Read the version embedded by the Windows release build."""
    version_file = _resource_path("VERSION.txt")
    try:
        version = version_file.read_text(encoding="utf-8").strip().removeprefix("v")
    except OSError:
        version = ""
    return version or "0.1.0"


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
TREE_KEYS = frozenset({"left", "right", "home"})


def is_text_input_class(widget_class: str) -> bool:
    """True for Tk widgets where the user types text (so bare-key shortcuts must stand down)."""
    return widget_class in TEXT_INPUT_CLASSES


def shortcut_blocked(widget_class: str, key: str) -> bool:
    """Whether a single-key shortcut must yield to the focused widget's own key handling.

    Text fields own every key. A focused button owns Space (it presses the
    button). A focused Treeview owns Left, Right, and Home (collapse, expand,
    first row), which keeps list navigation native.
    """
    if is_text_input_class(widget_class):
        return True
    if key == "space" and widget_class in BUTTON_CLASSES:
        return True
    return widget_class == "Treeview" and key in TREE_KEYS


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
def group_key(row: Mapping[str, Any]) -> str:
    """Stable Treeview id for a group: one row per status level and word."""
    return f"group:{row['level']}:{row['word']}"


def team_rows(
    groups: Sequence[Mapping[str, Any]],
    per_drone: Sequence[Mapping[str, Any]],
    query: str = "",
) -> list[dict[str, Any]]:
    """Plan the team list: one row per group, children per drone, filter applied.

    The presentation filter (``filter_group_rows``) never removes a late or
    lost group. Any other group keeps only its matching drones while a filter
    is active, and the label counts them against the whole group.
    """
    records = {str(record["drone_id"]): record for record in per_drone}
    rows = []
    for group in filter_group_rows(groups, query):
        members = [str(drone) for drone in group.get("drone_ids", [])]
        total = int(group.get("group_count", group.get("count", len(members))))
        label = fmt.group_label(total) if len(members) == total else f"{len(members)} of {total}"
        children = []
        for drone in members:
            record = records.get(drone, {})
            children.append({
                "iid": drone,
                "label": str(record.get("compact_name", drone)),
                "values": (
                    "",
                    fmt.format_value(record.get("target_link_delay_ms"), "ms") or fmt.EN_DASH,
                    fmt.format_value(record.get("target_packet_loss_pct"), "pct") or fmt.EN_DASH,
                    "",
                ),
            })
        rows.append({
            "iid": group_key(group),
            "level": str(group["level"]),
            "label": label,
            "members": members,
            "values": (
                fmt.status_word(str(group["level"]), group.get("word")),
                fmt.format_value(group.get("target_link_delay_ms"), "ms") or fmt.EN_DASH,
                fmt.format_value(group.get("target_packet_loss_pct"), "pct") or fmt.EN_DASH,
                fmt.format_value(group.get("target_age_s"), "s") or fmt.EN_DASH,
            ),
            "children": children,
        })
    return rows


def desired_team_selection(rows: Sequence[Mapping[str, Any]], selected: str, open_groups: set[str]) -> str | None:
    """Which row should look selected: the drone if its group is open, else the group."""
    for row in rows:
        if selected in row["members"]:
            return selected if row["iid"] in open_groups else str(row["iid"])
    return None


def drone_for_team_row(rows: Sequence[Mapping[str, Any]], iid: str, selected: str) -> str | None:
    """Map a selected Treeview row back to a drone id (a group keeps the current drone)."""
    for row in rows:
        if iid == row["iid"]:
            members = row["members"]
            if selected in members:
                return selected
            return members[0] if members else None
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
    def _build(self, scene_class: Any) -> None:
        tk, ttk, root = self.tk, self.ttk, self.root
        root.title("Zephyr Mesh")
        root.minsize(self.px(LAYOUT["windowMinWidth"]), self.px(LAYOUT["windowMinHeight"]))
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
        ttk.Frame(root, style="Line.TFrame", height=1).grid(row=1, column=0, sticky="ew")
        self._build_strip(root).grid(row=2, column=0, sticky="ew")
        ttk.Frame(root, style="Line.TFrame", height=1).grid(row=3, column=0, sticky="ew")
        self._build_middle(root, scene_class).grid(row=4, column=0, sticky="nsew")
        ttk.Frame(root, style="Line.TFrame", height=1).grid(row=5, column=0, sticky="ew")
        self._build_timeline(root).grid(row=6, column=0, sticky="ew")
        ttk.Frame(root, style="Line.TFrame", height=1).grid(row=7, column=0, sticky="ew")
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

    def _build_title(self, parent: Any) -> Any:
        ttk = self.ttk
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=(self.px(SPACE["lg"]), self.px(SPACE["md"])))
        ttk.Label(frame, text="Zephyr Mesh", style="Surface.Title.TLabel").pack(side="left")
        ttk.Label(frame, text=run_name(self.run), style="Surface.Secondary.TLabel").pack(
            side="left", padx=(self.px(SPACE["md"]), 0), pady=(self.px(3), 0))
        ttk.Label(frame, text=UI_TEXT["honesty"]["badge"], style="Badge.TLabel").pack(
            side="left", padx=(self.px(SPACE["md"]), 0), pady=(self.px(3), 0))
        about = ttk.Button(frame, text=f" {UI_TEXT['honesty']['aboutButton']}", style="Tool.TButton", compound="left",
                           image=self.images.icon("info", "textSecondary", 14), command=self._show_about)
        about.pack(side="right")
        self.notice = ttk.Label(frame, text="", style="Surface.Caption.TLabel")
        self.notice.pack(side="right", padx=(0, self.px(SPACE["md"])))
        return frame

    def _build_strip(self, parent: Any) -> Any:
        ttk = self.ttk
        frame = ttk.Frame(parent, style="Base.TFrame", padding=(self.px(SPACE["sm"]), self.px(SPACE["md"])))
        labels = UI_TEXT["statusStrip"]
        self.strip_values: dict[str, Any] = {}
        column = 0
        for index, label in enumerate(labels):
            if index:
                ttk.Frame(frame, style="Line.TFrame", width=1).grid(row=0, column=column, sticky="ns", pady=self.px(2))
                column += 1
            cell = ttk.Frame(frame, style="Base.TFrame", padding=(self.px(SPACE["md"]), 0))
            cell.grid(row=0, column=column, sticky="nsew")
            frame.columnconfigure(column, weight=2 if index == 0 else 1, uniform="strip")
            ttk.Label(cell, text=label, style="Base.Caption.TLabel").pack(anchor="w")
            row = ttk.Frame(cell, style="Base.TFrame", height=self.fonts["metric"].metrics("linespace") + self.px(2))
            row.pack(anchor="w", fill="x", pady=(self.px(2), 0))
            row.pack_propagate(False)
            if index == 0:
                glyph = ttk.Label(row, style="Base.Glyph.TLabel")
                glyph.pack(side="left", padx=(0, self.px(SPACE["sm"])))
                self.strip_glyph = glyph
            # Fixed character widths keep the layout still while values change frame to frame.
            value = ttk.Label(row, style="Base.Metric.TLabel", anchor="w")
            value.pack(side="left", fill="both", expand=True)
            self.strip_values[label] = value
            column += 1
        # A fixed-height strip keeps value changes from re-laying out the whole window.
        caption, metric = self.fonts["caption"], self.fonts["metric"]
        height = caption.metrics("linespace") + metric.metrics("linespace") + self.px(2 + 2 * SPACE["md"] + 4)
        frame.configure(height=height)
        frame.grid_propagate(False)
        return frame

    def _build_middle(self, parent: Any, scene_class: Any) -> Any:
        ttk = self.ttk
        frame = ttk.Frame(parent, style="Base.TFrame")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(2, weight=1, minsize=self.px(LAYOUT["sceneMinWidth"]))
        self._build_team(frame).grid(row=0, column=0, sticky="nsw")
        ttk.Frame(frame, style="Line.TFrame", width=1).grid(row=0, column=1, sticky="ns")
        self._build_scene(frame, scene_class).grid(row=0, column=2, sticky="nsew")
        self.inspector_line = ttk.Frame(frame, style="Line.TFrame", width=1)
        self.inspector_line.grid(row=0, column=3, sticky="ns")
        self.inspector_panel = self._build_inspector(frame)
        self.inspector_panel.grid(row=0, column=4, sticky="nse")
        return frame

    def _build_team(self, parent: Any) -> Any:
        ttk, tk = self.ttk, self.tk
        width = self.px(LAYOUT["sidebarWidthWindows"])
        frame = ttk.Frame(parent, style="Surface.TFrame", width=width, padding=(self.px(SPACE["md"]), self.px(SPACE["md"])))
        frame.grid_propagate(False)
        frame.pack_propagate(False)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        self.team_summary = ttk.Label(frame, style="Surface.Strong.TLabel", wraplength=width - self.px(2 * SPACE["md"]), width=36)
        self.team_summary.grid(row=0, column=0, columnspan=2, sticky="w")
        self.filter_entry = ttk.Entry(frame, textvariable=self.filter_var, style="Filter.TEntry", font=self.fonts["body"])
        self.filter_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(self.px(SPACE["sm"]), self.px(SPACE["sm"])))
        self.widgets.Placeholder(self.filter_entry, self.filter_var, f"Filter drones ({fmt.accelerator('focusFilter')})",
                                 "Field.Placeholder.TLabel")
        self.filter_var.trace_add("write", lambda *_args: self._on_filter())
        self.filter_entry.bind("<Escape>", self._clear_filter)
        columns = ("status", "delay", "loss", "age")
        tree = ttk.Treeview(frame, columns=columns, show="tree headings", selectmode="browse", style="Zephyr.Treeview", takefocus=True)
        inner = width - self.px(2 * SPACE["md"]) - self.px(12)
        fixed = {"status": self.px(56), "delay": self.px(48), "loss": self.px(60), "age": self.px(46)}
        widths = {"#0": inner - sum(fixed.values()), **fixed}
        headings = {"#0": "Drones", "status": "Status", "delay": "Delay", "loss": "Loss", "age": "Age"}
        for column in ("#0",) + columns:
            tree.heading(column, text=headings[column], anchor="w" if column in ("#0", "status") else "e")
            tree.column(column, width=widths[column], minwidth=widths[column], stretch=column == "#0",
                        anchor="w" if column in ("#0", "status") else "e")
        scroll = ttk.Scrollbar(frame, orient="vertical", style="Thin.Vertical.TScrollbar", command=tree.yview)
        tree.configure(yscrollcommand=self._auto_scroll(scroll, row=2, column=1))
        tree.grid(row=2, column=0, sticky="nsew")
        tree.bind("<<TreeviewSelect>>", self._on_select)
        tree.bind("<<TreeviewOpen>>", lambda _event: self._on_group_toggle(True))
        tree.bind("<<TreeviewClose>>", lambda _event: self._on_group_toggle(False))
        tree.bind("<Double-Button-1>", self._on_team_double)
        tree.bind("<Return>", lambda _event: self._focus_selected_drone())
        for sequence in self._context_sequences():
            tree.bind(sequence, self._on_team_context)
        self.fleet = tree
        self.team_hover = self.widgets.TreeHover(tree)
        return frame

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
        modes = ttk.Frame(frame, style="Base.TFrame")
        for value, text in (("3d", "3D"), ("top", "Top")):
            ttk.Radiobutton(modes, text=text, value=value, variable=self.view_var, style="Segment.Toolbutton",
                            command=lambda: self._set_view_mode(self.view_var.get()), takefocus=True).pack(side="left")
        modes.place(in_=self.canvas, x=self.px(SPACE["md"]), y=self.px(SPACE["md"]))
        return frame

    def _build_inspector(self, parent: Any) -> Any:
        ttk = self.ttk
        width = self.px(LAYOUT["inspectorWidth"])
        frame = ttk.Frame(parent, style="Surface.TFrame", width=width)
        frame.grid_propagate(False)
        frame.pack_propagate(False)
        scroll = self.widgets.ScrollFrame(frame)
        scroll.outer.pack(fill="both", expand=True)
        self.inspector_scroll = scroll
        self._theme_hooks.append(scroll.apply_theme)
        pad = self.px(SPACE["lg"])
        body = ttk.Frame(scroll.inner, style="Surface.TFrame", padding=(pad, self.px(SPACE["md"]), pad, pad))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        header = ttk.Frame(body, style="Surface.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        self.inspector_title = ttk.Label(header, style="Surface.Headline.TLabel")
        self.inspector_title.pack(side="left")
        self.inspector_word = ttk.Label(header, style="Surface.Body.TLabel", compound="left", width=10, anchor="e")
        self.inspector_word.pack(side="right")
        tiles = ttk.Frame(body, style="Surface.TFrame")
        tiles.grid(row=1, column=0, sticky="ew", pady=(self.px(SPACE["md"]), 0))
        tiles.columnconfigure(0, weight=1, uniform="tiles")
        tiles.columnconfigure(1, weight=1, uniform="tiles")
        self.tiles = {}
        captions = (("target_age", "Target age"), ("estimate", "Estimate error"),
                    ("delay", fmt.field_label("target_link_delay")), ("loss", fmt.field_label("target_packet_loss")))
        gap = self.px(SPACE["sm"])
        for index, (key, caption) in enumerate(captions):
            tile = self.widgets.MetricTile(tiles, caption, self.scale)
            tile.frame.grid(row=index // 2, column=index % 2, sticky="nsew",
                            padx=(0, gap) if index % 2 == 0 else (0, 0), pady=(0, gap))
            self.tiles[key] = tile
        label_width = self.px(118)
        wrap = width - 2 * pad - label_width - self.px(SPACE["md"] + 6)
        ttk.Label(body, text=UI_TEXT["inspector"]["positionAndMotion"], style="Surface.Strong.TLabel").grid(row=2, column=0, sticky="w",
                pady=(self.px(SPACE["md"]), self.px(SPACE["xs"])))
        self.motion_grid = self.widgets.KeyValueGrid(body, label_width=label_width, wrap=wrap, scale=self.scale, value_width=14)
        self.motion_grid.frame.grid(row=3, column=0, sticky="ew")
        ttk.Label(body, text=UI_TEXT["inspector"]["neighbors"], style="Surface.Strong.TLabel").grid(row=4, column=0, sticky="w",
                pady=(self.px(SPACE["md"]), self.px(SPACE["xs"])))
        self.neighbor_grid = self.widgets.KeyValueGrid(body, label_width=label_width, wrap=wrap, scale=self.scale, value_width=14)
        self.neighbor_grid.frame.grid(row=5, column=0, sticky="ew")
        ttk.Frame(body, style="Line.TFrame", height=1).grid(row=6, column=0, sticky="ew", pady=(self.px(SPACE["md"]), self.px(SPACE["xs"])))
        self.details = self.widgets.Disclosure(
            body, UI_TEXT["inspector"]["details"], self.images, pad=self.px(SPACE["xs"]),
            on_toggle=lambda opened: opened and self.root.after_idle(lambda: self.inspector_scroll.reveal(self.details.frame)))
        self.details.frame.grid(row=7, column=0, sticky="ew")
        self.details_grid = self.widgets.KeyValueGrid(self.details.body, wrap=width - 2 * pad - self.px(8), scale=self.scale, stacked=True)
        self.details_grid.frame.pack(fill="x", padx=(self.px(4), 0))
        return frame

    def _build_timeline(self, parent: Any) -> Any:
        ttk = self.ttk
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=(self.px(SPACE["lg"]), self.px(SPACE["sm"])))
        frame.columnconfigure(4, weight=1)
        transport = (
            ("restart", "Restart", self._restart, fmt.accelerator("restart")),
            ("step_back", "Step back", self._step_back, fmt.accelerator("stepBack")),
        )
        column = 0
        for icon, label, command, accel in transport:
            button = ttk.Button(frame, style="Icon.TButton", image=self.images.icon(icon, "text", 14), command=command)
            button.grid(row=0, column=column, padx=(0, self.px(SPACE["xs"])))
            self.widgets.Tooltip(button, f"{label}  ({accel})", lambda: self.theme, self.fonts["caption"])
            column += 1
        self.play_button = ttk.Button(frame, style="Tool.TButton", compound="left", width=7,
                                      image=self.images.icon("play", "text", 14), text=" Play", command=self._toggle)
        self.play_button.grid(row=0, column=column, padx=(0, self.px(SPACE["xs"])))
        column += 1
        forward = ttk.Button(frame, style="Icon.TButton", image=self.images.icon("step_forward", "text", 14), command=self._step)
        forward.grid(row=0, column=column)
        self.widgets.Tooltip(forward, f"Step forward  ({fmt.accelerator('stepForward')})", lambda: self.theme, self.fonts["caption"])
        column += 1
        self.slider = self.widgets.TimelineSlider(frame, self._seek_frame, self.scale)
        self.slider.canvas.grid(row=0, column=column, sticky="ew", padx=(self.px(SPACE["lg"]), self.px(SPACE["lg"])))
        self._theme_hooks.append(self.slider.apply_theme)
        column += 1
        self.time_label = ttk.Label(frame, style="Surface.Body.TLabel", width=12, anchor="e")
        self.time_label.grid(row=0, column=column, padx=(0, self.px(SPACE["lg"])))
        column += 1
        speeds = ttk.Frame(frame, style="Surface.TFrame")
        speeds.grid(row=0, column=column)
        ttk.Label(speeds, text="Speed", style="Surface.Caption.TLabel").pack(side="left", padx=(0, self.px(SPACE["sm"])))
        for speed in SPEEDS:
            ttk.Radiobutton(speeds, text=speed_label(speed), value=speed, variable=self.speed_var, style="Segment.Toolbutton",
                            command=lambda: self._set_speed(self.speed_var.get())).pack(side="left")
        return frame

    def _build_events(self, parent: Any) -> Any:
        ttk = self.ttk
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=(self.px(SPACE["lg"]), self.px(SPACE["sm"]),
                self.px(SPACE["lg"]), self.px(SPACE["sm"])))
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="Events", style="Surface.Caption.TLabel").grid(row=0, column=0, sticky="w", pady=(0, self.px(SPACE["xs"])))
        tree = ttk.Treeview(frame, columns=("word", "time", "text"), show="tree", selectmode="browse", height=4,
                            style="Events.Treeview", takefocus=True)
        tree.column("#0", width=self.px(30), minwidth=self.px(30), stretch=False)
        tree.column("word", width=self.px(76), minwidth=self.px(76), stretch=False, anchor="w")
        tree.column("time", width=self.px(64), minwidth=self.px(64), stretch=False, anchor="w")
        tree.column("text", width=self.px(600), stretch=True, anchor="w")
        scroll = ttk.Scrollbar(frame, orient="vertical", style="Thin.Vertical.TScrollbar", command=tree.yview)
        tree.configure(yscrollcommand=self._auto_scroll(scroll, row=1, column=1))
        tree.grid(row=1, column=0, sticky="ew")
        tree.bind("<<TreeviewSelect>>", self._on_event_select)
        self.events_tree = tree
        self.events_hover = self.widgets.TreeHover(tree)
        return frame

    def _auto_scroll(self, scrollbar: Any, *, row: int, column: int) -> Callable[[str, str], None]:
        state = {"shown": False}

        def update(first: str, last: str) -> None:
            scrollbar.set(first, last)
            needed = float(first) > 0.0 or float(last) < 1.0
            if needed != state["shown"]:
                state["shown"] = needed
                if needed:
                    scrollbar.grid(row=row, column=column, sticky="ns", padx=(self.px(2), 0))
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
                self.root.bind(shortcut.tk_binding, lambda _event, handler=handler: (handler(), "break")[1])

    def _shortcut(self, action: Callable[[], Any], key: str = "") -> Callable[[Any], Any]:
        """Wrap a single-key shortcut so it stands down while a text field (or owning widget) has focus."""
        def handler(event: Any) -> Any:
            widget = getattr(event, "widget", None)
            try:
                widget_class = widget.winfo_class() if widget is not None else ""
            except Exception:
                widget_class = ""
            if shortcut_blocked(widget_class, key):
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
        self.scene.apply_theme(theme)
        alive = []
        for hook in self._theme_hooks:
            try:
                hook(theme)
                alive.append(hook)
            except self.tk.TclError:
                continue
        self._theme_hooks = alive
        for window in (self.root, self._about_window, self._evidence_window, self._scenario_window):
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
        self._set(self.strip_glyph, image=self.images.status(headline["level"], 15, 16))
        self._set(self.strip_values[labels[0]], text=headline["text"], style="Base.Metric.TLabel")
        strip = presentation["strip"]
        self._set_metric(labels[1], fmt.format_value(presentation.get("estimate_error_p95_m"), "m"))
        self._set_metric(labels[2], fmt.format_value(strip.get("target_link_delay_ms"), "ms"))
        self._set_metric(labels[3], fmt.format_value(strip.get("target_packet_loss_pct"), "pct"))
        self._set_metric(labels[4], fmt.format_age_span(strip.get("newest_target_age_s"), strip.get("oldest_target_age_s")))

    def _render_fleet(self) -> None:
        tree = self.fleet
        presentation = self.session.presentation
        self._set(self.team_summary, text=presentation["team_summary"])
        plan = team_rows(presentation["groups"], presentation["per_drone"], self.filter_var.get())
        filtering = bool(self.filter_var.get().strip())
        wanted_groups = [row["iid"] for row in plan]
        for row in plan:
            if row["iid"] not in self._seen_groups:
                self._seen_groups.add(row["iid"])
                if row["level"] != "ok":
                    self._open_groups.add(row["iid"])
        existing_groups = set(tree.get_children(""))
        wanted_children = {child["iid"] for row in plan for child in row["children"]}
        for row in plan:
            image = self.images.status(row["level"], 10, 16)
            if tree.exists(row["iid"]):
                self._tree_item(row["iid"], text=row["label"], values=row["values"], image=image)
            else:
                tree.insert("", "end", iid=row["iid"], text=row["label"], values=row["values"], image=image)
        for index, row in enumerate(plan):
            if tree.parent(row["iid"]) != "" or tree.index(row["iid"]) != index:
                tree.move(row["iid"], "", index)
            for child_index, child in enumerate(row["children"]):
                if tree.exists(child["iid"]):
                    self._tree_item(child["iid"], text=child["label"], values=child["values"])
                    if tree.parent(child["iid"]) != row["iid"] or tree.index(child["iid"]) != child_index:
                        tree.move(child["iid"], row["iid"], child_index)
                else:
                    tree.insert(row["iid"], child_index, iid=child["iid"], text=child["label"], values=child["values"])
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
        self._sync_team_selection()

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
        """Make the tree selection match the session. Only calls selection_set when it differs."""
        desired = desired_team_selection(self._team_plan, self.session.selected_drone_id, self._open_now())
        current = tuple(self.fleet.selection())
        if desired is None:
            if current:
                self.fleet.selection_set(())
            return
        if current != (desired,):
            self.fleet.selection_set(desired)
            self.fleet.focus(desired)
        if reveal:
            self.fleet.see(desired)

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

    def _on_team_double(self, event: Any) -> str | None:
        row = self.fleet.identify_row(event.y)
        if row and row in {child["iid"] for plan in self._team_plan for child in plan["children"]}:
            self._focus_drone(row)
            return "break"
        return None

    def _on_team_context(self, event: Any) -> str:
        row = self.fleet.identify_row(event.y)
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
        finally:
            menu.grab_release()

    def _show_in_inspector(self, drone_id: str) -> None:
        self._select_drone(drone_id)
        if not self.session.show_inspector:
            self._set_inspector(True)

    def _copy_text(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _profile(self, profile_id: str | None) -> Mapping[str, Any] | None:
        return next((item for item in self.run.get("agent_profiles", []) if item.get("profile_id") == profile_id), None)

    def _render_inspector(self) -> None:
        record = self.session.record_for(self.session.selected_drone_id)
        if record is None:
            self.inspector_title.configure(text=UI_TEXT["inspector"]["emptyTitle"])
            self.inspector_word.configure(text="", image="")
            for tile in self.tiles.values():
                tile.set(None)
            self.motion_grid.set_rows(())
            self.neighbor_grid.set_rows(())
            self.details_grid.set_rows(())
            return
        self._set(self.inspector_title, text=record["heading_name"])
        self._set(self.inspector_word, text=f" {record['word']}", image=self.images.status(record["level"], 10, 12))
        self.tiles["target_age"].set(fmt.format_value(record.get("target_age_s"), "s"))
        self.tiles["estimate"].set(fmt.format_value(record.get("estimate_error_m"), "m"))
        self.tiles["delay"].set(fmt.format_value(record.get("target_link_delay_ms"), "ms"))
        self.tiles["loss"].set(fmt.format_value(record.get("target_packet_loss_pct"), "pct"))
        heading = record.get("heading_deg")
        self.motion_grid.set_rows((
            (UI_TEXT["inspector"]["altitude"], fmt.format_value(record.get("altitude_m"), "m")),
            (fmt.field_label("speed_mps"), fmt.format_value(record.get("speed_mps"), "mps")),
            (fmt.field_label("heading_deg"), fmt.format_value(heading, "deg") if heading is not None else "At rest"),
        ))
        self.neighbor_grid.set_rows((
            (fmt.field_label("neighbor_count"), fmt.format_scalar(record.get("neighbor_count"))),
            (fmt.field_label("nearest_neighbor_m"), fmt.format_value(record.get("nearest_neighbor_m"), "m")),
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
        self._set(self.play_button, text=" Pause" if playing else " Play",
                  image=self.images.icon("pause" if playing else "play", "text", 14))
        if abs(self.speed_var.get() - session.speed) > 1e-9:
            self.speed_var.set(session.speed)

    def _render_events(self) -> None:
        tree = self.events_tree
        events = self.session.events_through()
        signature = tuple((frame, event["text"]) for frame, event in events)
        if signature == self._event_signature:
            return
        keep = 0
        while keep < min(len(signature), len(self._event_signature)) and signature[keep] == self._event_signature[keep]:
            keep += 1
        for index in range(keep, len(self._event_signature)):
            iid = f"event{index}"
            if tree.exists(iid):
                self.events_hover.forget(iid)
                tree.delete(iid)
                self._event_frames.pop(iid, None)
        for index in range(keep, len(events)):
            frame, event = events[index]
            level, word, time_text, text = fmt.event_row(event)
            iid = f"event{index}"
            tree.insert("", "end", iid=iid, image=self.images.status(level, 11, 18), values=(word, time_text, text))
            self._event_frames[iid] = frame
        self._event_signature = signature
        children = tree.get_children("")
        if children:
            tree.see(children[-1])

    def _on_event_select(self, _event: Any = None) -> None:
        selection = self.events_tree.selection()
        if not selection:
            return
        frame = self._event_frames.get(selection[0])
        if frame is None or frame == self.session.frame_index:
            return
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

    def _toggle_inspector(self) -> None:
        self._set_inspector(not self.session.show_inspector)

    def _focus_filter(self) -> None:
        self.filter_entry.focus_set()
        self.filter_entry.select_range(0, "end")

    def _clear_filter(self, _event: Any = None) -> str:
        self.filter_var.set("")
        self.fleet.focus_set()
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

    def _about_details(self) -> list[tuple[str, Any, str]]:
        source = getattr(self.model, "source_path", None)
        digest = None
        if source is not None:
            try:
                digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            except OSError:
                digest = None
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
        self.notice.configure(text="Checking for updates...")

        def report(message: str, error: bool = False) -> None:
            from tkinter import messagebox

            self.notice.configure(text="")
            if error:
                messagebox.showwarning("Check for Updates", message, parent=self.root)
            else:
                messagebox.showinfo("Check for Updates", message, parent=self.root)

        def worker() -> None:
            error = False
            try:
                try:
                    from tools.release_updater import GitHubReleaseClient
                except ModuleNotFoundError:
                    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                    from tools.release_updater import GitHubReleaseClient
                client = GitHubReleaseClient()
                release, asset, newer = client.check(CURRENT_VERSION, platform=client.platform_name())
                if not newer:
                    message = f"Zephyr Mesh is up to date ({release.tag})."
                elif not getattr(sys, "frozen", False):
                    message = f"Update available: {release.tag}. Installing needs the packaged app."
                else:
                    self.root.after(0, lambda: self.notice.configure(text=f"Downloading {release.tag}..."))
                    staged = client.download_and_stage(asset)
                    install_dir = Path(sys.executable).resolve().parent
                    updater_candidates = (
                        install_dir / "ZephyrMeshUpdater.exe",
                        install_dir.parent / "ZephyrMeshUpdater.exe",
                    )
                    updater = next((candidate for candidate in updater_candidates if candidate.is_file()), updater_candidates[0])
                    launch = install_dir / "ZephyrMeshWindows.exe"
                    if not updater.is_file():
                        message = "The update was downloaded, but ZephyrMeshUpdater.exe is missing."
                        error = True
                    else:
                        updater = updater_launch_path(updater, install_dir, pid=os.getpid())
                        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        subprocess.Popen(
                            [str(updater), "--pid", str(os.getpid()), "--staged", str(staged.payload), "--install-dir",
                                    str(install_dir), "--launch", str(launch)],
                            creationflags=flags,
                            close_fds=True,
                        )
                        self.root.after(0, lambda: self.notice.configure(text=f"Installing {release.tag}..."))
                        self.root.after(700, self.root.destroy)
                        return
            except Exception as exc:
                message = f"The update check could not finish: {exc}"
                error = True
            self.root.after(0, lambda: report(message, error))

        threading.Thread(target=worker, daemon=True).start()

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

        from tkinter import messagebox

        if __package__ in (None, ""):
            from desktop.scenario_run import ScenarioRunFormatError
            from desktop.ui_viewers import scenario_window
        else:
            from .scenario_run import ScenarioRunFormatError
            from .ui_viewers import scenario_window
        try:
            run = load_scenario_run(path)
        except ScenarioRunFormatError as exc:
            messagebox.showerror("Scenario run", str(exc), parent=self.root)
            return False
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Scenario run", f"Could not open run: {exc}", parent=self.root)
            return False
        self._replace_window("_scenario_window")
        self._scenario_window = scenario_window(self._viewer_context(), run, Path(path))
        return True

    def _show_evidence_report(self, path: Path) -> bool:
        """Decode and display an evidence envelope without touching replay state."""

        from tkinter import messagebox

        loader, evidence_error = self._evidence_loader()
        try:
            report = loader(Path(path))
        except evidence_error as exc:
            messagebox.showerror("Evidence report", str(exc), parent=self.root)
            return False
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Evidence report", f"Could not open report: {exc}", parent=self.root)
            return False
        if not isinstance(report, dict):
            messagebox.showerror("Evidence report", "The report decoder returned an invalid object.", parent=self.root)
            return False
        self._evidence_report = report
        self._evidence_path = Path(path)
        self._render_evidence_report(report, self._evidence_path)
        return True

    def _render_evidence_report(self, report: dict[str, Any], path: Path) -> None:
        if __package__ in (None, ""):
            from desktop.ui_viewers import evidence_window
        else:
            from .ui_viewers import evidence_window
        self._replace_window("_evidence_window")
        self._evidence_window = evidence_window(self._viewer_context(), report, path)


def _initial_geometry(root: Any, scale: float) -> str:
    width = min(int(1440 * scale), max(root.winfo_screenwidth() - 80, int(LAYOUT["windowMinWidth"] * scale)))
    height = min(int(900 * scale), max(root.winfo_screenheight() - 120, int(LAYOUT["windowMinHeight"] * scale)))
    return f"{width}x{height}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="canonical S7 run.json")
    parser.add_argument("--scenario-run", type=Path, help="open a generated zephyr-s7-scenario-run-1 JSON in the read-only viewer")
    parser.add_argument("--headless", action="store_true", help="print the replay summary without opening Tk")
    parser.add_argument("--appearance", choices=("system", "dark", "light"), help="use this appearance for this session only")
    parser.add_argument("--smoke-ready", type=Path, help="write a readiness marker after the GUI is constructed")
    parser.add_argument("--smoke-step", action="store_true", help="step one replay frame before writing the readiness marker")
    parser.add_argument("--smoke-health", type=Path, help="write a second marker after the GUI remains healthy in its event loop")
    parser.add_argument("--smoke-evidence", type=Path, action="append",
            help="open and validate an evidence report during frozen GUI smoke; repeat for parity checks")
    parser.add_argument("--smoke-idle", type=Path, help="run the real event loop, then write whether the idle window stopped rendering")
    args = parser.parse_args(argv)
    if args.headless:
        if args.scenario_run:
            print(json.dumps(load_scenario_run(args.scenario_run).summary(), indent=2, sort_keys=True))
        else:
            print(json.dumps(ReplayModel.from_path(args.run).summary(), indent=2, sort_keys=True))
        return 0
    try:
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
        model = ReplayModel.from_path(args.run)
        root = tk.Tk()
        root.withdraw()
        app = WindowsReplayApp(root, model, appearance=args.appearance, persist_appearance=args.appearance is None)
        root.geometry(_initial_geometry(root, app.scale))
        root.deiconify()
        app._apply_title_bar(root, app.theme.dark)
        if args.scenario_run and not app._show_scenario_run(args.scenario_run):
            root.destroy()
            return 2
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
        show_startup_failure(exc, log_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
