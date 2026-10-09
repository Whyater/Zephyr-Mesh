"""Windows-friendly Tkinter replay preview for the Zephyr Mesh S7 fixture.

The app is intentionally offline and read-only. It is a desktop preview of the
canonical Python event log, not a vehicle controller or a live telemetry client.
Tkinter is imported only when ``main`` runs so the pure replay model remains
testable on machines without Tk support.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any

try:
    from .replay import ReplayModel, filter_agents
    from .scenario_run import ScenarioRunFormatError, load_scenario_run
    from .design_tokens import TOKENS
except ImportError:  # direct script execution for PyInstaller
    try:
        from desktop.replay import ReplayModel, filter_agents
        from desktop.scenario_run import ScenarioRunFormatError, load_scenario_run
        from desktop.design_tokens import TOKENS
    except ImportError:
        from replay import ReplayModel, filter_agents
        from scenario_run import ScenarioRunFormatError, load_scenario_run
        from design_tokens import TOKENS


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
    no console.  Keep the diagnostic outside the extracted application folder
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
        # directory is unavailable.  Never replace a useful startup error with
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
            # streams. Keep this reporting path best effort and never replace
            # the original startup failure with a reporting failure.
            pass
    # Source runs and unusual Windows environments can still provide a useful
    # fallback.  PyInstaller's windowed executable has no console, so this is
    # primarily for local development and test harnesses.
    try:
        stream = getattr(sys, "stderr", None)
        if stream is not None:
            print(message, file=stream)
    except Exception:
        pass


def mission_event_lines(model: Any, selected_id: str) -> tuple[str, ...]:
    """Build the compact event list shown by the macOS mission inspector.

    The Windows surface uses the same replay-only event vocabulary as the
    native cockpit, while keeping the output plain text for Tkinter.  Every
    line is derived from the current frame or manifest metadata.  It is an
    operator explanation of the fixture, not a live alert stream.
    """

    selected = next((agent for agent in model.frame.agents if agent.agent_id == selected_id), None)
    events = [
        f"INFO  Replay manifest loaded · {model.scenario_id} · seed {model.seed}",
        f"WARN  Command authority locked · {model.command_authority}",
        f"WARN  Failsafe boundary recorded · {model.failsafe}",
        (
            f"OK    Frame {model.frame.step_index} committed · "
            f"{model.frame.active_count} active vehicles · Δt {model.dt_s:.3f} s"
        ),
    ]
    minimum_confidence = min((agent.confidence for agent in model.frame.agents), default=1.0)
    if minimum_confidence < 0.98:
        events.append(f"WARN  Estimator confidence changed · minimum {minimum_confidence:.3f}")
    if selected is not None and selected.min_neighbor_distance_m > 0:
        events.append(
            f"OK    Separation margin observed · {selected.agent_id} nearest neighbor "
            f"{selected.min_neighbor_distance_m:.3f} m"
        )
    # The canonical S7 steps do not attach link events to frame rows.
    # Match the macOS worst-loss event contract with the replay-wide observed
    # envelope and label it as modeled replay evidence.
    link_losses = [
        link.loss_pct
        for agent in model.frame.agents
        if (link := model.link_stats_for(agent.agent_id)) is not None
    ]
    worst_loss = max(link_losses, default=0.0)
    if worst_loss >= 8:
        events.append(f"WARN  Link envelope visible · modeled loss {worst_loss:.1f}%")
    return tuple(events)


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


class WindowsReplayApp:
    """Small operator-facing replay surface using only the Python standard library."""

    def __init__(self, root: Any, model: ReplayModel) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.model = model
        self.playing = False
        self.selected_id = model.frame.agents[0].agent_id
        self.filter_var = None
        self.fleet_summary = None
        self.show_keepout = None
        self.canvas = None
        self.timeline = None
        self.fleet = None
        self.telemetry = None
        self.status = None
        self._disclosures: dict[str, tuple[Any, Any]] = {}
        self._evidence_report: dict[str, Any] | None = None
        self._evidence_path: Path | None = None
        self._evidence_window: Any | None = None
        self._scenario_window: Any | None = None
        self._min_x, self._max_x, self._min_y, self._max_y = self._bounds()
        self._view_center = ((self._min_x + self._max_x) / 2, (self._min_y + self._max_y) / 2)
        self._zoom = 1.0
        self._focus_selected = False
        self._build()
        self._render()

    def _build(self) -> None:
        root = self.root
        root.title("Zephyr Mesh")
        root.minsize(TOKENS.window_min_width, TOKENS.window_min_height)
        root.configure(background=TOKENS.background)
        style = self.ttk.Style(root)
        try:
            style.theme_use("vista")
        except self.tk.TclError:
            style.theme_use("clam")
        style.configure("Surface.TFrame", background=TOKENS.surface)
        style.configure("Night.TFrame", background=TOKENS.background)
        style.configure("Title.TLabel", background=TOKENS.surface, foreground=TOKENS.text, font=(TOKENS.ui_font, TOKENS.title_size, "semibold"))
        style.configure("Muted.TLabel", background=TOKENS.surface, foreground=TOKENS.muted, font=(TOKENS.ui_font, TOKENS.body_size))
        style.configure("Value.TLabel", background=TOKENS.surface, foreground=TOKENS.text, font=(TOKENS.mono_font, TOKENS.body_size))
        style.configure("Action.TButton", padding=(TOKENS.button_pad_x, TOKENS.button_pad_y))
        style.configure("Disclosure.TButton", padding=(0, TOKENS.disclosure_pad), anchor="w")
        style.configure("Fleet.Treeview", rowheight=TOKENS.fleet_row_height, font=(TOKENS.ui_font, TOKENS.body_size))
        style.configure("Fleet.Treeview.Heading", font=(TOKENS.ui_font, TOKENS.small_size, "bold"))

        menu = self.tk.Menu(root)
        file_menu = self.tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open scenario run…", command=self._open_scenario_run)
        file_menu.add_command(label="Open evidence report…", accelerator="Ctrl+O", command=self._open_evidence_report)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=root.destroy)
        menu.add_cascade(label="File", menu=file_menu)
        root.configure(menu=menu)

        header = self.ttk.Frame(root, style="Surface.TFrame", padding=(TOKENS.spacing_lg, TOKENS.header_vertical))
        header.pack(fill="x")
        self.ttk.Label(header, text="Zephyr Mesh", style="Title.TLabel").pack(side="left")
        self.ttk.Label(header, text=f"{self.model.scenario_id} · synthetic replay", style="Muted.TLabel").pack(side="right")

        body = self.ttk.Frame(root, style="Night.TFrame", padding=TOKENS.spacing_sm)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        fleet_panel = self.ttk.Frame(body, style="Surface.TFrame", padding=TOKENS.spacing_sm)
        fleet_panel.grid(row=0, column=0, sticky="nsew", padx=(0, TOKENS.spacing_sm))
        self.fleet_summary = self.ttk.Label(fleet_panel, style="Muted.TLabel")
        self.fleet_summary.pack(anchor="w")
        self.filter_var = self.tk.StringVar()
        filter_entry = self.ttk.Entry(fleet_panel, textvariable=self.filter_var, takefocus=True)
        filter_entry.pack(fill="x", pady=(TOKENS.table_top_gap, 0))
        filter_entry.bind("<KeyRelease>", lambda _event: self._on_filter())
        self.fleet = self.ttk.Treeview(fleet_panel, columns=("vehicle", "state", "battery", "link"), show="headings", selectmode="browse", style="Fleet.Treeview", takefocus=True)
        for column, label, width, anchor in (("vehicle", "Vehicle", TOKENS.fleet_column_vehicle, "w"), ("state", "State", TOKENS.fleet_column_state, "w"), ("battery", "Battery", TOKENS.fleet_column_metric, "e"), ("link", "Link", TOKENS.fleet_column_metric, "e")):
            self.fleet.heading(column, text=label)
            self.fleet.column(column, width=width, anchor=anchor, stretch=column == "vehicle")
        self.fleet.pack(fill="both", expand=True, pady=(TOKENS.table_top_gap, 0))
        self.fleet.bind("<<ListboxSelect>>", self._on_select)
        self.fleet.bind("<<TreeviewSelect>>", self._on_select)
        root.bind("<space>", lambda _event: self._toggle())
        root.bind("<Right>", lambda _event: self._step())
        root.bind("<f>", lambda _event: self._focus_selected_vehicle())
        root.bind("<F>", lambda _event: self._focus_selected_vehicle())
        root.bind("<r>", lambda _event: self._reset_view())
        root.bind("<R>", lambda _event: self._reset_view())
        root.bind("<o>", lambda _event: self._toggle_keepout())
        root.bind("<O>", lambda _event: self._toggle_keepout())
        root.bind("<Control-l>", lambda _event: (filter_entry.focus_set(), "break")[1])
        root.bind("<Control-o>", lambda _event: (self._open_evidence_report(), "break")[1])

        center = self.ttk.Frame(body, style="Night.TFrame")
        center.grid(row=0, column=1, sticky="nsew")
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)
        self.canvas = self.tk.Canvas(center, background=TOKENS.canvas, highlightthickness=0, relief="flat", takefocus=True)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._render())
        controls = self.ttk.Frame(center, style="Surface.TFrame", padding=(TOKENS.spacing_sm, TOKENS.tight_padding))
        controls.grid(row=1, column=0, sticky="ew", pady=(TOKENS.panel_gap, 0))
        controls.columnconfigure(1, weight=1)
        self.status = self.ttk.Label(controls, style="Muted.TLabel")
        self.status.grid(row=0, column=0, sticky="w")
        self.timeline = self.tk.Scale(controls, from_=0, to=max(0, self.model.frame_count - 1), orient="horizontal", showvalue=False, command=self._seek, background=TOKENS.control_background, troughcolor=TOKENS.control_track, highlightthickness=0, activebackground=TOKENS.control_active)
        self.timeline.set(0)
        self.timeline.grid(row=0, column=1, sticky="ew", padx=TOKENS.timeline_gap)
        self.ttk.Button(controls, text="Play", command=self._toggle, style="Action.TButton").grid(row=0, column=2, padx=TOKENS.button_gap)
        self.ttk.Button(controls, text="Step", command=self._step, style="Action.TButton").grid(row=0, column=3, padx=TOKENS.button_gap)
        self.ttk.Button(controls, text="Reset replay", command=self._reset, style="Action.TButton").grid(row=0, column=4, padx=TOKENS.button_gap)
        self.ttk.Button(controls, text="Focus selected", command=self._focus_selected_vehicle, style="Action.TButton").grid(row=0, column=5, padx=TOKENS.button_gap)
        self.ttk.Button(controls, text="Reset view", command=self._reset_view, style="Action.TButton").grid(row=0, column=6, padx=TOKENS.button_gap)
        self.show_keepout = self.tk.BooleanVar(value=True)
        self.ttk.Checkbutton(controls, text="Show keep-out volume", variable=self.show_keepout, command=self._render).grid(row=0, column=7, padx=TOKENS.button_gap)
        self.ttk.Button(controls, text="Check updates", command=self._check_update, style="Action.TButton").grid(row=0, column=8, padx=(TOKENS.update_gap, 0))

        inspector = self.ttk.Frame(body, style="Surface.TFrame", padding=TOKENS.spacing_sm)
        inspector.grid(row=0, column=2, sticky="nsew", padx=(TOKENS.spacing_sm, 0))
        self.ttk.Label(inspector, text="Selected vehicle", style="Muted.TLabel").pack(anchor="w")
        self.telemetry = self.ttk.Label(inspector, justify="left", anchor="nw", style="Value.TLabel", wraplength=TOKENS.inspector_text_width)
        self.telemetry.pack(fill="x", pady=(TOKENS.telemetry_top_gap, TOKENS.telemetry_bottom_gap))
        self._add_disclosure(inspector, "Flight state", self._flight_state_text)
        self._add_disclosure(inspector, "Cooperative context", self._cooperative_context_text)
        self._add_disclosure(inspector, "Parts profile", self._parts_profile_text)
        self._add_disclosure(inspector, "Mission events", self._mission_events_text)
        self._add_disclosure(inspector, "Replay notes", self._replay_notes_text)
        self._add_disclosure(inspector, "Provenance", self._provenance_text)

    def _add_disclosure(self, parent: Any, title: str, body: Any) -> None:
        section = self.ttk.Frame(parent, style="Surface.TFrame")
        section.pack(fill="x", pady=(0, TOKENS.spacing_sm))
        text = body() if callable(body) else body
        content = self.ttk.Label(section, text=text, justify="left", anchor="nw", style="Muted.TLabel", wraplength=TOKENS.inspector_text_width)
        self._disclosures[title] = (content, body)
        expanded = False

        def toggle() -> None:
            nonlocal expanded
            expanded = not expanded
            if expanded:
                content.pack(fill="x", pady=(TOKENS.content_pad, TOKENS.content_pad))
                button.configure(text=f"⌄  {title}")
            else:
                content.pack_forget()
                button.configure(text=f"›  {title}")

        button = self.ttk.Button(section, text=f"›  {title}", command=toggle, style="Disclosure.TButton")
        button.pack(fill="x")

    def _refresh_disclosures(self) -> None:
        for content, body in self._disclosures.values():
            if callable(body):
                content.configure(text=body())

    def _selected_agent(self) -> Any:
        return next((agent for agent in self.model.frame.agents if agent.agent_id == self.selected_id), self.model.frame.agents[0])

    def _flight_state_text(self) -> str:
        agent = self._selected_agent()
        constraints = ", ".join(agent.constraint_flags) if agent.constraint_flags else "Clear"
        return (
            f"Position     [{agent.position_m[0]:.2f}, {agent.position_m[1]:.2f}, {agent.position_m[2]:.2f}] m\n"
            f"Velocity     [{agent.velocity_mps[0]:.2f}, {agent.velocity_mps[1]:.2f}, {agent.velocity_mps[2]:.2f}] m/s\n"
            f"Active       {'Yes' if agent.active else 'No'}\n"
            f"Constraints  {constraints}"
        )

    def _cooperative_context_text(self) -> str:
        agent = self._selected_agent()
        link = self.model.link_stats_for(agent.agent_id)
        if link is None:
            link_text = "No packet events"
        else:
            delay = f"{link.mean_delay_ms:.1f} ms mean" if link.mean_delay_ms is not None else "delay unavailable"
            link_text = f"{delay} · {link.loss_pct:.1f}% loss"
        return (
            f"Neighbors    {agent.neighbor_count}\n"
            f"Separation   {agent.min_neighbor_distance_m:.3f} m\n"
            f"Link         {link_text}\n"
            f"Confidence   {agent.confidence:.3f}"
        )

    def _parts_profile_text(self) -> str:
        agent = self._selected_agent()
        profile = self.model.profile_for(agent.agent_id)
        if profile is None:
            return f"Profile      {agent.profile_id}\nParts data   Unavailable in this manifest"
        return (
            f"Profile      {profile.profile_id}\n"
            f"Motor        {profile.motor_part_id} · {profile.motor_max_power_w:.2f} W\n"
            f"Propeller    {profile.propeller_part_id} · {profile.propeller_diameter_m * 1000:.1f} mm\n"
            f"Configuration {profile.motor_count} motors · {profile.total_mass_kg:.3f} kg\n"
            f"Max thrust   {profile.estimated_max_thrust_n:.2f} N"
        )

    def _replay_notes_text(self) -> str:
        link = self.model.link_stats_for(self.selected_id)
        link_line = "Selected link: no packet events"
        if link is not None:
            delay = f"{link.mean_delay_ms:.1f} ms mean" if link.mean_delay_ms is not None else "delay unavailable"
            link_line = f"Selected link: {delay}, {link.loss_pct:.1f}% loss"
        return (
            f"Frame {self.model.index + 1} committed\n"
            f"Authority: {self.model.command_authority}\n"
            f"Failsafe: {self.model.failsafe}\n"
            f"{link_line}\n\n"
            "No live radio, camera, motor, or swarm command path is attached."
        )

    def _mission_events_text(self) -> str:
        """Return the same compact replay event vocabulary as the macOS app."""

        return "\n".join(mission_event_lines(self.model, self.selected_id))

    def _provenance_text(self) -> str:
        return (
            f"{self.model.schema}\n"
            f"Scenario {self.model.scenario_id}\n"
            f"Seed {self.model.seed} · Δt {self.model.dt_s:.3f} s\n"
            f"Profiles {len(self.model.agent_profiles)} · link events {self.model.summary()['link_event_count']}\n\n"
            f"{self.model.evidence_boundary}"
        )

    def _bounds(self) -> tuple[float, float, float, float]:
        points = [agent.position_m for frame in self.model.frames for agent in frame.agents]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), max(xs), min(ys), max(ys)

    def _project(self, x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        width = max(100, self.canvas.winfo_width())
        height = max(100, self.canvas.winfo_height())
        span_x = max(self._max_x - self._min_x, 1.0) / self._zoom
        span_y = max(self._max_y - self._min_y, 1.0) / self._zoom
        scale = min((width - TOKENS.canvas_extent) / span_x, (height - TOKENS.canvas_extent) / span_y)
        center_x, center_y = self._view_center
        center_z = 1.5
        if self._focus_selected:
            agent = self._selected_agent()
            center_x, center_y = agent.position_m[0], agent.position_m[1]
            center_z = agent.position_m[2]
        # A restrained oblique projection gives the Windows preview depth cues
        # while keeping the replay surface lightweight and deterministic.
        depth_x = (x - center_x) - 0.48 * (z - center_z)
        depth_y = (y - center_y) - 0.32 * (z - center_z)
        return width / 2 + depth_x * scale, height / 2 - depth_y * scale

    def _render(self) -> None:
        if self.canvas is None:
            return
        canvas = self.canvas
        canvas.delete("all")
        width = max(100, canvas.winfo_width())
        height = max(100, canvas.winfo_height())
        for fraction in (0.25, 0.5, 0.75):
            canvas.create_line(width * fraction, TOKENS.canvas_margin, width * fraction, height - TOKENS.canvas_margin, fill=TOKENS.grid)
            canvas.create_line(TOKENS.canvas_margin, height * fraction, width - TOKENS.canvas_margin, height * fraction, fill=TOKENS.grid)
        if self.show_keepout is not None and self.show_keepout.get():
            self._render_keepout(canvas)
        target_x, target_y = self._project(*self.model.frame.target_position_m)
        canvas.create_oval(target_x - TOKENS.target_radius, target_y - TOKENS.target_radius, target_x + TOKENS.target_radius, target_y + TOKENS.target_radius, outline=TOKENS.accent, width=2)
        canvas.create_line(target_x - TOKENS.target_cross, target_y, target_x + TOKENS.target_cross, target_y, fill=TOKENS.accent)
        canvas.create_line(target_x, target_y - TOKENS.target_cross, target_x, target_y + TOKENS.target_cross, fill=TOKENS.accent)
        for agent in self.model.frame.agents:
            self._render_drone(agent)
        canvas.create_text(TOKENS.canvas_label_x, TOKENS.canvas_label_y, text=f"Frame {self.model.index + 1} of {self.model.frame_count}  ·  T+{self.model.frame.time_s:.2f} s", fill=TOKENS.muted, anchor="nw", font=(TOKENS.mono_font, TOKENS.body_size))
        if self.status is not None:
            view = "focused" if self._focus_selected else "overview"
            keepout = "keep-out on" if self.show_keepout is not None and self.show_keepout.get() else "keep-out off"
            self.status.configure(text=f"T+{self.model.frame.time_s:.2f} s   |   active {self.model.frame.active_count}/{len(self.model.frame.agents)}   |   {view} · {keepout}   |   {self.model.status}")
        if self.timeline is not None and int(float(self.timeline.get())) != self.model.index:
            self.timeline.set(self.model.index)
        self._render_fleet()
        self._refresh_disclosures()
        self._render_telemetry()

    def _render_keepout(self, canvas: Any) -> None:
        """Draw the same fixed synthetic keep-out volume used by the macOS scene."""

        # Match the macOS SceneKit volume: a 1.3 m by 1.6 m by 1.3 m box.
        corners = {
            "near_low": self._project(-0.65, -2.30, 0.0),
            "near_high": self._project(-0.65, -0.70, 0.0),
            "far_low": self._project(0.65, -2.30, 0.0),
            "far_high": self._project(0.65, -0.70, 0.0),
            "top_near_low": self._project(-0.65, -2.30, 1.6),
            "top_near_high": self._project(-0.65, -0.70, 1.6),
            "top_far_low": self._project(0.65, -2.30, 1.6),
            "top_far_high": self._project(0.65, -0.70, 1.6),
        }
        canvas.create_polygon(
            corners["top_near_low"], corners["top_far_low"], corners["top_far_high"], corners["top_near_high"],
            outline=TOKENS.warning, fill="", width=2, dash=(5, 4),
        )
        for low, high in (("near_low", "top_near_low"), ("far_low", "top_far_low"), ("near_high", "top_near_high"), ("far_high", "top_far_high")):
            canvas.create_line(*corners[low], *corners[high], fill=TOKENS.warning, dash=(4, 4))
        canvas.create_line(*corners["near_low"], *corners["far_high"], fill=TOKENS.warning, dash=(4, 4))
        canvas.create_text(corners["top_near_low"][0] + 5, corners["top_near_low"][1] + 5, text="keep-out", fill=TOKENS.warning, anchor="nw", font=(TOKENS.ui_font, TOKENS.small_size))

    def _render_drone(self, agent: Any) -> None:
        """Draw a small depth-aware quadcopter glyph instead of a flat dot."""

        canvas = self.canvas
        if canvas is None:
            return
        x, y, z = agent.position_m
        heading = math.atan2(agent.velocity_mps[1], agent.velocity_mps[0]) if any(agent.velocity_mps[:2]) else 0.0
        color = TOKENS.warning if (not agent.active or agent.confidence < 0.98) else TOKENS.accent
        body = [self._project(x + dx, y + dy, z) for dx, dy in self._rotated_points(heading, 0.20, 0.11)]
        canvas.create_polygon(body, fill=TOKENS.surface, outline=color, width=1)
        for arm_angle in (heading + math.pi / 4, heading - math.pi / 4):
            start = self._project(x - 0.34 * math.cos(arm_angle), y - 0.34 * math.sin(arm_angle), z)
            end = self._project(x + 0.34 * math.cos(arm_angle), y + 0.34 * math.sin(arm_angle), z)
            canvas.create_line(*start, *end, fill=color, width=2)
        for rotor_angle in (heading + math.pi / 4, heading - math.pi / 4):
            for sign in (-1, 1):
                rotor_x = x + sign * 0.30 * math.cos(rotor_angle)
                rotor_y = y + sign * 0.30 * math.sin(rotor_angle)
                screen_x, screen_y = self._project(rotor_x, rotor_y, z + 0.04)
                canvas.create_oval(screen_x - 4, screen_y - 2, screen_x + 4, screen_y + 2, outline=color)
        if agent.agent_id == self.selected_id:
            selected_x, selected_y = self._project(x, y, z)
            canvas.create_oval(selected_x - 14, selected_y - 14, selected_x + 14, selected_y + 14, outline=TOKENS.selection_outline_color, width=2)
            canvas.create_text(selected_x + TOKENS.label_offset, selected_y - TOKENS.label_offset, text=agent.agent_id, fill=TOKENS.text, anchor="w", font=(TOKENS.mono_font, TOKENS.small_size, "bold"))

    @staticmethod
    def _rotated_points(heading: float, width: float, height: float) -> tuple[tuple[float, float], ...]:
        points = ((width, 0), (0, height), (-width, 0), (0, -height))
        return tuple((px * math.cos(heading) - py * math.sin(heading), px * math.sin(heading) + py * math.cos(heading)) for px, py in points)

    def _render_fleet(self) -> None:
        if self.fleet is None:
            return
        visible = filter_agents(self.model.frame.agents, self.filter_var.get() if self.filter_var is not None else "")
        for row in self.fleet.get_children():
            self.fleet.delete(row)
        for agent in visible:
            state = "Active" if agent.active else "Inactive"
            battery = f"{agent.battery_pct:.0f}%" if agent.battery_pct is not None else "—"
            link = self.model.link_stats_for(agent.agent_id)
            if agent.link_delay_ms is not None:
                link_text = f"{agent.link_delay_ms:.0f} ms"
            elif link is not None and link.mean_delay_ms is not None:
                link_text = f"{link.mean_delay_ms:.0f} ms"
            else:
                link_text = "—"
            self.fleet.insert("", "end", iid=agent.agent_id, values=(agent.agent_id, state, battery, link_text))
        if self.fleet_summary is not None:
            query = self.filter_var.get().strip() if self.filter_var is not None else ""
            suffix = f" · filter {query!r}" if query else ""
            self.fleet_summary.configure(text=f"Fleet · {len(visible)} of {len(self.model.frame.agents)} vehicles{suffix}")
        visible_ids = {agent.agent_id for agent in visible}
        if self.selected_id in visible_ids and self.fleet.selection() != (self.selected_id,):
            self.fleet.selection_set(self.selected_id)

    def _on_filter(self) -> None:
        self._render_fleet()
        self._render()

    def _focus_selected_vehicle(self) -> None:
        self._focus_selected = True
        self._zoom = 1.65
        self._render()

    def _reset_view(self) -> None:
        self._focus_selected = False
        self._zoom = 1.0
        self._view_center = ((self._min_x + self._max_x) / 2, (self._min_y + self._max_y) / 2)
        self._render()

    def _toggle_keepout(self) -> None:
        if self.show_keepout is not None:
            self.show_keepout.set(not self.show_keepout.get())
        self._render()

    def _render_telemetry(self) -> None:
        if self.telemetry is None:
            return
        agent = self._selected_agent()
        link = self.model.link_stats_for(agent.agent_id)
        if agent.link_delay_ms is not None:
            link_text = f"{agent.link_delay_ms:.0f} ms"
        elif link is not None and link.mean_delay_ms is not None:
            link_text = f"{link.mean_delay_ms:.1f} ms"
        else:
            link_text = "—"
        loss_text = f"{link.loss_pct:.1f}%" if link is not None else "—"
        battery_text = f"{agent.battery_pct:.0f}%" if agent.battery_pct is not None else "—"
        text = (
            f"{agent.agent_id} · {'Active' if agent.active else 'Inactive'}\n\n"
            f"Battery     {battery_text}\n"
            f"Confidence   {agent.confidence:.3f}\n"
            f"Link        {link_text}\n"
            f"Loss        {loss_text}\n"
            f"Neighbors    {agent.neighbor_count}\n"
            f"Separation   {agent.min_neighbor_distance_m:.3f} m\n\n"
            f"Position     [{agent.position_m[0]:.2f}, {agent.position_m[1]:.2f}, {agent.position_m[2]:.2f}] m\n"
            f"Velocity     [{agent.velocity_mps[0]:.2f}, {agent.velocity_mps[1]:.2f}, {agent.velocity_mps[2]:.2f}] m/s\n"
            f"Constraints  {', '.join(agent.constraint_flags) if agent.constraint_flags else 'Clear'}"
        )
        self.telemetry.configure(text=text)

    def _on_select(self, _event: Any) -> None:
        selection = self.fleet.selection()
        if selection:
            self.selected_id = selection[0]
            self._render()

    def _seek(self, value: str) -> None:
        try:
            index = int(float(value))
        except ValueError:
            return
        if index != self.model.index:
            self.model.reset()
            self.model.step(index)
        self._render()

    def _toggle(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self._tick()

    def _tick(self) -> None:
        if not self.playing:
            return
        if self.model.index >= self.model.frame_count - 1:
            self.playing = False
            return
        self.model.step()
        self._render()
        self.root.after(max(20, int(self.model.dt_s * 1000)), self._tick)

    def _step(self) -> None:
        self.playing = False
        self.model.step()
        self._render()

    def _reset(self) -> None:
        self.playing = False
        self.model.reset()
        self._render()

    def _check_update(self) -> None:
        if self.status is not None:
            self.status.configure(text="Checking GitHub release…")

        def worker() -> None:
            try:
                try:
                    from tools.release_updater import GitHubReleaseClient
                except ModuleNotFoundError:
                    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                    from tools.release_updater import GitHubReleaseClient
                client = GitHubReleaseClient()
                release, asset, newer = client.check(CURRENT_VERSION, platform=client.platform_name())
                if not newer:
                    message = f"Up to date: {release.tag}"
                elif not getattr(sys, "frozen", False):
                    message = f"Update available: {release.tag} · packaged app required to install"
                else:
                    self.root.after(0, lambda: self.status.configure(text=f"Downloading {release.tag}…"))
                    staged = client.download_and_stage(asset)
                    install_dir = Path(sys.executable).resolve().parent
                    updater_candidates = (
                        install_dir / "ZephyrMeshUpdater.exe",
                        install_dir.parent / "ZephyrMeshUpdater.exe",
                    )
                    updater = next((candidate for candidate in updater_candidates if candidate.is_file()), updater_candidates[0])
                    launch = install_dir / "ZephyrMeshWindows.exe"
                    if not updater.is_file():
                        message = "Update staged, but ZephyrMeshUpdater.exe is missing"
                    else:
                        updater = updater_launch_path(updater, install_dir, pid=os.getpid())
                        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        subprocess.Popen(
                            [str(updater), "--pid", str(os.getpid()), "--staged", str(staged.payload), "--install-dir", str(install_dir), "--launch", str(launch)],
                            creationflags=flags,
                            close_fds=True,
                        )
                        message = f"Installing {release.tag}…"
            except Exception as exc:
                message = f"Update check unavailable: {exc}"
            if self.status is not None:
                self.root.after(0, lambda: self.status.configure(text=message))
                if message.startswith("Installing "):
                    self.root.after(700, self.root.destroy)
        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _evidence_loader() -> tuple[Any, type[ValueError]]:
        """Load the shared report decoder without making Tk import it early."""

        try:
            from .evidence import EvidenceError, load_evidence_report
        except ImportError:  # direct script execution and PyInstaller
            try:
                from desktop.evidence import EvidenceError, load_evidence_report
            except ImportError:
                from evidence import EvidenceError, load_evidence_report
        return load_evidence_report, EvidenceError

    def _open_evidence_report(self) -> bool:
        """Ask for one portable report and show it in a read-only inspector."""

        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Open evidence report",
            filetypes=(("JSON evidence reports", "*.json"), ("All files", "*.*")),
        )
        if not selected:
            return False
        return self._show_evidence_report(Path(selected))

    def _open_scenario_run(self) -> bool:
        """Open one generated scenario run in a read-only frame inspector."""

        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Open synthetic scenario run",
            filetypes=(("JSON scenario runs", "*.json"), ("All files", "*.*")),
        )
        if not selected:
            return False
        return self._show_scenario_run(Path(selected))

    def _show_scenario_run(self, path: Path) -> bool:
        """Decode and display scenario summary plus frame rows without mutating replay state."""

        from tkinter import messagebox

        try:
            run = load_scenario_run(path)
        except ScenarioRunFormatError as exc:
            messagebox.showerror("Scenario run", str(exc), parent=self.root)
            return False
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Scenario run", f"Could not open run: {exc}", parent=self.root)
            return False
        if self._scenario_window is not None:
            try:
                self._scenario_window.destroy()
            except self.tk.TclError:
                pass
        window = self.tk.Toplevel(self.root)
        self._scenario_window = window
        window.title(f"Scenario run · {path.name}")
        window.minsize(650, 520)
        window.configure(background=TOKENS.background)
        window.transient(self.root)
        header = self.ttk.Frame(window, style="Surface.TFrame", padding=TOKENS.spacing_lg)
        header.pack(fill="x")
        self.ttk.Label(header, text="Synthetic scenario run", style="Title.TLabel").pack(anchor="w")
        self.ttk.Label(header, text=f"{path.name} · read-only replay", style="Muted.TLabel").pack(anchor="w", pady=(TOKENS.content_pad, 0))
        body = self.ttk.Frame(window, style="Night.TFrame", padding=TOKENS.spacing_lg)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        summary = run.summary()
        self.ttk.Label(body, text="Summary", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        summary_text = self.tk.Text(body, height=9, wrap="word", background=TOKENS.surface, foreground=TOKENS.text, relief="flat", borderwidth=0, padx=TOKENS.content_pad, pady=TOKENS.content_pad, font=(TOKENS.mono_font, TOKENS.body_size))
        summary_text.insert("1.0", json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        summary_text.configure(state="disabled")
        summary_text.grid(row=1, column=0, sticky="nsew", pady=(TOKENS.content_pad, TOKENS.spacing_sm))
        self.ttk.Label(body, text="Frames", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        frames_text = self.tk.Text(body, wrap="none", background=TOKENS.surface, foreground=TOKENS.text, relief="flat", borderwidth=0, padx=TOKENS.content_pad, pady=TOKENS.content_pad, font=(TOKENS.mono_font, TOKENS.body_size), takefocus=True)
        frame_lines = [
            f"frame {frame.step_index:>4}  t={frame.time_s:>8.3f} s  active={frame.active_count:>3}  agents={frame.agent_count:>3}"
            for frame in run.frames
        ]
        frames_text.insert("1.0", "\n".join(frame_lines))
        frames_text.configure(state="disabled")
        frames_text.grid(row=3, column=0, sticky="nsew", pady=(TOKENS.content_pad, 0))
        body.rowconfigure(3, weight=1)
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

    @staticmethod
    def _report_source(report: dict[str, Any]) -> dict[str, Any]:
        source = report.get("source")
        return source if isinstance(source, dict) else {}

    def _render_evidence_report(self, report: dict[str, Any], path: Path) -> None:
        """Render provenance and JSON values, retaining nulls and nested fields."""

        if self._evidence_window is not None:
            try:
                self._evidence_window.destroy()
            except self.tk.TclError:
                pass
        window = self.tk.Toplevel(self.root)
        self._evidence_window = window
        window.title(f"Evidence report · {path.name}")
        window.minsize(560, 460)
        window.configure(background=TOKENS.background)
        window.transient(self.root)

        source = self._report_source(report)
        kind = report.get("kind")
        status = report.get("status")
        digest = source.get("sha256")
        schema = source.get("schema") or report.get("schema")
        header = self.ttk.Frame(window, style="Surface.TFrame", padding=TOKENS.spacing_lg)
        header.pack(fill="x")
        self.ttk.Label(header, text="Evidence report", style="Title.TLabel").pack(anchor="w")
        self.ttk.Label(header, text=path.name, style="Muted.TLabel").pack(anchor="w", pady=(TOKENS.content_pad, 0))
        details = self.ttk.Frame(window, style="Surface.TFrame", padding=(TOKENS.spacing_lg, 0, TOKENS.spacing_lg, TOKENS.spacing_sm))
        details.pack(fill="x")
        self.ttk.Label(details, text=f"Kind  {kind or '—'}    Declared status  {status or '—'}", style="Value.TLabel").pack(anchor="w")
        self.ttk.Label(details, text=f"Source SHA-256 (declared)  {digest or '—'}", style="Muted.TLabel").pack(anchor="w", pady=(TOKENS.content_pad, 0))
        self.ttk.Label(details, text=f"Schema  {schema or '—'}", style="Muted.TLabel").pack(anchor="w")

        body = self.ttk.Frame(window, style="Night.TFrame", padding=(TOKENS.spacing_lg, TOKENS.spacing_sm, TOKENS.spacing_lg, TOKENS.spacing_lg))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        self.ttk.Label(body, text="Summary", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        text = self.tk.Text(body, wrap="word", background=TOKENS.surface, foreground=TOKENS.text, relief="flat", borderwidth=0, padx=TOKENS.content_pad, pady=TOKENS.content_pad, font=(TOKENS.mono_font, TOKENS.body_size), takefocus=True)
        scrollbar = self.ttk.Scrollbar(body, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.grid(row=1, column=0, sticky="nsew", pady=(TOKENS.content_pad, TOKENS.spacing_sm))
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(TOKENS.content_pad, TOKENS.spacing_sm))
        summary = report.get("summary")
        text.insert("1.0", json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        text.configure(state="disabled")

        limitations = report.get("limitations")
        limitation_text = limitations if isinstance(limitations, list) else []
        self.ttk.Label(body, text="Limitations", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        limits = self.ttk.Label(body, text="\n".join(f"• {item}" for item in limitation_text) or "None recorded", justify="left", anchor="w", style="Muted.TLabel", wraplength=510)
        limits.grid(row=3, column=0, sticky="ew", pady=(TOKENS.content_pad, 0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="canonical S7 run.json")
    parser.add_argument("--scenario-run", type=Path, help="open a generated zephyr-s7-scenario-run-1 JSON in the read-only inspector")
    parser.add_argument("--headless", action="store_true", help="print the replay summary without opening Tk")
    parser.add_argument("--smoke-ready", type=Path, help="write a readiness marker after the GUI is constructed")
    parser.add_argument("--smoke-step", action="store_true", help="step one replay frame before writing the readiness marker")
    parser.add_argument("--smoke-health", type=Path, help="write a second marker after the GUI remains healthy in its event loop")
    parser.add_argument("--smoke-evidence", type=Path, action="append", help="open and validate an evidence report during frozen GUI smoke; repeat for parity checks")
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
        app = WindowsReplayApp(root, model)
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
        if args.smoke_ready:
            suffix = f" evidence_kind={evidence_kind}" if evidence_kind else ""
            args.smoke_ready.write_text(f"ready frame={model.index + 1}/{model.frame_count}{suffix}\n", encoding="utf-8")
        if args.smoke_ready or args.smoke_health:
            # Hosted Windows runners can terminate a Tk event loop immediately,
            # and calling ``update`` in that state can make the frozen process
            # exit with code 1 without a useful stderr stream.  The smoke path
            # therefore uses the successful Tk construction above as its GUI
            # check, then stays alive with a bounded sleep so the validator can
            # observe both markers.  A guarded idle flush exercises pending Tk
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
                    args.smoke_health.write_text(f"healthy frame={model.index + 1}/{model.frame_count}{suffix}\n", encoding="utf-8")
                    health_written = True
                time.sleep(0.02)
            try:
                root.destroy()
            except tk.TclError:
                pass
        else:
            root.mainloop()
        return 0
    except Exception as exc:
        log_path = write_startup_diagnostic(exc)
        show_startup_failure(exc, log_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
