"""Windows-friendly Tkinter replay preview for the Zephyr Mesh S7 fixture.

The app is intentionally offline and read-only. It is a desktop preview of the
canonical Python event log, not a vehicle controller or a live telemetry client.
Tkinter is imported only when ``main`` runs so the pure replay model remains
testable on machines without Tk support.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

try:
    from .replay import ReplayModel
    from .design_tokens import TOKENS
except ImportError:  # direct script execution for PyInstaller
    from replay import ReplayModel
    from design_tokens import TOKENS


RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
DEFAULT_RUN = RESOURCE_ROOT / "runs" / "s7-swarm" / "run.json"
CURRENT_VERSION = "0.1.0"


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
        self.canvas = None
        self.timeline = None
        self.fleet = None
        self.telemetry = None
        self.status = None
        self._min_x, self._max_x, self._min_y, self._max_y = self._bounds()
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
        self.ttk.Label(fleet_panel, text=f"Fleet · {len(self.model.frame.agents)} vehicles", style="Muted.TLabel").pack(anchor="w")
        self.fleet = self.ttk.Treeview(fleet_panel, columns=("vehicle", "state", "battery", "link"), show="headings", selectmode="browse", style="Fleet.Treeview", takefocus=True)
        for column, label, width, anchor in (("vehicle", "Vehicle", TOKENS.fleet_column_vehicle, "w"), ("state", "State", TOKENS.fleet_column_state, "w"), ("battery", "Battery", TOKENS.fleet_column_metric, "e"), ("link", "Link", TOKENS.fleet_column_metric, "e")):
            self.fleet.heading(column, text=label)
            self.fleet.column(column, width=width, anchor=anchor, stretch=column == "vehicle")
        self.fleet.pack(fill="both", expand=True, pady=(TOKENS.table_top_gap, 0))
        for agent in self.model.frame.agents:
            state = "Active" if agent.active else "Inactive"
            battery = f"{agent.battery_pct:.0f}%" if agent.battery_pct is not None else "—"
            link = f"{agent.link_delay_ms:.0f} ms" if agent.link_delay_ms is not None else "—"
            self.fleet.insert("", "end", iid=agent.agent_id, values=(agent.agent_id, state, battery, link))
        self.fleet.selection_set(self.selected_id)
        self.fleet.bind("<<ListboxSelect>>", self._on_select)
        self.fleet.bind("<<TreeviewSelect>>", self._on_select)
        root.bind("<space>", lambda _event: self._toggle())
        root.bind("<Right>", lambda _event: self._step())

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
        self.ttk.Button(controls, text="Reset", command=self._reset, style="Action.TButton").grid(row=0, column=4, padx=TOKENS.button_gap)
        self.ttk.Button(controls, text="Check updates", command=self._check_update, style="Action.TButton").grid(row=0, column=5, padx=(TOKENS.update_gap, 0))

        inspector = self.ttk.Frame(body, style="Surface.TFrame", padding=TOKENS.spacing_sm)
        inspector.grid(row=0, column=2, sticky="nsew", padx=(TOKENS.spacing_sm, 0))
        self.ttk.Label(inspector, text="Selected vehicle", style="Muted.TLabel").pack(anchor="w")
        self.telemetry = self.ttk.Label(inspector, justify="left", anchor="nw", style="Value.TLabel", wraplength=TOKENS.inspector_text_width)
        self.telemetry.pack(fill="x", pady=(TOKENS.telemetry_top_gap, TOKENS.telemetry_bottom_gap))
        events = (
            "Manifest loaded\n"
            "Authority: replay only\n"
            "Boundary: synthetic data\n\n"
            "No live radio, camera, motor, or swarm command path."
        )
        self._add_disclosure(inspector, "Replay notes", events)
        provenance = f"{self.model.schema}\nseed {self.model.seed} · Δt {self.model.dt_s:.3f} s\n\n{self.model.evidence_boundary}"
        self._add_disclosure(inspector, "Provenance", provenance)

    def _add_disclosure(self, parent: Any, title: str, body: str) -> None:
        section = self.ttk.Frame(parent, style="Surface.TFrame")
        section.pack(fill="x", pady=(0, TOKENS.spacing_sm))
        content = self.ttk.Label(section, text=body, justify="left", anchor="nw", style="Muted.TLabel", wraplength=TOKENS.inspector_text_width)
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

    def _bounds(self) -> tuple[float, float, float, float]:
        points = [agent.position_m for frame in self.model.frames for agent in frame.agents]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), max(xs), min(ys), max(ys)

    def _project(self, x: float, y: float) -> tuple[float, float]:
        width = max(100, self.canvas.winfo_width())
        height = max(100, self.canvas.winfo_height())
        span_x = max(self._max_x - self._min_x, 1.0)
        span_y = max(self._max_y - self._min_y, 1.0)
        scale = min((width - TOKENS.canvas_extent) / span_x, (height - TOKENS.canvas_extent) / span_y)
        origin_x = (width - scale * span_x) / 2
        origin_y = (height + scale * span_y) / 2
        return origin_x + (x - self._min_x) * scale, origin_y - (y - self._min_y) * scale

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
        target_x, target_y = self._project(*self.model.frame.target_position_m[:2])
        canvas.create_oval(target_x - TOKENS.target_radius, target_y - TOKENS.target_radius, target_x + TOKENS.target_radius, target_y + TOKENS.target_radius, outline=TOKENS.accent, width=2)
        canvas.create_line(target_x - TOKENS.target_cross, target_y, target_x + TOKENS.target_cross, target_y, fill=TOKENS.accent)
        canvas.create_line(target_x, target_y - TOKENS.target_cross, target_x, target_y + TOKENS.target_cross, fill=TOKENS.accent)
        for agent in self.model.frame.agents:
            x, y = self._project(agent.position_m[0], agent.position_m[1])
            weak = (not agent.active) or agent.confidence < 0.98
            color = TOKENS.warning if weak else TOKENS.accent
            radius = 4 + min(5, max(0, agent.position_m[2]))
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="")
            if agent.agent_id == self.selected_id:
                canvas.create_oval(x - radius - TOKENS.selection_outline, y - radius - TOKENS.selection_outline, x + radius + TOKENS.selection_outline, y + radius + TOKENS.selection_outline, outline=TOKENS.selection_outline_color, width=2)
                canvas.create_text(x + TOKENS.label_offset, y - TOKENS.label_offset, text=agent.agent_id, fill=TOKENS.text, anchor="w", font=(TOKENS.mono_font, TOKENS.small_size, "bold"))
        canvas.create_text(TOKENS.canvas_label_x, TOKENS.canvas_label_y, text=f"Frame {self.model.index + 1} of {self.model.frame_count}  ·  T+{self.model.frame.time_s:.2f} s", fill=TOKENS.muted, anchor="nw", font=(TOKENS.mono_font, TOKENS.body_size))
        if self.status is not None:
            self.status.configure(text=f"T+{self.model.frame.time_s:.2f} s   |   active {self.model.frame.active_count}/{len(self.model.frame.agents)}   |   {self.model.status}")
        if self.timeline is not None and int(float(self.timeline.get())) != self.model.index:
            self.timeline.set(self.model.index)
        self._render_telemetry()

    def _render_telemetry(self) -> None:
        if self.telemetry is None:
            return
        agent = next((item for item in self.model.frame.agents if item.agent_id == self.selected_id), self.model.frame.agents[0])
        text = (
            f"{agent.agent_id} · {'Active' if agent.active else 'Inactive'}\n\n"
            f"Confidence   {agent.confidence:.3f}\n"
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
                    updater = install_dir.parent / "ZephyrMeshUpdater.exe"
                    launch = install_dir / "ZephyrMeshWindows.exe"
                    if not updater.is_file():
                        message = "Update staged, but ZephyrMeshUpdater.exe is missing"
                    else:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="canonical S7 run.json")
    parser.add_argument("--headless", action="store_true", help="print the replay summary without opening Tk")
    parser.add_argument("--smoke-ready", type=Path, help="write a readiness marker after the GUI is constructed")
    parser.add_argument("--smoke-step", action="store_true", help="step one replay frame before writing the readiness marker")
    parser.add_argument("--smoke-health", type=Path, help="write a second marker after the GUI remains healthy in its event loop")
    args = parser.parse_args(argv)
    if args.headless:
        print(json.dumps(ReplayModel.from_path(args.run).summary(), indent=2, sort_keys=True))
        return 0
    try:
        import tkinter as tk
    except ImportError as exc:
        raise SystemExit("Tkinter is required. Install Python with Tcl/Tk support on Windows.") from exc
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass
    model = ReplayModel.from_path(args.run)
    root = tk.Tk()
    app = WindowsReplayApp(root, model)
    if args.smoke_step:
        app._step()
    if args.smoke_ready:
        args.smoke_ready.write_text(f"ready frame={model.index + 1}/{model.frame_count}\n", encoding="utf-8")
    if args.smoke_health:
        root.after(250, lambda: args.smoke_health.write_text(f"healthy frame={model.index + 1}/{model.frame_count}\n", encoding="utf-8"))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
