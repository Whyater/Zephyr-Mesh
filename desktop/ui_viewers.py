"""About, scenario-run, and evidence-report windows for the Windows replay.

Each window leads with its plot or verdict, then a plain-language summary
built from the shared field labels, then a collapsed Details section holding
raw values, hashes, and versions. Decoding and validation stay in
``scenario_run`` and ``evidence``; nothing here loosens those checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .design_tokens import SPACE, UI_TEXT, WIN_LAYOUT
from .presentation import not_in_replay, run_name
from .scenario_run import (
    ScenarioRunCursor,
    clamp_scenario_rail_x,
    project_scenario_altitude,
    project_scenario_altitude_radius,
    project_scenario_point,
    project_scenario_radius,
    scenario_frame_geometry,
    scenario_rail_endpoints,
)
from . import ui_format as fmt
from .ui_theme import mix
from .ui_widgets import Disclosure, KeyValueGrid, ScrollFrame, TimelineSlider


WIN_TEXT = UI_TEXT["windowsText"]


@dataclass
class ViewerContext:
    root: Any
    theme: Callable[[], Any]
    fonts: Any
    images: Any
    scale: float
    add_theme_hook: Callable[[Callable[[Any], None]], None]
    apply_title_bar: Callable[[Any, bool], None]

    def px(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    @property
    def inset(self) -> int:
        """Side padding of every viewer, the same SPACE["lg"] as the main window."""
        return self.px(SPACE["lg"])

    @property
    def label_width(self) -> int:
        """One label column for every viewer, so About, Scenario run, and Evidence line up the same way."""
        return self.px(WIN_LAYOUT["viewerLabelWidth"])

    def hairline(self, parent: Any) -> Any:
        from tkinter import ttk

        return ttk.Frame(parent, style="Line.TFrame", height=self.px(WIN_LAYOUT["hairline"]))


class Rewrap:
    """Calls each registered function with the body's content width whenever the viewer is resized.

    Wrap lengths computed once for the default window size clip text when the
    window is narrower, which would hide the limitations and boundary text.
    """

    def __init__(self, ctx: ViewerContext, scroll: Any) -> None:
        self.ctx = ctx
        self.width = 0
        self.callbacks: list[Callable[[int], None]] = []
        scroll.canvas.bind("<Configure>", self._resized, add="+")

    def add(self, callback: Callable[[int], None]) -> None:
        self.callbacks.append(callback)
        if self.width:
            callback(self.width)

    def _resized(self, event: Any) -> None:
        width = max(1, int(event.width) - 2 * self.ctx.inset)
        if width == self.width:
            return
        self.width = width
        for callback in self.callbacks:
            callback(width)


def _window(ctx: ViewerContext, title: str, size: Mapping[str, int]) -> Any:
    import tkinter as tk

    window = tk.Toplevel(ctx.root)
    window.title(title)
    window.geometry(f"{ctx.px(size['width'])}x{ctx.px(size['height'])}")
    window.minsize(ctx.px(size["minWidth"]), ctx.px(size["minHeight"]))
    window.configure(background=ctx.theme().p.background)
    window.transient(ctx.root)
    window.bind("<Escape>", lambda _event: window.destroy())
    ctx.apply_title_bar(window, ctx.theme().dark)
    return window


def _header(ctx: ViewerContext, parent: Any, title: str, subtitle: str) -> Any:
    from tkinter import ttk

    frame = ttk.Frame(parent, style="Surface.TFrame", padding=(ctx.inset, ctx.px(SPACE["lg"])))
    ttk.Label(frame, text=title, style="Surface.Title.TLabel").pack(anchor="w")
    if subtitle:
        ttk.Label(frame, text=subtitle, style="Surface.Secondary.TLabel").pack(anchor="w", pady=(ctx.px(SPACE["xxs"]), 0))
    return frame


def _raw_text(ctx: ViewerContext, parent: Any, text: str, height: int = 10) -> Any:
    import tkinter as tk

    widget = tk.Text(parent, height=height, wrap="none", relief="flat", borderwidth=0, font=ctx.fonts["mono"],
                     padx=ctx.px(SPACE["sm"]), pady=ctx.px(SPACE["sm"]), highlightthickness=ctx.px(WIN_LAYOUT["hairline"]))
    widget.insert("1.0", text)
    widget.configure(state="disabled")

    def recolor(theme: Any) -> None:
        widget.configure(background=theme.p.background, foreground=theme.p.text, highlightbackground=theme.p.line,
                         highlightcolor=theme.p.accent, selectbackground=theme.p.selectionFill, selectforeground=theme.p.text,
                         insertbackground=theme.p.text)

    recolor(ctx.theme())
    ctx.add_theme_hook(recolor)
    return widget


def _section_title(ctx: ViewerContext, parent: Any, title: str, *, first: bool = False) -> None:
    """Every section title in every viewer uses the same Headline style and spacing."""
    from tkinter import ttk

    top = ctx.px(SPACE["sm"]) if first else ctx.px(SPACE["lg"])
    ttk.Label(parent, text=title, style="Surface.Headline.TLabel").pack(anchor="w", pady=(top, ctx.px(SPACE["xs"])))


def _value_wrap(ctx: ViewerContext, width: int) -> int:
    """Wrap length of a value column: content width minus the label column and its gap."""
    return max(ctx.px(SPACE["xxl"]), width - ctx.label_width - ctx.px(SPACE["md"]))


def _grid(ctx: ViewerContext, parent: Any, rows: Sequence[tuple], rewrap: Rewrap, *, label_wrap: bool = False) -> KeyValueGrid:
    grid = KeyValueGrid(parent, label_width=ctx.label_width, wrap=ctx.px(SPACE["xxl"]), scale=ctx.scale,
                        label_wrap=ctx.label_width - ctx.px(SPACE["md"]) if label_wrap else 0)  # md: the label's right gap
    grid.frame.pack(fill="x")
    grid.set_rows(rows)
    rewrap.add(lambda width: grid.set_wrap(_value_wrap(ctx, width)))
    return grid


def _scroll_body(ctx: ViewerContext, window: Any) -> tuple[Any, Any, Rewrap]:
    from tkinter import ttk

    scroll = ScrollFrame(window, scale=ctx.scale)
    scroll.outer.pack(fill="both", expand=True)
    scroll.apply_theme(ctx.theme())
    ctx.add_theme_hook(scroll.apply_theme)
    body = ttk.Frame(scroll.inner, style="Surface.TFrame", padding=(ctx.inset, ctx.px(SPACE["md"]), ctx.inset, ctx.px(SPACE["xl"])))
    body.pack(fill="both", expand=True)

    def wheel(event: Any) -> None:
        widget = window.winfo_containing(event.x_root, event.y_root)
        if widget is not None and scroll.contains(widget):
            scroll.scroll(event.delta)

    window.bind("<MouseWheel>", wheel, add="+")
    return scroll, body, Rewrap(ctx, scroll)


def _sentence(value: Any) -> str:
    """Show a raw data string with a leading capital; the words themselves are unchanged."""
    text = str(value)
    return text[:1].upper() + text[1:]


def unavailable_items(run: Mapping[str, Any]) -> str:
    """The presentation's "Not in this replay" list as one sentence for the About window."""
    items = not_in_replay(run)
    text = ", ".join(items[:1] + [item[:1].lower() + item[1:] for item in items[1:]])
    return text[:1].upper() + text[1:]


def _copy_button(ctx: ViewerContext, parent: Any, window: Any, text: str) -> Any:
    """A Copy button that says "Copied" for a moment, and reports a busy clipboard instead of raising."""
    from tkinter import TclError, ttk

    button = ttk.Button(parent, text="Copy", style="Tool.TButton", width=max(len("Copy"), len(WIN_TEXT["copied"])))

    def copy() -> None:
        try:
            window.clipboard_clear()
            window.clipboard_append(text)
        except TclError:
            from tkinter import messagebox

            messagebox.showwarning("Copy", WIN_TEXT["clipboardBusy"], parent=window)
            return
        button.configure(text=WIN_TEXT["copied"])

        def restore() -> None:
            try:
                button.configure(text="Copy")
            except TclError:
                pass

        button.after(1200, restore)

    button.configure(command=copy)
    return button


def _details(ctx: ViewerContext, body: Any, rows: Sequence[tuple], rewrap: Rewrap) -> Disclosure:
    """The collapsed Details section: a hairline, the disclosure, and a grid aligned with the summary above."""
    ctx.hairline(body).pack(fill="x", pady=(ctx.px(SPACE["lg"]), ctx.px(SPACE["xs"])))
    disclosure = Disclosure(body, "Details", ctx.images, pad=ctx.px(SPACE["xs"]))
    disclosure.frame.pack(fill="x")
    _grid(ctx, disclosure.body, rows, rewrap, label_wrap=True)
    return disclosure


def about_window(ctx: ViewerContext, run: Mapping[str, Any], details: Sequence[tuple[str, Any, str]]) -> Any:
    from tkinter import ttk

    window = _window(ctx, UI_TEXT["honesty"]["aboutButton"], WIN_LAYOUT["aboutWindow"])
    _header(ctx, window, UI_TEXT["honesty"]["aboutButton"], run_name(run)).pack(fill="x")
    ctx.hairline(window).pack(fill="x")
    _scroll, body, rewrap = _scroll_body(ctx, window)
    intro = ttk.Label(body, text="This window replays a recorded simulation file. It is not connected to any drone.",
                      style="Surface.Body.TLabel", justify="left")
    intro.pack(anchor="w", pady=(ctx.px(SPACE["xs"]), ctx.px(SPACE["md"])))
    rewrap.add(lambda width: intro.configure(wraplength=width))
    rows = []
    for field in ("status", "evidence_boundary", "command_authority", "failsafe"):
        if field in run:
            rows.append((fmt.field_label(field), _sentence(run[field])))
    rows.append((UI_TEXT["notInReplay"]["title"], unavailable_items(run)))
    _grid(ctx, body, rows, rewrap)
    command = run.get("generation_command")
    if command:
        _section_title(ctx, body, fmt.field_label("generation_command"))
        command_row = ttk.Frame(body, style="Surface.TFrame")
        command_row.pack(fill="x")
        _copy_button(ctx, command_row, window, str(command)).pack(side="right", padx=(ctx.px(SPACE["sm"]), 0))
        text = _raw_text(ctx, command_row, str(command), height=1 if len(str(command)) <= 72 else 2)
        text.configure(wrap="word", font=ctx.fonts["body"])
        text.pack(side="left", fill="x", expand=True)
    _details(ctx, body, [(label, None if value is None else value, kind) for label, value, kind in details], rewrap)
    return window


def _frame_line(run: Any, cursor: ScenarioRunCursor) -> str:
    frame = run.frames[cursor.index]
    volumes = len(frame.keep_out_spheres)
    noun = "keep-out volume" if volumes == 1 else "keep-out volumes"
    return (f"Frame {cursor.index + 1} of {len(run.frames)}, {frame.time_s:.2f} s, "
            f"{frame.active_count} of {frame.agent_count} active, {volumes} {noun}")


def scenario_window(ctx: ViewerContext, run: Any, path: Path) -> Any:
    import tkinter as tk
    from tkinter import ttk

    summary = run.summary()
    window = _window(ctx, f"Scenario run: {path.name}", WIN_LAYOUT["scenarioWindow"])
    _header(ctx, window, "Scenario run", path.name).pack(fill="x")
    ctx.hairline(window).pack(fill="x")
    _scroll, body, rewrap = _scroll_body(ctx, window)
    plot = tk.Canvas(body, height=ctx.px(WIN_LAYOUT["scenarioPlotHeight"]), highlightthickness=ctx.px(WIN_LAYOUT["hairline"]),
                     borderwidth=0)
    plot.pack(fill="x", pady=(ctx.px(SPACE["sm"]), ctx.px(SPACE["xs"])))
    legend = ttk.Label(body, text=WIN_TEXT["scenarioLegend"], style="Surface.Caption.TLabel", justify="left")
    legend.pack(anchor="w", pady=(0, ctx.px(SPACE["sm"])))
    rewrap.add(lambda width: legend.configure(wraplength=width))
    cursor = ScenarioRunCursor(len(run.frames))
    controls = ttk.Frame(body, style="Surface.TFrame")
    controls.pack(fill="x")
    frame_text = ttk.Label(body, style="Surface.Secondary.TLabel")
    frame_text.pack(anchor="w", pady=(ctx.px(SPACE["xs"]), 0))
    # Drawn geometry in device-independent pixels (marks on the plot, not layout spacing).
    target_ring, agent_dot, rail_dot, rail_tick = 7.0, 3.5, 3.0, 6.0

    def draw() -> None:
        theme = ctx.theme()
        p = theme.p
        floor = theme.scene.floor.hex
        plot.configure(background=floor, highlightbackground=p.line, highlightcolor=p.accent)
        plot.delete("all")
        geometry = scenario_frame_geometry(run.frames[cursor.index])
        width = max(1.0, float(plot.winfo_width()))
        height = max(1.0, float(plot.winfo_height()))
        rail_space = (SPACE["xxl"] + SPACE["xl"] + SPACE["lg"]) * ctx.scale
        top_width = max(width - rail_space, 1.0)
        caption = ctx.fonts["caption"]
        margin = ctx.px(SPACE["md"])
        plot.create_text(margin, margin, text="Top view", fill=p.textSecondary, anchor="nw", font=caption)
        keep_out = theme.scene.keep_out_volume
        for label, center_x, center_y, radius in geometry.keep_out_spheres:
            x, y = project_scenario_point((center_x, center_y), geometry.bounds, top_width, height)
            radius_px = project_scenario_radius(radius, geometry.bounds, top_width, height)
            plot.create_oval(x - radius_px, y - radius_px, x + radius_px, y + radius_px, outline=keep_out.hex,
                             fill=mix(keep_out.hex, floor, keep_out.alpha), width=max(1, ctx.px(1.5)))
            plot.create_text(x, y + radius_px + ctx.px(SPACE["xs"]), text=label, fill=p.textSecondary, anchor="n", font=caption)
        target_color = theme.scene.target_ring.hex
        target_x, target_y = project_scenario_point(geometry.target, geometry.bounds, top_width, height)
        ring = target_ring * ctx.scale
        plot.create_oval(target_x - ring, target_y - ring, target_x + ring, target_y + ring, outline=target_color,
                         width=max(2, ctx.px(SPACE["xxs"])))
        for _, agent_x, agent_y, active in geometry.agents:
            x, y = project_scenario_point((agent_x, agent_y), geometry.bounds, top_width, height)
            color = theme.scene.drone_body.hex if active else mix(theme.scene.drone_body.hex, floor, 0.4)
            dot = agent_dot * ctx.scale
            plot.create_oval(x - dot, y - dot, x + dot, y + dot, fill=color, outline=color)
        rail_x = clamp_scenario_rail_x(width, min(width - (SPACE["xl"] + SPACE["xxs"]) * ctx.scale,
                                                  top_width + (SPACE["xxl"] + SPACE["xs"]) * ctx.scale))
        rail_top, rail_bottom = scenario_rail_endpoints(height)
        plot.create_line(rail_x, rail_top, rail_x, rail_bottom, fill=p.lineStrong)
        plot.create_text(rail_x, margin, text="Altitude", fill=p.textSecondary, anchor="n", font=caption)
        target_z = project_scenario_altitude(geometry.target_altitude, geometry.altitude_bounds, height, inset=30.0)
        tick = rail_tick * ctx.scale
        plot.create_line(rail_x - tick, target_z, rail_x + tick, target_z, fill=target_color, width=max(2, ctx.px(SPACE["xxs"])))
        for altitude, (_, _, _, active) in zip(geometry.agent_altitudes, geometry.agents):
            y = project_scenario_altitude(altitude, geometry.altitude_bounds, height, inset=30.0)
            color = theme.scene.drone_body.hex if active else mix(theme.scene.drone_body.hex, floor, 0.4)
            dot = rail_dot * ctx.scale
            plot.create_oval(rail_x - dot, y - dot, rail_x + dot, y + dot, fill=color, outline=color)
        for center_z, (_label, _cx, _cy, radius) in zip(geometry.keep_out_altitudes, geometry.keep_out_spheres):
            center_y = project_scenario_altitude(center_z, geometry.altitude_bounds, height, inset=30.0)
            radius_y = project_scenario_altitude_radius(radius, geometry.altitude_bounds, height, inset=30.0)
            offset = ctx.px(SPACE["sm"]) + ctx.px(SPACE["xxs"])
            plot.create_line(rail_x + offset, center_y - radius_y, rail_x + offset, center_y + radius_y,
                             fill=keep_out.hex, width=max(2, ctx.px(3)))

    def update(index: int) -> None:
        cursor.seek(index)
        frame_text.configure(text=_frame_line(run, cursor))
        slider.set_state(cursor.index, cursor.frame_count, [])
        previous.state(["!disabled"] if cursor.index > 0 else ["disabled"])
        restart.state(["!disabled"] if cursor.index > 0 else ["disabled"])
        following.state(["!disabled"] if cursor.index < cursor.frame_count - 1 else ["disabled"])
        draw()

    icon = WIN_LAYOUT["iconSize"]
    restart = ttk.Button(controls, style="Icon.TButton", image=ctx.images.icon("restart", "text", icon),
            command=lambda: update(cursor.reset()))
    previous = ttk.Button(controls, style="Icon.TButton", image=ctx.images.icon("step_back", "text", icon),
            command=lambda: update(cursor.step(-1)))
    following = ttk.Button(controls, style="Icon.TButton", image=ctx.images.icon("step_forward", "text", icon),
            command=lambda: update(cursor.step()))
    for index, button in enumerate((restart, previous, following)):
        button.pack(side="left", padx=(0 if index == 0 else ctx.px(SPACE["xs"]), 0))
    slider = TimelineSlider(controls, lambda index: update(index), ctx.scale)
    if cursor.frame_count > 1:
        # The slider keeps SPACE["sm"] inside each end for its thumb, so the visible gap is SPACE["lg"].
        slider.canvas.pack(side="left", fill="x", expand=True, padx=(ctx.px(SPACE["lg"]) - ctx.px(SPACE["sm"]), 0))
    slider.apply_theme(ctx.theme())

    def recolor(theme: Any) -> None:
        slider.apply_theme(theme)
        draw()

    ctx.add_theme_hook(recolor)
    plot.bind("<Configure>", lambda _event: draw())

    parameters = run.parameters
    target_link = parameters.get("target_link", {})
    duration = summary["dt_s"] * max(summary["frame_count"] - 1, 0)
    rows = [
        (fmt.field_label("agent_count"), summary["agent_count"]),
        (fmt.field_label("frame_count"), summary["frame_count"]),
        (_unitless(fmt.field_label("dt_s")), fmt.format_value(summary["dt_s"], "s")),
        (_unitless(fmt.field_label("duration_s")), fmt.format_value(duration, "s")),
        (fmt.field_label("link_event_count"), summary["link_event_count"]),
        (fmt.field_label("profile_count"), summary["profile_count"]),
        (fmt.field_label("target_link"), fmt.link_text(target_link)),
        (fmt.field_label("neighbor_link"), fmt.link_text(parameters.get("neighbor_link", {}))),
        (fmt.field_label("status"), _sentence(summary["status"])),
        (fmt.field_label("evidence_boundary"), _sentence(summary["evidence_boundary"])),
    ]
    _section_title(ctx, body, "Summary")
    _grid(ctx, body, rows, rewrap)
    disclosure = _details(ctx, body, [
        (fmt.field_label("scenario_id"), summary["scenario_id"], "mono"),
        (fmt.field_label("scenario_hash"), summary["scenario_hash"], "mono"),
        (fmt.field_label("payload_sha256"), summary["payload_sha256"], "mono"),
        (fmt.field_label("schema"), summary["schema"], "mono"),
        (fmt.field_label("version"), summary["version"]),
        (fmt.field_label("generator"), summary["generator"], "mono"),
        (fmt.field_label("python"), summary["python"]),
        (fmt.field_label("numpy"), summary["numpy"]),
        (fmt.field_label("code_revision"), summary["code_revision"], "mono"),
        (fmt.field_label("code_revision_semantics"), summary["code_revision_semantics"]),
    ], rewrap)
    ttk.Label(disclosure.body, text="Raw parameters", style="Surface.Caption.TLabel").pack(anchor="w",
            pady=(ctx.px(SPACE["sm"]), ctx.px(SPACE["xxs"])))
    _raw_text(ctx, disclosure.body, json.dumps(parameters, indent=2, sort_keys=True, ensure_ascii=False), height=12).pack(fill="x")
    update(0)
    return window


def _unitless(label: str) -> str:
    """Drop a trailing "(s)" unit from a label whose value already carries its unit ("0.05 s")."""
    return re.sub(r" \([^)]*\)$", "", label)


def _nice_step(span: float) -> float:
    """A round tick step (1, 2, or 5 times a power of ten) for about four ticks."""
    import math

    raw = max(span, 1e-9) / 4.0
    power = 10 ** math.floor(math.log10(raw))
    return next(step * power for step in (1, 2, 5, 10) if step * power >= raw)


def _next_step(step: float) -> float:
    """The next round step on the 1-2-5 scale (2 after 1, 5 after 2, 10 after 5)."""
    import math

    power = 10 ** math.floor(math.log10(step) + 1e-9)
    leading = round(step / power)
    return (2 if leading == 1 else 5 if leading == 2 else 10) * power


def timing_value_text(stat: Mapping[str, Any]) -> str:
    """The gutter text for one timing row: median and p95, or the single sample."""
    if stat.get("count") == 1:
        return f"{WIN_TEXT['timingSingle']}, {fmt.stat_value(stat, 'p50' if 'p50' in stat else 'max')}"
    parts = [f"median {fmt.stat_value(stat, 'p50')}"] if fmt.stat_seconds(stat, "p50") is not None else []
    if fmt.stat_seconds(stat, "p95") is not None:
        parts.append(f"p95 {fmt.stat_value(stat, 'p95')}")
    return ", ".join(parts) or f"max {fmt.stat_value(stat, 'max')}"


def timing_rows(timings: Sequence[tuple[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Plot rows in milliseconds; missing or non-numeric statistics are skipped, never drawn as zero."""
    rows = []
    for label, stat in timings:
        median, p95 = fmt.stat_seconds(stat, "p50"), fmt.stat_seconds(stat, "p95")
        low = fmt.stat_seconds(stat, "min")
        high = fmt.stat_seconds(stat, "max")
        known = [value for value in (low, median, p95, high) if value is not None]
        if not known:
            continue
        rows.append({"label": label, "low": min(known), "high": max(known), "median": median, "p95": p95,
                     "text": timing_value_text(stat), "single": stat.get("count") == 1 or min(known) == max(known)})
    return rows


def timing_plot(ctx: ViewerContext, parent: Any, timings: Sequence[tuple[str, Mapping[str, Any]]]) -> Any:
    """Range plot on one shared millisecond axis: bar min to max, filled dot median, ring p95.

    The values sit in a gutter to the right of the axis, so a label never
    crosses a grid line. The p95 ring is drawn after the median dot and a
    little larger, so it still shows when p95 equals the median.
    """
    import tkinter as tk

    rows = timing_rows(timings)
    row_height = ctx.px(WIN_LAYOUT["timingRowHeight"])
    caption, body = ctx.fonts["caption"], ctx.fonts["body"]
    axis_room = caption.metrics("linespace") + ctx.px(SPACE["md"])
    height = row_height * len(rows) + axis_room + ctx.px(SPACE["sm"])
    canvas = tk.Canvas(parent, height=height, highlightthickness=0, borderwidth=0)

    def draw(_event: Any = None) -> None:
        theme = ctx.theme()
        p = theme.p
        canvas.configure(background=p.surface)
        canvas.delete("all")
        label_width = min(ctx.label_width, max((body.measure(row["label"]) for row in rows), default=0) + ctx.px(SPACE["md"]))
        gutter = max((caption.measure(row["text"]) for row in rows), default=0) + ctx.px(SPACE["lg"])
        width = max(float(canvas.winfo_width()), label_width + gutter + ctx.px(SPACE["xxl"]) * 4)
        tick_room = caption.measure("000 ms") / 2.0
        left, right = label_width + tick_room, width - gutter - tick_room
        top_value = max((row["high"] for row in rows), default=0.0)
        step = _nice_step(top_value)
        limit = step * max(1, -(-top_value // step))
        # On a narrow window, step up the 1-2-5 scale until neighboring tick labels have room.
        room = caption.measure(f"{limit:.0f} ms") + ctx.px(SPACE["md"])
        while limit and (right - left) * step / limit < room:
            step = _next_step(step)
            limit = step * max(1, -(-top_value // step))

        def x_of(ms: float) -> float:
            return left + (right - left) * (ms / limit if limit else 0.0)

        axis_y = row_height * len(rows) + ctx.px(SPACE["xs"])
        canvas.create_line(left, axis_y, right, axis_y, fill=p.line)
        tick = 0.0
        while tick <= limit + 1e-9:
            x = x_of(tick)
            canvas.create_line(x, ctx.px(SPACE["xs"]), x, axis_y, fill=mix(p.line, p.surface, 0.35))
            canvas.create_text(x, axis_y + ctx.px(SPACE["xs"]), text=f"{tick:.0f} ms", anchor="n", fill=p.textSecondary, font=caption)
            tick += step
        dot = ctx.px(SPACE["xs"])
        halo = dot + ctx.px(SPACE["xxs"]) + ctx.px(WIN_LAYOUT["hairline"])
        for index, row in enumerate(rows):
            y = row_height * index + row_height / 2
            canvas.create_text(0, y, text=row["label"], anchor="w", fill=p.textSecondary, font=body)
            if not row["single"]:
                canvas.create_line(x_of(row["low"]), y, x_of(row["high"]), y, fill=p.lineStrong, width=ctx.px(5), capstyle="round")
            if row["median"] is not None:
                x = x_of(row["median"])
                canvas.create_oval(x - dot, y - dot, x + dot, y + dot, outline=p.text, fill=p.text)
            if row["p95"] is not None:
                x = x_of(row["p95"])
                canvas.create_oval(x - halo, y - halo, x + halo, y + halo, outline=p.text, width=max(1, ctx.px(1.5)), fill="")
            canvas.create_text(right + tick_room + ctx.px(SPACE["lg"]), y, anchor="w", fill=p.text, font=caption, text=row["text"])

    canvas.bind("<Configure>", draw)
    ctx.add_theme_hook(lambda _theme: draw())
    return canvas


def evidence_window(ctx: ViewerContext, report: Mapping[str, Any], path: Path) -> Any:
    from tkinter import ttk

    kind = str(report.get("kind", ""))
    status = str(report.get("status", ""))
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    view = fmt.report_view(summary)
    window = _window(ctx, f"{fmt.field_label(kind)} evidence report", WIN_LAYOUT["evidenceWindow"])
    _header(ctx, window, f"{fmt.field_label(kind)} evidence report", fmt.humanize_filename(path.name)).pack(fill="x")
    ctx.hairline(window).pack(fill="x")
    _scroll, body, rewrap = _scroll_body(ctx, window)

    verdict = ttk.Frame(body, style="Surface.TFrame")
    verdict.pack(fill="x", pady=(ctx.px(SPACE["sm"]), 0))
    ttk.Label(verdict, text="Declared status", style="Surface.Caption.TLabel").pack(anchor="w")
    ttk.Label(verdict, text=fmt.field_label(status) if status else fmt.NOT_RECORDED, style="Surface.MetricSmall.TLabel").pack(anchor="w")

    # Limitations are the most important caveat on the page, so they sit in a banner right under the verdict.
    limitations = [item for item in report.get("limitations", []) if isinstance(item, str)]
    banner = ttk.Frame(body, style="Tile.TFrame", padding=ctx.px(SPACE["md"]))
    banner.pack(fill="x", pady=(ctx.px(SPACE["md"]), 0))
    ttk.Label(banner, text=fmt.field_label("limitations"), style="Tile.Caption.TLabel").pack(anchor="w", pady=(0, ctx.px(SPACE["xs"])))
    bullet_indent = ctx.fonts["body"].measure("•") + ctx.px(SPACE["sm"])
    for item in limitations or ["None recorded"]:
        row = ttk.Frame(banner, style="Tile.Holder.TFrame")
        row.pack(fill="x", pady=(0, ctx.px(SPACE["xxs"])))
        ttk.Label(row, text="•", style="Tile.Body.TLabel").pack(side="left", anchor="n", padx=(0, ctx.px(SPACE["sm"])))
        text = ttk.Label(row, text=item, style="Tile.Body.TLabel", justify="left")
        text.pack(side="left", anchor="w")
        # The banner adds its own padding on both sides; wrap inside it.
        rewrap.add(lambda width, text=text: text.configure(wraplength=max(1, width - 2 * ctx.px(SPACE["md"]) - bullet_indent)))

    timings = view["timings"]
    if timings and timing_rows(timings):
        _section_title(ctx, body, "Timing")
        timing_plot(ctx, body, timings).pack(fill="x")
        legend = ttk.Label(body, text="Bar: minimum to maximum.  Filled dot: median.  Ring: 95th percentile.",
                           style="Surface.Caption.TLabel", justify="left")
        legend.pack(anchor="w", pady=(ctx.px(SPACE["xxs"]), 0))
        rewrap.add(lambda width: legend.configure(wraplength=width))

    _section_title(ctx, body, fmt.field_label("summary"))
    for title, rows in view["sections"]:
        if title:
            ttk.Label(body, text=title, style="Surface.Strong.TLabel").pack(anchor="w", pady=(ctx.px(SPACE["md"]), ctx.px(SPACE["xs"])))
        _grid(ctx, body, rows, rewrap, label_wrap=True)

    source = report.get("source") if isinstance(report.get("source"), Mapping) else {}
    rows = [
        ("Report file", path.name, "mono"),
        ("Source file", source.get("name"), "mono"),
        ("Source SHA-256 (declared)", source.get("sha256"), "mono"),
        (fmt.field_label("source_schema"), source.get("schema"), "mono"),
        ("Report schema", report.get("schema"), "mono"),
    ]
    rows += [(label, value, "mono") for label, value in view["details"]]
    disclosure = _details(ctx, body, rows, rewrap)
    ttk.Label(disclosure.body, text="Raw summary", style="Surface.Caption.TLabel").pack(
        anchor="w", pady=(ctx.px(SPACE["sm"]), ctx.px(SPACE["xxs"])))
    _raw_text(ctx, disclosure.body, json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), height=14).pack(fill="x")
    return window


__all__ = ["Rewrap", "ViewerContext", "about_window", "evidence_window", "scenario_window", "timing_plot", "timing_rows",
           "timing_value_text", "unavailable_items"]
