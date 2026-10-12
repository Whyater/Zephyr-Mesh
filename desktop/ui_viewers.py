"""About, scenario-run, and evidence-report windows for the Windows replay.

Each window leads with its plot or verdict, then a plain-language summary
built from the shared field labels, then a collapsed Details section holding
raw values, hashes, and versions. Decoding and validation stay in
``scenario_run`` and ``evidence``; nothing here loosens those checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .design_tokens import SPACE, UI_TEXT
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


def _window(ctx: ViewerContext, title: str, size: tuple[int, int], minimum: tuple[int, int]) -> Any:
    import tkinter as tk

    window = tk.Toplevel(ctx.root)
    window.title(title)
    window.geometry(f"{ctx.px(size[0])}x{ctx.px(size[1])}")
    window.minsize(ctx.px(minimum[0]), ctx.px(minimum[1]))
    window.configure(background=ctx.theme().p.background)
    window.transient(ctx.root)
    window.bind("<Escape>", lambda _event: window.destroy())
    ctx.apply_title_bar(window, ctx.theme().dark)
    return window


def _header(ctx: ViewerContext, parent: Any, title: str, subtitle: str) -> Any:
    from tkinter import ttk

    frame = ttk.Frame(parent, style="Surface.TFrame", padding=(ctx.px(SPACE["xl"]), ctx.px(SPACE["lg"])))
    ttk.Label(frame, text=title, style="Surface.Title.TLabel").pack(anchor="w")
    if subtitle:
        ttk.Label(frame, text=subtitle, style="Surface.Secondary.TLabel").pack(anchor="w", pady=(ctx.px(SPACE["xxs"]), 0))
    return frame


def _raw_text(ctx: ViewerContext, parent: Any, text: str, height: int = 10) -> Any:
    import tkinter as tk

    widget = tk.Text(parent, height=height, wrap="none", relief="flat", borderwidth=0, font=ctx.fonts["mono"],
                     padx=ctx.px(SPACE["sm"]), pady=ctx.px(SPACE["sm"]), highlightthickness=1)
    widget.insert("1.0", text)
    widget.configure(state="disabled")

    def recolor(theme: Any) -> None:
        widget.configure(background=theme.p.background, foreground=theme.p.text, highlightbackground=theme.p.line,
                         highlightcolor=theme.p.accent, selectbackground=theme.p.selectionFill, selectforeground=theme.p.text,
                         insertbackground=theme.p.text)

    recolor(ctx.theme())
    ctx.add_theme_hook(recolor)
    return widget


def _section(ctx: ViewerContext, parent: Any, title: str | None, rows: Sequence[tuple], wrap: int, label_width: int) -> None:
    from tkinter import ttk

    if title:
        ttk.Label(parent, text=title, style="Surface.Strong.TLabel").pack(anchor="w", pady=(ctx.px(SPACE["md"]), ctx.px(SPACE["xs"])))
    grid = KeyValueGrid(parent, label_width=ctx.px(label_width), wrap=wrap, scale=ctx.scale)
    grid.frame.pack(fill="x")
    grid.set_rows(rows)


def _scroll_body(ctx: ViewerContext, window: Any) -> tuple[Any, Any]:
    from tkinter import ttk

    scroll = ScrollFrame(window)
    scroll.outer.pack(fill="both", expand=True)
    scroll.apply_theme(ctx.theme())
    ctx.add_theme_hook(scroll.apply_theme)
    body = ttk.Frame(scroll.inner, style="Surface.TFrame", padding=(ctx.px(SPACE["xl"]), ctx.px(SPACE["md"]),
            ctx.px(SPACE["xl"]), ctx.px(SPACE["xl"])))
    body.pack(fill="both", expand=True)

    def wheel(event: Any) -> None:
        widget = window.winfo_containing(event.x_root, event.y_root)
        if widget is not None and scroll.contains(widget):
            scroll.scroll(event.delta)

    window.bind("<MouseWheel>", wheel, add="+")
    return scroll, body


def _sentence(value: Any) -> str:
    """Show a raw data string with a leading capital; the words themselves are unchanged."""
    text = str(value)
    return text[:1].upper() + text[1:]


def unavailable_items(run: Mapping[str, Any]) -> str:
    """What this replay cannot show, stated from the data rather than assumed."""
    items = ["live radio link", "camera", "motor commands"]
    agents = [agent for step in run.get("steps", []) for agent in step.get("agents", [])]
    if not any(agent.get("battery_pct") is not None for agent in agents):
        items.append("battery level")
    text = ", ".join(items)
    return text[:1].upper() + text[1:]


def about_window(ctx: ViewerContext, run: Mapping[str, Any], details: Sequence[tuple[str, Any, str]]) -> Any:
    from tkinter import ttk

    window = _window(ctx, UI_TEXT["honesty"]["aboutButton"], (640, 620), (520, 420))
    _header(ctx, window, UI_TEXT["honesty"]["aboutButton"], fmt.plain_run_name(run)).pack(fill="x")
    ttk.Frame(window, style="Line.TFrame", height=1).pack(fill="x")
    _scroll, body = _scroll_body(ctx, window)
    wrap = ctx.px(640 - 2 * SPACE["xl"] - 190 - 24)
    ttk.Label(body, text="This window replays a recorded simulation file. It is not connected to any drone.",
              style="Surface.Body.TLabel", wraplength=ctx.px(560)).pack(anchor="w", pady=(ctx.px(SPACE["xs"]), ctx.px(SPACE["md"])))
    rows = []
    for field in ("status", "evidence_boundary", "command_authority", "failsafe"):
        if field in run:
            rows.append((fmt.field_label(field), _sentence(run[field])))
    rows.append(("Not in this replay", unavailable_items(run)))
    _section(ctx, body, None, rows, wrap, 150)
    command = run.get("generation_command")
    if command:
        ttk.Label(body, text=fmt.field_label("generation_command"), style="Surface.Strong.TLabel").pack(
            anchor="w", pady=(ctx.px(SPACE["md"]), ctx.px(SPACE["xs"])))
        command_row = ttk.Frame(body, style="Surface.TFrame")
        command_row.pack(fill="x")
        def copy() -> None:
            window.clipboard_clear()
            window.clipboard_append(str(command))

        ttk.Button(command_row, text="Copy", style="Tool.TButton", command=copy).pack(side="right",
                padx=(ctx.px(SPACE["sm"]), 0), anchor="n")
        text = _raw_text(ctx, command_row, str(command), height=1 if len(str(command)) <= 72 else 2)
        text.configure(wrap="word", font=ctx.fonts["body"])
        text.pack(side="left", fill="x", expand=True)
    ttk.Frame(body, style="Line.TFrame", height=1).pack(fill="x", pady=(ctx.px(SPACE["lg"]), ctx.px(SPACE["xs"])))
    disclosure = Disclosure(body, "Details", ctx.images, pad=ctx.px(SPACE["xs"]))
    disclosure.frame.pack(fill="x")
    grid = KeyValueGrid(disclosure.body, label_width=ctx.px(150), wrap=wrap, scale=ctx.scale)
    grid.frame.pack(fill="x", padx=(ctx.px(4), 0))
    grid.set_rows([(label, None if value is None else value, kind) for label, value, kind in details])
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
    window = _window(ctx, f"Scenario run: {path.name}", (760, 780), (620, 560))
    _header(ctx, window, "Scenario run", path.name).pack(fill="x")
    ttk.Frame(window, style="Line.TFrame", height=1).pack(fill="x")
    _scroll, body = _scroll_body(ctx, window)
    plot = tk.Canvas(body, height=ctx.px(260), highlightthickness=1, borderwidth=0)
    plot.pack(fill="x", pady=(ctx.px(SPACE["sm"]), ctx.px(SPACE["sm"])))
    cursor = ScenarioRunCursor(len(run.frames))
    controls = ttk.Frame(body, style="Surface.TFrame")
    controls.pack(fill="x")
    frame_text = ttk.Label(body, style="Surface.Secondary.TLabel")
    frame_text.pack(anchor="w", pady=(ctx.px(SPACE["xs"]), 0))

    def draw() -> None:
        theme = ctx.theme()
        p = theme.p
        floor = theme.scene.floor.hex
        plot.configure(background=floor, highlightbackground=p.line, highlightcolor=p.accent)
        plot.delete("all")
        geometry = scenario_frame_geometry(run.frames[cursor.index])
        width = max(1.0, float(plot.winfo_width()))
        height = max(1.0, float(plot.winfo_height()))
        rail_space = 72.0 * ctx.scale
        top_width = max(width - rail_space, 1.0)
        caption = ctx.fonts["caption"]
        plot.create_text(ctx.px(SPACE["md"]), ctx.px(SPACE["md"]), text="Top view", fill=p.textSecondary, anchor="nw", font=caption)
        keep_out = theme.scene.keep_out_volume
        for label, center_x, center_y, radius in geometry.keep_out_spheres:
            x, y = project_scenario_point((center_x, center_y), geometry.bounds, top_width, height)
            radius_px = project_scenario_radius(radius, geometry.bounds, top_width, height)
            plot.create_oval(x - radius_px, y - radius_px, x + radius_px, y + radius_px, outline=keep_out.hex,
                             fill=mix(keep_out.hex, floor, keep_out.alpha), width=max(1, ctx.px(1.5)))
            plot.create_text(x, y + radius_px + 4 * ctx.scale, text=label, fill=p.textSecondary, anchor="n", font=caption)
        target_color = theme.scene.target_ring.hex
        target_x, target_y = project_scenario_point(geometry.target, geometry.bounds, top_width, height)
        ring = 7 * ctx.scale
        plot.create_oval(target_x - ring, target_y - ring, target_x + ring, target_y + ring, outline=target_color, width=max(2, ctx.px(2)))
        for _, agent_x, agent_y, active in geometry.agents:
            x, y = project_scenario_point((agent_x, agent_y), geometry.bounds, top_width, height)
            color = theme.scene.drone_body.hex if active else mix(theme.scene.drone_body.hex, floor, 0.4)
            dot = 3.5 * ctx.scale
            plot.create_oval(x - dot, y - dot, x + dot, y + dot, fill=color, outline=color)
        rail_x = clamp_scenario_rail_x(width, min(width - 26.0 * ctx.scale, top_width + 36.0 * ctx.scale))
        rail_top, rail_bottom = scenario_rail_endpoints(height)
        plot.create_line(rail_x, rail_top, rail_x, rail_bottom, fill=p.lineStrong)
        plot.create_text(rail_x, ctx.px(SPACE["md"]), text="Altitude", fill=p.textSecondary, anchor="n", font=caption)
        target_z = project_scenario_altitude(geometry.target_altitude, geometry.altitude_bounds, height, inset=30.0)
        plot.create_line(rail_x - 6 * ctx.scale, target_z, rail_x + 6 * ctx.scale, target_z, fill=target_color, width=max(2, ctx.px(2)))
        for altitude, (_, _, _, active) in zip(geometry.agent_altitudes, geometry.agents):
            y = project_scenario_altitude(altitude, geometry.altitude_bounds, height, inset=30.0)
            color = theme.scene.drone_body.hex if active else mix(theme.scene.drone_body.hex, floor, 0.4)
            dot = 3 * ctx.scale
            plot.create_oval(rail_x - dot, y - dot, rail_x + dot, y + dot, fill=color, outline=color)
        for center_z, (_label, _cx, _cy, radius) in zip(geometry.keep_out_altitudes, geometry.keep_out_spheres):
            center_y = project_scenario_altitude(center_z, geometry.altitude_bounds, height, inset=30.0)
            radius_y = project_scenario_altitude_radius(radius, geometry.altitude_bounds, height, inset=30.0)
            plot.create_line(rail_x + 10 * ctx.scale, center_y - radius_y, rail_x + 10 * ctx.scale, center_y + radius_y,
                             fill=keep_out.hex, width=max(2, ctx.px(3)))

    def update(index: int) -> None:
        cursor.seek(index)
        frame_text.configure(text=_frame_line(run, cursor))
        slider.set_state(cursor.index, cursor.frame_count, [])
        previous.state(["!disabled"] if cursor.index > 0 else ["disabled"])
        restart.state(["!disabled"] if cursor.index > 0 else ["disabled"])
        following.state(["!disabled"] if cursor.index < cursor.frame_count - 1 else ["disabled"])
        draw()

    restart = ttk.Button(controls, style="Icon.TButton", image=ctx.images.icon("restart", "text", 14),
            command=lambda: update(cursor.reset()))
    restart.pack(side="left", padx=(0, ctx.px(SPACE["xs"])))
    previous = ttk.Button(controls, style="Icon.TButton", image=ctx.images.icon("step_back", "text", 14),
            command=lambda: update(cursor.step(-1)))
    previous.pack(side="left", padx=(0, ctx.px(SPACE["xs"])))
    following = ttk.Button(controls, style="Icon.TButton", image=ctx.images.icon("step_forward", "text", 14),
            command=lambda: update(cursor.step()))
    following.pack(side="left")
    slider = TimelineSlider(controls, lambda index: update(index), ctx.scale)
    slider.canvas.pack(side="left", fill="x", expand=True, padx=(ctx.px(SPACE["lg"]), 0))
    slider.apply_theme(ctx.theme())

    def recolor(theme: Any) -> None:
        slider.apply_theme(theme)
        draw()

    ctx.add_theme_hook(recolor)
    plot.bind("<Configure>", lambda _event: draw())

    parameters = run.parameters
    target_link = parameters.get("target_link", {})
    wrap = ctx.px(760 - 2 * SPACE["xl"] - 190 - 24)
    rows = [
        (fmt.field_label("agent_count"), summary["agent_count"]),
        (fmt.field_label("frame_count"), summary["frame_count"]),
        (fmt.field_label("dt_s"), fmt.format_scalar(summary["dt_s"])),
        (fmt.field_label("duration_s"), fmt.format_scalar(summary["dt_s"] * max(summary["frame_count"] - 1, 0))),
        (fmt.field_label("link_event_count"), summary["link_event_count"]),
        (fmt.field_label("profile_count"), summary["profile_count"]),
        (fmt.field_label("target_link"), fmt.link_text(target_link)),
        (fmt.field_label("neighbor_link"), fmt.link_text(parameters.get("neighbor_link", {}))),
        (fmt.field_label("status"), _sentence(summary["status"])),
        (fmt.field_label("evidence_boundary"), _sentence(summary["evidence_boundary"])),
    ]
    _section(ctx, body, "Summary", rows, wrap, 170)
    ttk.Frame(body, style="Line.TFrame", height=1).pack(fill="x", pady=(ctx.px(SPACE["lg"]), ctx.px(SPACE["xs"])))
    disclosure = Disclosure(body, "Details", ctx.images, pad=ctx.px(SPACE["xs"]))
    disclosure.frame.pack(fill="x")
    details = KeyValueGrid(disclosure.body, label_width=ctx.px(170), wrap=wrap, scale=ctx.scale)
    details.frame.pack(fill="x", padx=(ctx.px(4), 0))
    details.set_rows([
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
    ])
    ttk.Label(disclosure.body, text="Raw parameters", style="Surface.Caption.TLabel").pack(anchor="w",
            pady=(ctx.px(SPACE["sm"]), ctx.px(SPACE["xxs"])))
    _raw_text(ctx, disclosure.body, json.dumps(parameters, indent=2, sort_keys=True, ensure_ascii=False), height=12).pack(fill="x")
    update(0)
    return window


EVIDENCE_LABEL_WIDTH = 210


def _nice_step(span: float) -> float:
    """A round tick step (1, 2, or 5 times a power of ten) for about four ticks."""
    import math

    raw = max(span, 1e-9) / 4.0
    power = 10 ** math.floor(math.log10(raw))
    return next(step * power for step in (1, 2, 5, 10) if step * power >= raw)


def timing_plot(ctx: ViewerContext, parent: Any, timings: Sequence[tuple[str, Mapping[str, Any]]]) -> Any:
    """Small range plot on one shared millisecond axis: bar min to max, filled dot median, ring p95."""
    import tkinter as tk

    row_height = ctx.px(30)
    label_width = ctx.px(EVIDENCE_LABEL_WIDTH)
    height = row_height * len(timings) + ctx.px(46)
    canvas = tk.Canvas(parent, height=height, highlightthickness=0, borderwidth=0)

    def draw(_event: Any = None) -> None:
        theme = ctx.theme()
        p = theme.p
        canvas.configure(background=p.surface)
        canvas.delete("all")
        width = max(float(canvas.winfo_width()), label_width + 200.0)
        left, right = label_width, width - ctx.px(110)
        top_value = max(float(stat.get("max", 0) or 0) for _label, stat in timings) * 1000.0
        step = _nice_step(top_value)
        limit = step * max(1, -(-top_value // step))
        def x_of(ms: float) -> float:
            return left + (right - left) * (ms / limit if limit else 0.0)
        caption, body = ctx.fonts["caption"], ctx.fonts["body"]
        axis_y = row_height * len(timings) + ctx.px(10)
        canvas.create_line(left, axis_y, right, axis_y, fill=p.line)
        tick = 0.0
        while tick <= limit + 1e-9:
            x = x_of(tick)
            canvas.create_line(x, ctx.px(4), x, axis_y, fill=mix(p.line, p.surface, 0.55))
            canvas.create_text(x, axis_y + ctx.px(4), text=f"{tick:.0f} ms", anchor="n", fill=p.textSecondary, font=caption)
            tick += step
        dot = ctx.px(4)
        for index, (label, stat) in enumerate(timings):
            y = row_height * index + row_height / 2 + ctx.px(2)
            canvas.create_text(0, y, text=label, anchor="w", fill=p.textSecondary, font=body)
            low = float(stat.get("min", stat.get("p50", 0)) or 0) * 1000.0
            high = float(stat.get("max", stat.get("p95", 0)) or 0) * 1000.0
            canvas.create_line(x_of(low), y, x_of(high), y, fill=p.lineStrong, width=ctx.px(5), capstyle="round")
            if stat.get("p95") is not None:
                x = x_of(float(stat["p95"]) * 1000.0)
                canvas.create_oval(x - dot - 1, y - dot - 1, x + dot + 1, y + dot + 1, outline=p.text, width=max(1, ctx.px(1.5)),
                                   fill=p.surface)
            if stat.get("p50") is not None:
                x = x_of(float(stat["p50"]) * 1000.0)
                canvas.create_oval(x - dot, y - dot, x + dot, y + dot, outline=p.text, fill=p.text)
                canvas.create_text(x_of(high) + ctx.px(12), y, anchor="w", fill=p.text, font=caption,
                                   text=f"median {fmt.stat_value(stat, 'p50')}")

    canvas.bind("<Configure>", draw)
    ctx.add_theme_hook(lambda _theme: draw())
    return canvas


def evidence_window(ctx: ViewerContext, report: Mapping[str, Any], path: Path) -> Any:
    from tkinter import ttk

    kind = str(report.get("kind", ""))
    status = str(report.get("status", ""))
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    view = fmt.report_view(summary)
    window = _window(ctx, f"{fmt.field_label(kind)} evidence report", (720, 780), (580, 480))
    _header(ctx, window, f"{fmt.field_label(kind)} evidence report", fmt.humanize_filename(path.name)).pack(fill="x")
    ttk.Frame(window, style="Line.TFrame", height=1).pack(fill="x")
    _scroll, body = _scroll_body(ctx, window)
    label_width = ctx.px(EVIDENCE_LABEL_WIDTH)
    wrap_full = ctx.px(720 - 2 * SPACE["xl"] - 24)
    wrap = ctx.px(720 - 2 * SPACE["xl"] - EVIDENCE_LABEL_WIDTH - 36)

    verdict = ttk.Frame(body, style="Surface.TFrame")
    verdict.pack(fill="x", pady=(ctx.px(SPACE["sm"]), 0))
    ttk.Label(verdict, text="Declared status", style="Surface.Caption.TLabel").pack(anchor="w")
    ttk.Label(verdict, text=fmt.field_label(status) if status else fmt.NOT_RECORDED, style="Surface.MetricSmall.TLabel").pack(anchor="w")
    timings = view["timings"]
    if timings:
        ttk.Label(body, text="Timing", style="Surface.Strong.TLabel").pack(anchor="w", pady=(ctx.px(SPACE["md"]), ctx.px(SPACE["xs"])))
        timing_plot(ctx, body, timings).pack(fill="x")
        legend = "Bar: minimum to maximum.  Filled dot: median.  Ring: 95th percentile."
        ttk.Label(body, text=legend, style="Surface.Caption.TLabel").pack(anchor="w", pady=(ctx.px(SPACE["xxs"]), 0))
    limitations = [item for item in report.get("limitations", []) if isinstance(item, str)]
    ttk.Label(body, text=fmt.field_label("limitations"), style="Surface.Strong.TLabel").pack(
        anchor="w", pady=(ctx.px(SPACE["md"]), ctx.px(SPACE["xs"])))
    for item in limitations or ["None recorded"]:
        row = ttk.Frame(body, style="Surface.TFrame")
        row.pack(fill="x", pady=(0, ctx.px(SPACE["xs"])))
        ttk.Label(row, text="\u2022", style="Surface.Body.TLabel").pack(side="left", anchor="n", padx=(0, ctx.px(SPACE["sm"])))
        ttk.Label(row, text=item, style="Surface.Body.TLabel", wraplength=wrap_full - ctx.px(16), justify="left").pack(
            side="left", anchor="w")

    ttk.Label(body, text=fmt.field_label("summary"), style="Surface.Headline.TLabel").pack(anchor="w", pady=(ctx.px(SPACE["lg"]), 0))
    for title, rows in view["sections"]:
        if title:
            ttk.Label(body, text=title, style="Surface.Strong.TLabel").pack(anchor="w", pady=(ctx.px(SPACE["md"]), ctx.px(SPACE["xs"])))
        grid = KeyValueGrid(body, label_width=label_width, wrap=wrap, scale=ctx.scale, label_wrap=label_width - ctx.px(12))
        grid.frame.pack(fill="x", pady=(ctx.px(SPACE["xs"]) if not title else 0, 0))
        grid.set_rows(rows)

    source = report.get("source") if isinstance(report.get("source"), Mapping) else {}
    ttk.Frame(body, style="Line.TFrame", height=1).pack(fill="x", pady=(ctx.px(SPACE["lg"]), ctx.px(SPACE["xs"])))
    disclosure = Disclosure(body, "Details", ctx.images, pad=ctx.px(SPACE["xs"]))
    disclosure.frame.pack(fill="x")
    details = KeyValueGrid(disclosure.body, label_width=label_width - ctx.px(4), wrap=wrap, scale=ctx.scale,
                           label_wrap=label_width - ctx.px(16))
    details.frame.pack(fill="x", padx=(ctx.px(4), 0))
    rows = [
        ("Report file", path.name, "mono"),
        ("Source file", source.get("name"), "mono"),
        ("Source SHA-256 (declared)", source.get("sha256"), "mono"),
        (fmt.field_label("source_schema"), source.get("schema"), "mono"),
        ("Report schema", report.get("schema"), "mono"),
    ]
    rows += [(label, value, "mono") for label, value in view["details"]]
    details.set_rows(rows)
    ttk.Label(disclosure.body, text="Raw summary", style="Surface.Caption.TLabel").pack(
        anchor="w", pady=(ctx.px(SPACE["sm"]), ctx.px(SPACE["xxs"])))
    _raw_text(ctx, disclosure.body, json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), height=14).pack(fill="x")
    return window


__all__ = ["ViewerContext", "about_window", "evidence_window", "scenario_window", "timing_plot", "unavailable_items"]
