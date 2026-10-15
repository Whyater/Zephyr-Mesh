"""Delay-and-loss small multiples for the Results window, drawn on one Tk canvas.

Four panels (no loss, then 10, 30, and 50 % loss of the chosen type) share
one x scale (0 to 300 ms) and one y scale computed once from the whole
fixture, so nothing jumps when the controls change. The recovery tracker is
a solid accent line with filled dots and its interval band; the basic
tracker is a dashed neutral line with hollow dots and interval caps. The
red failure line is the only alert color on the chart, and it carries a
word label. All geometry choices that are not drawing live in
``results_data`` and are unit tested there.
"""
from __future__ import annotations

from typing import Any, Callable

from .design_tokens import LAYOUT, SPACE, WIN_LAYOUT
from . import results_data as rd
from .ui_widgets import Tooltip


ARMS = ("naive", "recovery")  # Up moves toward the basic tracker, Down toward recovery


def _width(ctx: Any, token: float, minimum: int = 1) -> float:
    """A stroke width in pixels for the current DPI, never under ``minimum``."""
    return max(float(minimum), float(token) * ctx.scale)


class DelayChart:
    """The 2 x 2 grid of panels. Selection and hover are overlays; the data layer redraws only when it changes."""

    def __init__(self, ctx: Any, parent: Any, data: rd.ResultsData, on_select: Callable[[rd.Point | None], None]) -> None:
        import tkinter as tk

        self.ctx = ctx
        self.data = data
        self.on_select = on_select
        self.motion = "constant_velocity"
        self.loss_type = "independent"
        self.selected: rd.Point | None = None
        self.hover: rd.Point | None = None
        self.cursor_panel = 0
        self.focused = False
        self.draw_count = 0
        self._points: list[tuple[float, float, rd.Point]] = []
        self._panel_boxes: list[tuple[float, float, float, float]] = []
        self._drawn_key: tuple[Any, ...] | None = None
        fonts = ctx.fonts
        self.title_h = int(fonts["bodyStrong"].metrics("linespace")) + ctx.px(SPACE["xs"])
        self.tick_h = int(fonts["caption"].metrics("linespace")) + ctx.px(SPACE["xs"])
        self.axis_title = int(fonts["caption"].metrics("linespace")) + ctx.px(SPACE["xs"])
        self.panel_h = ctx.px(WIN_LAYOUT["chartPanelHeight"])
        self.gap = ctx.px(SPACE["lg"])
        height = 2 * self.panel_h + self.gap + self.axis_title
        self.canvas = tk.Canvas(parent, height=height, highlightthickness=0, borderwidth=0, takefocus=1)
        canvas = self.canvas
        canvas.bind("<Configure>", lambda _event: self.redraw())
        canvas.bind("<Motion>", self._on_motion)
        canvas.bind("<Leave>", lambda _event: self._set_hover(None))
        canvas.bind("<Button-1>", self._on_click)
        canvas.bind("<FocusIn>", lambda _event: self._set_focus(True))
        canvas.bind("<FocusOut>", lambda _event: self._set_focus(False))
        for key in ("Left", "Right", "Up", "Down"):
            canvas.bind(f"<{key}>", lambda _event, key=key: self.key(key))
        canvas.bind("<bracketright>", lambda _event: self.key("next") or "break")
        canvas.bind("<bracketleft>", lambda _event: self.key("previous") or "break")
        canvas.bind("<Tab>", lambda _event: self.key("next"))
        for sequence in ("<Shift-Tab>", "<ISO_Left_Tab>"):
            try:
                canvas.bind(sequence, lambda _event: self.key("previous"))
            except tk.TclError:
                pass
        canvas.bind("<Escape>", lambda _event: self.key("Escape"))
        self.tooltip = Tooltip(canvas, "", ctx.theme, fonts["caption"], scale=ctx.scale, text_for=self._tip)
        # A hint still waiting for its delay when the window closes must not fire into a destroyed widget.
        canvas.bind("<Destroy>", lambda _event: self.tooltip._hide(), add="+")

    # ---- state ---------------------------------------------------------------------------
    def panel_keys(self) -> list[tuple[str, float]]:
        return rd.panels(self.loss_type)

    def panel_of(self, point: rd.Point) -> int:
        keys = self.panel_keys()
        key = (point.loss_model, point.loss_probability)
        return keys.index(key) if key in keys else 0

    def point_in_panel(self, panel: int, arm: str, index: int) -> rd.Point:
        model, level = self.panel_keys()[panel]
        return rd.Point(self.motion, model, level, arm, index)

    def set_view(self, motion: str, loss_type: str) -> None:
        """Change path or loss type; a selection moves to the same panel, tracker, and delay."""
        if (motion, loss_type) == (self.motion, self.loss_type):
            return
        panel = self.panel_of(self.selected) if self.selected else None
        self.motion, self.loss_type = motion, loss_type
        self.hover = None
        if self.selected is not None and panel is not None:
            self.selected = self.point_in_panel(panel, self.selected.arm, self.selected.index)
            self.on_select(self.selected)
        self.redraw()

    def select(self, point: rd.Point | None, *, notify: bool = True) -> None:
        if point is not None:
            self.cursor_panel = self.panel_of(point)
        if point == self.selected:
            return
        self.selected = point
        self.draw_overlay()
        if notify:
            self.on_select(point)

    def key(self, key: str) -> str | None:
        """Keyboard handling; returns "break" when the key was used, so Tab and Escape fall through at the ends."""
        count, delays = len(self.panel_keys()), len(self.data.delays_s)
        if key in ("next", "previous"):
            target = rd.step_panel(self.cursor_panel, 1 if key == "next" else -1, count)
            if target is None:
                return None
            self.cursor_panel = target
            if self.selected is not None:
                self.select(self.point_in_panel(target, self.selected.arm, self.selected.index))
            else:
                self.draw_overlay()
            return "break"
        if key == "Escape":
            if self.selected is None:
                return None
            self.select(None)
            return "break"
        if self.selected is None:
            self.select(self.point_in_panel(self.cursor_panel, "recovery", 0))
            return "break"
        self.select(rd.move_point(self.selected, key, delays, ARMS))
        return "break"

    # ---- events --------------------------------------------------------------------------
    def _hit(self, x: float, y: float) -> rd.Point | None:
        return rd.nearest(self._points, x, y, self.ctx.px(WIN_LAYOUT["hitRadius"]))

    def _tip(self, x: int, y: int) -> str | None:
        point = self._hit(x, y)
        return rd.hover_text(self.data, point) if point is not None else None

    def _on_motion(self, event: Any) -> None:
        self._set_hover(self._hit(event.x, event.y))

    def _set_hover(self, point: rd.Point | None) -> None:
        if point != self.hover:
            self.hover = point
            self.canvas.configure(cursor="hand2" if point is not None else "")
            self.draw_overlay()

    def _on_click(self, event: Any) -> None:
        self.canvas.focus_set()
        point = self._hit(event.x, event.y)
        if point is not None:
            self.select(point)
            return
        for index, box in enumerate(self._panel_boxes):
            if box[0] <= event.x <= box[2] and box[1] <= event.y <= box[3]:
                self.cursor_panel = index
                self.draw_overlay()

    def _set_focus(self, value: bool) -> None:
        self.focused = value
        self.draw_overlay()

    def apply_theme(self, _theme: Any = None) -> None:
        self._drawn_key = None
        self.redraw()

    # ---- geometry ------------------------------------------------------------------------
    def _layout(self, width: float) -> list[tuple[tuple[float, float, float, float], rd.Scale]]:
        """Panel boxes and plot scales. Every panel has the same plot size; the left column and the bottom
        row also carry the shared axis titles in the margin outside the panels."""
        ctx = self.ctx
        y_title = self.axis_title
        column = (width - y_title - self.gap) / 2.0
        inset = ctx.px(WIN_LAYOUT["chartPlotInset"])
        # Room for half of the last tick label ("300"), so it is never clipped at the panel edge.
        right = max(ctx.px(SPACE["sm"]), ctx.fonts["caption"].measure(str(rd.X_TICKS_MS[-1])) / 2.0 + ctx.px(SPACE["xxs"]))
        top = ctx.px(SPACE["xs"])
        result = []
        for index in range(4):
            row, col = divmod(index, 2)
            left = y_title + col * (column + self.gap)
            top_y = row * (self.panel_h + self.gap)
            box = (left, top_y, left + column, top_y + self.panel_h)
            scale = rd.Scale(x0=left + inset, x1=left + column - right, y0=top_y + self.panel_h - self.tick_h,
                             y1=top_y + self.title_h + top, x_max_ms=float(rd.X_TICKS_MS[-1]), y_max=self.data.y_max)
            result.append((box, scale))
        return result

    def label_slots(self, layout: list[tuple[Any, rd.Scale]], width: float, height: float) -> list[int | None]:
        """Where each panel puts the failure-line label, clear of that panel's data: 0 right end above the
        line, 1 right end below, 2 left end above, 3 left end below, or ``None`` (top right of the panel,
        above the plot) when the data covers all four."""
        threshold = float(self.data.s6["failure_p95_m"])
        pad = self.ctx.px(SPACE["xxs"])
        slots: list[int | None] = []
        for (_box, scale), (model, level) in zip(layout, self.panel_keys()):
            line_y = scale.y(threshold)
            candidates = [
                (scale.x1 - width, line_y - height - pad, scale.x1, line_y - pad),
                (scale.x1 - width, line_y + pad, scale.x1, line_y + height + pad),
                (scale.x0 + pad, line_y - height - pad, scale.x0 + pad + width, line_y - pad),
                (scale.x0 + pad, line_y + pad, scale.x0 + pad + width, line_y + height + pad),
            ]
            obstacles = []
            for arm in ARMS:
                series = self.data.series(self.motion, model, level, arm)
                if series is None:
                    continue
                xs = [scale.x(rd.delay_ms(delay)) for delay in self.data.delays_s]
                ys = [scale.y(value) for value in series["p95_median_m"]]
                for i, x in enumerate(xs):
                    high, low = scale.y(series["p95_ci_high_m"][i]), scale.y(series["p95_ci_low_m"][i])
                    obstacles.append((x - 2 * pad, high - pad, x + 2 * pad, low + pad))
                for i in range(len(xs) - 1):  # the connecting lines, sampled
                    for step in range(9):
                        t = step / 8.0
                        sx, sy = xs[i] + (xs[i + 1] - xs[i]) * t, ys[i] + (ys[i + 1] - ys[i]) * t
                        obstacles.append((sx - pad, sy - pad, sx + pad, sy + pad))
            slots.append(rd.free_slot(candidates, obstacles))
        return slots

    # ---- drawing -------------------------------------------------------------------------
    def redraw(self) -> None:
        """Redraw the data layer only when the size, theme, or view changed."""
        canvas = self.canvas
        width = float(canvas.winfo_width())
        if width <= 1:
            return
        theme = self.ctx.theme()
        key = (width, theme.name, self.motion, self.loss_type)
        if key != self._drawn_key:
            self._drawn_key = key
            self._draw(width, theme)
        self.draw_overlay()

    def _draw(self, width: float, theme: Any) -> None:
        ctx, canvas, p = self.ctx, self.canvas, theme.p
        fonts = ctx.fonts
        self.draw_count += 1
        canvas.configure(background=p.surface)
        canvas.delete("all")
        layout = self._layout(width)
        self._panel_boxes = [box for box, _scale in layout]
        self._points = []
        hairline = _width(ctx, WIN_LAYOUT["strokeHairline"])
        line_width = _width(ctx, LAYOUT["chartLineWidth"])
        marker = ctx.px(WIN_LAYOUT["chartMarkerSize"]) / 2.0
        cap = ctx.px(WIN_LAYOUT["chartCapWidth"]) / 2.0
        dash = tuple(max(1, int(round(value * ctx.scale))) for value in WIN_LAYOUT["dashPattern"])
        ticks = rd.y_ticks(self.data.y_max)
        threshold = float(self.data.s6["failure_p95_m"])
        label = rd.TEXT["failureLineLabel"].format(value=f"{threshold:g}")
        label_w = fonts["caption"].measure(label)
        label_h = int(fonts["caption"].metrics("linespace"))
        slots = self.label_slots(layout, label_w, label_h)
        for panel, ((box, scale), (model, level)) in enumerate(zip(layout, self.panel_keys())):
            row, col = divmod(panel, 2)
            canvas.create_text(box[0] + ctx.px(SPACE["xxs"]), box[1], text=rd.panel_title(level), anchor="nw",
                               fill=p.text, font=fonts["bodyStrong"])
            for tick in ticks:
                y = scale.y(tick)
                if tick > 0:
                    canvas.create_line(scale.x0, y, scale.x1, y, fill=p.line, width=hairline)
                canvas.create_text(scale.x0 - ctx.px(SPACE["xs"]), y, text=rd.tick_text(tick, ticks), anchor="e",
                                   fill=p.textSecondary, font=fonts["caption"])
            canvas.create_line(scale.x0, scale.y1, scale.x0, scale.y0, fill=p.lineStrong, width=hairline)
            canvas.create_line(scale.x0, scale.y0, scale.x1, scale.y0, fill=p.lineStrong, width=hairline)
            for ms in rd.X_TICKS_MS:
                x = scale.x(ms)
                canvas.create_line(x, scale.y0, x, scale.y0 + ctx.px(SPACE["xxs"]), fill=p.lineStrong, width=hairline)
                canvas.create_text(x, scale.y0 + ctx.px(SPACE["xxs"]), text=str(ms), anchor="n", fill=p.textSecondary,
                                   font=fonts["caption"])
            if col == 0:
                canvas.create_text(box[0] - self.axis_title / 2.0, (scale.y0 + scale.y1) / 2.0, text=rd.TEXT["axisError"],
                                   angle=90, fill=p.textSecondary, font=fonts["caption"])
            if row == 1:
                canvas.create_text((scale.x0 + scale.x1) / 2.0, box[3] + ctx.px(SPACE["xs"]), text=rd.TEXT["axisDelay"],
                                   anchor="n", fill=p.textSecondary, font=fonts["caption"])
            naive = self.data.series(self.motion, model, level, "naive")
            recovery = self.data.series(self.motion, model, level, "recovery")
            xs = [scale.x(rd.delay_ms(delay)) for delay in self.data.delays_s]
            if recovery is not None:
                band = rd.band_polygon(scale, self.data.delays_s, recovery["p95_ci_low_m"], recovery["p95_ci_high_m"])
                canvas.create_polygon(*band, fill=p.bandFill, outline="")
            line_y = scale.y(threshold)
            canvas.create_line(scale.x0, line_y, scale.x1, line_y, fill=p.alert, width=_width(ctx, LAYOUT["failureLineWidth"]))
            xxs = ctx.px(SPACE["xxs"])
            slot = slots[panel]
            if slot is None:
                canvas.create_text(box[2] - xxs, box[1] + xxs, text=label, anchor="ne", fill=p.textSecondary, font=fonts["caption"])
            else:
                label_x = scale.x1 - label_w if slot in (0, 1) else scale.x0 + xxs
                label_y = line_y - xxs if slot in (0, 2) else line_y + xxs
                canvas.create_text(label_x, label_y, text=label, anchor="sw" if slot in (0, 2) else "nw", fill=p.textSecondary,
                                   font=fonts["caption"])
            if naive is not None:
                ys = [scale.y(value) for value in naive["p95_median_m"]]
                for x, low, high in zip(xs, naive["p95_ci_low_m"], naive["p95_ci_high_m"]):
                    top, bottom = scale.y(high), scale.y(low)
                    canvas.create_line(x, top, x, bottom, fill=p.lineStrong, width=hairline)
                    canvas.create_line(x - cap, top, x + cap, top, fill=p.lineStrong, width=hairline)
                    canvas.create_line(x - cap, bottom, x + cap, bottom, fill=p.lineStrong, width=hairline)
                canvas.create_line(*[value for pair in zip(xs, ys) for value in pair], fill=p.lineStrong, width=line_width, dash=dash)
                for index, (x, y) in enumerate(zip(xs, ys)):
                    canvas.create_oval(x - marker, y - marker, x + marker, y + marker, outline=p.lineStrong, fill=p.surface,
                                       width=line_width)
                    self._points.append((x, y, rd.Point(self.motion, model, level, "naive", index)))
            if recovery is not None:
                ys = [scale.y(value) for value in recovery["p95_median_m"]]
                canvas.create_line(*[value for pair in zip(xs, ys) for value in pair], fill=p.accent, width=line_width)
                for index, (x, y) in enumerate(zip(xs, ys)):
                    canvas.create_oval(x - marker, y - marker, x + marker, y + marker, outline=p.accent, fill=p.accent)
                    self._points.append((x, y, rd.Point(self.motion, model, level, "recovery", index)))

    def point_xy(self, point: rd.Point) -> tuple[float, float] | None:
        return next(((x, y) for x, y, key in self._points if key == point), None)

    def draw_overlay(self) -> None:
        """Hover ring, selection ring, and the keyboard focus frame: the only items that change on mouse moves."""
        canvas, ctx = self.canvas, self.ctx
        try:
            canvas.delete("overlay")
        except Exception:
            return
        if not self._points:
            return
        p = ctx.theme().p
        ring = ctx.px(LAYOUT["chartHoverRing"]) / 2.0 + ctx.px(SPACE["xxs"])
        for point, color in ((self.hover, p.textSecondary), (self.selected, p.accent)):
            if point is None or (point == self.selected and color == p.textSecondary):
                continue
            xy = self.point_xy(point)
            if xy is None:
                continue
            x, y = xy
            canvas.create_oval(x - ring, y - ring, x + ring, y + ring, outline=color, tags=("overlay",),
                               width=_width(ctx, WIN_LAYOUT["strokeStrong"] if point == self.selected else WIN_LAYOUT["strokeRegular"]))
        if self.focused and self._panel_boxes:
            box = self._panel_boxes[self.cursor_panel]
            pad = ctx.px(SPACE["xxs"])
            canvas.create_rectangle(box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad, outline=p.accent,
                                    width=_width(ctx, WIN_LAYOUT["strokeHairline"]), tags=("overlay",))


__all__ = ["ARMS", "DelayChart"]
