"""Tolerance atlas matrix for the Results window, drawn on one Tk canvas.

Rows are the stratum's target paths, columns its trackers, and a last
column names the tracker that takes the most delay. Each cell shows the most
delay that tracker tolerated, tinted with one of four accent ramp stops (or
the plain surface plus a small alert square when it fails even at 0 ms).
Cells shrink toward a measured minimum on a narrow window; past that the
matrix scrolls sideways instead of clipping text.
"""
from __future__ import annotations

import math
from typing import Any, Callable

from .design_tokens import LAYOUT, RADIUS, SPACE, WIN_LAYOUT
from . import results_data as rd
from .ui_widgets import Tooltip


Selection = tuple[str, str]  # (variant, arm)


def rounded_points(x0: float, y0: float, x1: float, y1: float, radius: float) -> list[float]:
    """Polygon points that Tk's ``smooth`` option turns into a rectangle with rounded corners."""
    r = max(0.0, min(radius, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    return [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r,
            x0, y0 + r, x0, y0]


class AtlasGrid:
    """The matrix. ``on_select`` gets (variant, arm) or ``None``."""

    def __init__(self, ctx: Any, parent: Any, data: rd.ResultsData, on_select: Callable[[Selection | None], None]) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.ctx = ctx
        self.data = data
        self.on_select = on_select
        self.stratum = data.stratum_order[0]
        self.loss = rd.level_loss_key("independent", 0.0)
        self.selected: Selection | None = None
        self.cursor: tuple[int, int] = (0, 0)
        self.hover: tuple[int, int] | None = None
        self.focused = False
        self.draw_count = 0
        self._cells: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        self._total_width = 0.0
        self._drawn_key: tuple[Any, ...] | None = None
        self.frame = ttk.Frame(parent, style="Surface.TFrame")
        self.canvas = tk.Canvas(self.frame, highlightthickness=0, borderwidth=0, takefocus=1)
        self.canvas.pack(fill="x")
        self.scrollbar = ttk.Scrollbar(self.frame, orient="horizontal", style="Results.Horizontal.TScrollbar",
                                       command=self.canvas.xview)
        self._scroll_shown = False
        self.canvas.configure(xscrollcommand=self._on_xscroll)
        canvas = self.canvas
        canvas.bind("<Configure>", lambda _event: self.redraw())
        canvas.bind("<Motion>", self._on_motion)
        canvas.bind("<Leave>", lambda _event: self._set_hover(None))
        canvas.bind("<Button-1>", self._on_click)
        canvas.bind("<FocusIn>", lambda _event: self._set_focus(True))
        canvas.bind("<FocusOut>", lambda _event: self._set_focus(False))
        for key in ("Left", "Right", "Up", "Down", "Home", "End"):
            canvas.bind(f"<{key}>", lambda _event, key=key: self.key(key))
        canvas.bind("<Escape>", lambda _event: self.key("Escape"))
        canvas.bind("<Shift-MouseWheel>", lambda event: self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units"))
        self.tooltip = Tooltip(canvas, "", ctx.theme, ctx.fonts["caption"], scale=ctx.scale, text_for=self._tip)
        # A hint still waiting for its delay when the window closes must not fire into a destroyed widget.
        canvas.bind("<Destroy>", lambda _event: self.tooltip._hide(), add="+")
        self._measure()

    # ---- state ---------------------------------------------------------------------------
    @property
    def variants(self) -> list[str]:
        return list(self.data.strata[self.stratum]["variants"])

    @property
    def arms(self) -> list[str]:
        return list(self.data.strata[self.stratum]["arms"])

    def entry_at(self, row: int, column: int) -> Any:
        return self.data.entry(self.stratum, self.variants[row], self.loss, self.arms[column])

    def set_view(self, stratum: str, loss: str) -> None:
        """Change conditions or loss; the selection stays on the same path and tracker when they still exist."""
        if (stratum, loss) == (self.stratum, self.loss):
            return
        changed_stratum = stratum != self.stratum
        self.stratum, self.loss = stratum, loss
        self.hover = None
        if changed_stratum:
            self._measure()
            self.canvas.xview_moveto(0)
        if self.selected is not None:
            variant, arm = self.selected
            if variant in self.variants and arm in self.arms:
                self.cursor = (self.variants.index(variant), self.arms.index(arm))
                self.on_select(self.selected)
            else:
                self.selected = None
                self.cursor = (0, 0)
                self.on_select(None)
        else:
            self.cursor = (min(self.cursor[0], len(self.variants) - 1), min(self.cursor[1], len(self.arms) - 1))
        self.redraw()

    def select(self, selection: Selection | None, *, notify: bool = True) -> None:
        if selection is not None:
            variant, arm = selection
            self.cursor = (self.variants.index(variant), self.arms.index(arm))
        if selection == self.selected:
            return
        self.selected = selection
        self.draw_overlay()
        self.reveal(*self.cursor)
        if notify:
            self.on_select(selection)

    def key(self, key: str) -> str | None:
        if key == "Escape":
            if self.selected is None:
                return None
            self.select(None)
            return "break"
        row, column = self.cursor
        if self.selected is not None:
            row, column = rd.move_cell(row, column, key, len(self.variants), len(self.arms))
        self.select((self.variants[row], self.arms[column]))
        return "break"

    # ---- events --------------------------------------------------------------------------
    def cell_at(self, x: float, y: float) -> tuple[int, int] | None:
        x = self.canvas.canvasx(x)
        for key, box in self._cells.items():
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                return key
        return None

    def _tip(self, x: int, y: int) -> str | None:
        """Hover hint for a cell whose text is shortened: the tracker name and its full delay text."""
        cell = self.cell_at(x, y)
        if cell is None:
            return None
        entry = self.entry_at(*cell)
        return f"{self.data.arm_label(self.arms[cell[1]])}: {rd.tolerated_text(entry) or rd.TEXT['notComputed']}"

    def _on_motion(self, event: Any) -> None:
        self._set_hover(self.cell_at(event.x, event.y))

    def _set_hover(self, cell: tuple[int, int] | None) -> None:
        if cell != self.hover:
            self.hover = cell
            self.canvas.configure(cursor="hand2" if cell is not None else "")
            self.draw_overlay()

    def _on_click(self, event: Any) -> None:
        self.canvas.focus_set()
        cell = self.cell_at(event.x, event.y)
        if cell is not None:
            self.select((self.variants[cell[0]], self.arms[cell[1]]))

    def _set_focus(self, value: bool) -> None:
        self.focused = value
        self.draw_overlay()

    def _on_xscroll(self, first: str, last: str) -> None:
        self.scrollbar.set(first, last)
        needed = float(first) > 0.001 or float(last) < 0.999
        if needed != self._scroll_shown:
            self._scroll_shown = needed
            if needed:
                self.scrollbar.pack(fill="x", pady=(self.ctx.px(SPACE["xs"]), 0))
            else:
                self.scrollbar.pack_forget()

    def reveal(self, row: int, column: int) -> None:
        """Scroll sideways so a selected cell is fully visible."""
        box = self._cells.get((row, column))
        width = float(self.canvas.winfo_width())
        total = self._total_width
        if box is None or total <= width or width <= 1:
            return
        left = self.canvas.canvasx(0)
        if box[0] < left:
            self.canvas.xview_moveto(max(0.0, (box[0] - self.ctx.px(SPACE["xs"])) / total))
        elif box[2] > left + width:
            self.canvas.xview_moveto(min(1.0, (box[2] + self.ctx.px(SPACE["xs"]) - width) / total))

    def apply_theme(self, theme: Any) -> None:
        from tkinter import ttk

        p = theme.p
        style = ttk.Style(self.canvas)
        style.layout("Results.Horizontal.TScrollbar", [("Horizontal.Scrollbar.trough", {"sticky": "we", "children": [
            ("Horizontal.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})])
        style.configure("Results.Horizontal.TScrollbar", troughcolor=p.surface, background=p.lineStrong, bordercolor=p.surface,
                        lightcolor=p.lineStrong, darkcolor=p.lineStrong, gripcount=0, gripsize=0, relief="flat",
                        arrowsize=self.ctx.px(SPACE["sm"]))
        style.map("Results.Horizontal.TScrollbar", background=[("active", p.textSecondary)])
        self._drawn_key = None
        self.redraw()

    # ---- geometry ------------------------------------------------------------------------
    def _measure(self) -> None:
        """Column widths from the real text of this stratum, so nothing is cut off and nothing jumps when the loss changes."""
        ctx, fonts, data = self.ctx, self.ctx.fonts, self.data
        body, caption = fonts["body"], fonts["caption"]
        pad = ctx.px(SPACE["sm"])
        cell_pad = ctx.px(SPACE["xs"])
        stratum = data.strata[self.stratum]
        widest_title = max(body.measure(data.variant_title(variant)) for variant in stratum["variants"])
        self.header_w = min(ctx.px(WIN_LAYOUT["atlasRowHeaderWidth"]), widest_title + ctx.px(SPACE["md"]))
        square = ctx.px(WIN_LAYOUT["statusMarkerSize"])
        fails = square + ctx.px(SPACE["xs"]) + body.measure(rd.ATLAS_TEXT["failsAtZero"])
        self.min_cell = max(body.measure(rd.ATLAS_TEXT["censored"]), fails, caption.measure(rd.TEXT["notComputed"])) + 2 * cell_pad
        texts = [rd.ATLAS_TEXT["mostDelayColumn"]]
        for variant in stratum["variants"]:
            for loss in stratum["loss_conditions"]:
                texts.extend(text for text in rd.most_delay_text(data, data.cell(self.stratum, variant, loss)) if text)
        self.most_w = max(caption.measure(text) for text in texts) + 2 * pad
        self.gap = ctx.px(SPACE["xs"])
        self.cell_h = ctx.px(WIN_LAYOUT["atlasCellHeight"])
        self.head_h = 2 * int(caption.metrics("linespace")) + ctx.px(SPACE["sm"])
        rows = len(stratum["variants"])
        self.canvas.configure(height=self.head_h + rows * (self.cell_h + self.gap) + 2 * self.gap)

    def cell_width(self, available: float) -> float:
        columns = len(self.arms)
        # Total width is header + gap + columns * (cell + gap) + most + gap, so a cell gets what is left.
        room = math.floor((available - self.header_w - self.most_w - (columns + 2) * self.gap) / max(columns, 1))
        return max(float(self.min_cell), min(float(self.ctx.px(WIN_LAYOUT["atlasCellWidth"])), room))

    # ---- drawing -------------------------------------------------------------------------
    def redraw(self) -> None:
        width = float(self.canvas.winfo_width())
        if width <= 1:
            return
        theme = self.ctx.theme()
        key = (width, theme.name, self.stratum, self.loss)
        if key != self._drawn_key:
            self._drawn_key = key
            self._draw(width, theme)
        self.draw_overlay()

    def _draw(self, width: float, theme: Any) -> None:
        ctx, canvas, p, data = self.ctx, self.canvas, theme.p, self.data
        fonts = ctx.fonts
        self.draw_count += 1
        canvas.configure(background=p.surface)
        canvas.delete("all")
        stops = {0: p.surface, 1: p.atlasStop1, 2: p.atlasStop2, 3: p.atlasStop3, 4: p.atlasStop4}
        cell_w = self.cell_width(width)
        gap, cell_h = self.gap, self.cell_h
        radius = ctx.px(RADIUS["small"])
        hairline = max(1, ctx.px(WIN_LAYOUT["strokeHairline"]))
        winner_w = max(1, ctx.px(LAYOUT["atlasWinnerBorder"]))
        square = ctx.px(WIN_LAYOUT["statusMarkerSize"])
        line = int(fonts["caption"].metrics("linespace"))
        self._cells = {}
        columns_x = [self.header_w + gap + index * (cell_w + gap) for index in range(len(self.arms))]
        most_x = (columns_x[-1] + cell_w + gap) if columns_x else self.header_w + gap
        self._total_width = most_x + self.most_w + gap
        for index, arm in enumerate(self.arms):
            canvas.create_text(columns_x[index] + cell_w / 2.0, self.head_h - ctx.px(SPACE["xs"]), text=data.arm_label(arm),
                               anchor="s", justify="center", width=cell_w, fill=p.textSecondary, font=fonts["caption"])
        canvas.create_text(most_x + ctx.px(SPACE["sm"]), self.head_h - ctx.px(SPACE["xs"]), text=rd.ATLAS_TEXT["mostDelayColumn"],
                           anchor="sw", fill=p.textSecondary, font=fonts["caption"])
        for row, variant in enumerate(self.variants):
            top = self.head_h + gap + row * (cell_h + gap)
            middle = top + cell_h / 2.0
            canvas.create_text(0, middle, text=data.variant_title(variant), anchor="w", fill=p.text, font=fonts["body"],
                               width=self.header_w - ctx.px(SPACE["xs"]))
            cell = data.cell(self.stratum, variant, self.loss)
            winners = set(cell["arms"]) if cell and cell["kind"] == "winner" else set()
            for column, arm in enumerate(self.arms):
                x0 = columns_x[column]
                box = (x0, top, x0 + cell_w, top + cell_h)
                self._cells[(row, column)] = box
                entry = data.entry(self.stratum, variant, self.loss, arm)
                canvas.create_polygon(*rounded_points(*box, radius), smooth=True, fill=stops[rd.entry_stop(entry)],
                                      outline=p.line, width=hairline, tags=("cell",))
                if arm in winners:
                    inset = hairline + winner_w / 2.0
                    canvas.create_polygon(*rounded_points(box[0] + inset, box[1] + inset, box[2] - inset, box[3] - inset,
                                                          max(1, radius - hairline)),
                                          smooth=True, fill="", outline=p.accent, width=winner_w)
                text = rd.entry_cell_text(entry)
                cx = (box[0] + box[2]) / 2.0
                if entry is None:
                    canvas.create_text(cx, middle, text=text, fill=p.textSecondary, font=fonts["caption"])
                elif entry["kind"] == "fails_at_zero":
                    total = square + ctx.px(SPACE["xs"]) + fonts["body"].measure(text)
                    left = cx - total / 2.0
                    canvas.create_rectangle(left, middle - square / 2.0, left + square, middle + square / 2.0, fill=p.alert, outline="")
                    canvas.create_text(left + square + ctx.px(SPACE["xs"]), middle, text=text, anchor="w", fill=p.text,
                                       font=fonts["body"])
                else:
                    canvas.create_text(cx, middle, text=text, fill=p.text, font=fonts["body"])
            label, delay = rd.most_delay_text(data, cell)
            x = most_x + ctx.px(SPACE["sm"])
            if delay:
                canvas.create_text(x, middle - line / 2.0, text=label, anchor="w", fill=p.text, font=fonts["caption"])
                canvas.create_text(x, middle + line / 2.0, text=delay, anchor="w", fill=p.textSecondary, font=fonts["caption"])
            else:
                canvas.create_text(x, middle, text=label, anchor="w", fill=p.text if cell else p.textSecondary, font=fonts["caption"])
        height = int(canvas.cget("height"))
        canvas.configure(scrollregion=(0, 0, self._total_width, height))

    def draw_overlay(self) -> None:
        """Hover outline, the selection ring (2 px outside the cell), and the keyboard focus ring."""
        canvas, ctx = self.canvas, self.ctx
        try:
            canvas.delete("overlay")
        except Exception:
            return
        if not self._cells:
            return
        p = ctx.theme().p
        radius = ctx.px(RADIUS["small"])
        strong = max(1, ctx.px(WIN_LAYOUT["strokeStrong"]))
        hairline = max(1, ctx.px(WIN_LAYOUT["strokeHairline"]))
        selected = None
        if self.selected is not None and self.selected[0] in self.variants and self.selected[1] in self.arms:
            selected = (self.variants.index(self.selected[0]), self.arms.index(self.selected[1]))
        if self.hover is not None and self.hover != selected and self.hover in self._cells:
            canvas.create_polygon(*rounded_points(*self._cells[self.hover], radius), smooth=True, fill="", outline=p.lineStrong,
                                  width=hairline, tags=("overlay",))
        if selected is not None:
            x0, y0, x1, y1 = self._cells[selected]
            out = strong / 2.0 + hairline
            canvas.create_polygon(*rounded_points(x0 - out, y0 - out, x1 + out, y1 + out, radius + out), smooth=True, fill="",
                                  outline=p.accent, width=strong, tags=("overlay",))
        elif self.focused and self.cursor in self._cells:
            x0, y0, x1, y1 = self._cells[self.cursor]
            out = hairline * 2
            canvas.create_polygon(*rounded_points(x0 - out, y0 - out, x1 + out, y1 + out, radius + out), smooth=True, fill="",
                                  outline=p.accent, width=hairline, tags=("overlay",))


__all__ = ["AtlasGrid", "rounded_points"]
