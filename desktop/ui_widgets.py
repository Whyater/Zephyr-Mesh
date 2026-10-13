"""Small Tk widgets shared by the Windows replay window and its viewers.

Each widget takes its colors from a ``Theme`` through ``apply_theme`` so a
light or dark switch never rebuilds the window.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from .design_tokens import SPACE, WIN_LAYOUT
from .ui_format import NOT_RECORDED


Row = tuple  # (label, value or None, kind) where kind is "body" or "mono"


class Disclosure:
    """A real collapsible section: a chevron button that shows or hides ``body``."""

    def __init__(self, parent: Any, title: str, images: Any, *, bg: str = "Surface", expanded: bool = False,
                 on_toggle: Callable[[bool], None] | None = None, pad: int = 0) -> None:
        from tkinter import ttk

        self.frame = ttk.Frame(parent, style=f"{bg}.TFrame")
        self._closed = images.icon("chevron_right", "textSecondary", WIN_LAYOUT["chevronSize"])
        self._open = images.icon("chevron_down", "textSecondary", WIN_LAYOUT["chevronSize"])
        self.button = ttk.Button(self.frame, text=f" {title}", style="Link.TButton", compound="left",
                                 image=self._closed, command=self.toggle, takefocus=True)
        self.button.pack(fill="x")
        self.body = ttk.Frame(self.frame, style=f"{bg}.TFrame")
        self._pad = pad
        self._on_toggle = on_toggle
        self.expanded = False
        if expanded:
            self.toggle()

    def toggle(self) -> None:
        self.expanded = not self.expanded
        if self.expanded:
            self.body.pack(fill="x", pady=(self._pad, 0))
            self.button.configure(image=self._open)
        else:
            self.body.pack_forget()
            self.button.configure(image=self._closed)
        if self._on_toggle is not None:
            self._on_toggle(self.expanded)


class KeyValueGrid:
    """Label/value rows on a grid: labels in textSecondary, values in text. Reuses widgets."""

    def __init__(self, parent: Any, *, bg: str = "Surface", label_width: int = 0, wrap: int = 0, scale: float = 1.0,
                 stacked: bool = False, value_width: int = 0, label_wrap: int = 0) -> None:
        from tkinter import ttk

        self.ttk = ttk
        self.bg = bg
        self.frame = ttk.Frame(parent, style=f"{bg}.TFrame")
        self.frame.columnconfigure(1, weight=1)
        if label_width:
            self.frame.columnconfigure(0, minsize=label_width)
            self.frame.columnconfigure(1, weight=1)
        self.wrap = wrap
        self.scale = scale
        self.stacked = stacked
        self.value_width = value_width
        self.label_wrap = label_wrap
        self._rows: list[tuple[Any, Any]] = []
        self._shown: list[tuple[str, str, str]] = []

    def set_wrap(self, wrap: int) -> None:
        """Re-wrap every value for a new width (viewers call this when their window is resized)."""
        wrap = max(1, int(wrap))
        if wrap == self.wrap:
            return
        self.wrap = wrap
        for _label, value in self._rows:
            value.configure(wraplength=wrap)

    def set_rows(self, rows: Sequence[Row]) -> None:
        ttk = self.ttk
        gap = int(round(SPACE["xs"] * self.scale))
        while len(self._rows) < len(rows):
            index = len(self._rows)
            label = ttk.Label(self.frame, style=f"{self.bg}.Secondary.TLabel", anchor="nw", justify="left",
                              wraplength=self.label_wrap or 0)
            value = ttk.Label(self.frame, style=f"{self.bg}.Body.TLabel", anchor="nw", justify="left", width=self.value_width or None)
            if self.stacked:
                label.configure(style=f"{self.bg}.Caption.TLabel")
                label.grid(row=2 * index, column=0, columnspan=2, sticky="nw")
                value.grid(row=2 * index + 1, column=0, columnspan=2, sticky="nw", pady=(0, int(round(SPACE["sm"] * self.scale))))
            else:
                label.grid(row=index, column=0, sticky="nw", padx=(0, int(round(SPACE["md"] * self.scale))), pady=(0, gap))
                value.grid(row=index, column=1, sticky="nw", pady=(0, gap))
            self._rows.append((label, value))
        while len(self._rows) > len(rows):
            label, value = self._rows.pop()
            label.destroy()
            value.destroy()
        rendered = []
        for row in rows:
            kind = row[2] if len(row) > 2 else "body"
            missing = row[1] is None
            style = "Secondary" if missing else ("Mono" if kind == "mono" else "Body")
            rendered.append((str(row[0]), NOT_RECORDED if missing else str(row[1]), f"{self.bg}.{style}.TLabel"))
        for index, ((label_widget, value_widget), (label, value, style)) in enumerate(zip(self._rows, rendered)):
            if index < len(self._shown) and self._shown[index] == (label, value, style):
                continue
            label_widget.configure(text=label)
            value_widget.configure(text=value, style=style, wraplength=self.wrap or 0)
        self._shown = rendered


class MetricTile:
    """Bordered tile: a caption label and one number in metricSmall."""

    def __init__(self, parent: Any, caption: str, scale: float = 1.0) -> None:
        from tkinter import ttk

        pad = int(round(SPACE["sm"] * scale))
        self.frame = ttk.Frame(parent, style="Tile.TFrame", padding=(pad, pad, pad, pad))
        self.caption = ttk.Label(self.frame, text=caption, style="Tile.Caption.TLabel")
        self.caption.pack(anchor="w")
        from tkinter import font as tkfont

        # A fixed-height holder: switching between a number and "Not recorded" never moves the rows below.
        holder = ttk.Frame(self.frame, style="Tile.Holder.TFrame",
                           height=tkfont.nametofont("ZephyrFontMetricSmall").metrics("linespace"))
        holder.pack(fill="x", pady=(int(round(SPACE["xxs"] * scale)), 0))
        holder.pack_propagate(False)
        self.value = ttk.Label(holder, style="Tile.Value.TLabel", anchor="w")
        self.value.pack(side="left", fill="both", expand=True)

    def set(self, text: str | None) -> None:
        shown = (NOT_RECORDED, "Tile.Missing.TLabel") if text is None else (text, "Tile.Value.TLabel")
        if shown != getattr(self, "_shown", None):
            self.value.configure(text=shown[0], style=shown[1])
            self._shown = shown


class ScrollFrame:
    """Vertical scrolling container; the scrollbar appears only when content overflows."""

    def __init__(self, parent: Any, *, bg: str = "Surface", scale: float = 1.0) -> None:
        """The scrollbar's column is always reserved (an empty spacer stands in when it is hidden), so the
        content width, and with it the right inset, never changes when scrolling becomes possible."""
        import tkinter as tk
        from tkinter import ttk

        self.scale = scale
        self.outer = ttk.Frame(parent, style=f"{bg}.TFrame")
        self.canvas = tk.Canvas(self.outer, highlightthickness=0, borderwidth=0, takefocus=0)
        self.scrollbar = ttk.Scrollbar(self.outer, orient="vertical", style="Thin.Vertical.TScrollbar", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style=f"{bg}.TFrame")
        self._window = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self._on_scroll)
        self.reserve = max(1, self.scrollbar.winfo_reqwidth())
        self.spacer = ttk.Frame(self.outer, style=f"{bg}.TFrame", width=self.reserve)
        self.spacer.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner.bind("<Configure>", lambda _event: self._sync())
        self.canvas.bind("<Configure>", self._on_resize)
        self._bg = bg
        self._shown = False

    def _on_resize(self, event: Any) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)
        self._sync()

    def _sync(self) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all") or (0, 0, 0, 0))

    def _on_scroll(self, first: str, last: str) -> None:
        self.scrollbar.set(first, last)
        needed = float(first) > 0.0 or float(last) < 1.0
        if needed and not self._shown:
            self.spacer.pack_forget()
            self.scrollbar.pack(side="right", fill="y", before=self.canvas)
            self._shown = True
        elif not needed and self._shown:
            self.scrollbar.pack_forget()
            self.spacer.pack(side="right", fill="y", before=self.canvas)
            self._shown = False
            self.canvas.yview_moveto(0)

    def reveal(self, widget: Any) -> None:
        """Scroll so ``widget`` starts near the top of the visible area."""
        self.canvas.update_idletasks()
        total = max(self.inner.winfo_height(), 1)
        top = widget.winfo_rooty() - self.inner.winfo_rooty()
        if total > self.canvas.winfo_height():
            # Land the top edge in the blank gap above the widget, so no line of text is sliced in half.
            self.canvas.yview_moveto(max(0.0, round(top - SPACE["sm"] * self.scale) / total))

    def contains(self, widget: Any) -> bool:
        """True when ``widget`` sits inside this scroll area (used by the app's wheel router)."""
        path = str(widget) if widget is not None else ""
        return path == str(self.canvas) or path.startswith(str(self.inner))

    def scroll(self, delta: float) -> None:
        if not self._shown or not delta:
            return
        steps = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        self.canvas.yview_scroll(steps, "units")

    def apply_theme(self, theme: Any) -> None:
        color = theme.p.surface if self._bg == "Surface" else theme.p.background
        self.canvas.configure(background=color)


class TimelineSlider:
    """Seekable frame slider drawn on a Canvas, with neutral tick marks at event frames."""

    # Drawn geometry in device-independent pixels. The thumb (hover radius) fits inside the SPACE["sm"] end inset.
    TRACK = 3
    TICK = SPACE["xs"] + 1
    THUMB = SPACE["xs"] + SPACE["xxs"]
    THUMB_HOVER = THUMB + 1

    def __init__(self, parent: Any, on_seek: Callable[[int], None], scale: float = 1.0) -> None:
        import tkinter as tk

        self.scale = scale
        self.height = int(round(WIN_LAYOUT["sliderHeight"] * scale))
        # No box around the whole strip on focus: the thumb itself gets an accent ring instead.
        self.canvas = tk.Canvas(parent, height=self.height, highlightthickness=0, borderwidth=0, takefocus=1, cursor="hand2")
        self._focused = False
        self.on_seek = on_seek
        self.index = 0
        self.count = 1
        self.ticks: list[int] = []
        self._theme: Any = None
        self._hover = False
        self._items: dict[str, Any] = {}
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._press)
        self.canvas.bind("<Enter>", lambda _event: self._set_hover(True))
        self.canvas.bind("<Leave>", lambda _event: self._set_hover(False))
        self.canvas.bind("<FocusIn>", lambda _event: self._set_focus(True))
        self.canvas.bind("<FocusOut>", lambda _event: self._set_focus(False))
        self.canvas.bind("<Left>", lambda _event: self._nudge(-1))
        self.canvas.bind("<Right>", lambda _event: self._nudge(1))

    def _set_focus(self, value: bool) -> None:
        self._focused = value
        self.draw()

    def _nudge(self, delta: int) -> str:
        """Arrow keys move the focused slider one frame, like dragging the thumb."""
        index = max(0, min(self.count - 1, self.index + delta))
        if index != self.index:
            self.on_seek(index)
        return "break"

    def _set_hover(self, value: bool) -> None:
        self._hover = value
        self.draw()

    def _inset(self) -> float:
        """Room at each end for the thumb, so the visible track starts exactly ``SPACE["sm"]`` in."""
        return SPACE["sm"] * self.scale

    def _x_for(self, index: int, width: float) -> float:
        inset = self._inset()
        span = max(width - 2 * inset, 1.0)
        return inset + span * (index / max(self.count - 1, 1))

    def index_at(self, x: float) -> int:
        width = max(float(self.canvas.winfo_width()), 1.0)
        inset = self._inset()
        fraction = (float(x) - inset) / max(width - 2 * inset, 1.0)
        return max(0, min(self.count - 1, int(round(fraction * (self.count - 1)))))

    def _press(self, event: Any) -> None:
        self.canvas.focus_set()
        index = self.index_at(event.x)
        if index != self.index:
            self.on_seek(index)

    def set_state(self, index: int, count: int, ticks: Sequence[int]) -> None:
        changed = (index, count, list(ticks)) != (self.index, self.count, self.ticks)
        self.index, self.count, self.ticks = index, max(1, count), list(ticks)
        if changed:
            self.draw()

    def apply_theme(self, theme: Any) -> None:
        self._theme = theme
        p = theme.p
        self.canvas.configure(background=p.surface, highlightbackground=p.surface, highlightcolor=p.accent)
        self.draw()

    def draw(self) -> None:
        if self._theme is None:
            return
        p = self._theme.p
        canvas = self.canvas
        width = max(float(canvas.winfo_width()), 40.0)
        middle = self.height / 2.0
        track = max(2, int(round(self.TRACK * self.scale)))
        if not self._items:
            self._items["track"] = canvas.create_line(0, 0, 0, 0, capstyle="round")
            self._items["progress"] = canvas.create_line(0, 0, 0, 0, capstyle="round")
            self._items["thumb"] = canvas.create_oval(0, 0, 0, 0)
        for item in canvas.find_withtag("tick"):
            canvas.delete(item)
        left, right = self._x_for(0, width), self._x_for(self.count - 1, width)
        current = self._x_for(self.index, width)
        canvas.coords(self._items["track"], left, middle, right, middle)
        canvas.itemconfigure(self._items["track"], fill=p.line, width=track)
        canvas.coords(self._items["progress"], left, middle, current, middle)
        canvas.itemconfigure(self._items["progress"], fill=p.accent, width=track)
        tick_height = self.TICK * self.scale
        for frame in self.ticks:
            x = self._x_for(frame, width)
            gap = SPACE["xxs"] * self.scale
            canvas.create_line(x, middle + track + gap, x, middle + track + gap + tick_height,
                               fill=p.textSecondary, width=max(1, int(round(WIN_LAYOUT["strokeHairline"] * self.scale))),
                               tags=("tick",))
        radius = (self.THUMB_HOVER if self._hover else self.THUMB) * self.scale
        canvas.coords(self._items["thumb"], current - radius, middle - radius, current + radius, middle + radius)
        ring = p.text if self._focused else p.surface
        canvas.itemconfigure(self._items["thumb"], fill=p.accent, outline=ring,
                             width=max(1, int(round(WIN_LAYOUT["strokeStrong"] * self.scale))))
        canvas.tag_raise(self._items["thumb"])


class Placeholder:
    """Secondary hint text inside an empty ttk.Entry (ttk has no placeholder option)."""

    def __init__(self, entry: Any, variable: Any, text: str, style: str, inset: int = 0) -> None:
        from tkinter import ttk

        self.inset = inset
        self.entry = entry
        self.variable = variable
        self.label = ttk.Label(entry, text=text, style=style, cursor="xterm")
        self.label.bind("<Button-1>", lambda _event: entry.focus_set())
        variable.trace_add("write", lambda *_args: self.update())
        entry.bind("<FocusIn>", lambda _event: self.update(), add="+")
        entry.bind("<FocusOut>", lambda _event: self.update(), add="+")
        self.update()

    def update(self) -> None:
        if self.variable.get():
            self.label.place_forget()
        else:
            self.label.place(relx=0, rely=0.5, x=self.inset, anchor="w")


class TreeHover:
    """Hover feedback for Treeview rows using a tag; never touches the selection."""

    def __init__(self, tree: Any, tag: str = "hover") -> None:
        self.tree = tree
        self.tag = tag
        self.row = ""
        tree.bind("<Motion>", self._move, add="+")
        tree.bind("<Leave>", lambda _event: self._set(""), add="+")

    def _move(self, event: Any) -> None:
        self._set(self.tree.identify_row(event.y))

    def _set(self, row: str) -> None:
        if row == self.row:
            return
        if self.row and self.tree.exists(self.row):
            tags = tuple(tag for tag in self.tree.item(self.row, "tags") if tag != self.tag)
            self.tree.item(self.row, tags=tags)
        self.row = row
        if row and self.tree.exists(row):
            tags = tuple(self.tree.item(row, "tags")) + (self.tag,)
            self.tree.item(row, tags=tags)

    def forget(self, row: str) -> None:
        if row == self.row:
            self.row = ""


class Tooltip:
    """Delayed hover hint. ``text_for(x, y)`` can pick the text per position (for example one table cell)."""

    def __init__(self, widget: Any, text: str, theme_source: Callable[[], Any], font: Any, *, scale: float = 1.0,
                 text_for: Callable[[int, int], str | None] | None = None) -> None:
        self.widget = widget
        self.text = text
        self.theme_source = theme_source
        self.font = font
        self.scale = scale
        self.text_for = text_for
        self._job: Any = None
        self._window: Any = None
        self._shown_text: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        if text_for is not None:
            widget.bind("<Motion>", self._move, add="+")

    def px(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    def set_text(self, text: str) -> None:
        self.text = text

    def _current(self, event: Any = None) -> str | None:
        if self.text_for is None:
            return self.text
        if event is None:
            return None
        return self.text_for(event.x, event.y)

    def _move(self, event: Any) -> None:
        text = self._current(event)
        if text == self._shown_text and (self._window is not None or self._job is not None):
            return
        self._hide()
        if text:
            self._schedule(event)

    def _schedule(self, event: Any = None) -> None:
        self._cancel()
        text = self._current(event)
        if not text:
            return
        self._shown_text = text
        self._job = self.widget.after(WIN_LAYOUT["tooltipDelayMs"], self._show)

    def _cancel(self) -> None:
        if self._job is not None:
            try:
                self.widget.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    def _show(self) -> None:
        import tkinter as tk

        self._job = None
        if not self._shown_text:
            return
        window = None
        try:
            theme = self.theme_source()
            window = tk.Toplevel(self.widget)
            window.wm_overrideredirect(True)
            try:
                window.wm_attributes("-topmost", True)
            except tk.TclError:
                pass
            border = tk.Frame(window, background=theme.p.lineStrong, padx=self.px(WIN_LAYOUT["hairline"]),
                              pady=self.px(WIN_LAYOUT["hairline"]))
            border.pack()
            tk.Label(border, text=self._shown_text, background=theme.p.surface, foreground=theme.p.text, font=self.font,
                     padx=self.px(SPACE["sm"]), pady=self.px(SPACE["xs"])).pack()
            window.update_idletasks()  # lay out (and paint) before placing; some Tk builds show an empty box otherwise
            gap = self.px(SPACE["xs"])
            if self.text_for is not None:
                x, y = self.widget.winfo_pointerx(), self.widget.winfo_pointery() + self.px(SPACE["lg"])
            else:
                x = self.widget.winfo_rootx()
                top = self.widget.winfo_rooty()
                # Controls in the lower half (the timeline bar) get their hint above, so it never covers the Events list.
                below = top + self.widget.winfo_height() + gap
                above = top - window.winfo_reqheight() - gap
                y = above if top > self.widget.winfo_screenheight() / 2 and above > 0 else below
            window.wm_geometry(f"+{x}+{y}")
        except tk.TclError:
            if window is not None:  # created but not finished: never leave an empty window behind
                try:
                    window.destroy()
                except tk.TclError:
                    pass
            return
        self._window = window

    def _hide(self, _event: Any = None) -> None:
        self._cancel()
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None
        if _event is not None:
            self._shown_text = None


__all__ = ["Disclosure", "KeyValueGrid", "MetricTile", "Placeholder", "ScrollFrame", "TimelineSlider", "Tooltip", "TreeHover"]
