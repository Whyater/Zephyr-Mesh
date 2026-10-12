"""Small Tk widgets shared by the Windows replay window and its viewers.

Each widget takes its colors from a ``Theme`` through ``apply_theme`` so a
light or dark switch never rebuilds the window.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from .design_tokens import SPACE
from .ui_format import NOT_RECORDED


Row = tuple  # (label, value or None, kind) where kind is "body" or "mono"


class Disclosure:
    """A real collapsible section: a chevron button that shows or hides ``body``."""

    def __init__(self, parent: Any, title: str, images: Any, *, bg: str = "Surface", expanded: bool = False,
                 on_toggle: Callable[[bool], None] | None = None, pad: int = 0) -> None:
        from tkinter import ttk

        self.frame = ttk.Frame(parent, style=f"{bg}.TFrame")
        self._closed = images.icon("chevron_right", "textSecondary", 12)
        self._open = images.icon("chevron_down", "textSecondary", 12)
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
                value.grid(row=2 * index + 1, column=0, columnspan=2, sticky="nw", pady=(0, gap + gap))
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
        self.frame = ttk.Frame(parent, style="Tile.TFrame", padding=(pad + 2, pad, pad, pad))
        self.caption = ttk.Label(self.frame, text=caption, style="Tile.Caption.TLabel")
        self.caption.pack(anchor="w")
        from tkinter import font as tkfont

        holder = ttk.Frame(self.frame, style="Tile.Holder.TFrame",
                           height=tkfont.nametofont("ZephyrFontMetricSmall").metrics("linespace") + int(round(2 * scale)))
        holder.pack(fill="x", pady=(int(round(2 * scale)), 0))
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

    def __init__(self, parent: Any, *, bg: str = "Surface") -> None:
        import tkinter as tk
        from tkinter import ttk

        self.outer = ttk.Frame(parent, style=f"{bg}.TFrame")
        self.canvas = tk.Canvas(self.outer, highlightthickness=0, borderwidth=0, takefocus=0)
        self.scrollbar = ttk.Scrollbar(self.outer, orient="vertical", style="Thin.Vertical.TScrollbar", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style=f"{bg}.TFrame")
        self._window = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self._on_scroll)
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
            self.scrollbar.pack(side="right", fill="y", before=self.canvas)
            self._shown = True
        elif not needed and self._shown:
            self.scrollbar.pack_forget()
            self._shown = False
            self.canvas.yview_moveto(0)

    def reveal(self, widget: Any) -> None:
        """Scroll so ``widget`` starts near the top of the visible area."""
        self.canvas.update_idletasks()
        total = max(self.inner.winfo_height(), 1)
        top = widget.winfo_rooty() - self.inner.winfo_rooty()
        if total > self.canvas.winfo_height():
            self.canvas.yview_moveto(max(0.0, (top - 8) / total))

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

    def __init__(self, parent: Any, on_seek: Callable[[int], None], scale: float = 1.0) -> None:
        import tkinter as tk

        self.scale = scale
        self.height = int(round(28 * scale))
        self.canvas = tk.Canvas(parent, height=self.height, highlightthickness=1, borderwidth=0, takefocus=1, cursor="hand2")
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

    def _set_hover(self, value: bool) -> None:
        self._hover = value
        self.draw()

    def _inset(self) -> float:
        return 10 * self.scale

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
        track = max(2, int(round(3 * self.scale)))
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
        tick_height = 5 * self.scale
        for frame in self.ticks:
            x = self._x_for(frame, width)
            canvas.create_line(x, middle + track + 2 * self.scale, x, middle + track + 2 * self.scale + tick_height,
                               fill=p.textSecondary, width=max(1, int(round(self.scale))), tags=("tick",))
        radius = (7 if self._hover else 6) * self.scale
        canvas.coords(self._items["thumb"], current - radius, middle - radius, current + radius, middle + radius)
        canvas.itemconfigure(self._items["thumb"], fill=p.accent, outline=p.surface, width=max(1, int(round(2 * self.scale))))
        canvas.tag_raise(self._items["thumb"])


class Placeholder:
    """Secondary hint text inside an empty ttk.Entry (ttk has no placeholder option)."""

    def __init__(self, entry: Any, variable: Any, text: str, style: str) -> None:
        from tkinter import ttk

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
            self.label.place(relx=0, rely=0.5, x=10, anchor="w")


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
    """Delayed hover hint for icon-only buttons."""

    def __init__(self, widget: Any, text: str, theme_source: Callable[[], Any], font: Any) -> None:
        self.widget = widget
        self.text = text
        self.theme_source = theme_source
        self.font = font
        self._job: Any = None
        self._window: Any = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: Any = None) -> None:
        self._cancel()
        self._job = self.widget.after(550, self._show)

    def _cancel(self) -> None:
        if self._job is not None:
            self.widget.after_cancel(self._job)
            self._job = None

    def _show(self) -> None:
        import tkinter as tk

        self._job = None
        theme = self.theme_source()
        window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        try:
            window.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        border = tk.Frame(window, background=theme.p.lineStrong, padx=1, pady=1)
        border.pack()
        tk.Label(border, text=self.text, background=theme.p.surface, foreground=theme.p.text, font=self.font,
                 padx=8, pady=3).pack()
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        window.wm_geometry(f"+{x}+{y}")
        self._window = window

    def _hide(self, _event: Any = None) -> None:
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None


__all__ = ["Disclosure", "KeyValueGrid", "MetricTile", "Placeholder", "ScrollFrame", "TimelineSlider", "Tooltip", "TreeHover"]
