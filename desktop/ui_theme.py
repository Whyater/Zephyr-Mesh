"""Horizon theme for the Windows Tk surface: palette, fonts, ttk styles, and images.

Everything visual comes from the generated ``design_tokens`` module. The ttk
``clam`` theme is the base because it honors color options on every platform;
each element it draws is restyled here for both the dark and light palettes.
"""
from __future__ import annotations

import sys
from typing import Any, Callable

from .design_tokens import FONT_FALLBACKS, LAYOUT, SPACE, TYPE, UI_TEXT, palette, scene3d
from .ui_icons import icon_png, photo_data, status_png


LEVELS = UI_TEXT["statusLevels"]
STATUS_GLYPHS = UI_TEXT["statusShapes"]["windows"]


def mix(foreground: str, background: str, alpha: float) -> str:
    """Blend two token colors; Tk has no alpha, so translucency is precomputed."""
    def channels(color: str) -> tuple[int, int, int]:
        clean = color.lstrip("#")
        return int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)

    weight = min(1.0, max(0.0, float(alpha)))
    top, bottom = channels(foreground), channels(background)
    blended = (int(a * weight + b * (1.0 - weight) + 0.5) for a, b in zip(top, bottom))
    return "#" + "".join(f"{value:02X}" for value in blended)


def ui_scale(root: Any) -> float:
    """Pixel multiplier for token sizes: Windows DPI / 96, else 1.0 (macOS already uses points)."""
    if not sys.platform.startswith("win"):
        return 1.0
    try:
        dpi = float(root.winfo_fpixels("1i"))
    except Exception:
        return 1.0
    return max(1.0, round(dpi / 96.0 * 4.0) / 4.0)


class Theme:
    """One resolved palette plus helpers for status colors."""

    def __init__(self, name: str) -> None:
        self.name = "light" if name == "light" else "dark"
        self.p = palette(self.name)
        self.scene = scene3d(self.name)

    @property
    def dark(self) -> bool:
        return self.name == "dark"

    def color(self, token: str) -> str:
        return str(getattr(self.p, token))

    def status_color(self, level: str) -> str:
        spec = LEVELS.get(level, LEVELS["unavailable"])
        return self.color(str(spec["color"]))

    def event_color(self, level: str) -> str:
        return self.status_color(level)


def status_shape(level: str) -> str:
    """Shape name for a level from UI_TEXT; an unknown level falls back to the unavailable shape."""
    spec = LEVELS.get(level, LEVELS["unavailable"])
    return str(spec["shape"])


class Fonts:
    """Named Tk fonts for every type role, with Windows family fallbacks."""

    def __init__(self, root: Any, scale: float) -> None:
        from tkinter import font as tkfont

        available = set(tkfont.families(root))
        default = tkfont.nametofont("TkDefaultFont", root=root).actual("family")
        fixed = tkfont.nametofont("TkFixedFont", root=root).actual("family")
        self.family = next((name for name in FONT_FALLBACKS["ui"] if name in available), default)
        semibold = f"{self.family} Semibold"
        self.strong_family = semibold if semibold in available else None
        self.mono_family = next((name for name in FONT_FALLBACKS["mono"] if name in available), fixed)
        self.scale = scale
        self._fonts: dict[str, Any] = {}
        for role, spec in TYPE.items():
            pixels = int(round(abs(spec[1]) * scale))
            bold = len(spec) > 2 and spec[2] == "bold"
            if role == "mono":
                family, weight = self.mono_family, "normal"
            elif bold and self.strong_family:
                family, weight = self.strong_family, "normal"
            else:
                family, weight = self.family, "bold" if bold else "normal"
            name = f"ZephyrFont{role[:1].upper()}{role[1:]}"
            try:
                font = tkfont.Font(root=root, name=name, family=family, size=-pixels, weight=weight)
            except Exception:
                font = tkfont.nametofont(name, root=root)
                font.configure(family=family, size=-pixels, weight=weight)
            self._fonts[role] = font
        body = TYPE["body"]
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkTooltipFont"):
            try:
                tkfont.nametofont(name, root=root).configure(family=self.family, size=-int(round(abs(body[1]) * scale)))
            except Exception:
                pass

    def __getitem__(self, role: str) -> Any:
        return self._fonts[role]


class ImageBank:
    """Themed PhotoImages. Recoloring reloads data in place, so widgets update themselves."""

    def __init__(self, root: Any, scale: float) -> None:
        self.root = root
        self.scale = scale
        self._images: dict[tuple[Any, ...], Any] = {}
        self._recipes: dict[tuple[Any, ...], Callable[[Theme], bytes]] = {}
        self._theme: Theme | None = None

    def px(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    def _get(self, key: tuple[Any, ...], recipe: Callable[[Theme], bytes]) -> Any:
        import tkinter as tk

        if key not in self._images:
            self._recipes[key] = recipe
            data = photo_data(recipe(self._theme)) if self._theme else None
            self._images[key] = tk.PhotoImage(master=self.root, data=data) if data else tk.PhotoImage(master=self.root)
        return self._images[key]

    def status(self, level: str, size: float = 10, box: float | None = None) -> Any:
        """Status glyph for a presentation level (or an event kind such as ``info``)."""
        pixels, boxed = self.px(size), self.px(box) if box else None
        shape = status_shape(level)
        return self._get(("status", level, pixels, boxed), lambda theme: status_png(shape, theme.event_color(level), pixels, boxed))

    def icon(self, name: str, token: str = "text", size: float = 14) -> Any:
        pixels = self.px(size)
        return self._get(("icon", name, token, pixels), lambda theme: icon_png(name, theme.color(token), pixels))

    def apply(self, theme: Theme) -> None:
        self._theme = theme
        for key, image in self._images.items():
            image.configure(data=photo_data(self._recipes[key](theme)))


def apply_title_bar(window: Any, dark: bool) -> None:
    """Ask Windows 10/11 for a dark or light title bar; silently skipped elsewhere."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1 if dark else 0)
        for attribute in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
    except Exception:
        pass


def configure_styles(style: Any, theme: Theme, fonts: Fonts, scale: float) -> None:
    """Restyle every ttk element the app uses, for one palette."""
    p = theme.p

    def px(value: float) -> int:
        return max(1, int(round(value * scale)))

    sm, xs, md = SPACE["sm"], SPACE["xs"], SPACE["md"]
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(
        ".",
        background=p.background, foreground=p.text, bordercolor=p.line, darkcolor=p.background,
        lightcolor=p.background, troughcolor=p.surface, focuscolor=p.accent, selectbackground=p.selectionFill,
        selectforeground=p.text, insertcolor=p.text, fieldbackground=p.surface, font=fonts["body"],
        relief="flat", borderwidth=0,
    )
    style.map(".", foreground=[("disabled", p.textSecondary)])

    for name, color in (("Base", p.background), ("Surface", p.surface)):
        style.configure(f"{name}.TFrame", background=color)
        roles = {
            "Title": (fonts["title"], p.text), "Body": (fonts["body"], p.text),
            "Strong": (fonts["bodyStrong"], p.text), "Secondary": (fonts["body"], p.textSecondary),
            "Caption": (fonts["caption"], p.textSecondary), "Headline": (fonts["headline"], p.text),
            "Metric": (fonts["metric"], p.text), "MetricMissing": (fonts["metricSmall"], p.textSecondary),
            "MetricSmall": (fonts["metricSmall"], p.text), "Mono": (fonts["mono"], p.text),
            "Glyph": (fonts["body"], p.text),
        }
        for role, (font, foreground) in roles.items():
            style.configure(f"{name}.{role}.TLabel", background=color, foreground=foreground, font=font)
    style.configure("Line.TFrame", background=p.line)
    style.configure("Tile.TFrame", background=p.surface, bordercolor=p.line, lightcolor=p.surface, darkcolor=p.surface,
            relief="solid", borderwidth=1)
    style.configure("Tile.Holder.TFrame", background=p.surface)
    style.configure("Tile.Caption.TLabel", background=p.surface, foreground=p.textSecondary, font=fonts["caption"])
    style.configure("Tile.Value.TLabel", background=p.surface, foreground=p.text, font=fonts["metricSmall"])
    style.configure("Tile.Missing.TLabel", background=p.surface, foreground=p.textSecondary, font=fonts["body"])
    style.configure(
        "Badge.TLabel", background=p.surface, foreground=p.textSecondary, font=fonts["caption"], relief="solid",
        borderwidth=1, bordercolor=p.lineStrong, lightcolor=p.surface, darkcolor=p.surface, padding=(px(sm), px(2)),
    )

    button_layout = [("Button.border", {"sticky": "nswe", "border": "1", "children": [
        ("Button.padding", {"sticky": "nswe", "children": [("Button.label", {"sticky": "nswe"})]}),
    ]})]
    for name, padding in (("Tool.TButton", (px(md), px(xs + 1))), ("Icon.TButton", (px(xs + 2), px(xs + 1))),
            ("Link.TButton", (px(xs), px(2)))):
        style.layout(name, button_layout)
        border = p.surface if name == "Link.TButton" else p.line
        style.configure(
            name, background=p.surface, foreground=p.text, bordercolor=border, lightcolor=p.surface,
            darkcolor=p.surface, padding=padding, relief="solid", borderwidth=1, width=0, anchor="center",
            font=fonts["body"],
        )
        style.map(
            name,
            background=[("pressed", p.selectionFill), ("active", p.hoverFill)],
            lightcolor=[("pressed", p.selectionFill), ("active", p.hoverFill)],
            darkcolor=[("pressed", p.selectionFill), ("active", p.hoverFill)],
            bordercolor=[("focus", p.accent), ("active", p.lineStrong if name != "Link.TButton" else p.hoverFill)],
            foreground=[("disabled", p.textSecondary)],
        )
    style.configure("Link.TButton", anchor="w", font=fonts["bodyStrong"])

    style.layout("Segment.Toolbutton", [("Toolbutton.border", {"sticky": "nswe", "border": "1", "children": [
        ("Toolbutton.padding", {"sticky": "nswe", "children": [("Toolbutton.label", {"sticky": "nswe"})]}),
    ]})])
    style.configure(
        "Segment.Toolbutton", background=p.surface, foreground=p.textSecondary, bordercolor=p.line,
        lightcolor=p.surface, darkcolor=p.surface, padding=(px(sm + 2), px(xs)), relief="solid", borderwidth=1,
        anchor="center", font=fonts["body"],
    )
    style.map(
        "Segment.Toolbutton",
        background=[("selected", p.selectionFill), ("active", p.hoverFill)],
        lightcolor=[("selected", p.selectionFill), ("active", p.hoverFill)],
        darkcolor=[("selected", p.selectionFill), ("active", p.hoverFill)],
        foreground=[("selected", p.text), ("active", p.text)],
        bordercolor=[("focus", p.accent), ("selected", p.lineStrong)],
    )

    style.configure(
        "Filter.TEntry", fieldbackground=p.background, foreground=p.text, bordercolor=p.line,
        lightcolor=p.background, darkcolor=p.background, insertcolor=p.text, padding=(px(sm), px(xs + 1)),
        selectbackground=p.selectionFill, selectforeground=p.text,
    )
    style.map("Filter.TEntry", bordercolor=[("focus", p.accent)], lightcolor=[("focus", p.background)])

    item_layout = [("Treeitem.padding", {"sticky": "nswe", "children": [
        ("Treeitem.indicator", {"side": "left", "sticky": ""}),
        ("Treeitem.image", {"side": "left", "sticky": ""}),
        ("Treeitem.text", {"side": "left", "sticky": ""}),
    ]})]
    for name in ("Zephyr.Treeview", "Events.Treeview"):
        style.layout(f"{name}.Item", item_layout)
        style.configure(f"{name}.Item", foreground=p.textSecondary, indicatorsize=px(9))
        style.configure(
            name, background=p.surface, fieldbackground=p.surface, foreground=p.text, bordercolor=p.surface,
            lightcolor=p.surface, darkcolor=p.surface, rowheight=px(LAYOUT["rowHeight"]), borderwidth=1,
            relief="solid", font=fonts["body"], indent=px(SPACE["lg"] + 2),
        )
        style.map(
            name,
            background=[("selected", p.selectionFill)], foreground=[("selected", p.text)],
            bordercolor=[("focus", p.accent)],
        )
        style.configure(
            f"{name}.Heading", background=p.surface, foreground=p.textSecondary, font=fonts["caption"], relief="flat",
            bordercolor=p.surface, lightcolor=p.surface, darkcolor=p.surface, padding=(px(xs), px(xs)),
        )
        style.map(f"{name}.Heading", background=[("active", p.surface)], foreground=[("active", p.textSecondary)])

    scrollbar_layout = [("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
        ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"}),
    ]})]
    style.layout("Thin.Vertical.TScrollbar", scrollbar_layout)
    style.configure(
        "Thin.Vertical.TScrollbar", troughcolor=mix(p.line, p.surface, 0.35), background=p.lineStrong, bordercolor=p.surface,
        lightcolor=p.lineStrong, darkcolor=p.lineStrong, arrowsize=px(8), gripcount=0, relief="flat",
    )
    style.map(
        "Thin.Vertical.TScrollbar",
        background=[("active", p.textSecondary)], lightcolor=[("active", p.textSecondary)], darkcolor=[("active", p.textSecondary)],
    )


__all__ = [
    "Fonts", "ImageBank", "LEVELS", "STATUS_GLYPHS", "Theme", "apply_title_bar", "configure_styles", "mix",
    "status_shape", "ui_scale",
]
