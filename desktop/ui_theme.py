"""Horizon theme for the Windows Tk surface: palette, fonts, ttk styles, and images.

Everything visual comes from the generated ``design_tokens`` module. The ttk
``clam`` theme is the base because it honors color options on every platform;
each element it draws is restyled here for both the dark and light palettes.
"""
from __future__ import annotations

import sys
from typing import Any, Callable

from .design_tokens import FONT_FALLBACKS, LAYOUT, RADIUS, SPACE, TYPE, UI_TEXT, WIN_LAYOUT, palette, scene3d
from .ui_icons import icon_png, photo_data, png_bytes, render_png, status_png


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
        """A palette token, plus a few derived colors for controls drawn on the accent."""
        p = self.p
        derived = {
            "onAccent": p.onAlert,  # the palette's text-on-strong-color value: background in dark, white in light
            "accentHover": mix(p.accent, p.text, 0.88),
            "accentPressed": mix(p.accent, p.surface, 0.75),
            "badgeFill": mix(p.lineStrong, p.surface, 0.22),
        }
        return derived[token] if token in derived else str(getattr(p, token))

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


# Glyphs the shared icon set does not have, drawn with the same rasterizer (unit-square coordinates).
EXTRA_GLYPHS = {
    "search": (("ring", 0.42, 0.42, 0.25, 0.1), ("line", 0.6, 0.6, 0.84, 0.84, 0.12)),
    "clear": (("line", 0.3, 0.3, 0.7, 0.7, 0.11), ("line", 0.7, 0.3, 0.3, 0.7, 0.11)),
}


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

    def glyph(self, name: str, token: str = "textSecondary", size: float = 14) -> Any:
        """One of ``EXTRA_GLYPHS`` in a theme color."""
        pixels = self.px(size)
        return self._get(("glyph", name, token, pixels), lambda theme: render_png(EXTRA_GLYPHS[name], theme.color(token), pixels))

    def apply(self, theme: Theme) -> None:
        self._theme = theme
        for key, image in self._images.items():
            image.configure(data=photo_data(self._recipes[key](theme)))


# ---- rounded control borders --------------------------------------------------------------
# ``clam`` draws square, notched borders. These 9-slice images give every control the
# RADIUS["small"] corner from the tokens. They are rendered at the real pixel size for
# the current DPI scale (so 125% and 150% stay crisp) and recolored in place on a theme
# switch, because a ttk element can be created only once per interpreter.
_SUPER = 4


def _rounded_coverage(x: float, y: float, width: float, height: float, radius: float) -> bool:
    if x < 0 or y < 0 or x > width or y > height:
        return False
    cx = min(max(x, radius), width - radius)
    cy = min(max(y, radius), height - radius)
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius


# Width of the stretchable middle of each 9-slice image. Tk fills a control by tiling this
# middle piece, so a 1-pixel middle means thousands of tiny draws per redraw (very slow on
# macOS). A wide middle keeps a redraw to a handful of draws.
ROUNDED_CENTER = 48


def rounded_png(radius: int, border: int, fill: str, edge: str | None, center: int = ROUNDED_CENTER) -> bytes:
    """A square rounded rectangle: ``edge`` border ``border`` px wide around ``fill``, corners of ``radius``.

    The image is ``2 * (radius + border) + center`` pixels on a side. Antialiased
    with 4x4 supersampling. ``edge=None`` draws a borderless filled pill.
    """
    side = 2 * (radius + border) + center
    outer_r = float(radius + border)
    inner_r = max(0.0, float(radius))

    def channels(color: str) -> tuple[int, int, int]:
        clean = color.lstrip("#")
        return int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)

    fill_rgb = channels(fill)
    edge_rgb = channels(edge) if edge else fill_rgb
    corner = radius + border  # only the four corner squares need antialiasing
    total = _SUPER * _SUPER

    def corner_pixel(px: int, py: int) -> tuple[int, int, int, int]:
        outer = inner = 0
        for sy in range(_SUPER):
            y = py + (sy + 0.5) / _SUPER
            for sx in range(_SUPER):
                x = px + (sx + 0.5) / _SUPER
                if _rounded_coverage(x, y, side, side, outer_r):
                    outer += 1
                    if edge is None or _rounded_coverage(x - border, y - border, side - 2 * border, side - 2 * border, inner_r):
                        inner += 1
        if not outer:
            return (0, 0, 0, 0)
        weight = inner / outer
        color = tuple(int(round(f * weight + e * (1.0 - weight))) for f, e in zip(fill_rgb, edge_rgb))
        return (*color, int(round(255 * outer / total)))

    edge_pixel = (*edge_rgb, 255)
    fill_pixel = (*fill_rgb, 255)
    rows = []
    for py in range(side):
        row = []
        for px in range(side):
            in_corner = (px < corner or px >= side - corner) and (py < corner or py >= side - corner)
            if in_corner:
                row.append(corner_pixel(px, py))
            elif edge is not None and min(px, py, side - 1 - px, side - 1 - py) < border:
                row.append(edge_pixel)
            else:
                row.append(fill_pixel)
        rows.append(row)
    return png_bytes(side, side, rows)


class RoundedElements:
    """Image border elements for buttons, segments, the filter field, tiles, and the badge."""

    # element name -> ordered (state spec, (fill token, edge token or None)); the last entry is the default.
    SPECS: dict[str, tuple[tuple[str, tuple[str, str | None]], ...]] = {
        "Quiet": (("disabled", ("surface", "line")), ("pressed", ("selectionFill", "lineStrong")),
                  ("focus active", ("hoverFill", "accent")), ("focus", ("surface", "accent")),
                  ("active", ("hoverFill", "lineStrong")), ("", ("surface", "line"))),
        "Primary": (("pressed", ("accentPressed", None)), ("focus active", ("accentHover", "text")),
                    ("focus", ("accent", "text")), ("active", ("accentHover", None)), ("", ("accent", None))),
        "Segment": (("selected focus", ("selectionFill", "accent")), ("selected", ("selectionFill", "lineStrong")),
                    ("focus", ("surface", "accent")), ("active", ("hoverFill", "line")), ("", ("surface", "line"))),
        "Field": (("focus", ("background", "accent")), ("", ("background", "line"))),
        "Tile": (("", ("surface", "line")),),
        "Badge": (("", ("badgeFill", None)),),
    }

    def __init__(self, style: Any, scale: float) -> None:
        import tkinter as tk

        self.style = style
        self.scale = scale
        self.radius = max(2, int(round(RADIUS["small"] * scale)))
        self.border = max(1, int(round(WIN_LAYOUT["hairline"] * scale)))
        self.images: dict[tuple[str, int], Any] = {}
        master = style.master
        for name, specs in self.SPECS.items():
            for index, _spec in enumerate(specs):
                self.images[(name, index)] = tk.PhotoImage(master=master)

    def apply(self, theme: Theme) -> None:
        color = theme.color
        for name, specs in self.SPECS.items():
            for index, (_state, (fill, edge)) in enumerate(specs):
                png = rounded_png(self.radius, self.border, color(fill), color(edge) if edge else None)
                self.images[(name, index)].configure(data=photo_data(png))
        existing = set(self.style.element_names())
        slice_border = self.radius + self.border
        for name, specs in self.SPECS.items():
            element = f"Zephyr{name}.border"
            if element in existing:
                continue
            states = [(state, self.images[(name, index)]) for index, (state, _colors) in enumerate(specs[:-1])]
            default = self.images[(name, len(specs) - 1)]
            # padding=border keeps the content inset at one hairline, so control_padding stays exact.
            # width/height: the element's own size is just its two corners, never the (wide) image itself.
            self.style.element_create(element, "image", default, *states, border=slice_border, padding=self.border,
                                      width=2 * slice_border, height=2 * slice_border, sticky="nswe")

    @staticmethod
    def layout(name: str, child: str, inner: str) -> list[Any]:
        """A layout whose border is the rounded element ``name`` around a padding and ``inner`` element."""
        return [(f"Zephyr{name}.border", {"sticky": "nswe", "children": [
            (f"{child}.padding", {"sticky": "nswe", "children": [(inner, {"sticky": "nswe"})]}),
        ]})]


def rounded_elements(style: Any, scale: float) -> RoundedElements:
    """One set per Tk interpreter (elements and images belong to it), kept on the root window itself."""
    master = style.master
    bank = getattr(master, "_zephyr_rounded", None)
    if bank is None or bank.scale != scale:
        bank = RoundedElements(style, scale)
        master._zephyr_rounded = bank
    return bank


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


def _fill(total: int, content: int, border: int) -> tuple[int, int]:
    """Split the space left around ``content`` into (top, bottom) so the control is exactly ``total`` tall."""
    spare = max(0, total - content - 2 * border)
    return spare // 2, spare - spare // 2


def control_padding(fonts: Fonts, scale: float) -> dict[str, tuple[int, int, int, int]]:
    """ttk padding (left, top, right, bottom) that makes every control ``controlHeight`` tall.

    One control height for the whole app: text buttons, icon buttons,
    segmented choices, and the filter field all come out the same height,
    whatever the font metrics are on this machine.
    """
    def px(value: float) -> int:
        return max(1, int(round(value * scale)))

    border = px(WIN_LAYOUT["hairline"])
    height = px(WIN_LAYOUT["controlHeight"])
    text_top, text_bottom = _fill(height, int(fonts["body"].metrics("linespace")), border)
    icon_top, icon_bottom = _fill(height, px(WIN_LAYOUT["iconSize"]), border)
    side = icon_top  # square icon buttons
    return {
        "text": (px(SPACE["md"]), text_top, px(SPACE["md"]), text_bottom),
        "icon": (side, icon_top, icon_bottom, icon_bottom),
        "segment": (px(SPACE["sm"]), text_top, px(SPACE["sm"]), text_bottom),
        # The filter field keeps room for its search glyph (left) and clear button (right).
        "field": (px(SPACE["sm"]) + px(WIN_LAYOUT["iconSize"]) + px(SPACE["xs"]), text_top,
                  px(SPACE["sm"]) + px(WIN_LAYOUT["iconSize"]), text_bottom),
    }


def configure_styles(style: Any, theme: Theme, fonts: Fonts, scale: float) -> None:
    """Restyle every ttk element the app uses, for one palette."""
    p = theme.p

    def px(value: float) -> int:
        return max(1, int(round(value * scale)))

    xxs, xs = SPACE["xxs"], SPACE["xs"]
    hairline = px(WIN_LAYOUT["hairline"])
    pads = control_padding(fonts, scale)
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
            "Metric": (fonts["metric"], p.text), "MetricMissing": (fonts["metric"], p.textSecondary),
            "MetricSmall": (fonts["metricSmall"], p.text), "Mono": (fonts["mono"], p.text),
            "Glyph": (fonts["body"], p.text),
        }
        for role, (font, foreground) in roles.items():
            style.configure(f"{name}.{role}.TLabel", background=color, foreground=foreground, font=font)
    style.configure("Line.TFrame", background=p.line)
    rounded = rounded_elements(style, scale)
    rounded.apply(theme)
    style.layout("Tile.TFrame", [("ZephyrTile.border", {"sticky": "nswe"})])
    style.configure("Tile.TFrame", background=p.surface)
    style.configure("Tile.Holder.TFrame", background=p.surface)
    style.configure("Tile.Caption.TLabel", background=p.surface, foreground=p.textSecondary, font=fonts["caption"])
    style.configure("Tile.Value.TLabel", background=p.surface, foreground=p.text, font=fonts["metricSmall"])
    # Same font as a value, only the color changes, so a tile never changes height when data goes missing.
    style.configure("Tile.Missing.TLabel", background=p.surface, foreground=p.textSecondary, font=fonts["metricSmall"])
    style.configure("Tile.Body.TLabel", background=p.surface, foreground=p.text, font=fonts["body"])
    # The honesty badge is quiet text on a tinted pill: no border, so it never looks like a button.
    style.layout("Badge.TLabel", RoundedElements.layout("Badge", "Label", "Label.label"))
    style.configure("Badge.TLabel", background=p.surface, foreground=p.textSecondary, font=fonts["caption"],
                    padding=(px(SPACE["sm"]), px(xxs)))

    for name, element, padding in (("Tool.TButton", "Quiet", pads["text"]), ("Icon.TButton", "Quiet", pads["icon"]),
                                   ("Primary.Icon.TButton", "Primary", pads["icon"])):
        style.layout(name, RoundedElements.layout(element, "Button", "Button.label"))
        style.configure(name, background=p.surface, foreground=p.text, padding=padding, width=0, anchor="center",
                        font=fonts["body"], focuscolor=p.surface)
        style.map(name, foreground=[("disabled", p.textSecondary)])
    style.layout("Link.TButton", [("Button.border", {"sticky": "nswe", "border": "1", "children": [
        ("Button.padding", {"sticky": "nswe", "children": [("Button.label", {"sticky": "nswe"})]}),
    ]})])
    style.configure(
        "Link.TButton", background=p.surface, foreground=p.text, bordercolor=p.surface, lightcolor=p.surface,
        darkcolor=p.surface, padding=(px(xs), px(xxs)), relief="solid", borderwidth=hairline, width=0, anchor="w",
        font=fonts["bodyStrong"],
    )
    style.map(
        "Link.TButton",
        background=[("pressed", p.selectionFill), ("active", p.hoverFill)],
        lightcolor=[("pressed", p.selectionFill), ("active", p.hoverFill)],
        darkcolor=[("pressed", p.selectionFill), ("active", p.hoverFill)],
        bordercolor=[("focus", p.accent), ("active", p.hoverFill)],
        foreground=[("disabled", p.textSecondary)],
    )
    # A small borderless icon button that sits inside the filter field (clear the filter).
    style.layout("FieldIcon.TButton", [("Button.padding", {"sticky": "nswe", "children": [("Button.label", {"sticky": "nswe"})]})])
    style.configure("FieldIcon.TButton", background=p.background, padding=px(xxs), relief="flat", borderwidth=0)
    style.map("FieldIcon.TButton", background=[("pressed", p.selectionFill), ("active", p.hoverFill)])

    style.layout("Segment.Toolbutton", RoundedElements.layout("Segment", "Toolbutton", "Toolbutton.label"))
    style.configure(
        "Segment.Toolbutton", background=p.surface, foreground=p.textSecondary, padding=pads["segment"],
        anchor="center", font=fonts["body"],
    )
    style.map(
        "Segment.Toolbutton",
        background=[("selected", p.selectionFill), ("active", p.hoverFill)],
        foreground=[("selected", p.text), ("active", p.text)],
    )

    style.layout("Filter.TEntry", [("ZephyrField.border", {"sticky": "nswe", "children": [
        ("Entry.padding", {"sticky": "nswe", "children": [("Entry.textarea", {"sticky": "nswe"})]}),
    ]})])
    style.configure(
        "Filter.TEntry", fieldbackground=p.background, background=p.background, foreground=p.text,
        insertcolor=p.text, padding=pads["field"], selectbackground=p.selectionFill, selectforeground=p.text,
    )

    item_layout = [("Treeitem.padding", {"sticky": "nswe", "children": [
        ("Treeitem.indicator", {"side": "left", "sticky": ""}),
        ("Treeitem.image", {"side": "left", "sticky": ""}),
        ("Treeitem.text", {"side": "left", "sticky": ""}),
    ]})]
    for name in ("Zephyr.Treeview", "Events.Treeview"):
        style.layout(f"{name}.Item", item_layout)
        style.configure(f"{name}.Item", foreground=p.textSecondary, indicatorsize=px(SPACE["sm"]))
        style.configure(
            name, background=p.surface, fieldbackground=p.surface, foreground=p.text, bordercolor=p.surface,
            lightcolor=p.surface, darkcolor=p.surface, rowheight=px(LAYOUT["rowHeight"]), borderwidth=hairline,
            relief="solid", font=fonts["body"], indent=px(SPACE["lg"]),
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
        lightcolor=p.lineStrong, darkcolor=p.lineStrong, arrowsize=px(SPACE["sm"]), gripcount=0, relief="flat",
    )
    style.map(
        "Thin.Vertical.TScrollbar",
        background=[("active", p.textSecondary)], lightcolor=[("active", p.textSecondary)], darkcolor=[("active", p.textSecondary)],
    )


__all__ = [
    "EXTRA_GLYPHS", "Fonts", "ImageBank", "LEVELS", "RoundedElements", "STATUS_GLYPHS", "Theme", "apply_title_bar",
    "configure_styles", "control_padding", "mix", "rounded_png",
    "status_shape", "ui_scale",
]
