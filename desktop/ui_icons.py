"""Antialiased status glyphs and transport icons as PNG bytes, standard library only.

Tk 8.6 and 9.0 both decode PNG photo images with an alpha channel, so these
small shapes stay crisp on any row or button background without Pillow.
Shapes are described in unit coordinates (0 to 1) and rasterized with 4 x 4
supersampling. Colors are passed in from the generated tokens.
"""
from __future__ import annotations

import base64
import math
import struct
import zlib
from typing import Iterable, Sequence

Shape = tuple
_SAMPLES = 4

# Status shape names come from UI_TEXT["statusShapes"]; this maps each to geometry.
STATUS_SHAPES: dict[str, tuple[Shape, ...]] = {
    "circle": (("circle", 0.5, 0.5, 0.36),),
    "triangle": (("poly", ((0.5, 0.12), (0.9, 0.84), (0.1, 0.84))),),
    "square": (("poly", ((0.17, 0.17), (0.83, 0.17), (0.83, 0.83), (0.17, 0.83))),),
    "hollow circle": (("ring", 0.5, 0.5, 0.36, 0.13),),
    "dash": (("line", 0.2, 0.5, 0.8, 0.5, 0.14),),
    "info": (
        ("ring", 0.5, 0.5, 0.47, 0.11),
        ("circle", 0.5, 0.29, 0.075),
        ("line", 0.5, 0.45, 0.5, 0.73, 0.13),
    ),
}

ICONS: dict[str, tuple[Shape, ...]] = {
    "play": (("poly", ((0.3, 0.18), (0.82, 0.5), (0.3, 0.82))),),
    "pause": (
        ("poly", ((0.27, 0.2), (0.43, 0.2), (0.43, 0.8), (0.27, 0.8))),
        ("poly", ((0.57, 0.2), (0.73, 0.2), (0.73, 0.8), (0.57, 0.8))),
    ),
    "step_back": (
        ("poly", ((0.66, 0.2), (0.66, 0.8), (0.24, 0.5))),
        ("poly", ((0.7, 0.2), (0.8, 0.2), (0.8, 0.8), (0.7, 0.8))),
    ),
    "step_forward": (
        ("poly", ((0.34, 0.2), (0.76, 0.5), (0.34, 0.8))),
        ("poly", ((0.2, 0.2), (0.3, 0.2), (0.3, 0.8), (0.2, 0.8))),
    ),
    "restart": (
        ("poly", ((0.2, 0.2), (0.3, 0.2), (0.3, 0.8), (0.2, 0.8))),
        ("poly", ((0.8, 0.2), (0.8, 0.8), (0.34, 0.5))),
    ),
    "info": (
        ("ring", 0.5, 0.5, 0.42, 0.09),
        ("circle", 0.5, 0.31, 0.065),
        ("line", 0.5, 0.46, 0.5, 0.72, 0.11),
    ),
    "chevron_right": (("line", 0.4, 0.26, 0.64, 0.5, 0.12), ("line", 0.64, 0.5, 0.4, 0.74, 0.12)),
    "chevron_down": (("line", 0.26, 0.4, 0.5, 0.64, 0.12), ("line", 0.5, 0.64, 0.74, 0.4, 0.12)),
}


def _inside_poly(x: float, y: float, points: Sequence[tuple[float, float]]) -> bool:
    inside = False
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing:
                inside = not inside
    return inside


def _segment_distance(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    length = dx * dx + dy * dy
    t = 0.0 if length == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length))
    return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))


def _covers(shape: Shape, x: float, y: float) -> bool:
    kind = shape[0]
    if kind == "circle":
        _, cx, cy, radius = shape
        return (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius
    if kind == "ring":
        _, cx, cy, radius, thickness = shape
        distance = math.hypot(x - cx, y - cy)
        return radius - thickness <= distance <= radius
    if kind == "poly":
        return _inside_poly(x, y, shape[1])
    if kind == "line":
        _, x1, y1, x2, y2, thickness = shape
        return _segment_distance(x, y, x1, y1, x2, y2) <= thickness / 2.0
    raise ValueError(f"unknown shape kind {kind!r}")


def coverage(shapes: Iterable[Shape], size: int) -> list[list[float]]:
    """Return per-pixel coverage (0 to 1) for shapes drawn in a size x size box."""
    items = tuple(shapes)
    size = max(1, int(size))
    rows = []
    step = 1.0 / (_SAMPLES * size)
    for py in range(size):
        row = []
        for px in range(size):
            hits = 0
            for sy in range(_SAMPLES):
                y = (py * _SAMPLES + sy + 0.5) * step
                for sx in range(_SAMPLES):
                    x = (px * _SAMPLES + sx + 0.5) * step
                    if any(_covers(shape, x, y) for shape in items):
                        hits += 1
            row.append(hits / (_SAMPLES * _SAMPLES))
        rows.append(row)
    return rows


def _rgb(color: str) -> tuple[int, int, int]:
    clean = color.lstrip("#")
    if len(clean) != 6:
        raise ValueError("colors must be six-digit hex strings")
    return int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)


def png_bytes(width: int, height: int, pixels: Sequence[Sequence[tuple[int, int, int, int]]]) -> bytes:
    """Encode RGBA rows as a minimal PNG."""
    raw = b"".join(b"\x00" + b"".join(struct.pack("BBBB", *pixel) for pixel in row) for row in pixels)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def render_png(shapes: Iterable[Shape], color: str, size: int, box: int | None = None) -> bytes:
    """Rasterize shapes in one color; ``box`` pads the glyph into a wider canvas."""
    red, green, blue = _rgb(color)
    glyph = coverage(shapes, size)
    width = max(int(box or size), size)
    left = (width - size) // 2
    rows = []
    for row in glyph:
        pixels = [(red, green, blue, 0)] * left
        pixels += [(red, green, blue, int(round(alpha * 255))) for alpha in row]
        pixels += [(red, green, blue, 0)] * (width - left - size)
        rows.append(pixels)
    return png_bytes(width, size, rows)


def status_png(shape_name: str, color: str, size: int, box: int | None = None) -> bytes:
    """PNG for a status shape named in ``UI_TEXT["statusShapes"]``."""
    return render_png(STATUS_SHAPES[shape_name], color, size, box)


def icon_png(name: str, color: str, size: int) -> bytes:
    """PNG for one transport or disclosure icon."""
    return render_png(ICONS[name], color, size)


def photo_data(png: bytes) -> str:
    """Base64 text accepted by ``tkinter.PhotoImage(data=...)``."""
    return base64.b64encode(png).decode("ascii")


__all__ = ["ICONS", "STATUS_SHAPES", "coverage", "icon_png", "photo_data", "png_bytes", "render_png", "status_png"]
