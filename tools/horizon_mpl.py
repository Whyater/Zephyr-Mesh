"""Horizon palette and Matplotlib style for plots in this repository.

Plots in the repo should use this module instead of picking their own colors:

    from tools import horizon_mpl
    horizon_mpl.apply_style("light")
    colors = horizon_mpl.palette("light")

Every color is read from design/horizon_tokens.json at runtime, so the plots
follow the same palette as the apps. Red (`alert`) and amber (`caution`) are
status colors. Use them for alerts only, for example a failure line, never to
tell data series apart. The style adds no shadows and no gradients.

Text uses Matplotlib's bundled DejaVu Sans so the output does not depend on
fonts installed on a given machine.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl

TOKENS_PATH = Path(__file__).resolve().parents[1] / "design" / "horizon_tokens.json"
THEMES = ("light", "dark")
FONT_FAMILY = "DejaVu Sans"


@lru_cache(maxsize=1)
def _load_tokens() -> dict:
    with TOKENS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve(entry) -> str | None:
    """Return a hex string for a token entry, or None if it has no flat color."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        if "value" in entry:
            return entry["value"]
        if "hex" in entry:
            return entry["hex"]
    return None


def palette(theme: str = "light") -> dict[str, str]:
    """Token name -> hex string for a theme. Entries marked proposed are resolved to their value."""
    if theme not in THEMES:
        raise ValueError(f"theme must be one of {THEMES}, got {theme!r}")
    out: dict[str, str] = {}
    for name, entry in _load_tokens()["colors"][theme].items():
        value = _resolve(entry)
        if value is not None:
            out[name] = value
    return out


def _to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(max(0.0, min(1.0, c)) * 255):02X}" for c in rgb)


def mix(color_a: str, color_b: str, t: float) -> str:
    """Linear blend of two hex colors: t=0 gives color_a, t=1 gives color_b."""
    a, b = _to_rgb(color_a), _to_rgb(color_b)
    return _to_hex(tuple(x + (y - x) * t for x, y in zip(a, b)))  # type: ignore[arg-type]


def ramp(stops: list[str], n: int) -> list[str]:
    """n colors evenly spaced along a piecewise-linear path through the given hex stops."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if n == 1 or len(stops) == 1:
        return [stops[0]] * n
    out = []
    for i in range(n):
        pos = i / (n - 1) * (len(stops) - 1)
        lo = min(int(pos), len(stops) - 2)
        out.append(mix(stops[lo], stops[lo + 1], pos - lo))
    return out


def sequential_ramp(token_names: list[str], n: int, theme: str = "light") -> list[str]:
    """Like ramp(), with the stops given as token names from the theme palette."""
    pal = palette(theme)
    return ramp([pal[name] for name in token_names], n)


def with_alpha(hex_color: str, alpha: float, over: str) -> str:
    """Flat color equal to hex_color at the given alpha drawn over another hex color."""
    return mix(over, hex_color, alpha)


def apply_style(theme: str = "light") -> dict[str, str]:
    """Set Matplotlib rcParams for the Horizon style and return the palette used."""
    pal = palette(theme)
    mpl.rcParams.update(
        {
            "figure.facecolor": pal["background"],
            "figure.edgecolor": pal["background"],
            "savefig.facecolor": pal["background"],
            "savefig.edgecolor": pal["background"],
            "axes.facecolor": pal["surface"],
            "axes.edgecolor": pal["lineStrong"],
            "axes.labelcolor": pal["textSecondary"],
            "axes.titlecolor": pal["text"],
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.titlesize": 10.5,
            "axes.titleweight": "regular",
            "axes.titlepad": 8,
            "axes.labelsize": 9.5,
            "grid.color": pal["line"],
            "grid.linewidth": 0.6,
            "xtick.color": pal["textSecondary"],
            "ytick.color": pal["textSecondary"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "text.color": pal["text"],
            "font.family": FONT_FAMILY,
            "font.size": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": pal["text"],
            "lines.solid_capstyle": "butt",
            "patch.linewidth": 0.0,
        }
    )
    return pal
