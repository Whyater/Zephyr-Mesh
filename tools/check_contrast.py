#!/usr/bin/env python3
"""Check WCAG contrast and derived fills in the Horizon token source."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEXT_COLORS = ("text", "textSecondary", "accent", "target", "keepOut", "ok", "caution", "alert", "info")
FILL_NAMES = ("selectionFill", "hoverFill")


def unwrap(item: Any) -> Any:
    """Read a token value whether it is scalar or annotated."""
    if isinstance(item, dict) and "value" in item:
        return item["value"]
    return item


def rgb(hex_value: str) -> tuple[float, float, float]:
    """Convert a six digit hex color to normalized RGB."""
    clean = hex_value.lstrip("#")
    return tuple(int(clean[index:index + 2], 16) / 255 for index in (0, 2, 4))


def linear(channel: float) -> float:
    """Convert an sRGB channel to linear light."""
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def luminance(hex_value: str) -> float:
    """Calculate relative luminance."""
    red, green, blue = (linear(channel) for channel in rgb(hex_value))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(foreground: str, background: str) -> float:
    """Calculate the WCAG contrast ratio."""
    first = luminance(foreground)
    second = luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def blend(foreground: str, background: str, alpha: float) -> str:
    """Blend foreground over background using token alpha."""
    foreground_rgb = rgb(foreground)
    background_rgb = rgb(background)
    channels = []
    for first, second in zip(foreground_rgb, background_rgb):
        channel = first * alpha + second * (1 - alpha)
        channels.append(int(channel * 255 + 0.5))
    return "#" + "".join(f"{channel:02X}" for channel in channels)


def pair_name(theme: str, foreground: str, background: str) -> tuple[str, str, str]:
    """Build the stable key used by needs-approval entries."""
    return theme, foreground, background


def approval_pairs(document: dict[str, Any]) -> set[tuple[str, str, str]]:
    """Read explicitly marked contrast pairs from JSON."""
    entries = document.get("contrast", {}).get("needs-approval", [])
    return {
        pair_name(str(entry["theme"]), str(entry["foreground"]), str(entry["background"]))
        for entry in entries
    }


def check_derived_fills(colors: dict[str, Any], theme: str) -> list[str]:
    """Return errors for stale stored fill hex values."""
    errors = []
    resolved = {key: str(unwrap(item)) for key, item in colors.items()}
    for fill_name in FILL_NAMES:
        fill = colors[fill_name]
        expected = blend(resolved[fill["base"]], resolved[fill["over"]], float(fill["alpha"]))
        stored = str(fill.get("hex", "")).upper()
        if stored != expected.upper():
            errors.append(f"{theme} {fill_name} stores {stored}, expected {expected}")
    return errors


def print_pair(
    theme: str,
    foreground: str,
    background: str,
    ratio: float,
    required: float,
    marked: bool,
) -> bool:
    """Print one result and return whether it is an unapproved failure."""
    passed = ratio >= required
    if passed and marked:
        result = "STALE-MARK"
    elif passed:
        result = "PASS"
    elif marked:
        result = "NEEDS-APPROVAL"
    else:
        result = "FAIL"
    print(f"{theme:5} {foreground:13} {background:13} {ratio:8.2f} {required:9.1f} {result}")
    return not passed and not marked


def main() -> int:
    """Run all palette contrast and derived-fill checks."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "design" / "horizon_tokens.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    marked = approval_pairs(document)
    failures = []
    fill_errors = []
    print("theme pair ratio requirement result")
    print("----- ---- ----- --------- ------")
    for theme, colors in document["colors"].items():
        resolved = {key: str(unwrap(item)) for key, item in colors.items()}
        fill_errors.extend(check_derived_fills(colors, theme))
        base_pairs = [
            ("text", "background"),
            ("textSecondary", "background"),
            ("accent", "background"),
            ("target", "background"),
            ("keepOut", "background"),
            ("ok", "background"),
            ("caution", "background"),
            ("alert", "background"),
            ("info", "background"),
            ("text", "surface"),
            ("textSecondary", "surface"),
            ("accent", "surface"),
            ("target", "surface"),
            ("keepOut", "surface"),
            ("ok", "surface"),
            ("caution", "surface"),
            ("alert", "surface"),
            ("info", "surface"),
            ("lineStrong", "surface"),
            ("onAlert", "alert"),
            ("onAlert", "caution"),
        ]
        for foreground, background in base_pairs:
            required = 3.0 if foreground == "lineStrong" else 4.5
            ratio = contrast(resolved[foreground], resolved[background])
            marked_pair = pair_name(theme, foreground, background) in marked
            if print_pair(theme, foreground, background, ratio, required, marked_pair):
                failures.append((theme, foreground, background, ratio, required))
            elif marked_pair and ratio >= required:
                failures.append((theme, foreground, background, ratio, required))
                print(
                    f"{theme} {foreground} on {background}: stale needs-approval mark",
                    file=sys.stderr,
                )
        for fill_name in FILL_NAMES:
            fill = colors[fill_name]
            fill_hex = str(fill["hex"])
            for foreground in TEXT_COLORS:
                required = 4.5
                ratio = contrast(resolved[foreground], fill_hex)
                marked_pair = pair_name(theme, foreground, fill_name) in marked
                if print_pair(theme, foreground, fill_name, ratio, required, marked_pair):
                    failures.append((theme, foreground, fill_name, ratio, required))
                elif marked_pair and ratio >= required:
                    failures.append((theme, foreground, fill_name, ratio, required))
                    print(
                        f"{theme} {foreground} on {fill_name}: stale needs-approval mark",
                        file=sys.stderr,
                    )
    if fill_errors:
        print("\nDerived fill errors:", file=sys.stderr)
        for error in fill_errors:
            print(error, file=sys.stderr)
    if failures:
        print("\nContrast failures require approval before changing source values:", file=sys.stderr)
        for theme, foreground, background, ratio, required in failures:
            print(f"{theme} {foreground} on {background}: {ratio:.2f}:1, needs {required:.1f}:1", file=sys.stderr)
    return 1 if failures or fill_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
