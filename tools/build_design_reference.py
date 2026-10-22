#!/usr/bin/env python3
"""Render tools/design/reference.html, a one-page visual reference of the Horizon tokens.

The page is built only from tools/design/horizon_tokens.json, tools/design/ui_text.json
and the checked-in icon PNG. It is static and self-contained: no scripts, no
external fonts, no network requests. The page itself is styled with the dark
Horizon palette.

Usage:
  python3 tools/build_design_reference.py          write tools/design/reference.html
  python3 tools/build_design_reference.py --check  fail if the file is stale
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import struct
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.check_contrast import contrast, unwrap  # noqa: E402

TOKENS_PATH = ROOT / "tools" / "design" / "horizon_tokens.json"
UI_PATH = ROOT / "tools" / "design" / "ui_text.json"
ICON_PATH = ROOT / "tools" / "design" / "icon" / "ZephyrMesh-256.png"
ICO_PATH = ROOT / "tools" / "design" / "icon" / "ZephyrMesh.ico"
OUTPUT_PATH = ROOT / "tools" / "design" / "reference.html"

# What each color is for. The token files hold values; the jobs live here and in
# tools/design/README.md, and a test checks that every palette token has one.
JOBS = {
    "background": "Window background and the 3D floor",
    "surface": "Panels: team list, inspector, status strip, timeline",
    "line": "Hairline dividers, tile borders, scroll track, the scene grid",
    "lineStrong": "Badge border, event ticks, hovered button border, heavier grid lines",
    "text": "Primary text and values",
    "textSecondary": "Labels, captions, \"Not recorded\", the drone glyph",
    "accent": "Selected drone, focus ring, scrubber fill and thumb",
    "target": "The target ring in the scene",
    "keepOut": "Keep-out volume in the scene",
    "ok": "Status OK only",
    "caution": "Status Late only (alert color)",
    "alert": "Status Lost or Inactive, and the load-failure symbol (alert color)",
    "info": "Info events",
    "onAlert": "Text placed on an alert or caution fill",
    "selectionFill": "Selected row, pressed button, selected segment",
    "hoverFill": "Row and button under the pointer",
    "bandFill": "Interval band behind the recovery line in the Results charts",
    "atlasStop1": "Tolerance atlas cell, 0 to 25 ms",
    "atlasStop2": "Tolerance atlas cell, 50 to 100 ms",
    "atlasStop3": "Tolerance atlas cell, 150 to 200 ms",
    "atlasStop4": "Tolerance atlas cell, more than 300 ms",
}
TEXT_ROLES = ("text", "textSecondary", "accent", "target", "keepOut", "ok", "caution", "alert", "info")
# Inline SVG for each status shape, drawn in a 16 by 16 box.
SHAPES = {
    "circle": '<circle cx="8" cy="8" r="5.5" fill="{c}"/>',
    "triangle": '<path d="M8 2.2 14 13.4H2Z" fill="{c}"/>',
    "square": '<rect x="2.8" y="2.8" width="10.4" height="10.4" fill="{c}"/>',
    "hollow circle": '<circle cx="8" cy="8" r="4.8" fill="none" stroke="{c}" stroke-width="2"/>',
    "dash": '<rect x="3" y="7" width="10" height="2" rx="1" fill="{c}"/>',
    "info": '<circle cx="8" cy="8" r="6.4" fill="none" stroke="{c}" stroke-width="1.6"/>'
            '<circle cx="8" cy="4.9" r="1.1" fill="{c}"/><rect x="7.1" y="6.9" width="1.8" height="5" rx=".9" fill="{c}"/>',
}


def ico_frames(data: bytes) -> dict[int, bytes]:
    """PNG frames from an .ico, keyed by pixel size (the small frames carry the small-size rule)."""
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    if (reserved, kind) != (0, 1) or len(data) < 6 + 16 * count:
        raise ValueError("not an .ico file")
    frames = {}
    for index in range(count):
        width, _, _, _, _, _, length, offset = struct.unpack("<BBBBHHII", data[6 + 16 * index:22 + 16 * index])
        frame = data[offset:offset + length]
        if len(frame) != length or frame[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"ico entry {index} is not a complete PNG frame")
        frames[width or 256] = frame
    for size in (16, 32):
        if size not in frames:
            raise ValueError(f"ico has no {size} px frame")
    return frames


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def hex_of(colors: dict[str, Any], name: str) -> str:
    item = colors[name]
    if isinstance(item, dict) and "hex" in item:
        return str(item["hex"])
    return str(unwrap(item))


def status_of(item: Any) -> str:
    return str(item.get("status", "")) if isinstance(item, dict) else ""


def ratio_cell(foreground: str, background: str, required: float) -> str:
    ratio = contrast(foreground, background)
    verdict = "pass" if ratio >= required else "below"
    return f'<span class="ratio {verdict}">{ratio:.2f}:1</span>'


def color_cards(theme: str, colors: dict[str, Any]) -> str:
    background = hex_of(colors, "background")
    surface = hex_of(colors, "surface")
    cards = []
    for name, item in colors.items():
        value = hex_of(colors, name)
        notes = []
        if name in TEXT_ROLES:
            notes.append(f"on background {ratio_cell(value, background, 4.5)}")
            notes.append(f"on surface {ratio_cell(value, surface, 4.5)}")
        elif name == "lineStrong":
            notes.append(f"on surface {ratio_cell(value, surface, 3.0)} (needs 3:1)")
        elif name == "onAlert":
            notes.append(f"on alert {ratio_cell(value, hex_of(colors, 'alert'), 4.5)}")
            notes.append(f"on caution {ratio_cell(value, hex_of(colors, 'caution'), 4.5)}")
        elif isinstance(item, dict) and "base" in item:
            notes.append(f"{esc(item['base'])} at {float(item['alpha']):g} over {esc(item['over'])}")
        tag = ' <span class="tag">proposed</span>' if status_of(item) == "proposed" else ""
        cards.append(
            '<div class="card">'
            f'<div class="chip" style="background:{esc(value)}"></div>'
            f'<div class="name">{esc(name)}{tag}</div>'
            f'<div class="hex">{esc(value)}</div>'
            f'<div class="job">{esc(JOBS.get(name, ""))}</div>'
            + "".join(f'<div class="note">{note}</div>' for note in notes)
            + "</div>"
        )
    return f'<div class="cards {esc(theme)}">' + "".join(cards) + "</div>"


def avoided_pairs(tokens: dict[str, Any]) -> str:
    rows = []
    for entry in tokens["contrast"]["needs-approval"]:
        colors = tokens["colors"][entry["theme"]]
        foreground = hex_of(colors, entry["foreground"])
        background = hex_of(colors, entry["background"])
        rows.append(
            f'<tr><td>{esc(entry["theme"])}</td><td>{esc(entry["foreground"])} on {esc(entry["background"])}</td>'
            f'<td><span class="pair" style="color:{esc(foreground)};background:{esc(background)}">Sample text</span></td>'
            f"<td>{ratio_cell(foreground, background, 4.5)}</td></tr>"
        )
    return "<table><tr><th>Theme</th><th>Pair</th><th>Looks like</th><th>Ratio (needs 4.5:1)</th></tr>" + "".join(rows) + "</table>"


def type_rows(tokens: dict[str, Any]) -> str:
    rows = []
    for name, spec in tokens["typeScale"].items():
        if name == "families":
            continue
        weight = 600 if spec.get("weight") == "semibold" else 400
        family = "ui-monospace, Menlo, Consolas, monospace" if name == "mono" else "inherit"
        numeric = "font-variant-numeric: tabular-nums;" if spec.get("tabularDigits") else ""
        sample = "a1b2c3d4" if name == "mono" else ("12.34 m" if name.startswith("metric") else "Tracking held")
        use = spec.get("use", "")
        rows.append(
            f'<tr><td class="mono">{esc(name)}</td><td>{spec["size"]} pt, {"semibold" if weight == 600 else "regular"}</td>'
            f'<td style="font-size:{spec["size"]}px;font-weight:{weight};font-family:{family};{numeric}">{esc(sample)}</td>'
            f"<td>{esc(use)}</td></tr>"
        )
    families = tokens["typeScale"]["families"]
    return (
        "<table><tr><th>Role</th><th>Size and weight</th><th>Sample</th><th>Use</th></tr>" + "".join(rows) + "</table>"
        f'<p class="small">macOS: {esc(families["macOS"])}. Windows: {esc(families["windows"])}.</p>'
    )


def spacing_rows(tokens: dict[str, Any]) -> str:
    rows = []
    for name, number in tokens["spacing"].items():
        rows.append(
            f'<tr><td class="mono">{esc(name)}</td><td>{number} pt</td>'
            f'<td><div class="bar" style="width:{number * 4}px"></div></td></tr>'
        )
    return "<table><tr><th>Token</th><th>Value</th><th>Bar (drawn 4x)</th></tr>" + "".join(rows) + "</table>"


def radii_block(tokens: dict[str, Any]) -> str:
    radii = tokens["radii"]
    boxes = []
    for name in ("small", "medium"):
        boxes.append(
            f'<div class="radius" style="border-radius:{radii[name] * 2}px"><b>{esc(name)}</b>'
            f'<span>{radii[name]} pt</span><span class="small">{esc(radii.get(name + "Use", ""))}</span></div>'
        )
    return '<div class="row">' + "".join(boxes) + '</div><p class="small">Drawn at 2x so the corners are visible.</p>'


def status_rows(tokens: dict[str, Any], ui: dict[str, Any]) -> str:
    rows = []
    for level, spec in ui["statusLevels"].items():
        cells = []
        for theme in ("dark", "light"):
            colors = tokens["colors"][theme]
            color = hex_of(colors, spec["color"])
            svg = SHAPES[spec["shape"]].format(c=esc(color))
            cells.append(
                f'<td class="{theme}-cell"><span class="status"><svg width="16" height="16" viewBox="0 0 16 16" '
                f'aria-hidden="true">{svg}</svg><b style="color:{esc(hex_of(colors, "text"))}">{esc(spec["word"])}</b>'
                "</span></td>"
            )
        alternate = f' or "{esc(spec["alternateWord"])}"' if "alternateWord" in spec else ""
        rows.append(
            f'<tr><td class="mono">{esc(level)}</td><td>"{esc(spec["word"])}"{alternate}</td><td>{esc(spec["shape"])}</td>'
            f'<td class="mono">{esc(spec["color"])}</td>' + "".join(cells) + "</tr>"
        )
    thresholds = ui["displayThresholds"]
    late = thresholds["targetAgeLateS"]["value"]
    lost = thresholds["targetAgeLostS"]["value"]
    return (
        "<table><tr><th>Level</th><th>Word</th><th>Shape</th><th>Color token</th><th>Dark</th><th>Light</th></tr>"
        + "".join(rows)
        + "</table>"
        f'<p class="small">Display thresholds: target age of {late} s or more reads Late; {lost} s or more reads Lost. '
        f"{esc(thresholds['targetAgeLateS']['note'][:1].upper() + thresholds['targetAgeLateS']['note'][1:])}.</p>"
    )


def key_value_table(mapping: dict[str, Any], unit: str = "") -> str:
    rows = "".join(f'<tr><td class="mono">{esc(key)}</td><td>{esc(value)}{unit}</td></tr>' for key, value in mapping.items())
    return f"<table>{rows}</table>"


def shortcut_rows(ui: dict[str, Any]) -> str:
    rows = []
    names = {"primary": "Command or Control", "shift": "Shift"}
    for spec in ui["shortcuts"].values():
        keys = [names.get(item, item) for item in spec.get("modifiers", [])] + [spec["key"].title()]
        rows.append(f"<tr><td>{esc(spec['label'])}</td><td>{esc(' + '.join(keys))}</td></tr>")
    return (
        "<table><tr><th>Action</th><th>Keys</th></tr>" + "".join(rows) + "</table>"
        f'<p class="small">{esc(ui["shortcutRule"])}</p>'
    )


def stylesheet(colors: dict[str, Any], light: dict[str, Any], tokens: dict[str, Any]) -> str:
    c = {name: hex_of(colors, name) for name in colors}
    l = {name: hex_of(light, name) for name in light}
    space = tokens["spacing"]
    radius = tokens["radii"]["small"]
    return f"""
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: {c['background']}; color: {c['text']};
  font: 13px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif; }}
main {{ max-width: 1120px; margin: 0 auto; padding: {space['xxl']}px {space['xl']}px; }}
header {{ display: flex; align-items: center; gap: {space['lg']}px; margin-bottom: {space['xl']}px; }}
h1 {{ font-size: 20px; font-weight: 600; margin: 0; }}
h2 {{ font-size: 15px; font-weight: 600; margin: {space['xxl']}px 0 {space['sm']}px; }}
h3 {{ font-size: 13px; font-weight: 600; margin: {space['lg']}px 0 {space['sm']}px; color: {c['textSecondary']}; }}
p {{ margin: {space['xs']}px 0 {space['sm']}px; max-width: 760px; }}
.small, .job, .note, .hex {{ font-size: 12px; color: {c['textSecondary']}; }}
.mono, .hex {{ font-family: ui-monospace, Menlo, Consolas, monospace; }}
section {{ background: {c['surface']}; border: 1px solid {c['line']}; border-radius: {radius}px;
  padding: {space['lg']}px; margin-top: {space['sm']}px; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ text-align: left; font-size: 12px; font-weight: 400; color: {c['textSecondary']};
  border-bottom: 1px solid {c['line']}; padding: {space['xs']}px {space['sm']}px; }}
td {{ border-bottom: 1px solid {c['line']}; padding: {space['sm']}px; vertical-align: middle; }}
tr:last-child td {{ border-bottom: 0; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: {space['sm']}px; }}
.cards.light {{ background: {l['background']}; padding: {space['sm']}px; border-radius: {radius}px; }}
.card {{ background: {c['background']}; border: 1px solid {c['line']}; border-radius: {radius}px; padding: {space['sm']}px; }}
.cards.light .card {{ background: {l['surface']}; border-color: {l['line']}; color: {l['text']}; }}
.cards.light .job, .cards.light .note, .cards.light .hex {{ color: {l['textSecondary']}; }}
.chip {{ height: 40px; border-radius: {radius}px; border: 1px solid {c['lineStrong']}; margin-bottom: {space['sm']}px; }}
.name {{ font-weight: 600; }}
.tag {{ font-size: 12px; font-weight: 400; border: 1px solid currentColor; border-radius: {radius}px; padding: 0 {space['xs']}px; }}
.ratio.pass {{ color: inherit; }}
.ratio.below {{ text-decoration: underline; }}
.pair {{ padding: {space['xxs']}px {space['sm']}px; border-radius: {radius}px; }}
.bar {{ height: 12px; background: {c['accent']}; border-radius: 2px; }}
.row {{ display: flex; gap: {space['lg']}px; flex-wrap: wrap; align-items: flex-end; }}
.radius {{ width: 200px; height: 96px; border: 2px solid {c['lineStrong']}; display: flex; flex-direction: column;
  justify-content: center; padding: {space['md']}px; gap: {space['xxs']}px; }}
.status {{ display: inline-flex; align-items: center; gap: {space['xs']}px; }}
.dark-cell {{ background: {c['surface']}; }}
.light-cell {{ background: {l['surface']}; }}
.icon {{ background-image: var(--icon); background-size: contain; background-repeat: no-repeat; flex: none; }}
.icon-light {{ background: {'#ECECEC'}; padding: {space['lg']}px; border-radius: {radius}px; }}
.icon-dark {{ background: {'#1E1E1E'}; padding: {space['lg']}px; border-radius: {radius}px; }}
footer {{ margin-top: {space['xxl']}px; color: {c['textSecondary']}; font-size: 12px; }}
"""


def render(tokens: dict[str, Any], ui: dict[str, Any], icon_png: bytes, ico: bytes) -> str:
    dark = tokens["colors"]["dark"]
    light = tokens["colors"]["light"]
    icon = data_url(icon_png)
    small = ico_frames(ico)
    icon_sizes = "".join(f'<div class="icon" style="width:{size}px;height:{size}px"></div>' for size in (128, 64))
    icon_sizes += "".join(
        f'<img src="{data_url(small[size])}" width="{size}" height="{size}" alt="{size} px frame">' for size in (32, 16))
    motion = tokens["motion"]
    sections = [
        ("Colors, dark (default)", "<p>Red (alert) and amber (caution) carry status only. They are never decoration "
         "or brand color. Ratios use the WCAG formula from tools/check_contrast.py; text needs 4.5:1.</p>"
         + color_cards("dark", dark)),
        ("Colors, light", "<p>Tokens tagged proposed are still waiting for approval in the token file.</p>"
         + color_cards("light", light)),
        ("Pairs that are avoided", "<p>These text and fill pairs fall below 4.5:1. The token file marks them "
         "needs-approval, and the apps keep status words and accent text off these fills; a selected row "
         "shows its word in the text color.</p>" + avoided_pairs(tokens)),
        ("Status levels", "<p>Every status is a shape, a color and a word together, so color is never the only "
         "signal.</p>" + status_rows(tokens, ui)),
        ("Type scale", type_rows(tokens)),
        ("Spacing", spacing_rows(tokens)),
        ("Radii", radii_block(tokens)),
        ("Motion", f"<p>Fast {motion['fastMs']} ms, standard {motion['standardMs']} ms, {esc(motion['curve'])}. "
         f"Only movement the user starts is animated. With Reduce Motion on, durations are "
         f"{motion['reducedMotionMs']} ms.</p>"),
        ("Layout sizes", key_value_table(tokens["layout"], " pt")),
        ("Keyboard shortcuts", shortcut_rows(ui)),
        ("App icon", '<p>A gust from the west wraps into an orbit of five drones linked to the point they '
         'track together. At 16 and 32 px the links and the second breeze are left out. The 128 and 64 px '
         'samples are scaled from ZephyrMesh-256.png; the 32 and 16 px samples are the real small frames '
         'from ZephyrMesh.ico.</p>'
         f'<div class="row"><div class="row icon-light">{icon_sizes}</div><div class="row icon-dark">{icon_sizes}</div></div>'),
    ]
    body = "".join(f"<h2>{esc(title)}</h2><section>{content}</section>" for title, content in sections)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Zephyr Mesh design reference</title>\n"
        f"<style>{stylesheet(dark, light, tokens)}</style>\n</head>\n"
        f'<body style="--icon:url({icon})">\n<main>\n'
        f'<header><div class="icon" style="width:64px;height:64px"></div><div><h1>Zephyr Mesh design reference</h1>'
        f'<p class="small">Horizon tokens for the macOS and Windows replay apps. Generated by '
        f"tools/build_design_reference.py from tools/design/horizon_tokens.json and tools/design/ui_text.json; "
        f"do not edit by hand. The written guide is tools/design/README.md.</p></div></header>\n"
        f"{body}\n"
        "<footer>Values on this page come from the token files. Contrast ratios are computed, not measured "
        "on a screen.</footer>\n</main>\n</body>\n</html>\n"
    )


def build(tokens_path: Path = TOKENS_PATH, ui_path: Path = UI_PATH, icon_path: Path = ICON_PATH,
          ico_path: Path = ICO_PATH) -> str:
    return render(load(tokens_path), load(ui_path), icon_path.read_bytes(), ico_path.read_bytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render tools/design/reference.html from the token files.")
    parser.add_argument("--check", action="store_true", help="fail if tools/design/reference.html is stale")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    expected = build()
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != expected:
            print(f"{args.output} is stale; run python3 tools/build_design_reference.py", file=sys.stderr)
            return 1
        return 0
    args.output.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
