#!/usr/bin/env python3
"""Generate the shared design modules from the JSON source files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pprint
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOKENS_PATH = ROOT / "tools" / "design" / "horizon_tokens.json"
UI_PATH = ROOT / "tools" / "design" / "ui_text.json"
SWIFT_PATH = ROOT / "tools" / "macos" / "ZephyrMeshApp" / "Sources" / "ZephyrMeshApp" / "DesignSystem.swift"
PY_PATH = ROOT / "tools" / "desktop" / "design_tokens.py"

PALETTE_FIELDS = (
    "background", "surface", "line", "lineStrong", "text", "textSecondary",
    "accent", "target", "keepOut", "ok", "caution", "alert", "info", "onAlert",
    "selectionFill", "hoverFill", "bandFill", "atlasStop1", "atlasStop2", "atlasStop3", "atlasStop4",
)


def load(path: Path) -> dict[str, Any]:
    """Read one JSON source file."""
    return json.loads(path.read_text(encoding="utf-8"))


def value(item: Any) -> Any:
    """Unwrap token objects that store their value under ``value``."""
    if isinstance(item, dict) and "value" in item:
        return item["value"]
    return item


def swift_string(text: str) -> str:
    """Quote a string for Swift source."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def identifier(key: str) -> str:
    """Convert a JSON key to a valid lowerCamelCase source identifier."""
    parts = re.findall(r"[A-Za-z0-9]+", key)
    if not parts:
        raise ValueError(f"JSON key {key!r} has no identifier characters")
    result = parts[0][:1].lower() + parts[0][1:]
    result += "".join(part[:1].upper() + part[1:] for part in parts[1:])
    if result[:1].isdigit():
        result = "_" + result
    return result


def identifiers(keys: list[str]) -> dict[str, str]:
    """Return identifiers and fail if sanitizing creates a collision."""
    result: dict[str, str] = {}
    owners: dict[str, str] = {}
    for key in keys:
        name = identifier(key)
        if name in owners:
            raise ValueError(
                f"identifier collision: {key!r} and {owners[name]!r} both become {name!r}"
            )
        owners[name] = key
        result[key] = name
    return result


def color_hex(colors: dict[str, Any], name: str) -> str:
    """Return a color hex value from a palette."""
    item = colors[name]
    raw = item["hex"] if isinstance(item, dict) and "hex" in item else value(item)
    return str(raw)


def scene_color(colors: dict[str, Any], spec: Any) -> tuple[str, float]:
    """Resolve a scene color reference and its optional alpha."""
    if isinstance(spec, str):
        if spec.startswith("#"):
            return spec, 1.0
        return color_hex(colors, spec), 1.0
    if "hex" in spec:
        return str(spec["hex"]), float(spec.get("alpha", 1.0))
    return color_hex(colors, str(spec["base"])), float(spec.get("alpha", 1.0))


def swift_color(hex_value: str, alpha: float = 1.0) -> str:
    """Render a Swift Color initializer."""
    return f"Color(hex: {swift_string(hex_value.lstrip('#'))}, opacity: {alpha:g})"


def swift_palette(name: str, colors: dict[str, Any]) -> list[str]:
    """Render one Swift color palette."""
    lines = [f"    static let {name} = Palette("]
    for key in PALETTE_FIELDS:
        lines.append(f"        {key}: {swift_color(color_hex(colors, key))},")
    lines[-1] = lines[-1].rstrip(",")  # older Swift rejects a trailing comma
    lines.append("    )")
    return lines


def swift_scene_palette(name: str, scene: dict[str, Any], colors: dict[str, Any]) -> list[str]:
    """Render one Swift scene palette, including transparent volumes."""
    lines = [f"    static let {name} = ScenePalette("]
    for key, spec in scene.items():
        hex_value, alpha = scene_color(colors, spec)
        lines.append(f"        {identifier(key)}: {swift_color(hex_value, alpha)},")
    lines[-1] = lines[-1].rstrip(",")  # older Swift rejects a trailing comma
    lines.append("    )")
    return lines


def tk_binding(key: str, modifiers: list[str]) -> str:
    """Build the Tk binding used by the Windows preview."""
    names = {"right": "Right", "left": "Left", "home": "Home", "space": "space"}
    key_name = names.get(key, key)
    if len(key_name) == 1:
        key_name = key_name.upper() if "shift" in {item.lower() for item in modifiers} else key_name.lower()
    prefix = "-".join("Control" if item == "primary" else item.title() for item in modifiers)
    return f"<{prefix + '-' if prefix else ''}{key_name}>"


def shortcut_rows(ui: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize shortcut records."""
    shortcuts = ui.get("shortcuts", {})
    if not isinstance(shortcuts, dict):
        raise ValueError("shortcuts must be an object keyed by action id")
    identifiers(list(shortcuts))
    rows = []
    for action_id, spec in shortcuts.items():
        if not isinstance(spec, dict):
            raise ValueError(f"shortcut {action_id!r} must be an object")
        key = str(spec["key"])
        modifiers = [str(item) for item in spec.get("modifiers", [])]
        rows.append(
            {
                "action_id": action_id,
                "label": str(spec["label"]),
                "key": key,
                "modifiers": modifiers,
                "single_key": bool(spec.get("singleKey", False)),
                "tk_binding": tk_binding(key, modifiers),
            }
        )
    return rows


def transport_rows(ui: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Resolve transport labels, reusing a shortcut label when the entry names one."""
    shortcuts = ui.get("shortcuts", {})
    rows: dict[str, dict[str, str]] = {}
    for key, spec in ui.get("transport", {}).items():
        if "label" in spec:
            label = spec["label"]
        elif spec.get("shortcut") in shortcuts:
            label = shortcuts[spec["shortcut"]]["label"]
        else:
            raise ValueError(f"transport {key!r} needs a label or a known shortcut")
        rows[key] = {"label": str(label), "help": str(spec["help"])}
    return rows


def validate_ui(ui: dict[str, Any]) -> None:
    """Fail when two shared keys that must agree have drifted apart."""
    unavailable = ui["statusLevels"]["unavailable"]["word"]
    if "notRecorded" in ui and ui["notRecorded"] != unavailable:
        raise ValueError("notRecorded must equal statusLevels.unavailable.word")
    pinned = ui.get("teamFilter", {}).get("pinnedLevels", [])
    unknown = [level for level in pinned if level not in ui["statusLevels"]]
    if unknown:
        raise ValueError(f"teamFilter.pinnedLevels has unknown levels: {unknown}")


def swift_string_constant(name: str, text: str, indent: str = "        ", limit: int = 140) -> list[str]:
    """Render one string constant, splitting a long literal over lines with ``+``."""
    single = f"{indent}static let {name} = {swift_string(text)}"
    if len(single) <= limit:
        return [single]
    room = limit - len(indent) - 8  # quotes plus the " +" tail
    words = text.split(" ")
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(swift_string(candidate)) > room and current:
            chunks.append(current + " ")
            current = word
        else:
            current = candidate
    chunks.append(current)
    lines = [f"{indent}static let {name} ="]
    for index, chunk in enumerate(chunks):
        tail = " +" if index < len(chunks) - 1 else ""
        lines.append(f"{indent}    {swift_string(chunk)}{tail}")
    return lines


def swift_text_enum(name: str, section: dict[str, Any]) -> list[str]:
    """Render a nested Swift enum of string, string-list, and integer constants."""
    ids = identifiers(list(section))
    lines = [f"    enum {name} {{"]
    for key, item in section.items():
        if isinstance(item, str):
            lines.extend(swift_string_constant(ids[key], item))
        elif isinstance(item, bool):
            continue
        elif isinstance(item, int):
            lines.append(f"        static let {ids[key]}: Int = {item}")
        elif isinstance(item, list) and all(isinstance(entry, str) for entry in item):
            values = ", ".join(swift_string(entry) for entry in item)
            lines.append(f"        static let {ids[key]}: [String] = [{values}]")
    lines.append("    }")
    return lines


def swift_weight(spec: dict[str, Any]) -> str:
    """Map a JSON font weight to a SwiftUI weight."""
    return ".semibold" if spec.get("weight") == "semibold" else ".regular"


def swift_type_lines(type_scale: dict[str, Any]) -> list[str]:
    """Render Swift font definitions from the type scale."""
    lines = ["enum TypeScale {"]
    for name, spec in type_scale.items():
        if name == "families":
            continue
        design = ", design: .monospaced" if name == "mono" else ""
        digits = ".monospacedDigit()" if spec.get("tabularDigits") else ""
        line = (
            f"    static let {identifier(name)} = Font.system(size: {spec['size']}, "
            f"weight: {swift_weight(spec)}{design}){digits}"
        )
        lines.append(line)
    lines.extend(["}", ""])
    return lines


def family_names(text: str) -> tuple[str, ...]:
    """Parse comma or slash separated font family descriptions."""
    cleaned = text.replace("fallback", "")
    names = [part.strip() for part in re.split(r"[,/]", cleaned) if part.strip()]
    return tuple(names)


def python_type_table(
    type_scale: dict[str, Any],
) -> tuple[dict[str, tuple[Any, ...]], dict[str, tuple[str, ...]]]:
    """Build Tk type and fallback tables from JSON."""
    families = type_scale["families"]
    ui_families = family_names(str(families["windows"]))
    mono_families = family_names(str(type_scale["mono"]["windows"]))
    table: dict[str, tuple[Any, ...]] = {}
    for name, spec in type_scale.items():
        if name == "families":
            continue
        family = mono_families if name == "mono" else ui_families
        row: tuple[Any, ...] = (family[0], -int(spec["size"]))
        if spec.get("weight") == "semibold":
            row += ("bold",)
        table[name] = row
    return table, {"ui": ui_families, "mono": mono_families}


def generate_swift(tokens: dict[str, Any], ui: dict[str, Any]) -> str:
    """Render the generated Swift design module."""
    dark = tokens["colors"]["dark"]
    light = tokens["colors"]["light"]
    type_scale = tokens["typeScale"]
    shortcuts = shortcut_rows(ui)
    event_ids = identifiers(list(ui.get("eventTemplates", {})))
    lines = [
        "// GENERATED FILE. Do not edit. Regenerate with: python tools/generate_design.py",
        "import AppKit",
        "import SwiftUI",
        "",
        "struct ScenePalette {",
        "    let droneBody, droneArms, rotorDisc, targetRing, keepOutVolume, grid, floor: Color",
        "    var droneBodyNS: NSColor { NSColor(droneBody) }",
        "    var droneArmsNS: NSColor { NSColor(droneArms) }",
        "    var rotorDiscNS: NSColor { NSColor(rotorDisc) }",
        "    var targetRingNS: NSColor { NSColor(targetRing) }",
        "    var keepOutVolumeNS: NSColor { NSColor(keepOutVolume) }",
        "    var gridNS: NSColor { NSColor(grid) }",
        "    var floorNS: NSColor { NSColor(floor) }",
        "}",
        "",
        "enum Theme {",
        "    struct Palette {",
        "        let background, surface, line, lineStrong, text, textSecondary: Color",
        "        let accent, target, keepOut, ok, caution, alert, info, onAlert: Color",
        "        let selectionFill, hoverFill: Color",
        "        let bandFill, atlasStop1, atlasStop2, atlasStop3, atlasStop4: Color",
    ]
    for field in PALETTE_FIELDS:
        lines.append(f"        var {identifier(field)}NS: NSColor {{ NSColor({identifier(field)}) }}")
    lines.extend(["    }", *swift_palette("dark", dark), *swift_palette("light", light)])
    lines.extend(swift_scene_palette("darkScene3D", tokens["scene3D"]["dark"], dark))
    lines.extend(swift_scene_palette("lightScene3D", tokens["scene3D"]["light"], light))
    lines.extend(
        [
            "    static func palette(for colorScheme: ColorScheme) -> Palette {",
            "        colorScheme == .light ? light : dark",
            "    }",
            "    static func scene3D(for colorScheme: ColorScheme) -> ScenePalette {",
            "        colorScheme == .light ? lightScene3D : darkScene3D",
            "    }",
            "}",
            "",
            "private extension Color {",
            "    init(hex: String, opacity: Double = 1) {",
            "        let value = UInt64(hex, radix: 16) ?? 0",
            "        self.init(.sRGB, red: Double((value >> 16) & 0xff) / 255,",
            "            green: Double((value >> 8) & 0xff) / 255,",
            "            blue: Double(value & 0xff) / 255, opacity: opacity)",
            "    }",
            "}",
            "",
        ]
    )
    lines.extend(swift_type_lines(type_scale))
    lines.extend(["enum Space {"])
    for key, number in tokens["spacing"].items():
        lines.append(f"    static let {identifier(key)}: CGFloat = {number}")
    lines.extend(["}", "", "enum Radius {"])
    for key, number in tokens["radii"].items():
        if isinstance(number, (int, float)):
            lines.append(f"    static let {identifier(key)}: CGFloat = {number}")
    lines.extend(["}", "", "enum Motion {"])
    lines.append(f"    static let fast: TimeInterval = {tokens['motion']['fastMs'] / 1000:g}")
    lines.append(f"    static let standard: TimeInterval = {tokens['motion']['standardMs'] / 1000:g}")
    lines.extend(
        [
            "    static func duration(_ value: TimeInterval, reduceMotion: Bool) -> TimeInterval {",
            "        reduceMotion ? 0 : value",
            "    }",
            "}",
            "",
            "enum LayoutMetrics {",
        ]
    )
    # Every shared layout key is emitted, including sidebarWidthWindows, which the Windows app no longer
    # reads (it uses windowsLayout.sidebarWidth); it stays so the generated Swift file keeps its shape.
    for key, number in tokens["layout"].items():
        lines.append(f"    static let {identifier(key)}: CGFloat = {number}")
    lines.extend(
        [
            "}",
            "",
            "enum UIText {",
            f"    static let honestyBadge = {swift_string(ui['honesty']['badge'])}",
            f"    static let aboutButton = {swift_string(ui['honesty']['aboutButton'])}",
            "    static let update: [String: String] = [",
        ]
    )
    for key, item in ui.get("update", {}).items():
        lines.append(f"        {swift_string(key)}: {swift_string(str(item))},")
    lines.append("    ]")
    strip_ids = identifiers(list(ui["statusStrip"]))
    for label in ui["statusStrip"]:
        lines.append(f"    static let {strip_ids[label]} = {swift_string(label)}")
    thresholds = ui["displayThresholds"]
    lines.extend(
        [
            f"    static let targetAgeLateS: Double = {thresholds['targetAgeLateS']['value']}",
            f"    static let targetAgeLostS: Double = {thresholds['targetAgeLostS']['value']}",
            "    struct StatusLevel {",
            "        let word: String",
            "        let alternateWord: String?",
            "        let shape: String",
            "        let color: String",
            "    }",
            "    static let statusLevels: [String: StatusLevel] = [",
        ]
    )
    for key, spec in ui["statusLevels"].items():
        alternate = swift_string(spec["alternateWord"]) if "alternateWord" in spec else "nil"
        lines.append(
            f"        {swift_string(key)}: StatusLevel(word: {swift_string(spec['word'])}, "
            f"alternateWord: {alternate}, shape: {swift_string(spec['shape'])}, color: {swift_string(spec['color'])}),"
        )
    lines.extend(["    ]", "    static let statusShapesMacOS: [String: String] = ["])
    for key, item in ui["statusShapes"]["macOS"].items():
        lines.append(f"        {swift_string(key)}: {swift_string(item)},")
    lines.extend(["    ]", "    static let statusShapesWindows: [String: String] = ["])
    for key, item in ui["statusShapes"]["windows"].items():
        lines.append(f"        {swift_string(key)}: {swift_string(item)},")
    lines.extend(["    ]", "    static let fieldLabels: [String: String] = ["])
    for key, item in sorted(ui["fieldLabels"].items()):
        lines.append(f"        {swift_string(key)}: {swift_string(item)},")
    lines.extend(
        [
            "    ]",
            "    static let aboutBodyFields: [String] = "
            f"{json.dumps(ui['honesty']['aboutBodyFields'], ensure_ascii=False)}",
            f"    static let shortcutRule = {swift_string(ui['shortcutRule'])}",
            f"    static let droneCompactName = {swift_string(ui['droneNaming']['compact'])}",
            f"    static let droneHeadingName = {swift_string(ui['droneNaming']['heading'])}",
            f"    static let targetAgeLateNote = {swift_string(thresholds['targetAgeLateS']['note'])}",
            f"    static let targetAgeLostNote = {swift_string(thresholds['targetAgeLostS']['note'])}",
            "    struct Shortcut {",
            "        let actionID: String",
            "        let label: String",
            "        let key: String",
            "        let modifiers: [String]",
            "        let singleKey: Bool",
            "    }",
            "    static let shortcuts: [Shortcut] = [",
        ]
    )
    for row in shortcuts:
        mods = ", ".join(swift_string(item) for item in row["modifiers"])
        lines.append(
            f"        Shortcut(actionID: {swift_string(row['action_id'])}, label: {swift_string(row['label'])}, "
            f"key: {swift_string(row['key'])}, modifiers: [{mods}], singleKey: {str(row['single_key']).lower()}),"
        )
    lines.extend(["    ]"])
    for key, item in ui.get("eventTemplates", {}).items():
        event_name = event_ids[key][:1].upper() + event_ids[key][1:]
        lines.append(f"    static let event{event_name} = {swift_string(item)}")
    # Shared copy sections. Each one is a nested enum so a merge touches one block.
    if "notRecorded" in ui:
        lines.append(f"    static let notRecorded = {swift_string(ui['notRecorded'])}")
    if "notInReplay" in ui:
        lines.append(f"    static let notInReplayTitle = {swift_string(ui['notInReplay']['title'])}")
    for json_key, enum_name in (
        ("eventDroneList", "EventDroneList"),
        ("teamFilter", "TeamFilter"),
        ("menus", "Menus"),
        ("inspector", "Inspector"),
        ("results", "Results"),
        ("atlas", "Atlas"),
    ):
        if json_key in ui:
            lines.extend(swift_text_enum(enum_name, ui[json_key]))
    transport = transport_rows(ui)
    if transport:
        transport_ids = identifiers(list(transport))
        lines.extend(
            [
                "    struct TransportControl {",
                "        let label: String",
                "        let help: String",
                "    }",
                "    enum Transport {",
            ]
        )
        for key, row in transport.items():
            lines.append(
                f"        static let {transport_ids[key]} = TransportControl(label: {swift_string(row['label'])}, "
                f"help: {swift_string(row['help'])})"
            )
        lines.append("    }")
    lines.extend(["}", ""])
    return "\n".join(lines)


def generate_python(tokens: dict[str, Any], ui: dict[str, Any]) -> str:
    """Render the generated Python design module."""
    colors = tokens["colors"]
    type_table, fallbacks = python_type_table(tokens["typeScale"])
    rows = shortcut_rows(ui)

    def palette(theme: str) -> str:
        values = [color_hex(colors[theme], field) for field in PALETTE_FIELDS]
        return f"Palette({', '.join(repr(item) for item in values)})"

    def scene_palette(theme: str) -> str:
        values = []
        for spec in tokens["scene3D"][theme].values():
            hex_value, alpha = scene_color(colors[theme], spec)
            values.append(f"SceneColor({hex_value!r}, {alpha!r})")
        return f"ScenePalette({', '.join(values)})"

    lines = [
        "# GENERATED FILE. Do not edit. Regenerate with: python tools/generate_design.py",
        "from __future__ import annotations",
        "from dataclasses import dataclass",
        "",
        "@dataclass(frozen=True)",
        "class Palette:",
    ]
    lines.extend(f"    {identifier(field)}: str" for field in PALETTE_FIELDS)
    lines.extend(
        [
            "",
            "@dataclass(frozen=True)",
            "class SceneColor:",
            "    hex: str",
            "    alpha: float = 1.0",
            "",
            "@dataclass(frozen=True)",
            "class ScenePalette:",
            "    drone_body: SceneColor",
            "    drone_arms: SceneColor",
            "    rotor_disc: SceneColor",
            "    target_ring: SceneColor",
            "    keep_out_volume: SceneColor",
            "    grid: SceneColor",
            "    floor: SceneColor",
            "",
            f"DARK = {palette('dark')}",
            f"LIGHT = {palette('light')}",
            f"DARK_SCENE3D = {scene_palette('dark')}",
            f"LIGHT_SCENE3D = {scene_palette('light')}",
            "",
            "def palette(theme: str = 'dark') -> Palette:",
            "    return LIGHT if str(theme).lower() == 'light' else DARK",
            "",
            "def scene3d(theme: str = 'dark') -> ScenePalette:",
            "    return LIGHT_SCENE3D if str(theme).lower() == 'light' else DARK_SCENE3D",
            "",
            f"FONT_FALLBACKS = {fallbacks!r}",
            "",
            "TYPE = {",
        ]
    )
    for name, spec in type_table.items():
        lines.append(f"    {name!r}: {spec!r},")
    lines.extend(
        [
            "}",
            "",
            "def tk_font(role: str):",
            "    return TYPE[role]",
            "",
            f"SPACE = {tokens['spacing']!r}",
            f"RADIUS = {tokens['radii']!r}",
            "MOTION_MS = {",
            f"    'fast': {tokens['motion']['fastMs']},",
            f"    'standard': {tokens['motion']['standardMs']},",
            f"    'curve': {tokens['motion']['curve']!r},",
            f"    'reduced': {tokens['motion']['reducedMotionMs']},",
            "}",
            f"LAYOUT = {tokens['layout']!r}",
            # Windows-only control, column, and window sizes; the Swift output has no use for them.
            f"WIN_LAYOUT = {pprint.pformat({key: value for key, value in tokens.get('windowsLayout', {}).items() if key != 'comment'}, width=100, sort_dicts=False)}",
            "",
            "@dataclass(frozen=True)",
            "class Shortcut:",
            "    action_id: str",
            "    label: str",
            "    key: str",
            "    modifiers: tuple[str, ...]",
            "    tk_binding: str",
            "    single_key: bool",
            "",
            "SHORTCUTS = (",
        ]
    )
    for row in rows:
        lines.append(
            f"    Shortcut({row['action_id']!r}, {row['label']!r}, {row['key']!r}, "
            f"{tuple(row['modifiers'])!r}, {row['tk_binding']!r}, {row['single_key']!r}),"
        )
    lines.extend([")", "", f"UI_TEXT = {pprint.pformat(ui, width=100, sort_dicts=False)}", ""])
    transport = transport_rows(ui)
    if transport:
        lines.extend(
            [
                "# Transport labels with shortcut labels resolved, so callers never re-derive them.",
                f"TRANSPORT = {pprint.pformat(transport, width=100, sort_dicts=False)}",
                "",
            ]
        )
    return "\n".join(lines)


def outputs(
    tokens_path: Path = TOKENS_PATH,
    ui_path: Path = UI_PATH,
    root: Path = ROOT,
) -> dict[Path, str]:
    """Return expected generated files."""
    tokens = load(tokens_path)
    ui = load(ui_path)
    validate_ui(ui)
    swift_output = root / SWIFT_PATH.relative_to(ROOT)
    python_output = root / PY_PATH.relative_to(ROOT)
    return {
        swift_output: generate_swift(tokens, ui),
        python_output: generate_python(tokens, ui),
    }


def main(argv: list[str] | None = None) -> int:
    """Generate files or check that checked-in outputs are current."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--tokens", type=Path, default=TOKENS_PATH)
    parser.add_argument("--ui-text", type=Path, default=UI_PATH)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    expected = outputs(args.tokens, args.ui_text, args.output_root)
    mismatches = []
    for path, content in expected.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                mismatches.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    if mismatches:
        print("generated files differ:", *mismatches, sep="\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
