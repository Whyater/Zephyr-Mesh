"""Keep macOS UI source aligned with generated design and copy rules."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "macos" / "ZephyrMeshApp" / "Sources"
# Empty since the macOS rebuild deleted LegacyViews.swift and LegacyDesignShim.swift;
# every UI source is now covered by every rule below.
TEMPORARY_ALLOWLIST: set[str] = set()
# Generated tokens plus the strict parsers kept byte-for-byte from the foundation
# commit. They hold no views; they are exempt from the line-length rule only.
LINE_LENGTH_EXEMPT = {"DesignSystem.swift", "EvidenceInspector.swift", "ReleaseUpdater.swift", "ScenarioRunInspector.swift"}
STRING = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
VISIBLE_STRING = re.compile(
    r'\b(?:Text|Label|Button|Toggle|TextField|Alert|SectionTitle|DataSection|DataRow|MetricTile|accessibilityLabel|help)\s*\([^\n"]*"([^"\\]*(?:\\.[^"\\]*)*)"'
)
SNAKE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b")
RGB = re.compile(r"\b(?:NSColor|Color)\s*\(\s*(?:red|green|blue|white|black|gray|hue|calibratedRed|srgbRed)\s*[:=]")


def source_files() -> list[Path]:
    return sorted(SOURCE_ROOT.rglob("*.swift"))


def test_temporary_allowlist_entries_exist() -> None:
    existing = {path.name for path in source_files()}
    missing = sorted(TEMPORARY_ALLOWLIST - existing)
    assert not missing, f"temporary allowlist names missing files: {missing}"


def test_macos_ui_hygiene() -> None:
    violations: list[str] = []
    synthetic_count = 0
    for path in source_files():
        if path.name in TEMPORARY_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        for line_number, line in enumerate(text.splitlines(), 1):
            if chr(0x2014) in line:
                violations.append(f"{relative}:{line_number}: em dash")
            if path.name not in LINE_LENGTH_EXEMPT and len(line) > 140:
                violations.append(f"{relative}:{line_number}: line exceeds 140 characters")
        if path.name != "DesignSystem.swift":
            if (
                re.search(r"#[0-9A-Fa-f]{3,8}\b", text) or RGB.search(text)
            ):
                violations.append(f"{relative}: hex or RGB color literal")
            if any(token in text for token in ("NSColor.system", "Color.accentColor", "controlAccentColor", "labelColor")):
                violations.append(f"{relative}: system or accent color literal")
            if "Font.system(size:" in text:
                violations.append(f"{relative}: direct system font literal")
            for line_number, line in enumerate(text.splitlines(), 1):
                for match in VISIBLE_STRING.finditer(line):
                    value = match.group(1)
                    if "synthetic" in value.lower():
                        synthetic_count += 1
                    if SNAKE.search(value):
                        violations.append(f"{relative}:{line_number}: user-visible snake_case string literal")
    if synthetic_count > 2:
        violations.append(f"user-visible synthetic string literals: {synthetic_count}")
    assert not violations, "\n".join(violations)


def test_honesty_badge_text_comes_only_from_generated_ui_text() -> None:
    """The badge words appear once in the window, so no view may spell them itself."""
    offenders = [
        str(path.relative_to(ROOT))
        for path in source_files()
        if path.name != "DesignSystem.swift" and "Simulation replay" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


def test_scene_never_uses_default_camera_control() -> None:
    """The scene uses its own clamped orbit controller, never SceneKit's free camera."""
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        assert "allowsCameraControl = true" not in text, path.name
        assert "defaultCameraController" not in text, path.name
