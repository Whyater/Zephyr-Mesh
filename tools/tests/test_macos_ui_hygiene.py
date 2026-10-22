"""Keep macOS UI source aligned with generated design and copy rules."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "tools" / "macos" / "ZephyrMeshApp" / "Sources"
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


# A non-zero number written straight into a padding, frame, spacing, minimum length or corner
# radius. Those sizes come from the generated tokens (Space, Radius, LayoutMetrics); 0 is allowed.
RAW_LAYOUT_NUMBER = re.compile(
    r"(?:\.padding\((?:\.[a-zA-Z]+(?:, )?|\[[^\]]*\], )?-?"
    r"|\.frame\([^)]*?\b(?:min|max|ideal)?(?:[wW]idth|[hH]eight): "
    r"|\bspacing: |\bminLength: |\bcornerRadius: )"
    r"(?:[1-9][0-9]*(?:\.[0-9]+)?|0\.[0-9]*[1-9])\b"
)
# Generated tokens, and files owned by other work (the updater and the resource loader).
RAW_LAYOUT_EXEMPT = {"DesignSystem.swift", "ReleaseUpdater.swift", "ResourceLocator.swift"}


def test_layout_sizes_come_from_tokens() -> None:
    violations = []
    for path in source_files():
        if path.name in RAW_LAYOUT_EXEMPT:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if RAW_LAYOUT_NUMBER.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")
    assert not violations, "raw layout number, use a design token:\n" + "\n".join(violations)


def test_raw_layout_number_rule_catches_literals() -> None:
    """The rule must flag real literals, or the test above proves nothing."""
    assert RAW_LAYOUT_NUMBER.search(".padding(12)")
    assert RAW_LAYOUT_NUMBER.search(".padding(.horizontal, 8)")
    assert RAW_LAYOUT_NUMBER.search(".frame(width: 136, alignment: .leading)")
    assert RAW_LAYOUT_NUMBER.search(".frame(maxWidth: .infinity, minHeight: 56)")
    assert not RAW_LAYOUT_NUMBER.search("SCNCylinder(radius: 0.005, height: 1)"), "scene geometry is metres, not layout"
    assert RAW_LAYOUT_NUMBER.search("HStack(spacing: 6)")
    assert RAW_LAYOUT_NUMBER.search("RoundedRectangle(cornerRadius: 4)")
    assert not RAW_LAYOUT_NUMBER.search("VStack(spacing: 0)")
    assert not RAW_LAYOUT_NUMBER.search(".padding(Space.md)")
    assert not RAW_LAYOUT_NUMBER.search(".frame(width: LayoutMetrics.eventWordWidth)")
