"""Independent checks for the generated Horizon design foundation."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from tools.generate_design import shortcut_rows, tk_binding

ROOT = Path(__file__).parents[1]
PYTHON = sys.executable


def run_generator(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the generator with the repository Python interpreter."""
    return subprocess.run(
        [PYTHON, "tools/generate_design.py", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_generated_files_are_current() -> None:
    """Checked-in generated files must match both source JSON documents."""
    result = run_generator("--check")
    assert result.returncode == 0, result.stderr


def test_shortcut_bindings_are_exact() -> None:
    """Every JSON shortcut has the intended Tk spelling."""
    ui = json.loads((ROOT / "design/ui_text.json").read_text())
    expected = {
        "playPause": "<space>",
        "stepForward": "<Right>",
        "stepBack": "<Left>",
        "restart": "<Home>",
        "focusSelected": "<f>",
        "resetCamera": "<r>",
        "toggleTopView": "<t>",
        "toggleKeepOut": "<k>",
        "toggleInspector": "<i>",
        "openEvidenceReport": "<Control-o>",
        "openScenarioRun": "<Control-Shift-O>",
        "focusFilter": "<Control-l>",
        "aboutReplay": "<Control-i>",
    }
    rows = {row["action_id"]: row["tk_binding"] for row in shortcut_rows(ui)}
    assert rows == expected
    assert tk_binding("o", ["primary"]) == "<Control-o>"
    assert tk_binding("o", ["primary", "shift"]) == "<Control-Shift-O>"


def test_changing_type_size_and_scene_color_changes_both_outputs(tmp_path: Path) -> None:
    """A temporary JSON edit must flow into both generated modules."""
    tokens = json.loads((ROOT / "design/horizon_tokens.json").read_text())
    tokens["typeScale"]["body"]["size"] = 17
    tokens["scene3D"]["dark"]["droneBody"] = "accent"
    token_path = tmp_path / "horizon_tokens.json"
    token_path.write_text(json.dumps(tokens))
    ui_path = tmp_path / "ui_text.json"
    ui_path.write_text((ROOT / "design/ui_text.json").read_text())
    output_root = tmp_path / "out"
    output_root.mkdir()

    result = run_generator(
        "--tokens",
        str(token_path),
        "--ui-text",
        str(ui_path),
        "--output-root",
        str(output_root),
    )
    assert result.returncode == 0, result.stderr
    swift = (output_root / "macos/ZephyrMeshApp/Sources/ZephyrMeshApp/DesignSystem.swift").read_text()
    python = (output_root / "desktop/design_tokens.py").read_text()
    assert "Font.system(size: 17" in swift
    assert "'body': ('Segoe UI Variable Text', -17)" in python
    assert "SceneColor('#7CC4FF'" in python
    assert 'droneBody: Color(hex: "7CC4FF"' in swift


def test_contrast_check_passes() -> None:
    """Marked contrast candidates are reported without hiding stale-fill errors."""
    result = subprocess.run([PYTHON, "tools/check_contrast.py"], cwd=ROOT)
    assert result.returncode == 0


def run_contrast(path: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run the contrast checker, optionally against a temporary token copy."""
    arguments = [PYTHON, "tools/check_contrast.py"]
    if path is not None:
        arguments.append(str(path))
    return subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True)


def test_contrast_checker_covers_each_required_pair_once_per_theme() -> None:
    """Every required contrast pair is emitted once for both themes."""
    result = run_contrast()
    assert result.returncode == 0, result.stderr
    colors = ("text", "textSecondary", "accent", "target", "keepOut", "ok", "caution", "alert", "info")
    expected = {
        *{(foreground, background) for foreground in colors for background in ("background", "surface")},
        *{(foreground, fill) for foreground in colors for fill in ("selectionFill", "hoverFill")},
        ("lineStrong", "surface"),
        ("onAlert", "alert"),
        ("onAlert", "caution"),
    }
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0] in ("dark", "light"):
            rows.append((fields[0], fields[1], fields[2]))
    counts = Counter(rows)
    assert set((foreground, background) for _, foreground, background in rows) == expected
    assert all(counts[(theme, foreground, background)] == 1 for theme in ("dark", "light") for foreground, background in expected)


def test_unmarked_failing_pair_exits_one(tmp_path: Path) -> None:
    """An unmarked failing contrast pair fails the checker."""
    document = json.loads((ROOT / "design/horizon_tokens.json").read_text())
    document["colors"]["light"]["lineStrong"] = "#FFFFFF"
    token_path = tmp_path / "horizon_tokens.json"
    token_path.write_text(json.dumps(document))

    result = run_contrast(token_path)
    assert result.returncode == 1
    assert "light lineStrong on surface" in result.stderr
    assert "FAIL" in result.stdout


def test_passing_needs_approval_is_stale(tmp_path: Path) -> None:
    """A needs-approval mark on a passing pair is rejected as stale."""
    document = json.loads((ROOT / "design/horizon_tokens.json").read_text())
    document["contrast"]["needs-approval"].append(
        {"theme": "dark", "foreground": "text", "background": "background"}
    )
    token_path = tmp_path / "horizon_tokens.json"
    token_path.write_text(json.dumps(document))

    result = run_contrast(token_path)
    assert result.returncode == 1
    assert "STALE-MARK" in result.stdout
    assert "dark text on background: stale needs-approval mark" in result.stderr


def test_status_levels_are_complete_and_json_has_no_em_dash() -> None:
    """Status records are complete and source text follows the punctuation rule."""
    ui = json.loads((ROOT / "design/ui_text.json").read_text())
    for level in ("ok", "caution", "alert", "waiting", "unavailable", "info"):
        assert {"word", "shape", "color"} <= ui["statusLevels"][level].keys()
    for path in (ROOT / "design/horizon_tokens.json", ROOT / "design/ui_text.json"):
        assert "\u2014" not in path.read_text()


def test_body_size_and_generated_swift_line_length() -> None:
    """The body size remains readable and generated Swift stays bounded."""
    tokens = json.loads((ROOT / "design/horizon_tokens.json").read_text())
    assert tokens["typeScale"]["body"]["size"] >= 13
    lines = (ROOT / "macos/ZephyrMeshApp/Sources/ZephyrMeshApp/DesignSystem.swift").read_text().splitlines()
    assert max(map(len, lines), default=0) <= 160


def test_generated_swift_identifiers_and_braces_are_valid() -> None:
    """Generated Swift declarations use identifiers and balanced braces."""
    source = (ROOT / "macos/ZephyrMeshApp/Sources/ZephyrMeshApp/DesignSystem.swift").read_text()
    declarations = re.findall(r"\b(?:enum|struct|class|func|let|var|case)\s+([^\s:{(=,]+)", source)
    assert all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in declarations)
    without_strings = re.sub(r'"(?:\\.|[^"\\])*"', '""', source)
    depth = 0
    for character in without_strings:
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
        assert depth >= 0
    assert depth == 0


def test_shared_copy_sections_are_emitted_and_transport_reuses_shortcut_labels() -> None:
    """New copy sections reach both outputs; transport labels are not duplicated in the JSON."""
    import pytest

    from desktop.design_tokens import TRANSPORT, UI_TEXT
    from tools.generate_design import transport_rows, validate_ui

    ui = json.loads((ROOT / "design" / "ui_text.json").read_text(encoding="utf-8"))
    assert TRANSPORT == transport_rows(ui)
    assert TRANSPORT["restart"]["label"] == ui["shortcuts"]["restart"]["label"]
    assert TRANSPORT["stepBack"]["label"] == ui["shortcuts"]["stepBack"]["label"]
    assert TRANSPORT["stepForward"]["label"] == ui["shortcuts"]["stepForward"]["label"]
    assert "label" not in ui["transport"]["restart"]
    assert all(row["help"] for row in TRANSPORT.values())
    assert UI_TEXT["notRecorded"] == UI_TEXT["statusLevels"]["unavailable"]["word"] == "Not recorded"
    swift = (ROOT / "macos/ZephyrMeshApp/Sources/ZephyrMeshApp/DesignSystem.swift").read_text(encoding="utf-8")
    for line in (
        'static let notRecorded = "Not recorded"',
        "enum Menus {",
        "enum Inspector {",
        "enum TeamFilter {",
        'static let pinnedLevels: [String] = ["caution", "alert"]',
        'static let restart = TransportControl(label: "Restart", help:',
        'static let more = "+{count} more"',
    ):
        assert line in swift, line
    drifted = json.loads(json.dumps(ui))
    drifted["notRecorded"] = "Unknown"
    with pytest.raises(ValueError, match="notRecorded"):
        validate_ui(drifted)
    drifted = json.loads(json.dumps(ui))
    drifted["teamFilter"]["pinnedLevels"] = ["late"]
    with pytest.raises(ValueError, match="pinnedLevels"):
        validate_ui(drifted)
