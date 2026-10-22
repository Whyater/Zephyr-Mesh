"""Source hygiene for the Windows replay UI: tokens only, honest wording, readable code."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tools.desktop.design_tokens import SHORTCUTS, UI_TEXT


ROOT = Path(__file__).resolve().parents[2]
UI_MODULES = (
    "tools/desktop/windows_preview.py",
    "tools/desktop/ui_theme.py",
    "tools/desktop/ui_widgets.py",
    "tools/desktop/ui_icons.py",
    "tools/desktop/ui_format.py",
    "tools/desktop/ui_viewers.py",
    "tools/desktop/scene_view.py",
    "tools/desktop/scene_projection.py",
    "tools/desktop/ui_settings.py",
    "tools/desktop/replay_session.py",
    "tools/desktop/results_data.py",
    "tools/desktop/ui_results.py",
    "tools/desktop/ui_results_chart.py",
    "tools/desktop/ui_results_atlas.py",
)
VISIBLE_KEYWORDS = {"text", "label", "title", "message", "accelerator"}
VISIBLE_CALLS = {"title", "showinfo", "showwarning", "showerror", "askopenfilename", "add_command", "heading"}
SNAKE_CASE = re.compile(r"\b[a-z]+_[a-z0-9_]+\b")


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def _strings(node: ast.AST, skip: set[int] | None = None) -> list[str]:
    """String constants inside an expression, including f-string literal parts."""
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str) and id(child) not in (skip or set()):
            found.append(child.value)
    return found


def _key_constants(tree: ast.AST) -> set[int]:
    """Constants used as data keys (``record["heading_name"]``, ``.get(...)``, ``field_label(...)``), not shown."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            ids.update(id(child) for child in ast.walk(node.slice) if isinstance(child, ast.Constant))
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in {"get", "field_label", "format_value", "accelerator", "icon", "status", "humanize_key"}:
                ids.update(id(argument) for argument in node.args if isinstance(argument, ast.Constant))
    return ids


def _visible_strings(tree: ast.AST) -> list[str]:
    keys = _key_constants(tree)
    visible = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg in VISIBLE_KEYWORDS:
                visible.extend(_strings(keyword.value, keys))
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name in VISIBLE_CALLS:
            for argument in node.args:
                visible.extend(_strings(argument, keys))
    return visible


def test_legacy_tokens_are_gone_and_unused():
    assert not (ROOT / "tools" / "desktop" / "legacy_tokens.py").exists()
    for relative in UI_MODULES:
        assert "legacy_tokens" not in _source(relative)


@pytest.mark.parametrize("relative", UI_MODULES)
def test_no_hex_colors_long_lines_or_em_dashes(relative):
    source = _source(relative)
    assert not re.search(r"#[0-9A-Fa-f]{6}\b|#[0-9A-Fa-f]{3}\b", source), "colors come from tools/desktop/design_tokens.py only"
    assert "\u2014" not in source
    long_lines = [number for number, line in enumerate(source.splitlines(), 1) if len(line) > 140]
    assert not long_lines, f"lines over 140 characters: {long_lines}"


@pytest.mark.parametrize("relative", UI_MODULES)
def test_font_sizes_are_never_positive_literals(relative):
    source = _source(relative)
    assert not re.search(r"font\s*=\s*\(\s*[^()]*?,\s*\d", source), "use the generated named fonts"
    assert not re.search(r"\bsize\s*=\s*\d", source), "Tk sizes are negative pixels from TYPE"


@pytest.mark.parametrize("relative", UI_MODULES)
def test_no_user_visible_snake_case(relative):
    tree = ast.parse(_source(relative))
    offenders = [text for text in _visible_strings(tree) if SNAKE_CASE.search(text)]
    assert not offenders, offenders


def test_synthetic_appears_in_at_most_two_user_strings():
    count = 0
    for relative in UI_MODULES:
        tree = ast.parse(_source(relative))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                count += "synthetic" in node.value.lower()
    assert count <= 2, f"{count} string literals mention synthetic; the badge text comes from UI_TEXT"


def test_status_words_come_from_ui_text():
    words = {spec["word"] for spec in UI_TEXT["statusLevels"].values()}
    words |= {spec["alternateWord"] for spec in UI_TEXT["statusLevels"].values() if "alternateWord" in spec}
    for relative in UI_MODULES:
        tree = ast.parse(_source(relative))
        literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert not (literals & words), f"{relative} hardcodes status words {sorted(literals & words)}"
    from tools.desktop.ui_icons import STATUS_SHAPES
    from tools.desktop.ui_theme import LEVELS, STATUS_GLYPHS

    assert LEVELS is UI_TEXT["statusLevels"]
    assert STATUS_GLYPHS == UI_TEXT["statusShapes"]["windows"]
    assert {spec["shape"] for spec in LEVELS.values()} <= set(STATUS_SHAPES)


def test_menu_shortcut_table_equals_ui_text_shortcuts():
    from tools.desktop import ui_format as fmt
    from tools.desktop.windows_preview import MENU_LAYOUT, SHORTCUT_HANDLERS, menu_accelerators

    assert set(SHORTCUT_HANDLERS) == set(UI_TEXT["shortcuts"]) == {item.action_id for item in SHORTCUTS}
    menu_actions = {action for _menu, items in MENU_LAYOUT for _label, action, _kind in items if action}
    assert menu_actions == set(UI_TEXT["shortcuts"]) - {"focusFilter"}, "every shortcut except the filter focus has a menu item"
    for action, text in menu_accelerators().items():
        spec = UI_TEXT["shortcuts"][action]
        assert text == fmt.accelerator(action)
        assert text.split("+")[-1].lower() == spec["key"]
        assert ("Ctrl" in text) == ("primary" in spec["modifiers"])
        assert ("Shift" in text) == ("shift" in spec["modifiers"])
