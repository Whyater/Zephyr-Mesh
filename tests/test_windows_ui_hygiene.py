from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPORARY_ALLOWLIST = {"desktop/windows_preview.py", "desktop/legacy_tokens.py"}
UI_MODULES = {
    "desktop/scene_projection.py",
    "desktop/ui_settings.py",
    "desktop/replay_session.py",
}


def test_temporary_allowlist_points_to_existing_files():
    assert all((ROOT / path).is_file() for path in TEMPORARY_ALLOWLIST)


def test_new_ui_modules_follow_source_hygiene_rules():
    snake_case_text = re.compile(r"\b[A-Za-z]+_[A-Za-z0-9_]+\b")
    for relative in UI_MODULES:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "\u2014" not in source
        assert not re.search(r"#[0-9A-Fa-f]{3,8}\b", source)
        assert not any(len(line) > 140 for line in source.splitlines())
        for line in source.splitlines():
            if "text=" in line or "label=" in line:
                assert not snake_case_text.search(line)
