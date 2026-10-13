"""The design reference page is current, self-contained, and covers every token."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from tools.build_design_reference import JOBS

ROOT = Path(__file__).parents[1]
PAGE = ROOT / "design" / "reference.html"


def test_reference_page_is_current() -> None:
    result = subprocess.run([sys.executable, "tools/build_design_reference.py", "--check"], cwd=ROOT,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_reference_page_is_self_contained() -> None:
    text = PAGE.read_text(encoding="utf-8")
    for banned in ("<script", "<link", "@import", "http://", "https://", "src=\"/", "url(/"):
        assert banned not in text, f"reference.html must not contain {banned!r}"
    assert chr(0x2014) not in text, "no em dashes"


def test_every_palette_token_has_a_job_and_appears_on_the_page() -> None:
    tokens = json.loads((ROOT / "design" / "horizon_tokens.json").read_text(encoding="utf-8"))
    text = PAGE.read_text(encoding="utf-8")
    for theme, colors in tokens["colors"].items():
        for name in colors:
            assert name in JOBS, f"{theme} {name} has no job in tools/build_design_reference.py"
            assert f">{name}<" in text or f">{name} <" in text, f"{name} is missing from the page"
    ui = json.loads((ROOT / "design" / "ui_text.json").read_text(encoding="utf-8"))
    for spec in ui["statusLevels"].values():
        assert spec["word"] in text
