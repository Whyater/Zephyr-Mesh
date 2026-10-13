"""The checkout VERSION file is the one source of the app version."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_version_file_is_one_plain_release_number():
    value = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value), value


def test_fallback_versions_match_the_version_file():
    # Used only when no bundle version is available (`swift run`, a source checkout).
    expected = (ROOT / "VERSION").read_text(encoding="utf-8").strip() + "-dev"
    swift = (ROOT / "macos/ZephyrMeshApp/Sources/ZephyrMeshApp/ReleaseUpdater.swift").read_text(encoding="utf-8")
    assert f'?? "{expected}"' in swift
    windows = (ROOT / "desktop/windows_preview.py").read_text(encoding="utf-8")
    assert f'else "{expected}"' in windows


def test_local_build_defaults_to_the_dev_version():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_app", ROOT / "macos" / "build_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.development_version() == (ROOT / "VERSION").read_text(encoding="utf-8").strip() + "-dev"
    assert module.DEFAULT_OUTPUT == ROOT / "Zephyr Mesh.app"


def test_windows_build_reads_the_version_file_and_has_no_fallback_number():
    script = (ROOT / "desktop" / "build_windows.ps1").read_text(encoding="utf-8")
    assert 'Join-Path $repoRoot "VERSION"' in script
    assert '+ "-dev"' in script
    assert re.search(r'"[0-9]+\.[0-9]+\.[0-9]+', script) is None, "no hard-coded version number"


def test_watch_loop_opens_the_bundle_the_builder_writes():
    import importlib.util
    import sys
    sys.path.insert(0, str(ROOT / "macos"))
    try:
        spec = importlib.util.spec_from_file_location("watch_app", ROOT / "macos" / "watch_app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / "macos"))
    assert module.BUNDLE == ROOT / "Zephyr Mesh.app"


def test_docs_do_not_point_at_the_old_bundle_path():
    for doc in ("README.md", "desktop/README.md", "macos/ZephyrMeshApp/README.md"):
        assert "macos/ZephyrMesh.app" not in (ROOT / doc).read_text(encoding="utf-8"), doc
