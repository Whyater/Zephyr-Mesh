from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.build_release_manifest import build_manifest
from tools.package_release import create_archive, package, platform_name, sha256_file


def _app(root: Path, name: str = "ZephyrMeshWindows") -> Path:
    app = root / name
    (app / "_internal").mkdir(parents=True)
    (app / f"{name}.exe").write_bytes(b"preview")
    (app / "_internal" / "runs.json").write_text('{"schema":"fixture"}\n', encoding="utf-8")
    return app


def test_package_writes_one_root_archive_and_checksum(tmp_path: Path):
    app = _app(tmp_path)
    first = package(app, tmp_path / "release", platform="windows-x86_64", version="v0.1.0")
    second = package(app, tmp_path / "release-2", platform="windows-x86_64", version="0.1.0")

    assert first["filename"] == "ZephyrMesh-windows-x86_64-v0.1.0.zip"
    assert first["sha256"] == second["sha256"]
    assert Path(first["checksum_path"]).read_text(encoding="utf-8").startswith(f"{first['sha256']}  ")
    with ZipFile(first["path"]) as archive:
        roots = {name.split("/", 1)[0] for name in archive.namelist() if name}
        assert roots == {"ZephyrMeshWindows"}
        assert "ZephyrMeshWindows/ZephyrMeshWindows.exe" in archive.namelist()


def test_package_rejects_symlinked_source_entries(tmp_path: Path):
    app = _app(tmp_path)
    target = tmp_path / "target.txt"
    target.write_text("outside", encoding="utf-8")
    link = app / "link.txt"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink"):
        create_archive(app, tmp_path / "release.zip")


def test_manifest_recomputes_asset_digest_and_release_url(tmp_path: Path):
    app = _app(tmp_path, "ZephyrMesh.app")
    archive = package(app, tmp_path / "release", platform="macos-arm64", version="0.1.0")
    manifest = build_manifest([Path(archive["path"])], version="v0.1.0", repository="Whyater/Zephyr-Mesh")
    asset = manifest["assets"][0]
    assert manifest["schema"] == "zephyr-release-manifest-1"
    assert asset["sha256"] == sha256_file(Path(archive["path"]))
    assert asset["url"].endswith("/v0.1.0/ZephyrMesh-macos-arm64-v0.1.0.zip")


def test_manifest_rejects_duplicate_platforms(tmp_path: Path):
    app = _app(tmp_path)
    one = package(app, tmp_path / "one", platform="windows-x86_64", version="0.1.0")
    two = package(app, tmp_path / "two", platform="windows-x86_64", version="0.1.0")
    with pytest.raises(ValueError, match="duplicate"):
        build_manifest([Path(one["path"]), Path(two["path"])], version="0.1.0", repository="Whyater/Zephyr-Mesh")


def test_manifest_rejects_macos_asset_version_mismatch(tmp_path: Path):
    app = _app(tmp_path, "ZephyrMesh.app")
    archive = package(app, tmp_path / "release", platform="macos-arm64", version="0.2.0")
    with pytest.raises(ValueError, match="version"):
        build_manifest([Path(archive["path"])], version="0.1.0", repository="Whyater/Zephyr-Mesh")


def test_manifest_rejects_windows_asset_version_mismatch(tmp_path: Path):
    app = _app(tmp_path)
    archive = package(app, tmp_path / "release", platform="windows-x86_64", version="0.2.0")
    with pytest.raises(ValueError, match="version does not match"):
        build_manifest([Path(archive["path"])], version="0.1.0", repository="Whyater/Zephyr-Mesh")


def test_platform_name_maps_common_runner_architectures():
    assert platform_name("macos", "arm64") == "macos-arm64"
    assert platform_name("windows", "AMD64") == "windows-x86_64"
