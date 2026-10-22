#!/usr/bin/env python3
"""Build a double-clickable ``Zephyr Mesh.app`` bundle on macOS.

The Swift package remains the source of truth. This script only assembles the
release executable and its SwiftPM resource bundle into the conventional
macOS application layout so a local checkout can be opened from Finder.
The generated bundle is intentionally ignored by git because it is a
machine-specific binary artifact.
"""
from __future__ import annotations

import argparse
import plistlib
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tools" / "macos" / "ZephyrMeshApp"
DEFAULT_OUTPUT = ROOT / "Zephyr Mesh.app"
VERSION_FILE = ROOT / "VERSION"


def development_version() -> str:
    """Return the checkout version used for local builds."""
    value = VERSION_FILE.read_text(encoding="utf-8").strip().removeprefix("v")
    if not value or any(character.isspace() for character in value):
        raise ValueError("VERSION must contain one non-empty tag-safe value")
    return f"{value}-dev"


def build_bundle(output: Path, *, version: str | None = None) -> Path:
    """Compile the Swift release product and assemble an app bundle."""
    if version is None:
        version = development_version()
    version = version.removeprefix("v")
    if not version or any(character.isspace() for character in version):
        raise ValueError("version must be a non-empty tag-safe value")
    subprocess.run(["swift", "build", "-c", "release"], cwd=PACKAGE, check=True)
    products = PACKAGE / ".build" / "release"
    executable = products / "ZephyrMeshApp"
    resource_bundle = products / "ZephyrMeshApp_ZephyrMeshApp.bundle"
    if not executable.is_file() or not resource_bundle.is_dir():
        raise FileNotFoundError("SwiftPM release product or resource bundle is missing")

    output = output.resolve()
    if output.exists():
        shutil.rmtree(output)
    macos_dir = output / "Contents" / "MacOS"
    resources_dir = output / "Contents" / "Resources"
    macos_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)
    shutil.copy2(executable, macos_dir / "ZephyrMeshApp")
    shutil.copytree(resource_bundle, resources_dir / resource_bundle.name)
    icon_source = ROOT / "tools" / "design" / "icon" / "ZephyrMesh.icns"
    icon_installed = False
    if icon_source.is_file():
        shutil.copy2(icon_source, resources_dir / icon_source.name)
        icon_installed = True
    else:
        print(f"Warning: macOS app icon is missing: {icon_source}")

    info = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": "Zephyr Mesh",
        "CFBundleExecutable": "ZephyrMeshApp",
        "CFBundleIdentifier": "org.zephyrmesh.cockpit",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": "Zephyr Mesh",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
    }
    if icon_installed:
        info["CFBundleIconFile"] = icon_source.name
    with (output / "Contents" / "Info.plist").open("wb") as handle:
        plistlib.dump(info, handle, sort_keys=False)
    # An ad-hoc signature keeps the local Finder launch path smooth without
    # pretending this development artifact is notarized or publisher-signed.
    codesign = shutil.which("codesign")
    if codesign:
        subprocess.run([codesign, "--force", "--deep", "--sign", "-", str(output)], check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local Zephyr Mesh macOS app bundle")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="app bundle path")
    parser.add_argument("--version", help="bundle version, with or without a leading v")
    parser.add_argument("--install", action="store_true", help="copy the bundle into ~/Applications")
    args = parser.parse_args()
    bundle = build_bundle(args.output, version=args.version)
    if args.install:
        applications = Path.home() / "Applications"
        applications.mkdir(parents=True, exist_ok=True)
        destination = applications / bundle.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(bundle, destination)
        print(f"Installed {destination}")
    print(f"Built {bundle}")
    print(f"Open it with: open {bundle}")


if __name__ == "__main__":
    main()
