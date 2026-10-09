#!/usr/bin/env python3
"""Build a double-clickable ``ZephyrMesh.app`` bundle on macOS.

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


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "macos" / "ZephyrMeshApp"
DEFAULT_OUTPUT = ROOT / "macos" / "ZephyrMesh.app"


def build_bundle(output: Path, *, version: str = "0.1.0") -> Path:
    """Compile the Swift release product and assemble an app bundle."""
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
    parser.add_argument("--version", default="0.1.0", help="bundle version, with or without a leading v")
    args = parser.parse_args()
    bundle = build_bundle(args.output, version=args.version)
    print(f"Built {bundle}")
    print(f"Open it with: open {bundle}")


if __name__ == "__main__":
    main()
