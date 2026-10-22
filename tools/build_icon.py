#!/usr/bin/env python3
"""Rebuild the Zephyr Mesh app icon files in tools/design/icon/.

Runs tools/make_icon.swift (CoreGraphics drawing, colors from
tools/design/horizon_tokens.json), then macOS iconutil for the .icns and
tools/pack_ico.py for the Windows .ico. Needs macOS with Swift and iconutil.

Writes exactly:
  tools/design/icon/ZephyrMesh.icns   macOS app icon, 16 to 1024 px
  tools/design/icon/ZephyrMesh.ico    Windows icon, 16, 24, 32, 48, 64, 128, 256 px
  tools/design/icon/ZephyrMesh-256.png
  tools/design/icon/ZephyrMesh-1024.png
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.pack_ico import pack_ico  # noqa: E402

ICON_DIR = ROOT / "tools" / "design" / "icon"
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(command)} failed:\n{result.stdout}{result.stderr}")


def main() -> int:
    for tool in ("swift", "iconutil"):
        if shutil.which(tool) is None:
            print(f"build_icon: {tool} not found; the icon is built on macOS only", file=sys.stderr)
            return 1
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zephyr-icon-") as scratch:
        work = Path(scratch)
        run(["swift", str(ROOT / "tools" / "make_icon.swift"), str(work)])
        iconset = work / "ZephyrMesh.iconset"
        run(["iconutil", "-c", "icns", str(iconset), "-o", str(ICON_DIR / "ZephyrMesh.icns")])
        frames = [(work / "windows" / f"ZephyrMesh-{size}.png").read_bytes() for size in ICO_SIZES]
        (ICON_DIR / "ZephyrMesh.ico").write_bytes(pack_ico(frames))
        shutil.copyfile(iconset / "icon_256x256.png", ICON_DIR / "ZephyrMesh-256.png")
        shutil.copyfile(iconset / "icon_512x512@2x.png", ICON_DIR / "ZephyrMesh-1024.png")
    for name in ("ZephyrMesh.icns", "ZephyrMesh.ico", "ZephyrMesh-256.png", "ZephyrMesh-1024.png"):
        path = ICON_DIR / name
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
