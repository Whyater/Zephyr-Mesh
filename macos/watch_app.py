#!/usr/bin/env python3
"""Rebuild and relaunch the local macOS bundle when source files change.

This is a development convenience, not part of the release updater. It keeps
the double-clickable bundle aligned with SwiftUI and fixture edits without
requiring an IDE preview session.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "macos" / "ZephyrMeshApp"
BUNDLE = ROOT / "macos" / "ZephyrMesh.app"


def snapshot() -> dict[Path, int]:
    files = list((PACKAGE / "Sources").rglob("*.swift")) + list((PACKAGE / "Resources").rglob("*")) + [ROOT / "macos" / "build_app.py"]
    return {path: path.stat().st_mtime_ns for path in files if path.is_file()}


def rebuild() -> None:
    subprocess.run([sys.executable, str(ROOT / "macos" / "build_app.py")], cwd=ROOT, check=True)
    subprocess.run(["open", "-n", str(BUNDLE)], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=1.0, help="polling interval in seconds")
    args = parser.parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    rebuild()
    previous = snapshot()
    print("Watching SwiftUI sources and fixture resources. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(args.interval)
            current = snapshot()
            if current != previous:
                rebuild()
                previous = current
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
