#!/usr/bin/env python3
"""Build the macOS app, copy it out of the repo, hide SwiftPM's build folder, and run it.

This is the check that would have caught the v0.2.2 launch crash: SwiftPM's
``Bundle.module`` fell back to the absolute ``.build`` path of the machine that
compiled the app, so a local build "worked" only because that folder existed.
Here the app runs from a temporary copy while ``tools/macos/ZephyrMeshApp/.build`` is
renamed away, and the folder is put back afterwards even if the check fails or
is interrupted with Ctrl-C.

``--launch`` also opens the copy normally (``open -n``), waits, checks that it
is still running and left no new crash report, then quits it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "tools" / "macos" / "ZephyrMeshApp"
APP = ROOT / "Zephyr Mesh.app"
BUILD_DIR = PACKAGE / ".build"
HIDDEN_DIR = PACKAGE / ".build-smoke-hidden"
CRASH_REPORTS = Path.home() / "Library" / "Logs" / "DiagnosticReports"
SELF_TEST_TIMEOUT_S = 30


def restore_leftover_build_dir() -> None:
    """Undo a hide that an earlier run could not undo (for example after kill -9)."""
    if HIDDEN_DIR.exists() and not BUILD_DIR.exists():
        HIDDEN_DIR.rename(BUILD_DIR)
        print(f"Restored {BUILD_DIR} left hidden by an earlier run")


def crash_reports() -> set[str]:
    try:
        return {path.name for path in CRASH_REPORTS.glob("ZephyrMeshApp*.ips")}
    except OSError:
        return set()


def run_self_test(app: Path, cwd: Path) -> bool:
    executable = app / "Contents" / "MacOS" / "ZephyrMeshApp"
    started = time.monotonic()
    try:
        result = subprocess.run([str(executable), "--self-test"], cwd=cwd, capture_output=True, text=True,
                                timeout=SELF_TEST_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        print(f"FAIL: --self-test did not exit within {SELF_TEST_TIMEOUT_S} s (did it open a window?)")
        return False
    output = (result.stdout + result.stderr).strip()
    print(f"--self-test exit={result.returncode} in {time.monotonic() - started:.2f} s: {output}")
    if result.returncode != 0 or "self-test ok" not in result.stdout:
        print("FAIL: the copied app could not load its bundled resources")
        return False
    return True


def launch_check(app: Path, wait_s: float) -> bool:
    executable = str((app / "Contents" / "MacOS" / "ZephyrMeshApp").resolve())
    before = crash_reports()
    subprocess.run(["/usr/bin/open", "-n", str(app)], check=True)
    time.sleep(wait_s)
    listing = subprocess.run(["/bin/ps", "-axo", "pid=,comm="], capture_output=True, text=True, check=False).stdout
    pids = [line.split(None, 1)[0] for line in listing.splitlines() if line.strip().split(None, 1)[-1] == executable]
    new_reports = sorted(crash_reports() - before)
    alive = bool(pids)
    print(f"launch: alive after {wait_s:g} s = {alive} (pids {', '.join(pids) or 'none'}); new crash reports: {new_reports or 'none'}")
    for pid in pids:
        subprocess.run(["/bin/kill", "-TERM", pid], check=False)
    return alive and not new_reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-build", action="store_true", help="reuse the existing Zephyr Mesh.app")
    parser.add_argument("--launch", action="store_true", help="also open the copy normally and check it stays up")
    parser.add_argument("--launch-wait", type=float, default=5.0, help="seconds to wait before checking the launched app")
    args = parser.parse_args(argv)

    restore_leftover_build_dir()
    if not args.skip_build:
        subprocess.run([sys.executable, str(ROOT / "tools" / "macos" / "build_app.py")], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix="zephyr-app-smoke-") as temporary:
        copied_app = Path(temporary) / APP.name
        shutil.copytree(APP, copied_app, symlinks=True)
        print(f"Copied the app to {copied_app}")
        moved = False
        try:
            if BUILD_DIR.exists():
                if HIDDEN_DIR.exists():
                    shutil.rmtree(HIDDEN_DIR)
                BUILD_DIR.rename(HIDDEN_DIR)
                moved = True
                print(f"Hid {BUILD_DIR}")
            ok = run_self_test(copied_app, Path(temporary))
            if ok and args.launch:
                ok = launch_check(copied_app, args.launch_wait)
        finally:
            if moved:
                if BUILD_DIR.exists():
                    shutil.rmtree(BUILD_DIR)
                HIDDEN_DIR.rename(BUILD_DIR)
                print(f"Restored {BUILD_DIR}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
