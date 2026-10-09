"""Validate a PyInstaller Windows preview bundle before publishing it."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desktop.replay import ReplayModel


def validate(bundle_dir: Path, *, gui_smoke: bool = False) -> dict[str, str]:
    bundle_dir = Path(bundle_dir).resolve()
    executable = bundle_dir / "ZephyrMeshWindows.exe"
    updater = bundle_dir.parent / "ZephyrMeshUpdater.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"missing preview executable: {executable}")
    if not updater.is_file():
        raise FileNotFoundError(f"missing sibling updater: {updater}")
    resource_roots = (bundle_dir / "_internal", bundle_dir)
    run = next((root / "runs" / "s7-swarm" / "run.json" for root in resource_roots if (root / "runs" / "s7-swarm" / "run.json").is_file()), None)
    if run is None:
        raise FileNotFoundError("bundled canonical S7 run.json is missing")
    result = {"bundle": str(bundle_dir), "preview": str(executable), "updater": str(updater), "run": str(run)}
    # A PyInstaller windowed executable has no reliable stdout contract. Parse
    # the bundled canonical replay directly so this validator remains useful
    # on CI and on Windows without launching a GUI process.
    model = ReplayModel.from_path(run)
    if model.frame_count != 6 or len(model.frame.agents) != 50:
        raise RuntimeError(f"unexpected bundled replay: {model.frame_count} frames, {len(model.frame.agents)} agents")
    result["replay"] = "parsed"
    if gui_smoke:
        # Exercise the actual frozen executable. A windowed PyInstaller build
        # has no stdout contract, so success means it constructs, stays alive
        # long enough to initialize Tk and the bundled replay, then terminates
        # cleanly when the validator closes it.
        marker = Path(tempfile.gettempdir()) / f"zephyr-mesh-smoke-{os.getpid()}.ready"
        health = Path(tempfile.gettempdir()) / f"zephyr-mesh-smoke-{os.getpid()}.healthy"
        marker.unlink(missing_ok=True)
        health.unlink(missing_ok=True)
        process = subprocess.Popen([str(executable), "--smoke-ready", str(marker), "--smoke-step", "--smoke-health", str(health)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not marker.is_file():
                time.sleep(0.05)
            if not marker.is_file():
                process.terminate()
                _, stderr = process.communicate(timeout=5.0)
                details = (stderr or "").strip()[-4000:]
                raise RuntimeError(f"frozen GUI did not publish its readiness marker (exit={process.returncode}); stderr={details}")
            marker_text = marker.read_text(encoding="utf-8").strip()
            if marker_text != "ready frame=2/6":
                raise RuntimeError(f"unexpected frozen GUI readiness marker: {marker_text}")
            if process.poll() is not None:
                raise RuntimeError("frozen GUI exited after publishing readiness")
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not health.is_file():
                time.sleep(0.05)
            if not health.is_file() or health.read_text(encoding="utf-8").strip() != "healthy frame=2/6":
                raise RuntimeError("frozen GUI did not remain healthy after readiness")
            if process.poll() is not None:
                raise RuntimeError("frozen GUI exited after health handshake")
        finally:
            process.terminate()
            process.wait(timeout=5.0)
            marker.unlink(missing_ok=True)
            health.unlink(missing_ok=True)
        result["gui"] = "frozen executable initialized and stepped"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="dist/ZephyrMeshWindows directory")
    parser.add_argument("--gui-smoke", action="store_true", help="on a Windows desktop, launch the frozen GUI and verify initialization")
    args = parser.parse_args()
    print(json.dumps(validate(args.bundle, gui_smoke=args.gui_smoke), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
