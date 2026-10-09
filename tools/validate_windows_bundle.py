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
from desktop.evidence import load_evidence_report


def validate(bundle_dir: Path, *, gui_smoke: bool = False) -> dict[str, str]:
    bundle_dir = Path(bundle_dir).resolve()
    executable = bundle_dir / "ZephyrMeshWindows.exe"
    updater = bundle_dir.parent / "ZephyrMeshUpdater.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"missing preview executable: {executable}")
    if not updater.is_file():
        raise FileNotFoundError(f"missing sibling updater: {updater}")
    readme = bundle_dir / "README-Windows.txt"
    if not readme.is_file():
        raise FileNotFoundError(f"missing Windows extraction guidance: {readme}")
    readme_text = readme.read_text(encoding="utf-8")
    for required_text in ("Extract the complete ZIP", "startup.log"):
        if required_text not in readme_text:
            raise RuntimeError(f"Windows extraction guidance is missing {required_text!r}")
    resource_roots = (bundle_dir / "_internal", bundle_dir)
    run = next((root / "runs" / "s7-swarm" / "run.json" for root in resource_roots if (root / "runs" / "s7-swarm" / "run.json").is_file()), None)
    if run is None:
        raise FileNotFoundError("bundled canonical S7 run.json is missing")
    result = {"bundle": str(bundle_dir), "preview": str(executable), "updater": str(updater), "readme": str(readme), "run": str(run)}
    evidence_paths = []
    for filename, kind, status in (("evidence_report_example.json", "espnow", "fixture"), ("investigation_report_example.json", "investigation", "synthetic"), ("s7_report_example.json", "swarm", "synthetic")):
        evidence = next((root / "desktop" / filename for root in resource_roots if (root / "desktop" / filename).is_file()), None)
        if evidence is None:
            raise FileNotFoundError(f"bundled {filename} is missing")
        evidence_report = load_evidence_report(evidence)
        if evidence_report["kind"] != kind or evidence_report["status"] != status:
            raise RuntimeError(f"bundled {filename} has an unexpected kind or status")
        evidence_paths.append(evidence)
    result["evidence"] = ",".join(str(path) for path in evidence_paths)
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
        command = [str(executable), "--smoke-ready", str(marker), "--smoke-step", "--smoke-health", str(health)]
        for evidence_path in evidence_paths:
            command.extend(("--smoke-evidence", str(evidence_path)))
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
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
            if marker_text != "ready frame=2/6 evidence_kind=espnow,investigation,swarm":
                raise RuntimeError(f"unexpected frozen GUI readiness marker: {marker_text}")
            if process.poll() is not None:
                raise RuntimeError("frozen GUI exited after publishing readiness")
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not health.is_file():
                time.sleep(0.05)
            if not health.is_file() or health.read_text(encoding="utf-8").strip() != "healthy frame=2/6 evidence_kind=espnow,investigation,swarm":
                process.terminate()
                _, stderr = process.communicate(timeout=5.0)
                details = (stderr or "").strip()[-4000:]
                raise RuntimeError(f"frozen GUI did not remain healthy after readiness (exit={process.returncode}); stderr={details}")
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
