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


WINDOWS_UI_MODULES = (
    "desktop.scene_projection", "desktop.ui_settings", "desktop.replay_session", "desktop.presentation",
    "desktop.ui_format", "desktop.ui_icons", "desktop.ui_theme", "desktop.ui_widgets", "desktop.scene_view",
    "desktop.ui_viewers", "desktop.results_data", "desktop.ui_results_chart", "desktop.ui_results_atlas", "desktop.ui_results",
)


def expected_smoke_marker(word: str, frame_count: int, evidence_kinds: list[str]) -> str:
    """The readiness line the frozen GUI prints after stepping one frame.

    Built from the bundled replay and evidence files rather than typed in, so
    regenerating the S7 run or adding an evidence kind does not break the gate.
    One step moves from frame 1 to frame 2, or stays on frame 1 for a one-frame run.
    """
    stepped_frame = min(2, frame_count)
    return f"{word} frame={stepped_frame}/{frame_count} evidence_kind={','.join(sorted(set(evidence_kinds)))}"


def validate(bundle_dir: Path, *, gui_smoke: bool = False) -> dict[str, str]:
    bundle_dir = Path(bundle_dir).resolve()
    executable = bundle_dir / "ZephyrMeshWindows.exe"
    updater = bundle_dir.parent / "ZephyrMeshUpdater.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"missing preview executable: {executable}")
    if not updater.is_file():
        raise FileNotFoundError(f"missing sibling updater: {updater}")
    version_file = bundle_dir / "VERSION.txt"
    if not version_file.is_file() or not version_file.read_text(encoding="utf-8").strip():
        # Without it the app reports a fallback version and the updater misjudges what is installed.
        raise FileNotFoundError(f"missing embedded version: {version_file}")
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
    result = {"bundle": str(bundle_dir), "preview": str(executable), "updater": str(updater), "readme": str(readme), "run": str(run), "ui_modules": ",".join(WINDOWS_UI_MODULES)}
    evidence_paths = []
    evidence_kinds = []
    for filename, kind, status in (("evidence_report_example.json", "espnow", "fixture"), ("investigation_report_example.json", "investigation", "synthetic"), ("s7_report_example.json", "swarm", "synthetic")):
        evidence = next((root / "desktop" / filename for root in resource_roots if (root / "desktop" / filename).is_file()), None)
        if evidence is None:
            raise FileNotFoundError(f"bundled {filename} is missing")
        evidence_report = load_evidence_report(evidence)
        if evidence_report["kind"] != kind or evidence_report["status"] != status:
            raise RuntimeError(f"bundled {filename} has an unexpected kind or status")
        evidence_paths.append(evidence)
        evidence_kinds.append(evidence_report["kind"])
    result["evidence"] = ",".join(str(path) for path in evidence_paths)
    # A PyInstaller windowed executable has no reliable stdout contract. Parse
    # the bundled canonical replay directly so this validator remains useful
    # on CI and on Windows without launching a GUI process.
    model = ReplayModel.from_path(run)
    # Compare the parsed model with the raw file, not with typed-in numbers.
    raw_steps = json.loads(run.read_text(encoding="utf-8"))["steps"]
    expected_frames = len(raw_steps)
    expected_agents = len(raw_steps[0]["agents"])
    if model.frame_count != expected_frames or len(model.frame.agents) != expected_agents:
        raise RuntimeError(f"unexpected bundled replay: {model.frame_count} frames, {len(model.frame.agents)} agents (file has {expected_frames} frames, {expected_agents} agents)")
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
            if marker_text != expected_smoke_marker("ready", model.frame_count, evidence_kinds):
                raise RuntimeError(f"unexpected frozen GUI readiness marker: {marker_text}")
            if process.poll() is not None:
                raise RuntimeError("frozen GUI exited after publishing readiness")
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not health.is_file():
                time.sleep(0.05)
            if not health.is_file() or health.read_text(encoding="utf-8").strip() != expected_smoke_marker("healthy", model.frame_count, evidence_kinds):
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
        # The update helper refuses any build whose --self-test fails, so a
        # release whose frozen exe cannot pass it could never be installed.
        # Only the exit code counts: a windowed build has no reliable stdout.
        try:
            self_test = subprocess.run([str(executable), "--self-test"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                       text=True, timeout=30.0, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("frozen --self-test did not exit within 30 s") from exc
        if self_test.returncode != 0:
            raise RuntimeError(f"frozen --self-test failed (exit={self_test.returncode}); stderr={(self_test.stderr or '').strip()[-2000:]}")
        result["self_test"] = "passed"
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
