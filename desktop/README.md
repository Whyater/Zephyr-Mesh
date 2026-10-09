# Cross-platform desktop preview

`windows_preview.py` is the Windows-first portability surface. It opens a native
Tkinter window and projects the canonical S7 event log with depth-aware
quadcopter glyphs, a target marker, and an optional keep-out volume. The
operator surface mirrors the macOS cockpit's replay controls, fleet filtering,
vehicle selection, telemetry disclosures, parts profile, link diagnostics,
provenance, mission-event history, focus/reset view controls, and GitHub release
check. The mission-event history uses the same replay-only vocabulary as the
macOS inspector, so authority and failsafe boundaries stay visible on both
platforms. Because canonical S7 link events are not attached to individual
frame rows, Windows derives the link-envelope line from the replay-wide
observed loss envelope and labels it as modeled replay evidence. It shares
`desktop/replay.py` with tests and uses no SVG or network data for the replay.

Keyboard access is built in: Space plays or pauses, Right Arrow advances one
frame, `F` focuses the selected vehicle, `R` resets the view, `O` toggles the
keep-out volume, and Control-L moves focus to the fleet filter. The Canvas
projection is intentionally lightweight and deterministic, while the macOS
surface provides the native SceneKit 3D renderer.

Both desktop surfaces open the same portable evidence envelope. Create one
from a raw trace or hardware profile with:

```bash
PYTHONPATH=. python tools/build_evidence_report.py vision sim/vision_trace_example.json --output /tmp/zephyr-vision-report.json
PYTHONPATH=. python tools/build_investigation_report.py sim/investigation_example.json --output /tmp/zephyr-s6-report.json
PYTHONPATH=. python tools/build_evidence_report.py swarm runs/s7-swarm/run.json --output /tmp/zephyr-s7-report.json
```

Choose **Open evidence report…** in the File menu, Control-O on Windows, or
Command-O on macOS. The inspector preserves nested values and explicit
`null` fields, shows the declared source SHA-256 and status, and keeps report
limitations visible. Reports never enter a vehicle command path.

On Windows, build a self-contained executable from the repository root with:

```powershell
powershell -ExecutionPolicy Bypass -File desktop/build_windows.ps1
```

The resulting `dist/ZephyrMeshWindows/ZephyrMeshWindows.exe` is a local
synthetic replay. Keep the sibling `dist/ZephyrMeshUpdater.exe` beside that
directory so the helper can replace the directory after the preview exits. The
build embeds the requested version in `VERSION.txt`, so later releases can be
compared by the updater without changing source code.
The tagged release workflow also runs the Python suite on a Windows runner before packaging. Before publishing a local Windows build, run
`python tools/validate_windows_bundle.py dist/ZephyrMeshWindows` on Windows;
it checks the bundled replay, sibling helper, and the expected six-frame,
50-vehicle contract without relying on stdout from a windowed executable.
On a Windows desktop, add `--gui-smoke` to launch the frozen Tk executable,
step one replay frame through its smoke handshake, verify a readiness marker,
and close it before publishing.
Use `python -m desktop.windows_preview --headless` for a
dependency-light smoke check on any platform. The macOS app remains the native
SwiftUI + SceneKit surface.

Both desktop surfaces use `tools/release_updater.py`. It checks a GitHub
release, selects the platform and architecture asset, requires a SHA-256
digest, stages a safe ZIP extraction, and applies it only after the running
process exits. Packaged Windows builds automatically start the sibling
or embedded `ZephyrMeshUpdater.exe` when a newer verified release is available;
embedded helpers are copied to a temporary path before swapping their own
application directory. No update is attempted from an unverified branch or a
missing digest.

## Release packaging

The repository contains a tag-driven workflow at
`.github/workflows/release.yml`. A tag such as `v0.1.5` builds native macOS
bundles on Apple Silicon (`macos-14`) and Intel (`macos-15-intel`), builds the
Windows preview and helper on `windows-2022`, validates the frozen Windows replay, and publishes
architecture-labeled ZIP assets. Each ZIP contains exactly one app directory,
a `.sha256` sidecar, and a `release-manifest.json` is attached for mirrors and
offline tooling.

The local packaging commands are useful when checking an artifact before a
tagged build:

```bash
python3 macos/build_app.py --version 0.1.5
python3 tools/package_release.py \
  --source macos/ZephyrMesh.app \
  --output release \
  --platform macos-arm64 \
  --version 0.1.5
python3 tools/build_release_manifest.py \
  --version 0.1.5 \
  --output release/release-manifest.json \
  release/ZephyrMesh-macos-arm64-v0.1.5.zip
```

On Windows, pass `-Package` to the build script to produce the Windows ZIP
and checksum in `dist/release`. The build embeds `VERSION.txt` so the update
check compares the installed package version instead of assuming the initial
development version:

```powershell
powershell -ExecutionPolicy Bypass -File desktop/build_windows.ps1 `
  -Version 0.1.5 -Package
```

The local macOS bundle is ad-hoc signed for Finder launch. The release workflow
does not claim notarization or Windows publisher signing; those are release
gates to complete before treating a public build as production-ready.
