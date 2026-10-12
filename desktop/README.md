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

Keyboard access is built in (single-key shortcuts are ignored while you type in the fleet filter): Space plays or pauses (the Play button reads Pause while playing), Right Arrow advances one
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

Generated `zephyr-s7-scenario-run-1` files can also be inspected without
starting the simulator. Choose **Open scenario run...** in the Windows File
menu, pass `--scenario-run /path/to/run.json`, or print its validated summary
with `--headless --scenario-run /path/to/run.json`. The macOS File menu has the
matching **Open synthetic scenario run...** action. Both inspectors show frame
counts and timing while retaining the synthetic, read-only evidence boundary.

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
it checks the bundled replay, sibling helper, and that the replay's frame and
vehicle counts match the bundled `run.json` itself, without relying on stdout from a windowed executable.
On a Windows desktop, add `--gui-smoke` to launch the frozen Tk executable,
step one replay frame through its smoke handshake, verify a readiness marker,
and close it before publishing.
Use `python -m desktop.windows_preview --headless` for a
dependency-light smoke check on any platform. The macOS app remains the native
SwiftUI + SceneKit surface.

Both desktop surfaces use `tools/release_updater.py`. It checks a GitHub
release and selects the asset named exactly
`ZephyrMesh-{platform}-v{version}.zip` (the name `tools/package_release.py`
writes). It downloads only from
`https://github.com/Whyater/Zephyr-Mesh/releases/download/`, requires a SHA-256
digest, stages a safe ZIP extraction, and applies it only after the running
process exits. Version tags compare as numbers (`0.2` equals `v0.2.0`), and a
pre-release such as `v0.3.0-rc1` is older than `v0.3.0`; the macOS updater
follows the same rules and both are tested against one shared case list
(`macos/ZephyrMeshApp/Tests/ZephyrMeshAppTests/Fixtures/updater-cases.json`).
Packaged Windows builds start the sibling or embedded `ZephyrMeshUpdater.exe`
when a newer release is available; embedded helpers are copied to a temporary
path before swapping their own application directory. If the swap fails, the
helper writes `update-result.txt` next to `startup.log`, starts the previous copy
again, and the app shows that message the next time it opens.

What the SHA-256 check does and does not do: it proves the downloaded file
matches the digest GitHub reported for that release asset, so it catches
corruption and tampering in transit. It is an integrity check only, not
publisher authentication. The digest and the file come from the same GitHub
account, so anyone who can publish a release there can publish a matching
pair. The Windows build is also unsigned, so Windows SmartScreen may warn when
you first run it.

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
