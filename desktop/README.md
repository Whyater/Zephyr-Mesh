# Cross-platform desktop preview

`windows_preview.py` is the Windows-first portability surface. It opens a native
Tkinter window, projects the canonical S7 event log on a Canvas, and exposes
play, step, reset, provenance, and a GitHub release check. It shares
`desktop/replay.py` with tests and uses no SVG or network data for the replay.

On Windows, build a self-contained executable from the repository root with:

```powershell
powershell -ExecutionPolicy Bypass -File desktop/build_windows.ps1
```

The resulting `dist/ZephyrMeshWindows/ZephyrMeshWindows.exe` is a local
synthetic replay. Keep the sibling `dist/ZephyrMeshUpdater.exe` beside that
directory so the helper can replace the directory after the preview exits.
Before publishing a Windows build, run
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
`ZephyrMeshUpdater.exe` when a newer verified release is available. No update
is attempted from an unverified branch or a missing digest.
