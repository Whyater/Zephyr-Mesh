# Zephyr Mesh native cockpit

This directory contains the native macOS operator cockpit for Zephyr Mesh. It is a SwiftUI + SceneKit executable that renders a local, offline fleet fixture with a 3D scene, compact fleet table, replay controls, provenance text, selected-vehicle telemetry, cooperative-neighbor diagnostics, and a short mission event stream.

The app is a **local synthetic cockpit preview**. It loads the compact projection of the canonical Python S7 event log from `runs/s7-swarm/run.json`; the Python simulator remains the source of truth for dynamics and evidence. Each frame now carries the S7 step index, target position, active count, neighbor count, separation margin, constraint flags, target estimate, fused target, packet age, and profile ID when those fields are present in the canonical run. The Swift fixture is intentionally labeled unavailable for live radio, camera input, manual authority, and flight performance.

The operator surface keeps the evidence boundary visible while a replay is running:

- The header keeps one plain-language synthetic-replay status and puts secondary actions in a menu.
- The left fleet rail supports filtering and shows per-vehicle battery, link delay, state, role, and loss in a compact table-like list.
- The right inspector exposes selected-vehicle flight state, link quality, and cooperative context first; replay notes and provenance are concise disclosure sections.
- The SceneKit view is an inspectable visual projection with a target marker, keep-out volume, grid, selected-vehicle ring, constrained orbit controls, focus-selected action, and frame HUD. It does not add a control or hardware path.

## Build and run

On macOS 14 or newer with Xcode 27 or Swift 6:

```bash
cd macos/ZephyrMeshApp
swift build
swift run
```

To assemble a Finder-launchable application bundle in the repository's
`macos/` folder, run this from the repository root:

```bash
python3 macos/build_app.py
open macos/ZephyrMesh.app
```

The generated `ZephyrMesh.app` is a local release artifact and is ignored by
git. Re-run the builder after pulling a code or fixture change. It embeds the
same checked-in demo fixture that `swift run` uses, so opening the bundle does
not start a server or require network access.

For a development loop that rebuilds and opens a fresh bundle when SwiftUI or
fixture files change, run `python3 macos/watch_app.py` from the repository
root. Stop the watcher with `Ctrl-C`; the release bundle remains ignored.

Alternatively, open `Package.swift` in Xcode and run the `ZephyrMeshApp` scheme. `swift test` runs the release-selector tests (exact asset name, no cross-platform matches), the version-ordering and download-URL cases shared with the Python updater, the SHA-256 `verify` checks, and the generated install script in temporary folders (good update, bad signature, hostile paths, symlink, two roots, and a failed final move that must restore the old app). The package has no third-party or network dependency. SceneKit is an Apple platform framework selected for the no-SVG, offline desktop surface documented in the private Decision Log.

The **Update** button and app-menu command query the latest GitHub release.
When a newer architecture-matched macOS ZIP exists (exactly
`ZephyrMesh-macos-<arch>-v<version>.zip`, downloaded only from
`https://github.com/Whyater/Zephyr-Mesh/releases/download/`), the app verifies
its SHA-256 digest and archive entries, checks the bundle signature, starts a
detached helper, exits, swaps the bundle, retains the previous bundle for
rollback, and relaunches. If the helper cannot finish, it moves the previous
bundle back if needed, writes the reason to
`~/Library/Logs/ZephyrMesh/last-update.txt`, and reopens the old app, which
shows that message once. `swift run` reports that it has no installable app
bundle; a Finder-launched `ZephyrMesh.app` is required.

The digest check is an integrity check only, not publisher authentication: it
shows the file matches what GitHub reported for that release asset, not who
published it. The bundle signature check accepts an ad hoc signature, which any
build can carry. The app is not notarized. The install script is tested in
temporary folders, but the whole path has not yet been run end to end against a
published release by a person.

## Evidence boundary

The app currently proves that the native surface builds and can render a deterministic projection with operator-facing diagnostics. It does not prove frame time on a target Mac, physics fidelity, collision safety, radio behavior, or flight performance. A parity test checks the projection's schema, frames, identities, positions, profile IDs, authority labels, and source contract against the checked-in Python artifact. The UI intentionally presents replay-derived diagnostics as modeled or synthetic rather than measured. Usability, accessibility, contrast, and reduced-motion behavior still need target-device measurement.
