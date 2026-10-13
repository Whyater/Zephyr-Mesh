# Zephyr Mesh native cockpit

This directory contains the macOS replay viewer for Zephyr Mesh, a SwiftUI and SceneKit app.
It replays the bundled fixture built from the canonical Python S7 run (`runs/s7-swarm/run.json`)
by `tools/build_native_fixture.py`. It is a recorded simulation, not a live connection: there
is no radio, camera, motor, or command path. Values the run does not contain read "Not
recorded". Where the numbers come from:

- Every metric (status, tracking headline, estimate error, link delay, packet loss, target age,
  altitude, speed, heading, neighbors, groups, and events) comes from the presentation records
  that `desktop/presentation.py` computes and the fixture embeds. The Swift code only formats
  them.
- The raw position, velocity and fused-target vectors and the hardware profile values under
  Details, and the scene geometry (drone and target positions, keep-out spheres, and each drone's
  arm length and propeller size), come from fixture fields that `tools/build_native_fixture.py`
  projects from `run.json` and its `manifest.json`. They are copied, not recomputed.
- The team summary line ("50 drones: 44 OK, 6 late") comes from the presentation too. A few
  small display steps happen in Swift: the propeller diameter is shown in millimetres, the filter
  caption counts the listed rows, and a long drone list in a narrow event row keeps fewer names
  and counts the rest in "+N more".

## What the window shows

- Title bar: the app name, the run named in plain words (for example "50-drone ring, seed 17"),
  and one "Simulation replay" badge. The toolbar holds view controls only, in one evenly spaced
  group: the 3D or Top choice, About this replay, and the inspector toggle. Playback lives in one
  place, the timeline under the scene.
- Status strip at the top of the detail area: the tracking headline (shape, color and words,
  the largest text on screen), then estimate error p95, target link delay, target packet loss,
  and target age, in the same order as the inspector tiles. The layout depends on the column
  width only: one row when the column is at least 960 points wide, otherwise the headline sits
  above the four numbers. It never changes height while the replay plays.
- Team list (sidebar): a one-line team summary, a filter field, and one row per status group
  with the group's age range, groups that need attention first. The Age (s) and Loss (%) column
  headers carry the units. Groups expand to their drones. While filtering, a caption reads
  "Showing N of 50", and adds "Late and lost drones stay listed." when those are shown without
  matching; when nothing matches, the list says so and offers Clear filter. The selected drone
  stays listed, marked with a pin, even when the filter would hide it. Right-click a row to focus
  the camera on it, show it in the inspector, or copy its ID.
- 3D view: a procedural quadcopter at true scale from the hardware profile (0.10 m arms, 78 mm
  propellers), the target ring with a stem to the floor, the target keep-out sphere the simulator
  enforces (0.75 m radius from the scenario constraints), and a floor grid with 1 m squares and
  heavier lines every 5 m that fades out before its edge. Late drones carry an amber triangle and
  lost drones a red square above them; these markers keep a fixed size on screen, so status stays
  visible when the whole swarm is in view and in the top view. Drones keep full opacity. The
  selection ring circles the selected drone at its altitude, also at a fixed size on screen, and
  steps aside once the camera is close enough that the drone itself is larger. A small legend in
  the top-left corner names the colors and the grid. Drag to orbit, scroll or pinch to zoom; both
  ease out and stay within limits (pitch 10 to 85 degrees, distance up to 3 times the swarm
  radius, no roll or panning). Click a drone to select it, double-click or press F to focus it,
  press R to reset, and T for a straight-down top view. Hovering a drone shows its name and
  status.
- Inspector (right): the selected drone's status, its four key numbers, position and motion,
  neighbors, and a collapsed Details section with raw vectors, constraint flags, and hardware
  profile parts.
- Timeline and events (bottom): restart, step back, play or pause (the larger button), step
  forward, a scrubber with marks at frames that have events, the replay time, and a speed choice
  (0.1x, 0.25x default, 1x). The event list grows as the replay advances (six rows, or four when
  the window is short); click an event to jump to it. A long list of drone names is shortened at
  a name boundary ("D06, D13 +7 more") when the row is narrow; the tooltip has the full text.

Menus follow the usual places: Check for Updates in the app menu; Open Evidence Report and Open
Scenario Run in File; view and camera commands in View; a Playback menu; and About This Replay
in Help. Settings has an Appearance choice (System, Dark, Light; Dark by default). Keyboard
shortcuts come from `design/ui_text.json`. Single-key shortcuts are disabled while the team
filter is being edited, so typing there never moves the replay.

Evidence reports and scenario runs open in their own windows after the same strict validation
as before (schema, size, key set, and hash checks). Files are read and checked off the main
thread, so a large file does not freeze the app; opening a file that is already open brings its
window forward, and closing a window releases the file. Each window shows its verdict or plot
first, then a plain summary with labels in one column and values beside them, then a collapsed
Details section with hashes, provenance, and the raw data as a collapsed tree (the raw summary
for an evidence report; the raw parameters and provenance for a scenario run). The scenario run
window states the run status above its plot, centres the plot on the data, draws a grid with a
scale bar and an altitude rail with its end values, and puts the summary in a second column
when the window is wide. Update results and file errors appear as alerts even when the main
window is closed.

Colors, type sizes, spacing and layout sizes come only from the generated
`Sources/ZephyrMeshApp/DesignSystem.swift`; `tests/test_macos_ui_hygiene.py` checks this.

## Build and run

On macOS 14 or newer with Xcode 27 or Swift 6:

```bash
cd macos/ZephyrMeshApp
swift build
swift run
```

To assemble a Finder-launchable application bundle in the repository root, run this from the repository root:

```bash
python3 macos/build_app.py
open "Zephyr Mesh.app"
```

The generated `Zephyr Mesh.app` is a local development artifact and is ignored by
git. Re-run the builder after pulling a code or fixture change. It embeds the
same checked-in demo fixture that `swift run` uses, so opening the bundle does
not start a server or require network access.

Local bundles use the repository `VERSION` with a `-dev` suffix. Release builds
pass an explicit version. Add `--install` to copy the finished bundle into
`~/Applications`.

For a development loop that rebuilds and opens a fresh bundle when SwiftUI or
fixture files change, run `python3 macos/watch_app.py` from the repository
root. Stop the watcher with `Ctrl-C`; the release bundle remains ignored.

Alternatively, open `Package.swift` in Xcode and run the `ZephyrMeshApp` scheme. `swift test` runs the strict fixture decoder tests (unknown or missing keys fail with a named message), the presentation formatting, team grouping and filter rules, playback and shortcut gating, the orbit camera limits and smoothing, a rendered hit test for hover and click, the release-selector tests (exact asset name, no cross-platform matches), the version-ordering and download-URL cases shared with the Python updater, the SHA-256 `verify` checks, and the generated install script in temporary folders (good update, bad signature, hostile paths, symlink, two roots, and a failed final move that must restore the old app). The package has no third-party or network dependency. SceneKit is an Apple platform framework selected for the no-SVG, offline desktop surface documented in the private Decision Log.

The **Check for Updates...** command in the app menu queries the latest GitHub release.
When a newer architecture-matched macOS ZIP exists (exactly
`ZephyrMesh-macos-<arch>-v<version>.zip`, downloaded only from
`https://github.com/Whyater/Zephyr-Mesh/releases/download/`), the app verifies
its SHA-256 digest while streaming it to disk (at most 512 MB), starts a
detached helper, and exits. The helper checks the digest again, screens the
archive entries, checks the bundle signature, and runs the new build's
`--self-test`; a build that fails any of these is never installed. It then
moves the current bundle to `~/Library/Caches/org.zephyrmesh.cockpit`, moves
the new one into place, opens it, and deletes the backup once the new app has
stayed running for a few seconds. If the helper cannot finish, it moves the previous
bundle back if needed, writes the reason to
`~/Library/Logs/ZephyrMesh/last-update.txt`, and reopens the old app, which
shows that message once. `swift run` reports that it has no installable app
bundle; a Finder-launched `Zephyr Mesh.app` is required.

The digest check is an integrity check only, not publisher authentication: it
shows the file matches what GitHub reported for that release asset, not who
published it. The bundle signature check accepts an ad hoc signature, which any
build can carry. The app is not notarized. The install script is tested in
temporary folders, but the whole path has not yet been run end to end against a
published release by a person.

## Evidence boundary

The app currently proves that the native surface builds and can render a deterministic projection with operator-facing diagnostics. It does not prove frame time on a target Mac, physics fidelity, collision safety, radio behavior, or flight performance. A parity test checks the projection's schema, frames, identities, positions, profile IDs, authority labels, and source contract against the checked-in Python artifact. The UI presents replay-derived values as a simulation replay, never as measured. Usability, accessibility, contrast, and reduced-motion behavior still need checks by people on target devices.
