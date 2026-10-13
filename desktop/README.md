# Cross-platform desktop preview

`windows_preview.py` is the Windows replay window. It is a native Tkinter app
that replays the canonical S7 event log (`runs/s7-swarm/run.json`). It reads a
file; it is not connected to a radio, camera, motor, or drone, and the one
badge in the title row ("Simulation replay") says so.

What the window shows, top to bottom:

- **Title row.** The app name, the run in plain words (for example "50-drone
  ring, seed 17"), the "Simulation replay" badge, and on the right the 3D/Top
  view choice and an **About this replay** button, like the Mac toolbar.
- **Status strip.** Tracking (a shape, a color, and a word, such as a filled
  circle and "Tracking held"), estimate error p95, target link delay, target
  packet loss, and target age. A value the replay does not contain reads
  "Not recorded"; it is never shown as 0.
- **Team list (left).** One row per status, word first ("Late · 3 drones"),
  with "Loss (%)" and "Age (s)" columns (units in the headings; loss always has
  one decimal); Age is a range when the drones differ. The list is as wide as its
  longest label needs, so on some fonts the window's minimum width grows a
  little past the 1180-pixel token (1200 measured on macOS Tk 9). Rows that
  are not OK sort first and open by default; open a row to see its drones,
  each with its own status shape and word ("D09 · Late"). Clicking a group row
  opens or closes it and never changes the selected drone; when the selected
  drone sits in a closed group, that group gets a light tint instead of the
  selection highlight. A dash in a cell
  means the value was not recorded: hovering it says so, and a line under the
  list explains the dash whenever one is showing. The filter narrows every
  group except late and lost ones, which stay listed; while it is active a
  caption says "Showing N of M" (and "No drones match ..." when nothing
  matches), and a clear button sits in the field. A summary line for the whole
  team sits above the list ("50 drones: 47 OK, 3 late").
- **Scene (center).** A Canvas drawing of the swarm: X-frame drone glyphs in
  one neutral color, the selected drone in the accent color, and drones with a
  late or lost target drawn faded with a marker above them in the shared status
  shape and color (triangle for Late, square for Lost), explained by a legend in
  the top-left corner. The target is a ring with a thin stem to the
  floor, over a 1 m grid with a stronger line every 5 m. Drag to orbit, scroll
  to zoom (both eased and clamped), double-click a drone or press F to focus
  it, R to reset the framing, T for the straight-down Top view. Keep-out
  volumes are drawn only when the replay frame contains them; the canonical S7
  run has none. Drones do not show a heading marker when the data says they
  are at rest. Each drone has four translucent rotor discs (Tk has no alpha, so
  the translucency is a blend with the floor color); the rotors do not spin.
  The camera stays within 6x the swarm radius in both views, which is enough to
  frame the whole ring in a wide, short window (the Mac app keeps a 3x limit).
  Projection and camera math live in `scene_projection.py` and are unit tested.
- **Inspector (right, toggle with I).** The selected drone's status, four
  metric tiles in the same order as the status strip (estimate error, target
  link delay, target packet loss, target age), position and motion, neighbors
  (including neighbor link delay and loss), and a collapsed **Details** section
  with the raw vectors, the hardware profile parts, and constraint flags.
  Heading reads "At rest" only when a speed was recorded and is below
  0.01 m/s; a drone with no recorded velocity reads "Not recorded". With no
  drone selected the panel shows one hint line.
- **Timeline and events (bottom).** Four square buttons (restart, step back,
  Play/Pause in the accent color, step forward) with tooltips, a slider with
  tick marks at event frames, the time label, and playback speed (0.1x, 0.25x,
  1x; 0.25x by default). The controls in the title row and the timeline are
  styled to one height (`layout.controlHeight` in `design/horizon_tokens.json`)
  with rounded corners drawn from `RADIUS.small`; that was measured on macOS
  Tk 9 only, not yet on Windows or at 125% or 150% scaling. The event list below
  uses the shared presentation rules, newest at the bottom; events from the
  current frame are in full text color and earlier ones are dimmer. Clicking an
  event (or pressing Enter on it) jumps to the clicked row, even if it is
  already selected; the jump happens on the press, so it still lands on the
  right row when the list scrolls. Behavior described here was checked on macOS
  Tk 9.

Every number comes from `desktop/presentation.py` through `ReplaySession`, and
every color, font size, and label comes from the generated
`desktop/design_tokens.py` (built from `design/*.json`). The theme is ttk's
`clam` restyled for the dark and light palettes; choose **View > Appearance**
(System, Dark, or Light; Dark is the default). The choice is saved in
`~/.zephyr-mesh/settings.json`, and System follows the Windows app theme
setting when it can be read. On Windows the app uses Segoe UI Variable (or
Segoe UI) and scales sizes with the display DPI; when Windows animations are
turned off, the camera moves without easing.

Status so far: this window has only been run on macOS with Tk 9.0. Nothing has
run on real Windows yet, so the Segoe UI fonts, DPI scaling, the dark title bar,
the Windows theme and animation settings, and the native menus are untested.
On macOS Tk, one frame of the 0.25 s replay takes about 42 to 44 ms on average
to render (worst seen about 89 ms), which is borderline for 1x playback (50 ms
per frame); 0.25x, the default, has 200 ms per frame.

Menus: **File** (Open Evidence Report..., Open Scenario Run..., Exit),
**View** (3D View, Top View, Show Keep-out Volume, Show Inspector, Focus
Selected Drone, Reset Camera, Appearance), **Playback** (Play/Pause, Step
Forward, Step Back, Restart, Speed), and **Help** (About This Replay, Check for
Updates). Keyboard shortcuts match `design/ui_text.json`: Space play or pause,
Left and Right step, Home restart, F focus, R reset camera, T 3D or Top, K
keep-out volume, I inspector, Control-L team filter, Control-O evidence report,
Control-Shift-O scenario run, Control-I about (on a Mac, Command works too).
Single-key shortcuts stand down while you type in a text field. A focused
button keeps Space, and the team list keeps Left, Right, and Space for opening
and closing groups; the Events list hands every key to the replay. A file that
cannot be opened gets one plain sentence (missing, not readable, not UTF-8,
not JSON, or the decoder's reason), the same in both viewers. Right-click a drone in the list or the scene for Focus
Camera, Show in Inspector, and Copy Drone ID.

Both desktop surfaces open the same portable evidence envelope. Create one
from a raw trace or hardware profile with:

```bash
PYTHONPATH=. python tools/build_evidence_report.py vision sim/vision_trace_example.json --output /tmp/zephyr-vision-report.json
PYTHONPATH=. python tools/build_investigation_report.py sim/investigation_example.json --output /tmp/zephyr-s6-report.json
PYTHONPATH=. python tools/build_evidence_report.py swarm runs/s7-swarm/run.json --output /tmp/zephyr-s7-report.json
```

Choose **Open Evidence Report...** in the File menu, Control-O on Windows, or
Command-O on macOS. The Windows viewer leads with the declared status and the
report's limitations, then a labeled summary, then a collapsed **Details**
section with the source SHA-256, schemas, and the raw summary JSON. Reports
never enter a vehicle command path.

Generated `zephyr-s7-scenario-run-1` files can also be inspected without
starting the simulator. Choose **Open scenario run...** in the Windows File
menu, pass `--scenario-run /path/to/run.json`, or print its validated summary
with `--headless --scenario-run /path/to/run.json`. The macOS File menu has the
matching **Open synthetic scenario run...** action. The Windows viewer shows a
top-down plot with an altitude rail for the selected frame first, then a
summary, then a collapsed **Details** section with hashes, versions, and the
raw parameters. Both viewers keep the read-only evidence boundary visible.

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
dependency-light smoke check on any platform. On any desktop with Tk,
`python -m desktop.windows_preview --smoke-idle idle.txt` opens the window,
runs the real event loop, and writes `idle renders=0 ...` when the window
stops redrawing once start-up settles (or `busy` with the count if it does not). The macOS app remains the native
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
  --source "Zephyr Mesh.app" \
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
