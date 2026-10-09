# Zephyr Mesh

## What is Zephyr Mesh

Zephyr Mesh is an offline-first drone tracking and swarm simulation workspace. It lets you test how latency, packet loss, sensing noise, and recovery strategies affect cooperative flight before connecting the model to measured hardware data.

The simulator is intentionally transparent. Python owns the dynamics and replay data; the desktop apps provide operator views over the same checked-in fixtures. Current runs are synthetic and read-only. They do not control a drone or claim flight performance.

## Download

### macOS

The native app requires macOS 14 or newer. Download the architecture that matches your Mac from [Zephyr Mesh v0.2.2](https://github.com/Whyater/Zephyr-Mesh/releases/tag/v0.2.2):

- [Apple Silicon](https://github.com/Whyater/Zephyr-Mesh/releases/download/v0.2.2/ZephyrMesh-macos-arm64-v0.2.2.zip)
- [Intel](https://github.com/Whyater/Zephyr-Mesh/releases/download/v0.2.2/ZephyrMesh-macos-x86_64-v0.2.2.zip)

Unzip the download and open `ZephyrMesh.app`. SHA-256 sidecars are attached to the release. The package is ad hoc signed and may require a one-time confirmation in macOS Privacy & Security. To build the local bundle instead, install the Xcode Command Line Tools and run:

```bash
python3 macos/build_app.py
open macos/ZephyrMesh.app
```

The app does not require a server or network connection to show the bundled replay.

### Windows

Download the [Windows x86_64 package](https://github.com/Whyater/Zephyr-Mesh/releases/download/v0.2.2/ZephyrMesh-windows-x86_64-v0.2.2.zip) from [Zephyr Mesh v0.2.2](https://github.com/Whyater/Zephyr-Mesh/releases/tag/v0.2.2). Extract the complete ZIP before opening `ZephyrMeshWindows\ZephyrMeshWindows.exe`; keep the `_internal` folder beside it. The release also provides a separate .zip.sha256 asset for verification.

To build locally, install Python 3.11 or newer, open PowerShell at the repository root, and run:

```powershell
powershell -ExecutionPolicy Bypass -File desktop/build_windows.ps1
```

The script creates `dist\ZephyrMeshWindows\ZephyrMeshWindows.exe` and `dist\ZephyrMeshUpdater.exe`. Keep the updater beside the application directory. The app's update action only installs a published release asset with a matching platform architecture and GitHub SHA-256 digest.

## Current features

- Six-degree-of-freedom rigid-body simulation with quaternion attitude, gravity, configurable wind, quadratic relative-air drag, and a tested RK4 integrator.
- Cascaded position and attitude control with transparent, inspectable inputs.
- Deterministic communication fixtures for delay, jitter, independent loss, burst loss, serialized contention, packet age, duplicates, and out-of-order delivery.
- Versioned ESP-NOW trace schema and replay bridge that preserve packet identity, measured outcomes, timestamps, RSSI, retries, and clock boundaries without fitting unmeasured radio behavior.
- Trace summary CLI with unresolved-outcome counts, shared-clock delay summaries, observed-record loss bounds, RSSI summaries, and provenance.
- Versioned camera/tracker trace schema and summary CLI that preserve detection, miss, and unresolved outcomes, confidence, image-space boxes, optional frame-labelled positions, clock boundaries, and provenance.
- Seeded sensing and tracking fixtures with independent or burst sensor dropouts, bias, covariance, out-of-order rejection, and recovery metrics; S7 adds a separate deterministic agent dropout schedule.
- Bounded latency, loss, noise, and target-acceleration investigations with reproducible JSON outputs.
- Stage 7 coordination foundation with a seeded multi-agent event log, stable identities, formation slots, cooperative target fusion, dropout events, separation checks, and keep-out constraints.
- Hardware profile and actuator foundation with SI-unit motor and propeller inputs, first-order motor lag, RPM saturation, torque mixing, power draw, and battery sag.
- Strict SI-unit hardware-profile documents with custom motor and propeller part IDs, status/source/calibration provenance, JSON Schema validation, round-trip loading, and a normalization CLI.
- Versioned bench trace schema and summary CLI for RPM, thrust, voltage, current, temperature, test conditions, incomplete readings, and transparent electrical input power summaries without curve fitting.
- Portable evidence reports for the macOS and Windows inspectors, preserving source hashes, declared status, nested values, and limitations.
- S6 investigation reports for latency, loss probability, sensing noise, target acceleration, estimator error, and scenario-specific failure labels. These are deterministic research fixtures, not tolerance or flight-performance results.
- S7 replay analysis for per-route outcomes, abstract packet age, estimator coverage and age, bracketed reacquisition intervals, constraint flags, and observed separation. These metrics describe the checked-in replay and do not establish radio tolerance or flight safety.
- Typed drone adapter, authority arbitration, and replayable link or battery failsafe contracts.
- Native macOS SwiftUI and SceneKit cockpit plus a Windows Tkinter desktop surface built from the same canonical replay, including compact mission-event history and replay-only authority or failsafe context.
- Portable deterministic S7 scenario runs with parameter hashes, payload digests, strict cross-platform validation, bounded linear moving keep-out spheres, and read-only frame inspectors with matching manual controls on both desktop surfaces. Inspectors expose the canonical digests, runtime provenance, base revision semantics, and evidence boundary; frame order and timestamps are validated across readers. The selected-frame view includes a compact synthetic altitude profile beside the top-down geometry, with the long integrity details collapsed to reduce clutter.
- Model-provenance helpers identify the S7 coordinator as a bounded point-mass fixture and keep declared hardware-profile identity separate from actuator state, energy telemetry, and 6-DOF flight simulation.
- Local release updater contract with digest verification, path-safe extraction, rollback staging, and architecture-aware asset selection.

## Run from source

The Python simulator and replay demo run offline from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
TMP_DIR=$(mktemp -d)
PYTHONPATH=. python tools/generate_s7_fixture.py --output "$TMP_DIR/s7-swarm"
PYTHONPATH=. python tools/analyze_trace.py sim/trace_example.json
PYTHONPATH=. python tools/analyze_vision_trace.py sim/vision_trace_example.json
PYTHONPATH=. python tools/validate_hardware_profile.py sim/hardware_profile_example.json
PYTHONPATH=. python tools/analyze_bench_trace.py sim/bench_trace_example.json
PYTHONPATH=. python tools/build_evidence_report.py espnow sim/trace_example.json --output /tmp/zephyr-espnow-report.json
PYTHONPATH=. python tools/build_investigation_report.py sim/investigation_example.json --output /tmp/zephyr-s6-report.json
PYTHONPATH=. python tools/analyze_s7_run.py runs/s7-swarm/run.json
PYTHONPATH=. python tools/run_scenario.py --config sim/scenario_example.json --output /tmp/zephyr-s7-scenario-run.json --summary
python -m demo.demo --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` in a browser on the same computer. The demo is read-only and serves synthetic fixtures. The generation command writes a temporary fixture and leaves the checked-in files unchanged. A scenario config may add `velocity_mps` to a keep-out sphere for a bounded linear moving constraint; the generated frame snapshots retain each synthetic center for inspection. The canonical artifacts are [`runs/s0-baseline`](runs/s0-baseline), [`runs/s2-physics`](runs/s2-physics), and [`runs/s7-swarm`](runs/s7-swarm). The native and Windows surfaces consume the S7 replay through [`desktop/replay.py`](desktop/replay.py) and the checked-in macOS fixture projection. See [`ROADMAP.md`](ROADMAP.md) for evidence gates.

To inspect a trace or profile in either desktop app, create a portable report
with `tools/build_evidence_report.py`, then choose **Open evidence report…**
from the File menu. Reports are read-only. Their declared status and
limitations stay attached to the source summary. A digest identifies the
input file without establishing flight performance.

The published v0.2.2 binaries are the replay baseline. The scenario-run CLI and strict scenario inspectors documented below are in the current source checkout and will be included in a later desktop release after target-host validation.

## Planned features

- Measure ESP-NOW latency, loss, update rate, and contention on real hardware, then feed those traces into the link model.
- Size camera and sensor noise from recorded measurements and validate tracker recovery against those traces.
- Calibrate motor, propeller, battery, and airframe parameters from bench data, including the limits of the current transparent model.
- Add hardware adapters and a ground-station interface behind the existing `DroneAdapter` contract.
- Add supervised mission authoring, replay export, explicit controller assignment, team-command arbitration, and link-loss recovery workflows.
- Measure Windows and macOS target-device parity for frame time, memory, accessibility, reduced motion, and cooperative diagnostics.
- Add publisher signing and notarization for broad macOS distribution.
- Validate the model with a small, non-contact indoor demonstration. Payloads, contact interception, targeting people, and covert surveillance are outside the project.

## Known Bugs/Limitations

- The checked-in event logs and desktop views are synthetic replays. They do not represent a measured radio link, camera, motor, controller, or flight test.
- No measured ESP-NOW or camera trace is included yet. The versioned schemas are ready for documented hardware captures, but no radio tolerance, camera calibration, tracker, or flight-performance result is claimed.
- Motor, propeller, battery, drag, and airflow values are scenario inputs or calibration placeholders. The profile document loader validates units, provenance fields, and schema structure, but it does not calibrate coefficients. Its thrust field is a declared-coefficient estimate at the lower motor/propeller RPM limit and does not apply torque or electrical power limits. Downwash, ground effect, propeller wake interaction, and venue airflow still need measured models.
- The bench trace boundary and electrical input power summary are synthetic until a documented run records instruments, part identity, calibration, units, clock conditions, and repeatable test settings. No fitted motor map is included.
- The release workflow runs the Python suite, builds, and smoke-tests the Windows package, but physical Windows install, updater relaunch, accessibility, and frame-time measurements remain open. Linux packaging is later work.
- Scenario runs are deterministic point-mass coordination fixtures over abstract links. They support reproducible replay and contract testing, but do not establish radio tolerance, tracker performance, safety thresholds, or flight performance.
- Scenario temporal integrity uses an absolute 1e-9 second contract tolerance, and `code_revision` identifies the repository base revision rather than uncommitted working-tree edits. These are integrity and provenance fields, not performance measurements.
- The altitude profile is a bounded selected-frame projection, not a full 3D scene or a live vehicle view. Swarm profile provenance records declared parts only; it does not synthesize motor power, battery depletion, actuator state, or flight results.
- The v0.2.2 macOS package is not notarized. Broad distribution still needs a Developer ID signature and notarization.

## Contributing

Start with the core question: how much radio latency, packet loss, and tracking noise can a cooperative drone tracker tolerate, and what helps it recover? Keep changes reproducible, label synthetic and measured data separately, and run the test suite before opening a pull request.

Use the commands in [Run from source](#run-from-source) for setup and tests. On Windows, use `.venv\Scripts\activate` in place of `source .venv/bin/activate`. See [`ROADMAP.md`](ROADMAP.md) for the current evidence gates.

## License

MIT. See [LICENSE](LICENSE).
