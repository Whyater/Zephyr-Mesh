# Zephyr Mesh public roadmap

This roadmap is written for contributors and reviewers. It separates shipped code from local work and future proposals.

## Local now

- Deterministic S0-S6 Python fixtures for physics, communication, sensing, tracking, and bounded investigation.
- Stage 7 foundation: a seeded multi-agent scenario contract with stable identities, explicit obstacle and keep-out events, cooperative formation commands, and replayable telemetry. See the Python module and tests for the exact scope.
- Actuator foundation: first-order motor lag, rotor saturation, body torque mixing, power draw, and battery sag with independent limiting-case tests.
- Adapter foundation: typed `DroneAdapter`, deterministic `SimAdapter`, explicit command authority arbitration, and latched link/battery failsafe states.
- Native macOS cockpit preview in `macos/ZephyrMeshApp` using SwiftUI and SceneKit. It is synthetic and read-only, and it already consumes a compact projection of the canonical Python S7 event log.
- Windows-first Tkinter preview in `desktop/windows_preview.py`, with a depth-aware quadcopter projection, fleet filtering, focus/reset controls, keep-out volume, cooperative diagnostics, parts profiles, and the same replay controls as the macOS cockpit. It uses the same Python replay model and stays synthetic and read-only.
- Cross-platform evidence inspector: `tools/build_evidence_report.py` creates a strict, read-only envelope for ESP-NOW, vision, hardware-profile, and bench artifacts. Both desktop apps open the envelope with matching schema, status, source-hash, size, depth, duplicate-key, and limitation checks. The Windows frozen smoke path opens the bundled fixture.
- S6 investigation reports: the existing latency/loss/noise sweep is now serialized as `zephyr-s6-sweep-1` and opens in both inspectors as an `investigation` report. It keeps the tested grid, independent hand-check, criterion note, and synthetic boundary together.
- S7 replay analysis: `tools/analyze_s7_run.py` produces descriptive route, abstract-age, estimator-coverage, reacquisition, constraint, and observed-separation metrics. The same result opens as a `swarm` evidence report in both desktop inspectors, with synthetic and censored boundaries retained.
- Portable S7 scenario runs: tools/run_scenario.py validates bounded JSON configs, produces deterministic parameter and payload digests, and both desktop surfaces inspect the same read-only frames with matching contract checks. The inspectors show runtime provenance and the evidence boundary, while all readers reject reordered frames or timestamps outside the documented 1e-9 second contract tolerance. Linear moving keep-out spheres are captured at each frame time with a shared velocity bound; arbitrary trajectories and obstacle rendering remain future work.
- Shared release updater contract with digest verification, safe extraction, rollback backup, runtime version discovery, and post-exit replacement for the macOS bundle and Windows directory build. Release `v0.2.2` publishes Apple Silicon, Intel, and Windows x86_64 assets with SHA-256 sidecars and a manifest. The workflow runs a frozen Windows GUI smoke check; physical install and relaunch remain open.

## Next evidence gates

1. Run S7 analytic and limiting-case checks, including zero-command stability, one-agent dropout, and obstacle clearance.
2. Expand the canonical event-log projection and preserve parity as the native cockpit gains more views.
3. Use the pushed custom motor and propeller profile and bench-trace boundaries for measured records, then calibrate parameters from repeatable bench measurements.
4. Measure target-device frame time, memory, accessibility behavior, and offline loading before making UX or performance claims.
5. Capture a first documented ESP-NOW run, validate it against `sim/trace_schema.json`, replay it through `sim.link.replay_trace`, and inspect it with `tools/analyze_trace.py` before fitting any link parameters.
6. Capture a documented camera/tracker run, validate it against `sim/vision_trace_schema.json`, and inspect it with `tools/analyze_vision_trace.py` before fitting noise, latency, or recovery parameters.
7. Exercise the v0.2.2 downloads on physical macOS and Windows hosts, including updater staging and relaunch. macOS code signing/notarization and Windows publisher signing remain release gates.
8. Replace the checked-in fixture report with public measured reports only after issues [#1](https://github.com/Whyater/Zephyr-Mesh/issues/1), [#2](https://github.com/Whyater/Zephyr-Mesh/issues/2), and [#3](https://github.com/Whyater/Zephyr-Mesh/issues/3) produce documented artifacts and review them through the same inspector.

## Later

- SimAdapter and hardware adapters behind one state contract.
- Rotor/motor lag, thrust saturation, battery sag, downwash, weather, and venue airflow models with independent validation.
- Mission authoring, replay export, and a supervised non-contact indoor tag demo.

Payload, contact interception, targeting people, and covert surveillance remain outside the project.
