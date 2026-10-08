# Zephyr Mesh public roadmap

This roadmap is written for contributors and reviewers. It separates shipped code from local work and future proposals.

## Local now

- Deterministic S0-S6 Python fixtures for physics, communication, sensing, tracking, and bounded investigation.
- Stage 7 foundation: a seeded multi-agent scenario contract with stable identities, explicit obstacle and keep-out events, cooperative formation commands, and replayable telemetry. See the Python module and tests for the exact scope.
- Actuator foundation: first-order motor lag, rotor saturation, body torque mixing, power draw, and battery sag with independent limiting-case tests.
- Adapter foundation: typed `DroneAdapter`, deterministic `SimAdapter`, explicit command authority arbitration, and latched link/battery failsafe states.
- Native macOS cockpit preview in `macos/ZephyrMeshApp` using SwiftUI and SceneKit. It is synthetic and read-only, and it already consumes a compact projection of the canonical Python S7 event log.
- Windows-first Tkinter preview in `desktop/windows_preview.py`, with a PyInstaller build script and a detached updater helper for digest-verified GitHub release ZIPs. Linux packaging remains later work.
- Shared release updater contract with digest verification, safe extraction, rollback backup, and post-exit replacement for the macOS bundle and Windows directory build. The code is local and tested with offline fixtures; no published release has exercised install and relaunch yet.

## Next evidence gates

1. Run S7 analytic and limiting-case checks, including zero-command stability, one-agent dropout, and obstacle clearance.
2. Expand the canonical event-log projection and preserve parity as the native cockpit gains more views.
3. Add custom motor and propeller profile schemas with SI-unit validation, then calibrate parameters from bench measurements.
4. Measure target-device frame time, memory, accessibility behavior, and offline loading before making UX or performance claims.
5. Feed measured ESP-NOW and camera traces into the existing link and sensor contracts.
6. Publish architecture-matched, digest-published macOS and Windows release ZIPs so the in-app update actions can be exercised against a real release. macOS code signing and Windows publisher signing are release gates; neither is claimed for the local build.

## Later

- SimAdapter and hardware adapters behind one state contract.
- Rotor/motor lag, thrust saturation, battery sag, downwash, weather, and venue airflow models with independent validation.
- Mission authoring, replay export, and a supervised non-contact indoor tag demo.

Payload, contact interception, targeting people, and covert surveillance remain outside the project.
