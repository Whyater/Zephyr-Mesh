# Zephyr Mesh public roadmap

This roadmap is written for contributors and reviewers. It separates shipped code from local work and future proposals.

## Local now

- Deterministic S0-S6 Python fixtures for physics, communication, sensing, tracking, and bounded investigation.
- Stage 7 foundation: a seeded multi-agent scenario contract with stable identities, explicit obstacle and keep-out events, cooperative formation commands, and replayable telemetry. See the Python module and tests for the exact scope.
- Native macOS cockpit preview in `macos/ZephyrMeshApp` using SwiftUI and SceneKit. It is synthetic and read-only, and it already consumes a compact projection of the canonical Python S7 event log.

## Next evidence gates

1. Run S7 analytic and limiting-case checks, including zero-command stability, one-agent dropout, and obstacle clearance.
2. Expand the canonical event-log projection and preserve parity as the native cockpit gains more views.
3. Add custom motor and propeller profile schemas with SI-unit validation, then calibrate parameters from bench measurements.
4. Measure target-device frame time, memory, accessibility behavior, and offline loading before making UX or performance claims.
5. Feed measured ESP-NOW and camera traces into the existing link and sensor contracts.

## Later

- SimAdapter and hardware adapters behind one state contract.
- Rotor/motor lag, thrust saturation, battery sag, downwash, weather, and venue airflow models with independent validation.
- Mission authoring, replay export, and a supervised non-contact indoor tag demo.

Payload, contact interception, targeting people, and covert surveillance remain outside the project.
