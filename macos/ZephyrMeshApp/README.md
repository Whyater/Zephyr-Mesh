# Zephyr Mesh native cockpit

This directory contains the first native macOS surface for Zephyr Mesh. It is a SwiftUI + SceneKit executable that renders a local, offline fleet fixture with a 3D scene, fleet rail, link-health badges, safety gate, replay controls, and provenance labels.

The app is a **local synthetic cockpit preview**. It loads a compact projection of the canonical Python S7 event log from `runs/s7-swarm/run.json`; the Python simulator remains the source of truth for dynamics and evidence. The Swift fixture is intentionally labeled unavailable for live radio, camera input, manual authority, and flight performance.

## Build and run

On macOS 14 or newer with Xcode 27 or Swift 6:

```bash
cd macos/ZephyrMeshApp
swift build
swift run
```

Alternatively, open `Package.swift` in Xcode and run the `ZephyrMeshApp` scheme. The package has no third-party or network dependency. SceneKit is an Apple platform framework selected for the no-SVG, offline desktop surface documented in the private Decision Log.

## Evidence boundary

The app currently proves that the native surface builds and can render a deterministic projection. It does not prove frame time on a target Mac, physics fidelity, collision safety, radio behavior, or flight performance. A parity test checks the projection's schema, frames, identities, positions, authority labels, and source contract against the checked-in Python artifact.
