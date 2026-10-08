# Zephyr-Mesh

A drone flight simulator in Python, built without a physics engine, and the starting point for a project on cooperative drone tracking over imperfect radio links.

## What exists today

- **6-DOF rigid-body model** (`sim/drone.py`): 13-number state (position, velocity, orientation quaternion, body angular velocity), diagonal inertia tensor, and gyroscopic coupling. The quaternion is re-normalized every step.
- **Environment** (`sim/environment.py`): gravity, configurable wind, and quadratic drag from relative air velocity.

### S2 physics status

The S2 physics correction makes `step_physics` a true classical RK4 update and applies quadratic drag from explicit relative air velocity. Independent checks cover constant velocity, constant acceleration, a falling-body limit, an independent harmonic-oscillator fixture, rotation with nonzero angular velocity, drag, hover, quaternion normalization, and timestep convergence. The coefficients remain scenario parameters, not measured flight-model values.

The raw harmonic-oscillator convergence values are retained in [`runs/s2-physics/convergence.json`](runs/s2-physics/convergence.json). For this one-second scalar fixture, halving `dt` produced observed absolute-error ratios of 17.50, 17.04, and 16.60 across `dt` values 0.2, 0.1, 0.05, and 0.025 seconds. These are numerical fixture results only and do not establish flight performance.
- **Cascaded controller** (`sim/controller.py`): an outer PD position loop with gravity compensation, and an inner attitude loop that tilts the drone using the cross product of its current and desired "up" axes. It is PD, not PID.
- **Hover mission** (`main.py`): the drone climbs from the ground to a 3 m target at 100 Hz for 7 seconds and plots altitude over time.

The S2 force model is still an intentionally transparent reference model. It includes gravity, configured wind, relative-air quadratic drag, thrust, and rigid-body gyroscopic coupling. It does not yet model rotor thrust maps, motor dynamics, ground effect, propeller wake interaction, battery sag, or indoor airflow. Those additions need measured inputs and independent validation before the simulator can be used to estimate crash risk.

### S2 physical replay demo

S2 keeps the baseline replay read-only and adds a responsive physical flight view. Each recorded sample is shown as a small quadrotor over a ground frame, with position, velocity, quaternion-derived roll/pitch/yaw, and a thrust direction that follows the displayed attitude. The replay controls support play, pause, stepping, scrubbing, speed changes, and stopping the replay. The `Stop replay` action only stops the local timeline because this build has no flight-control path.

The browser payload is served by `demo/demo.py` with schema `zephyr-s2-demo-1`. `/api/run?limit=N` returns up to `N` samples from the retained telemetry and includes a `zephyr-s2-report-1` report. The report separates payload sample count and truncation from full-source metadata. SHA-256 checksums refer to the complete retained telemetry, manifest, and S2 convergence artifact, and labels radio latency/loss, sensor noise, and recovery as unavailable. `POST /api/rerun` accepts only an empty JSON object and replays the same retained fixture; it does not run a live simulation or accept control parameters. The downloadable `telemetry.csv` and `run.json` endpoints use that same retained source.

The physical view is an explanatory replay, not a flight display and not evidence of flight performance. Live radio, camera, manual controller input, swarm commands, and age-of-data timestamps are not implemented. Manual controller assignment, per-drone takeover and return authority, team-command arbitration, and swarm task views remain planned design work.

### S3 deterministic link replay

S3 adds `sim/link.py`, a seeded packet-link fixture for testing how a tracker records imperfect delivery. `LinkConfig` can apply a fixed delay with optional seeded jitter, independent loss, burst loss, or no loss. The `serialized` contention mode queues packets on one declared FIFO transmission resource. It is an explicit abstraction, not a model of ESP-NOW airtime, CSMA/CA, or radio backoff.

Each attempted packet retains sender, receiver, sequence number, send and receive times, packet age, loss reason, and duplicate or out-of-order flags. `SimulatedLink` keeps the event log deterministic for a given seed and exposes `advance()` and `deliver_all()` for replay fixtures. The `lag_error()` helper checks the first-order `v × L` relationship between target speed and communication delay. These are simulator checks only. They do not measure a radio or establish a tolerable field link budget.

The browser's **S3 link replay** panel calls `GET /api/link` and shows the retained synthetic fixture, including delay, loss, serialized contention, packet age, and delivery flags. The endpoint rejects query parameters and has no control or live-radio path. Its payload uses schema `zephyr-s3-link-fixture-1` and labels live radio, manual controller input, swarm commands, and measured link quality as unavailable.

### S4 sensing, S5 tracking, and S6 investigation

S4 adds `sim/sensor.py`, which keeps truth, measurement, timestamp, validity, bias, noise, and dropout reason separate. Independent and burst dropout, Gaussian noise, fixed bias, and seeded random-walk bias are model inputs. It is a sensing fixture, not a camera or IMU model, and no coefficients are presented as measured.

S5 adds `sim/tracker.py`, an inspectable three-dimensional constant-velocity Kalman filter. It predicts through missing observations, keeps covariance, rejects out-of-order timestamps instead of silently rewinding, and separates estimator prediction from controller recovery. The S5 endpoint is a synthetic comparison fixture, not flight performance.

S6 adds `sim/investigation.py`, a deterministic grid over observation delay, independent loss, position noise, and optional target acceleration. Each row preserves its configured delay, actual grid-sample observation age, held-last baseline, tracker error summary, sample count, and scenario-specific failure criterion. The independent check remains `v × L`: at 0.7 m/s and 0.10 s, a held-last-position estimate is off by 0.07 m before noise or dropout. This sweep does not establish a radio tolerance or a safety threshold.

The browser is now a phone-first Horizon replay cockpit. The scene is dominant, camera views and timeline controls are grouped together, packet details are behind disclosure, and S4, S5, and S6 load their synthetic fixtures on demand. The cockpit uses the existing APIs and marks simulated, recorded, and unavailable data explicitly. Its SVG scene is a visual replay model, not a full 3D rigid-body renderer.

The full Python suite currently passes with 85 tests:

```bash
.venv/bin/python -m pytest -q
```

### Simulator phase map

There are eight simulator phases. S0 baseline hooks, S1 metrics, S2 physics correctness, and S3 deterministic link replay are implemented. S4 seeded sensing, S5 constant-velocity tracking recovery, and S6 bounded latency/loss/noise investigation are implemented as synthetic fixtures. S7 now has a **local coordination foundation**, not a flight-ready swarm: `sim/swarm.py` models seeded target and neighbor packet paths, cooperative target fusion, formation slots, dropout schedules, pairwise separation, spherical keep-out constraints, and replayable event data. `sim/hardware.py` defines SI-unit motor, propeller, and airframe profiles with explicit torque, wattage, RPM, diameter, pitch, and transparent static-thrust estimates. `sim/scenarios.py` creates a deterministic 50-agent ring with unique synthetic profile IDs and a scenario manifest. The checked-in `runs/s7-swarm/` artifact is synthetic and not flight performance.

### Native macOS cockpit preview

`macos/ZephyrMeshApp` is a native SwiftUI + SceneKit desktop surface. It loads a compact projection of the canonical Python S7 event log and provides a 3D fleet view, 50-agent fleet rail, link-health badges, safety gate, obstacle toggle, and replay controls. The app is read-only and labels live radio, camera input, manual authority, and flight performance as unavailable. Build it on macOS 14 or newer with:

```bash
cd macos/ZephyrMeshApp
swift build
swift run
```

The renderer is local and dependency-free. It is a cockpit surface, not a substitute for the Python physics or a flight-validation result. Recreate the canonical fixture with `PYTHONPATH=. ./.venv/bin/python tools/generate_s7_fixture.py --output runs/s7-swarm --agents 50 --steps 6 --seed 17`.

### S1 baseline metrics

S1 adds a local browser demo and read-only report for the retained S0 telemetry. It is a **synthetic single-drone baseline replay**, not flight performance. The report covers step response and position-error metrics; radio latency or packet loss, sensor noise, packet age, detection reacquisition, and controller recovery are unavailable because this telemetry contains no link or sensor events.

The frozen historical 700-sample, 6.99-second post-S2 baseline reports a 0.28 s rise time, 8.35% overshoot, 2.99 s settling time, and 0.867 m RMS position error. These conventions use the first sample reaching 10% of the step amplitude for rise time, the first sample after which all values stay within ±5% of the step amplitude for settling, and overshoot divided by the absolute step amplitude. RMS error compares each sample with the fixed target `[0, 0, 3]` m. The retained timestamps use the S0 pre-step label for the state recorded after each step. Its manifest records the original revision and dirty tree state; it is not a provenance-matched run of the current working tree. Machine-specific paths are normalized for distribution.

Reproduce the JSON report with:

```bash
.venv/bin/python -m sim.report runs/s0-baseline --output /tmp/report.json
```

These metrics describe the tested simulator configuration and do not establish flight performance.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

To run the browser replay locally from the repository root:

```bash
.venv/bin/python -m demo.demo --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` in a browser on the same computer. The server serves the retained, synthetic replay, read-only S4-S6 fixtures, and the versioned S7 event-log projection at `/api/swarm`. It does not expose a radio, camera, motor, controller, or swarm command path.

Stop the local server with `Ctrl-C` when finished. The repository does not claim a hosted public deployment. See `--help` for local server options.

Run the test suite with:

```bash
.venv/bin/python -m pytest -q
```

## Planned (not built yet)

The core question: how much radio latency, packet loss, and tracking noise can a group of drones tolerate while tracking a target, and what helps them recover?

- Radio behavior measured with ESP-NOW hardware, including latency, loss, update rate, and contention
- Sensor and camera noise sized from real measurements
- measured sensing and camera noise inputs
- Kalman-filter target tracking recovery
- ESP-NOW radio experiments on ESP32 boards
- Ground-camera vision tracking (OpenCV, then YOLO)
- Multiple drones sharing information over the radio link
- Physical controller input with explicit per-drone assignment, takeover, return-authority confirmation, and link-loss behavior
- Swarm task commands with clear team-command and joystick arbitration
- A multi-drone physical view that shows each model, identity, command authority, and stale data state
- Small indoor drones for real flight tests, non-contact only
- Motor and propeller coefficients calibrated from bench measurements and replayed through the profile contract
- Native cockpit loading of full event-log timelines, with target-device frame-time and accessibility measurements

Everything in this section is a plan. It moves up to "What exists today" only when it works and has been checked.

## History

- **March 2026:** the idea took shape as theorycrafting about how low-cost drones could coordinate to track other drones.
- **Summer 2026:** simulator code written; first commit July 1, 2026.

## License

MIT. See `LICENSE`.
