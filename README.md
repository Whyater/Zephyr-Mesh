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

### S2 physical replay demo

S2 keeps the baseline replay read-only and adds a responsive physical flight view. Each recorded sample is shown as a small quadrotor over a ground frame, with position, velocity, quaternion-derived roll/pitch/yaw, and a thrust direction that follows the displayed attitude. The replay controls support play, pause, scrub, and stopping the replay. The visible `ABORT / STOP` control is disabled and labelled **NO FLIGHT ACTION** because this build has no flight-control path.

The browser payload is served by `demo/demo.py` with schema `zephyr-s2-demo-1`. `/api/run?limit=N` returns up to `N` samples from the retained telemetry and includes a `zephyr-s2-report-1` report. The report records SHA-256 checksums for the retained telemetry, manifest, and S2 convergence artifact, and labels radio latency/loss, sensor noise, and recovery as unavailable. `POST /api/rerun` accepts only an empty JSON object and replays the same retained fixture; it does not run a live simulation or accept control parameters. The downloadable `telemetry.csv` and `run.json` endpoints use that same retained source.

The physical view is an explanatory replay, not a flight display and not evidence of flight performance. Live radio, camera, manual controller input, swarm commands, and age-of-data timestamps are not implemented. Manual controller assignment, per-drone takeover and return authority, team-command arbitration, and swarm task views remain planned design work.

### S3 deterministic link replay

S3 adds `sim/link.py`, a seeded packet-link fixture for testing how a tracker records imperfect delivery. `LinkConfig` can apply a fixed delay with optional seeded jitter, independent loss, burst loss, or no loss. The `serialized` contention mode queues packets on one declared FIFO transmission resource. It is an explicit abstraction, not a model of ESP-NOW airtime, CSMA/CA, or radio backoff.

Each attempted packet retains sender, receiver, sequence number, send and receive times, packet age, loss reason, and duplicate or out-of-order flags. `SimulatedLink` keeps the event log deterministic for a given seed and exposes `advance()` and `deliver_all()` for replay fixtures. The `lag_error()` helper checks the first-order `v × L` relationship between target speed and communication delay. These are simulator checks only. They do not measure a radio or establish a tolerable field link budget.

The browser's **S3 link replay** panel calls `GET /api/link` and shows the retained synthetic fixture, including delay, loss, serialized contention, packet age, and delivery flags. The endpoint rejects query parameters and has no control or live-radio path. Its payload uses schema `zephyr-s3-link-fixture-1` and labels live radio, manual controller input, swarm commands, and measured link quality as unavailable. The full test suite currently passes with 44 tests:

```bash
.venv/bin/python -m pytest -q
```

### S1 baseline metrics

S1 adds a local browser demo and read-only report for the retained S0 telemetry. It is a **synthetic single-drone baseline replay**, not flight performance. The report covers step response and position-error metrics; radio latency or packet loss, sensor noise, packet age, detection reacquisition, and controller recovery are unavailable because this telemetry contains no link or sensor events.

The refreshed 700-sample, 6.99-second post-S2 baseline reports a 0.28 s rise time, 8.35% overshoot, 2.99 s settling time, and 0.867 m RMS position error. These conventions use the first sample reaching 10% of the step amplitude for rise time, the first sample after which all values stay within ±5% of the step amplitude for settling, and overshoot divided by the absolute step amplitude. RMS error compares each sample with the fixed target `[0, 0, 3]` m. The retained timestamps use the S0 pre-step label for the state recorded after each step.

Reproduce the JSON report with:

```bash
.venv/bin/python -m sim.report runs/s0-baseline --output /tmp/report.json
```

These metrics describe the tested simulator configuration and do not establish flight performance.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

To run the S2 browser replay on the Mac, use loopback by default:

```bash
.venv/bin/python -m demo.demo --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` on the Mac. For the same replay on a phone, bind explicitly to the Mac's Tailscale IP and open `http://<MAC_TAILSCALE_IP>:8765` on a phone that is on the same tailnet:

```bash
.venv/bin/python -m demo.demo --host <MAC_TAILSCALE_IP> --port 8765
```

This server is unauthenticated, so treat a Tailscale bind as a trusted-tailnet boundary. Keep the Mac awake while it runs, do not expose it with a public Tailscale Funnel, and stop the server with `Ctrl-C` when finished.

The phone view uses the same HTML, retained telemetry, and API as the Mac view. It requires the Tailscale app to be connected on the phone and the Mac to remain awake. This is a temporary private preview, not a public deployment.

Run the test suite with:

```bash
.venv/bin/python -m pytest -q
```

## Planned (not built yet)

The core question: how much radio latency, packet loss, and tracking noise can a group of drones tolerate while tracking a target, and what helps them recover?

- Radio behavior measured with ESP-NOW hardware, including latency, loss, update rate, and contention
- Sensor and camera noise sized from real measurements
- Kalman-filter target tracking
- ESP-NOW radio experiments on ESP32 boards
- Ground-camera vision tracking (OpenCV, then YOLO)
- Multiple drones sharing information over the radio link
- Physical controller input with explicit per-drone assignment, takeover, return-authority confirmation, and link-loss behavior
- Swarm task commands with clear team-command and joystick arbitration
- A multi-drone physical view that shows each model, identity, command authority, and stale data state
- Small indoor drones for real flight tests, non-contact only

Everything in this section is a plan. It moves up to "What exists today" only when it works and has been checked.

## History

- **March 2026:** the idea took shape as theorycrafting about how low-cost drones could coordinate to track other drones.
- **Summer 2026:** simulator code written; first commit July 1, 2026.

## License

MIT. See `LICENSE`.
