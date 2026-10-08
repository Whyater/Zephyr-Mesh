# Zephyr-Mesh

A drone flight simulator in Python, built without a physics engine, and the starting point for a project on cooperative drone tracking over imperfect radio links.

## What exists today

- **6-DOF rigid-body model** (`sim/drone.py`): 13-number state (position, velocity, orientation quaternion, body angular velocity), diagonal inertia tensor, and gyroscopic coupling. The quaternion is re-normalized every step.
- **Environment** (`sim/environment.py`): gravity, plus a quadratic drag function.

### Known issues (found Sep 28, 2026, not fixed yet)

- **The integrator is effectively first-order (Euler), not RK4.** `step_physics` computes midpoint values but never feeds them back into `compute_derivatives`, so both derivative calls return the same thing and the midpoint values go unused. Fixing this, and checking the fix against a case with a known exact answer, is the first planned change.
- **Drag is never applied.** `calculate_drag` is imported in `drone.py` but not called, so the simulated drone currently flies with no air resistance.
- **Cascaded controller** (`sim/controller.py`): an outer PD position loop with gravity compensation, and an inner attitude loop that tilts the drone using the cross product of its current and desired "up" axes. It is PD, not PID.
- **Hover mission** (`main.py`): the drone climbs from the ground to a 3 m target at 100 Hz for 7 seconds and plots altitude over time.

### S1 baseline replay

S1 adds a local browser demo and read-only report for the retained S0 telemetry. It is a **synthetic single-drone baseline replay**, not flight performance. The report covers step response and position-error metrics; radio latency or packet loss, sensor noise, packet age, detection reacquisition, and controller recovery are unavailable because this telemetry contains no link or sensor events.

The retained 700-sample, 6.99-second baseline reports a 0.28 s rise time, 11.21% overshoot, 3.03 s settling time, and 0.863 m RMS position error. These conventions use the first sample reaching 10% of the step amplitude for rise time, the first sample after which all values stay within ±5% of the step amplitude for settling, and overshoot divided by the absolute step amplitude. RMS error compares each sample with the fixed target `[0, 0, 3]` m. The retained timestamps use the S0 pre-step label for the state recorded after each step.

Reproduce the JSON report with:

```bash
.venv/bin/python -m sim.report runs/s0-baseline --output /tmp/report.json
```

Known simulation issues remain: the integrator labeled RK4 behaves as forward Euler, and drag is not applied. These metrics do not establish flight performance.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

To run the S1 browser demo on the Mac, use loopback by default:

```bash
.venv/bin/python -m demo.demo --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` on the Mac. For a phone preview, bind explicitly to the Mac's Tailscale IP and open `http://<MAC_TAILSCALE_IP>:8765` on a phone that is on the same tailnet:

```bash
.venv/bin/python -m demo.demo --host <MAC_TAILSCALE_IP> --port 8765
```

This server is unauthenticated, so treat a Tailscale bind as a trusted-tailnet boundary. Keep the Mac awake while it runs, do not expose it with a public Tailscale Funnel, and stop the server with `Ctrl-C` when finished.

Run the test suite with:

```bash
.venv/bin/python -m pytest -q
```

## Planned (not built yet)

The core question: how much radio latency, packet loss, and tracking noise can a group of drones tolerate while tracking a target, and what helps them recover?

- Simulated radio link (delay, loss, update rate) and camera noise, sized from real measurements
- Kalman-filter target tracking
- ESP-NOW radio experiments on ESP32 boards
- Ground-camera vision tracking (OpenCV, then YOLO)
- Multiple drones sharing information over the radio link
- Small indoor drones for real flight tests, non-contact only

Everything in this section is a plan. It moves up to "What exists today" only when it works and has been checked.

## History

- **March 2026:** the idea took shape as theorycrafting about how low-cost drones could coordinate to track other drones.
- **Summer 2026:** simulator code written; first commit July 1, 2026.

## License

MIT. See `LICENSE`.
