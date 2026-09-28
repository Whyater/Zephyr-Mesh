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

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Planned (not built yet)

The core question: how much radio latency, packet loss, and tracking noise can a group of drones tolerate while tracking a target, and what helps them recover?

- Step-response metrics computed from logged data
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

## Notes

Built by Abyan Kashif with substantial AI assistance.

## License

MIT. See `LICENSE`.
