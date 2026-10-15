# Results

All numbers on this page come from simulation. The link, sensor and radio inputs are assumed or taken from published sources, not measured, and nothing here is flight-tested. Run the commands from the repository root after the setup in [INSTALL.md](../INSTALL.md#run-from-source). The short version is in the [README](../README.md#results).

## Result so far

![Tracking error against link delay for a naive tracker and a capture-time recovery tracker, on a straight line and on a turn](../runs/s6-recovery/s6_recovery.png)

In simulation (target at 1 m/s, 10 Hz measurements, 5 cm noise, 50 seeds), the naive tracker's p95 error (the error it stays under 95 % of the time) grows by about speed x delay and crosses the 0.25 m failure line at 200 ms with no loss. Stamping each measurement with its capture time and predicting forward cuts the 300 ms error with no loss from 0.36 m to 0.16 m on a straight line (no failing cell up to 300 ms and 50 % random loss) and from 0.38 m to 0.29 m on a 1 m radius turn (the first failure moves from 200 ms to 300 ms). On the turn, loss takes over: at 30 % random loss it fails from 150 ms just like naive, at 50 % both fail at every delay and recovery is up to 2 cm worse, and with bursty loss it can be up to 13 cm worse.

Limits: simulated link and sensor, not measured; the tracker is given each measurement's exact capture time (perfectly synced clocks); one constant-velocity filter with one fixed tuning; a 1 m/s target and one 1 m turn; 10 Hz; the 0.25 m line is a research criterion, not a safety limit.

```bash
PYTHONPATH=. python tools/run_s6_recovery.py --output runs/s6-recovery --overwrite
PYTHONPATH=. python tools/make_s6_figure.py runs/s6-recovery/summary.json --output-dir runs/s6-recovery
```

### What else helps a delayed tracker

![Map of which tracker tolerates the most link delay, for nine target paths (straight lines, circles, a jinking path, a figure eight, start and stop) under random and bursty packet loss. Most cells are ties, and with 30 to 50 % bursty loss on the hardest paths no tracker passes even at zero delay](../runs/s6-atlas/atlas_which_method.png)

In simulation, each idea below was tried against the capture-time tracker above, with the same 0.25 m p95 research line. In the map, a cell fails when at least half its seeds fail; that rule is proposed, not yet adopted.

- Late packets: putting out-of-order packets back into the filter gains under 1 cm (paired median, at most 0.94 cm), so dropping them costs very little (`runs/s6-retro`).
- Turns: on a 1 m radius turn at 300 ms delay and 30 % loss, a turn-rate model stays under the line (0.20 m) where the straight-line recovery tracker fails (0.40 m). The price on straight lines is up to 2 cm with no loss, up to 3 cm with random loss, and up to 10 cm with 50 % bursty loss, and no single turn-aware tracker is best on every path (`runs/s6-turn`).
- Tuning check: a consistency check on the filter (NIS) flags a wrongly guessed sensor noise without knowing the truth, but it cannot see pure link delay (`runs/s6-tuning`).
- Clock sync: a simple sync exchange every 0.1 to 1 s keeps the extra error within 1 cm of perfect clocks. It only misses with a 10 s interval and half the sync exchanges lost (about 29 % of sync messages) (`runs/s6-clocks`).
- Sending fewer packets: sending only when the target has moved can stand in for a faster periodic rate on a turn, but not on a lossy straight line. With 5 cm noise and 100 ms delay on the turn with no loss, sending after every 10 cm of movement (about 20 packets/s) gives 11.6 cm p95, against 16.9 cm at 10 packets/s and 12.1 cm at 25 packets/s. On a straight line with 30 % loss, about 19 packets/s gives 12.2 cm, against 13.9 cm at 10 packets/s and 9.7 cm at 25 packets/s. Sending positions as 16-bit numbers at 1 mm saves about 10 % of the bytes on air and moves p95 by at most 2 % with periodic sending, and up to 6 % either way with event sending (`runs/s6-sending`).
- The map: most cells are ties between trackers (39 of the 55 cells where any tracker passes). In 8 of 63 cells nothing works even at zero delay, all with 30 to 50 % bursty loss (`runs/s6-atlas`).

Limits: simulated link and sensor, not measured; the map uses 20 seeds per cell and 5 cm noise at 10 Hz; the clock test uses one 100 ms link; the packet size on air comes from the literature, not from a measurement.

```bash
# Writes a fresh copy to /tmp; the checked-in map is runs/s6-atlas.
PYTHONPATH=. python tools/run_s6_atlas.py --output /tmp/s6-atlas --seeds 20 --imm-strata central,student_t,jitter --speed-variants-central-only --no-checkpoint
PYTHONPATH=. python tools/make_s6_atlas_figure.py /tmp/s6-atlas
```

### How much delay the drone's own control loop can take

In simulation, with two micro-quad profiles built from published specs (not measured):

- Delaying only the position the drone is told to fly to never made it unstable. It only adds lag error, in proportion to the delay.
- Delaying the drone's own position and velocity feedback is different. In an 80 s check on both profiles, at 0.33 s of delay the wobble after a 1 m step still shrinks, but slowly (its peak in each 10 s window falls from 0.73 m to 0.41 m over 80 s), and at 0.34 s it grows instead (up to 1.24 m). A hand calculation from the controller's phase margin predicts about 0.32 to 0.34 s.
- If the whole state is delayed, attitude included, the limit drops to about 20 to 30 ms.

Limits: simulated, with literature-derived profiles and one controller tuning; the 80 s check uses one seed per case; ground contact is not modeled. Data: `runs/a6-long-check`, `runs/a6-control-tolerance`.

### How the radio itself behaves

In a simulated ESP-NOW-style link (parameters from the literature, not measured):

- Radio delay stays small. The worst p95 radio latency in the tracker test was about 28 ms, well under the 150 to 200 ms where the tracker starts failing from delay. In this model the tracker fails from lost packets, not from radio delay.
- Twelve drones each sending 32-byte packets 50 times a second keep the channel about two thirds busy. With retries every packet arrives, with a p95 latency of 4.4 ms. Without retries about 98 % arrive.

Limits: no outside Wi-Fi or Bluetooth traffic, every drone hears every other drone, carrier sense is perfect, and the model's top packet rate is about 13 % below the one published measurement. Data: `runs/c-exp014-mac-r2`, `runs/c-exp017-tolerance-r2`.
