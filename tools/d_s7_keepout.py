"""Controlled reruns of the simulated S7 cooperative demo to explain keep-out entries.

Runs the same configuration as tools/d_generate_s7_coop.py (4 agents, seed 23,
20 s) with one knob changed at a time and reports keep-out ring entries.
Nothing is written into sim/runs/. Simulated only.

Knobs and how they are applied:
- max_accel_mps2 and use_avoidance are real CoopSwarmConfig fields.
- "truth estimates" has no config field. It is a monkeypatch (subclass
  override) of CoopSwarmSimulator._track_for that returns the true target
  position and velocity at time t with a tiny covariance, so slots, the
  velocity feed-forward and the keep-out projection all use truth.
- "truth keep-out only" is a monkeypatch of _project_keep_out that swaps the
  center (normally the drone's fused estimate) for the true target at time t,
  or at t + dt for the "lead" variant. Everything else still uses estimates.
- "estimate_lead" (rows prefixed P_PROPOSAL) is a PROPOSAL prototype, not a
  change to the simulator: the keep-out center is the drone's own fused
  estimate projected one step ahead (fused position + fused velocity * dt).

Two distance metrics are reported:
- registered: what summary.keep_out_entries counts (simulator.py, the loop
  after motion). It compares the drone position AFTER the step (time t + dt)
  with the target truth at time t.
- same_time: drone and target at the same instant t.

Usage:
  PYTHONPATH=. python tools/d_s7_keepout.py --output <dir>/variants.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import sim.coop.simulator as simmod
from sim.coop.simulator import CoopSwarmConfig, CoopSwarmSimulator

_ORIGINAL_SAFE_VELOCITY = simmod.safe_velocity


class InstrumentedSim(CoopSwarmSimulator):
    def __init__(self, config, *, truth_estimates=False, keepout_center=None):
        super().__init__(config)
        self.truth_estimates = truth_estimates
        self.keepout_center = keepout_center  # None, "truth", or "truth_lead"
        self.frames: list[dict] = []
        self.calls: list[dict] = []
        self._t = 0.0

    def _agent_id_for(self, position):
        for aid, agent in self.agents.items():
            if position is agent.position:
                return aid
        return None

    def _dropout_state(self, t):
        self._t = float(t)
        truth_pos, truth_vel = self._truth(t)
        self.frames.append({
            "t": float(t),
            "truth": truth_pos.copy(),
            "truth_vel": np.asarray(truth_vel, dtype=float).copy(),
            "pos": {aid: a.position.copy() for aid, a in self.agents.items()},
            "vel": {aid: a.velocity.copy() for aid, a in self.agents.items()},
        })
        self.calls.append({})
        super()._dropout_state(t)

    def _track_for(self, agent, t):
        if not self.truth_estimates:
            result = super()._track_for(agent, t)
        else:
            pos, vel = self._truth(t)
            result = np.r_[pos, vel], np.eye(6) * 1e-8, "truth", float(t)
        # Cache the latest fused track per agent; the movement loop's keep-out
        # call at time t follows the allocation loop's _track_for call at t.
        self.__dict__.setdefault("_last_track", {})[agent.agent_id] = (float(t), np.asarray(result[0], dtype=float).copy())
        return result

    def _project_keep_out(self, position, velocity, center):
        aid = self._agent_id_for(position)
        fused_center = np.asarray(center, dtype=float).copy()
        if self.keepout_center == "truth":
            center = self._truth(self._t)[0]
        elif self.keepout_center == "truth_lead":
            center = self._truth(self._t + self.config.control_dt_s)[0]
        elif self.keepout_center == "estimate_lead":
            # PROPOSAL prototype only: the drone's own fused estimate projected
            # one control step ahead (position + velocity * dt). No truth used.
            stamp, mean = self._last_track[aid]
            if abs(stamp - self._t) > 1e-9 or not np.allclose(mean[:3], fused_center):
                raise RuntimeError("cached fused track does not match the keep-out call")
            center = mean[:3] + mean[3:] * self.config.control_dt_s
        out, flag = super()._project_keep_out(position, velocity, center)
        if aid is not None:
            self.calls[-1][aid] = {
                "estimate_center": fused_center,
                "used_center": np.asarray(center, dtype=float).copy(),
                "v_old": self.agents[aid].velocity.copy(),
                "v_after_avoid": np.asarray(velocity, dtype=float).copy(),
                "v_after_keepout": np.asarray(out, dtype=float).copy(),
                "keep_out_flag": bool(flag),
                "orca": getattr(self, "_last_orca", {}).get(aid),
            }
        return out, flag


def _make_safe_velocity_wrapper(sim_ref):
    def wrapped(own_pos, own_vel, preferred_vel, neighbors, config, **kw):
        out, flags = _ORIGINAL_SAFE_VELOCITY(own_pos, own_vel, preferred_vel, neighbors, config, **kw)
        sim = sim_ref[0]
        if sim is not None:
            aid = sim._agent_id_for(own_pos)
            if aid is not None:
                sim.__dict__.setdefault("_last_orca", {})[aid] = {
                    "preferred": np.asarray(preferred_vel, dtype=float).copy(),
                    "flags": sorted(flags),
                }
        return out, flags
    return wrapped


def _inward_speed(velocity, drone_pos, target_pos):
    """Speed of the drone toward the target (positive means closing)."""
    away = drone_pos - target_pos
    norm = float(np.linalg.norm(away))
    if norm < 1e-12:
        return 0.0
    return float(-np.dot(velocity, away / norm))


def run_variant(name, config, *, truth_estimates=False, keepout_center=None):
    holder = [None]
    simmod.safe_velocity = _make_safe_velocity_wrapper(holder)
    try:
        sim = InstrumentedSim(config, truth_estimates=truth_estimates, keepout_center=keepout_center)
        holder[0] = sim
        started = time.perf_counter()
        result = sim.run().as_dict()
        runtime = time.perf_counter() - started
    finally:
        simmod.safe_velocity = _ORIGINAL_SAFE_VELOCITY
    c = config
    frames, calls = sim.frames, sim.calls
    n = len(frames) - 1
    r = c.keep_out_radius_m
    registered, same_time = [], []
    for k in range(n + 1):
        after = frames[k + 1]["pos"] if k < n else frames[k]["pos"]
        for aid in sorted(after):
            d_reg = float(np.linalg.norm(after[aid] - frames[k]["truth"]))
            if d_reg < r:
                registered.append({"k": k, "t": frames[k]["t"], "drone": aid, "distance_m": d_reg})
            d_same = float(np.linalg.norm(frames[k]["pos"][aid] - frames[k]["truth"]))
            if d_same < r:
                same_time.append({"k": k, "t": frames[k]["t"], "drone": aid, "distance_m": d_same})

    def context(entry, metric):
        k, aid = entry["k"], entry["drone"]
        call = calls[k].get(aid) if k < len(calls) else None
        out = dict(entry)
        out["penetration_m"] = r - entry["distance_m"]
        if call is None or k >= n:
            return out
        v_old = call["v_old"]
        v_new = frames[k + 1]["vel"][aid]
        dt = c.control_dt_s
        requested_accel = float(np.linalg.norm(call["v_after_keepout"] - v_old) / dt)
        realized_accel = float(np.linalg.norm(v_new - v_old) / dt)
        drone_t = frames[k]["pos"][aid]
        truth_t = frames[k]["truth"]
        orca = call["orca"] or {}
        preferred = orca.get("preferred")
        out.update({
            "speed_after_mps": float(np.linalg.norm(v_new)),
            "requested_accel_after_keepout_mps2": requested_accel,
            "realized_accel_mps2": realized_accel,
            "accel_at_clamp": bool(requested_accel > c.max_accel_mps2 + 1e-9),
            "keep_out_projection_fired": call["keep_out_flag"],
            "estimate_error_m": float(np.linalg.norm(call["estimate_center"] - truth_t)),
            "keepout_center_error_vs_truth_t_m": float(np.linalg.norm(call["used_center"] - truth_t)),
            "inward_speed_preferred_mps": None if preferred is None else _inward_speed(preferred, drone_t, truth_t),
            "inward_speed_after_avoidance_mps": _inward_speed(call["v_after_avoid"], drone_t, truth_t),
            "inward_speed_after_keepout_mps": _inward_speed(call["v_after_keepout"], drone_t, truth_t),
            "inward_speed_realized_mps": _inward_speed(v_new, drone_t, truth_t),
            "target_speed_toward_drone_mps": _inward_speed(-frames[k]["truth_vel"], drone_t, truth_t),
            "avoidance_flags": orca.get("flags"),
        })
        return out

    def describe(entries, metric):
        if not entries:
            return {"entries": 0, "deepest": None, "times_s": [], "drones": []}
        deepest = min(entries, key=lambda e: e["distance_m"])
        return {
            "entries": len(entries),
            "deepest": context(deepest, metric),
            "times_s": [round(e["t"], 3) for e in entries],
            "drones": sorted({e["drone"] for e in entries}),
            "all": [context(e, metric) for e in entries],
        }

    reported = int(result["summary"]["keep_out_entries"])
    return {
        "variant": name,
        "seed": c.seed,
        "max_accel_mps2": c.max_accel_mps2,
        "use_avoidance": c.use_avoidance,
        "truth_estimates": truth_estimates,
        "keepout_center": keepout_center or "fused_estimate",
        "simulator_reported_keep_out_entries": reported,
        "recount_matches_simulator": reported == len(registered),
        "registered_metric": describe(registered, "registered"),
        "same_time_metric": describe(same_time, "same_time"),
        "runtime_s": runtime,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10, help="extra baseline seeds (1..N) for the seed check")
    args = parser.parse_args()
    if "runs" in args.output.resolve().parts:
        raise SystemExit("refusing to write under sim/runs/")
    base = CoopSwarmConfig(n_agents=4, seed=23, duration_s=20.0)
    plan = [
        ("A_baseline", base, {}),
        ("B_truth_estimates", base, {"truth_estimates": True}),
        ("B_ko_truth_keepout_center_only", base, {"keepout_center": "truth"}),
        ("B_lead_truth_t_plus_dt_keepout_center_only", base, {"keepout_center": "truth_lead"}),
        ("C2_accel_x2", replace(base, max_accel_mps2=8.0), {}),
        ("C4_accel_x4", replace(base, max_accel_mps2=16.0), {}),
        ("Cinf_accel_unclamped", replace(base, max_accel_mps2=1e6), {}),
        ("D2_truth_estimates_accel_x2", replace(base, max_accel_mps2=8.0), {"truth_estimates": True}),
        ("D4_truth_estimates_accel_x4", replace(base, max_accel_mps2=16.0), {"truth_estimates": True}),
        ("G_no_avoidance", replace(base, use_avoidance=False), {}),
    ]
    for seed in range(1, args.seeds + 1):
        seeded = replace(base, seed=seed)
        plan.append((f"E_baseline_seed{seed}", seeded, {}))
        plan.append((f"E_truth_estimates_seed{seed}", seeded, {"truth_estimates": True}))
        plan.append((f"E_accel_x4_seed{seed}", replace(seeded, max_accel_mps2=16.0), {}))
    # Repair round: clean 2x2 grid on seed 23. Estimates stay as-is everywhere
    # except the keep-out centre. Existing rows above are unchanged.
    unclamped = replace(base, max_accel_mps2=1e6)
    plan += [
        ("R_truth_center_t_accel_off", unclamped, {"keepout_center": "truth"}),
        ("R_truth_center_t_plus_dt_accel_off", unclamped, {"keepout_center": "truth_lead"}),
        ("P_PROPOSAL_estimate_lead_center_accel_4", base, {"keepout_center": "estimate_lead"}),
        ("P_PROPOSAL_estimate_lead_center_accel_off", unclamped, {"keepout_center": "estimate_lead"}),
    ]
    for seed in range(1, args.seeds + 1):
        seeded = replace(base, seed=seed)
        plan.append((f"P_PROPOSAL_estimate_lead_accel_4_seed{seed}", seeded, {"keepout_center": "estimate_lead"}))
        plan.append((f"P_PROPOSAL_estimate_lead_accel_off_seed{seed}", replace(seeded, max_accel_mps2=1e6), {"keepout_center": "estimate_lead"}))
    results = []
    for name, config, kwargs in plan:
        row = run_variant(name, config, **kwargs)
        results.append(row)
        reg, same = row["registered_metric"], row["same_time_metric"]
        reg_min = "-" if reg["deepest"] is None else f"{reg['deepest']['distance_m']:.4f}"
        same_min = "-" if same["deepest"] is None else f"{same['deepest']['distance_m']:.4f}"
        print(
            f"{name:46s} reg={reg['entries']:3d} min={reg_min} "
            f"same={same['entries']:3d} min_same={same_min} "
            f"recount_ok={row['recount_matches_simulator']}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, (np.floating, float)):
            return round(float(value), 6)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    payload = {
        "schema": "d-s7-keepout-variants-1",
        "simulated_only": True,
        "keep_out_radius_m": base.keep_out_radius_m,
        "control_dt_s": base.control_dt_s,
        "base_max_accel_mps2": base.max_accel_mps2,
        "variants": clean(results),
    }
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
