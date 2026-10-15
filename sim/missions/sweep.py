"""Simulated scenario of a planned use case: area sweep with shared coverage."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from sim.adapters import Geofence
from sim.missions.engine import Mission, agent_frame, register_mission


@register_mission
class SweepMission(Mission):
    name = "sweep"
    rng_names = ()
    limits = (
        "Simulated only: coverage is a point-mass sensor footprint on a one metre grid.",
        "Synthetic radio loss and adapter dynamics are not field measurements.",
    )
    defaults = {
        "message_mode": "delta",
        "mode": None,
        "dropout": "none",
    }

    _DRONES = ("d1", "d2", "d3")
    _WIDTH = 40
    _HEIGHT = 20
    _ALTITUDE = 10.0
    _SPEED = 2.0
    _SHARE_HZ = 5.0
    _LOSS_TIMEOUT = 2.0

    def setup(self, ctx):
        selected_mode = ctx.params["message_mode"] if ctx.params.get("mode") is None else ctx.params["mode"]
        mode = str(selected_mode)
        dropout = str(ctx.params["dropout"])
        if mode not in {"delta", "cumulative"}:
            raise ValueError("message_mode must be delta or cumulative")
        if dropout not in {"none", "one"}:
            raise ValueError("dropout must be none or one")

        # Three equal sectors.  Grid centres are used as the near-left entry
        # and make the strip assignment deterministic.
        edges = np.linspace(0.0, 20.0, 4)
        strip_lanes: list[list[tuple[float, float, float]]] = []
        strip_routes: list[list[tuple[float, float, float]]] = []
        strip_masks: list[np.ndarray] = []
        starts = []
        for k in range(3):
            low, high = float(edges[k]), float(edges[k + 1])
            # Keep a two metre footprint comfortably inside each strip while
            # retaining four metre lane spacing.  Starts remain at the
            # near-left strip corner below the first lane.
            offsets = (1.0, 5.0 / 6.0, 2.0 / 3.0)
            ys = np.arange(low + offsets[k], high, 4.0, dtype=float)
            if ys.size == 0:
                ys = np.array([(low + high) / 2.0])
            lanes = []
            for j, y in enumerate(ys):
                left_to_right = (j % 2) == 0
                x = 38.5 if left_to_right else 1.5
                lanes.append((float(x), float(y), self._ALTITUDE))
            route = []
            for j, lane in enumerate(lanes):
                y = lane[1]
                if j % 2 == 0:
                    route.extend(((1.0, y, self._ALTITUDE), (39.0, y, self._ALTITUDE)))
                else:
                    route.extend(((39.0, y, self._ALTITUDE), (1.0, y, self._ALTITUDE)))
                if j + 1 < len(lanes):
                    route.append((route[-1][0], lanes[j + 1][1], self._ALTITUDE))
            strip_routes.append(route)
            strip_lanes.append(lanes)
            yy = np.arange(self._HEIGHT) + 0.5
            xx = np.arange(self._WIDTH) + 0.5
            strip_masks.append((yy[:, None] >= low) & (yy[:, None] < high) & np.ones((1, self._WIDTH), dtype=bool))
            starts.append((2.0, float(ys[0]), self._ALTITUDE))

        # The start points are the near-left corners of the three strips.
        # Assignment is nevertheless performed through the specified solver.
        cost = np.zeros((3, 3), dtype=float)
        for i, start in enumerate(starts):
            for j, lanes in enumerate(strip_lanes):
                entry = lanes[0]
                cost[i, j] = math.dist(start, entry)
        rows, cols = linear_sum_assignment(cost)
        assignment = {int(r): int(c) for r, c in zip(rows, cols)}

        geofence = Geofence((-5.0, -5.0, 0.0), (45.0, 25.0, 20.0), margin_m=0.5)
        for i, drone_id in enumerate(self._DRONES):
            ctx.add_drone(
                drone_id,
                initial_position_m=starts[i],
                geofence=geofence,
                max_speed_mps=self._SPEED,
            )

        for sender in self._DRONES:
            for receiver in (*self._DRONES, "ground"):
                if sender == receiver:
                    continue
                ctx.link(f"share:{sender}>>{receiver}")

        own = np.zeros((3, self._HEIGHT, self._WIDTH), dtype=bool)
        counts = np.zeros((self._HEIGHT, self._WIDTH), dtype=np.int16)
        beliefs = {d: own[k].copy() for k, d in enumerate(self._DRONES)}
        ctx.state.update(
            {
                "assignment": assignment,
                "message_mode": mode,
                "strip_lanes": strip_lanes,
                "strip_routes": strip_routes,
                "strip_masks": strip_masks,
                "strip_owner": {d: assignment[k] for k, d in enumerate(self._DRONES)},
                "own": own,
                "counts": counts,
                "beliefs": beliefs,
                "last_sent": {d: own[k].copy() for k, d in enumerate(self._DRONES)},
                "send_seq": {d: 0 for d in self._DRONES},
                "last_send_due": {d: -1 for d in self._DRONES},
                "last_msg": {r: {s: None for s in self._DRONES if s != r} for r in self._DRONES},
                "declared_lost": {r: set() for r in self._DRONES},
                "dropout_done": False,
                "alive": {d: True for d in self._DRONES},
                "queues": {d: list(strip_routes[assignment[i]]) for i, d in enumerate(self._DRONES)},
                "indices": {d: 0 for d in self._DRONES},
                "last_goals": {d: None for d in self._DRONES},
                "replans": 0,
                "belief_error_cells": 0,
                "false_loss_declarations": 0,
                "coverage_history": [],
                "t95": None,
                "t100": None,
                "ground_belief": np.zeros((self._HEIGHT, self._WIDTH), dtype=bool),
            }
        )
        ctx.start_log(self._DRONES)

    @staticmethod
    def _flat_to_mask(indices: Any, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        if indices is None:
            return mask
        flat = np.asarray(indices, dtype=int).reshape(-1)
        flat = flat[(flat >= 0) & (flat < shape[0] * shape[1])]
        mask.reshape(-1)[flat] = True
        return mask

    def _receive_shares(self, ctx, t: float) -> None:
        state = ctx.state
        beliefs = state["beliefs"]
        for receiver in self._DRONES:
            for sender in self._DRONES:
                if sender == receiver:
                    continue
                link = ctx.links[f"share:{sender}>>{receiver}"]
                for packet in link.deliveries_until(t):
                    payload = packet.message
                    if isinstance(payload, dict) and payload.get("mode") == "cumulative":
                        mask = np.asarray(payload["coverage"], dtype=bool)
                        if mask.shape == beliefs[receiver].shape:
                            beliefs[receiver] |= mask
                    elif isinstance(payload, dict):
                        beliefs[receiver] |= self._flat_to_mask(payload.get("indices", ()), beliefs[receiver].shape)
                    state["last_msg"][receiver][sender] = float(packet.receive_time_s)

        for sender in self._DRONES:
            link = ctx.links[f"share:{sender}>>ground"]
            for packet in link.deliveries_until(t):
                payload = packet.message
                if isinstance(payload, dict) and payload.get("mode") == "cumulative":
                    mask = np.asarray(payload["coverage"], dtype=bool)
                    if mask.shape == state["ground_belief"].shape:
                        state["ground_belief"] |= mask
                elif isinstance(payload, dict):
                    state["ground_belief"] |= self._flat_to_mask(payload.get("indices", ()), state["ground_belief"].shape)

    def _mark_true(self, ctx, states: dict[str, Any]) -> None:
        own = ctx.state["own"]
        counts = ctx.state["counts"]
        yy, xx = np.indices((self._HEIGHT, self._WIDTH), dtype=float)
        xx += 0.5
        yy += 0.5
        for k, drone in enumerate(self._DRONES):
            value = states[drone]
            p = np.asarray(value.position_m, dtype=float)
            # Tiny tolerance avoids dropping cells on the exact swath edge
            # because adapter positions and cell centres are floating point.
            newly = (xx - p[0]) ** 2 + (yy - p[1]) ** 2 <= (2.0 + 1e-9) ** 2
            newly &= ~own[k]
            if np.any(newly):
                own[k] |= newly
                counts[newly] += 1
            # A drone always knows its own current map.
            ctx.state["beliefs"][drone] |= own[k]

    def _send_shares(self, ctx, t: float) -> None:
        state = ctx.state
        mode = state["message_mode"]
        due = int(math.floor(float(t) * self._SHARE_HZ + 1e-9))
        for sender_i, sender in enumerate(self._DRONES):
            if not state["alive"][sender] or due <= state["last_send_due"][sender]:
                continue
            own = state["own"][sender_i]
            if mode == "cumulative":
                payload = {"mode": "cumulative", "coverage": own.copy()}
            else:
                delta = own & ~state["last_sent"][sender]
                payload = {"mode": "delta", "indices": sorted(int(v) for v in np.flatnonzero(delta))}
            seq = state["send_seq"][sender]
            for receiver in (*self._DRONES, "ground"):
                if receiver == sender:
                    continue
                # Cumulative payloads are copied independently per stream.
                message = payload if mode == "delta" else {"mode": "cumulative", "coverage": own.copy()}
                ctx.links[f"share:{sender}>>{receiver}"].send(seq, t, message)
            state["send_seq"][sender] = seq + 1
            state["last_send_due"][sender] = due
            state["last_sent"][sender] = own.copy()

    def _make_replan(self, ctx, lost: str, receiver: str, t: float) -> None:
        state = ctx.state
        strip_index = state["strip_owner"][lost]
        strip = state["strip_masks"][strip_index]
        believed = state["beliefs"][receiver]
        uncovered = strip & ~believed
        true_covered = np.any(state["own"][:, uncovered], axis=0) if np.any(uncovered) else np.array([], dtype=bool)
        # ``true_covered`` above is only used for counting, and indexing keeps
        # the scoring path separate from the ground-station belief decision.
        error = int(np.count_nonzero(true_covered)) if np.any(uncovered) else 0
        state["belief_error_cells"] += error

        coords = np.argwhere(uncovered)
        survivors = [d for d in self._DRONES if d != lost and state["alive"][d]]
        if not survivors:
            return
        halves: list[np.ndarray] = []
        if coords.size:
            median_x = float(np.median(coords[:, 1]))
            halves = [uncovered & (np.indices(uncovered.shape)[1] <= median_x),
                      uncovered & (np.indices(uncovered.shape)[1] > median_x)]
            halves = [m for m in halves if np.any(m)]
        if halves:
            positions = []
            for d in survivors:
                tel, _ = ctx.drones[d].telemetry_view(t)
                positions.append(np.asarray(tel.position_m[:2], dtype=float) if tel is not None else np.array([2.0, 10.0]))
            entries = []
            for half in halves:
                ys, xs = np.where(half)
                entries.append(np.array([float(np.min(xs) + 0.5), float(np.median(ys) + 0.5)]))
            cost = np.array([[math.dist(p, e) for e in entries] for p in positions], dtype=float)
            rr, cc = linear_sum_assignment(cost)
            assignments = {survivors[int(r)]: halves[int(c)] for r, c in zip(rr, cc)}
            # If there are fewer halves than survivors, idle survivors remain idle.
            for d, mask in assignments.items():
                ys, _ = np.where(mask)
                rows = sorted(set(int(y) for y in ys))
                selected = []
                last_y = None
                for y in rows:
                    if last_y is None or y - last_y >= 4:
                        selected.append(y)
                        last_y = y
                for lane_index, y in enumerate(selected):
                    xs = np.where(mask[y])[0]
                    if xs.size == 0:
                        continue
                    left, right = float(xs.min() + 0.5), float(xs.max() + 0.5)
                    y_m = float(y + 0.5)
                    if lane_index % 2 == 0:
                        points = ((left, y_m, self._ALTITUDE), (right, y_m, self._ALTITUDE))
                    else:
                        points = ((right, y_m, self._ALTITUDE), (left, y_m, self._ALTITUDE))
                    if state["queues"][d]:
                        previous = state["queues"][d][-1]
                        state["queues"][d].append((previous[0], y_m, self._ALTITUDE))
                    state["queues"][d].extend(points)
        state["replans"] += 1
        ctx.event(t, "replan", receiver, {"lost_peer": lost, "believed_uncovered": int(np.count_nonzero(uncovered)), "belief_error_cells": error, "survivors": len(survivors)})

    def _check_losses(self, ctx, t: float) -> None:
        state = ctx.state
        for receiver in self._DRONES:
            for sender in self._DRONES:
                if sender == receiver or sender in state["declared_lost"][receiver]:
                    continue
                last = state["last_msg"][receiver][sender]
                if last is None or t - float(last) <= self._LOSS_TIMEOUT + 1e-9:
                    continue
                state["declared_lost"][receiver].add(sender)
                if state["alive"][sender]:
                    state["false_loss_declarations"] += 1
                self._make_replan(ctx, sender, receiver, t)

    def _control(self, ctx, t: float) -> None:
        state = ctx.state
        for drone in self._DRONES:
            if not state["alive"][drone]:
                continue
            tel, _ = ctx.drones[drone].telemetry_view(t)
            idx = state["indices"][drone]
            queue = state["queues"][drone]
            if tel is not None and idx < len(queue):
                goal = np.asarray(queue[idx], dtype=float)
                if math.dist(np.asarray(tel.position_m, dtype=float), goal) <= 0.5:
                    idx += 1
                    state["indices"][drone] = idx
            if idx < len(queue):
                goal = queue[idx]
                state["last_goals"][drone] = goal
                ctx.drones[drone].set_goal(goal)
            elif state["last_goals"][drone] is not None:
                ctx.drones[drone].set_goal(state["last_goals"][drone])

    def tick(self, ctx, i: int, t: float):
        state = ctx.state
        self._receive_shares(ctx, t)
        self._check_losses(ctx, t)
        self._control(ctx, t)

        dropout_now = ctx.params["dropout"] == "one" and not state["dropout_done"] and t >= 10.0 - 1e-9
        if dropout_now:
            # Land before the step so the final share at t=10 describes the
            # last horizontal position.  The share is sent below, then the
            # drone is marked inactive for later sends.
            ctx.adapters["d2"].land()
            state["dropout_done"] = True

        result = {}
        for drone in self._DRONES:
            result[drone] = ctx.drones[drone].tick(i)

        self._mark_true(ctx, result)
        self._send_shares(ctx, t)
        if dropout_now and state["alive"]["d2"]:
            state["alive"]["d2"] = False
            ctx.event(t, "drone_dropout", "d2", {"scripted": True})

        union = np.any(state["own"], axis=0)
        fraction = float(np.mean(union))
        state["coverage_history"].append((float(t), fraction))
        if state["t95"] is None and fraction >= 0.95:
            state["t95"] = float(t)
        if state["t100"] is None and fraction >= 1.0:
            state["t100"] = float(t)
        if i % 100 == 0:
            ctx.event(t, "coverage_update", None, {"coverage_fraction": fraction})
        ctx.frame(i, t, None, {d: agent_frame(result[d]) for d in self._DRONES})

    def done(self, ctx):
        if ctx.state.get("t100") is not None:
            return "complete"
        return None

    def metrics(self, ctx):
        state = ctx.state
        counts = state["counts"]
        union = np.any(state["own"], axis=0)
        at60 = 0.0
        for t, fraction in state["coverage_history"]:
            if t <= 60.0 + 1e-9:
                at60 = float(fraction)
            else:
                break
        return {
            "t95_s": None if state["t95"] is None else float(state["t95"]),
            "t100_s": None if state["t100"] is None else float(state["t100"]),
            "duplicated_fraction": float(np.count_nonzero(counts >= 2) / (self._WIDTH * self._HEIGHT)),
            "missed_cells": int(np.count_nonzero(~union)),
            "replans": int(state["replans"]),
            "belief_error_cells": int(state["belief_error_cells"]),
            "false_loss_declarations": int(state["false_loss_declarations"]),
            "coverage_at_60s": float(at60),
        }
