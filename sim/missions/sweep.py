"""Simulated scenario of a planned use case: area sweep with shared coverage (EXP-031).

Three drones sweep an outdoor field 40 x 20 m at 10 m altitude and 2 m/s.  The
field is cut into three equal strips along x (each 13.33 m wide and the full
20 m deep).  Each strip is flown as a lawnmower: lanes run along x, 4 m apart,
the first lane centre 2 m inside the strip's near edge, so five lanes with a
2 m half swath tile the 20 m depth exactly.  Coverage is scored on a 1 m grid:
a cell counts as seen by a drone when the cell centre is within 2 m of that
drone's true horizontal position.

Each drone shares its coverage at 5 Hz with every other drone and with the
ground station, one MissionLink stream per directed pair
("share:<sender>><receiver>").  Messages are either "delta" (cells newly seen
since the previous message) or "cumulative" (the sender's whole map).  With
dropout "one", drone d2 lands at t = 10 s (a scripted fault through its
adapter's land(), not a failsafe).  A drone that hears nothing from a peer for
2.0 s declares it lost, splits the cells of the lost peer's strip that it
BELIEVES uncovered into two halves at the strip's x centre line, and the
survivors are matched to halves with scipy's linear_sum_assignment on
distance.  Each survivor plans on its own belief and only changes its own
route.  A living peer can be declared lost by mistake under loss; that is
counted, and the strip is re-planned anyway.

Planning uses only what has arrived: the drone's own coverage map (its own
sensor), delivered share messages and delivered telemetry.  True coverage is
used for scoring only.  Everything here is simulated.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from sim.adapters import Geofence
from sim.missions.engine import Mission, agent_frame, register_mission

FIELD_X_M = 40
FIELD_Y_M = 20
ALTITUDE_M = 10.0
SPEED_MPS = 2.0
HALF_SWATH_M = 2.0
LANE_SPACING_M = 4.0
N_STRIPS = 3
GEOFENCE_MARGIN_M = 5.0
SHARE_HZ = 5.0
LOSS_TIMEOUT_S = 2.0
DROPOUT_TIME_S = 10.0
DROPOUT_DRONE = "d2"
GROUND = "ground"

# Lane ends sit 1.05 m inside the outermost cell centre of the lane's band.
# A cell centre in the outer row of a band is 1.5 m off the lane, so the 2 m
# disc reaches it while the drone is within sqrt(2^2 - 1.5^2) = 1.32 m of it
# along the lane.  The drone moves on once it is within WAYPOINT_ACCEPT_M of
# the lane end, so it always gets to within 1.05 + 0.2 = 1.25 m (< 1.32 m).
# The next column centre beyond the band is 1 + 1.05 = 2.05 m away (> 2 m),
# so a lane end never spills into the neighbouring strip.
LANE_END_INSET_M = 1.05
WAYPOINT_ACCEPT_M = 0.2

_COLS = np.arange(FIELD_X_M, dtype=float) + 0.5   # cell centre x
_ROWS = np.arange(FIELD_Y_M, dtype=float) + 0.5   # cell centre y


def _strip_edges() -> np.ndarray:
    return np.linspace(0.0, float(FIELD_X_M), N_STRIPS + 1)


def strip_mask(k: int) -> np.ndarray:
    """Cells of strip k (rows = y, columns = x), assigned by cell centre."""
    edges = _strip_edges()
    in_strip = (_COLS >= edges[k]) & (_COLS < edges[k + 1])
    return np.broadcast_to(in_strip, (FIELD_Y_M, FIELD_X_M)).copy()


def lane_route(cells: np.ndarray, start_xy) -> list[tuple[float, float, float]]:
    """Lawnmower waypoints that sweep every cell in ``cells``.

    Rows are grouped in bands of four (one 4 m swath); the lane centre is 2 m
    above the band's lowest row edge (clamped so it stays 2 m inside the
    field).  Each lane runs between the band's outermost cell centres, pulled
    in by LANE_END_INSET_M.  Bands are flown from the end nearer
    ``start_xy``, and each lane starts from the end nearer the previous lane's
    end.
    """
    rows_with_cells = np.flatnonzero(cells.any(axis=1))
    lanes: list[tuple[float, float, float]] = []   # (y, x_low, x_high)
    k = 0
    while k < rows_with_cells.size:
        first = int(rows_with_cells[k])
        band = cells[first:first + 4]
        cols = np.flatnonzero(band.any(axis=0))
        lane_y = min(first + HALF_SWATH_M, FIELD_Y_M - HALF_SWATH_M)
        x_low = float(_COLS[cols[0]]) + LANE_END_INSET_M
        x_high = float(_COLS[cols[-1]]) - LANE_END_INSET_M
        if x_high < x_low:
            x_low = x_high = 0.5 * (x_low + x_high)
        lanes.append((float(lane_y), x_low, x_high))
        while k < rows_with_cells.size and rows_with_cells[k] < first + 4:
            k += 1
    if not lanes:
        return []
    x0, y0 = float(start_xy[0]), float(start_xy[1])
    if abs(lanes[-1][0] - y0) < abs(lanes[0][0] - y0):
        lanes.reverse()
    route: list[tuple[float, float, float]] = []
    cx = x0
    for lane_y, x_low, x_high in lanes:
        if abs(x_high - cx) < abs(x_low - cx):
            ends = (x_high, x_low)
        else:
            ends = (x_low, x_high)
        route.append((ends[0], lane_y, ALTITUDE_M))
        route.append((ends[1], lane_y, ALTITUDE_M))
        cx = ends[1]
    return route


@register_mission
class SweepMission(Mission):
    name = "sweep"
    rng_names = ()
    limits = (
        "Simulated only: the camera footprint is a 2 m radius disc scored on a 1 m grid at fixed altitude.",
        "Each survivor re-plans on its own belief; no consensus step between survivors.",
        "Lost-peer declaration is a 2.0 s silence rule; a declared peer is never reinstated.",
    )
    defaults = {"message_mode": "delta", "dropout": "none"}

    _DRONES = ("d1", "d2", "d3")

    def validate_params(self, params: dict) -> dict:
        merged = super().validate_params(params)
        if merged["message_mode"] not in {"delta", "cumulative"}:
            raise ValueError("message_mode must be 'delta' or 'cumulative'")
        if merged["dropout"] not in {"none", "one"}:
            raise ValueError("dropout must be 'none' or 'one'")
        return merged

    # setup ----------------------------------------------------------------
    def setup(self, ctx):
        drones = self._DRONES
        masks = [strip_mask(k) for k in range(N_STRIPS)]
        # Each drone starts above the first lane start of one strip (drone k at
        # strip k), so the assignment below is the identity; it still goes
        # through the solver, as the design specifies.
        entries = [lane_route(masks[k], (0.0, 0.0))[0] for k in range(N_STRIPS)]
        starts = list(entries)
        cost = np.array([[math.dist(starts[i][:2], entries[j][:2]) for j in range(N_STRIPS)]
                         for i in range(len(drones))])
        rows, cols = linear_sum_assignment(cost)
        strip_of = {drones[int(r)]: int(c) for r, c in zip(rows, cols)}

        geofence = Geofence((-GEOFENCE_MARGIN_M, -GEOFENCE_MARGIN_M, 0.0),
                            (FIELD_X_M + GEOFENCE_MARGIN_M, FIELD_Y_M + GEOFENCE_MARGIN_M, 2.0 * ALTITUDE_M),
                            margin_m=0.5)
        for k, drone in enumerate(drones):
            ctx.add_drone(drone, initial_position_m=starts[k], geofence=geofence, max_speed_mps=SPEED_MPS)
        for sender in drones:
            for receiver in (*drones, GROUND):
                if receiver != sender:
                    ctx.link(f"share:{sender}>{receiver}")

        shape = (FIELD_Y_M, FIELD_X_M)
        ctx.state.update({
            "mode": ctx.params["message_mode"],
            "strip_masks": masks,
            "strip_of": strip_of,
            "routes": {d: lane_route(masks[strip_of[d]], starts[k][:2]) for k, d in enumerate(drones)},
            "waypoint": {d: 0 for d in drones},
            "active": {d: True for d in drones},
            # Scoring truth: each drone's true footprint map and per-cell drone counts.
            "own": {d: np.zeros(shape, dtype=bool) for d in drones},
            "counts": np.zeros(shape, dtype=np.int16),
            "covered": 0,
            # Planning side: what each drone believes, from its own map and shares.
            "belief": {d: np.zeros(shape, dtype=bool) for d in drones},
            # The ground station's map, built from its share streams (operator
            # view only; the drones never read it).
            "ground_belief": np.zeros(shape, dtype=bool),
            "last_sent": {d: np.zeros(shape, dtype=bool) for d in drones},
            "share_seq": {d: 0 for d in drones},
            "share_due": {d: -1 for d in drones},
            "heard_at": {r: {s: 0.0 for s in drones if s != r} for r in drones},
            "peer_xy": {r: {s: None for s in drones if s != r} for r in drones},
            "declared": {r: [] for r in drones},
            "replans": 0,
            "belief_error_cells": 0,
            "false_loss_declarations": 0,
            "t95": None,
            "t100": None,
        })
        ctx.start_log(drones)

    # planning helpers (delivered data only) ---------------------------------
    def _planner_xy(self, ctx, drone: str, t: float):
        tel, _ = ctx.drones[drone].telemetry_view(t)
        if tel is None:
            return None
        return (float(tel.position_m[0]), float(tel.position_m[1]))

    def _receive_shares(self, ctx, t: float) -> None:
        state = ctx.state
        for sender in self._DRONES:
            for receiver in (*self._DRONES, GROUND):
                if receiver == sender:
                    continue
                packets = ctx.links[f"share:{sender}>{receiver}"].deliveries_until(t)
                if not packets:
                    continue
                belief = state["ground_belief"] if receiver == GROUND else state["belief"][receiver]
                for packet in packets:
                    message = packet.message
                    if message["mode"] == "cumulative":
                        belief |= message["coverage"]
                    else:
                        belief.reshape(-1)[message["cells"]] = True
                    if receiver != GROUND:
                        state["heard_at"][receiver][sender] = max(state["heard_at"][receiver][sender],
                                                                  float(packet.receive_time_s))
                        if message["position_xy"] is not None:
                            state["peer_xy"][receiver][sender] = message["position_xy"]

    def _check_losses(self, ctx, t: float) -> None:
        state = ctx.state
        for receiver in self._DRONES:
            if not state["active"][receiver]:
                continue   # a landed drone runs no mission logic
            for sender in self._DRONES:
                if sender == receiver or sender in state["declared"][receiver]:
                    continue
                if t - state["heard_at"][receiver][sender] <= LOSS_TIMEOUT_S + 1e-9:
                    continue
                state["declared"][receiver].append(sender)
                false_declaration = state["active"][sender]   # scoring only
                if false_declaration:
                    state["false_loss_declarations"] += 1
                self._replan(ctx, receiver, sender, t, false_declaration)

    def _replan(self, ctx, receiver: str, lost: str, t: float, false_declaration: bool) -> None:
        state = ctx.state
        k = state["strip_of"][lost]
        strip = state["strip_masks"][k]
        uncovered = strip & ~state["belief"][receiver]
        # Scoring only: believed uncovered but truly seen by some drone.
        error = int(np.count_nonzero(uncovered & (state["counts"] > 0)))
        state["belief_error_cells"] += error

        survivors = sorted(d for d in self._DRONES if d != lost and d not in state["declared"][receiver])
        edges = _strip_edges()
        centre_x = 0.5 * (edges[k] + edges[k + 1])
        left_cols = _COLS < centre_x
        halves = [uncovered & left_cols, uncovered & ~left_cols]
        own_xy = self._planner_xy(ctx, receiver, t)
        if own_xy is None:
            own_xy = state["routes"][receiver][0][:2]
        if len(survivors) == 1:
            mine = uncovered
        else:
            positions = []
            for d in survivors:
                if d == receiver:
                    positions.append(own_xy)
                elif state["peer_xy"][receiver][d] is not None:
                    positions.append(state["peer_xy"][receiver][d])
                else:   # never heard: fall back to the peer's planned strip start
                    positions.append(state["routes"][d][0][:2])
            centroids = []
            for half in halves:
                if np.any(half):
                    ys, xs = np.nonzero(half)
                    centroids.append((float(np.mean(_COLS[xs])), float(np.mean(_ROWS[ys]))))
                else:
                    centroids.append((centre_x, 0.5 * FIELD_Y_M))
            cost = np.array([[math.dist(p, c) for c in centroids] for p in positions])
            rows, cols = linear_sum_assignment(cost)
            pick = {survivors[int(r)]: int(c) for r, c in zip(rows, cols)}
            mine = halves[pick[receiver]]
        route = state["routes"][receiver]
        # New lanes go after the survivor's remaining route, so plan them from
        # where that route ends (or from here if it is already done).
        start_xy = route[-1][:2] if state["waypoint"][receiver] < len(route) else own_xy
        route.extend(lane_route(mine, start_xy))
        state["replans"] += 1
        ctx.event(t, "replan", receiver, {
            "lost_peer": lost,
            "false_declaration": bool(false_declaration),
            "believed_uncovered": int(np.count_nonzero(uncovered)),
            "assigned_cells": int(np.count_nonzero(mine)),
            "belief_error_cells": error,
            "survivors": list(survivors),
        })

    def _control(self, ctx, t: float) -> None:
        state = ctx.state
        for drone in self._DRONES:
            if not state["active"][drone]:
                continue
            route = state["routes"][drone]
            idx = state["waypoint"][drone]
            xy = self._planner_xy(ctx, drone, t)
            if xy is not None:
                while idx < len(route) and math.dist(xy, route[idx][:2]) <= WAYPOINT_ACCEPT_M:
                    idx += 1
                state["waypoint"][drone] = idx
            if idx < len(route):
                ctx.drones[drone].set_goal(route[idx])
            # With the route done, the last goal stays in force (hover).

    # truth scoring ------------------------------------------------------------
    def _mark_coverage(self, ctx, drone: str, position_m) -> None:
        state = ctx.state
        x, y = float(position_m[0]), float(position_m[1])
        c0 = max(0, int(math.ceil(x - HALF_SWATH_M - 0.5)))
        c1 = min(FIELD_X_M - 1, int(math.floor(x + HALF_SWATH_M - 0.5)))
        r0 = max(0, int(math.ceil(y - HALF_SWATH_M - 0.5)))
        r1 = min(FIELD_Y_M - 1, int(math.floor(y + HALF_SWATH_M - 0.5)))
        if c0 > c1 or r0 > r1:
            return
        dx = _COLS[c0:c1 + 1] - x
        dy = _ROWS[r0:r1 + 1] - y
        # The tiny tolerance keeps a cell centre exactly on the 2 m circle.
        inside = (dy[:, None] ** 2 + dx[None, :] ** 2) <= HALF_SWATH_M ** 2 + 1e-9
        own = state["own"][drone][r0:r1 + 1, c0:c1 + 1]
        newly = inside & ~own
        if not newly.any():
            return
        own |= newly
        counts = state["counts"][r0:r1 + 1, c0:c1 + 1]
        state["covered"] += int(np.count_nonzero(newly & (counts == 0)))
        counts[newly] += 1
        # The drone's own sensor map is part of its own belief.
        state["belief"][drone][r0:r1 + 1, c0:c1 + 1] |= newly

    # sharing ------------------------------------------------------------------
    def _send_shares(self, ctx, t: float) -> None:
        state = ctx.state
        due = int(math.floor(t * SHARE_HZ + 1e-9))
        for sender in self._DRONES:
            if not state["active"][sender] or due <= state["share_due"][sender]:
                continue
            own = state["own"][sender]
            position_xy = self._planner_xy(ctx, sender, t)
            if state["mode"] == "cumulative":
                message = {"mode": "cumulative", "coverage": own.copy(), "position_xy": position_xy}
            else:
                cells = np.flatnonzero(own & ~state["last_sent"][sender])
                message = {"mode": "delta", "cells": cells, "position_xy": position_xy}
                state["last_sent"][sender] = own.copy()
            seq = state["share_seq"][sender]
            for receiver in (*self._DRONES, GROUND):
                if receiver != sender:
                    ctx.links[f"share:{sender}>{receiver}"].send(seq, t, message)
            state["share_seq"][sender] = seq + 1
            state["share_due"][sender] = due

    # main loop ----------------------------------------------------------------
    def tick(self, ctx, i, t):
        state = ctx.state
        self._receive_shares(ctx, t)
        self._check_losses(ctx, t)
        self._control(ctx, t)

        dropout_now = (ctx.params["dropout"] == "one" and state["active"][DROPOUT_DRONE]
                       and t >= DROPOUT_TIME_S - 1e-9)
        if dropout_now:
            ctx.adapters[DROPOUT_DRONE].land()

        result = {d: ctx.drones[d].tick(i) for d in self._DRONES}
        for drone in self._DRONES:
            if state["active"][drone]:
                self._mark_coverage(ctx, drone, result[drone].position_m)
        # The dropout drone still sends the share due at the dropout step, then
        # goes silent.
        self._send_shares(ctx, t)
        if dropout_now:
            state["active"][DROPOUT_DRONE] = False
            ctx.event(t, "drone_dropout", DROPOUT_DRONE, {"scripted": True, "action": "land"})

        total = FIELD_X_M * FIELD_Y_M
        if state["t95"] is None and state["covered"] >= 0.95 * total:
            state["t95"] = round(t, 6)
        if state["t100"] is None and state["covered"] >= total:
            state["t100"] = round(t, 6)
        if i % 100 == 0:
            ctx.event(t, "coverage_update", None, {"coverage_fraction": state["covered"] / total})
        ctx.frame(i, t, None, {d: agent_frame(result[d]) for d in self._DRONES})

    def done(self, ctx):
        return "complete" if ctx.state["t100"] is not None else None

    def metrics(self, ctx):
        state = ctx.state
        total = FIELD_X_M * FIELD_Y_M
        # Failsafe landings: transitions to the landing state, except the
        # scripted dropout drone's own landing at or after the dropout time.
        dropout_t = DROPOUT_TIME_S if ctx.params["dropout"] == "one" else None
        landings = sum(
            1 for event in ctx.events
            if event["kind"] == "failsafe_transition"
            and event["detail"].get("state") == "landing"
            and not (dropout_t is not None and event["agent"] == DROPOUT_DRONE
                     and event["time_s"] >= dropout_t - 1e-9)
        )
        return {
            "t95_s": None if state["t95"] is None else float(state["t95"]),
            "t100_s": None if state["t100"] is None else float(state["t100"]),
            "duplicated_fraction": float(np.count_nonzero(state["counts"] >= 2) / total),
            "missed_cells": int(total - state["covered"]),
            "replans": int(state["replans"]),
            "belief_error_cells": int(state["belief_error_cells"]),
            "false_loss_declarations": int(state["false_loss_declarations"]),
            "failsafe_landings": int(landings),
        }
