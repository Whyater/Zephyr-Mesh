"""Simulated hover-and-move mission: the smallest end-to-end use of the adapter seam.

One drone takes a goal over the command link, flies there and holds.  It is the
example a student reads first, and the engine's own test mission.  Simulated
only.
"""
from __future__ import annotations

import math

from sim.missions.engine import Mission, agent_frame, register_mission


@register_mission
class HoverMission(Mission):
    name = "hover"
    rng_names = ()
    limits = ("One drone, no target; checks the command and telemetry links and the failsafe.",)
    defaults = {"start_m": [2.0, 2.0, 1.5], "goal_m": [3.0, 2.0, 1.5]}

    def setup(self, ctx):
        ctx.add_drone("d1", initial_position_m=ctx.params["start_m"], geofence=None)
        ctx.start_log(["d1"])
        ctx.drones["d1"].set_goal(ctx.params["goal_m"])
        ctx.state["max_link_age_s"] = 0.0

    def tick(self, ctx, i, t):
        state = ctx.drones["d1"].tick(i)
        ctx.state["last"] = state
        if state.link_age_s is not None:
            ctx.state["max_link_age_s"] = max(ctx.state["max_link_age_s"], state.link_age_s)
        ctx.frame(i, t, None, {"d1": agent_frame(state)})

    def metrics(self, ctx):
        state = ctx.state["last"]
        goal = ctx.params["goal_m"]
        error = math.dist([float(v) for v in state.position_m], goal)
        return {"final_error_m": error, "final_failsafe_state": state.failsafe_state,
                "max_link_age_s": ctx.state["max_link_age_s"]}
