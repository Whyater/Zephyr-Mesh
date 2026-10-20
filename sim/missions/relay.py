"""Simulated scenario of a planned use case: relay mission using Run C radio."""
from __future__ import annotations

import math

import numpy as np

from sim.adapters import Geofence
from sim.missions.comms import RadioMissionLink
from sim.missions.engine import Mission, agent_frame, register_mission
from sim.radio.channel import mean_rssi_dbm
from sim.radio.model import RadioLinkConfig
from sim.radio.params import PARAMS, channel_preset, default_mac


def _stress_channel():
    # Matches tools/c_exp017_tolerance.py:42 and :105, including its iid
    # shadowing, dual-slope breakpoint, far exponent, and orientation loss.
    wavelength_m = 299_792_458.0 / 2.437e9
    breakpoint_m = 4.0 * 1.5 * 1.5 / wavelength_m
    return channel_preset(
        "outdoor_open",
        shadowing_sigma_db=PARAMS["sigma_outdoor_open_db"].high,
        breakpoint_m=breakpoint_m,
        exponent_far=4.0,
        orientation_loss_db=PARAMS["orientation_loss_db"].high,
        shadowing_mode="iid",
    )


def _validate_params(params: dict) -> None:
    distance = float(params["distance_m"])
    if isinstance(params["distance_m"], bool) or not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_m must be finite and positive")
    fraction = params["relay_fraction"]
    if isinstance(fraction, bool) or (fraction is not None and float(fraction) not in (0.25, 0.4, 0.5, 0.6, 0.75)):
        raise ValueError("relay_fraction must be null or one of 0.25, 0.4, 0.5, 0.6, 0.75")
    rate = float(params["rate_hz"])
    if isinstance(params["rate_hz"], bool) or not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate_hz must be finite and positive")
    message_bytes = int(params["message_bytes"])
    if isinstance(params["message_bytes"], bool) or message_bytes != params["message_bytes"] or message_bytes < 1:
        raise ValueError("message_bytes must be a positive integer")
    altitude = float(params["altitude_m"])
    if isinstance(params["altitude_m"], bool) or not math.isfinite(altitude) or altitude <= 0.0:
        raise ValueError("altitude_m must be finite and positive")
    retries = int(params["retries"])
    if isinstance(params["retries"], bool) or retries != params["retries"] or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    delay = float(params["forward_delay_s"])
    if isinstance(params["forward_delay_s"], bool) or not math.isfinite(delay) or delay < 0.0:
        raise ValueError("forward_delay_s must be finite and nonnegative")


@register_mission
class RelayMission(Mission):
    name = "relay"
    rng_names = ("hop1", "hop2")
    limits = (
        "Simulated only: the radio channel and retry behavior are model outputs, not measurements.",
        "Radio geometry is fixed at the commanded hover positions; vehicle motion and antenna attitude are not coupled to path loss.",
        "retries_mean is null because RadioMissionLink does not expose per-packet retry counts in its public event records.",
        "Each hop is its own radio model, so the far drone's and the relay's transmissions never compete for airtime (one shared channel would add contention; at 10 Hz and 32 B that is small, Run C EXP-014).",
    )
    defaults = {
        "distance_m": 900.0,
        "relay_fraction": None,
        "rate_hz": 10.0,
        "message_bytes": 32,
        "altitude_m": 30.0,
        "retries": 5,
        "forward_delay_s": 0.001,
    }

    def validate_params(self, params: dict) -> dict:
        merged = super().validate_params(params)
        _validate_params(merged)
        return merged

    def setup(self, ctx):
        _validate_params(ctx.params)
        distance = float(ctx.params["distance_m"])
        altitude = float(ctx.params["altitude_m"])
        fraction = ctx.params["relay_fraction"]
        relay_x = 0.5 * distance if fraction is None else float(fraction) * distance
        positions = {
            "gs": (0.0, 0.0, altitude),
            "far": (distance, 0.0, altitude),
            "relay": (relay_x, 0.0, altitude),
        }
        geofence = Geofence(
            min_m=(-100.0, -100.0, 0.0),
            max_m=(distance + 100.0, 100.0, altitude + 100.0),
        )
        for agent_id in ("far", "relay"):
            drone = ctx.add_drone(
                agent_id,
                initial_position_m=positions[agent_id],
                geofence=geofence,
            )
            drone.set_goal(positions[agent_id])
            # Telemetry for this mission is emitted below through the Run C
            # radio links at the registered 10 Hz rate.
            drone._last_tel_due = 10**9
        ctx.start_log(["far", "relay"])

        channel = _stress_channel()
        radio_cfg = RadioLinkConfig(
            positions=positions,
            channel=channel,
            mac=default_mac(unicast_retry_limit=int(ctx.params["retries"])),
            payload_bytes=int(ctx.params["message_bytes"]),
            unicast=True,
        )
        relay_mode = fraction is not None
        if relay_mode:
            hop1_sender, hop1_receiver = "far", "relay"
        else:
            hop1_sender, hop1_receiver = "far", "gs"
        hop1 = RadioMissionLink(
            "radio:hop1",
            radio_config=radio_cfg,
            seed=ctx.seed_for("hop1"),
            sender_id=hop1_sender,
            receiver_id=hop1_receiver,
        )
        hop2 = None
        if relay_mode:
            hop2 = RadioMissionLink(
                "radio:hop2",
                radio_config=radio_cfg,
                seed=ctx.seed_for("hop2"),
                sender_id="relay",
                receiver_id="gs",
            )
        # Expose the radio links to the engine so their counters are included
        # in mission logs alongside the command links.
        # Bind the far drone's telemetry endpoint to the radio link used by
        # the experiment.  The mission emits the 10 Hz frames explicitly so
        # forwarding can preserve the original send timestamp.
        ctx.drones["far"].tel_link = hop1
        ctx.links["tel:far"] = hop1
        if hop2 is not None:
            ctx.links["tel:relay"] = hop2
        ctx.state.update(
            {
                "radio": {"hop1": hop1, "hop2": hop2, "channel": channel, "positions": positions},
                "relay_mode": relay_mode,
                "next_seq": 0,
                "next_send_time": 0.0,
                "send_times": {},
                "forward_queue": [],
                "latencies": [],
                "hop1_received": 0,
                "hop2_received": 0,
                "sent": 0,
                "states": {},
                "next_radio_process": 0.0,
            }
        )

    @staticmethod
    def _process_radio(ctx, now: float) -> None:
        state = ctx.state
        hop1 = state["radio"]["hop1"]
        hop2 = state["radio"]["hop2"]
        for packet in hop1.deliveries_until(now):
            state["hop1_received"] += 1
            if state["relay_mode"]:
                state["forward_queue"].append(
                    (float(packet.receive_time_s) + float(ctx.params["forward_delay_s"]), packet.seq)
                )
            else:
                sent_at = state["send_times"].get(packet.seq)
                if sent_at is not None:
                    state["latencies"].append(float(packet.receive_time_s) - sent_at)
        if hop2 is not None:
            state["forward_queue"].sort(key=lambda item: (item[0], item[1]))
            ready = []
            while state["forward_queue"] and state["forward_queue"][0][0] <= now + 1e-12:
                ready.append(state["forward_queue"].pop(0))
            for send_at, seq in ready:
                hop2.send(seq, send_at, None)
            for packet in hop2.deliveries_until(now):
                state["hop2_received"] += 1
                sent_at = state["send_times"].get(packet.seq)
                if sent_at is not None:
                    state["latencies"].append(float(packet.receive_time_s) - sent_at)

    def tick(self, ctx, i: int, t: float) -> None:
        states = {}
        for agent_id in ("far", "relay"):
            states[agent_id] = ctx.drones[agent_id].tick(i)
        state = ctx.state
        period = 1.0 / float(ctx.params["rate_hz"])
        due = int(math.floor(float(t) / period + 1e-9))
        while state["next_seq"] <= due and state["next_send_time"] <= ctx.scenario.duration_s + 1e-12:
            seq = int(state["next_seq"])
            send_at = float(state["next_send_time"])
            state["radio"]["hop1"].send(seq, send_at, states["far"])
            state["send_times"][seq] = send_at
            state["next_seq"] += 1
            state["next_send_time"] = state["next_seq"] * period
            state["sent"] += 1
        if t + 1e-12 >= state["next_radio_process"]:
            self._process_radio(ctx, t)
            state["next_radio_process"] += period
        state["states"] = states
        ctx.frame(i, t, None, {agent_id: agent_frame(value) for agent_id, value in states.items()})

    def metrics(self, ctx):
        # Drain packets already sent at the mission boundary.  A short fixed
        # tail is enough for the configured MAC retry and forwarding delay.
        end = float(ctx.scenario.duration_s)
        step = max(ctx.dt, 1.0 / float(ctx.params["rate_hz"]))
        now = end + step
        while now <= end + 2.0 + 1e-12:
            self._process_radio(ctx, now)
            now += step
        state = ctx.state
        sent = int(state["sent"])
        latencies = [float(v) for v in state["latencies"]]
        delivery = len(latencies) / sent if sent else 0.0
        relay_mode = bool(state["relay_mode"])
        hop1 = state["radio"]["hop1"]
        hop2 = state["radio"]["hop2"]
        hop1_stats = hop1.stats()
        hop2_stats = hop2.stats() if hop2 is not None else None
        distance = float(ctx.params["distance_m"])
        fraction = ctx.params["relay_fraction"]
        longest = distance if fraction is None else max(float(fraction) * distance, (1.0 - float(fraction)) * distance)
        rx_dbm = mean_rssi_dbm(longest, state["radio"]["channel"])
        return {
            "delivery_ratio": float(delivery),
            "median_latency_s": float(np.median(latencies)) if latencies else None,
            "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else None,
            "hop1_delivery": float(hop1_stats["delivered"] / hop1_stats["sent"]) if relay_mode and hop1_stats["sent"] else None,
            "hop2_delivery": float(hop2_stats["delivered"] / hop2_stats["sent"]) if relay_mode and hop2_stats and hop2_stats["sent"] else None,
            "retries_mean": None,
            "link_budget_rx_dbm": float(rx_dbm),
        }
