"""Mission command and telemetry plumbing for one simulated drone."""
from __future__ import annotations

import inspect
import math

import numpy as np

try:
    from sim.failsafe import SequenceLossWindow
except ImportError:  # supports running while the failsafe extension is pending
    class SequenceLossWindow:
        def __init__(self, window_packets=20):
            self.window_packets = int(window_packets)
            self._received = set()
            self._newest = None
        def record(self, seq):
            seq = int(seq)
            if self._newest is None or seq > self._newest:
                self._newest = seq
                self._received.add(seq)
        def loss_fraction(self, expected_next_seq=None):
            if self._newest is None:
                return None
            end = self._newest if expected_next_seq is None else int(expected_next_seq) - 1
            start = max(0, end - self.window_packets + 1)
            total = max(0, end - start + 1)
            return None if total == 0 else (total - sum(i in self._received for i in range(start, end + 1))) / total


class LinkedDrone:
    """Connect an adapter to command and telemetry mission links."""

    def __init__(self, adapter, *, cmd_link, tel_link, command_rate_hz: float = 20.0,
                 telemetry_rate_hz: float = 20.0) -> None:
        self.adapter = adapter
        self.cmd_link = cmd_link
        self.tel_link = tel_link
        self.command_rate_hz = float(command_rate_hz)
        self.telemetry_rate_hz = float(telemetry_rate_hz)
        if self.command_rate_hz <= 0 or self.telemetry_rate_hz <= 0:
            raise ValueError("rates must be positive")
        self._goal: np.ndarray | None = None
        self._cmd_seq = 0
        self._last_cmd_receive = 0.0
        self._newest_cmd_seq: int | None = None
        self._loss_window = SequenceLossWindow()
        self._last_cmd_due = -1
        self._last_tel_due = -1
        self._telemetry_cache = None
        if not adapter.connected:
            adapter.connect()

    def set_goal(self, position_m) -> None:
        point = np.asarray(tuple(position_m), dtype=float)
        if point.shape != (3,) or np.any(~np.isfinite(point)):
            raise ValueError("position_m must be a finite 3-vector")
        self._goal = point

    @staticmethod
    def _packet_time(packet, name: str) -> float:
        value = getattr(packet, name, None)
        if value is None and isinstance(packet, dict):
            value = packet.get(name)
        return float(value)

    def tick(self, i: int):
        if int(i) != i or i < 0:
            raise ValueError("i must be a nonnegative integer")
        i = int(i)
        dt = 0.01
        now = i * dt
        if self._goal is not None:
            due = int(math.floor(now * self.command_rate_hz + 1e-9))
            if due > self._last_cmd_due:
                self.cmd_link.send(self._cmd_seq, now, self._goal.copy())
                self._cmd_seq += 1
                self._last_cmd_due = due
        for packet in self.cmd_link.deliveries_until(now):
            message = getattr(packet, "message", None)
            if message is None and isinstance(packet, dict):
                message = packet.get("message")
            if message is not None:
                self.adapter.send_setpoint(np.asarray(message, dtype=float))
            received_at = self._packet_time(packet, "receive_time_s") if hasattr(packet, "receive_time_s") else self._packet_time(packet, "receive_time")
            self._last_cmd_receive = max(self._last_cmd_receive, received_at)
            if isinstance(packet, dict):
                seq = int(packet["seq"])
            else:
                seq = int(packet.seq)
            self._loss_window.record(seq)
            if self._newest_cmd_seq is None or seq > self._newest_cmd_seq:
                self._newest_cmd_seq = seq
        age = max(0.0, now - self._last_cmd_receive)
        step = self.adapter.step
        kwargs = {"link_age_s": age}
        if "recent_loss_fraction" in inspect.signature(step).parameters:
            kwargs["recent_loss_fraction"] = self._loss_window.loss_fraction(self._expected_next_seq(now))
        state = step(dt, **kwargs)
        tel_due = int(math.floor((now + dt) * self.telemetry_rate_hz + 1e-9))
        if tel_due > self._last_tel_due:
            self.tel_link.send(tel_due, now + dt, state)
            self._last_tel_due = tel_due
        return state

    def _expected_next_seq(self, now: float) -> int | None:
        """Sequence number the drone expects next, from what it can know.

        The drone knows the command rate but not the ground station's counter:
        each full command period since the newest arrival counts as one missing
        packet.  A packet still in flight is therefore never counted as lost.
        """
        if self._newest_cmd_seq is None:
            return None
        missed = int(math.floor((now - self._last_cmd_receive) * self.command_rate_hz + 1e-9))
        return self._newest_cmd_seq + 1 + max(0, missed)

    def telemetry_view(self, t: float):
        now = float(t)
        packets = self.tel_link.deliveries_until(now)
        if packets:
            packet = packets[-1]
            message = getattr(packet, "message", None)
            if message is None and isinstance(packet, dict):
                message = packet.get("message")
            sent_at = self._packet_time(packet, "send_time_s") if hasattr(packet, "send_time_s") else self._packet_time(packet, "send_time")
            if self._telemetry_cache is None or sent_at >= self._telemetry_cache[1]:
                self._telemetry_cache = (message, sent_at)
        if self._telemetry_cache is None:
            return None, math.inf
        message, sent = self._telemetry_cache
        return message, now - sent

