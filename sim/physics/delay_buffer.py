"""Deterministic fixed-delay, Bernoulli-loss sample channels.

The channel is deliberately small.  It models a packet stream sampled at a
fixed period, with a constant transport delay and independent packet loss.
The random draw is made for every offered packet, including when the loss
probability is zero.  This keeps common-random-number sweeps aligned.
"""

from __future__ import annotations

import copy

import numpy as np


_TIME_TOLERANCE_S = 1e-9


def _copy_value(value):
    """Return a detached value while retaining scalar and tuple types."""
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


class DelayLossChannel:
    """A sampled channel with fixed delay, independent loss, and hold-last.

    ``offer`` is expected at ``k * sample_period_s`` for increasing integer
    ``k``.  The optional ``initial_value`` is returned until the first packet
    that has both survived and arrived.  Values are copied when stored and
    returned so callers cannot mutate the held packet in the channel.
    """

    def __init__(
        self,
        delay_s: float,
        loss_p: float,
        sample_period_s: float,
        rng: np.random.Generator,
        initial_value=None,
    ) -> None:
        self.delay_s = float(delay_s)
        self.loss_p = float(loss_p)
        self.sample_period_s = float(sample_period_s)
        if not np.isfinite(self.delay_s) or self.delay_s < 0.0:
            raise ValueError("delay_s must be finite and non-negative")
        if not np.isfinite(self.loss_p) or not 0.0 <= self.loss_p <= 1.0:
            raise ValueError("loss_p must be finite and in [0, 1]")
        if not np.isfinite(self.sample_period_s) or self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be finite and positive")
        if not hasattr(rng, "random"):
            raise TypeError("rng must provide random()")
        self.rng = rng
        self.initial_value = _copy_value(initial_value)

        # Packets are kept in send-index order.  Integer indices avoid the
        # accumulation of binary floating-point error in k * sample_period.
        self._packets: list[tuple[int, float, object]] = []
        self._next_packet = 0
        self._latest_index: int | None = None
        self._latest_value = _copy_value(initial_value)
        self._last_query_s: float | None = None
        self._last_send_index: int | None = None
        self.offered_count = 0
        self.accepted_count = 0
        self.dropped_count = 0
        self._drop_indices: list[int] = []

    @property
    def drop_indices(self) -> tuple[int, ...]:
        """Indices of packets dropped so far, useful for CRN diagnostics."""
        return tuple(self._drop_indices)

    def _sample_index(self, t_send: float) -> int:
        t_send = float(t_send)
        if not np.isfinite(t_send):
            raise ValueError("t_send must be finite")
        index = int(round(t_send / self.sample_period_s))
        if abs(t_send - index * self.sample_period_s) > _TIME_TOLERANCE_S:
            raise ValueError("t_send must be on the sample grid")
        if self._last_send_index is not None and index <= self._last_send_index:
            raise ValueError("offers must use strictly increasing sample times")
        return index

    def offer(self, t_send: float, value) -> bool:
        """Offer one packet and return ``True`` when it survives the loss.

        One random draw is consumed for every call, even at ``loss_p == 0``.
        The return value is only a convenience; channel consumers should use
        :meth:`latest` to obtain the held value.
        """
        index = self._sample_index(t_send)
        # Deliberately draw unconditionally.  This is what makes loss levels
        # share the same random uniforms in a sweep.
        uniform = float(self.rng.random())
        dropped = uniform < self.loss_p
        self.offered_count += 1
        self._last_send_index = index
        if dropped:
            self.dropped_count += 1
            self._drop_indices.append(index)
            return False

        arrival_s = index * self.sample_period_s + self.delay_s
        self._packets.append((index, arrival_s, _copy_value(value)))
        self.accepted_count += 1
        return True

    def _rewind(self) -> None:
        """Recompute the held value for a non-monotonic time query."""
        self._next_packet = 0
        self._latest_index = None
        self._latest_value = _copy_value(self.initial_value)

    def latest(self, t_now: float):
        """Return the newest sent packet that has arrived by ``t_now``.

        A one-nanosecond tolerance handles the representation of times such
        as ``0.3`` that arise from repeated decimal additions.
        """
        t_now = float(t_now)
        if not np.isfinite(t_now):
            raise ValueError("t_now must be finite")
        if self._last_query_s is not None and t_now < self._last_query_s - _TIME_TOLERANCE_S:
            self._rewind()
        self._last_query_s = t_now

        while self._next_packet < len(self._packets):
            index, arrival_s, value = self._packets[self._next_packet]
            if arrival_s > t_now + _TIME_TOLERANCE_S:
                break
            self._latest_index = index
            self._latest_value = value
            self._next_packet += 1
        return _copy_value(self._latest_value)


__all__ = ["DelayLossChannel"]
