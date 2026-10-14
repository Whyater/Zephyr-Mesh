"""Sequence number tracking for the simulated radio receiver.

The old uint8 counter wraps every 256 packets, so a wrapped value can look
like a duplicate or an old packet.  A 32-bit serial counter, together with a
sliding window and RFC 1982 serial arithmetic, distinguishes a recent packet
from a packet that is behind the window.  At 50 Hz a 32-bit counter takes
about 2.7 years to wrap.  This module only tracks ordering and replay state;
it does not authenticate packets.
"""

from dataclasses import dataclass
import operator


SEQ_BITS = 32
SEQ_MOD = 1 << SEQ_BITS
_HALF_MOD = 1 << (SEQ_BITS - 1)


def _validate_seq(seq: int) -> int:
    if isinstance(seq, bool) or not hasattr(seq, "__index__"):
        raise ValueError("seq must be an integer in [0, 2**32)")
    seq = operator.index(seq)
    if not 0 <= seq < SEQ_MOD:
        raise ValueError("seq must be an integer in [0, 2**32)")
    return seq


def seq_next(s: int) -> int:
    """Return the next value in the 32-bit serial number space."""

    _validate_seq(s)
    return (s + 1) % SEQ_MOD


def seq_diff(a: int, b: int) -> int:
    """Return signed serial difference ``a - b`` using RFC 1982 arithmetic."""

    _validate_seq(a)
    _validate_seq(b)
    return ((a - b + _HALF_MOD) % SEQ_MOD) - _HALF_MOD


@dataclass(frozen=True)
class Observation:
    """The classification and extended sequence number of one observation."""

    status: str
    unwrapped: int
    gap: int
    accept: bool


class SequenceTracker:
    """Track per-sender sequence numbers with a sliding anti-replay window.

    ``window`` is the number of sequence positions retained behind the
    highest value.  A packet in that range is accepted once, even if it is
    reordered.  Values outside it are stale, and values already marked in the
    bitmap are duplicates.  ``counts["gap_packets"]`` counts packets skipped
    when a newer one arrives; a skipped packet that turns up later as
    "reordered" is not subtracted, so it is an upper bound on loss.
    """

    def __init__(self, window: int = 64):
        if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window < _HALF_MOD:
            raise ValueError("window must be an integer from 1 to 2**31 - 1")
        self.window = window
        self.counts: dict[str, int] = {
            "new": 0,
            "duplicate": 0,
            "reordered": 0,
            "stale": 0,
            "gap_packets": 0,
        }
        # sender -> [highest serial value, highest unwrapped value, bitmap]
        self._senders: dict[str, list[int]] = {}
        self._mask = (1 << window) - 1

    def observe(self, sender: str, seq: int) -> Observation:
        """Classify one packet and update the sender's receive window."""

        seq = _validate_seq(seq)
        state = self._senders.get(sender)
        if state is None:
            state = [seq, seq, 1]
            self._senders[sender] = state
            self.counts["new"] += 1
            return Observation("new", seq, 0, True)

        highest_seq, highest_unwrapped, bitmap = state
        delta = seq_diff(seq, highest_seq)
        if delta > 0:
            unwrapped = highest_unwrapped + delta
            gap = delta - 1
            if delta >= self.window:
                bitmap = 1
            else:
                bitmap = ((bitmap << delta) | 1) & self._mask
            state[0] = seq
            state[1] = unwrapped
            state[2] = bitmap
            self.counts["new"] += 1
            self.counts["gap_packets"] += gap
            return Observation("new", unwrapped, gap, True)

        unwrapped = highest_unwrapped + delta
        if delta == 0:
            self.counts["duplicate"] += 1
            return Observation("duplicate", unwrapped, 0, False)

        distance = -delta
        if distance >= self.window:
            self.counts["stale"] += 1
            return Observation("stale", unwrapped, 0, False)

        bit = 1 << distance
        if bitmap & bit:
            self.counts["duplicate"] += 1
            return Observation("duplicate", unwrapped, 0, False)

        state[2] = bitmap | bit
        self.counts["reordered"] += 1
        return Observation("reordered", unwrapped, 0, True)

