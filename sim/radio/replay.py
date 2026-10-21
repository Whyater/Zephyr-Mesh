"""Receiver-side replay and freshness checks for the drone simulator.

Name warning: "replay" here means a replay ATTACK, where someone re-sends an
old packet hoping the receiver accepts it again. This module is the guard that
rejects such stale or repeated frames. It does not replay a recorded ESP-NOW
capture. Replaying a captured trace packet by packet is
``sim.link.replay_trace``; reading an ESP32 echo log is
``tools/import_espnow_log.py``; fitting link parameters from a log is
``tools/fit_espnow_log.py``. (Renaming this module is a separate proposal.)

This module simulates a receiver-side check. It does not provide authenticity:
a party that can forge sequence numbers can defeat this check without a MAC or
encryption. ESP-NOW's own encryption is a separate topic covered in the
experiment notes. The check is defensive and simulation-only.
"""

from __future__ import annotations

import math
from typing import Dict, Set, Tuple

from .seq import SequenceTracker, _validate_seq


class ReplayGuard:
    """Track sequence windows independently for each sender session.

    A session identifier change starts a fresh sequence window. The immediately
    replaced identifier, and every identifier replaced before it, remain
    rejected as ``old_session``. A frame older than ``max_age_s`` is rejected
    before it can change session or sequence state.
    """

    def __init__(self, window: int = 64, max_age_s: float | None = None):
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")
        if max_age_s is not None:
            if isinstance(max_age_s, bool) or not isinstance(max_age_s, (int, float)):
                raise ValueError("max_age_s must be finite and nonnegative")
            if not math.isfinite(float(max_age_s)) or max_age_s < 0:
                raise ValueError("max_age_s must be finite and nonnegative")
            max_age_s = float(max_age_s)
        self.window = window
        self.max_age_s = max_age_s
        self._sessions: Dict[str, Tuple[str, SequenceTracker]] = {}
        self._superseded: Dict[str, Set[str]] = {}

    def check(
        self,
        sender: str,
        session_id: str,
        seq: int,
        *,
        sent_s: float | None = None,
        now_s: float | None = None,
    ) -> tuple[bool, str]:
        """Check one frame and return ``(accepted, reason)``.

        Reasons are ``fresh``, ``reordered_fresh``, ``replay``, ``stale``,
        ``too_old``, and ``old_session``. The age rule is applied only when a
        limit is configured and both timestamps are supplied. A frame rejected
        for age or session does not alter the sequence window.
        """
        if not isinstance(sender, str) or not isinstance(session_id, str):
            raise ValueError("sender and session_id must be strings")
        self._validate_seq(seq)
        self._validate_time(sent_s, "sent_s")
        self._validate_time(now_s, "now_s")

        current = self._sessions.get(sender)
        superseded = self._superseded.get(sender)
        if current is not None and session_id != current[0]:
            if superseded is not None and session_id in superseded:
                return False, "old_session"
            # A new session is allowed to reset state, but freshness still
            # applies before committing that reset.
            if self._too_old(sent_s, now_s):
                return False, "too_old"
            if superseded is None:
                superseded = set()
                self._superseded[sender] = superseded
            superseded.add(current[0])
            tracker = SequenceTracker(window=self.window)
            self._sessions[sender] = (session_id, tracker)
        elif current is None:
            if superseded is not None and session_id in superseded:
                return False, "old_session"
            if self._too_old(sent_s, now_s):
                return False, "too_old"
            tracker = SequenceTracker(window=self.window)
            self._sessions[sender] = (session_id, tracker)
        else:
            tracker = current[1]
            if self._too_old(sent_s, now_s):
                return False, "too_old"

        observation = tracker.observe(sender, seq)
        if observation.status == "new":
            return True, "fresh"
        if observation.status == "reordered":
            return True, "reordered_fresh"
        if observation.status == "duplicate":
            return False, "replay"
        if observation.status == "stale":
            return False, "stale"
        raise RuntimeError(f"unknown sequence status: {observation.status}")

    def _too_old(self, sent_s: float | None, now_s: float | None) -> bool:
        return (
            self.max_age_s is not None
            and sent_s is not None
            and now_s is not None
            and now_s - sent_s > self.max_age_s
        )

    @staticmethod
    def _validate_seq(seq: int) -> None:
        _validate_seq(seq)

    @staticmethod
    def _validate_time(value: float | None, name: str) -> None:
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
