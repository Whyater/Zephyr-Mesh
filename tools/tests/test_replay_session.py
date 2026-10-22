from __future__ import annotations

import json
from pathlib import Path

from tools.desktop.replay_session import ReplaySession


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "sim/runs/s7-swarm/run.json"


def test_session_transitions_and_cached_presentation():
    session = ReplaySession(RUN)
    assert session.speed == 0.25
    assert set(("headline", "strip", "groups", "per_drone", "events")) <= set(session.presentation)
    assert session.frame_index == 0
    session.step_forward()
    assert session.frame_index == 1
    session.step_back()
    assert session.frame_index == 0
    session.seek_frame(999)
    assert session.frame_index == session.frame_count - 1
    session.seek_time(-100)
    assert session.frame_index == 0
    session.view_mode = "top"
    session.show_keepout = False
    session.show_inspector = False
    session.team_filter = "D01"
    assert session.view_mode == "top"
    assert session.selected_drone_id


def test_play_pause_play_has_one_pending_tick():
    scheduled = []
    cancelled = []

    def schedule(callback, delay):
        scheduled.append((callback, delay))
        return len(scheduled)

    def cancel(job):
        cancelled.append(job)

    session = ReplaySession(RUN, schedule=schedule, cancel=cancel)
    session.play()
    session.pause()
    session.play()
    assert len(scheduled) == 2
    assert len(cancelled) == 1
    assert session.pending_tick is True
    callback, _delay = scheduled[-1]
    callback()
    assert session.frame_index == 1
    assert len(scheduled) == 3
    assert session.pending_tick is True


def test_default_selection_uses_first_non_ok_or_first_drone():
    run = json.loads(RUN.read_text())
    session = ReplaySession(run)
    assert session.selected_drone_id == "drone-001"
