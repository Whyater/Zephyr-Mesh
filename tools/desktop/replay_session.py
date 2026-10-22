"""Tk-free replay state and cached presentation for the desktop UI."""
from __future__ import annotations

import json
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from .presentation import present_run
from .replay import ReplayModel


Schedule = Callable[[Callable[[], None], int], Any]
Cancel = Callable[[Any], None]
SPEEDS = (0.1, 0.25, 1.0)


def _document(source: Mapping[str, Any] | str | Path | ReplayModel) -> Mapping[str, Any]:
    if isinstance(source, ReplayModel):
        if source.source_path is None:
            raise ValueError("ReplayModel must have source_path or be constructed from a document")
        return json.loads(source.source_path.read_text(encoding="utf-8"))
    if isinstance(source, Mapping):
        return source
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _frame_time(frame: Any, index: int) -> float:
    """A frame's time in seconds. NaN, infinity, or a non-number would break the playback timer, so they are rejected."""
    if not isinstance(frame, Mapping):
        raise ValueError(f"frame {index + 1} is not an object")
    raw = frame.get("time_s", frame.get("time", 0.0))
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise ValueError(f"frame {index + 1} time must be a finite number of seconds")
    return float(raw)


class ReplaySession:
    """Own replay transitions while leaving scheduling to the Tk surface."""

    def __init__(
        self,
        run: Mapping[str, Any] | str | Path | ReplayModel,
        schedule: Schedule | None = None,
        cancel: Cancel | None = None,
        *,
        speed: float = 0.25,
    ) -> None:
        self.run = _document(run)
        frames = self.run.get("steps", self.run.get("frames", []))
        if not isinstance(frames, list) or not frames:
            raise ValueError("run must contain at least one frame")
        self._times = tuple(_frame_time(frame, index) for index, frame in enumerate(frames))
        self._presentations = tuple(present_run(self.run)["frames"])
        self._index = 0
        self._playing = False
        self._speed = 0.25
        self._schedule = schedule
        self._cancel = cancel
        self._pending = False
        self._job: Any = None
        self._selected_drone_id = self._default_selection()
        self._team_filter = ""
        self._view_mode = "3d"
        self._show_keepout = True
        self._show_inspector = True
        self.speed = speed

    def _default_selection(self) -> str:
        for record in self._presentations[0]["per_drone"]:
            if record.get("level") != "ok":
                return str(record["drone_id"])
        records = self._presentations[0]["per_drone"]
        return str(records[0]["drone_id"]) if records else "D01"

    @property
    def frame_index(self) -> int:
        return self._index

    @property
    def index(self) -> int:
        return self._index

    @property
    def current_frame_index(self) -> int:
        return self._index

    @property
    def frame_count(self) -> int:
        return len(self._presentations)

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def pending_tick(self) -> bool:
        return self._pending

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        numeric = float(value)
        if numeric not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        self._speed = numeric

    @property
    def selected_drone_id(self) -> str:
        return self._selected_drone_id

    @selected_drone_id.setter
    def selected_drone_id(self, value: str) -> None:
        self._selected_drone_id = str(value)

    @property
    def team_filter(self) -> str:
        return self._team_filter

    @team_filter.setter
    def team_filter(self, value: str) -> None:
        self._team_filter = str(value)

    @property
    def view_mode(self) -> str:
        return self._view_mode

    @view_mode.setter
    def view_mode(self, value: str) -> None:
        if value not in {"3d", "top"}:
            raise ValueError("view_mode must be '3d' or 'top'")
        self._view_mode = value

    @property
    def show_keepout(self) -> bool:
        return self._show_keepout

    @show_keepout.setter
    def show_keepout(self, value: bool) -> None:
        self._show_keepout = bool(value)

    @property
    def show_inspector(self) -> bool:
        return self._show_inspector

    @show_inspector.setter
    def show_inspector(self, value: bool) -> None:
        self._show_inspector = bool(value)

    @property
    def presentation(self) -> Mapping[str, Any]:
        return self._presentations[self._index]

    @property
    def current_presentation(self) -> Mapping[str, Any]:
        return self.presentation

    @property
    def headline(self) -> Mapping[str, Any]:
        return self.presentation["headline"]

    @property
    def strip(self) -> Mapping[str, Any]:
        return self.presentation["strip"]

    @property
    def groups(self) -> list[dict[str, Any]]:
        return self.presentation["groups"]

    @property
    def per_drone(self) -> list[dict[str, Any]]:
        return self.presentation["per_drone"]

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.presentation["events"]

    @property
    def frame_time(self) -> float:
        return self._times[self._index]

    @property
    def times(self) -> tuple[float, ...]:
        return self._times

    @property
    def duration(self) -> float:
        return self._times[-1] if self._times else 0.0

    def presentation_at(self, index: int) -> Mapping[str, Any]:
        return self._presentations[max(0, min(int(index), self.frame_count - 1))]

    def events_through(self, index: int | None = None) -> list[tuple[int, Mapping[str, Any]]]:
        """Every timeline event up to and including a frame, oldest first, with its frame index."""
        last = self._index if index is None else max(0, min(int(index), self.frame_count - 1))
        return [(frame, event) for frame in range(last + 1) for event in self._presentations[frame]["events"]]

    def event_frames(self) -> list[int]:
        """Frames that carry at least one event, for timeline tick marks."""
        return [frame for frame, presentation in enumerate(self._presentations) if presentation["events"]]

    def record_for(self, drone_id: str) -> Mapping[str, Any] | None:
        """The current frame's presentation record for one drone."""
        return next((record for record in self.per_drone if record["drone_id"] == drone_id), None)

    def step_forward(self) -> Mapping[str, Any]:
        self._index = min(self._index + 1, self.frame_count - 1)
        return self.presentation

    def step(self, delta: int = 1) -> Mapping[str, Any]:
        if delta >= 0:
            for _ in range(delta):
                self.step_forward()
        else:
            for _ in range(-delta):
                self.step_back()
        return self.presentation

    def step_back(self) -> Mapping[str, Any]:
        self._index = max(self._index - 1, 0)
        return self.presentation

    def seek_frame(self, index: int) -> Mapping[str, Any]:
        if not isinstance(index, int):
            raise TypeError("frame index must be an integer")
        self._index = max(0, min(index, self.frame_count - 1))
        return self.presentation

    def seek_time(self, time_s: float) -> Mapping[str, Any]:
        value = float(time_s)
        nearest = min(range(self.frame_count), key=lambda index: abs(self._times[index] - value))
        return self.seek_frame(nearest)

    def restart(self) -> Mapping[str, Any]:
        self.pause()
        self._index = 0
        return self.presentation

    def _delay_ms(self) -> int:
        if len(self._times) < 2:
            return 1
        dt = max(self._times[1] - self._times[0], 0.001)
        # Bounded so a huge time step can never schedule an overflowing or hour-long timer.
        return max(1, min(60_000, round(dt * 1000.0 / self._speed)))

    def _schedule_tick(self) -> None:
        if not self._playing or self._pending or self._schedule is None:
            return
        self._pending = True
        try:
            delay = self._delay_ms()
            try:
                first_name = next(iter(inspect.signature(self._schedule).parameters), "")
            except (TypeError, ValueError):
                first_name = ""
            if first_name in {"delay", "delay_ms", "milliseconds", "ms"}:
                self._job = self._schedule(delay, self._tick)
            else:
                self._job = self._schedule(self._tick, delay)
        except Exception:
            self._pending = False
            self._job = None
            raise

    def _tick(self) -> None:
        self._pending = False
        self._job = None
        if not self._playing:
            return
        if self._index >= self.frame_count - 1:
            self._playing = False
            return
        self.step_forward()
        self._schedule_tick()

    def tick(self) -> None:
        """Run one scheduled tick; useful for deterministic UI tests."""
        self._tick()

    def play(self) -> None:
        if self._index >= self.frame_count - 1:
            self._index = 0
        self._playing = True
        self._schedule_tick()

    def pause(self) -> None:
        self._playing = False
        if self._pending:
            if self._cancel is not None:
                self._cancel(self._job)
            self._pending = False
            self._job = None

    def toggle_play(self) -> None:
        self.pause() if self._playing else self.play()

    def play_pause(self) -> None:
        self.toggle_play()

    def set_speed(self, value: float) -> None:
        self.speed = value

    def set_selected_drone(self, value: str) -> None:
        self.selected_drone_id = value

    def set_team_filter(self, value: str) -> None:
        self.team_filter = value

    def set_view_mode(self, value: str) -> None:
        self.view_mode = value

    def set_keepout_visible(self, value: bool) -> None:
        self.show_keepout = value

    def set_inspector_visible(self, value: bool) -> None:
        self.show_inspector = value


__all__ = ["Cancel", "ReplaySession", "SPEEDS", "Schedule"]
