"""Tk-free text formatting for the Windows replay surface.

Every value shown on screen comes from ``desktop.presentation`` (or a decoded
report) and passes through one of these helpers, so precision and the wording
for missing values stay identical everywhere. ``None`` always means the data
did not contain the value; callers show ``NOT_RECORDED`` for it, never zero.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .design_tokens import SHORTCUTS, UI_TEXT


LEVELS: Mapping[str, Mapping[str, Any]] = UI_TEXT["statusLevels"]
NOT_RECORDED: str = LEVELS["unavailable"]["word"]
FIELD_LABELS: Mapping[str, str] = UI_TEXT["fieldLabels"]
EN_DASH = "–"
SUMMARY_ORDER = ("ok", "caution", "alert", "waiting", "unavailable")

# Number formatting per value kind: (digits, unit suffix).
_KINDS = {
    "ms": (0, " ms"),
    "pct": (None, "%"),
    "s": (2, " s"),
    "m": (2, " m"),
    "mps": (2, " m/s"),
    "deg": (0, "°"),
    "count": (0, ""),
}

_UNIT_SUFFIXES = (
    ("_mps2", "m/s²"), ("_mps", "m/s"), ("_ms", "ms"), ("_pct", "%"), ("_nm", "N·m"),
    ("_wh", "Wh"), ("_kg", "kg"), ("_rpm", "rpm"), ("_deg", "°"), ("_hz", "Hz"),
    ("_s", "s"), ("_m", "m"), ("_w", "W"), ("_n", "N"), ("_v", "V"),
)
_WORDS = {
    "id": "ID", "ids": "IDs", "p50": "P50", "p95": "P95", "p99": "P99", "rssi": "RSSI",
    "sha256": "SHA-256", "espnow": "ESP-NOW", "phy": "PHY", "dt": "Time step", "numpy": "NumPy",
    "rpm": "RPM", "cpu": "CPU", "imu": "IMU", "gps": "GPS", "fps": "FPS",
}


def _number(value: float, digits: int | None) -> str:
    """Format one number; ``digits=None`` uses 0 or 1 decimals as needed."""
    if digits is None:
        rounded = round(float(value), 1)
        return f"{rounded:.0f}" if rounded == round(rounded) else f"{rounded:.1f}"
    return f"{float(value):.{digits}f}"


def format_value(value: Any, kind: str) -> str | None:
    """Format a presentation value; ranges ``{"min", "max"}`` use an en dash.

    Returns ``None`` when the value is missing, so the caller can show
    ``NOT_RECORDED`` in the secondary text color.
    """
    if value is None:
        return None
    digits, unit = _KINDS[kind]
    if isinstance(value, Mapping):
        low, high = value.get("min"), value.get("max")
        if low is None or high is None:
            return None
        span_digits = 0 if digits is None else digits  # ranges use whole percents so they fit table cells
        low_text, high_text = _number(low, span_digits), _number(high, span_digits)
        if low_text == high_text:
            return f"{low_text}{unit}"
        return f"{low_text}{EN_DASH}{high_text}{unit}"
    return f"{_number(value, digits)}{unit}"


def display(value: Any, kind: str) -> str:
    """Format a value or return the shared missing-value word."""
    text = format_value(value, kind)
    return NOT_RECORDED if text is None else text


def format_age_span(newest: float | None, oldest: float | None) -> str | None:
    """Format the team target-age span from the presentation strip."""
    if newest is None and oldest is None:
        return None
    if newest is None or oldest is None or _number(newest, 2) == _number(oldest, 2):
        return format_value(oldest if oldest is not None else newest, "s")
    return f"{_number(newest, 2)}{EN_DASH}{_number(oldest, 2)} s"


def format_vector(values: Sequence[float] | None, digits: int = 2) -> str:
    """Format a three-element vector without brackets."""
    if values is None:
        return NOT_RECORDED
    return ", ".join(f"{float(item):.{digits}f}" for item in values)


def status_word(level: str, word: str | None = None) -> str:
    """Return the record's word, or the generated word for a level."""
    if word:
        return str(word)
    return str(LEVELS.get(level, LEVELS["unavailable"])["word"])


def team_summary(per_drone: Iterable[Mapping[str, Any]]) -> str:
    """Build the one-line team summary, for example ``50 drones: 45 OK, 4 late``."""
    records = list(per_drone)
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (str(record.get("level", "unavailable")), status_word(record.get("level", ""), record.get("word")))
        counts[key] = counts.get(key, 0) + 1
    rank = {level: index for index, level in enumerate(SUMMARY_ORDER)}
    parts = []
    for (level, word), count in sorted(counts.items(), key=lambda item: (rank.get(item[0][0], 99), item[0][1])):
        shown = word if word.isupper() else word.lower()
        parts.append(f"{count} {shown}")
    noun = "drone" if len(records) == 1 else "drones"
    return f"{len(records)} {noun}: {', '.join(parts)}" if parts else f"0 {noun}"


def group_label(count: int) -> str:
    """Label for a team-list group row."""
    return f"{count} drone" if count == 1 else f"{count} drones"


def plain_run_name(run: Mapping[str, Any]) -> str:
    """Name the run in plain words, for example ``50-drone ring, seed 17``."""
    frames = run.get("steps", run.get("frames", []))
    first = frames[0] if frames else {}
    count = len(first.get("agents", first.get("drones", []))) if isinstance(first, Mapping) else 0
    identifier = str(run.get("scenario_id", run.get("run_id", ""))).lower()
    shape = "ring" if "ring" in identifier else "run"
    name = f"{count}-drone {shape}"
    seed = run.get("seed")
    return f"{name}, seed {seed}" if seed is not None else name


def time_label(current_s: float, total_s: float) -> str:
    """Timeline label, for example ``0.10 / 0.25 s``."""
    return f"{current_s:.2f} / {total_s:.2f} s"


def accelerator(action_id: str) -> str:
    """Return the Windows menu accelerator text for one generated shortcut."""
    shortcut = next(item for item in SHORTCUTS if item.action_id == action_id)
    names = {"space": "Space", "right": "Right", "left": "Left", "home": "Home"}
    parts = ["Ctrl" if item == "primary" else item.title() for item in shortcut.modifiers]
    parts.append(names.get(shortcut.key, shortcut.key.upper()))
    return "+".join(parts)


def humanize_key(key: str) -> str:
    """Turn a data key into a readable label with its unit in parentheses."""
    raw = str(key)
    unit = None
    lowered = raw.lower()
    for suffix, name in _UNIT_SUFFIXES:
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            unit = name
            lowered = lowered[: -len(suffix)]
            break
    words = [word for word in re.split(r"[_\s]+", lowered) if word]
    if not words:
        return raw
    shown = [_WORDS.get(word, word) for word in words]
    first = shown[0]
    shown[0] = first if first != first.lower() else first[:1].upper() + first[1:]
    text = " ".join(shown)
    return f"{text} ({unit})" if unit else text


def field_label(key: str) -> str:
    """Return the shared label for a data key, or a readable fallback."""
    return FIELD_LABELS.get(str(key)) or humanize_key(str(key))


def format_scalar(value: Any) -> str:
    """Format one decoded JSON value for a label/value row."""
    if value is None:
        return NOT_RECORDED
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return NOT_RECORDED
        if value == 0:
            return "0"
        if abs(value) < 1e-3:
            return f"{value:.2e}"
        if abs(value) >= 1000:
            return f"{value:.0f}"
        return f"{value:.4g}"
    if isinstance(value, str):
        if not value.strip():
            return NOT_RECORDED
        return value[:1].upper() + value[1:] if " " in value.strip() else value
    if isinstance(value, Sequence):
        items = list(value)
        if not items:
            return "None"
        if all(not isinstance(item, (Mapping, list, tuple)) for item in items):
            shown = ", ".join(format_scalar(item) for item in items[:6])
            return shown if len(items) <= 6 else f"{shown} and {len(items) - 6} more"
        return f"{len(items)} items, listed under Details"
    if isinstance(value, Mapping):
        if not value:
            return "None"
        if len(value) <= 8 and all(not isinstance(item, (Mapping, list, tuple)) for item in value.values()):
            return ", ".join(f"{field_label(key)} {format_scalar(item)}" for key, item in sorted(value.items()))
        return f"{len(value)} fields, listed under Details"
    return str(value)


def summary_sections(summary: Mapping[str, Any]) -> list[tuple[str | None, list[tuple[str, str]]]]:
    """Group a nested summary into titled sections of label/value rows.

    Top-level scalars come first under no title. Each nested object becomes
    one section; anything deeper is summarized and left to Details.
    """
    top: list[tuple[str, str]] = []
    sections: list[tuple[str | None, list[tuple[str, str]]]] = []
    for key in sorted(summary):
        value = summary[key]
        if isinstance(value, Mapping):
            rows = [(field_label(inner), format_scalar(item)) for inner, item in sorted(value.items())]
            sections.append((field_label(key), rows or [("Fields", "None")]))
        else:
            top.append((field_label(key), format_scalar(value)))
    return ([(None, top)] if top else []) + sections


__all__ = [
    "EN_DASH", "FIELD_LABELS", "LEVELS", "NOT_RECORDED", "accelerator", "display", "field_label",
    "format_age_span", "format_scalar", "format_value", "format_vector", "group_label",
    "humanize_key", "plain_run_name", "status_word", "summary_sections", "team_summary", "time_label",
]
