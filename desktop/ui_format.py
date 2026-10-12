"""Tk-free text formatting for the Windows replay surface.

Every value shown on screen comes from ``desktop.presentation`` (or a decoded
report) and passes through one of these helpers, so precision and the wording
for missing values stay identical everywhere. ``None`` always means the data
did not contain the value; callers show ``NOT_RECORDED`` for it, never zero.
"""
from __future__ import annotations

import math
from pathlib import Path
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


def event_row(event: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """One event-list row: (level for the glyph, word, time, text), always from UI_TEXT.

    An event level that UI_TEXT does not define shows as unavailable, so a row
    never appears with a bare shape and no word.
    """
    level = str(event.get("level", ""))
    if level not in LEVELS:
        level = "unavailable"
    time_text = f"{float(event['time_s']):.2f} s" if event.get("time_s") is not None else NOT_RECORDED
    return level, status_word(level), time_text, str(event.get("text", ""))


def quantity(value: Any, unit: str, factor: float = 1.0, digits: int | None = None) -> str:
    """A number with its unit, or ``NOT_RECORDED`` when the data lacks it (never 0)."""
    if value is None or isinstance(value, bool):
        return NOT_RECORDED
    try:
        number = float(value) * factor
    except (TypeError, ValueError):
        return NOT_RECORDED
    text = format_scalar(number) if digits is None else f"{number:.{digits}f}"
    return f"{text} {unit}"


def part_text(part_id: Any, detail: str) -> str:
    """Part id plus one key figure, for example ``drone-001-motor, 18.72 W max``."""
    name = str(part_id) if part_id else NOT_RECORDED
    return name if detail == NOT_RECORDED else f"{name}, {detail}"


def link_text(link: Mapping[str, Any] | None) -> str:
    """Plain words for a scenario link model; missing fields read as not recorded."""
    if not link:
        return NOT_RECORDED
    delay = link.get("delay_s")
    jitter = link.get("delay_jitter_s")
    parts = [f"{quantity(delay, 'ms', 1000.0, 0)} delay" if delay is not None else f"delay {NOT_RECORDED.lower()}"]
    if jitter:
        parts[0] += f" (jitter {quantity(jitter, 'ms', 1000.0, 0)})"
    model = link.get("loss_model")
    probability = link.get("loss_probability")
    if model == "independent":
        parts.append(f"{quantity(probability, '%', 100.0, 0).replace(' %', '%')} independent loss"
                     if probability is not None else f"independent loss, rate {NOT_RECORDED.lower()}")
    elif model == "burst":
        parts.append("burst loss")
    elif model == "none":
        parts.append("no loss")
    else:
        parts.append(f"loss model {NOT_RECORDED.lower()}")
    return ", ".join(parts)


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
        if value in FIELD_LABELS:
            return FIELD_LABELS[value]
        if re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)+", value):
            words = value.replace("_", " ")
            return words[:1].upper() + words[1:]
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


_STAT_KEYS = frozenset({"min", "max", "mean", "p50", "p95"})
_RAW_ID_KEYS = frozenset({
    "schema", "source_schema", "sha256", "payload_sha256", "scenario_hash", "trace_id", "scenario_id", "run_id",
    "device_ids", "code_revision", "calibration_id", "generator", "profile_id", "part_id",
})


def is_stat(value: Any) -> bool:
    """A descriptive statistics object such as ``{"p50": ..., "p95": ..., "unit": "s"}``."""
    return isinstance(value, Mapping) and "unit" in value and bool(_STAT_KEYS & set(value))


def is_raw_id(key: str) -> bool:
    """Keys that hold identifiers or hashes; these live under Details, not in the summary."""
    lowered = str(key).lower()
    return lowered in _RAW_ID_KEYS or lowered.endswith(("_id", "_ids", "_sha256", "_hash"))


def _stat_scale(stat: Mapping[str, Any]) -> tuple[str, float, int]:
    """Display unit, multiplier, and decimals for a statistics object."""
    unit = str(stat.get("unit", ""))
    values = [abs(float(stat[key])) for key in _STAT_KEYS if isinstance(stat.get(key), (int, float))]
    largest = max(values, default=0.0)
    if unit == "s" and largest < 1.0:
        return "ms", 1000.0, 0 if largest >= 0.01 else 1
    if unit == "m":
        return "m", 1.0, 2
    if unit in ("dBm", "dB"):
        return unit, 1.0, 0
    return unit, 1.0, 2


def stat_value(stat: Mapping[str, Any], key: str) -> str:
    """One statistic with its unit, for example ``12 ms``."""
    value = stat.get(key)
    if value is None or isinstance(value, bool):
        return NOT_RECORDED
    unit, factor, digits = _stat_scale(stat)
    number = float(value) * factor
    return f"{number:.{digits}f} {unit}".strip()


def stat_line(stat: Mapping[str, Any]) -> str:
    """One readable line: median, p95, max, and the sample count."""
    count = stat.get("count")
    if stat.get("available") is False or count == 0:
        return NOT_RECORDED
    parts = [f"median {stat_value(stat, 'p50')}" if "p50" in stat else None,
             f"p95 {stat_value(stat, 'p95')}" if "p95" in stat else None,
             f"max {stat_value(stat, 'max')}" if "max" in stat else None]
    text = ", ".join(part for part in parts if part)
    text = text[:1].upper() + text[1:]
    if isinstance(count, int) and not isinstance(count, bool):
        text += f" ({count} sample{'s' if count != 1 else ''})"
    return text


def humanize_filename(name: str) -> str:
    """``evidence_report_example.json`` becomes ``Evidence report example``."""
    stem = Path(str(name)).stem if "." in str(name) else str(name)
    words = re.sub(r"[_\-]+", " ", stem).strip()
    return words[:1].upper() + words[1:] if words else str(name)


def report_view(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Split a report summary into plain sections, timing statistics, and raw Details rows.

    ``sections``: (title or None, rows) with stats as one line and ids removed.
    ``timings``: (label, stat) for statistics measured in seconds, for the verdict plot.
    ``details``: (label, value) for raw ids, schemas, and hashes found anywhere.
    """
    sections: list[tuple[str | None, list[tuple[str, str]]]] = []
    timings: list[tuple[str, Mapping[str, Any]]] = []
    details: list[tuple[str, str]] = []
    top: list[tuple[str, str]] = []

    def visit(key: str, value: Any, rows: list[tuple[str, str]], prefix: str) -> None:
        label = field_label(key)
        if is_raw_id(key):
            details.append((f"{prefix}{label}", format_scalar(value)))
        elif is_stat(value):
            label = re.sub(r" \([^)]*\)$", "", label)
            rows.append((label, stat_line(value)))
            if value.get("unit") == "s" and value.get("available") is not False and value.get("count") != 0:
                timings.append((label, value))
        elif isinstance(value, Mapping) and value and all(not isinstance(item, (Mapping, list)) for item in value.values()) \
                and len(value) <= 8:
            rows.append((label, format_scalar(dict(value))))
        elif isinstance(value, Mapping) and len(value) > 8:
            rows.append((label, f"{len(value)} entries, listed under Details"))
        elif isinstance(value, Mapping):
            nested: list[tuple[str, str]] = []
            for inner in sorted(value):
                visit(inner, value[inner], nested, f"{label}: ")
            if nested:
                rows.extend((f"{label}: {inner_label}", text) for inner_label, text in nested)
        elif isinstance(value, list) and value and all(isinstance(item, Mapping) for item in value):
            rows.append((label, f"{len(value)} rows, listed under Details"))
        elif str(key).endswith("loss_rate") and isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
            rows.append((label, format_value(float(value) * 100.0, "pct") or NOT_RECORDED))
        else:
            rows.append((label, format_scalar(value)))

    for key in sorted(summary):
        value = summary[key]
        if isinstance(value, Mapping) and not is_stat(value) and not is_raw_id(key) and len(value) <= 8 or (
                isinstance(value, Mapping) and not is_stat(value) and any(isinstance(item, Mapping) for item in value.values())
                and not is_raw_id(key)):
            rows: list[tuple[str, str]] = []
            for inner in sorted(value):
                visit(inner, value[inner], rows, f"{field_label(key)}: ")
            if rows:
                sections.append((field_label(key), rows))
        else:
            visit(key, value, top, "")
    return {"sections": ([(None, top)] if top else []) + sections, "timings": timings, "details": details}


__all__ = [
    "EN_DASH", "FIELD_LABELS", "LEVELS", "NOT_RECORDED", "accelerator", "humanize_filename", "is_raw_id", "is_stat",
    "report_view", "stat_line", "stat_value", "display", "event_row", "field_label",
    "format_age_span", "format_scalar", "link_text", "part_text", "quantity", "format_value", "format_vector", "group_label",
    "humanize_key", "plain_run_name", "status_word", "summary_sections", "team_summary", "time_label",
]
