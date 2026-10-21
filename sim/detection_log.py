"""Line-based detection log reader and writer.

A detection log is a JSON Lines file: one header line with the trace manifest,
then one line per camera/tracker observation. This module is a format reader
and writer, not a camera model. It reuses the strict rules in
``sim.vision_trace`` for the manifest and for every observation, and adds only
the line-level checks (encoding, JSON hygiene, blank lines, duplicate ids).
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sim.vision_trace import VISION_TRACE_SCHEMA, VisionObservation, VisionTrace

DETECTION_LOG_SCHEMA = "zephyr-detection-log-1"


class DetectionLogError(ValueError):
    """A detection log problem. The message starts with ``<path>:<line>: ``."""


def _reject_constant(token: str) -> Any:
    raise ValueError(f"{token} is not allowed (non-finite number)")


def _parse_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"number {token} is not finite")
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_reject_constant,
        parse_float=_parse_float,
    )


def _parse(items: list[str | bytes], source_name: str) -> VisionTrace:
    def fail(line: int, message: str) -> DetectionLogError:
        return DetectionLogError(f"{source_name}:{line}: {message}")

    if not items:
        raise fail(0, "empty file: a detection log needs a header line")

    header: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = []
    first_line_for_id: dict[str, int] = {}
    clock_domain = ""
    line_no = 0

    for line_no, item in enumerate(items, start=1):
        if isinstance(item, bytes):
            try:
                text = item.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise fail(line_no, f"invalid UTF-8: {exc.reason} at byte {exc.start} of the line") from exc
        else:
            text = item.removesuffix("\n")
            if "\n" in text:
                raise fail(line_no, "line contains an embedded newline")
        if line_no == 1 and text.startswith("\ufeff"):
            raise fail(line_no, "file starts with a UTF-8 byte order mark (BOM); save the log as UTF-8 without a BOM")
        if "\r" in text:
            raise fail(line_no, "carriage return found; use LF line endings, CRLF files are not allowed")
        if not text.strip():
            raise fail(line_no, "blank line; only one final newline is allowed")
        try:
            value = _loads(text)
        except ValueError as exc:
            raise fail(line_no, f"invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise fail(line_no, "line must be a JSON object")

        if header is None:
            if value.get("schema") != DETECTION_LOG_SCHEMA:
                raise fail(line_no, f"missing header: line 1 must be an object with schema {DETECTION_LOG_SCHEMA}")
            if "observations" in value:
                raise fail(line_no, "header must not contain observations; they go on the following lines")
            manifest = {key: field for key, field in value.items() if key != "schema"}
            try:
                parsed = VisionTrace.from_dict({**manifest, "schema": VISION_TRACE_SCHEMA, "observations": []})
            except (ValueError, OverflowError) as exc:
                raise fail(line_no, f"header: {exc}") from exc
            header = manifest
            clock_domain = parsed.clock_domain
            continue

        try:
            observation = VisionObservation.from_dict(value)
        except (ValueError, OverflowError) as exc:
            raise fail(line_no, f"observation: {exc}") from exc
        if observation.observation_id in first_line_for_id:
            raise fail(
                line_no,
                f"duplicate observation_id {observation.observation_id!r} (first seen on line "
                f"{first_line_for_id[observation.observation_id]})",
            )
        first_line_for_id[observation.observation_id] = line_no
        if (
            clock_domain == "shared_monotonic"
            and observation.processing_time_s is not None
            and observation.processing_time_s < observation.capture_time_s
        ):
            raise fail(line_no, "observation: shared_monotonic processing_time_s cannot precede capture_time_s")
        observations.append(value)

    assert header is not None
    try:
        return VisionTrace.from_dict({**header, "schema": VISION_TRACE_SCHEMA, "observations": observations})
    except (ValueError, OverflowError) as exc:
        raise fail(line_no, str(exc)) from exc


def parse_detection_log_lines(lines: Iterable[str], *, source_name: str = "<memory>") -> VisionTrace:
    """Validate and parse log lines (with or without a trailing newline each)."""

    return _parse(list(lines), source_name)


def read_detection_log(path: str | Path) -> VisionTrace:
    """Read a detection log file, raising DetectionLogError on the first problem."""

    path = Path(path)
    raw_lines = path.read_bytes().split(b"\n")
    if raw_lines[-1] == b"":
        raw_lines.pop()
    return _parse(list(raw_lines), str(path))


def detection_log_lines(trace: VisionTrace, *, source_name: str = "<memory>") -> list[str]:
    """Return the header line and one line per observation, without newlines."""

    def dump(value: dict[str, Any], line: int, where: str) -> str:
        try:
            return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
        except ValueError as exc:
            raise DetectionLogError(f"{source_name}:{line}: cannot write {where}: {exc}") from exc

    data = trace.as_dict()
    observations = data.pop("observations")
    data["schema"] = DETECTION_LOG_SCHEMA
    lines = [dump(data, 1, "header")]
    for line, observation in enumerate(observations, start=2):
        lines.append(dump({key: value for key, value in observation.items() if value is not None}, line,
                          f"observation {observation.get('observation_id')!r}"))
    return lines


def write_detection_log(trace: VisionTrace, path: str | Path) -> None:
    """Write a trace as a detection log with LF line endings and one final newline."""

    text = "\n".join(detection_log_lines(trace, source_name=str(path))) + "\n"
    Path(path).write_bytes(text.encode("utf-8"))
