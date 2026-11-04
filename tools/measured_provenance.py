"""One rule for when an input that says "measured" may be called measured downstream.

The intakes refuse a measured bench file without provenance, and the trace readers
refuse a measured trace without its provenance block. The files made after that (fits,
latency analyses, link JSONs) are what the reruns and the latency budget read. Each
reader asks ``measured_problem`` here: an input that says ``measured`` but carries no
checkable record of where it came from is reported as ``unverified`` with the reason,
never as measured.

What counts as checkable (hashes must be present and well formed; nothing here re-reads
the raw files to compare them):

- ESP-NOW fit (``zephyr-espnow-fit-1``): ``source_trace.provenance`` with non-empty
  ``operator``, ``captured_at`` and ``code_revision`` and a 64-hex ``raw_sha256``, the
  block ``tools/fit_espnow_log.py`` copies from a measured trace.
- Vision noise fit (``zephyr-vision-noise-fit-1``): the same four fields in
  ``input.provenance``, copied from a measured detection log by ``sim/vision_noise_fit.py``.
- Camera latency analysis (``zephyr-vision-latency-1``): 64-hex hashes of ``frames.npz``
  and ``display_log.csv`` under ``inputs`` and a ``run_meta`` with the recorder keys,
  which ``tools/vision_latency.py analyze`` writes for a recorded run.
- Small link JSON (anything else with a status): ``inputs`` listing every source file
  with a name and a 64-hex sha256, the block ``tools/vision_latency_to_link.py`` writes.
"""
from __future__ import annotations

import re
from typing import Any

UNVERIFIED = "unverified"
ESPNOW_FIT_SCHEMA = "zephyr-espnow-fit-1"
VISION_FIT_SCHEMA = "zephyr-vision-noise-fit-1"
LATENCY_SCHEMA = "zephyr-vision-latency-1"
PROVENANCE_TEXT = ("operator", "captured_at", "code_revision")
# The same keys tools/intake_vision.py requires in a measured run's run_meta.json.
RECORDER_KEYS = ("schema", "opencv_version", "platform", "clock")
LATENCY_RUN_FILES = ("frames.npz", "display_log.csv")
_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_HEX.fullmatch(value) is not None


def block_problem(block: Any, where: str) -> str | None:
    """Why a capture provenance block cannot back a measured label, or None when it can."""
    if not isinstance(block, dict):
        return f"it has no {where} block (operator, captured_at, code_revision, raw_sha256)"
    missing = [key for key in PROVENANCE_TEXT if not _text(block.get(key))]
    if missing:
        return f"{where} has no {', '.join(missing)}"
    if not _sha256(block.get("raw_sha256")):
        return f"{where}.raw_sha256 is not a 64-character hex hash of the raw bench file"
    return None


def _inputs_problem(inputs: Any) -> str | None:
    entries = list(inputs.values()) if isinstance(inputs, dict) else []
    if not entries:
        return "it lists no source files under inputs"
    if not all(isinstance(e, dict) and _text(e.get("file")) and _sha256(e.get("sha256")) for e in entries):
        return "not every entry under inputs has a file name and a 64-character hex sha256"
    return None


def _latency_problem(data: dict[str, Any]) -> str | None:
    inputs = data.get("inputs") if isinstance(data.get("inputs"), dict) else {}
    missing = [name for name in LATENCY_RUN_FILES if not _sha256(inputs.get(name))]
    if missing:
        return f"inputs has no sha256 for {', '.join(missing)}"
    meta = data.get("run_meta")
    if not isinstance(meta, dict):
        return "it has no run_meta (the recorder metadata)"
    missing = [key for key in RECORDER_KEYS if not meta.get(key)]
    if missing:
        return f"run_meta lacks the recorder keys {', '.join(missing)}"
    return None


def measured_problem(data: Any) -> str | None:
    """For an input that says measured: why its provenance cannot be checked, or None when it can.

    The caller decides what "says measured" means for its input kind and only asks then.
    """
    if not isinstance(data, dict):
        return "it is not a JSON object"
    schema = data.get("schema")
    if schema == ESPNOW_FIT_SCHEMA:
        source = data.get("source_trace") if isinstance(data.get("source_trace"), dict) else {}
        return block_problem(source.get("provenance"), "source_trace.provenance")
    if schema == VISION_FIT_SCHEMA:
        block = data.get("input") if isinstance(data.get("input"), dict) else {}
        return block_problem(block.get("provenance"), "input.provenance")
    if schema == LATENCY_SCHEMA:
        return _latency_problem(data)
    return _inputs_problem(data.get("inputs"))


def checked_status(declared: Any, data: Any) -> tuple[Any, str | None]:
    """Return (status, note). A declared "measured" without checkable provenance becomes unverified."""
    if declared != "measured":
        return declared, None
    problem = measured_problem(data)
    if problem is None:
        return declared, None
    return UNVERIFIED, f"says measured, but {problem}, so it is not treated as measured"
