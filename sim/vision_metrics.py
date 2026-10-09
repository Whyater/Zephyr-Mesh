"""Transparent summaries for validated camera/tracker traces.

These summaries describe observed records. They do not infer missed targets
outside the trace, camera calibration quality, or tracker performance.
"""
from __future__ import annotations

from collections import Counter
from math import floor, fsum
from typing import Any

from .vision_trace import VisionTrace

SUMMARY_SCHEMA = "zephyr-vision-trace-summary-1"


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return (1.0 - fraction) * ordered[lower] + fraction * ordered[upper]


def _distribution(values: list[float], *, unit: str) -> dict[str, Any]:
    if not values:
        return {"available": False, "unit": unit, "count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "available": True,
        "unit": unit,
        "count": len(values),
        "min": min(values),
        "mean": fsum(value / len(values) for value in values),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "max": max(values),
    }


def summarize_vision_trace(trace: VisionTrace) -> dict[str, Any]:
    """Return a JSON-safe, evidence-bounded summary of one vision trace."""

    observations = list(trace.observations)
    total = len(observations)
    detected = sum(item.outcome == "detected" for item in observations)
    missed = sum(item.outcome == "missed" for item in observations)
    unknown = sum(item.outcome == "unknown" for item in observations)
    processing_latency = [
        item.processing_time_s - item.capture_time_s
        for item in observations
        if item.processing_time_s is not None and trace.clock_domain == "shared_monotonic"
    ]
    confidence = [item.confidence for item in observations if item.outcome == "detected" and item.confidence is not None]
    subjects = {item.subject_id for item in observations}
    frames = {item.frame_id for item in observations}
    complete = unknown == 0
    duplicate_values = [item.duplicate for item in observations]
    order_values = [item.out_of_order for item in observations]
    duplicate_known = [value for value in duplicate_values if value is not None]
    order_known = [value for value in order_values if value is not None]
    return {
        "schema": SUMMARY_SCHEMA,
        "evidence_boundary": "observed vision trace summary; not camera calibration or tracker performance",
        "provenance": {
            "trace_id": trace.trace_id,
            "status": trace.status,
            "source": trace.source,
            "source_schema": "zephyr-vision-trace-1",
            "code_revision": trace.code_revision,
            "notes": trace.notes,
            "clock_domain": trace.clock_domain,
            "clock_uncertainty_s": trace.clock_uncertainty_s,
            "capture": trace.capture,
        },
        "observations": {
            "observed_records": total,
            "frame_count": len(frames),
            "subject_count": len(subjects),
            "detected": detected,
            "missed": missed,
            "unknown_unresolved": unknown,
            "unknown_reason_counts": dict(sorted(Counter(item.unknown_reason for item in observations if item.outcome == "unknown").items())),
            "miss_reason_counts": dict(sorted(Counter(item.miss_reason for item in observations if item.outcome == "missed").items())),
            "complete_outcome_denominator": complete,
            "observed_detection_rate": detected / total if total and complete else None,
            "detection_rate_bounds": {
                "interpretation": "identification bounds over observed records; not confidence intervals",
                "lower": detected / total if total else None,
                "upper": (detected + unknown) / total if total else None,
            },
            "censoring": "not inferred; unknown reasons alone do not establish a terminal frame or mechanism",
            "duplicates": sum(value is True for value in duplicate_known) if duplicate_known else None,
            "duplicate_flag_unknown": sum(value is None for value in duplicate_values),
            "duplicate_metadata_available": bool(duplicate_known),
            "out_of_order": sum(value is True for value in order_known) if order_known else None,
            "out_of_order_flag_unknown": sum(value is None for value in order_values),
            "out_of_order_metadata_available": bool(order_known),
        },
        "processing_latency": _distribution(processing_latency, unit="s") if trace.clock_domain == "shared_monotonic" else {
            "available": False,
            "unit": "s",
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
            "reason": "clock_domain does not support capture-to-processing latency",
        },
        "confidence": _distribution(confidence, unit="probability"),
        "aggregation": {
            "latency_population": "one sample per observed record with both timestamps on a shared clock",
            "confidence_population": "one sample per detected record with supplied confidence",
            "weighting": "equal weight per observed record; frames and subjects are pooled",
            "quantile_method": "sorted empirical samples with linear interpolation at (n-1)*p; descriptive, not population-tail estimates",
            "subject_identity": "subject_id is capture-provided; equivalence to physical ground truth is not inferred",
        },
    }
