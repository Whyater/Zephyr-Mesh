"""Evidence-bounded summaries for motor/propeller bench traces."""
from __future__ import annotations

from math import floor, fsum
from typing import Any

from .bench_trace import BenchTrace

SUMMARY_SCHEMA = "zephyr-bench-trace-summary-1"


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


def summarize_bench_trace(trace: BenchTrace) -> dict[str, Any]:
    """Summarize observed bench records without fitting a transfer model."""

    valid = [sample for sample in trace.samples if sample.quality == "valid"]
    invalid = [sample for sample in trace.samples if sample.quality == "invalid"]
    unknown = [sample for sample in trace.samples if sample.quality == "unknown"]
    rpm = [sample.rpm for sample in valid if sample.rpm is not None]
    thrust = [sample.thrust_n for sample in valid if sample.thrust_n is not None]
    voltage = [sample.voltage_v for sample in valid if sample.voltage_v is not None]
    current = [sample.current_a for sample in valid if sample.current_a is not None]
    duration = [sample.duration_s for sample in valid if sample.duration_s is not None]
    temperature = [sample.temperature_c for sample in valid if sample.temperature_c is not None]
    electrical_power = []
    electrical_power_nonfinite = 0
    for sample in valid:
        if sample.voltage_v is None or sample.current_a is None:
            continue
        power = sample.voltage_v * sample.current_a
        if power == float("inf") or power == float("-inf"):
            electrical_power_nonfinite += 1
        else:
            electrical_power.append(power)
    return {
        "schema": SUMMARY_SCHEMA,
        "evidence_boundary": "observed bench trace summary; not a fitted motor map or flight performance",
        "provenance": {
            "trace_id": trace.trace_id,
            "status": trace.status,
            "source": trace.source,
            "calibration_id": trace.calibration_id,
            "code_revision": trace.code_revision,
            "notes": trace.notes,
            "profile_id": trace.profile_id,
            "motor_part_id": trace.motor_part_id,
            "propeller_part_id": trace.propeller_part_id,
            "measurement_scope": trace.measurement_scope,
            "clock_domain": trace.clock_domain,
            "capture": trace.capture,
        },
        "samples": {
            "observed_records": len(trace.samples),
            "valid": len(valid),
            "invalid": len(invalid),
            "unknown_unresolved": len(unknown),
            "unknown_reason_counts": {reason: sum(sample.quality_reason == reason for sample in unknown) for reason in sorted({sample.quality_reason for sample in unknown})},
            "invalid_reason_counts": {reason: sum(sample.quality_reason == reason for sample in invalid) for reason in sorted({sample.quality_reason for sample in invalid})},
            "complete_quality_denominator": not unknown,
            "electrical_power_nonfinite_excluded": electrical_power_nonfinite,
        },
        "measurements": {
            "rpm": _distribution(rpm, unit="rev/min"),
            "thrust": _distribution(thrust, unit="N"),
            "voltage": _distribution(voltage, unit="V"),
            "current": _distribution(current, unit="A"),
            "electrical_input_power": _distribution(electrical_power, unit="W"),
            "duration": _distribution(duration, unit="s"),
            "temperature": _distribution(temperature, unit="degC"),
        },
        "method": {
            "electrical_input_power": "computed per valid record as voltage_v * current_a; not mechanical shaft power",
            "weighting": f"equal weight per valid record within declared measurement_scope={trace.measurement_scope}",
            "quantile_method": "sorted empirical samples with linear interpolation at (n-1)*p; descriptive only",
            "fit_status": "not fitted; use a frozen calibration group and held-out validation before changing simulator coefficients",
        },
    }
