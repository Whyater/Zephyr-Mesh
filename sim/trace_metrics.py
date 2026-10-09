"""Transparent summaries for validated ESP-NOW traces.

The summary reports what the capture contains. It keeps unknown outcomes as
unresolved observations and only computes one-way delay when the trace
declares a shared monotonic clock. It does not fit a link model or estimate
radio tolerance.
"""
from __future__ import annotations

from collections import Counter
from math import floor, fsum
from typing import Any

from .link import replay_trace
from .trace import EspNowTrace

SUMMARY_SCHEMA = "zephyr-espnow-trace-summary-1"


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


def summarize_trace(trace: EspNowTrace) -> dict[str, Any]:
    """Return a JSON-safe, evidence-bounded summary of one trace."""

    events = replay_trace(trace)
    attempted = len(events)
    received = sum(event.outcome == "received" for event in events)
    lost = sum(event.outcome == "lost" for event in events)
    unknown = sum(event.outcome == "unknown" for event in events)
    delays = [event.packet_age_s for event in events if event.outcome == "received" and event.packet_age_s is not None]
    rssi = [packet.rssi_dbm for packet in trace.packets if packet.rssi_dbm is not None]
    complete = unknown == 0
    identities = {(event.sender, event.receiver, event.sender_session_id, event.seq) for event in events}
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "evidence_boundary": "observed trace summary; not radio tolerance or flight performance",
        "provenance": {
            "trace_id": trace.trace_id,
            "status": trace.status,
            "source": trace.source,
            "source_schema": "zephyr-espnow-trace-1",
            "code_revision": trace.code_revision,
            "notes": trace.notes,
            "clock_domain": trace.clock_domain,
            "clock_uncertainty_s": trace.clock_uncertainty_s,
            "capture": trace.capture,
        },
        "packets": {
            "observed_transmission_records": attempted,
            "denominator_contract": "one record per observed transmission attempt; repeated logical identities remain separate records; unlogged attempts are not inferred",
            "logical_identity_count": len(identities),
            "identity_fields": ["sender_id", "receiver_id", "sender_session_id", "seq"],
            "received": received,
            "lost": lost,
            "unknown_unresolved": unknown,
            "unknown_reason_counts": dict(sorted(Counter(event.unknown_reason for event in events if event.outcome == "unknown").items())),
            "censoring": "not inferred; unknown reasons alone do not establish censoring time or mechanism",
            "duplicates": sum(event.duplicate for event in events),
            "out_of_order": sum(event.out_of_order for event in events),
            "complete_outcome_denominator": complete,
            "loss_rate": lost / attempted if attempted and complete else None,
            "loss_rate_bounds": {
                "interpretation": "identification bounds over observed records; not confidence intervals",
                "lower": lost / attempted if attempted else None,
                "upper": (lost + unknown) / attempted if attempted else None,
            },
        },
        "delay": _distribution(delays, unit="s") if trace.clock_domain == "shared_monotonic" else {
            "available": False,
            "unit": "s",
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
            "reason": "clock_domain does not support one-way age",
        },
        "rssi": _distribution(rssi, unit="dBm"),
        "aggregation": {
            "delay_population": "one sample per received record with shared-clock age, including repeated logical identities",
            "rssi_population": "one sample per record with supplied RSSI, including repeated logical identities",
            "weighting": "equal weight per observed record; routes and capture conditions are pooled",
            "quantile_method": "sorted empirical samples with linear interpolation at (n-1)*p; descriptive, not population-tail estimates",
        },
    }
    return summary
