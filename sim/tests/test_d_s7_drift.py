"""Pins the S7 v1 drift fixes: an event's optional outcome must match what happened to it.

Pending events (no receive time, no loss reason) stay valid in v1, as before,
but may not claim an outcome. Summaries keep reporting them as "unknown",
which is a summary label, not an event outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from sim.s7_metrics import _outcome

ROOT = Path(__file__).parents[2]

BASE = {"sender": "a", "receiver": "b", "seq": 0, "send_time": 0.0, "duplicate": False, "out_of_order": False}
RECEIVED = dict(BASE, receive_time=0.1, loss_reason=None, packet_age=0.1)
LOST = dict(BASE, receive_time=None, loss_reason="link", packet_age=None)
PENDING = dict(BASE, receive_time=None, loss_reason=None, packet_age=None)


def _validator():
    schema = json.loads((ROOT / "sim/s7_run_schema.json").read_text(encoding="utf-8"))
    return Draft202012Validator({**schema["$defs"]["linkEvent"], "$defs": schema["$defs"]})


def test_outcome_must_match_event_state():
    validator = _validator()
    assert not list(validator.iter_errors(dict(RECEIVED, outcome="received")))
    assert not list(validator.iter_errors(dict(LOST, outcome="lost")))
    assert list(validator.iter_errors(dict(RECEIVED, outcome="lost")))
    assert list(validator.iter_errors(dict(LOST, outcome="received")))


def test_pending_events_stay_valid_but_cannot_claim_an_outcome():
    validator = _validator()
    assert not list(validator.iter_errors(PENDING))
    assert list(validator.iter_errors(dict(PENDING, outcome="received")))
    assert list(validator.iter_errors(dict(PENDING, outcome="lost")))
    assert _outcome(PENDING) == "unknown"


def test_frozen_s7_run_still_validates():
    schema = json.loads((ROOT / "sim/s7_run_schema.json").read_text(encoding="utf-8"))
    document = json.loads((ROOT / "sim/runs/s7-swarm/run.json").read_text(encoding="utf-8"))
    assert not list(Draft202012Validator(schema).iter_errors(document))
