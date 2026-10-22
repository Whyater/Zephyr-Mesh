import json

import jsonschema
import numpy as np
import pytest

from sim.link import LinkConfig, SimulatedLink
from sim.radio.channel import ChannelConfig
from sim.radio.mac import MacConfig, data_airtime_s
from sim.radio.model import (BackgroundSource, NetworkModelConfig, RadioLinkConfig, events_to_trace,
                             make_network_model)
from sim.radio.params import default_phy
from sim.trace import EspNowTrace


def _dicts(events):
    return [e.as_dict() for e in events]


@pytest.mark.parametrize("config", [
    LinkConfig(),
    LinkConfig(delay_s=0.05, delay_jitter_s=0.02, loss_model="independent", loss_probability=0.3),
    LinkConfig(delay_s=0.1, loss_model="burst", burst_start_probability=0.05, burst_end_probability=0.2),
    LinkConfig(delay_s=0.01, contention_mode="serialized", packet_duration_s=0.004),
])
def test_abstract_mode_is_simulated_link_event_for_event(config):
    """The abstract mode must not change a single draw: EXP-006 and other callers depend on it."""
    reference = SimulatedLink(config, seed=123)
    model = make_network_model(NetworkModelConfig(mode="abstract", link=config), seed=123)
    out_ref, out_model = [], []
    for k in range(300):
        t = 0.01 * k
        out_ref.append(reference.send("a", "b", k, t).as_dict())
        out_model.append(model.send("a", "b", k, t).as_dict())
        if k % 7 == 0:
            out_ref.extend(_dicts(reference.advance(t)))
            out_model.extend(_dicts(model.advance(t)))
    out_ref.extend(_dicts(reference.deliver_all()))
    out_model.extend(_dicts(model.deliver_all()))
    assert out_model == out_ref
    assert _dicts(model.events) == _dicts(reference.events)


def _clean_channel(**kw):
    # Strong, unshadowed link: PER is effectively zero at 5 m.
    return ChannelConfig(tx_power_dbm=19.5, pl0_db=40.2, exponent=2.0, **kw)


def _radio(positions, seed=7, **kw):
    return make_network_model(NetworkModelConfig(mode="radio", radio=RadioLinkConfig(positions=positions, **kw)),
                              seed=seed)


def test_radio_single_link_delay_is_difs_plus_airtime_plus_processing():
    """Hand check: one sender, idle medium, backoff off.

    Delay = DIFS (50 us) + airtime of a 32 B ESP-NOW frame (792 us) + processing.
    """
    phy = default_phy()
    mac = MacConfig(backoff=False, unicast_retry_limit=0, processing_delay_s=0.0005)
    model = _radio({"drone": (0, 0, 1), "gs": (5, 0, 1)}, channel=_clean_channel(), phy=phy, mac=mac,
                   unicast=False)
    for k in range(20):
        model.send("drone", "gs", k, 0.1 * k)
    delivered = model.deliver_all()
    assert len(delivered) == 20
    expected = phy.difs_s + data_airtime_s(32, phy) + 0.0005
    assert expected == pytest.approx(50e-6 + 792e-6 + 0.0005, abs=1e-12)
    for e in delivered:
        assert e.receive_time - e.send_time == pytest.approx(expected, abs=1e-9)
        assert e.outcome == "received" and not e.duplicate


def test_radio_unicast_with_ack_adds_no_receiver_delay_and_reports_retries():
    phy = default_phy()
    model = _radio({"drone": (0, 0, 1), "gs": (5, 0, 1)}, channel=_clean_channel(), phy=phy,
                   mac=MacConfig(backoff=False, unicast_retry_limit=3), unicast=True)
    for k in range(10):
        model.send("drone", "gs", k, 0.05 * k)
    delivered = model.deliver_all()
    assert [e.seq for e in delivered] == list(range(10))
    assert all(e.retry_count == 0 for e in model.events)


def test_radio_out_of_range_packets_are_lost_with_a_reason():
    model = _radio({"drone": (0, 0, 1), "gs": (1e6, 0, 1)}, channel=_clean_channel(), unicast=True)
    for k in range(5):
        model.send("drone", "gs", k, 0.1 * k)
    assert model.deliver_all() == []
    assert all(e.outcome == "lost" and e.loss_reason for e in model.events)


def test_radio_contention_from_background_traffic_costs_delay_or_delivery():
    """Twelve background senders at 50 Hz (channel use near 1) must hurt the tracked stream."""
    positions = {"drone": (0, 0, 1), "gs": (10, 0, 1)}
    background = []
    for i in range(12):
        name = f"bg{i}"
        positions[name] = (2.0 * i, 5.0, 1.0)
        background.append(BackgroundSource(name, "gs", 50.0, stop_s=5.0))
    quiet = _radio(dict(positions), channel=_clean_channel(), background=())
    busy = _radio(dict(positions), channel=_clean_channel(), background=tuple(background))
    for model in (quiet, busy):
        for k in range(40):
            model.send("drone", "gs", k, 0.5 + 0.1 * k)
        model.deliver_all()
    quiet_delay = np.mean([e.packet_age_s for e in quiet.events if e.outcome == "received"])
    busy_ok = [e for e in busy.events if e.outcome == "received"]
    busy_delay = np.mean([e.packet_age_s for e in busy_ok]) if busy_ok else np.inf
    assert len(busy_ok) < 40 or busy_delay > 2 * quiet_delay


def test_radio_mode_is_deterministic_per_seed():
    def run(seed):
        cfg = NetworkModelConfig(mode="radio", radio=RadioLinkConfig(
            positions={"drone": (0, 0, 1), "gs": (30, 0, 1), "x": (10, 10, 1)},
            background=(BackgroundSource("x", "gs", 40.0, stop_s=3.0),)))
        model = make_network_model(cfg, seed=seed)
        for k in range(25):
            model.send("drone", "gs", k, 0.1 * k)
        model.deliver_all()
        return _dicts(model.events)

    assert run(5) == run(5)


def test_events_export_as_a_valid_synthetic_trace(tmp_path):
    model = _radio({"drone": (0, 0, 1), "gs": (20, 0, 1)}, channel=_clean_channel(shadowing_sigma_db=3.0))
    for k in range(30):
        model.send("drone", "gs", k, 0.1 * k)
    model.deliver_all()
    doc = events_to_trace(model.events, trace_id="radio-model-test")
    schema = json.loads(open("sim/trace_schema.json").read())
    jsonschema.validate(doc, schema)
    trace = EspNowTrace.from_dict(doc)
    assert trace.status == "synthetic"
    assert len(trace.packets) == 30


def test_mode_switch_rejects_unknown_modes():
    with pytest.raises(ValueError):
        make_network_model(NetworkModelConfig(mode="mesh"), seed=1)
    with pytest.raises(ValueError):
        make_network_model(NetworkModelConfig(mode="radio"), seed=1)
