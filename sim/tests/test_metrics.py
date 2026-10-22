import numpy as np
import pytest
from sim.metrics import step_response, tracking_error, packet_delivery, packet_age, reacquisition_time, miss_distance_time, controller_recovery_time

def test_step_fixture_hand_answer():
    r=step_response([0,1,2,3,4],[0,1,2,3,3],3)
    assert r["overshoot_pct"]==0 and r["settling_time_s"]==3

def test_constant_vector_error_percentiles():
    r=tracking_error(np.zeros((20,3)),np.tile([3.,4.,0.],(20,1)))
    assert np.isclose(r["rms_m"],5) and np.isclose(r["p95_m"],5)

def test_unique_loss_and_duplicate_do_not_change_delivery():
    rows=[{"sender_id":"s","seq":i,"sent":True,"received":True} for i in range(3)]
    rows += [{"sender_id":"s","seq":1,"received":True}]
    r=packet_delivery(rows); assert r["received_unique"]==3 and r["duplicates"]==1 and r["loss_rate"]==0

def test_packet_delivery_accepts_simulated_link_event_schema_and_separates_routes():
    rows = [
        {"sender": "A", "receiver": "B", "seq": 0, "sent": True, "received": True},
        {"sender": "C", "receiver": "B", "seq": 0, "sent": True, "received": True},
    ]
    result = packet_delivery(rows)
    assert result["sent_unique"] == 2
    assert result["received_unique"] == 2
    assert result["duplicates"] == 0

def test_packet_delivery_infers_sent_and_received_from_link_events():
    rows = [
        {"sender": "A", "receiver": "B", "seq": 0, "send_time": 0.0, "receive_time": 0.1},
        {"sender": "A", "receiver": "B", "seq": 1, "send_time": 0.1, "receive_time": None},
    ]
    result = packet_delivery(rows)
    assert result["sent_unique"] == 2 and result["received_unique"] == 1
    assert result["loss_rate"] == 0.5

def test_guardrails_and_reacquisition():
    with pytest.raises(ValueError): packet_age(1,0,same_clock=False)
    assert reacquisition_time([0,1,2,3],[True,False,False,True])["time_s"]==2
    assert miss_distance_time([0,1,2],[3,2,1],capture_radius=1)["time_to_capture_s"]==2

def test_empty_metrics_are_deterministically_censored():
    r=step_response([], [], 1)
    assert r["count"] == 0 and all(r["censored"].values())
    assert reacquisition_time([], [], consecutive=1) == {"time_s":None,"censored":True}
    assert miss_distance_time([], [], capture_radius=1)["censored"]

def test_receive_only_packets_have_unknown_denominator():
    r=packet_delivery([{"sender_id":"s", "seq":7, "received":True}])
    assert r["sent_unique"] is None and r["lost_unique"] is None and r["loss_rate"] is None

def test_explicit_denominator_uses_observed_sequence_origin_and_duplicates():
    rows=[{"sender_id":"s","seq":10,"received":True}, {"sender_id":"s","seq":12,"received":True}, {"sender_id":"s","seq":12,"received":True}]
    r=packet_delivery(rows, sent_denominator=3)
    assert r["missing_sequences"] == [11] and r["duplicates"] == 1

def test_invalid_nonfinite_and_consecutive_inputs_rejected():
    with pytest.raises(ValueError): step_response([0,1], [0,np.nan], 1)
    with pytest.raises(ValueError): miss_distance_time([0,1], [0,np.inf])
    with pytest.raises(ValueError): reacquisition_time([0,1], [1,np.nan])
    with pytest.raises(ValueError): reacquisition_time([0,1], [1,1], consecutive=0)
    with pytest.raises(ValueError): controller_recovery_time([0,1], [0,2])

def test_closest_sampled_miss_distance_and_zero_step():
    assert miss_distance_time([0,1,2], [4,1.5,2], capture_radius=1)["miss_distance_m"] == 1.5
    r=step_response([0,1], [2,2], 2)
    assert all(r["censored"].values())

def test_missing_packets_use_actual_routes_and_noncontiguous_sent_ids():
    rows = [
        {"sender": "A", "receiver": "B", "seq": 10, "sent": True, "received": True},
        {"sender": "A", "receiver": "B", "seq": 40, "sent": True},
        {"sender": "C", "receiver": "B", "seq": 10, "sent": True},
    ]
    result = packet_delivery(rows)
    assert result["lost_unique"] == 2 and result["loss_rate"] == pytest.approx(2 / 3)
    assert result["missing_sequences"] is None
    assert result["missing_packets"] == [
        {"sender_id": "A", "receiver_id": "B", "seq": 40},
        {"sender_id": "C", "receiver_id": "B", "seq": 10},
    ]
    assert packet_delivery(rows[:2])["missing_sequences"] == [40]


def test_explicit_denominator_does_not_invent_sequence_gaps_across_routes():
    rows = [
        {"sender": "A", "receiver": "B", "seq": 10, "received": True},
        {"sender": "C", "receiver": "B", "seq": 10, "received": True},
    ]
    result = packet_delivery(rows, sent_denominator=3)
    assert result["lost_unique"] == 1
    assert result["missing_sequences"] is None and result["missing_packets"] is None


def test_out_of_order_uses_route_high_water_and_excludes_duplicate_arrivals():
    rows = [{"sender": "A", "receiver": "B", "seq": seq, "received": True}
            for seq in (3, 1, 2, 1)]
    rows.append({"sender": "C", "receiver": "B", "seq": 1, "received": True})
    result = packet_delivery(rows)
    assert result["out_of_order"] == 2 and result["duplicates"] == 1


def test_tracking_error_scalar_series_has_independent_rms_and_percentiles():
    result = tracking_error([0, 0], [-3, 4])
    assert result["count"] == 2 and not result["empty"]
    assert result["rms_m"] == pytest.approx(np.sqrt(12.5))
    assert result["median_m"] == 3.5
    assert result["per_axis"]["rms_m"] == pytest.approx([np.sqrt(12.5)])


@pytest.mark.parametrize("shape", [(0,), (0, 3)])
def test_tracking_error_equal_empty_arrays_report_no_samples(shape):
    result = tracking_error(np.empty(shape), np.empty(shape))
    assert result["empty"] and result["count"] == 0 and result["rms_m"] is None


@pytest.mark.parametrize("truth,estimate", [([0], []), ([0], [[0]]), ([[0, 0]], [[0]])])
def test_tracking_error_shape_mismatch_is_invalid_not_empty(truth, estimate):
    with pytest.raises(ValueError, match="equal shape"):
        tracking_error(truth, estimate)


def test_censored_recovery_and_capture_do_not_report_success():
    assert controller_recovery_time([0, 1, 2], [0, 0, 0]) == {"time_s": None, "censored": True}
    assert controller_recovery_time([], []) == {"time_s": None, "censored": True}
    assert reacquisition_time([0, 1, 2], [0, 0, 1], consecutive=2) == {"time_s": None, "censored": True}
    result = miss_distance_time([0, 1, 2], [3, 2, 1.5], capture_radius=1)
    assert result == {"miss_distance_m": 1.5, "time_to_capture_s": None, "censored": True}
