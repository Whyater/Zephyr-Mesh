import numpy as np

from sim.missions.comms import LossProcess, MissionLink, OutageWindow
from sim.missions.presets import PRESET_NAMES, get_preset


def test_presets_and_long_run_rates():
    assert PRESET_NAMES == (
        "clean", "iid_1", "iid_5", "iid_10", "iid_30",
        "burst_1_0.5s", "burst_10_0.2s", "burst_10_0.5s", "burst_30_0.5s",
    )
    for name in PRESET_NAMES:
        preset = get_preset(name)
        process = LossProcess(preset, seed=17, duration_s=2000.0)
        values = np.array([process.is_lost("cmd:a", i, i * 0.01) for i in range(200000)])
        if preset.loss_model == "burst":
            assert abs(values.mean() - preset.loss_probability) < 0.01
        elif preset.loss_model == "none":
            assert not values.any()
        else:
            se = np.sqrt(preset.loss_probability * (1 - preset.loss_probability) / values.size)
            assert abs(values.mean() - preset.loss_probability) <= 3 * se


def test_burst_run_length_and_shared_environment():
    process = LossProcess("burst_10_0.5s", seed=1, duration_s=2000.0)
    bad = np.array([process.is_lost("cmd:a", i, i * 0.01) for i in range(200000)], dtype=bool)
    runs = []
    i = 0
    while i < bad.size:
        if bad[i]:
            j = i
            while j < bad.size and bad[j]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    assert abs(np.mean(runs) - 50.0) / 50.0 < 0.10
    assert all(process.is_lost("cmd:a", i, i * 0.01) == process.is_lost("tel:a", i, i * 0.01)
               for i in range(200000))


def test_outage_determinism_and_clean_delay():
    loss = LossProcess("clean", seed=4, duration_s=2.0)
    outage = OutageWindow(0.5, 1.0, streams=("cmd:a",))
    link = MissionLink("cmd:a", delay_s=0.1, loss=loss, outages=(outage,), seed=8)
    for seq in range(20):
        link.send(seq, seq * 0.05, {"seq": seq})
    first = link.events()
    assert all(e["reason"] == "outage" for e in first if 0.5 <= e["send_time_s"] < 1.0)
    assert all(e["reason"] is None for e in first if not 0.5 <= e["send_time_s"] < 1.0)
    again = MissionLink("cmd:a", delay_s=0.1, loss=LossProcess("clean", seed=4, duration_s=2.0),
                        outages=(outage,), seed=8)
    for seq in range(20):
        again.send(seq, seq * 0.05, {"seq": seq})
    assert first == again.events()
    clean = MissionLink("tel:a", delay_s=0.07, loss=LossProcess("clean", seed=5, duration_s=1.0), seed=9)
    clean.send(0, 0.2, "ok")
    packet = clean.deliveries_until(0.27)[0]
    assert packet.receive_time_s - packet.send_time_s == 0.07

