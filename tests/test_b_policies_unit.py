"""Fast unit tests for the sender policies (quantizer, periodic, event, packet sizes)."""
import numpy as np
import pytest

from sim.estimation.base import EstimatorSettings
from sim.estimation.policies import SenderOutput, apply_sender, dequantize, packet_bytes, quantize

SETTINGS = EstimatorSettings()


def _stream(n=201, rate=50.0, noise=0.05, seed=3, speed=1.0):
    t = np.arange(n) / rate
    truth = np.stack([speed * t, np.zeros(n), np.zeros(n)], axis=1)
    rng = np.random.default_rng(seed)
    return t, truth + rng.normal(0.0, noise, truth.shape)


def test_quantize_codes_are_int16_and_clip():
    q = 0.001
    codes, z_hat = quantize(np.array([[1e9, -1e9, 0.0004]]), q)
    assert codes.dtype == np.int16
    assert list(codes[0]) == [32767, -32768, 0]
    assert z_hat[0, 0] == pytest.approx(32767 * q)
    assert z_hat[0, 1] == pytest.approx(-32768 * q)
    assert np.array_equal(dequantize(codes, q), z_hat)


def test_quantize_error_is_uniform_with_std_q_over_sqrt12():
    q = 0.01
    rng = np.random.default_rng(11)
    z = rng.uniform(-100.0, 100.0, size=(100_000, 3))
    err = quantize(z, q)[1] - z
    assert np.all(np.abs(err) <= q / 2 + 1e-12)
    expected = q / np.sqrt(12.0)
    assert np.std(err, axis=0) == pytest.approx([expected] * 3, rel=0.02)
    assert np.all(np.abs(err.mean(axis=0)) < 0.01 * q)


def test_quantize_rejects_bad_step():
    for q in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            quantize([1.0, 2.0, 3.0], q)


def test_periodic_ten_hz_from_fifty_hz_sends_every_fifth():
    t, z = _stream(n=51)
    out = apply_sender(t, z, {"kind": "periodic", "rate_hz": 10.0}, SETTINGS, 0)
    assert isinstance(out, SenderOutput)
    assert list(out.send_indices) == list(range(0, 51, 5))
    assert np.array_equal(out.sent_measurements, z[out.send_indices])


def test_default_sender_sends_everything_as_float32_sized_packets():
    t, z = _stream(n=30)
    out = apply_sender(t, z, {}, SETTINGS, 0)
    assert list(out.send_indices) == list(range(30))
    assert out.data_bytes_per_packet == 18
    assert out.on_air_bytes_per_packet == 18 + 43


def test_packet_bytes_model():
    assert packet_bytes(False) == {"data": 18, "on_air": 61}
    assert packet_bytes(True) == {"data": 12, "on_air": 55}
    t, z = _stream(n=10)
    out = apply_sender(t, z, {"quantization_m": 0.001}, SETTINGS, 0)
    assert (out.data_bytes_per_packet, out.on_air_bytes_per_packet) == (12, 55)


def test_quantized_output_is_on_the_grid():
    q = 0.005
    t, z = _stream(n=40)
    out = apply_sender(t, z, {"quantization_m": q}, SETTINGS, 0)
    assert np.allclose(out.sent_measurements / q, np.round(out.sent_measurements / q), atol=1e-9)
    assert np.max(np.abs(out.sent_measurements - z)) <= q / 2 + 1e-12


@pytest.mark.parametrize("noise", [0.05, 0.0])
def test_event_with_zero_delta_sends_every_sample(noise):
    t, z = _stream(noise=noise)
    event = apply_sender(t, z, {"kind": "event", "delta_m": 0.0}, SETTINGS, 0)
    periodic = apply_sender(t, z, {"kind": "periodic", "rate_hz": 50.0}, SETTINGS, 0)
    assert np.array_equal(event.send_indices, periodic.send_indices)
    assert event.send_indices.size == t.size
    assert np.array_equal(event.sent_measurements, periodic.sent_measurements)


def test_event_sends_fewer_packets_as_delta_grows():
    t, z = _stream(noise=0.05)
    counts = [apply_sender(t, z, {"kind": "event", "delta_m": d}, SETTINGS, 0).send_indices.size
              for d in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 5.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == t.size
    assert counts[-1] < counts[0] and counts[-1] >= 1
    first = apply_sender(t, z, {"kind": "event", "delta_m": 5.0}, SETTINGS, 0)
    assert first.send_indices[0] == 0


def test_event_first_sample_is_always_sent_and_indices_increase():
    t, z = _stream()
    out = apply_sender(t, z, {"kind": "event", "delta_m": 0.1}, SETTINGS, 0)
    assert out.send_indices[0] == 0
    assert np.all(np.diff(out.send_indices) > 0)


def test_event_quantized_value_is_what_the_replica_fuses():
    # With a huge step the sent values collapse to the grid and the trigger
    # still sees the raw measurement, so the output stays on the grid.
    q = 0.5
    t, z = _stream(noise=0.05)
    out = apply_sender(t, z, {"kind": "event", "delta_m": 0.3, "quantization_m": q}, SETTINGS, 0)
    assert np.allclose(out.sent_measurements / q, np.round(out.sent_measurements / q))


def test_heartbeat_bounds_the_gap_between_sends():
    t, z = _stream(n=501, noise=0.0)  # perfect line: the replica soon predicts it, so few events
    plain = apply_sender(t, z, {"kind": "event", "delta_m": 0.5}, SETTINGS, 0)
    heart = apply_sender(t, z, {"kind": "event", "delta_m": 0.5, "heartbeat_s": 0.5}, SETTINGS, 0)
    def longest_silence(out):
        stamps = np.r_[t[out.send_indices], t[-1]]  # include the quiet stretch at the end
        return np.diff(stamps).max()

    assert longest_silence(plain) > 2.0  # only meaningful if the heartbeat matters
    assert longest_silence(heart) <= 0.5 + 1e-9
    assert heart.send_indices.size > plain.send_indices.size


@pytest.mark.parametrize("params", [
    {"bogus": 1},
    {"kind": "weird"},
    {"kind": "event"},
    {"kind": "event", "delta_m": -0.1},
    {"kind": "event", "delta_m": 0.1, "rate_hz": 10.0},
    {"kind": "periodic", "delta_m": 0.1},
    {"kind": "periodic", "heartbeat_s": 1.0},
    {"kind": "periodic", "rate_hz": 0.0},
    {"kind": "periodic", "rate_hz": 500.0},
    {"quantization_m": 0.0},
    {"kind": "event", "delta_m": 0.1, "heartbeat_s": 0.0},
])
def test_bad_params_raise(params):
    t, z = _stream(n=20)
    with pytest.raises(ValueError):
        apply_sender(t, z, params, SETTINGS, 0)


def test_bad_arrays_raise():
    t, z = _stream(n=20)
    with pytest.raises(ValueError):
        apply_sender(t, z[:, :2], {}, SETTINGS, 0)
    with pytest.raises(ValueError):
        apply_sender(t[::-1], z, {}, SETTINGS, 0)


@pytest.mark.parametrize("rate", [0.0, -5.0, float("nan"), float("inf")])
def test_one_sample_periodic_sender_still_validates_the_rate(rate):
    """With one sample the sensor rate cannot be inferred, but a bad rate must still raise."""
    import numpy as np
    from sim.estimation.base import EstimatorSettings
    from sim.estimation.policies import apply_sender
    with pytest.raises(ValueError):
        apply_sender(np.array([0.0]), np.zeros((1, 3)), {"kind": "periodic", "rate_hz": rate}, EstimatorSettings(), 0)
