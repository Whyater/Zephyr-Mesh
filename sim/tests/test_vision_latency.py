"""Tests for the on-screen pattern latency method (synthetic recordings only, no camera).

The main check is independent of the analysis code: synth_recording builds
frames from a known, injected latency using its own clock bookkeeping, and the
analysis has to get that latency back within the error it states itself.
"""
import dataclasses

import numpy as np
import pytest

from sim.vision_latency import (
    Decoded,
    DisplayLog,
    Settings,
    SynthConfig,
    analyze,
    code_bits,
    decode_cells,
    find_strip_roi,
    gray_decode,
    gray_encode,
    match_frame,
    pattern_strip_box,
    render_pattern,
    sample_cells,
    synth_recording,
)


def run(cfg: SynthConfig, **settings):
    rec = synth_recording(cfg)
    settings.setdefault("display_lag_s", (cfg.display_lag_s, cfg.display_lag_s))
    settings.setdefault("n_bits", cfg.n_bits)
    settings.setdefault("bootstrap", 1000)
    return rec, analyze(rec.frames, rec.t_read_s, rec.log, Settings(**settings))


def apparent_truth(rec, cfg):
    """True apparent latency: camera latency plus display lag, mean over frames."""
    return float(np.mean(rec.true_latency_s)) + cfg.display_lag_s


# --- Gray code ------------------------------------------------------------------

def test_gray_neighbours_differ_in_exactly_one_bit_including_the_wrap():
    n = 8
    for k in range(1 << n):
        a, b = code_bits(k, n), code_bits((k + 1) % (1 << n), n)
        assert int(np.sum(a != b)) == 1


def test_gray_round_trip():
    for k in range(4096):
        assert gray_decode(gray_encode(k)) == k


# --- decoding ---------------------------------------------------------------------

def test_render_then_decode_every_code_both_ways_round():
    n = 8
    box = pattern_strip_box(640, 360)
    for code in range(0, 256, 7):
        image = render_pattern(code, n, 640, 360)
        assert decode_cells(sample_cells(image, box, n), n).codes == (code,)
        mirrored = image[:, ::-1]
        w = image.shape[1]
        mbox = (w - box[2], box[1], w - box[0], box[3])
        decoded = decode_cells(sample_cells(mirrored, mbox, n), n)
        assert decoded.codes == (code,) and decoded.mirrored


def test_one_grey_cell_gives_the_two_neighbouring_codes():
    n = 8
    # Average the cells for codes 41 and 42: exactly one bit cell turns grey.
    a = np.concatenate(([200.0, 30.0], 30 + 170 * code_bits(41, n), [30.0, 30.0]))
    b = np.concatenate(([200.0, 30.0], 30 + 170 * code_bits(42, n), [30.0, 30.0]))
    assert decode_cells(0.5 * (a + b), n).codes == (41, 42)
    # Across the wrap: 255 and 0.
    a = np.concatenate(([200.0, 30.0], 30 + 170 * code_bits(255, n), [30.0, 30.0]))
    b = np.concatenate(([200.0, 30.0], 30 + 170 * code_bits(0, n), [30.0, 30.0]))
    assert decode_cells(0.5 * (a + b), n).codes == (255, 0)


def test_two_grey_cells_are_rejected():
    n = 8
    a = np.concatenate(([200.0, 30.0], 30 + 170 * code_bits(40, n), [30.0, 30.0]))
    c = np.concatenate(([200.0, 30.0], 30 + 170 * code_bits(42, n), [30.0, 30.0]))
    decoded = decode_cells(0.5 * (a + c), n)
    assert decoded.codes == () and "grey cells" in decoded.reason


def test_flat_frame_is_rejected_for_no_contrast():
    decoded = decode_cells(np.full(12, 100.0), 8)
    assert decoded.codes == () and "contrast" in decoded.reason


# --- matching, checked against a hand calculation -------------------------------------

def test_match_frame_hand_calculation():
    # Codes 0, 1, 2, 3, 4 go up at 0.00, 0.05, 0.11, 0.16, 0.22 s.
    log = DisplayLog(np.arange(5), np.arange(5), np.array([0.00, 0.05, 0.11, 0.16, 0.22]), 8)
    # A frame read at 0.20 s that shows code 1. Code 1 was up from 0.05 to 0.11,
    # so the frame is between 0.20 - 0.11 = 0.09 and 0.20 - 0.05 = 0.15 s old,
    # best guess the middle, 0.12 s.
    sample, _ = match_frame(Decoded((1,), False, 170.0), 0.20, log, max_latency_s=2.0)
    assert (sample.lo_s, sample.hi_s, sample.mid_s) == pytest.approx((0.09, 0.15, 0.12))
    # The same read showing codes 1 and 2 caught the change at 0.11 s:
    # best guess 0.20 - 0.11 = 0.09, bounds 0.20 - 0.16 = 0.04 to 0.15.
    sample, _ = match_frame(Decoded((1, 2), False, 170.0), 0.20, log, max_latency_s=2.0)
    assert sample.straddle
    assert (sample.lo_s, sample.hi_s, sample.mid_s) == pytest.approx((0.04, 0.15, 0.09))


def test_match_frame_rejects_codes_not_yet_shown_and_the_open_last_event():
    log = DisplayLog(np.arange(3), np.arange(3), np.array([0.0, 0.05, 0.10]), 8)
    assert match_frame(Decoded((2,), False, 170.0), 0.04, log, max_latency_s=2.0)[0] is None
    sample, reason = match_frame(Decoded((2,), False, 170.0), 0.30, log, max_latency_s=2.0)
    assert sample is None and "no end time" in reason


# --- recovering a known, injected latency -------------------------------------------------

@pytest.mark.parametrize("fps", [15.0, 30.0, 60.0])
@pytest.mark.parametrize("latency", [0.0, 0.033, 0.100, 0.200, 0.300])
def test_recovers_injected_latency_within_stated_error(latency, fps):
    # 15 fps gets a longer run: too few frames and the locking guard (rightly) switches on.
    cfg = SynthConfig(latency_s=latency, fps=fps, duration_s=20.0 if fps < 20 else 12.0,
                      seed=int(1000 * latency + fps))
    rec, result = run(cfg)
    a = result["apparent_latency_s"]
    assert result["frames_used"] >= 0.9 * result["frames_total"]
    assert abs(a["mean"] - apparent_truth(rec, cfg)) <= a["stated_error"]
    # The stated error should be small compared with one camera frame,
    # otherwise the method is not telling us much.
    assert a["stated_error"] < 0.25 / fps


def test_short_slow_run_switches_on_the_locking_guard_and_still_covers_the_truth():
    cfg = SynthConfig(latency_s=0.0, fps=15.0, duration_s=10.0, seed=15)
    rec, result = run(cfg)
    a = result["apparent_latency_s"]
    assert not result["display_holds_s"]["random"]
    assert a["stated_error_parts"]["locking_guard"] > 0
    assert abs(a["mean"] - apparent_truth(rec, cfg)) <= a["stated_error"]


def test_limiting_case_zero_latency_zero_lag_gives_zero():
    cfg = SynthConfig(latency_s=0.0, display_lag_s=0.0, duration_s=12.0, seed=5)
    _rec, result = run(cfg)
    a = result["apparent_latency_s"]
    assert abs(a["mean"]) <= a["stated_error"]
    assert result["camera_latency_s"]["low"] <= 0.0 <= result["camera_latency_s"]["high"]


def test_limiting_case_instant_display_changes_and_instant_exposure_is_exact():
    # Codes change every 1 ms and the shutter is instant, so each frame pins its
    # age to within half a millisecond. The answer must be essentially exact.
    cfg = SynthConfig(latency_s=0.150, exposure_s=0.0, hold_s=(0.001, 0.001), duration_s=6.0,
                      n_bits=10, seed=11)
    # 1024 codes at 1 ms each repeat after about 1 s, so the latency assumption must be below that.
    # The shutter is instant, so no widening for exposure is needed either.
    _rec, result = run(cfg, max_latency_s=0.5, exposure_max_s=0.0)
    a = result["apparent_latency_s"]
    assert a["per_frame_half_width_mean"] == pytest.approx(0.0005, abs=1e-6)
    assert abs(a["mean"] - 0.150) <= 0.0006
    assert abs(a["mean"] - 0.150) <= a["stated_error"]


def test_display_lag_range_brackets_the_true_camera_latency():
    cfg = SynthConfig(latency_s=0.110, display_lag_s=0.030, duration_s=12.0, seed=21)
    rec, result = run(cfg, display_lag_s=(0.0, 0.050))
    a = result["apparent_latency_s"]
    # The apparent latency includes the display lag ...
    assert abs(a["mean"] - 0.140) <= a["stated_error"] + 1e-3
    # ... and the bounds on the camera latency contain the true 110 ms.
    cam = result["camera_latency_s"]
    true_cam = float(np.mean(rec.true_latency_s))
    assert cam["low"] <= true_cam <= cam["high"]
    assert cam["high"] - cam["low"] == pytest.approx(0.050 + 2 * a["stated_error"])


def test_phase_locked_fixed_holds_switch_on_the_locking_guard_and_still_cover_the_truth():
    # Fixed 2-frame holds at 30 fps, with the shutter always just after a change:
    # every midpoint is off the same way, so the scatter alone would understate the error.
    cfg = SynthConfig(latency_s=0.090, fps=30.0, hold_s=(2 / 30, 2 / 30), exposure_s=0.002,
                      camera_phase_s=0.004, duration_s=12.0, seed=31)
    rec, result = run(cfg)
    a = result["apparent_latency_s"]
    assert not result["display_holds_s"]["random"]
    assert not any("not longer than exposure_max_s" in w for w in result["warnings"])
    assert a["stated_error_parts"]["locking_guard"] > 0.02
    assert abs(a["mean"] - apparent_truth(rec, cfg)) <= a["stated_error"]


def test_long_exposure_and_heavy_noise_still_recover():
    cfg = SynthConfig(latency_s=0.180, exposure_s=0.033, noise_std=12.0, duration_s=12.0, seed=41)
    rec, result = run(cfg)
    a = result["apparent_latency_s"]
    assert result["straddle_fraction"] > 0.2
    assert result["frames_used"] >= 0.8 * result["frames_total"]
    assert abs(a["mean"] - apparent_truth(rec, cfg)) <= a["stated_error"]


def test_mirrored_recording_is_detected_and_recovered():
    cfg = SynthConfig(latency_s=0.070, mirrored=True, duration_s=12.0, seed=51)
    rec, result = run(cfg)
    assert result["mirrored_fraction"] == 1.0
    assert abs(result["apparent_latency_s"]["mean"] - apparent_truth(rec, cfg)) <= result["apparent_latency_s"]["stated_error"]


@pytest.mark.parametrize("seed", [61, 62, 63, 64])
def test_jittery_latency_mean_and_jitter_from_straddle_frames(seed):
    # 20 ms of one-sided jitter: the latency distribution is lopsided, which is
    # where a median would drift and the mean must not.
    cfg = SynthConfig(latency_s=0.100, latency_jitter_s=0.020, duration_s=20.0, seed=seed)
    rec, result = run(cfg)
    a = result["apparent_latency_s"]
    assert abs(a["mean"] - apparent_truth(rec, cfg)) <= a["stated_error"]
    true_std = float(np.std(rec.true_latency_s))
    assert a["straddle_frames"] >= 10
    assert 0.7 * true_std <= a["jitter_std_from_straddles"] <= 1.3 * true_std


def test_no_jitter_gives_a_small_jitter_estimate():
    cfg = SynthConfig(latency_s=0.100, duration_s=20.0, seed=65)
    _rec, result = run(cfg)
    # Straddle midpoints are off by at most a quarter of the 10 ms exposure.
    assert result["apparent_latency_s"]["jitter_std_from_straddles"] < 0.003


def test_warnings_for_a_short_run_and_dropped_frames():
    cfg = SynthConfig(latency_s=0.100, duration_s=12.0, seed=66)
    rec = synth_recording(cfg)
    keep = np.ones(len(rec.frames), dtype=bool)
    keep[100:104] = False   # four frames lost in a row
    result = analyze(rec.frames[keep], rec.t_read_s[keep], rec.log, Settings(roi=cfg.strip_box, bootstrap=300))
    assert any("dropped frames" in w for w in result["warnings"])
    # A 5 s run has fewer than 128 changes, so the top bit never changes and the
    # automatic box misses it. Nothing decodes; the warning must say why.
    short = SynthConfig(latency_s=0.100, duration_s=5.0, seed=67)
    rec = synth_recording(short)
    result = analyze(rec.frames, rec.t_read_s, rec.log, Settings(bootstrap=300))
    assert "error" in result
    assert any("top bit never changed" in w for w in result["warnings"])
    # With the box given by hand the same short run works.
    result = analyze(rec.frames, rec.t_read_s, rec.log, Settings(roi=short.strip_box, bootstrap=300))
    assert "error" not in result


def test_clean_frame_bounds_are_widened_for_exposure():
    log = DisplayLog(np.arange(5), np.arange(5), np.array([0.00, 0.05, 0.11, 0.16, 0.22]), 8)
    sample, _ = match_frame(Decoded((1,), False, 170.0), 0.20, log, max_latency_s=2.0, widen_s=0.008)
    assert (sample.lo_s, sample.hi_s, sample.mid_s) == pytest.approx((0.082, 0.158, 0.12))
    sample, _ = match_frame(Decoded((1, 2), False, 170.0), 0.20, log, max_latency_s=2.0, widen_s=0.008)
    assert (sample.lo_s, sample.hi_s, sample.mid_s) == pytest.approx((0.04, 0.15, 0.09))


def test_codes_wrap_around_many_times():
    # 5 bits = 32 codes, about 1.8 s per cycle with the default holds.
    cfg = SynthConfig(latency_s=0.300, n_bits=5, duration_s=12.0, seed=71)
    rec, result = run(cfg, max_latency_s=1.0)
    assert rec.log.seq[-1] > 3 * 32
    assert abs(result["apparent_latency_s"]["mean"] - apparent_truth(rec, cfg)) <= result["apparent_latency_s"]["stated_error"]


def test_refuses_to_run_when_codes_repeat_faster_than_the_latency_assumption():
    # 32 codes at 40 to 70 ms repeat after about 1.8 s. A frame 2.5 s old would
    # quietly match a later event, so analyze must refuse rather than misread.
    cfg = SynthConfig(latency_s=2.5, n_bits=5, duration_s=12.0, seed=81)
    rec = synth_recording(cfg)
    with pytest.raises(ValueError, match="codes repeat"):
        analyze(rec.frames, rec.t_read_s, rec.log, Settings(n_bits=5, max_latency_s=5.0))


def test_default_eight_bits_leave_room_for_the_default_latency_assumption():
    # 256 codes at no less than 40 ms each repeat after at least 10 s, far past 2 s.
    assert 256 * SynthConfig().hold_s[0] > 5 * Settings().max_latency_s


def test_hidden_strip_frames_are_counted_as_rejected():
    cfg = SynthConfig(latency_s=0.120, duration_s=12.0, seed=91)
    rec = synth_recording(cfg)
    frames = rec.frames.copy()
    x0, y0, x1, y1 = cfg.strip_box
    frames[::10, y0:y1, x0:x1] = 110   # someone's hand in front of the mirror
    result = analyze(frames, rec.t_read_s, rec.log, Settings(roi=cfg.strip_box, bootstrap=500))
    hidden = len(frames[::10])
    assert sum(result["frames_rejected"].values()) == hidden
    assert result["frames_used"] == len(frames) - hidden


def test_find_strip_roi_matches_where_the_strip_is():
    cfg = SynthConfig(duration_s=12.0, seed=3)
    rec = synth_recording(cfg)
    x0, y0, x1, y1 = find_strip_roi(rec.frames, cfg.n_bits)
    tx0, ty0, tx1, ty1 = cfg.strip_box
    cell = (tx1 - tx0) / (cfg.n_bits + 4)
    assert abs(x0 - tx0) <= 0.3 * cell and abs(x1 - tx1) <= 0.3 * cell
    assert abs(y0 - ty0) <= 2 and abs(y1 - ty1) <= 2


# --- input checks ------------------------------------------------------------------

def test_bad_inputs_raise():
    cfg = SynthConfig(duration_s=4.0, seed=2)
    rec = synth_recording(cfg)
    with pytest.raises(ValueError):
        analyze(rec.frames[:-1], rec.t_read_s, rec.log)
    with pytest.raises(ValueError):
        analyze(rec.frames, rec.t_read_s[::-1], rec.log)
    with pytest.raises(ValueError):
        analyze(rec.frames, rec.t_read_s, rec.log, Settings(display_lag_s=(0.05, 0.0)))
    with pytest.raises(ValueError):
        DisplayLog(np.array([0, 2]), np.array([0, 2]), np.array([0.0, 1.0]), 8)
    with pytest.raises(ValueError):
        DisplayLog(np.array([0, 1]), np.array([0, 1]), np.array([1.0, 1.0]), 8)
    with pytest.raises(ValueError):
        sample_cells(rec.frames[0], (0, 0, 10_000, 10), 8)


def test_settings_are_plain_dataclasses():
    # The CLI builds Settings from arguments; make sure replace works for it.
    assert dataclasses.replace(Settings(), n_bits=6).n_bits == 6


def test_confidence_is_labelled_approximate_in_json_and_summary():
    from tools.vision_latency import summary_text
    _rec, result = run(SynthConfig(latency_s=0.100, duration_s=12.0, seed=66))
    assert "approximate" in result["settings"]["confidence_note"]
    assert "97%" in result["settings"]["confidence_note"]
    assert "approximately" in summary_text(result)


def test_jitter_estimate_is_none_with_fewer_than_ten_straddle_frames():
    cfg = SynthConfig(latency_s=0.100, exposure_s=0.002, fps=30.0, duration_s=8.0, seed=67)
    _rec, result = run(cfg)
    a = result["apparent_latency_s"]
    assert a["straddle_frames"] < 10
    assert a["jitter_std_from_straddles"] is None
