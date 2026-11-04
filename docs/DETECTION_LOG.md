# Detection logs and the vision noise fit

This page describes a line-based log that a camera plus a detector can write while it runs, and a script that turns one such log into the noise settings the simulator already uses.

**Status: proposal, tested on synthetic data only.** The repo has no real camera log yet, so this page doesn't say how well any real camera or detector performs.

Run every command from the repo root with the project's virtual environment set up and activated (see [INSTALL.md](INSTALL.md)), so that `python` is the environment's Python.

---

## Pieces

| Piece | File | Job |
| --- | --- | --- |
| Format | this page | One header line, then one line per observation |
| Reader and writer | `sim/detection_log.py` | Reads a log and stops at the first bad line, naming its line number |
| Fit | `sim/vision_noise_fit.py`, `tools/fit_vision_noise.py` | Turns one log with ground truth into bias, spread, dropout and latency, each with an interval, and checks for common camera log problems |
| Synthetic logs | `tools/make_detection_fixture.py` | Writes a log from known settings using the sim's own sensor model, so you can check that the fit gets them back |

---

## The format: `zephyr-detection-log-1`

A detection log is a UTF-8 text file with the `.jsonl` extension. Every line is one JSON object, and lines end with a plain newline (LF).

It is the existing `zephyr-vision-trace-1` vision trace split into lines, so the field rules are the ones `sim/vision_trace.py` already checks (the full schema is `sim/vision_trace_schema.json`). A camera program can append one line per frame and target while it runs. The format allows several targets per frame, told apart by `subject_id`.

### Line 1: the header

```json
{"schema":"zephyr-detection-log-1","trace_id":"practice-run-1","status":"synthetic","clock_domain":"shared_monotonic","capture":{"source_type":"camera","source_id":"desk-camera","device_ids":["laptop"],"model":"marker-detector","resolution_px":{"width":1280,"height":720},"frame_rate_hz":30}}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `schema` | yes | Always `zephyr-detection-log-1` |
| `trace_id` | yes | A name for this run |
| `status` | yes | `measured`, `synthetic` or `fixture` |
| `clock_domain` | yes | `shared_monotonic` if capture and processing times come from the same clock; otherwise `device_and_host` or `unknown` |
| `capture` | yes | `source_type` (`camera`, `tracker` or `recording`), `source_id`, `device_ids` (a list), `model`, and optionally `calibration_id`, `resolution_px`, `frame_rate_hz` |
| `source`, `provenance` | yes when `status` is `measured` | See the next section |
| `code_revision`, `clock_uncertainty_s`, `notes` | no | Free text, a number in seconds, free text |

### Extra header fields for a measured log

A log marked `"status":"measured"` must also have `source` (text) and a `provenance` object with these keys:

| Key | Rule |
| --- | --- |
| `instrument` | An object with exactly `kind`, `model` and `settings` (`settings` is an object) |
| `calibration_ref` | The SHA-256 hash (64 hex characters) of the camera calibration file, or `null` together with `calibration_ref_null_reason` (text) |
| `raw_sha256` | The SHA-256 hash of the raw recording (64 hex characters) |
| `code_revision` | Text, such as a git commit |
| `captured_at` | An ISO 8601 time with a UTC offset, such as `2026-11-01T14:00:00-06:00` |
| `operator` | Text |
| `ground_truth` | An object with `method` (text), `values` (object) and `uncertainty` (object), or `null` together with `ground_truth_null_reason` |

Here is a header that passes the reader. **All of these values are made up for illustration.** Replace them with your own.

```json
{"schema":"zephyr-detection-log-1","trace_id":"example-measured-header","status":"measured","clock_domain":"shared_monotonic","source":"example-run.jsonl","capture":{"source_type":"camera","source_id":"desk-camera","device_ids":["laptop"],"model":"marker-detector","resolution_px":{"width":1280,"height":720},"frame_rate_hz":30},"provenance":{"instrument":{"kind":"camera","model":"example-camera","settings":{"exposure":"auto"}},"calibration_ref":null,"calibration_ref_null_reason":"example only, no calibration file","raw_sha256":"0000000000000000000000000000000000000000000000000000000000000000","code_revision":"example","captured_at":"2026-11-01T14:00:00-06:00","operator":"example","ground_truth":{"method":"tape measure","values":{"distance_m":2.0},"uncertainty":{"distance_m":0.01}}}}
```

### Every other line: one observation

```json
{"observation_id":"obs-0001","frame_id":"frame-0001","subject_id":"marker-1","capture_time_s":1.000,"processing_time_s":1.012,"outcome":"detected","confidence":0.94,"position_m":[0.20,-0.10,2.40],"position_frame":"camera","ground_truth":{"position_m":[0.20,-0.10,2.39],"uncertainty_m":0.01,"method":"tape"}}
{"observation_id":"obs-0002","frame_id":"frame-0002","subject_id":"marker-1","capture_time_s":1.033,"processing_time_s":1.047,"outcome":"missed","miss_reason":"not_found"}
```

| Field | Reader needs it | Unit | Used by the fit for |
| --- | --- | --- | --- |
| `observation_id` | yes, unique in the file | text | |
| `frame_id` | yes | text | |
| `subject_id` | yes | text | choosing which target to fit |
| `capture_time_s` | yes | seconds | ordering, latency |
| `outcome` | yes | `detected`, `missed` or `unknown` | dropout |
| `processing_time_s` | no | seconds; on a `shared_monotonic` clock it may not be earlier than `capture_time_s` | latency |
| `confidence` | no, detections only | 0 to 1 | not used |
| `bbox_px` | a detection needs this or `position_m` | pixels: `x` and `y` zero or more, `width` and `height` above zero | not used |
| `position_m` | a detection needs this or `bbox_px` | meters, 3 numbers | bias and spread |
| `position_frame` | yes when `position_m` is present | text, such as `camera` | the fit needs one frame across the chosen subject's lines (the reader does not check this) |
| `miss_reason` | yes for `missed` | text | |
| `unknown_reason` | yes for `unknown` | text | |
| `ground_truth` | no, but the fit needs it (below) | an object with all three of `position_m` (meters, same frame as `position_m`), `uncertainty_m` (meters) and `method` | bias and spread; the largest `uncertainty_m` widens the bias intervals |
| `duplicate`, `out_of_order` | no | true or false | not used |

**The fit needs at least 2 `detected` lines that have both `position_m` and `ground_truth`.** With fewer, the fit stops with an error. Box-only logs and logs without ground truth can be read, but not fitted. The file doesn't record which frame `ground_truth` is in, so you must record it in the same frame as `position_m`.

`sim/detection_log_example.jsonl` is a small example: a header plus 10 observations, written by hand for the tests.

### Rejected lines

`read_detection_log` stops at the first problem and raises `DetectionLogError` with the file name and line number, like `run.jsonl:57: observation: missed observations require miss_reason`. It rejects:

- bytes that are not UTF-8, carriage returns (Windows CRLF line endings), and blank lines in the middle of the file
- a line that is not JSON, or JSON that is not an object
- the same key twice in one line
- `NaN`, `Infinity`, or numbers too large to be finite
- a missing or wrong header, or a header with an `observations` field
- any observation that breaks a `zephyr-vision-trace-1` rule (a detection with no position or box, a miss with no reason, and so on)
- a repeated `observation_id` (the error gives the line of the second one)
- a processing time earlier than its capture time on a shared clock
- an empty file, or a file that starts with a UTF-8 byte order mark (BOM)

If the bad line is the last line and the file doesn't end with a newline, the error adds that the recording was probably cut off mid-write (the camera program stopped or crashed). Delete that last line and read the log again. A bad line anywhere else doesn't get this hint (`_parse` in `sim/detection_log.py`).

The file may end with one final newline. `write_detection_log` writes the same format back, and reading what it writes gives the same trace.

---

## Fitting a log

```sh
PYTHONPATH=. python tools/fit_vision_noise.py run.jsonl --output run-fit.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--output PATH` | print to the screen | Where to write the result. It must not be the input file, and an existing file is replaced |
| `--subject ID` | none | Which `subject_id` to fit. Needed when the log has more than one |
| `--dropout-mode` | `independent` | `independent` or `burst`: which dropout model goes in `sensor_config.mode` |
| `--bootstrap N` | 1000 | Number of resamples for the intervals. 0 uses formulas instead |
| `--seed S` | 0 | Seed for the resampling. The same log and seed always give the same output |

A file that does not end in `.jsonl` is read as a `zephyr-vision-trace-1` JSON file. On any error the script prints the reason and exits with code 2. The intervals are 95%. Another level is only available from Python (`fit_vision_noise(..., confidence_level=0.9)`).

### What comes out

The output is one JSON file (`zephyr-vision-noise-fit-1`).

- `sensor_config` has exactly the fields of `SensorConfig` in `sim/sensor.py`, so `SensorConfig(**fit["sensor_config"])` works.
  - `bias_m`: the average of measured minus true position, per axis.
  - `noise_std_m`: one spread for all three axes, the square root of the average of the three per-axis variances.
  - `dropout_probability`: missed frames divided by detected plus missed frames. `unknown` lines are left out.
  - `burst_start_probability`, `burst_end_probability`: the same misses fitted as on/off bursts. They are filled in either way when the log has enough steps to fit them (otherwise they fall back to 0 and 1 and their intervals are empty). `mode` decides which one the sim uses.
  - `random_walk_std_m`: always 0. It is not fitted.
- `scenario_delay` holds `measurement_delay_s` (average latency) and `measurement_jitter_s` (its standard deviation). Copy them into a mission `Scenario` (`sim/missions/api.py`), which has fields with the same names. It is `null`, with a reason in `scenario_delay_null_reason`, when the clock domain is not `shared_monotonic` or fewer than 2 lines have `processing_time_s`.
- `uncertainty` gives each number an estimate, a low and high end, the method, and how many lines it used. Bias, spread and latency intervals come from resampling short blocks of neighbouring frames (a block bootstrap). The dropout interval uses the Wilson formula, and the burst intervals come from simulating the fitted on/off pattern many times.
- `per_axis` gives the spread on each axis, an outlier-resistant spread (1.4826 times the median absolute deviation), the share of large outliers, and the correlation between neighbouring frames.
- `assumptions` lists, in plain words, every assumption the fit made for this log, plus any warnings from the checks below. Read it.

### Checks for a real camera log

A real webcam log can be wrong in ways the format rules can't see: frames that never got a line, the same frame written twice, a frozen stream, pose flips, or numbers in the wrong unit. The reader accepts these logs, so `fit_vision_noise` in `sim/vision_noise_fit.py` checks for them itself.

- Two problems are **refused**: positions in the wrong unit stop the whole fit, and a latency over 5 seconds stops the latency part.
- Everything else is **flagged**: the fit adds a line starting with `WARNING:` to `assumptions`.
- A check never changes a fitted number. A log with none of these problems fits exactly as it did before the checks existed.

`tools/intake_vision.py` prints every `assumptions` line under "Fit notes" in `INTAKE.md`. When a fit has any `WARNING` lines, its section also gets a `Warnings: N` line near the top. Read those lines before you use the numbers.

In this section, "detections with truth" means `detected` lines of the fitted subject that have both `position_m` and `ground_truth`. All checks look at the chosen `subject_id` only.

#### Refused: positions in the wrong unit

`_check_units` takes each detection with truth and divides the length of `position_m` by the length of `ground_truth.position_m`. Both lengths are distances from the origin of the position frame. If the median of those ratios is above 10 or below 0.1, the fit stops with an error that starts `position_m is probably not in meters`. The command line tool prints it and exits with code 2.

A factor near 1000 means one side is in millimeters. Pixel coordinates against meter truth also give a big factor.

**What to do:** give the detector the marker size in meters (0.150, not 150), make sure `position_m` and `ground_truth.position_m` are both in meters and in the same frame, and fit again.

#### The 5 cm skip near the origin

Only detections whose true position is at least 5 cm from the frame's origin (`_UNITS_MIN_TRUTH_M = 0.05`) count toward that ratio. A detection whose measured position sits on the origin itself (length 1e-6 m or less) is left out too. The reason: when the target is much closer to the origin than the noise, the measured distance is mostly noise (about 1.7 times the noise spread), so the ratio can pass 10 even with the right units.

In the camera frame this almost never matters, because the target is far more than 5 cm from the lens. It matters in a world frame, when the target sits right on top of the room's origin point.

If every true position is closer than 5 cm, the check is skipped and the fit adds a note that starts `Units check skipped`. This note is not a `WARNING` line, so it is not counted in the `INTAKE.md` warning number. **When you see it, nothing has checked your units.** For example, `position_m` in millimeters against `ground_truth` in meters passes without a refusal if the truth is always within 5 cm of the origin.

**What to do:** check by hand that both positions are in meters. If you can, put the origin point at least 5 cm from the target, so the check can run.

#### Refused: latency over 5 seconds

This check runs only when `clock_domain` is `shared_monotonic`. If any line of the subject (detected, missed or unknown) has `processing_time_s - capture_time_s` above 5 seconds (`MAX_LATENCY_S = 5.0`), the latency fit is refused instead of skewed. `scenario_delay` is `null`, and `scenario_delay_null_reason` says how many lines were too slow and the largest value. A `WARNING: latency not fitted` line repeats the reason. Bias, spread and dropout are still fitted. Latencies of 5 seconds or less are kept.

**What to do:** this is almost never real detector latency. Usually `processing_time_s` came from a different clock than `capture_time_s` (such as wall-clock time), or the program stalled. Fix the time stamps or remove those frames, then fit again.

#### Flagged: warning lines

These come from `_quirk_warnings`, in this order.

| Warning | When it fires | Why it matters | What to do |
| --- | --- | --- | --- |
| Gaps in `capture_time_s` | The subject has at least 11 different capture times, and one step between neighbouring times is longer than 1.9 times the median step. The median step is used, not the header's `frame_rate_hz`, because a webcam often runs slower than it says. The warning estimates the missing frames as each long step divided by the median step, rounded, minus 1. | A frame with no line is not counted as missed, so `dropout_probability` and the burst fit can come out too low. | Make the capture program write a line for every frame, misses included. Normal timing jitter doesn't fire this, and neither does a camera that stops for good (the log is just shorter). |
| Repeated capture times | Some lines share a `capture_time_s` with another line of the same subject. | If they are one camera frame written twice, its error counts twice and the intervals come out too narrow. | Remove the repeats or check the time stamps. |
| Time going backward | In file order, a line has an earlier `capture_time_s` than the line before it, and it is not marked `"out_of_order": true`. | The fit sorts by capture time, so its numbers don't change, but a monotonic camera clock never runs backward. | Check that every `capture_time_s` comes from one monotonic clock. Mark lines that really arrived late as `out_of_order`. |
| Frames only under other ids | Some `frame_id` values have lines for other `subject_id` values but none for the chosen one. | For the chosen subject those frames are neither detected nor missed, so dropout doesn't count them. If the other id is a misread of this marker, those frames are really misses. | Find out whether the other id is a second real target or the detector misreading this marker's id. |
| Behind the camera | `position_frame` is `camera`, and some detections with truth have `position_m` z at 0 or below. In the camera frame (OpenCV convention) z points out of the lens, so that puts the target behind the camera. | These are almost always pose flips. They inflate `noise_std_m` and shift `bias_m`. | Find those frames and explain them before you use the fit. |
| Outliers | At least 20 detections with truth, and on some axis either the standard deviation is more than 2 times the outlier-resistant spread, or at least 3 errors are more than 4 standard deviations from the mean. | `noise_std_m` is a plain standard deviation, so a few wild frames (pose flips, id mix-ups) set its size. | Find and explain those frames before you use `noise_std_m`. |
| Frozen stream | At least 20 detections with truth, and the number that repeat the previous `position_m` exactly (in capture order) is at least 3 and at least 5% of them. | Real noise almost never gives the same value to every digit, so the camera probably sent the same frame again. Repeated frames shrink `noise_std_m`. | Check the video for frozen stretches. |
| Very far target | The median length of `ground_truth.position_m` is above 100 m. | On a bench run this means both positions are probably in pixels or millimeters. They agree with each other, so the units ratio is near 1 and the refusal above doesn't catch it. | Convert both to meters and fit again. |

#### Clearer errors

- A log with more than one `subject_id` and no `--subject` now lists each id with its number of lines. If one id has only a few lines, the detector probably misread the marker id on those frames. Pick the real one with `--subject`.
- A log with a header but no observation lines reads fine, but the fit stops with `trace has no observations to fit`, because the camera program wrote no frames.

### Checking the fit on known settings

```sh
PYTHONPATH=. python tools/make_detection_fixture.py /tmp/fixture.jsonl --frames 3000 --rate 30 \
  --noise-std 0.04 --bias 0.02 -0.01 0.05 --dropout 0.1 \
  --latency-mean 0.03 --latency-std 0.008 --seed 7
PYTHONPATH=. python tools/fit_vision_noise.py /tmp/fixture.jsonl --output /tmp/fixture-fit.json
```

The first command writes a synthetic log and, next to it, `/tmp/fixture.params.json` with the true settings (the fit never reads that file). It makes the measurements with the simulator's own `SensorModel`. Other options: `--trajectory circle|static` (default `circle`), and `--mode burst` with `--burst-start` and `--burst-end` for burst dropout (in burst mode `--dropout` must be 0). Run it with `--help` to see them all.

On the machine this page was checked on, the two commands gave these values, rounded to 4 decimals. The same commands give the same file on the same Python and NumPy versions.

| Setting | True | Fitted | 95% interval |
| --- | --- | --- | --- |
| bias x (m) | 0.02 | 0.0192 | 0.0178 to 0.0208 |
| bias y (m) | -0.01 | -0.0113 | -0.0128 to -0.0096 |
| bias z (m) | 0.05 | 0.0493 | 0.0481 to 0.0508 |
| noise std (m) | 0.04 | 0.0400 | 0.0394 to 0.0406 |
| dropout | 0.1 | 0.1010 | 0.0907 to 0.1123 |
| latency mean (s) | 0.03 | 0.0303 | 0.0299 to 0.0306 |
| latency std (s) | 0.008 | 0.0082 | 0.0079 to 0.0084 |

`tools/tests/test_vision_noise_recovery.py` checks the same idea automatically on bigger logs. It also runs 40 seeds on 1,000-frame logs and requires the 95% intervals for `noise_std_m` and `dropout_probability` to contain the true value in at least 32 of the 40 seeds each, and the three bias intervals together in at least 106 of 120.

---

## Limits (read before using a real log)

- **Fit one log per condition.** Error usually grows with distance, so fit one log per distance and lighting setup. Simple camera geometry also suggests that error along the viewing direction (depth) grows faster than error across the image. This is not measured here. Look at `per_axis` before trusting `noise_std_m`.
- **Positions only.** Bias, spread, dropout and latency all come from the same fit, which needs positions in meters with ground truth. Pixel boxes are not turned into meters, because the format has no camera intrinsics.
- **Ground truth is a full 3D position.** A tape measure gives only a distance, so a bench run has to record the target position in the camera frame some other way, or this needs a distance-only mode (not built).
- **Latency is processing time only.** `processing_time_s - capture_time_s` covers what happens after the capture time stamp. Anything before that stamp, such as exposure and readout, is not seen. How close the stamp is to the real exposure time depends on the camera software and is not checked here.
- **Correlated frames make intervals too narrow.** Short blocks handle short-range correlation. A slow drift, or a static target whose error is the same for the whole run, still gives intervals that are too narrow. `per_axis.lag1_autocorrelation` is a warning sign.
- **Outliers inflate the spread.** `noise_std_m` is a plain standard deviation. If it is much bigger than the outlier-resistant spread in `per_axis`, a bell curve is a poor description of that log. The outlier warning points at this, but the fit never removes outliers for you.
- **Missing lines are hard to see.** The burst fit assumes the log lists every frame, including misses. Frames dropped before they reach the log cannot be counted as misses. The gap warning catches pauses in the middle of a run, but not frames lost at the start or end, or a frame rate that is slower than expected the whole time.
- **Duplicates are not removed.** Lines marked `duplicate` or `out_of_order` are used as they are. Repeated capture times get a warning, but they stay in the fit.
