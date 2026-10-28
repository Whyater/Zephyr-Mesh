"""Camera latency from a filmed on-screen pattern (proposal, EXP-005 method).

The idea, in plain words
------------------------
macOS OpenCV does not say when a frame was captured. It only hands the frame
over, and the program can read its own clock at that moment (the "read time").
To learn how old a frame is when it arrives, the laptop shows a changing
pattern on its own screen and the camera films that screen (through a mirror).

* The display side writes a log: "code 17 went on screen at t = 3.412 s".
* The camera side writes a log: "frame 40 was read at t = 3.538 s".
* This module reads the code inside frame 40. If it shows code 17, the light
  in that frame left the screen while code 17 was up, so the frame is about
  3.538 - 3.412 s old, give or take how long code 17 stayed up.

Both logs use the same laptop clock, so no capture timestamp is needed.

The code is drawn as a strip of black and white cells using a Gray code: two
neighbouring codes differ in exactly one cell. If the camera's exposure catches
the moment the screen changes, only one cell looks grey, so the frame still
pins the time down to two neighbouring codes instead of becoming garbage.

What the number means
---------------------
The measured "apparent latency" is

    apparent = camera latency + display lag

where camera latency runs from the middle of the exposure to the moment
``read()`` returned, and display lag runs from the moment the display program
logged the code to the moment the pixels actually changed. The display lag is
not measured here. It is passed in as a range (``display_lag_s``) and turns the
apparent latency into a lower and upper bound on the camera latency.

This file needs only NumPy, so it runs in the repo's .venv. The capture script
(``tools/vision_latency_run.py``) needs OpenCV and runs in the camera venv.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "zephyr-vision-latency-1"

# Strip layout, left to right: [white, black, bit cells (most significant first), black, black].
# The white-black pair on one end tells the decoder which way round the strip is
# (a mirror flips it), and gives the white and black reference brightness.
N_REF_LEFT = 2
N_REF_RIGHT = 2

# A bit cell whose brightness sits closer than this fraction of the
# white-to-black contrast to the halfway threshold counts as "grey" (ambiguous).
DEFAULT_AMBIGUOUS_MARGIN = 0.25
# Reject a frame if the white and black reference cells differ by less than this
# (in the frame's own brightness units): the strip is not visible.
DEFAULT_MIN_CONTRAST = 20.0


# --- Gray code ----------------------------------------------------------------

def gray_encode(k: int) -> int:
    """Binary number k to its Gray code (neighbours differ in one bit)."""
    return k ^ (k >> 1)


def gray_decode(g: int) -> int:
    """Gray code back to the plain binary number."""
    k = 0
    while g:
        k ^= g
        g >>= 1
    return k


def code_bits(code: int, n_bits: int) -> np.ndarray:
    """The Gray-coded bits of ``code`` as 0/1, most significant first."""
    g = gray_encode(code % (1 << n_bits))
    return np.array([(g >> (n_bits - 1 - i)) & 1 for i in range(n_bits)], dtype=np.uint8)


# --- drawing the pattern ---------------------------------------------------------

def n_cells(n_bits: int) -> int:
    return N_REF_LEFT + n_bits + N_REF_RIGHT


def cell_values(code: int, n_bits: int) -> np.ndarray:
    """Brightness of every strip cell, 1.0 = white, 0.0 = black."""
    bits = code_bits(code, n_bits).astype(float)
    return np.concatenate(([1.0, 0.0], bits, [0.0, 0.0]))


def render_pattern(code: int, n_bits: int, width: int, height: int) -> np.ndarray:
    """The full-screen image for one code: a grey background and the strip.

    The strip fills the middle 90% of the width and 50% of the height so it is
    still large after the camera sees it small in a mirror. Returns uint8 grey.
    """
    image = np.full((height, width), 128, dtype=np.uint8)
    x0, x1 = round(0.05 * width), round(0.95 * width)
    y0, y1 = round(0.25 * height), round(0.75 * height)
    values = cell_values(code, n_bits)
    edges = np.linspace(x0, x1, len(values) + 1).round().astype(int)
    for value, left, right in zip(values, edges[:-1], edges[1:]):
        image[y0:y1, left:right] = 255 if value > 0.5 else 0
    return image


def pattern_strip_box(width: int, height: int) -> tuple[int, int, int, int]:
    """Where render_pattern puts the strip (x0, y0, x1, y1), for tests and previews."""
    return (round(0.05 * width), round(0.25 * height),
            round(0.95 * width), round(0.75 * height))


# --- reading the code back out of a camera frame -----------------------------------

@dataclass(frozen=True)
class Decoded:
    """What one frame shows.

    ``codes`` holds one code (clean frame) or two neighbouring codes (the
    exposure caught a change, one grey cell). Empty means rejected, and
    ``reason`` says why.
    """
    codes: tuple[int, ...]
    mirrored: bool | None
    contrast: float
    reason: str = ""


def sample_cells(frame: np.ndarray, roi: tuple[int, int, int, int], n_bits: int) -> np.ndarray:
    """Mean brightness of each strip cell inside the box ``roi`` = (x0, y0, x1, y1).

    Each cell is sampled in its middle half (left to right) and the middle 40%
    of the strip height, so blurred cell edges and a slightly tilted strip do
    not leak into the neighbouring cell.
    """
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= frame.shape[1] and 0 <= y0 < y1 <= frame.shape[0]):
        raise ValueError(f"roi {roi} is outside the {frame.shape[1]}x{frame.shape[0]} frame")
    count = n_cells(n_bits)
    cell_w = (x1 - x0) / count
    if cell_w < 2:
        raise ValueError(f"roi is only {x1 - x0} px wide for {count} cells; it needs at least 2 px per cell")
    ya = math.floor(y0 + 0.30 * (y1 - y0))
    yb = max(ya + 1, math.ceil(y0 + 0.70 * (y1 - y0)))
    means = np.empty(count)
    for i in range(count):
        xa = math.floor(x0 + (i + 0.25) * cell_w)
        xb = max(xa + 1, math.ceil(x0 + (i + 0.75) * cell_w))
        means[i] = float(frame[ya:yb, xa:xb].mean())
    return means


def decode_cells(means: np.ndarray, n_bits: int, *, mirror: str = "auto",
                 ambiguous_margin: float = DEFAULT_AMBIGUOUS_MARGIN,
                 min_contrast: float = DEFAULT_MIN_CONTRAST) -> Decoded:
    """Turn cell brightnesses into a code (or two neighbouring codes)."""
    if mirror not in ("auto", "yes", "no"):
        raise ValueError("mirror must be 'auto', 'yes' or 'no'")
    if len(means) != n_cells(n_bits):
        raise ValueError(f"expected {n_cells(n_bits)} cells, got {len(means)}")
    normal = means
    flipped = means[::-1]

    def contrast_of(cells: np.ndarray) -> float:
        # Left reference pair is white then black; the right pair is black, black.
        return float(cells[0] - cells[1])

    if mirror == "no":
        cells = normal
    elif mirror == "yes":
        cells = flipped
    else:
        # Pick the orientation whose left end looks like white-then-black and
        # whose far end looks like black-black.
        score_n = contrast_of(normal) - abs(float(normal[-1] - normal[1]))
        score_f = contrast_of(flipped) - abs(float(flipped[-1] - flipped[1]))
        cells = flipped if score_f > score_n else normal
    mirrored = cells is flipped
    contrast = contrast_of(cells)
    if contrast < min_contrast:
        return Decoded((), mirrored, contrast, "no contrast between the white and black reference cells")
    white, black = float(cells[0]), float(cells[1])
    threshold = 0.5 * (white + black)
    bits_raw = cells[N_REF_LEFT:N_REF_LEFT + n_bits]
    margin = np.abs(bits_raw - threshold) / (white - black)
    bits = (bits_raw > threshold).astype(int)
    unsure = np.flatnonzero(margin < ambiguous_margin)
    if len(unsure) > 1:
        return Decoded((), mirrored, contrast, f"{len(unsure)} grey cells (more than one display change during the exposure, or blur)")

    def to_code(bit_values: np.ndarray) -> int:
        g = 0
        for b in bit_values:
            g = (g << 1) | int(b)
        return gray_decode(g)

    if len(unsure) == 0:
        return Decoded((to_code(bits),), mirrored, contrast)
    # One grey cell: the exposure saw both values of that bit. In a Gray code the
    # two readings are neighbouring codes (k and k+1, possibly across the wrap).
    a = bits.copy()
    b = bits.copy()
    a[unsure[0]] = 0
    b[unsure[0]] = 1
    ca, cb = to_code(a), to_code(b)
    modulus = 1 << n_bits
    if (cb - ca) % modulus == 1:
        pair = (ca, cb)
    elif (ca - cb) % modulus == 1:
        pair = (cb, ca)
    else:  # a grey cell that is not the bit that changed (noise, blur) gives codes that are not neighbours
        return Decoded((), mirrored, contrast, "grey cell does not give neighbouring codes")
    return Decoded(pair, mirrored, contrast)


def find_strip_roi(frames: np.ndarray, n_bits: int, *, change_fraction: float = 0.5) -> tuple[int, int, int, int]:
    """Guess where the strip is from the pixels that flicker.

    Bit cells switch between black and white during the recording; the
    reference cells and the background stay put. So the box around pixels whose
    brightness range (max minus min over all frames) is large covers the bit
    cells. The box is then widened by the reference cells on both ends.
    Needs a recording long enough that every bit changes at least once
    (with codes starting at 0, that is more than 2**(n_bits-1) display changes).
    """
    if frames.ndim != 3 or len(frames) < 2:
        raise ValueError("need at least two frames to find the strip")
    span = frames.max(axis=0).astype(float) - frames.min(axis=0).astype(float)
    peak = float(np.percentile(span, 99.9))
    if peak <= 0:
        raise ValueError("nothing changes between frames, so the strip cannot be found")
    mask = span > change_fraction * peak
    # Drop single noisy pixels: keep rows and columns where a real run of pixels changed.
    cols = np.flatnonzero(mask.sum(axis=0) >= max(2, 0.2 * mask.sum(axis=0).max()))
    rows = np.flatnonzero(mask.sum(axis=1) >= max(2, 0.2 * mask.sum(axis=1).max()))
    if len(cols) == 0 or len(rows) == 0:
        raise ValueError("could not find a flickering region")
    bx0, bx1 = int(cols[0]), int(cols[-1]) + 1
    by0, by1 = int(rows[0]), int(rows[-1]) + 1
    cell_w = (bx1 - bx0) / n_bits
    x0 = max(0, round(bx0 - N_REF_LEFT * cell_w))
    x1 = min(frames.shape[2], round(bx1 + N_REF_RIGHT * cell_w))
    return x0, by0, x1, by1


# --- matching frames to display events -----------------------------------------------

@dataclass(frozen=True)
class DisplayLog:
    """One row per pattern change: sequence number, code shown, laptop time it was logged."""
    seq: np.ndarray
    code: np.ndarray
    t_shown_s: np.ndarray
    n_bits: int
    t_drawn_s: np.ndarray | None = None   # optional: clock after the draw call returned (a check only)

    def __post_init__(self) -> None:
        if not (len(self.seq) == len(self.code) == len(self.t_shown_s)):
            raise ValueError("display log columns differ in length")
        if len(self.seq) < 2:
            raise ValueError("display log needs at least two rows")
        if np.any(np.diff(self.seq) != 1):
            raise ValueError("display log seq must count up by 1 with no gaps")
        if np.any(np.diff(self.t_shown_s) <= 0):
            raise ValueError("display log times must increase")
        if np.any(self.code != self.seq % (1 << self.n_bits)):
            raise ValueError("display log code must equal seq modulo 2**n_bits")


def read_display_log(path: Path, n_bits: int) -> DisplayLog:
    """Read the CSV written by the capture script: columns seq, code, t_shown_s."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = {"seq", "code", "t_shown_s"} - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    drawn = np.array([float(r["t_drawn_s"]) for r in rows]) if "t_drawn_s" in rows[0] else None
    return DisplayLog(np.array([int(r["seq"]) for r in rows]),
                      np.array([int(r["code"]) for r in rows]),
                      np.array([float(r["t_shown_s"]) for r in rows]), n_bits, drawn)


@dataclass(frozen=True)
class Sample:
    """One frame's latency: the apparent latency lies in [lo_s, hi_s]; mid_s is the best single value."""
    frame_index: int
    t_read_s: float
    seq: int
    straddle: bool
    lo_s: float
    hi_s: float
    mid_s: float


def match_frame(decoded: Decoded, t_read_s: float, log: DisplayLog, *,
                max_latency_s: float, widen_s: float = 0.0) -> tuple[Sample | None, str]:
    """Find which display event a decoded frame came from and bound its age.

    A clean frame shows code k. The pattern for event j (code k) was on screen
    from s[j] to s[j+1] (plus display lag), so the apparent latency is between
    t_read - s[j+1] and t_read - s[j]. The midpoint is the best guess.

    A clean frame can still have up to ``ambiguous_margin`` of its exposure in a
    neighbouring event (that cell is not grey enough to count). Its exposure
    middle still lies inside event j: with less than a quarter of the exposure
    outside, the middle sits at least a quarter exposure inside. ``widen_s``
    widens the clean bounds by margin x exposure on each side anyway, as spare
    room; it makes the bounds (and the locking guard built from them) wider
    than they need to be, never narrower. The midpoint is not moved.

    A straddling frame shows k and k+1: the exposure caught the change at
    s[j+1], so the best guess is t_read - s[j+1] and the bounds widen to
    [t_read - s[j+2], t_read - s[j]].

    Codes repeat every 2**n_bits events. The right event is the latest one with
    this code that was shown before the read. That is only true if the real
    latency is shorter than one full cycle of codes; a frame older than a cycle
    would quietly match a later event. ``analyze`` refuses to run unless every
    cycle in the log is longer than ``max_latency_s``, so ``max_latency_s`` is
    the assumption that rules this out.
    """
    if not decoded.codes:
        return None, decoded.reason
    first = decoded.codes[0]
    s = log.t_shown_s
    candidates = np.flatnonzero((log.code == first) & (s <= t_read_s))
    if len(candidates) == 0:
        return None, "code was never shown before this frame was read"
    j = int(candidates[-1])
    straddle = len(decoded.codes) == 2
    end_index = j + (2 if straddle else 1)
    if end_index >= len(s):
        return None, "frame is from the last display event, which has no end time"
    lo = t_read_s - s[end_index] - (0.0 if straddle else widen_s)
    hi = t_read_s - s[j] + (0.0 if straddle else widen_s)
    mid = t_read_s - s[j + 1] if straddle else t_read_s - 0.5 * (s[j] + s[j + 1])
    if hi > max_latency_s:
        return None, f"older than max_latency_s ({max_latency_s} s)"
    return Sample(-1, float(t_read_s), int(log.seq[j]), straddle, float(lo), float(hi), float(mid)), ""


# --- putting it together ------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    n_bits: int = 8
    roi: tuple[int, int, int, int] | None = None   # None = find it from the flicker
    mirror: str = "auto"
    display_lag_s: tuple[float, float] = (0.0, 0.060)
    max_latency_s: float = 2.0
    exposure_max_s: float = 0.034   # longest exposure assumed, for widening per-frame bounds (one 30 fps frame)
    bootstrap: int = 2000
    confidence: float = 0.99
    seed: int = 0
    ambiguous_margin: float = DEFAULT_AMBIGUOUS_MARGIN
    min_contrast: float = DEFAULT_MIN_CONTRAST
    display_refresh_hz: float | None = None   # screen refresh rate; None = no refresh-phase term


CONFIDENCE_NOTE = ("approximate; on synthetic runs the stated error covered the truth in 97% of runs "
                   "(96% without the locking guard), not 99%")


def cluster_bootstrap_mean_ci(values: np.ndarray, groups: np.ndarray, *, n: int, confidence: float,
                              rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean, resampling whole groups.

    Frames that saw the same display event share that event's timing error, so
    they are not independent. Resampling events (with all their frames)
    instead of single frames keeps that sharing in the interval.
    """
    _, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(float)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = 0.5 * (1.0 - confidence)
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def analyze(frames: np.ndarray, t_read_s: np.ndarray, log: DisplayLog, settings: Settings | None = None) -> dict:
    """Decode every frame, match it to the display log and summarise the latency.

    Returns a JSON-ready dict. All times are in seconds.

    Stated error. The headline is the MEAN of the per-frame midpoints. Each
    midpoint is off by up to half a hold time, but because the display holds
    each code for a random time the camera does not lock to the display, so
    those errors scatter evenly around zero and cancel in the mean. (The median
    does not have that property when the latency itself is lopsided, so it is
    reported only as a description.) The error has two parts, reported
    separately and added:
    * statistical: half the width of a 99% bootstrap interval for the mean that
      resamples display events, not frames, because frames that saw the same
      event share its timing error. Treat the 99% as approximate: in a check
      over 300 random synthetic recordings (tools/vision_latency_calibration.py)
      it covered the injected latency in 292 (97%), or 178 of the 186 runs
      without the locking guard (96%), and the worst miss was 1.6 times the
      stated error (2.8 ms). No bias was found (mean signed error
      -0.02 +/- 0.04 ms over 80 runs). All of those recordings used a screen
      that changes instantly, with no refresh grid, so the honest figure for
      a real screen is "about 96% in synthetic checks without vsync" plus the
      refresh-phase term below;
    * locking guard: if the hold times barely vary compared with the camera
      frame period, camera and display can lock in step and the midpoint errors
      stop cancelling. Then the mean half-width of the per-frame bounds is
      added as a worst case;
    * refresh phase (only when ``display_refresh_hz`` is given): a real screen
      only changes on its refresh grid, one step g = 1 / refresh apart. A
      camera whose frame period is a whole number of steps (30 fps on a 60 Hz
      screen) takes every exposure at the same offset after a refresh, so the
      midpoint errors no longer cancel even when the holds are random. The
      expected leftover offset is the camera's position inside a refresh step
      minus half a step, so it lies between -g/2 and +g/2, and recording
      longer does not shrink it. g/2 is added as a worst case. A camera that
      drifts against the grid averages it out, so this is conservative for
      it. Limits: the scatter around that expected offset (which frames of an
      event survive as clean) is only covered by the statistical part, so
      very short recordings can exceed it; and the synthetic checks
      (tools/vision_latency_calibration.py vsync, and a phase sweep in the
      tests) use exposures up to 33 ms and holds of 40 to 70 ms, so exposures
      longer than two refresh steps or holds shorter than one are untested.
    The display lag range is a separate one-sided correction: camera latency =
    apparent latency minus display lag.

    Not modelled (real effects the synthetic tests do not contain): rolling
    shutter (rows exposed at slightly different times; the strip is
    horizontal, so all its cells sit on the same rows, but the row offset
    still adds a fixed amount to the latency), a tilted strip seen in a
    mirror, brightness that varies along the strip, display lag that changes
    from event to event, and auto exposure changing during the run.
    """
    settings = settings or Settings()
    if frames.ndim != 3:
        raise ValueError("frames must be a 3D array (frame, row, column) of grey values")
    if len(frames) != len(t_read_s):
        raise ValueError("frames and t_read_s differ in length")
    if np.any(np.diff(t_read_s) <= 0):
        raise ValueError("t_read_s must increase")
    lag_min, lag_max = settings.display_lag_s
    if not (0 <= lag_min <= lag_max):
        raise ValueError("display_lag_s must be (min, max) with 0 <= min <= max")
    if settings.display_refresh_hz is not None and not settings.display_refresh_hz > 0:
        raise ValueError("display_refresh_hz must be positive")
    if log.n_bits != settings.n_bits:
        raise ValueError(f"display log uses {log.n_bits} bits but settings say {settings.n_bits}")
    period = 1 << log.n_bits
    if len(log.t_shown_s) > period:
        shortest_cycle = float(np.min(log.t_shown_s[period:] - log.t_shown_s[:-period]))
        if shortest_cycle <= settings.max_latency_s:
            raise ValueError(f"codes repeat after {shortest_cycle:.3f} s, which is not longer than "
                             f"max_latency_s = {settings.max_latency_s} s, so old frames could be misread. "
                             "Use more bits, longer holds, or a smaller max_latency_s.")
    roi = settings.roi or find_strip_roi(frames, settings.n_bits)

    samples: list[Sample] = []
    rejected: dict[str, int] = {}
    mirrored_votes = 0
    for i, (frame, t) in enumerate(zip(frames, t_read_s)):
        decoded = decode_cells(sample_cells(frame, roi, settings.n_bits), settings.n_bits,
                               mirror=settings.mirror, ambiguous_margin=settings.ambiguous_margin,
                               min_contrast=settings.min_contrast)
        sample, reason = match_frame(decoded, float(t), log, max_latency_s=settings.max_latency_s,
                                     widen_s=settings.ambiguous_margin * settings.exposure_max_s)
        if sample is None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        mirrored_votes += 1 if decoded.mirrored else 0
        samples.append(Sample(i, sample.t_read_s, sample.seq, sample.straddle, sample.lo_s, sample.hi_s, sample.mid_s))

    result: dict = {
        "schema": SCHEMA,
        "status": "analysis",
        "settings": {"n_bits": settings.n_bits, "roi": list(roi), "mirror": settings.mirror,
                     "display_lag_s": [lag_min, lag_max], "max_latency_s": settings.max_latency_s,
                     "bootstrap": settings.bootstrap, "confidence": settings.confidence,
                     "confidence_note": CONFIDENCE_NOTE, "seed": settings.seed,
                     "exposure_max_s": settings.exposure_max_s},
        "frames_total": len(frames),
        "frames_used": len(samples),
        "frames_rejected": rejected,
        "mirrored_fraction": (mirrored_votes / len(samples)) if samples else None,
        "warnings": [],
    }
    if settings.display_refresh_hz is not None:   # only when set, so older outputs stay byte-identical
        result["settings"]["display_refresh_hz"] = settings.display_refresh_hz
    if settings.roi is None and len(log.seq) <= period // 2:
        result["warnings"].append(f"only {len(log.seq)} display changes, so the top bit never changed and the "
                                  "automatic strip box may miss it. Record longer or pass --roi.")
    if len(samples) < 10:
        result["error"] = f"only {len(samples)} usable frames; need at least 10"
        return result

    mid = np.array([s.mid_s for s in samples])
    half = 0.5 * np.array([s.hi_s - s.lo_s for s in samples])
    seqs = np.array([s.seq for s in samples])
    straddle = np.array([s.straddle for s in samples])
    rng = np.random.default_rng(settings.seed)
    mean = float(np.mean(mid))
    mean_lo, mean_hi = cluster_bootstrap_mean_ci(mid, seqs, n=settings.bootstrap,
                                                 confidence=settings.confidence, rng=rng)
    holds = np.diff(log.t_shown_s)
    gaps = np.diff(t_read_s)
    frame_period = float(np.median(gaps))
    # Random holds make the display's timing drift against the camera's like a
    # random walk: after N changes the spread is about sqrt(N) x std(hold). If that
    # spread covers at least two camera frames, every phase between the two gets
    # visited and the midpoint errors cancel. Otherwise assume they may be locked.
    random_holds = bool(math.sqrt(len(holds)) * np.std(holds) >= 2.0 * frame_period)
    statistical = 0.5 * (mean_hi - mean_lo)
    locking_guard = 0.0 if random_holds else float(np.mean(half))
    stated = statistical + locking_guard
    stated_parts = {"statistical": statistical, "locking_guard": locking_guard}
    if settings.display_refresh_hz is not None:
        stated_parts["refresh_phase"] = 0.5 / settings.display_refresh_hz
        stated += stated_parts["refresh_phase"]
    # Straddle frames caught a display change mid-exposure, so their midpoint is
    # off by at most a quarter of the exposure. Their spread is therefore close to
    # the real frame-to-frame jitter, unlike clean frames whose midpoints scatter
    # by up to half a hold. They still read high: the estimate has a floor of
    # roughly exposure / 7 (about 5 ms at a 33 ms exposure, even with zero true
    # jitter). It is not subtracted.
    jitter = float(np.std(mid[straddle], ddof=1)) if int(straddle.sum()) >= 10 else None

    warnings = result["warnings"]
    rejected_total = sum(rejected.values())
    if rejected_total > 0.3 * len(frames):
        warnings.append(f"{rejected_total} of {len(frames)} frames were rejected; check the aim and preview.png.")
    if float(np.max(gaps)) > 1.5 * frame_period:
        warnings.append(f"frame gaps up to {1000 * float(np.max(gaps)):.0f} ms against a median of "
                        f"{1000 * frame_period:.0f} ms: the camera dropped frames or changed rate.")
    if float(np.min(holds)) <= settings.exposure_max_s:
        warnings.append("some holds are not longer than exposure_max_s, so one exposure can see two changes.")
    draw_ms = None
    if log.t_drawn_s is not None:
        draw = log.t_drawn_s - log.t_shown_s
        draw_ms = {"median": 1000 * float(np.median(draw)), "max": 1000 * float(np.max(draw))}
        if float(np.median(draw)) > lag_max:
            warnings.append(f"the draw call alone took a median {draw_ms['median']:.0f} ms, more than the assumed "
                            f"display lag maximum; raise --display-lag-ms.")

    result.update({
        "apparent_latency_s": {
            "mean": mean,
            "mean_ci": [mean_lo, mean_hi],
            "stated_error": stated,
            "stated_error_parts": stated_parts,
            "median": float(np.median(mid)),
            "p05": float(np.quantile(mid, 0.05)),
            "p95": float(np.quantile(mid, 0.95)),
            "per_frame_half_width_mean": float(np.mean(half)),
            "jitter_std_from_straddles": jitter,
            "straddle_frames": int(straddle.sum()),
        },
        "camera_latency_s": {
            "low": mean - stated - lag_max,
            "high": mean + stated - lag_min,
            "note": "camera latency = apparent latency minus display lag; the display lag range is an input, not a measurement",
        },
        "display_holds_s": {"mean": float(np.mean(holds)), "min": float(np.min(holds)),
                            "max": float(np.max(holds)), "random": random_holds},
        "draw_call_ms": draw_ms,
        "camera_frame_period_s": frame_period,
        "straddle_fraction": float(np.mean(straddle)),
        "samples": [{"frame": s.frame_index, "t_read_s": s.t_read_s, "seq": s.seq, "straddle": s.straddle,
                     "lo_s": s.lo_s, "hi_s": s.hi_s, "mid_s": s.mid_s} for s in samples],
    })
    return result


# --- synthetic recordings with a known delay (for tests and dry runs) ------------------------

@dataclass(frozen=True)
class SynthConfig:
    """A fake recording. Every time comes from the same pretend laptop clock.

    latency_s is the true camera latency (middle of exposure to read()).
    display_lag_s is the true delay between logging a code and the pixels changing.
    vsync_hz, if set, makes the screen refresh on a fixed grid like a real
    display: a change only shows at the next refresh after logged time plus
    display_lag_s, so each event gets an extra lag between 0 and one refresh
    step. None (the default) is a screen that changes the instant it is told to.
    """
    latency_s: float = 0.120
    latency_jitter_s: float = 0.0
    display_lag_s: float = 0.0
    fps: float = 30.0
    exposure_s: float = 0.010
    duration_s: float = 20.0
    hold_s: tuple[float, float] = (0.040, 0.070)   # each code stays up a random time in this range
    n_bits: int = 8
    frame_size: tuple[int, int] = (240, 320)        # rows, columns of the camera frame
    strip_box: tuple[int, int, int, int] = (30, 90, 290, 150)  # where the strip lands in the frame
    white: float = 210.0
    black: float = 35.0
    background: float = 110.0
    noise_std: float = 4.0
    mirrored: bool = False
    camera_phase_s: float = 0.0
    vsync_hz: float | None = None
    seed: int = 1


def synth_display_log(cfg: SynthConfig, rng: np.random.Generator) -> DisplayLog:
    lo, hi = cfg.hold_s
    times = [0.0]
    while times[-1] < cfg.duration_s + cfg.latency_s + 1.0:
        times.append(times[-1] + (rng.uniform(lo, hi) if hi > lo else lo))
    seq = np.arange(len(times))
    return DisplayLog(seq, seq % (1 << cfg.n_bits), np.array(times), cfg.n_bits)


@dataclass(frozen=True)
class SynthRecording:
    frames: np.ndarray
    t_read_s: np.ndarray
    log: DisplayLog
    true_latency_s: np.ndarray   # per frame, read time minus exposure middle (the answer key)


def synth_recording(cfg: SynthConfig) -> SynthRecording:
    """Make frames, read times and a display log with a known, injected latency.

    The frame image is what an ideal camera would see: each strip cell's
    brightness is the time-average of what the screen showed during the
    exposure window, so a frame that catches a change gets a grey cell, just
    like a real one. The read time is exposure middle + latency (+ jitter).
    Nothing in here uses the analysis code, apart from drawing cell values.
    """
    rng = np.random.default_rng(cfg.seed)
    log = synth_display_log(cfg, rng)
    on_screen = log.t_shown_s + cfg.display_lag_s       # when the pixels really change
    if cfg.vsync_hz is not None:
        on_screen = np.ceil(on_screen * cfg.vsync_hz) / cfg.vsync_hz   # next refresh
    period = 1.0 / cfg.fps
    first = 0.5 + cfg.camera_phase_s                    # skip the first half second
    centres = np.arange(first, cfg.duration_s, period)
    rows, cols = cfg.frame_size
    x0, y0, x1, y1 = cfg.strip_box
    count = n_cells(cfg.n_bits)
    edges = np.linspace(x0, x1, count + 1).round().astype(int)
    frames = np.empty((len(centres), rows, cols), dtype=np.uint8)
    t_read = np.empty(len(centres))
    for f, centre in enumerate(centres):
        a, b = centre - 0.5 * cfg.exposure_s, centre + 0.5 * cfg.exposure_s
        if cfg.exposure_s <= 0:
            j = int(np.searchsorted(on_screen, centre, side="right") - 1)
            weights = {j: 1.0}
        else:
            weights = {}
            j0 = int(np.searchsorted(on_screen, a, side="right") - 1)
            j = j0
            while j < len(on_screen) and (j < 0 or on_screen[j] < b):
                start = max(a, on_screen[j]) if j >= 0 else a
                end = min(b, on_screen[j + 1]) if j + 1 < len(on_screen) else b
                if end > start and j >= 0:
                    weights[j] = (end - start) / cfg.exposure_s
                j += 1
        cells = sum(w * cell_values(int(log.code[j]), cfg.n_bits) for j, w in weights.items())
        image = np.full((rows, cols), cfg.background)
        for value, left, right in zip(cells, edges[:-1], edges[1:]):
            image[y0:y1, left:right] = cfg.black + value * (cfg.white - cfg.black)
        if cfg.mirrored:
            image = image[:, ::-1]
        image += rng.normal(0.0, cfg.noise_std, image.shape)
        frames[f] = np.clip(np.round(image), 0, 255).astype(np.uint8)
        jitter = abs(rng.normal(0.0, cfg.latency_jitter_s)) if cfg.latency_jitter_s > 0 else 0.0
        t_read[f] = centre + cfg.latency_s + jitter
    # A big latency jump could put two reads out of order. A real read loop
    # returns frames in order, so the later frame waits for the earlier one.
    for f in range(1, len(t_read)):
        t_read[f] = max(t_read[f], t_read[f - 1] + 1e-6)
    return SynthRecording(frames, t_read, log, t_read - centres)


def write_display_log(path: Path, log: DisplayLog) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["seq", "code", "t_shown_s"])
        for seq, code, t in zip(log.seq, log.code, log.t_shown_s):
            writer.writerow([int(seq), int(code), f"{float(t):.6f}"])
