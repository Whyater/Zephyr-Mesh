#!/usr/bin/env python3
"""Record a camera latency run: show a changing pattern and film it (proposal, EXP-005).

NOT TESTED WITH A REAL CAMERA. Its loop is tested with a fake camera and a fake
screen only. Opening the camera needs the user's own approval and macOS camera
permission for Terminal.app.

Runs in the camera venv (OpenCV), not the repo's .venv:

    uv venv -p 3.13 ~/zephyr-camera-venv
    uv pip install -p ~/zephyr-camera-venv/bin/python "opencv-python==5.0.0.93" "numpy==2.5.4"

Then, from the repo folder, in Terminal.app (not an editor's built-in terminal,
which may never get the camera permission prompt):

    PYTHONPATH=. ~/zephyr-camera-venv/bin/python tools/vision_latency_run.py \\
        --out ~/zephyr-latency/run1

Setup: prop a mirror in front of the laptop so the camera sees the screen.
The script first shows an aiming view (the pattern on top, the camera's view
underneath). Move the mirror until the striped bar is fully inside the camera
view, then press Space. The pattern then fills the screen for --seconds
(default 30). Do not touch anything while it runs. Esc stops early.

What it writes into --out:
  frames.npz        grey frames (downscaled) and t_read_s, the laptop clock
                    right after each cap.read() returned
  display_log.csv   seq, code, t_shown_s (laptop clock just BEFORE the draw
                    call, so the screen can only change after it), t_drawn_s
                    (after cv2.waitKey returned, for checking only)
  run_meta.json     settings, versions, measured frame rate, status "measured"

Then analyse with the repo's .venv:

    PYTHONPATH=. .venv/bin/python tools/vision_latency.py analyze ~/zephyr-latency/run1
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim.vision_latency import render_pattern

WINDOW = "zephyr latency pattern"


@dataclass
class Recording:
    frames: list[np.ndarray] = field(default_factory=list)
    t_read_s: list[float] = field(default_factory=list)
    display: list[tuple[int, int, float, float]] = field(default_factory=list)  # seq, code, t_shown, t_drawn
    camera_error: str = ""


def camera_loop(read: Callable[[], np.ndarray | None], shrink: Callable[[np.ndarray], np.ndarray],
                rec: Recording, stop: threading.Event, clock: Callable[[], float]) -> None:
    """Read frames until told to stop. The clock is read first, before any conversion."""
    misses = 0
    while not stop.is_set():
        frame = read()
        t = clock()
        if frame is None:
            misses += 1
            if misses > 30:
                rec.camera_error = "camera stopped returning frames"
                return
            continue
        misses = 0
        rec.frames.append(shrink(frame))
        rec.t_read_s.append(t)


def display_loop(show: Callable[[np.ndarray], None], key_pressed: Callable[[], bool],
                 patterns: Callable[[int], np.ndarray], rec: Recording, *, n_bits: int, seconds: float,
                 hold_s: tuple[float, float], clock: Callable[[], float], rng: random.Random,
                 sleep: Callable[[float], None] = time.sleep) -> None:
    """Show code 0, 1, 2, ... each for a random hold time, logging when each was drawn.

    t_shown is read BEFORE show(), so the screen cannot change before the
    logged time and the display lag in the analysis is never negative.
    The random hold stops the camera from locking in step with the display.
    """
    start = clock()
    seq = 0
    while clock() - start < seconds:
        code = seq % (1 << n_bits)
        image = patterns(code)
        t_shown = clock()
        show(image)
        if key_pressed():
            return
        rec.display.append((seq, code, t_shown, clock()))
        until = t_shown + rng.uniform(*hold_s)
        while clock() < until:
            if key_pressed():
                return
            sleep(0.001)   # short naps keep the CPU free for the camera thread
        seq += 1


def record(read, shrink, show, key_pressed, *, n_bits: int, seconds: float, hold_s: tuple[float, float],
           screen_size: tuple[int, int], clock: Callable[[], float] = time.monotonic,
           seed: int | None = None) -> Recording:
    """Run the camera thread and the display loop together (the display stays on this thread,
    because macOS only allows windows on the main thread)."""
    rec = Recording()
    width, height = screen_size
    cache: dict[int, np.ndarray] = {}

    def patterns(code: int) -> np.ndarray:
        if code not in cache:
            cache[code] = render_pattern(code, n_bits, width, height)
        return cache[code]

    stop = threading.Event()
    thread = threading.Thread(target=camera_loop, args=(read, shrink, rec, stop, clock), daemon=True)
    thread.start()
    try:
        display_loop(show, key_pressed, patterns, rec, n_bits=n_bits, seconds=seconds, hold_s=hold_s,
                     clock=clock, rng=random.Random(seed))
    finally:
        stop.set()
        thread.join(timeout=5.0)
    return rec


def save(rec: Recording, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frames = np.stack(rec.frames) if rec.frames else np.zeros((0, 1, 1), dtype=np.uint8)
    np.savez_compressed(out / "frames.npz", frames=frames, t_read_s=np.array(rec.t_read_s))
    with (out / "display_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["seq", "code", "t_shown_s", "t_drawn_s"])
        for seq, code, shown, drawn in rec.display:
            writer.writerow([seq, code, f"{shown:.6f}", f"{drawn:.6f}"])
    gaps = np.diff(rec.t_read_s) if len(rec.t_read_s) > 1 else np.array([])
    meta = dict(meta)
    meta.update({"frames": len(rec.frames), "display_events": len(rec.display),
                 "measured_fps": (1.0 / float(np.median(gaps))) if len(gaps) else None,
                 "frame_gap_ms": ({"median": 1000 * float(np.median(gaps)), "max": 1000 * float(np.max(gaps))}
                                  if len(gaps) else None),
                 "camera_error": rec.camera_error})
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="folder for the run (must not exist yet)")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="0 is usually the built-in camera, but an iPhone (Continuity Camera) can take 0")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=30.0, help="requested camera frame rate")
    parser.add_argument("--width", type=int, default=1280, help="requested camera width")
    parser.add_argument("--height", type=int, default=720, help="requested camera height")
    parser.add_argument("--save-width", type=int, default=400, help="frames are shrunk to this width before saving")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--hold-ms", default="40,70", help="random hold per code, min,max in ms")
    parser.add_argument("--screen", default="1440,900", help="pattern image size; the window stretches it to full screen")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} already exists; pick a new folder so no run is overwritten")
    hold = tuple(float(x) / 1000 for x in args.hold_ms.split(","))
    screen = tuple(int(x) for x in args.screen.split(","))
    if len(hold) != 2 or not 0 < hold[0] <= hold[1] or len(screen) != 2:
        parser.error("--hold-ms needs min,max with 0 < min <= max and --screen needs width,height")
    if 2 ** args.bits * hold[0] <= 5.0:
        parser.error("codes would repeat in under 5 s; use more --bits or longer holds")

    try:
        import cv2
    except ImportError:
        print("OpenCV is not installed in this Python. Use the camera venv described in --help.", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Could not open the camera. Check System Settings > Privacy & Security > Camera "
              "for Terminal, and run this from Terminal.app.", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    def read() -> np.ndarray | None:
        ok, frame = cap.read()
        return frame if ok else None

    def shrink(frame: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = grey.shape
        return cv2.resize(grey, (args.save_width, round(h * args.save_width / w)), interpolation=cv2.INTER_AREA)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Aiming view: pattern on top, the live camera view underneath, until Space (or Esc to quit).
    print("Aim the mirror so the whole striped bar shows in the camera view, then press Space. Esc quits.")
    top = render_pattern(5, args.bits, screen[0], int(0.6 * screen[1]))
    while True:
        frame = read()
        if frame is not None:
            view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h = screen[1] - top.shape[0]
            w = round(view.shape[1] * h / view.shape[0])
            view = cv2.resize(view, (w, h))
            canvas = np.full((screen[1], screen[0]), 128, dtype=np.uint8)
            canvas[:top.shape[0]] = top
            x = max(0, (screen[0] - w) // 2)
            canvas[top.shape[0]:, x:x + min(w, screen[0])] = view[:, :min(w, screen[0])]
            cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return 1
        if key == 32:
            break

    pressed = {"esc": False}

    def show(image: np.ndarray) -> None:
        cv2.imshow(WINDOW, image)
        if cv2.waitKey(1) & 0xFF == 27:
            pressed["esc"] = True

    print(f"Recording for {args.seconds:.0f} s. Keep still.")
    rec = record(read, shrink, show, lambda: pressed["esc"], n_bits=args.bits, seconds=args.seconds,
                 hold_s=hold, screen_size=screen)
    cap.release()
    cv2.destroyAllWindows()
    meta = {"schema": "zephyr-vision-latency-run-1", "status": "measured", "n_bits": args.bits,
            "camera_index": args.camera_index, "requested": {"width": args.width, "height": args.height,
                                                             "fps": args.fps},
            "hold_ms": [1000 * hold[0], 1000 * hold[1]], "seconds": args.seconds, "stopped_early": pressed["esc"],
            "opencv_version": cv2.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "clock": "time.monotonic (same clock for frames and display)",
            "notes": ["t_read_s is taken right after cap.read() returns; OpenCV on macOS gives no capture time.",
                      "t_shown_s is taken just before cv2.imshow, so display lag is >= 0."]}
    save(rec, args.out, meta)
    print(f"Saved {len(rec.frames)} frames and {len(rec.display)} pattern changes to {args.out}")
    if rec.camera_error:
        print(f"Camera problem: {rec.camera_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
