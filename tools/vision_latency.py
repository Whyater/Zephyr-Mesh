#!/usr/bin/env python3
"""Measure camera latency from a filmed on-screen pattern (proposal, EXP-005 method).

Runs in the repo's .venv (NumPy and Matplotlib only, no OpenCV, no camera).

Commands
--------
analyze   Read a recording (frames.npz and display_log.csv written by
          tools/vision_latency_run.py) and write latency samples and a summary.

              PYTHONPATH=. .venv/bin/python tools/vision_latency.py analyze \\
                  RUN_DIR --output RUN_DIR/latency.json

synth     Write a fake recording with a known, injected latency, in the same
          format, for a dry run of the whole chain. Labelled synthetic.

              PYTHONPATH=. .venv/bin/python tools/vision_latency.py synth \\
                  /tmp/vl-synth --latency-ms 120 --fps 30

preview   Save the first frame as a PNG with the strip box drawn on it, to check
          that the strip was found (or to read off a box for --roi).

              PYTHONPATH=. .venv/bin/python tools/vision_latency.py preview \\
                  RUN_DIR --output RUN_DIR/preview.png

What the result means: "apparent latency" is camera latency plus display lag.
It is an upper bound on camera latency (display lag cannot be negative). The
camera latency range uses the --display-lag-ms range, which is an assumption,
not a measurement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from sim.vision_latency import (
    Settings,
    SynthConfig,
    analyze,
    find_strip_roi,
    read_display_log,
    synth_recording,
    write_display_log,
)

FRAMES_FILE = "frames.npz"
LOG_FILE = "display_log.csv"
META_FILE = "run_meta.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair(text: str) -> tuple[float, float]:
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two numbers like 0,60")
    lo, hi = float(parts[0]), float(parts[1])
    if not (0 <= lo <= hi):
        raise argparse.ArgumentTypeError("expected 0 <= min <= max")
    return lo, hi


def _roi(text: str) -> tuple[int, int, int, int]:
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x0,y0,x1,y1")
    return parts[0], parts[1], parts[2], parts[3]


def load_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(run_dir / FRAMES_FILE)
    meta_path = run_dir / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return data["frames"], data["t_read_s"], meta


def cmd_analyze(args: argparse.Namespace) -> int:
    frames, t_read, meta = load_run(args.run_dir)
    n_bits = args.bits or int(meta.get("n_bits", 8))
    log = read_display_log(args.run_dir / LOG_FILE, n_bits)
    settings = Settings(n_bits=n_bits, roi=args.roi, mirror=args.mirror,
                        display_lag_s=(args.display_lag_ms[0] / 1000, args.display_lag_ms[1] / 1000),
                        max_latency_s=args.max_latency_ms / 1000, seed=args.seed)
    result = analyze(frames, t_read, log, settings)
    result["status"] = meta.get("status", "unknown")
    result["inputs"] = {name: _sha256(args.run_dir / name)
                        for name in (FRAMES_FILE, LOG_FILE, META_FILE) if (args.run_dir / name).exists()}
    result["run_meta"] = meta
    samples = result.pop("samples", [])
    output = args.output or (args.run_dir / "latency.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with output.with_suffix(".samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "t_read_s", "seq", "straddle", "lo_s", "hi_s", "mid_s"])
        for s in samples:
            writer.writerow([s["frame"], f"{s['t_read_s']:.6f}", s["seq"], int(s["straddle"]),
                             f"{s['lo_s']:.6f}", f"{s['hi_s']:.6f}", f"{s['mid_s']:.6f}"])
    print(summary_text(result))
    return 0 if "error" not in result else 1


def summary_text(result: dict) -> str:
    lines = [f"Frames used: {result['frames_used']} of {result['frames_total']}"]
    for reason, count in sorted(result["frames_rejected"].items()):
        lines.append(f"  rejected {count}: {reason}")
    if "error" in result:
        lines.append(f"ERROR: {result['error']}")
        return "\n".join(lines)
    a = result["apparent_latency_s"]
    cam = result["camera_latency_s"]
    lag = result["settings"]["display_lag_s"]
    lines += [
        (f"Apparent latency (camera + display lag): mean {1000 * a['mean']:.1f} ms "
         f"+/- {1000 * a['stated_error']:.1f} ms (median {1000 * a['median']:.1f}, "
         f"p05 {1000 * a['p05']:.1f}, p95 {1000 * a['p95']:.1f} ms)"),
        f"  The +/- is a {100 * result['settings']['confidence']:.0f}% interval, but only approximately. Note: {result['settings']['confidence_note']}.",
        f"  This is an upper bound on camera latency: at most {1000 * (a['mean'] + a['stated_error']):.1f} ms.",
        (f"Camera latency, assuming display lag {1000 * lag[0]:.0f} to {1000 * lag[1]:.0f} ms: "
         f"{1000 * cam['low']:.1f} to {1000 * cam['high']:.1f} ms"),
        ("Frame-to-frame jitter (std, from frames that caught a change): "
         + (f"{1000 * a['jitter_std_from_straddles']:.1f} ms from {a['straddle_frames']} frames"
            if a["jitter_std_from_straddles"] is not None else "not enough such frames")),
        f"Status of the input: {result.get('status')}",
    ]
    lines += [f"WARNING: {w}" for w in result.get("warnings", [])]
    return "\n".join(lines)


def cmd_synth(args: argparse.Namespace) -> int:
    cfg = SynthConfig(latency_s=args.latency_ms / 1000, latency_jitter_s=args.jitter_ms / 1000,
                      display_lag_s=args.display_lag_ms / 1000, fps=args.fps, duration_s=args.duration_s,
                      exposure_s=args.exposure_ms / 1000, mirrored=args.mirrored, seed=args.seed)
    rec = synth_recording(cfg)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.run_dir / FRAMES_FILE, frames=rec.frames, t_read_s=rec.t_read_s)
    write_display_log(args.run_dir / LOG_FILE, rec.log)
    meta = {"schema": "zephyr-vision-latency-run-1", "status": "synthetic", "n_bits": cfg.n_bits,
            "injected": {"latency_ms": args.latency_ms, "jitter_ms": args.jitter_ms,
                         "display_lag_ms": args.display_lag_ms, "fps": args.fps,
                         "exposure_ms": args.exposure_ms, "mirrored": args.mirrored, "seed": args.seed,
                         "true_mean_latency_ms": 1000 * float(np.mean(rec.true_latency_s))}}
    (args.run_dir / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote a SYNTHETIC recording ({len(rec.frames)} frames) to {args.run_dir}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames, _, meta = load_run(args.run_dir)
    n_bits = args.bits or int(meta.get("n_bits", 8))
    roi = args.roi or find_strip_roi(frames, n_bits)
    fig, ax = plt.subplots(figsize=(8, 8 * frames.shape[1] / frames.shape[2]))
    ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255)
    x0, y0, x1, y1 = roi
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#4cc9f0", linewidth=2))
    ax.set_title(f"strip box x0,y0,x1,y1 = {x0},{y0},{x1},{y1}")
    output = args.output or (args.run_dir / "preview.png")
    fig.savefig(output, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output} (strip box {x0},{y0},{x1},{y1})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="latency from a recording")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output", type=Path, help="summary JSON (default RUN_DIR/latency.json); samples go next to it")
    p.add_argument("--roi", type=_roi, help="strip box x0,y0,x1,y1 in frame pixels (default: found from the flicker)")
    p.add_argument("--mirror", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--bits", type=int, help="code bits (default from run_meta.json, else 8)")
    p.add_argument("--display-lag-ms", type=_pair, default=(0.0, 60.0),
                   help="assumed display lag range min,max in ms (default 0,60; an assumption, see the docs)")
    p.add_argument("--max-latency-ms", type=float, default=2000.0, help="frames older than this are rejected")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("synth", help="write a synthetic recording with a known latency")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--latency-ms", type=float, default=120.0)
    p.add_argument("--jitter-ms", type=float, default=0.0)
    p.add_argument("--display-lag-ms", type=float, default=0.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--exposure-ms", type=float, default=10.0)
    p.add_argument("--duration-s", type=float, default=20.0)
    p.add_argument("--mirrored", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("preview", help="PNG of the first frame with the strip box")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--roi", type=_roi)
    p.add_argument("--bits", type=int)
    p.set_defaults(func=cmd_preview)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
