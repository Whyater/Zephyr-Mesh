"""Turn a camera latency result into the link JSON that tools/s6_link_rerun.py reads.

Runs in the repo's .venv (no OpenCV, no camera).

    PYTHONPATH=. .venv/bin/python tools/vision_latency_to_link.py \\
        RUN_DIR/latency.json --output RUN_DIR/link.json \\
        [--processing processing_time.json | --vision-fit vision_noise_fit.json]

Inputs
------
latency.json      written by ``tools/vision_latency.py analyze``
                  (schema zephyr-vision-latency-1).
--processing      optional detector cost per frame, schema
                  zephyr-processing-time-1 with median_s, p95_s and n
                  (status is optional; without it the status is unknown).
                  Made by tools/aruco_processing_time.py. A file that says
                  measured must carry provenance (operator, captured_at,
                  code_revision, raw_sha256) or it is refused.
--vision-fit      optional, instead of --processing: a zephyr-vision-noise-fit-1
                  whose scenario_delay holds the detector latency from the
                  detection log (processing_time_s - capture_time_s).

Both optional inputs measure the same thing, detector time per frame, so at
most one may be given (adding both would count it twice). Adding either one to
the camera latency assumes no overlap: the camera latency ends at the frame
read, and the processing time is taken to start there. Nothing checks where a
detection log puts capture_time_s (docs/DETECTION_LOG.md), so this is an
assumption and the output says so.

How the link is built
---------------------
delay_s   camera latency UPPER bound + detector processing time. The camera
          range is lower = apparent - stated error - max display lag, upper =
          apparent + stated error - min display lag. The upper bound is the
          cautious choice: it only assumes display lag is at least its minimum
          (0 ms by default), while the lower bound depends on the assumed
          maximum. Lower, mid and upper are all kept in ``components``.
          The processing term is the median of a processing-time JSON or the
          mean of the fit's scenario_delay. The link wants a mean, and
          processing time is usually right-skewed, so a median can understate
          it; the output warns and keeps p95_s next to it.
jitter_s  standard deviation, the camera jitter (from frames that caught a
          display change) and the fit's detector jitter added in quadrature
          (assumes the two are independent). A processing-time JSON gives no
          standard deviation, so it adds no jitter; a warning says so.
loss_probability  0: the camera path does not drop packets in this model.
status    synthetic if any input is synthetic, measured only if every input
          is measured, otherwise unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA = "zephyr-vision-latency-link-1"
LATENCY_SCHEMA = "zephyr-vision-latency-1"
PROCESSING_SCHEMA = "zephyr-processing-time-1"
FIT_SCHEMA = "zephyr-vision-noise-fit-1"
STATUSES = {"synthetic", "measured", "unknown"}
PROCESSING_PROVENANCE_TEXT = ("operator", "captured_at", "code_revision")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
NO_OVERLAP = ("no overlap assumed: camera latency ends at the frame read and processing time starts there; "
              "if the log stamps capture earlier, the overlap is counted twice")


class LinkConversionError(ValueError):
    pass


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LinkConversionError(f"{name} must be a finite number")
    return float(value)


def _nonnegative(value: Any, name: str) -> float:
    result = _number(value, name)
    if result < 0:
        raise LinkConversionError(f"{name} must be zero or more")
    return result


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LinkConversionError(f"{name} must be a JSON object")
    return value


def _status(value: Any) -> str:
    if value is None:
        return "unknown"
    if not isinstance(value, str):
        raise LinkConversionError("status must be a string")
    return value if value in STATUSES else "unknown"


def combine_status(statuses: list[str]) -> str:
    """Synthetic wins, measured needs every input measured, anything else is unknown."""
    if "synthetic" in statuses:
        return "synthetic"
    if statuses and all(status == "measured" for status in statuses):
        return "measured"
    return "unknown"


def _camera(latency: Any) -> dict[str, Any]:
    latency = _object(latency, "latency input")
    if latency.get("schema") != LATENCY_SCHEMA:
        raise LinkConversionError(f"latency input must have schema {LATENCY_SCHEMA}, got {latency.get('schema')!r}")
    if "error" in latency:
        raise LinkConversionError(f"the latency analysis failed: {latency['error']}")
    camera = _object(latency.get("camera_latency_s"), "camera_latency_s")
    apparent = _object(latency.get("apparent_latency_s"), "apparent_latency_s")
    settings = _object(latency.get("settings"), "settings")
    low = _number(camera.get("low"), "camera_latency_s.low")
    high = _number(camera.get("high"), "camera_latency_s.high")
    if low > high:
        raise LinkConversionError("camera_latency_s.low is above camera_latency_s.high")
    if high <= 0:
        raise LinkConversionError("the camera latency upper bound is not positive; check the recording")
    lag = settings.get("display_lag_s")
    if not isinstance(lag, list) or len(lag) != 2:
        raise LinkConversionError("settings.display_lag_s must be a [min, max] list")
    lag_min, lag_max = (_nonnegative(v, "settings.display_lag_s") for v in lag)
    if lag_min > lag_max:
        raise LinkConversionError("settings.display_lag_s min is above max")
    warnings = latency.get("warnings", [])
    if not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
        raise LinkConversionError("warnings must be a list of strings")
    jitter = apparent.get("jitter_std_from_straddles")
    if jitter is not None:
        jitter = _nonnegative(jitter, "apparent_latency_s.jitter_std_from_straddles")
    return {
        "status": _status(latency.get("status")),
        "latency_s": {"lower": low, "mid": 0.5 * (low + high), "upper": high},
        "used": "upper",
        "apparent_latency_mean_s": _number(apparent.get("mean"), "apparent_latency_s.mean"),
        "stated_error_s": _nonnegative(apparent.get("stated_error"), "apparent_latency_s.stated_error"),
        "display_lag_assumed_s": [lag_min, lag_max],
        "display_lag_note": "an assumption passed through from the analysis, not a measurement",
        "jitter_s": jitter,
        "jitter_note": ("std of frames that caught a display change; reads high by roughly exposure / 7"
                        if jitter is not None else "not measured: fewer than 10 frames caught a display change"),
        "warnings": list(warnings),
    }


def _require_processing_provenance(data: dict[str, Any]) -> None:
    """A processing-time file that says measured must say who, when, which code and which raw timings."""
    prov = data.get("provenance")
    if not isinstance(prov, dict):
        raise LinkConversionError("the processing-time file says measured but has no provenance block (operator, "
                                  "captured_at, code_revision, raw_sha256), so it was not used")
    for key in PROCESSING_PROVENANCE_TEXT:
        value = prov.get(key)
        if not isinstance(value, str) or not value.strip():
            raise LinkConversionError(f"provenance.{key} must be non-empty text in a measured processing-time file")
    digest = prov.get("raw_sha256")
    if not isinstance(digest, str) or not _SHA256_HEX.fullmatch(digest):
        raise LinkConversionError("provenance.raw_sha256 must be 64 lowercase hex characters (the hash of the raw "
                                  "timings) in a measured processing-time file")


def _processing_from_json(data: Any) -> dict[str, Any]:
    data = _object(data, "processing input")
    if data.get("schema") != PROCESSING_SCHEMA:
        raise LinkConversionError(f"processing input must have schema {PROCESSING_SCHEMA}, got {data.get('schema')!r}")
    if _status(data.get("status")) == "measured":
        _require_processing_provenance(data)
    median = _nonnegative(data.get("median_s"), "median_s")
    p95 = _nonnegative(data.get("p95_s"), "p95_s")
    if p95 < median:
        raise LinkConversionError("p95_s is below median_s")
    n = data.get("n")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise LinkConversionError("n must be a positive integer")
    return {"source": PROCESSING_SCHEMA, "status": _status(data.get("status")), "delay_s": median,
            "statistic": "median", "p95_s": p95, "n": n, "jitter_s": None, "assumption": NO_OVERLAP,
            "jitter_note": "not included: a processing-time JSON gives no standard deviation"}


def _processing_from_fit(data: Any) -> dict[str, Any]:
    data = _object(data, "vision fit input")
    if data.get("schema") != FIT_SCHEMA:
        raise LinkConversionError(f"vision fit input must have schema {FIT_SCHEMA}, got {data.get('schema')!r}")
    block = data.get("scenario_delay")
    if block is None:
        reason = data.get("scenario_delay_null_reason", "no reason given")
        raise LinkConversionError(f"the vision fit has no scenario_delay: {reason}")
    block = _object(block, "scenario_delay")
    status = _object(data.get("input", {}), "input").get("status")
    return {"source": FIT_SCHEMA + " scenario_delay", "status": _status(status),
            "delay_s": _nonnegative(block.get("measurement_delay_s"), "scenario_delay.measurement_delay_s"),
            "statistic": "mean", "assumption": NO_OVERLAP,
            "jitter_s": _nonnegative(block.get("measurement_jitter_s"), "scenario_delay.measurement_jitter_s"),
            "jitter_note": "std of processing_time_s - capture_time_s in the detection log"}


def build_link(latency: Any, processing: Any = None, vision_fit: Any = None) -> dict[str, Any]:
    """Build the link JSON from parsed inputs. Pass at most one of processing and vision_fit."""
    if processing is not None and vision_fit is not None:
        raise LinkConversionError("give a processing-time JSON or a vision fit, not both: "
                                  "both measure detector time, so adding them would count it twice")
    camera = _camera(latency)
    warnings = list(camera.pop("warnings"))
    proc = None
    if processing is not None:
        proc = _processing_from_json(processing)
    elif vision_fit is not None:
        proc = _processing_from_fit(vision_fit)
    delay = camera["latency_s"]["upper"] + (proc["delay_s"] if proc else 0.0)
    parts = [j for j in (camera["jitter_s"], proc["jitter_s"] if proc else None) if j is not None]
    jitter = math.sqrt(sum(j * j for j in parts))
    if camera["latency_s"]["lower"] < 0:
        warnings.append("the camera latency lower bound is negative: the assumed display lag maximum is larger "
                        "than the apparent latency, so lower and mid are not physical")
    if proc is not None and proc["statistic"] == "median":
        warnings.append("delay_s adds the processing median, but the link delay is a mean; processing time is "
                        "usually right-skewed, so the median can understate it (p95_s is in components)")
    if camera["jitter_s"] is None:
        warnings.append("camera jitter was not measured, so jitter_s leaves it out")
    if proc is None:
        warnings.append("no processing time given, so delay_s is camera latency only")
    elif proc["jitter_s"] is None:
        warnings.append("the processing-time JSON has no standard deviation, so jitter_s leaves processing spread out")
    statuses = [camera["status"]] + ([proc["status"]] if proc else [])
    source = "camera latency upper bound" + (f" + processing ({proc['source']}, {proc['statistic']})" if proc else "")
    return {"schema": SCHEMA, "delay_s": delay, "jitter_s": jitter, "loss_probability": 0.0,
            "source": source, "status": combine_status(statuses),
            "components": {"camera": camera, "processing": proc}, "warnings": warnings}


def _read(path: Path) -> tuple[Any, dict[str, str]]:
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LinkConversionError(f"cannot read {path.name}: {exc}") from exc
    return data, {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def _check_output(output: Path) -> Path:
    resolved = output.resolve()
    runs = (_REPO_ROOT / "sim" / "runs").resolve()
    if resolved == runs or runs in resolved.parents:
        raise LinkConversionError("output must not be inside sim/runs")
    return resolved


def convert_files(latency_path: Path, output: Path, *, processing_path: Path | None = None,
                  vision_fit_path: Path | None = None) -> dict[str, Any]:
    resolved = _check_output(output)
    if any(resolved == p.resolve() for p in (latency_path, processing_path, vision_fit_path) if p is not None):
        raise LinkConversionError("output would overwrite an input file")
    latency, latency_ref = _read(latency_path)
    inputs = {"latency": latency_ref}
    processing = vision_fit = None
    if processing_path is not None:
        processing, inputs["processing"] = _read(processing_path)
    if vision_fit_path is not None:
        vision_fit, inputs["vision_fit"] = _read(vision_fit_path)
    link = build_link(latency, processing=processing, vision_fit=vision_fit)
    link["inputs"] = inputs
    resolved.write_text(json.dumps(link, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return link


def summary_text(link: dict[str, Any]) -> str:
    cam = link["components"]["camera"]["latency_s"]
    lines = [(f"Link delay {1000 * link['delay_s']:.1f} ms, jitter {1000 * link['jitter_s']:.1f} ms, loss 0 "
              f"(status: {link['status']})"),
             (f"  Camera latency {1000 * cam['lower']:.1f} to {1000 * cam['upper']:.1f} ms "
              f"(mid {1000 * cam['mid']:.1f}); the upper bound is used"),
             f"  Source: {link['source']}"]
    lines += [f"WARNING: {w}" for w in link["warnings"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("latency", type=Path, help="latency.json from tools/vision_latency.py analyze")
    parser.add_argument("--output", type=Path, required=True, help="link JSON to write (not inside sim/runs)")
    extra = parser.add_mutually_exclusive_group()
    extra.add_argument("--processing", type=Path, help="processing-time JSON (zephyr-processing-time-1)")
    extra.add_argument("--vision-fit", type=Path, help="vision noise fit JSON whose scenario_delay is used")
    args = parser.parse_args(argv)
    try:
        link = convert_files(args.latency, args.output, processing_path=args.processing,
                             vision_fit_path=args.vision_fit)
    except (LinkConversionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(summary_text(link))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
