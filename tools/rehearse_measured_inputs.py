"""Dress rehearsal for the measured-input path (simulated inputs only; nothing here is a measurement).

Runs in the repo's .venv (NumPy, SciPy and Matplotlib only; no camera, no radio, no device).

    PYTHONPATH=. .venv/bin/python tools/rehearse_measured_inputs.py [--workdir DIR]

It builds synthetic radio and vision inputs from generators written separately from the code that
reads them, then pushes them through the same commands bench data will use:

1. tools/make_espnow_echo_fixture.py makes a sender and an echoer log with known settings;
2. tools/make_detection_fixture.py makes a detection log with known noise, bias and latency;
3. tools/vision_latency.py synth makes a latency recording with a known injected latency;
4. tools/intake_espnow.py and tools/intake_vision.py take them in (pilot slice);
5. tools/s6_combined_rerun.py reruns the S6 cell with the radio link, the vision link and the noise fit
   together (pilot slice).

Then it checks, and prints one PASS or FAIL line per check:

- every recovered value sits inside the interval the tool states for it, and the interval holds the
  generator's own value (for the latency recording the generator's answer key, not the nominal knobs,
  because its jitter is half-normal and shifts the mean);
- every artifact it wrote says synthetic, and the combined rerun is complete;
- the whole chain took less than 3 minutes;
- the registry file is unchanged (the tests also check that sim/runs gained nothing).

Everything goes into a new folder (a temporary one unless --workdir is given), never under sim/runs.
A pass shows that the chain runs end to end on known inputs. It says nothing about a real radio or a
real camera.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.vision_latency import SynthConfig, synth_recording
from tools import (
    intake_espnow,
    intake_vision,
    make_detection_fixture,
    s6_combined_rerun,
    vision_latency,
)
from tools.make_espnow_echo_fixture import EchoParams, write_fixture

TIME_LIMIT_S = 180.0
REGISTRY = REPO_ROOT / "tools" / "reproduce_registry.json"

RADIO_SEED = 20261020
RADIO_PARAMS = EchoParams()  # the generator's defaults: 2.5 ms delay, 0.8 ms jitter, burst 0.02 / 0.3
N_BOOT_RADIO = 200

DETECTION_SEED = 3
DETECTION_FRAMES = 600
DETECTION_NOISE_M = 0.03
DETECTION_BIAS_M = (0.02, -0.01, 0.05)
DETECTION_DROPOUT = 0.1
DETECTION_LATENCY_S = 0.04
DETECTION_LATENCY_STD_S = 0.01

LATENCY_NOMINAL_MS = 120.0
LATENCY_JITTER_MS = 8.0
LATENCY_SEED = 1  # synth's own default seed; the answer key below is rebuilt from it
PROCESSING = {"schema": "zephyr-processing-time-1", "median_s": 0.015, "p95_s": 0.021, "n": 300,
              "status": "synthetic"}


class RehearsalError(Exception):
    """A step of the chain failed before the checks could run."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quiet(func, *args) -> int:
    """Call a tool's main() and keep its console output out of the rehearsal report."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _inside(name: str, truth: float, low: float, high: float, estimate: float | None = None,
            scale: float = 1.0, unit: str = "") -> dict[str, Any]:
    """The generator's value must sit inside the tool's stated interval."""
    detail = f"truth {truth * scale:.4g}{unit} in [{low * scale:.4g}, {high * scale:.4g}]{unit}"
    if estimate is not None:
        detail += f" (estimate {estimate * scale:.4g}{unit})"
    return _check(name, low <= truth <= high, detail)


def _interval(entry: dict[str, Any]) -> tuple[float, float]:
    low, high = entry["ci"]
    return float(low), float(high)


def _all_synthetic(paths: list[Path]) -> list[str]:
    """Names of artifacts whose status field is missing or not synthetic."""
    bad: list[str] = []
    for path in paths:
        try:
            data = _load(path)
        except (OSError, ValueError):
            bad.append(f"{path.name} (unreadable)")
            continue
        status = _find_status(data)
        if status is None or "synthetic" not in str(status):
            bad.append(f"{path.name} ({status!r})")
    return bad


def _find_status(data: Any) -> str | None:
    """The artifact's overall status: a top-level status, else the source trace or input status."""
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("status"), str):
        return data["status"]
    for key in ("source_trace", "input"):
        inner = data.get(key)
        if isinstance(inner, dict) and isinstance(inner.get("status"), str):
            return inner["status"]
    return None


def build_inputs(work: Path) -> dict[str, Path]:
    """Write the three synthetic inputs and the processing-time file. Returns their paths."""
    inputs = work / "inputs"
    inputs.mkdir()
    write_fixture(inputs / "radio", RADIO_PARAMS, RADIO_SEED)
    log = inputs / "cal-a.jsonl"
    code = _quiet(make_detection_fixture.main, [
        str(log), "--frames", str(DETECTION_FRAMES), "--rate", "30", "--noise-std", str(DETECTION_NOISE_M),
        "--bias", *(str(b) for b in DETECTION_BIAS_M), "--dropout", str(DETECTION_DROPOUT),
        "--latency-mean", str(DETECTION_LATENCY_S), "--latency-std", str(DETECTION_LATENCY_STD_S),
        "--seed", str(DETECTION_SEED)])
    if code != 0:
        raise RehearsalError("the detection fixture generator failed")
    run = inputs / "lat-a"
    code = _quiet(vision_latency.main, ["synth", str(run), "--latency-ms", str(LATENCY_NOMINAL_MS),
                                        "--jitter-ms", str(LATENCY_JITTER_MS), "--seed", str(LATENCY_SEED)])
    if code != 0:
        raise RehearsalError("the latency recording generator failed")
    processing = inputs / "processing.json"
    processing.write_text(json.dumps(PROCESSING), encoding="utf-8")
    return {"sender": inputs / "radio" / "sender.csv", "echoer": inputs / "radio" / "echoer.csv",
            "truth": inputs / "radio" / "truth.json", "log": log, "latency_run": run, "processing": processing}


def run_chain(work: Path, inputs: dict[str, Path]) -> dict[str, Any]:
    """Run both intakes and the combined pilot rerun. Returns the output folders and per-step seconds."""
    out = {"radio": work / "radio-intake", "vision": work / "vision-intake", "combined": work / "combined"}
    seconds: dict[str, float] = {}

    start = time.perf_counter()
    code = _quiet(intake_espnow.main, [
        "--sender", str(inputs["sender"]), "--echoer", str(inputs["echoer"]), "--status", "synthetic",
        "--output", str(out["radio"]), "--n-boot", str(N_BOOT_RADIO), "--seeds", "1", "--pilot"])
    seconds["radio intake"] = time.perf_counter() - start
    if code != 0:
        raise RehearsalError("tools/intake_espnow.py failed")

    start = time.perf_counter()
    code = _quiet(intake_vision.main, [
        "--log", str(inputs["log"]), "--latency-run", str(inputs["latency_run"]),
        "--processing", str(inputs["processing"]), "--output", str(out["vision"]), "--pilot"])
    seconds["vision intake"] = time.perf_counter() - start
    if code != 0:
        raise RehearsalError("tools/intake_vision.py failed")

    start = time.perf_counter()
    code = _quiet(s6_combined_rerun.main, [
        "--radio", str(out["radio"] / "fit.json"),
        "--vision-link", str(out["vision"] / "links" / "lat-a.link.json"),
        "--vision-fit", str(out["vision"] / "fits" / "cal-a.fit.json"),
        "--output", str(out["combined"]), "--pilot"])
    seconds["combined rerun"] = time.perf_counter() - start
    if code != 0:
        raise RehearsalError("tools/s6_combined_rerun.py failed")
    return {"out": out, "seconds": seconds}


def radio_checks(radio_out: Path, truth_path: Path) -> list[dict[str, Any]]:
    """Radio values against the echo generator's truth file and its burst settings."""
    expected = _load(truth_path)["expected"]
    fit = _load(radio_out / "fit.json")
    one_way = fit["delay"]["one_way_estimate"]
    chain = fit["loss"]["burst_chain"]
    rows = [
        ("radio one-way delay", expected["echo_visible_delay_s"], one_way["delay_s"], 1e3, " ms"),
        ("radio delay jitter", expected["echo_visible_jitter_s"], one_way["delay_jitter_s"], 1e3, " ms"),
        ("radio burst start probability", expected["outbound_p_received_to_lost"],
         chain["burst_start_probability"], 1.0, ""),
        ("radio burst end probability", RADIO_PARAMS.burst_end, chain["burst_end_probability"], 1.0, ""),
        ("radio stationary loss", expected["outbound_stationary_loss"], chain["stationary_loss_rate"], 1.0, ""),
        ("radio mean loss run", expected["outbound_mean_loss_run_packets"], chain["mean_loss_run_packets"],
         1.0, " packets"),
    ]
    checks = []
    for name, truth, entry, scale, unit in rows:
        low, high = _interval(entry)
        checks.append(_inside(name, truth, low, high, float(entry["value"]), scale, unit))
    return checks


def vision_checks(vision_out: Path) -> list[dict[str, Any]]:
    """Vision values against the detection generator's settings and the latency answer key."""
    unc = _load(vision_out / "fits" / "cal-a.fit.json")["uncertainty"]
    rows = [("vision noise", DETECTION_NOISE_M, unc["noise_std_m"], 1e3, " mm"),
            ("vision measurement delay", DETECTION_LATENCY_S, unc["measurement_delay_s"], 1e3, " ms")]
    rows += [(f"vision bias {axis}", value, unc[f"bias_m_{axis}"], 1e3, " mm")
             for axis, value in zip("xyz", DETECTION_BIAS_M)]
    checks = []
    for name, truth, entry, scale, unit in rows:
        checks.append(_inside(name, truth, float(entry["ci_low"]), float(entry["ci_high"]),
                              float(entry["estimate"]), scale, unit))

    # The latency synth draws abs(normal(0, jitter)) on top of the injected latency, so the truth is the
    # realized mean and std of its answer key, not the 120 ms and 8 ms that were typed in.
    import numpy as np  # local: only the answer key needs it

    key = synth_recording(SynthConfig(latency_s=LATENCY_NOMINAL_MS / 1e3, latency_jitter_s=LATENCY_JITTER_MS / 1e3,
                                      seed=LATENCY_SEED)).true_latency_s
    true_mean, true_std = float(np.mean(key)), float(np.std(key, ddof=1))
    analysis = _load(vision_out / "latency" / "lat-a.latency.json")
    stated = float(analysis["apparent_latency_s"]["stated_error"])
    link = _load(vision_out / "links" / "lat-a.link.json")
    camera = link["components"]["camera"]
    upper = float(camera["latency_s"]["upper"])
    # The stated range is [lower, upper] (lower assumes the largest display lag, upper none), so the
    # answer key's mean must sit inside it. The link keeps the upper end on purpose, so the key mean
    # must also be at or below it and within twice the tool's own stated error of it.
    lower = float(camera["latency_s"]["lower"])
    checks.append(_inside("latency answer key mean inside the stated range", true_mean, lower, upper,
                          scale=1e3, unit=" ms"))
    checks.append(_check("latency answer key mean vs camera upper bound",
                         true_mean <= upper <= true_mean + 2 * stated,
                         f"key mean {1e3 * true_mean:.2f} ms <= upper {1e3 * upper:.2f} ms <= "
                         f"{1e3 * (true_mean + 2 * stated):.2f} ms (stated error {1e3 * stated:.2f} ms)"))
    checks.append(_check("latency differs from the nominal knob",
                         abs(upper - LATENCY_NOMINAL_MS / 1e3) > 0.004,
                         f"upper {1e3 * upper:.2f} ms is more than 4 ms from the typed {LATENCY_NOMINAL_MS:.0f} ms"))
    jitter = float(camera["jitter_s"])
    # The straddle estimate reads high by roughly exposure / 7 (10 ms / 7 = 1.4 ms); allow 2 ms above the key.
    checks.append(_check("latency jitter vs answer key std",
                         0.9 * true_std <= jitter <= true_std + 0.002,
                         f"key std {1e3 * true_std:.2f} ms, link jitter {1e3 * jitter:.2f} ms "
                         f"(allowed {1e3 * 0.9 * true_std:.2f} to {1e3 * (true_std + 0.002):.2f} ms)"))
    checks.append(_check("link delay is camera upper bound plus processing median (arithmetic only)",
                         abs(float(link["delay_s"]) - (upper + PROCESSING["median_s"])) < 1e-9,
                         f"link delay {1e3 * float(link['delay_s']):.3f} ms = upper + "
                         f"{1e3 * PROCESSING['median_s']:.0f} ms"))
    return checks


def status_checks(chain: dict[str, Any]) -> list[dict[str, Any]]:
    out = chain["out"]
    jsons = [out["radio"] / "fit.json", out["radio"] / "trace.json", out["vision"] / "fits" / "cal-a.fit.json",
             out["vision"] / "latency" / "lat-a.latency.json", out["vision"] / "links" / "lat-a.link.json",
             out["combined"] / "summary.json"]
    missing = [path.name for path in jsons if not path.exists()]
    bad = _all_synthetic([path for path in jsons if path.exists()])
    checks = [_check("every JSON artifact says synthetic", not bad and not missing,
                     f"{len(jsons) - len(missing)} of {len(jsons)} files read"
                     + (f"; missing: {', '.join(missing)}" if missing else "")
                     + (f"; not synthetic: {', '.join(bad)}" if bad else ""))]
    notes = {"radio INTAKE.md": out["radio"] / "INTAKE.md", "vision INTAKE.md": out["vision"] / "INTAKE.md",
             "combined README.txt": out["combined"] / "README.txt"}
    for label, path in notes.items():
        text = path.read_text(encoding="utf-8").lower() if path.exists() else ""
        checks.append(_check(f"{label} says synthetic", "synthetic" in text and "measured fit" not in text,
                             "mentions synthetic" if "synthetic" in text else "no mention of synthetic"))
    summary = _load(out["combined"] / "summary.json")
    checks.append(_check("combined rerun is complete and synthetic",
                         summary.get("complete") is True and summary.get("input_status") == "synthetic",
                         f"complete={summary.get('complete')}, input_status={summary.get('input_status')}"))
    checks.append(_check("combined rerun used all three inputs",
                         sorted(summary.get("included", [])) == ["radio", "vision_fit", "vision_link"],
                         f"included {summary.get('included')}"))
    return checks


def run_rehearsal(work: Path) -> dict[str, Any]:
    """Run the whole chain inside ``work`` (a new or empty folder) and return the checks and timings."""
    work = Path(work)
    registry_before = _sha256(REGISTRY) if REGISTRY.exists() else None
    start = time.perf_counter()
    inputs = build_inputs(work)
    chain = run_chain(work, inputs)
    checks = radio_checks(chain["out"]["radio"], inputs["truth"])
    checks += vision_checks(chain["out"]["vision"])
    checks += status_checks(chain)
    elapsed = time.perf_counter() - start
    checks.append(_check("whole chain under 3 minutes", elapsed < TIME_LIMIT_S,
                         f"{elapsed:.1f} s of {TIME_LIMIT_S:.0f} s"))
    registry_after = _sha256(REGISTRY) if REGISTRY.exists() else None
    checks.append(_check("registry file unchanged", registry_before == registry_after,
                         "tools/reproduce_registry.json has the same sha256"))
    return {"checks": checks, "elapsed_s": elapsed, "seconds": chain["seconds"], "work": work}


def report_lines(result: dict[str, Any]) -> list[str]:
    lines = ["Dress rehearsal (synthetic inputs only; this is not a measurement)"]
    for step, seconds in result["seconds"].items():
        lines.append(f"  step {step}: {seconds:.1f} s")
    for check in result["checks"]:
        lines.append(f"  {'PASS' if check['ok'] else 'FAIL'}  {check['name']}: {check['detail']}")
    failed = [c for c in result["checks"] if not c["ok"]]
    lines.append(f"{len(result['checks']) - len(failed)} of {len(result['checks'])} checks passed in "
                 f"{result['elapsed_s']:.1f} s")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic dress rehearsal of the measured-input path.")
    parser.add_argument("--workdir", type=Path, help="new or empty folder (default: a temporary one)")
    args = parser.parse_args(argv)
    if args.workdir is not None:
        args.workdir.mkdir(parents=True, exist_ok=True)
        if any(args.workdir.iterdir()):
            parser.error("--workdir must be empty")
        result = run_rehearsal(args.workdir)
    else:
        with tempfile.TemporaryDirectory(prefix="zephyr-rehearsal-") as tmp:
            result = run_rehearsal(Path(tmp))
    print("\n".join(report_lines(result)))
    return 0 if all(c["ok"] for c in result["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
