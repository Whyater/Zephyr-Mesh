"""One command from bench vision files to a pilot rerun and a summary (a proposal, not a result).

Runs in the repo's .venv (NumPy and Matplotlib only, no OpenCV, no camera).

    PYTHONPATH=. .venv/bin/python tools/intake_vision.py \\
        --log cal-1.jsonl [--log cal-2.jsonl ...] \\
        --latency-run lat-1 [--latency-run lat-2 ...] \\
        [--processing processing_time.json] --output DIR [--pilot]

For each detection log it fits the vision noise (tools/fit_vision_noise.py).
For each latency run folder it analyzes the recording (tools/vision_latency.py
analyze) and converts the result to a link JSON
(tools/vision_latency_to_link.py), adding the processing-time JSON when one is
given. For each fit it then reruns the S6 main slice with the fitted noise
(tools/s6_measured_rerun.py). Everything goes into a new folder, never under
sim/runs, and INTAKE.md lists every input and output with its sha256, each
status, and the fitted values.

Status: the overall status is synthetic if any input is synthetic (or a test
fixture), measured only if every input says measured, otherwise unknown. A log
or latency run that says "measured" without its provenance is refused before
any stage runs. A log whose provenance is present but malformed is refused by
the fit stage, and the folder this command made is removed.

What the rerun applies: only the fitted noise (the registered S6 rerun applies
nothing else). The link JSON is written for tools/s6_link_rerun.py and is not
applied by this command. Display lag in the latency result is an assumption.
One processing-time JSON is added to the link of every latency run, and a
processing-time file has no provenance fields, so its status is taken as
written. Inputs are never modified.

--pilot runs one seed on 8 of the 56 main-slice cells (both motions, delay 0
and 0.10 s, loss 0 and 0.30). Without it the default is 5 seeds on every cell.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import fit_vision_noise, vision_latency
from tools.s6_measured_rerun import (
    BRACKET_LABELS,
    RerunError,
    protected_runs_dirs,
    run_rerun,
)
from tools.vision_latency_to_link import (
    LinkConversionError,
    _processing_from_json,
    combine_status,
    convert_files,
)

PILOT_DELAYS = (0.0, 0.10)
PILOT_LOSSES = (0.0, 0.30)
PILOT_SEEDS = 1
DEFAULT_SEEDS = 5
LATENCY_FILES = ("frames.npz", "display_log.csv", "run_meta.json")
RECORDER_KEYS = ("schema", "opencv_version", "platform", "clock")
VOLATILE = ("INTAKE.md", "manifest.json", "README.txt")
STATUSES = ("synthetic", "measured", "fixture", "unknown")
STATUS_RULE = ("The overall status is synthetic if any input is synthetic or a test fixture, measured only if "
               "every input says measured, otherwise unknown.")
SCOPE_NOTE = ("Scope: simulation inputs only. This does not claim flight performance, and a synthetic or fixture "
              "input says nothing about a real camera.")
LINK_NOTE = ("The link JSON files (links/) are for tools/s6_link_rerun.py. The S6 rerun below does not use "
             "them.")
RERUN_NOTE = ("Only the fitted noise is applied. Fitted bias, dropout and latency are recorded in the fit but "
              "not applied, and the tracker measurement std stays at 0.05 m as registered. The link and "
              "tracker are simulated.")
VOLATILE_NOTE = ("manifest.json and README.txt inside each S6 rerun folder are volatile (they hold paths and "
                 "runtime), so they are not hashed here. INTAKE.md does not hash itself. comparison.csv copies "
                 "registered columns from sim/runs when that file exists, so its hash depends on the checkout.")
PILOT_NOTE = ("Pilot: the range below spans one-seed medians on a subset of cells. It shows that the chain "
              "runs. It is not a result.")
INTERVAL_NOTE = "(the +/- is an approximate interval)"


class IntakeError(ValueError):
    """An input, an option or the output folder cannot be used."""


def pilot_filter(cell: dict[str, Any]) -> bool:
    """Keep 8 of the 56 main-slice cells: both motions, two delays, two loss levels."""
    return cell["delay_s"] in PILOT_DELAYS and cell["loss_probability"] in PILOT_LOSSES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status(value: Any) -> str:
    return value if isinstance(value, str) and value in STATUSES else "unknown"


def _ms(seconds: float | None) -> str:
    return "n/a" if seconds is None else f"{1000 * seconds:.1f} ms"


def _num(value: float | None, digits: int = 4) -> str:
    return "not fitted" if value is None else f"{value:.{digits}f}"


# ---------------------------------------------------------------- preflight

def _check_output(output: Path, inputs: list[Path]) -> tuple[Path, bool]:
    """Return the resolved output folder and whether it already existed (empty)."""
    resolved = output.resolve()
    for runs_dir in protected_runs_dirs():
        if resolved == runs_dir or resolved.is_relative_to(runs_dir):
            raise IntakeError(f"output must not be inside sim/runs ({runs_dir}); registered outputs live there")
    if resolved.exists():
        if not resolved.is_dir():
            raise IntakeError(f"output path is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise IntakeError(f"output folder is not empty: {resolved}; pick a new folder")
    for path in inputs:
        if path.resolve().is_relative_to(resolved):
            raise IntakeError(f"input {path.name} must not be inside the output folder")
    return resolved, resolved.exists()


def _unique_names(paths: list[Path], kind: str, name_of) -> dict[str, Path]:
    named: dict[str, Path] = {}
    for path in paths:
        name = name_of(path)
        if name in named:
            raise IntakeError(f"two {kind} inputs share the name {name!r} ({named[name].name} and {path.name}); "
                              "rename one so their outputs do not collide")
        named[name] = path
    return named


def _check_log(name: str, path: Path) -> None:
    """Stop early, with a plain message, when a log says measured but gives no provenance."""
    if not path.is_file():
        raise IntakeError(f"detection log {path.name}: file not found")
    with path.open("rb") as handle:
        first = handle.readline()
    try:
        header = json.loads(first.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return  # the fit step reports a malformed log with its own message
    if isinstance(header, dict) and header.get("status") == "measured":
        missing = [key for key in ("provenance", "source") if not header.get(key)]
        if missing:
            raise IntakeError(f"detection log {path.name} ({name}): status is measured but {' and '.join(missing)} "
                              "is missing; a measured log needs its provenance (operator, capture time, code "
                              "revision, instrument), so this input was not used")


def _read_processing(path: Path) -> dict[str, Any]:
    """Check a processing-time file before any stage runs; return its parsed fields."""
    if not path.is_file():
        raise IntakeError(f"processing-time file {path.name}: file not found")
    try:
        return _processing_from_json(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, LinkConversionError) as exc:
        raise IntakeError(f"processing-time file {path.name}: {exc}") from exc


def _check_latency_run(name: str, run: Path) -> None:
    if not run.is_dir():
        raise IntakeError(f"latency run {name}: not a folder")
    for required in ("frames.npz", "display_log.csv"):
        if not (run / required).is_file():
            raise IntakeError(f"latency run {name}: {required} is missing")
    meta_path = run / "run_meta.json"
    if not meta_path.is_file():
        return  # status will read as unknown
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntakeError(f"latency run {name}: cannot read run_meta.json: {exc}") from exc
    if isinstance(meta, dict) and meta.get("status") == "measured":
        missing = [key for key in RECORDER_KEYS if not meta.get(key)]
        if missing:
            raise IntakeError(f"latency run {name}: status is measured but run_meta.json lacks the recorder "
                              f"provenance ({', '.join(missing)}); a measured run comes from "
                              "tools/vision_latency_run.py, so this input was not used")


# ---------------------------------------------------------------- stages

def _quiet(func, *args) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = func(*args)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
        except (OSError, ValueError, KeyError, ArithmeticError, zipfile.BadZipFile) as exc:
            err.write(f"{type(exc).__name__}: {exc}\n")
            code = 2
    return int(code or 0), out.getvalue(), err.getvalue()


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "no message"


def _fit_log(name: str, log: Path, output: Path, *, subject: str | None, dropout_mode: str, bootstrap: int,
             seed: int) -> dict[str, Any]:
    argv = [str(log), "--output", str(output), "--dropout-mode", dropout_mode, "--bootstrap", str(bootstrap),
            "--seed", str(seed)]
    if subject:
        argv += ["--subject", subject]
    code, _, err = _quiet(fit_vision_noise.main, argv)
    if code != 0:
        raise IntakeError(f"fit failed for {log.name}: {_last_line(err)}")
    return json.loads(output.read_text(encoding="utf-8"))


def _analyze_run(name: str, run: Path, output: Path) -> dict[str, Any]:
    code, _, err = _quiet(vision_latency.main, ["analyze", str(run), "--output", str(output)])
    if code == 2 or not output.exists():
        raise IntakeError(f"latency analysis failed for {name}: {_last_line(err)}")
    result = json.loads(output.read_text(encoding="utf-8"))
    if "error" in result:
        raise IntakeError(f"latency analysis failed for {name}: {result['error']}")
    return result


def _run_s6(fit_path: Path, output: Path, *, bracket: tuple[str, ...], depth_axis: str | None, seeds: int,
            pilot: bool) -> dict[str, Any]:
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            return run_rerun(fit_path, output, bracket=bracket, depth_axis=depth_axis, seeds_count=seeds,
                             processes=1, cell_filter=pilot_filter if pilot else None)
    except (RerunError, OSError) as exc:
        raise IntakeError(f"S6 rerun failed for {fit_path.name}: {exc}") from exc


# ---------------------------------------------------------------- INTAKE.md

def _interval_rows(fit: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    unc = fit["uncertainty"]
    rows = [("noise_std_m (pooled)", unc["noise_std_m"])]
    axes = fit["per_axis"]["axes"]
    for index, axis in enumerate(axes):
        rows.append((f"noise std, {axis} axis (m)", {"estimate": fit["per_axis"]["std_m"][index],
                                                      "ci_low": fit["per_axis"]["std_ci_low_m"][index],
                                                      "ci_high": fit["per_axis"]["std_ci_high_m"][index]}))
    for axis in ("x", "y", "z"):
        key = f"bias_m_{axis}"
        if key in unc:
            rows.append((f"bias, {axis} (m)", unc[key]))
    for label, key in (("dropout probability", "dropout_probability"),
                       ("measurement delay (s)", "measurement_delay_s"),
                       ("measurement jitter (s)", "measurement_jitter_s")):
        rows.append((label, unc[key]))
    return rows


def _fit_section(name: str, log: Path, fit: dict[str, Any]) -> list[str]:
    block = fit["input"]
    level = fit.get("bootstrap", {}).get("confidence_level")
    level_text = f"{100 * level:.0f}%" if isinstance(level, (int, float)) else "stated"
    counts = block["counts"]
    frames = f"{counts['observed']} observed, {counts['detected']} detected, {counts['missed']} missed"
    lines = [f"### {name}", "",
             f"- Log: {log.name}",
             f"- Status: {_status(block.get('status'))}",
             f"- Trace id: {block.get('trace_id')}, position frame: {fit['per_axis'].get('position_frame')}",
             f"- Frames: {frames}",
             "", f"| Parameter | Estimate | {level_text} interval |", "|---|---|---|"]
    for label, record in _interval_rows(fit):
        est, low, high = record.get("estimate"), record.get("ci_low"), record.get("ci_high")
        interval = "n/a" if low is None or high is None else f"{low:.4f} to {high:.4f}"
        lines.append(f"| {label} | {_num(est)} | {interval} |")
    if fit.get("assumptions"):
        lines += ["", "Fit notes:"] + [f"- {note}" for note in fit["assumptions"]]
    return lines + [""]


def _latency_section(name: str, run: Path, result: dict[str, Any], link: dict[str, Any]) -> list[str]:
    a = result["apparent_latency_s"]
    cam = result["camera_latency_s"]
    lag = result["settings"]["display_lag_s"]
    jitter = a.get("jitter_std_from_straddles")
    camera = (f"- Camera latency, assuming display lag {_ms(lag[0])} to {_ms(lag[1])} "
              f"(an assumption, not a measurement): {_ms(cam['low'])} to {_ms(cam['high'])}")
    lines = [f"### {name}", "",
             f"- Run folder: {run.name}",
             f"- Status: {_status(result.get('status'))}",
             f"- Frames used: {result['frames_used']} of {result['frames_total']}",
             (f"- Apparent latency (camera plus display lag): mean {_ms(a['mean'])} +/- {_ms(a['stated_error'])} "
              f"{INTERVAL_NOTE}"),
             camera,
             f"- Frame-to-frame jitter (std, straddle frames): {_ms(jitter)}",
             f"- Link: delay {_ms(link['delay_s'])}, jitter {_ms(link['jitter_s'])}, status {link['status']}",
             f"- Link source: {link['source']}"]
    lines += [f"- Warning: {w}" for w in link["warnings"]]
    return lines + [""]


def _rerun_section(name: str, manifest: dict[str, Any], output: Path) -> list[str]:
    lines = [f"### {name}", "",
             f"- Status: {manifest['status']}",
             f"- Seeds per cell: {manifest['seeds_count']}, cells per bracket: "
             f"{', '.join(f'{k} {v}' for k, v in manifest['n_cells'].items())}"
             + (" (pilot subset)" if manifest["cell_filter_used"] else " (full main slice)"),
             ""]
    if manifest["cell_filter_used"] or manifest["seeds_count"] == 1:
        lines += [PILOT_NOTE, ""]
    lines += ["| Bracket | Noise used (m) | Source | Recovery median p95 error, min to max (m) |",
              "|---|---|---|---|"]
    for entry in manifest["bracket"]:
        summary = json.loads((output / entry["label"] / "summary.json").read_text(encoding="utf-8"))
        p95 = [cell["arms"]["recovery"]["p95_median_m"] for cell in summary["cells"]]
        lines.append(f"| {entry['label']} | {entry['noise_std_m']:.4f} | {entry['source']} | "
                     f"{min(p95):.4f} to {max(p95):.4f} |")
    return lines + [""]


def _intake_md(*, overall: str, input_rows: list[tuple[str, str, str]], fits: list[list[str]],
               latency: list[list[str]], reruns: list[list[str]], output_rows: list[tuple[str, str]],
               options: dict[str, Any], has_link: bool) -> str:
    lines = ["# Vision intake", "",
             f"Overall status: {overall}", "",
             STATUS_RULE, "",
             SCOPE_NOTE, "",
             f"Options: {options['mode']}, {options['seeds']} seed(s) per cell, bracket {options['bracket']}.", "",
             "## Inputs", "", "| File | sha256 | Status |", "|---|---|---|"]
    lines += [f"| {label} | {digest} | {status} |" for label, digest, status in input_rows]
    lines += ["", "## Vision noise fits", ""]
    lines += [line for block in fits for line in block] if fits else ["No detection log was given.", ""]
    lines += ["## Latency runs and links", ""]
    if latency:
        lines += [line for block in latency for line in block]
    else:
        lines += ["No latency run was given, so no link JSON was written.", ""]
    if has_link:
        lines += [LINK_NOTE, ""]
    lines += ["## S6 rerun with the fitted noise", "",
              RERUN_NOTE, ""]
    lines += [line for block in reruns for line in block] if reruns else ["No detection log was given, so no rerun.", ""]
    lines += ["## Output files", "",
              VOLATILE_NOTE, "",
              "| File | sha256 |", "|---|---|"]
    lines += [f"| {path} | {digest} |" for path, digest in output_rows]
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------- driver

def _clear_contents(folder: Path) -> None:
    for child in folder.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def run_intake(*, logs: list[Path], latency_runs: list[Path], processing: Path | None, output: Path,
               pilot: bool, seeds: int | None, bracket: tuple[str, ...], depth_axis: str | None,
               subject: str | None, dropout_mode: str, bootstrap: int, fit_seed: int) -> Path:
    if not logs and not latency_runs:
        raise IntakeError("give at least one --log or --latency-run")
    if processing is not None and not latency_runs:
        raise IntakeError("--processing needs a latency run to add to (give --latency-run)")
    if seeds is not None and seeds < 1:
        raise IntakeError("seeds must be positive")
    named_logs = _unique_names(logs, "log", lambda p: p.stem)
    named_runs = _unique_names(latency_runs, "latency run", lambda p: p.resolve().name)
    for name, path in named_logs.items():
        _check_log(name, path)
    for name, path in named_runs.items():
        _check_latency_run(name, path)
    processing_info = None
    if processing is not None:
        processing_info = _read_processing(processing)
    inputs = [*logs, *latency_runs, *([processing] if processing else [])]
    resolved, existed = _check_output(output, inputs)
    n_seeds = seeds if seeds is not None else (PILOT_SEEDS if pilot else DEFAULT_SEEDS)

    resolved.mkdir(parents=True, exist_ok=True)
    try:
        return _run_stages(named_logs, named_runs, processing, processing_info, resolved, pilot=pilot, seeds=n_seeds,
                           bracket=bracket, depth_axis=depth_axis, subject=subject, dropout_mode=dropout_mode,
                           bootstrap=bootstrap, fit_seed=fit_seed)
    except BaseException:
        # Only the folder this command made (or emptied) is cleared, so a failed intake leaves no half result.
        if existed:
            _clear_contents(resolved)
        else:
            shutil.rmtree(resolved, ignore_errors=True)
        raise


def _run_stages(named_logs, named_runs, processing, processing_info, out: Path, *, pilot, seeds, bracket, depth_axis, subject,
                dropout_mode, bootstrap, fit_seed) -> Path:
    input_rows: list[tuple[str, str, str]] = []
    statuses: list[str] = []
    fit_blocks: list[list[str]] = []
    latency_blocks: list[list[str]] = []
    rerun_blocks: list[list[str]] = []

    for name, log in named_logs.items():
        fit_path = out / "fits" / f"{name}.fit.json"
        fit_path.parent.mkdir(exist_ok=True)
        fit = _fit_log(name, log, fit_path, subject=subject, dropout_mode=dropout_mode, bootstrap=bootstrap,
                       seed=fit_seed)
        status = _status(fit["input"].get("status"))
        statuses.append(status)
        input_rows.append((log.name, _sha256(log), status))
        fit_blocks.append(_fit_section(name, log, fit))
        manifest = _run_s6(fit_path, out / "s6" / name, bracket=bracket, depth_axis=depth_axis, seeds=seeds,
                           pilot=pilot)
        rerun_blocks.append(_rerun_section(name, manifest, out / "s6" / name))

    if processing is not None:
        statuses.append(processing_info["status"])
        input_rows.append((processing.name, _sha256(processing), processing_info["status"]))

    for name, run in named_runs.items():
        (out / "latency").mkdir(exist_ok=True)
        (out / "links").mkdir(exist_ok=True)
        latency_path = out / "latency" / f"{name}.latency.json"
        result = _analyze_run(name, run, latency_path)
        run_status = _status(result.get("status"))
        statuses.append(run_status)
        for file_name in LATENCY_FILES:
            if (run / file_name).is_file():
                input_rows.append((f"{name}/{file_name}", _sha256(run / file_name),
                                   run_status if file_name == "run_meta.json" else "-"))
        try:
            link = convert_files(latency_path, out / "links" / f"{name}.link.json", processing_path=processing)
        except (LinkConversionError, OSError) as exc:
            raise IntakeError(f"link conversion failed for {name}: {exc}") from exc
        latency_blocks.append(_latency_section(name, run, result, link))

    overall = combine_status(["synthetic" if status == "fixture" else status for status in statuses])
    output_rows = sorted(
        (path.relative_to(out).as_posix(), _sha256(path)) for path in out.rglob("*")
        if path.is_file() and path.name not in VOLATILE)
    options = {"mode": "pilot subset" if pilot else "full main slice", "seeds": seeds,
               "bracket": ",".join(bracket)}
    text = _intake_md(overall=overall, input_rows=input_rows, fits=fit_blocks, latency=latency_blocks,
                      reruns=rerun_blocks, output_rows=output_rows, options=options, has_link=bool(named_runs))
    (out / "INTAKE.md").write_text(text, encoding="utf-8")
    return out / "INTAKE.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, action="append", default=[],
                        help="detection log .jsonl or vision trace .json (repeatable)")
    parser.add_argument("--latency-run", type=Path, action="append", default=[],
                        help="latency run folder from tools/vision_latency_run.py (repeatable)")
    parser.add_argument("--processing", type=Path, help="processing-time JSON (zephyr-processing-time-1)")
    parser.add_argument("--output", type=Path, required=True, help="new or empty folder; never under sim/runs")
    parser.add_argument("--pilot", action="store_true", help="1 seed on a small subset of cells")
    parser.add_argument("--seeds", type=int, help=f"seeds per cell (default {PILOT_SEEDS} with --pilot, "
                                                  f"else {DEFAULT_SEEDS})")
    parser.add_argument("--bracket", default=",".join(BRACKET_LABELS), help="comma list from: pooled, depth")
    parser.add_argument("--depth-axis", choices=("x", "y", "z"))
    parser.add_argument("--subject", help="subject_id to fit (when a log has more than one)")
    parser.add_argument("--dropout-mode", choices=("independent", "burst"), default="independent")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for the fits")
    parser.add_argument("--fit-seed", type=int, default=0)
    args = parser.parse_args(argv)
    labels = tuple(part.strip() for part in args.bracket.split(",") if part.strip())
    start = time.perf_counter()
    try:
        path = run_intake(logs=args.log, latency_runs=args.latency_run, processing=args.processing,
                          output=args.output, pilot=args.pilot, seeds=args.seeds, bracket=labels,
                          depth_axis=args.depth_axis, subject=args.subject, dropout_mode=args.dropout_mode,
                          bootstrap=args.bootstrap, fit_seed=args.fit_seed)
    except IntakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {path} in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
