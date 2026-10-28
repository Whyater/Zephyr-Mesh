"""One command from ESP-NOW echo logs to a summary (a proposal, not a result).

Runs in the repo's .venv (NumPy, SciPy and Matplotlib only; no device is touched).

    PYTHONPATH=. .venv/bin/python tools/intake_espnow.py \\
        --sender sender.csv --echoer echoer.csv --status synthetic --output DIR [--pilot]

    PYTHONPATH=. .venv/bin/python tools/intake_espnow.py \\
        --sender-raw sender_raw.txt --echoer-raw echoer_raw.txt --status measured \\
        --operator NAME --captured-at WHEN --code-revision REV --instrument-model MODEL \\
        --source WHERE --output DIR

The stages, in order (each one is the existing tool, called in this process):

1. raw captures only: tools/extract_echo_log.py cuts one clean log out of each capture;
2. tools/import_espnow_log.py turns the two logs into trace.json;
3. tools/fit_espnow_log.py fits the link (fit.json);
4. tools/burst_bridge.py --json reads the same trace (burst_bridge.json, burst_bridge.txt);
5. tools/s6_link_rerun.py reruns the S6 main slice with the fitted link (s6/).

Everything goes into a new folder, never under sim/runs. INTAKE.md lists every input and output with its
sha256, the status, the fitted values with their intervals, and the S6 numbers. Inputs are never modified.

Status: the one given with --status travels through the trace, the fit and the rerun, and the command stops
if any stage reports a different one. "measured" is refused, before any stage runs, unless --operator,
--captured-at, --code-revision, --instrument-model and --source are all given (the importer writes them into
the trace's provenance block). A fixture or synthetic status is never upgraded.

Two burst rules read the same z statistic with different cutoffs: the fit calls a log bursty above 1.645
(one-sided, at 95% confidence) and the bridge above 2.0 (two-sided). INTAKE.md shows both verdicts side by
side and does not pick one. Both logs are required, because without the echoer log no outbound loss can be
fitted.

--pilot runs one seed on 4 cells of the main slice. It shows that the chain runs and is not a result. Without
it the default is 50 seeds on every cell, as in tools/s6_link_rerun.py.
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
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import (
    burst_bridge,
    extract_echo_log,
    fit_espnow_log,
    import_espnow_log,
    s6_link_rerun,
)

STATUSES = ("measured", "synthetic", "fixture")
PROVENANCE_FLAGS = ("operator", "captured_at", "code_revision", "instrument_model", "source")
PILOT_SEEDS = 1
DEFAULT_SEEDS = 50
DEFAULT_TRACE_ID = "espnow-echo-intake"
VOLATILE = ("INTAKE.md",)

STATUS_NOTE = ("The status is the one given with --status. It is written into trace.json and read back from "
               "trace.json, fit.json and the S6 manifest; the command stops if any of them disagrees. A "
               "synthetic or fixture status says nothing about a real radio link.")
SCOPE_NOTE = ("Scope: simulation inputs only. This does not claim flight performance. The rerun below uses "
              "the fitted link in a simulated tracker with a simulated sensor.")
DELAY_NOTE = ("One-way delay is the round trip divided by two. That assumes both legs take the same mean "
              "time, and nothing in an echo log can check it (fit notes below).")
PILOT_NOTE = ("Pilot: the numbers below come from one seed on a subset of the main slice. They show that the "
              "chain runs. They are not a result.")
VERDICT_NOTE = ("Both rules read the same burstiness z from the same trace. They differ only in the cutoff. "
                "This report shows both and does not combine them or pick one. The S6 rerun below follows "
                "the fit's choice, because it reads the fit's link settings.")


class IntakeError(ValueError):
    """An input, an option or the output folder cannot be used."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "no message"


def _quiet(func, argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = func(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
        except (OSError, ValueError, KeyError, TypeError) as exc:
            err.write(f"{type(exc).__name__}: {exc}\n")
            code = 2
    return int(code or 0), out.getvalue(), err.getvalue()


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- preflight

def check_provenance(status: str, values: dict[str, str | None]) -> None:
    """A measured run needs all five provenance values, non-blank, before anything else happens."""
    if status != "measured":
        return
    missing = [name for name in PROVENANCE_FLAGS if not (values.get(name) or "").strip()]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise IntakeError(f"--status measured needs {flags}; a measured run must say who captured it, when, "
                          "with which code, on which instrument, and from where, so nothing was run")


def _check_output(output: Path, inputs: list[Path]) -> tuple[Path, bool]:
    resolved = output.resolve()
    runs_dir = (REPO_ROOT / "sim" / "runs").resolve()
    if resolved == runs_dir or resolved.is_relative_to(runs_dir):
        raise IntakeError("output must not be inside sim/runs; registered outputs live there")
    if resolved.exists():
        if not resolved.is_dir():
            raise IntakeError(f"output path is not a directory: {resolved.name}")
        if any(resolved.iterdir()):
            raise IntakeError(f"output folder is not empty: {resolved.name}; pick a new folder")
    for path in inputs:
        if path.resolve().is_relative_to(resolved):
            raise IntakeError(f"input {path.name} must not be inside the output folder")
    return resolved, resolved.exists()


def _clear_contents(folder: Path) -> None:
    for child in folder.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


# ---------------------------------------------------------------- stages

def _extract(raw: Path, out: Path, role: str) -> dict[str, Any]:
    try:
        return extract_echo_log.extract_file(raw, out=out, audit_path=out.with_suffix(".audit.json"))
    except (ValueError, OSError) as exc:
        hint = ("; for a capture with board resets, cut it with tools/extract_echo_log.py --split and give "
                "the block you want as a clean log" if "second format line" in str(exc) else "")
        raise IntakeError(f"{role} capture {raw.name}: {exc}{hint}") from exc


def _import_trace(sender: Path, echoer: Path, trace_path: Path, *, trace_id: str, status: str, source: str | None,
                  notes: str | None, provenance: dict[str, str | None]) -> dict[str, Any]:
    argv = ["--sender", str(sender), "--echoer", str(echoer), "--trace-id", trace_id, "--status", status,
            "--out", str(trace_path)]
    for flag, value in (("--source", source), ("--notes", notes)):
        if value:
            argv += [flag, value]
    if status == "measured":
        for name in PROVENANCE_FLAGS[:4]:
            argv += ["--" + name.replace("_", "-"), str(provenance[name])]
    code, _, err = _quiet(import_espnow_log.main, argv)
    if code != 0:
        raise IntakeError(f"import failed: {_last_line(err)}")
    return _load(trace_path)


def _fit(trace_path: Path, fit_path: Path, *, n_boot: int, seed: int) -> dict[str, Any]:
    code, _, err = _quiet(fit_espnow_log.main, [str(trace_path), "--out", str(fit_path), "--n-boot", str(n_boot),
                                                  "--seed", str(seed)])
    if code != 0:
        raise IntakeError(f"fit failed: {_last_line(err)}")
    fit = _load(fit_path)
    if fit.get("s6_deliver") is None:
        raise IntakeError("the fit gave no link settings, so there is nothing to rerun: "
                          f"{fit.get('link_config_null_reason') or 'no reason given'}")
    return fit


def _bridge(trace_path: Path, json_path: Path, txt_path: Path) -> dict[str, Any]:
    code, out, err = _quiet(burst_bridge.main, [str(trace_path), "--format", "trace", "--json", str(json_path)])
    if code != 0:
        raise IntakeError(f"burst bridge failed: {_last_line(err)}")
    _write(txt_path, out)
    return _load(json_path)


def _rerun(fit_path: Path, out: Path, *, pilot: bool, seeds: int) -> dict[str, Any]:
    argv = ["fit.json", "--output", "s6", "--seeds", str(seeds)] + (["--pilot"] if pilot else [])
    try:
        return s6_link_rerun.run_rerun(fit_path, out, pilot=pilot, seeds_count=seeds, argv=argv)
    except (s6_link_rerun.LinkRerunError, OSError) as exc:
        raise IntakeError(f"S6 rerun failed: {exc}") from exc


def verdicts_agree(fit_burst: bool, bridge_verdict: str) -> bool:
    """True when the fit's burst/independent choice matches the bridge's verdict.

    "Anti-bursty" counts as not bursty (the fit has no such label). A bridge that could not decide never agrees.
    """
    if bridge_verdict.startswith(burst_bridge.VERDICT_UNKNOWN):
        return False
    return fit_burst == bridge_verdict.startswith(burst_bridge.VERDICT_BURSTY)


# ---------------------------------------------------------------- INTAKE.md

def _num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _entry(record: dict[str, Any] | None, scale: float = 1.0) -> tuple[float | None, float | None, float | None]:
    if not record or record.get("value") is None:
        return None, None, None
    ci = record.get("ci") or [None, None]
    low, high = (None if v is None else scale * v for v in ci)
    return scale * record["value"], low, high


def _fit_rows(fit: dict[str, Any]) -> list[tuple[str, tuple[float | None, float | None, float | None]]]:
    one_way = fit["delay"]["one_way_estimate"]
    loss = fit["loss"]
    chain = loss["burst_chain"]
    rate = loss["loss_rate"]
    low, high = (loss["loss_rate_ci"] or [None, None])
    return [
        ("one-way delay (ms)", _entry(one_way["delay_s"], 1000.0)),
        ("one-way delay jitter (ms)", _entry(one_way["delay_jitter_s"], 1000.0)),
        ("outbound loss rate (lost fraction)", (rate, low, high)),
        ("burst start probability", _entry(chain["burst_start_probability"])),
        ("burst end probability", _entry(chain["burst_end_probability"])),
        ("stationary loss rate", _entry(chain["stationary_loss_rate"])),
        ("mean loss run (packets)", _entry(chain["mean_loss_run_packets"])),
    ]


def _fit_section(fit: dict[str, Any]) -> list[str]:
    level = f"{100 * fit['confidence']:.0f}%"
    lines = ["## Link fit", "",
             (f"- Trace id: {fit['source_trace']['trace_id']}, status {fit['source_trace']['status']}, "
              f"{fit['source_trace']['n_packets']} packets"),
             f"- Echoes used for delay: {fit['delay']['n']}",
             f"- Packets with a known outbound outcome: {fit['loss']['n']} ({fit['loss']['n_lost']} lost)",
             (f"- Return leg: {fit['return_leg']['n_no_echo']} of {fit['return_leg']['n_echoer_logged']} "
              "logged packets had no echo back (this includes echoes the echoer chose not to send)"),
             "", DELAY_NOTE, "",
             f"| Parameter | Estimate | {level} interval |", "|---|---|---|"]
    for label, (estimate, low, high) in _fit_rows(fit):
        interval = "n/a" if low is None or high is None else f"{low:.4f} to {high:.4f}"
        lines.append(f"| {label} | {_num(estimate)} | {interval} |")
    lines += ["", "Fit notes:"] + [f"- {note}" for note in fit["delay"]["assumptions"]]
    lines += [f"- {note}" for note in fit["delay"]["consequences"]]
    return lines + [""]


def _verdict_section(fit: dict[str, Any], bridge: dict[str, Any]) -> list[str]:
    check = fit["loss"]["independence_check"]
    choice = fit["loss"]["model_choice"]["loss_model"]
    fit_burst = choice == "burst"
    z_fit = check["approx_z"]
    z_bridge = bridge["gilbert"]["burstiness"]["approx_z"]
    cutoff = check["threshold"]
    lines = ["## Burst verdicts (two rules, not combined)", "", VERDICT_NOTE, "",
             "| Rule | z | Cutoff | Verdict |", "|---|---|---|---|",
             (f"| fit (tools/fit_espnow_log.py) | {_num(z_fit, 3)} | burst if z > {_num(cutoff, 3)} "
              f"(one-sided, {100 * fit['confidence']:.0f}% confidence) and end probability < 1 | "
              f"{choice} model |"),
             (f"| bridge (tools/burst_bridge.py) | {_num(z_bridge, 3)} | bursty if z >= "
              f"{burst_bridge.IID_Z_CUTOFF}, anti-bursty if z <= -{burst_bridge.IID_Z_CUTOFF}, otherwise "
              f"independent | {bridge['verdict']} |"), ""]
    if bridge["verdict"].startswith(burst_bridge.VERDICT_UNKNOWN):
        lines.append("The bridge could not give a verdict, so there is nothing to compare.")
    elif verdicts_agree(fit_burst, bridge["verdict"]):
        word = "bursty" if fit_burst else "not bursty"
        lines.append(f"The two rules agree on this log: both say {word}.")
    else:
        lines.append(f"The two rules disagree on this log: the fit chooses the {choice} model and the bridge says "
                     f"\"{bridge['verdict']}\". This happens when z lies between the two cutoffs.")
    return lines + [""]


def _rerun_section(manifest: dict[str, Any], out: Path) -> list[str]:
    summary = _load(out / "summary.json")
    prov = summary["provenance"]
    lines = ["## S6 rerun with the fitted link", "",
             f"- Status label: {summary['status']}",
             f"- Link input: {prov['input_kind']}, status {prov['input_status']}, trace id {prov['trace_id']}",
             (f"- Link: delay {1000 * summary['link']['delay_s']:.3f} ms, jitter "
              f"{1000 * summary['link']['jitter_s']:.3f} ms, loss {summary['link']['loss_probability']:.4f}, "
              f"model {summary['link']['loss_model']}"),
             f"- Seeds per cell: {manifest['seeds_count']}, cells: {manifest['n_cells']}"
             + (" (pilot subset)" if manifest["pilot"] else " (full main slice)"),
             f"- Limits: {'; '.join(summary['limits'])}", ""]
    if manifest["pilot"]:
        lines += [PILOT_NOTE, ""]
    lines += ["| Motion | Sensor noise (m) | Naive p95 median (m) | Recovery p95 median (m) |", "|---|---|---|---|"]
    for cell in summary["cells"]:
        lines.append(f"| {cell['motion']} | {cell['noise_std_m']:.2f} | "
                     f"{cell['arms']['naive']['p95_median_m']:.4f} | {cell['arms']['recovery']['p95_median_m']:.4f} |")
    return lines + [""]


def _intake_md(*, status: str, input_rows: list[tuple[str, str, str]], provenance: dict[str, str | None],
               audits: list[tuple[str, dict[str, Any]]], fit: dict[str, Any], bridge: dict[str, Any],
               manifest: dict[str, Any], s6_dir: Path, output_rows: list[tuple[str, str]],
               options: dict[str, Any]) -> str:
    lines = ["# ESP-NOW echo intake", "", f"Overall status: {status}", "", STATUS_NOTE, "", SCOPE_NOTE, "",
             (f"Options: {options['mode']}, {options['seeds']} seed(s) per cell, {options['n_boot']} bootstrap "
              f"resamples, fit seed {options['seed']}."), "",
             "## Inputs", "", "| File | sha256 | Role |", "|---|---|---|"]
    lines += [f"| {name} | {digest} | {role} |" for name, digest, role in input_rows]
    if status == "measured":
        lines += ["", "Provenance (as given on the command line, written into trace.json):", ""]
        lines += [f"- {name.replace('_', ' ')}: {provenance[name]}" for name in PROVENANCE_FLAGS]
    if audits:
        lines += ["", "## Extraction from raw captures", ""]
        for role, audit in audits:
            rows = sum(block["data_rows"] for block in audit["blocks"])
            lines.append(f"- {role}: {len(audit['blocks'])} block(s), {rows} data rows, "
                         f"{len(audit['dropped'])} line(s) left out (boot text and similar), raw sha256 "
                         f"{audit['raw_sha256']}")
    lines += [""] + _fit_section(fit) + _verdict_section(fit, bridge) + _rerun_section(manifest, s6_dir)
    lines += ["## Output files", "",
              "INTAKE.md does not hash itself. burst_bridge.txt is the bridge's own printed report.", "",
              "| File | sha256 |", "|---|---|"]
    lines += [f"| {path} | {digest} |" for path, digest in output_rows]
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------- driver

def run_intake(*, sender: Path | None, echoer: Path | None, sender_raw: Path | None, echoer_raw: Path | None,
               status: str, output: Path, trace_id: str = DEFAULT_TRACE_ID, source: str | None = None,
               notes: str | None = None, operator: str | None = None, captured_at: str | None = None,
               code_revision: str | None = None, instrument_model: str | None = None, pilot: bool = False,
               seeds: int | None = None, n_boot: int = 2000, seed: int = 0) -> Path:
    clean, raw = (sender, echoer), (sender_raw, echoer_raw)
    if any(clean) and any(raw):
        raise IntakeError("give either --sender and --echoer (clean logs) or --sender-raw and --echoer-raw "
                          "(raw captures), not both")
    pair = clean if any(clean) else raw
    if not all(pair):
        raise IntakeError("give both logs: --sender and --echoer, or --sender-raw and --echoer-raw; without "
                          "the echoer log no outbound loss can be fitted")
    if status not in STATUSES:
        raise IntakeError(f"--status must be one of {', '.join(STATUSES)}")
    provenance = {"operator": operator, "captured_at": captured_at, "code_revision": code_revision,
                  "instrument_model": instrument_model, "source": source}
    check_provenance(status, provenance)
    if (seeds is not None and seeds < 1) or n_boot < 1:
        raise IntakeError("--seeds and --n-boot must be positive")
    for path in pair:
        if not path.is_file():
            raise IntakeError(f"input file not found: {path.name}")
    resolved, existed = _check_output(output, list(pair))
    n_seeds = seeds if seeds is not None else (PILOT_SEEDS if pilot else DEFAULT_SEEDS)
    resolved.mkdir(parents=True, exist_ok=True)
    try:
        return _run_stages(resolved, pair, is_raw=bool(any(raw)), status=status, trace_id=trace_id, source=source,
                           notes=notes, provenance=provenance, pilot=pilot, seeds=n_seeds, n_boot=n_boot, seed=seed)
    except BaseException:
        # Only the folder this command made (or emptied) is cleared, so a failed intake leaves no half result.
        if existed:
            _clear_contents(resolved)
        else:
            shutil.rmtree(resolved, ignore_errors=True)
        raise


def _run_stages(out: Path, pair: tuple[Path, Path], *, is_raw: bool, status: str, trace_id: str,
                source: str | None, notes: str | None, provenance: dict[str, str | None], pilot: bool,
                seeds: int, n_boot: int, seed: int) -> Path:
    roles = ("sender", "echoer")
    input_rows: list[tuple[str, str, str]] = []
    audits: list[tuple[str, dict[str, Any]]] = []
    logs: list[Path] = []
    if is_raw:
        (out / "logs").mkdir()
        for role, raw in zip(roles, pair):
            input_rows.append((raw.name, _sha256(raw), f"{role} raw capture"))
            clean = out / "logs" / f"{role}.csv"
            audits.append((role, _extract(raw, clean, role)))
            logs.append(clean)
    else:
        for role, path in zip(roles, pair):
            input_rows.append((path.name, _sha256(path), f"{role} log"))
        logs = list(pair)

    trace_path, fit_path = out / "trace.json", out / "fit.json"
    trace = _import_trace(logs[0], logs[1], trace_path, trace_id=trace_id, status=status, source=source,
                          notes=notes, provenance=provenance)
    fit = _fit(trace_path, fit_path, n_boot=n_boot, seed=seed)
    bridge = _bridge(trace_path, out / "burst_bridge.json", out / "burst_bridge.txt")
    manifest = _rerun(fit_path, out / "s6", pilot=pilot, seeds=seeds)

    seen = {"trace.json": trace["status"], "fit.json": fit["source_trace"]["status"],
            "s6/manifest.json": manifest["provenance"]["input_status"]}
    wrong = {name: value for name, value in seen.items() if value != status}
    if wrong:
        raise IntakeError(f"status changed between stages (asked for {status}): "
                          + ", ".join(f"{name} says {value}" for name, value in wrong.items()))

    output_rows = sorted(
        (path.relative_to(out).as_posix(), _sha256(path)) for path in out.rglob("*")
        if path.is_file() and path.name not in VOLATILE)
    options = {"mode": "pilot subset" if pilot else "full main slice", "seeds": seeds, "n_boot": n_boot,
               "seed": seed}
    _write(out / "INTAKE.md", _intake_md(status=status, input_rows=input_rows, provenance=provenance,
                                         audits=audits, fit=fit, bridge=bridge, manifest=manifest,
                                         s6_dir=out / "s6", output_rows=output_rows, options=options))
    return out / "INTAKE.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sender", type=Path, help="clean sender log (zephyr-espnow-echo-log-1)")
    parser.add_argument("--echoer", type=Path, help="clean echoer log")
    parser.add_argument("--sender-raw", type=Path, help="raw serial capture of the sender board")
    parser.add_argument("--echoer-raw", type=Path, help="raw serial capture of the echoer board")
    parser.add_argument("--status", required=True, choices=STATUSES)
    parser.add_argument("--output", type=Path, required=True, help="new or empty folder; never under sim/runs")
    parser.add_argument("--trace-id", default=DEFAULT_TRACE_ID)
    parser.add_argument("--source", help="where the capture came from (needed for measured)")
    parser.add_argument("--notes")
    parser.add_argument("--operator", help="who captured it (needed for measured)")
    parser.add_argument("--captured-at", help="when it was captured (needed for measured)")
    parser.add_argument("--code-revision", help="firmware and tool revision (needed for measured)")
    parser.add_argument("--instrument-model", help="board and radio model (needed for measured)")
    parser.add_argument("--pilot", action="store_true", help="1 seed on 4 cells of the main slice")
    parser.add_argument("--seeds", type=int, help=f"seeds per cell (default {PILOT_SEEDS} with --pilot, "
                                                  f"else {DEFAULT_SEEDS})")
    parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for the fit")
    parser.add_argument("--seed", type=int, default=0, help="fit seed")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    try:
        path = run_intake(sender=args.sender, echoer=args.echoer, sender_raw=args.sender_raw,
                          echoer_raw=args.echoer_raw, status=args.status, output=args.output,
                          trace_id=args.trace_id, source=args.source, notes=args.notes, operator=args.operator,
                          captured_at=args.captured_at, code_revision=args.code_revision,
                          instrument_model=args.instrument_model, pilot=args.pilot, seeds=args.seeds,
                          n_boot=args.n_boot, seed=args.seed)
    except IntakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {path} in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
