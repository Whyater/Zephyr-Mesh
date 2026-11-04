"""One command from a bench data folder to one plain-language verdict file (a proposal, not a result).

Runs in the repo's .venv (NumPy, SciPy and Matplotlib only; no camera, no radio, no device).

    PYTHONPATH=. .venv/bin/python tools/bench_verdict.py "$D" --output "$D/out/verdict-pilot" --pilot \\
        [--radio-status measured --operator NAME --captured-at WHEN --code-revision REV \\
         --instrument-model MODEL --source WHERE] [--noise pooled|depth] [--tracker-r registered|measured]

It runs the existing tools in order, each as its own process, with only their public command-line flags:

1. tools/preflight_bench_data.py checks the folder. A failing preflight stops everything here.
2. tools/intake_vision.py fits every detection log and converts every latency run (vision-intake/).
3. tools/intake_espnow.py takes in the radio logs, only when the folder has them (radio-intake/).
4. tools/s6_combined_rerun.py reruns the S6 cell with the worst latency link, the noisiest vision fit and
   the radio fit (combined/). Without radio logs it gets radio-assumed.link.json: 40 ms, no jitter, no loss,
   status "assumed", the same radio term the budget assumes.
5. tools/latency_budget.py adds the delays for each latency run and reads the S6 grid row (budget/). The
   noise row it reads is the smallest S6 noise level at or above the noise the combined rerun applied.

Then it writes VERDICT.md and verdict.json: what went in (with status and sha256), the delay and noise,
where they sit on the S6 grid, PASS, MARGINAL or FAIL with the reasons, the caveats that always apply,
and every command it ran with its exit code.

Verdict rules (worst wins):
FAIL      the delay is above the largest S6 grid delay, or the noise is above the largest S6 noise level,
          or the recovery arm fails on the grid row read, or in the combined rerun, for either motion
          (a cell fails when half or more of its seeds fail).
MARGINAL  nothing failed, but the next grid delay up fails for recovery (or the row read is already the
          largest delay), or the 95% interval on a recovery fail rate reaches one half.
PASS      otherwise.
The data status is separate: synthetic if anything says synthetic or fixture; measured only if every
input and every output in the chain says measured and a measured radio fit was used; otherwise
"not fully measured". A synthetic verdict is a rehearsal, never a bench result.

Exit code: 0 when a verdict was written (PASS, MARGINAL or FAIL), 1 when the chain stopped before a
verdict (VERDICT.md says where and why), 2 for a usage error (such as an output folder that is not new).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.s6_recovery import wilson_interval  # noqa: E402
from tools.preflight_bench_data import Report, find_inputs  # noqa: E402

SCHEMA = "zephyr-bench-verdict-1"
S6_SUMMARY = REPO_ROOT / "sim" / "runs" / "s6-recovery" / "summary.json"
MOTIONS = ("constant_velocity", "constant_turn")
ASSUMED_RADIO_S = 0.040   # tools/latency_budget.py's default radio term (assumed, simulated)
ASSUMED_RADIO_STATUS = "assumed (simulated)"
SYNTHETIC = ("synthetic", "fixture")
GRID_TOLERANCE = 1e-9

ALWAYS_CAVEATS = [
    "The grid row is the registered S6 main slice at 0 loss and 0 jitter. A pass there means a typical, "
    "constant delay on a lossless link, not a worst case. Loss and jitter enter only through the combined rerun.",
    "D adds the camera latency upper bound, the detector processing median and a radio term. The processing "
    "median and the radio term (the top of the interval on the MEAN one-way delay, or the 40 ms assumption) "
    "are not upper bounds, so D is not a worst-case delay.",
    "Motion, link timing and the tracker are simulated. The vision noise comes from a static marker, one "
    "camera, one room and one lighting setup. None of this is flight performance.",
    "The budget D and the combined rerun delay differ by design: the budget uses the camera upper bound and "
    "the top of the radio interval, the combined rerun uses the link means. Both are printed.",
    "The combining rule in the combined rerun (delays add, jitters add in quadrature) is an assumption.",
]


class VerdictError(ValueError):
    """A usage problem: nothing was run."""


class Stop(Exception):
    """The chain stopped before a verdict."""

    def __init__(self, step: str, message: str):
        super().__init__(message)
        self.step = step
        self.message = message


# ---------------------------------------------------------------- the verdict rules


def fail_rate_high(recovery: dict[str, Any]) -> float:
    """Upper end of the 95% interval on the fail rate: the tool's own value, else the same Wilson interval."""
    high = recovery.get("fail_rate_ci_high")
    if isinstance(high, (int, float)) and not isinstance(high, bool) and math.isfinite(high):
        return float(high)
    return wilson_interval(int(recovery["fail_count"]), int(recovery["n_seeds"]))[1]


def _fails(recovery: dict[str, Any]) -> bool:
    return 2 * int(recovery["fail_count"]) >= int(recovery["n_seeds"])


def _counts(recovery: dict[str, Any]) -> str:
    return f"{recovery['fail_count']} of {recovery['n_seeds']} seeds fail"


def decide(grid: dict[str, Any], combined: dict[str, Any]) -> tuple[str, list[str]]:
    """PASS, MARGINAL or FAIL, with one reason line per rule that fired. See the module docstring."""
    fails: list[str] = []
    marginal: list[str] = []
    if grid.get("noise_level_m") is None:
        fails.append("the fitted noise is above the largest S6 noise level, so no grid row applies")
    elif not grid.get("on_grid"):
        fails.append(f"D = {1000 * grid['total_delay_s']:.1f} ms is off the grid (above the largest S6 delay), "
                     "so no grid row applies")
    else:
        point = 1000 * grid["grid_point_s"]
        for row in grid["rows"]:
            rec = row["recovery"]
            if _fails(rec):
                fails.append(f"grid row {point:.0f} ms: recovery fails for {row['motion']} ({_counts(rec)})")
            elif fail_rate_high(rec) >= 0.5:
                marginal.append(f"grid row {point:.0f} ms: the 95% interval on the recovery fail rate for "
                                f"{row['motion']} reaches {fail_rate_high(rec):.2f} ({_counts(rec)})")
        if grid.get("next_rows") is None:
            marginal.append(f"the row read ({point:.0f} ms) is the largest S6 delay, so the headroom above it "
                            "is unknown")
        else:
            nxt = 1000 * grid["next_grid_point_s"]
            for row in grid["next_rows"]:
                if _fails(row["recovery"]):
                    marginal.append(f"the next grid row up ({nxt:.0f} ms) fails for recovery in {row['motion']} "
                                    f"({_counts(row['recovery'])}): one grid step of headroom")
    for cell in combined.get("cells", []):
        rec = cell["arms"]["recovery"]
        if _fails(rec):
            fails.append(f"combined rerun: recovery fails for {cell['motion']} ({_counts(rec)})")
        elif fail_rate_high(rec) >= 0.5:
            marginal.append(f"combined rerun: the 95% interval on the recovery fail rate for {cell['motion']} "
                            f"reaches {fail_rate_high(rec):.2f} ({_counts(rec)}), so these seeds cannot rule "
                            "out a failing cell")
    if fails:
        return "FAIL", fails + marginal
    if marginal:
        return "MARGINAL", marginal
    return "PASS", ["recovery passes on the grid row read, on the next row up, and in the combined rerun, and "
                    "every 95% interval on its fail rate stays under one half"]


def noise_level(noise_m: float, levels: tuple[float, ...]) -> float | None:
    """The smallest S6 noise level at or above ``noise_m``, or None above the largest level."""
    for level in sorted(levels):
        if noise_m <= level + GRID_TOLERANCE:
            return level
    return None


def _is_synthetic(status: Any) -> bool:
    return isinstance(status, str) and (status in SYNTHETIC or "synthetic" in status or "fixture" in status)


def data_status(statuses: dict[str, Any], *, radio_assumed: bool) -> tuple[str, list[str]]:
    """synthetic, measured or "not fully measured", with the reasons. Worst wins."""
    synthetic = [name for name, status in statuses.items() if _is_synthetic(status)]
    if synthetic:
        return "synthetic", [f"{name} says {statuses[name]}" for name in synthetic]
    why = [f"{name} says {status if status else 'nothing (no status)'}"
           for name, status in statuses.items() if status != "measured"]
    if radio_assumed:
        why.append(f"the radio term is {ASSUMED_RADIO_STATUS} (no radio logs in the folder)")
    if not why and statuses:
        return "measured", [f"all {len(statuses)} inputs and outputs say measured, and preflight passed"]
    return "not fully measured", why or ["no statuses were found"]


# ---------------------------------------------------------------- reading files


def _load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(data: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def _fit_noise(fit: Any) -> float | None:
    for value in (_get(fit, "sensor_config", "noise_std_m"), _get(fit, "per_axis", "pooled_std_m")):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def s6_rows(summary: dict[str, Any], noise_m: float) -> dict[float, list[dict[str, Any]]]:
    """The registered S6 main-slice rows at this noise, no loss, no jitter: {delay: [row per motion]}."""
    out: dict[float, list[dict[str, Any]]] = {}
    for c in summary.get("cells", []):
        if (isinstance(c, dict) and c.get("slice") == "main" and c.get("loss_model") == "independent"
                and c.get("loss_probability") == 0.0 and c.get("jitter_s") == 0.0
                and math.isclose(float(c.get("noise_std_m", -1)), noise_m, abs_tol=1e-12)):
            out.setdefault(float(c["delay_s"]), []).append(
                {"motion": c["motion"], "recovery": c["arms"]["recovery"], "naive": c["arms"]["naive"]})
    return out


def s6_noise_levels(summary: dict[str, Any]) -> tuple[float, ...]:
    return tuple(sorted({float(c["noise_std_m"]) for c in summary.get("cells", [])
                         if isinstance(c, dict) and c.get("slice") == "main" and c.get("loss_probability") == 0.0
                         and c.get("jitter_s") == 0.0}))


def _radio_role(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
        if line.strip().lower().replace(" ", "") in ("#role=sender", "#role=echoer"):
            return line.split("=", 1)[1].strip().lower()
    return None


# ---------------------------------------------------------------- running the tools


class Runner:
    def __init__(self, out: Path):
        self.out = out
        self.commands: list[dict[str, Any]] = []
        (out / "logs").mkdir(parents=True, exist_ok=True)

    def run(self, tool: str, args: list[str]) -> subprocess.CompletedProcess:
        python = Path(sys.executable).as_posix()
        argv = [python, f"tools/{tool}", *args]
        command = f"PYTHONPATH=. {shlex.quote(python)} tools/{tool} {shlex.join(args)}".rstrip()
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
        proc = subprocess.run(argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        log = self.out / "logs" / f"{len(self.commands) + 1:02d}-{Path(tool).stem}.txt"
        log.write_text(f"$ {command}\n\n{proc.stdout}\n{proc.stderr}", encoding="utf-8", newline="\n")
        self.commands.append({"tool": tool, "command": command, "exit_code": proc.returncode,
                              "log": log.relative_to(self.out).as_posix()})
        return proc

    def must(self, step: str, tool: str, args: list[str]) -> subprocess.CompletedProcess:
        proc = self.run(tool, args)
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:])
            raise Stop(step, f"{tool} exited with code {proc.returncode}. Its last lines:\n\n{tail}")
        return proc


def _p(path: Path) -> str:
    return Path(path).as_posix()


def check_output(output: Path) -> Path:
    resolved = Path(output).resolve()
    runs = (REPO_ROOT / "sim" / "runs").resolve()
    if resolved == runs or runs in resolved.parents:
        raise VerdictError("the output folder must never be under sim/runs")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise VerdictError(f"the output folder {output} must be new or empty")
    return resolved


def run_chain(folder: Path, out: Path, args: argparse.Namespace, runner: Runner, state: dict[str, Any]) -> None:
    pilot = ["--pilot"] if args.pilot else []
    vision_out, radio_out, combined_out = out / "vision-intake", out / "radio-intake", out / "combined"

    # 1. preflight
    pre_args = [_p(folder), "--output", _p(vision_out)]
    if args.frame_size:
        pre_args += ["--frame-size", args.frame_size]
    proc = runner.run("preflight_bench_data.py", pre_args)
    state["preflight"] = proc.stdout.strip()
    if proc.returncode != 0:
        raise Stop("preflight", "Preflight did not pass, so nothing else ran. Its full output:\n\n"
                   + (proc.stdout + proc.stderr).strip())

    found = find_inputs(folder, Report(folder=folder))
    senders = [p for p in found.radio if _radio_role(p) == "sender"]
    echoers = [p for p in found.radio if _radio_role(p) == "echoer"]
    state["inputs"] = [{"kind": kind, "path": _p(p), "sha256": _sha256(p)}
                       for kind, paths in (("detection log", found.logs), ("processing time", found.processing),
                                           ("calibration", found.calibrations), ("radio log", found.radio))
                       for p in paths]
    state["inputs"] += [{"kind": "latency run", "path": _p(f), "sha256": _sha256(f)}
                        for run in found.latency_runs for f in sorted(run.iterdir()) if f.is_file()]
    if not found.logs:
        raise Stop("inputs", "No detection log in the folder. The verdict needs at least one (runbook step 4).")
    if not found.latency_runs:
        raise Stop("inputs", "No latency run in the folder. The verdict needs at least one (runbook step 6).")
    if len(found.processing) != 1:
        raise Stop("inputs", f"Found {len(found.processing)} processing-time files; the verdict needs exactly one "
                   "(runbook step 5), because the budget adds its median.")
    if found.radio and args.radio_status is None:
        raise Stop("inputs", "The folder has radio logs, but --radio-status was not given. Radio logs carry no "
                   "status of their own. Give --radio-status measured (with --operator, --captured-at, "
                   "--code-revision, --instrument-model and --source) or --radio-status synthetic.")
    if found.radio and (len(senders) != 1 or len(echoers) != 1):
        raise Stop("inputs", f"Found {len(senders)} sender and {len(echoers)} echoer radio logs; the verdict "
                   "takes exactly one of each.")
    processing = found.processing[0]

    # 2. vision intake
    v_args = [a for log in found.logs for a in ("--log", _p(log))]
    v_args += [a for run in found.latency_runs for a in ("--latency-run", _p(run))]
    v_args += ["--processing", _p(processing), "--output", _p(vision_out), *pilot]
    runner.must("vision intake", "intake_vision.py", v_args)

    # 3. radio intake
    radio_fit = None
    if found.radio:
        r_args = ["--sender", _p(senders[0]), "--echoer", _p(echoers[0]), "--status", args.radio_status,
                  "--output", _p(radio_out)]
        for flag in ("operator", "captured_at", "code_revision", "instrument_model", "source"):
            value = getattr(args, flag)
            if value is not None:
                r_args += ["--" + flag.replace("_", "-"), value]
        runner.must("radio intake", "intake_espnow.py", r_args + pilot)
        radio_fit = radio_out / "fit.json"
    state["radio_assumed"] = radio_fit is None

    # Pick the worst latency link (largest delay) and the noisiest vision fit (largest pooled noise).
    links = {run: vision_out / "links" / f"{run.name}.link.json" for run in found.latency_runs}
    fits = {log: vision_out / "fits" / f"{log.stem}.fit.json" for log in found.logs}
    for path in [*links.values(), *fits.values()]:
        if not path.is_file():
            raise Stop("vision intake", f"intake_vision.py did not write {_p(path)}")
    worst_run = max(links, key=lambda run: float(_load(links[run])["delay_s"]))
    noises = {log: _fit_noise(_load(fit)) for log, fit in fits.items()}
    if any(value is None for value in noises.values()):
        raise Stop("vision intake", "a vision fit has no pooled noise value (sensor_config.noise_std_m)")
    worst_log = max(noises, key=lambda log: noises[log])
    state["picked"] = {"latency_run": worst_run.name, "detection_log": worst_log.name,
                       "pooled_noise_by_log_m": {log.name: value for log, value in noises.items()}}

    # 4. combined rerun
    if radio_fit is None:
        radio_link = out / "radio-assumed.link.json"
        radio_link.write_text(json.dumps({
            "delay_s": ASSUMED_RADIO_S, "jitter_s": 0.0, "loss_probability": 0.0, "loss_model": "independent",
            "status": ASSUMED_RADIO_STATUS,
            "note": "stand-in radio term written by tools/bench_verdict.py because the folder has no radio logs: "
                    "the same 40 ms tools/latency_budget.py assumes, with no jitter and no loss; not a measurement"},
            indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    else:
        radio_link = radio_fit
    c_args = ["--radio", _p(radio_link), "--vision-link", _p(links[worst_run]), "--vision-fit", _p(fits[worst_log]),
              "--noise", args.noise, "--tracker-r", args.tracker_r, "--output", _p(combined_out), *pilot]
    runner.must("combined rerun", "s6_combined_rerun.py", c_args)
    combined = _load(combined_out / "summary.json")
    state["combined_summary"] = combined
    applied_noise = float(combined["settings"]["noise_std_m"])

    # 5. latency budget, once per latency run, at the noise level at or above the applied noise
    summary = _load(S6_SUMMARY)
    level = noise_level(applied_noise, s6_noise_levels(summary))
    state["noise_level_m"] = level
    budgets: dict[str, dict[str, Any]] = {}
    if level is not None:
        (out / "budget").mkdir()
        for run in found.latency_runs:
            b_out = out / "budget" / f"{run.name}.json"
            b_args = ["--latency", _p(vision_out / "latency" / f"{run.name}.latency.json"),
                      "--processing", _p(processing), "--noise-std-m", repr(level), "--out", _p(b_out)]
            if radio_fit is not None:
                b_args += ["--fit", _p(radio_fit)]
            runner.must("latency budget", "latency_budget.py", b_args)
            budgets[run.name] = _load(b_out)
    state["budgets"] = budgets
    state["summary"] = summary


def build_grid(state: dict[str, Any]) -> dict[str, Any]:
    level = state["noise_level_m"]
    if level is None:
        return {"on_grid": False, "total_delay_s": None, "grid_point_s": None, "noise_level_m": None, "rows": [],
                "next_rows": None, "next_grid_point_s": None, "worst_run": None}
    worst = max(state["budgets"], key=lambda name: float(state["budgets"][name]["total_delay_s"]))
    b = state["budgets"][worst]
    point = b.get("grid_point_s")
    rows_by_delay = s6_rows(state["summary"], level)
    grid = {"on_grid": point is not None, "total_delay_s": float(b["total_delay_s"]), "grid_point_s": point,
            "noise_level_m": level, "worst_run": worst, "budget_verdict": b.get("verdict"),
            "budget_status": b.get("status"), "components": b.get("components", {}), "next_rows": None,
            "next_grid_point_s": None, "rows": []}
    if point is not None:
        grid["rows"] = rows_by_delay.get(float(point), [])
        if len(grid["rows"]) != len(MOTIONS):
            raise Stop("latency budget", f"the S6 summary has {len(grid['rows'])} rows at {point} s; expected 2")
        above = sorted(d for d in rows_by_delay if d > float(point) + GRID_TOLERANCE)
        if above:
            grid["next_grid_point_s"] = above[0]
            grid["next_rows"] = rows_by_delay[above[0]]
    return grid


def collect_statuses(out: Path, state: dict[str, Any]) -> dict[str, Any]:
    vision = out / "vision-intake"
    statuses: dict[str, Any] = {}
    for entry in state["inputs"]:
        if entry["kind"] == "processing time":
            statuses[f"processing-time file {Path(entry['path']).name}"] = _get(_load(Path(entry["path"])), "status")
    for fit in sorted((vision / "fits").glob("*.fit.json")):
        data = _load(fit)
        statuses[f"vision fit {fit.name}"] = _get(data, "input", "status") or _get(data, "status")
    for name in ("latency", "links"):
        for path in sorted((vision / name).glob("*.json")):
            statuses[f"{name} {path.name}"] = _get(_load(path), "status")
    fit = out / "radio-intake" / "fit.json"
    if fit.is_file():
        statuses["radio fit fit.json"] = _get(_load(fit), "source_trace", "status")
    statuses["combined rerun inputs"] = state["combined_summary"].get("input_status")
    for name, b in state["budgets"].items():
        statuses[f"latency budget {name}"] = b.get("status")
    return statuses


# ---------------------------------------------------------------- writing the verdict


def _ms(value: Any) -> str:
    return "n/a" if value is None else f"{1000 * float(value):.1f} ms"


def _rec(recovery: dict[str, Any]) -> str:
    return f"recovery {recovery['label']} ({_counts(recovery)}, 95% upper {fail_rate_high(recovery):.2f})"


def verdict_markdown(data: dict[str, Any]) -> str:
    verdict = data["verdict"]
    status = data["data_status"]
    if verdict is None:
        head = f"# Bench verdict: NO VERDICT (stopped at {data['stopped_at']})"
    elif status == "synthetic":
        head = f"# Bench verdict: {verdict} on synthetic data (rehearsal only, not a bench result)"
    elif status == "measured":
        head = f"# Bench verdict: {verdict}"
    else:
        head = f"# Bench verdict: {verdict} on data that is not fully measured"
    lines = [head, "", f"Data status: {status}"]
    lines += [f"- {why}" for why in data["data_status_reasons"]]
    lines += ["", "Written by tools/bench_verdict.py. Every number below comes from the commands listed at the end, "
              "so rerunning them regenerates it."]
    if data.get("stop_message"):
        lines += ["", "## Why it stopped", "", "```", data["stop_message"], "```"]
    if data["inputs"]:
        lines += ["", "## What went in", "", "| Kind | File | sha256 |", "|---|---|---|"]
        lines += [f"| {e['kind']} | {e['path']} | `{e['sha256']}` |" for e in data["inputs"]]
    if data.get("statuses"):
        lines += ["", "Status of each input and tool output, as the files state it:", ""]
        lines += [f"- {name}: {value if value else 'no status'}" for name, value in data["statuses"].items()]
    if verdict is not None:
        g, c = data["grid"], data["combined"]
        comp = g.get("components") or {}
        lines += ["", "## Delay and noise", ""]
        if g["total_delay_s"] is not None:
            lines.append(f"- Budget D = {_ms(g['total_delay_s'])} for the worst latency run ({g['worst_run']}): "
                         f"camera upper bound {_ms(_get(comp, 'camera', 'value_s'))} + processing median "
                         f"{_ms(_get(comp, 'processing', 'value_s'))} + radio term {_ms(_get(comp, 'radio', 'value_s'))}"
                         f" ({_get(comp, 'radio', 'label') or _get(comp, 'radio', 'status')}).")
        picked = data["picked"]
        lines.append(f"- Vision noise applied: {1000 * data['noise_std_m']:.1f} mm ({data['noise_label']}), from the "
                     f"fit of {picked['detection_log']}, the noisiest of {len(picked['pooled_noise_by_log_m'])} "
                     "log(s) by pooled noise.")
        lines.append(f"- Combined rerun applied: delay {_ms(c['delay_s'])}, jitter {_ms(c['jitter_s'])}, loss "
                     f"{c['loss_probability']:.3f} ({c['loss_model']}), noise {1000 * c['noise_std_m']:.1f} mm, tracker "
                     f"R {1000 * c['tracker_measurement_std_m']:.1f} mm, {c['n_seeds']} seeds per motion.")
        if data["radio_assumed"]:
            lines.append(f"- Radio: no radio logs, so both steps use the {ASSUMED_RADIO_STATUS} 40 ms term with no "
                         "jitter and no loss (radio-assumed.link.json).")
        lines += ["", "## Where it sits on the S6 grid", ""]
        if g["noise_level_m"] is None:
            lines.append("- The noise is above the largest S6 noise level: no grid row applies.")
        elif not g["on_grid"]:
            lines.append(f"- {g.get('budget_verdict')}")
        else:
            lines.append(f"- Row read: {_ms(g['grid_point_s'])} delay at {100 * g['noise_level_m']:.0f} cm noise, "
                         "0 loss and 0 jitter (the smallest grid delay at or above D, the smallest noise level at or "
                         "above the applied noise).")
            lines += [f"  - {r['motion']}: {_rec(r['recovery'])}" for r in g["rows"]]
            if g["next_rows"] is None:
                lines.append("- Next row up: none (the row read is the largest S6 delay).")
            else:
                lines.append(f"- Next row up: {_ms(g['next_grid_point_s'])}.")
                lines += [f"  - {r['motion']}: {_rec(r['recovery'])}" for r in g["next_rows"]]
        lines.append("- Combined rerun:")
        lines += [f"  - {cell['motion']}: {_rec(cell['arms']['recovery'])}" for cell in c["cells"]]
        lines += ["", f"## Verdict: {verdict}", ""]
        lines += [f"- {reason}" for reason in data["reasons"]]
    lines += ["", "## Caveats (always printed)", ""]
    lines += [f"- {text}" for text in data["caveats"]]
    if data.get("preflight"):
        lines += ["", "## Preflight", "", "```", data["preflight"], "```"]
    lines += ["", "## Commands run, in order, from the repo root", ""]
    lines += [f"{i}. exit {cmd['exit_code']}, output in `{cmd['log']}`:\n   `{cmd['command']}`"
              for i, cmd in enumerate(data["commands"], 1)]
    lines += ["", "## Verdict rules", "", "```", __doc__.split("Verdict rules (worst wins):\n", 1)[1].split(
        "\n\nExit code:", 1)[0], "```", ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="the bench data folder (holds aruco/, latency/, radio/)")
    parser.add_argument("--output", type=Path, required=True, help="new or empty folder; never under sim/runs")
    parser.add_argument("--pilot", action="store_true", help="few seeds in every step (a smoke test of the chain)")
    parser.add_argument("--frame-size", help="expected frame size for the preflight, such as 1280x720")
    parser.add_argument("--noise", choices=("pooled", "depth"), default="pooled",
                        help="which fitted noise the combined rerun applies (default pooled)")
    parser.add_argument("--tracker-r", choices=("registered", "measured"), default="registered",
                        help="tracker measurement std in the combined rerun (default registered 0.05 m)")
    parser.add_argument("--radio-status", choices=("measured", "synthetic", "fixture"),
                        help="status of the radio logs (needed when the folder has them)")
    for flag in ("--operator", "--captured-at", "--code-revision", "--instrument-model", "--source"):
        parser.add_argument(flag, help="radio provenance passed to tools/intake_espnow.py (needed for measured)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        if not args.folder.is_dir():
            raise VerdictError(f"{args.folder} is not a folder; give the bench data folder")
        out = check_output(args.output)
    except VerdictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    folder = args.folder.resolve()
    runner = Runner(out)
    state: dict[str, Any] = {"inputs": [], "radio_assumed": None}
    caveats = list(ALWAYS_CAVEATS)
    data: dict[str, Any] = {"schema": SCHEMA, "folder": _p(folder), "pilot": args.pilot, "verdict": None,
                            "reasons": [], "stopped_at": None, "stop_message": None}
    try:
        run_chain(folder, out, args, runner, state)
        grid = build_grid(state)
        verdict, reasons = decide(grid, state["combined_summary"])
        statuses = collect_statuses(out, state)
        status, status_why = data_status(statuses, radio_assumed=state["radio_assumed"])
        combined = state["combined_summary"]
        settings = dict(combined["settings"])
        settings["n_seeds"] = _get(combined, "design", "n_seeds")
        settings["cells"] = [{"motion": cell["motion"], "arms": {"recovery": cell["arms"]["recovery"]}}
                             for cell in combined["cells"]]
        if state["radio_assumed"]:
            caveats.append("No radio logs: the radio term is 40 ms, assumed from simulation, with no jitter and no "
                           "loss, in both the budget and the combined rerun.")
        if args.pilot:
            caveats.append(f"Pilot run: {settings['n_seeds']} seeds per motion in the combined rerun. A pilot "
                           "verdict is a smoke test of the chain, not a result.")
        if len(state["picked"]["pooled_noise_by_log_m"]) > 1:
            caveats.append("The noisiest log is picked by pooled noise, even with --noise depth.")
        data.update({"verdict": verdict, "reasons": reasons, "data_status": status,
                     "data_status_reasons": status_why, "measured": status == "measured", "statuses": statuses,
                     "total_delay_s": grid["total_delay_s"], "noise_std_m": float(settings["noise_std_m"]),
                     "noise_label": args.noise, "grid": grid, "combined": settings, "picked": state["picked"],
                     "radio_assumed": state["radio_assumed"]})
        code = 0
    except Stop as stop:
        data.update({"stopped_at": stop.step, "stop_message": stop.message, "data_status": "no verdict",
                     "data_status_reasons": ["the chain stopped before a verdict"], "measured": False,
                     "radio_assumed": state.get("radio_assumed")})
        code = 1
    data.update({"preflight": state.get("preflight"), "inputs": state["inputs"], "caveats": caveats,
                 "commands": runner.commands})
    data.setdefault("total_delay_s", None)
    data.setdefault("noise_std_m", None)
    data.setdefault("grid", None)
    data.setdefault("combined", None)
    (out / "verdict.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                      newline="\n")
    text = verdict_markdown(data)
    (out / "VERDICT.md").write_text(text, encoding="utf-8", newline="\n")
    print(text.split("\n## ", 1)[0])
    print(f"\nWrote {_p(out / 'VERDICT.md')} and verdict.json.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
