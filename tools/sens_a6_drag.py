"""Side run: does the EXP-021 arm b delay limit depend on drag being off?

Both literature-derived profiles set the drag coefficient to 0, and the
control-tolerance harness builds ``ActuatedQuadrotor`` without drag
arguments, so the registered d_crit (sim/runs/a6-control-tolerance) has no
aerodynamic drag.  This wrapper reruns arm b, step reference, zero loss,
seed 0, with quadratic drag switched on.

With drag on, the EXP-021 verdict is not monotone in delay (a run can pass
at 0.5 s and fail at 0.8 s), so bisection can land on the wrong boundary.
The drag variants are therefore scanned on the full 10 ms delay grid, and
each flight records both the EXP-021 verdict and the stricter "failed to
fly" verdict (the flight itself is the same for both).  Controls: the
registered ``find_d_crit`` search and the same grid scan, both with drag
off, must reproduce the registered arm b entry exactly before any drag
variant runs.

The registered harness is imported read-only.  Drag is set by replacing the
module attribute ``sim.physics.control_tolerance.ActuatedQuadrotor`` with a
``functools.partial`` for the duration of one search or scan, and the drag
values on every vehicle actually built are recorded so the output proves
the patch took effect.  Everything runs in one process (no worker pool).

Outputs are simulated only and go to a folder outside ``sim/runs``.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sim.physics.control_tolerance as ct
from sim.vehicle import ActuatedQuadrotor

PROFILES = {
    "crazyflie-class": ROOT / "sim" / "physics" / "profiles" / "crazyflie_class.json",
    "esp-fly-class": ROOT / "sim" / "physics" / "profiles" / "esp_fly_class.json",
}
CONFIG_PATH = ROOT / "tools" / "a_control_tolerance_config.json"
REGISTERED_DCRIT = ROOT / "sim" / "runs" / "a6-control-tolerance" / "d_crit.json"
# Drag areas already used by the disturbance experiment (tools/a_disturbance_config.json).
AREAS_M2 = (0.002, 0.005, 0.012)
DRAG_COEFFICIENT = 1.0
EPISODE_DURATION_S = 20.0
SEED = 0
# Delay grid for the scan, in 10 ms control ticks: 0.25 s to 0.60 s.
SCAN_TICKS = tuple(range(25, 61))
# Fields find_d_crit keeps for the stable and unstable sides.
SIDE_KEYS = ("rms_error_m", "max_error_m", "whole_run_max_error_m", "whole_run_max_tilt_deg",
             "whole_run_min_z_m", "unstable_reason", "unstable_time_s")
ROW_KEYS = SIDE_KEYS + ("unstable", "failed_to_fly", "failed_to_fly_reason", "failed_to_fly_time_s")


@contextlib.contextmanager
def drag_patch(drag_coefficient: float | None, area_m2: float | None) -> Iterator[list[tuple[float, float]]]:
    """Build every harness vehicle with the given drag, and log what was built.

    ``None`` for both values leaves the registered constructor in place (the
    profile's own drag values are used) but still logs the built vehicles.
    """
    built: list[tuple[float, float]] = []
    kwargs: dict[str, float] = {}
    if drag_coefficient is not None:
        kwargs["drag_coefficient"] = float(drag_coefficient)
    if area_m2 is not None:
        kwargs["cross_sectional_area"] = float(area_m2)
    make = functools.partial(ActuatedQuadrotor, **kwargs)

    def logged(*args: Any, **kw: Any) -> ActuatedQuadrotor:
        vehicle = make(*args, **kw)
        built.append((float(vehicle.drone.drag_coefficient), float(vehicle.drone.cross_sectional_area)))
        return vehicle

    original = ct.ActuatedQuadrotor
    ct.ActuatedQuadrotor = logged
    try:
        yield built
    finally:
        ct.ActuatedQuadrotor = original


def search(profile_path: Path, drag_coefficient: float | None, area_m2: float | None) -> dict[str, Any]:
    """Run the registered arm b, p = 0 d_crit search with drag patched in."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bracket = tuple(config["d_crit"]["arm_b"]["bracket"])
    with drag_patch(drag_coefficient, area_m2) as built:
        answer = ct.find_d_crit(
            str(profile_path), arm="b", resolution_s=ct.CONTROL_DT_S, bracket=bracket,
            duration_s=EPISODE_DURATION_S, scoring_windows=config.get("scoring_windows"),
        )
    effective = sorted(set(built))
    if len(effective) != 1:
        raise RuntimeError(f"expected one drag setting across the search, got {effective}")
    answer["episode_duration_s"] = EPISODE_DURATION_S
    answer["effective_drag_coefficient"], answer["effective_area_m2"] = effective[0]
    answer["episodes_run"] = len(built)
    return answer


def _first_failure(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """First failing grid delay, and whether any later delay passes again."""
    failing = [row["delay_s"] for row in rows if row[key]]
    if not failing:
        return {"first_fail_delay_s": None, "monotone": True}
    first = failing[0]
    passes_after = [row["delay_s"] for row in rows if row["delay_s"] > first and not row[key]]
    return {"first_fail_delay_s": first, "monotone": not passes_after,
            "passes_after_first_fail_s": passes_after}


def scan(profile_path: Path, drag_coefficient: float | None, area_m2: float | None,
         ticks: tuple[int, ...] = SCAN_TICKS) -> dict[str, Any]:
    """Run every grid delay once and summarize both failure verdicts."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = []
    with drag_patch(drag_coefficient, area_m2) as built:
        for tick in ticks:
            out = ct.run_episode(
                str(profile_path), "b", "step", tick * ct.CONTROL_DT_S, 0.0, SEED,
                duration_s=EPISODE_DURATION_S, scoring_windows=config.get("scoring_windows"),
                failed_to_fly_rule=True,
            )
            row = {"delay_s": round(out["delay_s"], 10), **{key: out[key] for key in ROW_KEYS}}
            rows.append(row)
    effective = sorted(set(built))
    if len(effective) != 1:
        raise RuntimeError(f"expected one drag setting across the scan, got {effective}")
    return {
        "effective_drag_coefficient": effective[0][0],
        "effective_area_m2": effective[0][1],
        "exp021": _first_failure(rows, "unstable"),
        "failed_to_fly": _first_failure(rows, "failed_to_fly"),
        # Runs the EXP-021 rules call stable although the vehicle went below z = 0.
        "exp021_stable_below_ground_s": [row["delay_s"] for row in rows
                                         if not row["unstable"] and row["whole_run_min_z_m"] < 0.0],
        "rows": rows,
    }


def scan_matches_registered(result: dict[str, Any], registered: dict[str, Any]) -> bool:
    """The drag-off scan must give the registered boundary and side diagnostics."""
    # Compare on whole control ticks: 35 * 0.01 is not exactly 0.35 in floating point.
    def tick(delay_s: float | None) -> int | None:
        return None if delay_s is None else round(delay_s / ct.CONTROL_DT_S)

    by_tick = {tick(row["delay_s"]): row for row in result["rows"]}
    stable = by_tick.get(tick(registered["stable_delay_s"]))
    unstable = by_tick.get(tick(registered["unstable_delay_s"]))
    if stable is None or unstable is None:
        return False
    return (tick(result["exp021"]["first_fail_delay_s"]) == tick(registered["d_crit_s"])
            and all(stable[key] == registered["stable_side"][key] for key in SIDE_KEYS)
            and all(unstable[key] == registered["unstable_side"][key] for key in SIDE_KEYS))


def registered_arm_b(profile_id: str) -> dict[str, Any]:
    doc = json.loads(REGISTERED_DCRIT.read_text(encoding="utf-8"))
    return doc["profiles"][profile_id]["arm_b"]["p0"]


def control_matches(answer: dict[str, Any], registered: dict[str, Any]) -> bool:
    """True when every registered field is reproduced exactly."""
    return all(answer.get(key) == value for key, value in registered.items())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(output_dir: Path, areas: tuple[float, ...] = AREAS_M2,
        profiles: tuple[str, ...] = tuple(PROFILES)) -> dict[str, Any]:
    """Run the controls, then the drag variants only if every control passes."""
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to((ROOT / "sim" / "runs").resolve()):
        raise ValueError("side-run output must not go under sim/runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.process_time()
    result: dict[str, Any] = {
        "schema": "zephyr-mesh-sens-a6-drag-1",
        "simulated_only": True,
        "arm": "b", "reference": "step", "loss_p": 0.0, "seed": SEED,
        "drag_coefficient": DRAG_COEFFICIENT,
        "controls": {}, "variants": {},
    }
    all_controls_pass = True
    for pid in profiles:
        registered = registered_arm_b(pid)
        controls = {
            # Registered path: profile drag values (Cd 0, area 0.001).
            "profile_drag": search(PROFILES[pid], None, None),
            # Drag switched on with zero area must also give zero drag force.
            "cd1_area0": search(PROFILES[pid], DRAG_COEFFICIENT, 0.0),
        }
        for answer in controls.values():
            answer["matches_registered"] = control_matches(answer, registered)
        grid = scan(PROFILES[pid], None, None)
        grid["matches_registered"] = scan_matches_registered(grid, registered)
        controls["profile_drag_scan"] = grid
        all_controls_pass = all_controls_pass and all(c["matches_registered"] for c in controls.values())
        result["controls"][pid] = controls
    result["controls_pass"] = all_controls_pass
    if all_controls_pass:
        for pid in profiles:
            result["variants"][pid] = {
                f"{area:g}": scan(PROFILES[pid], DRAG_COEFFICIENT, area) for area in areas
            }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
    inputs = [Path(__file__).resolve(), CONFIG_PATH, REGISTERED_DCRIT, *(PROFILES[p] for p in profiles)]
    manifest = {
        "command": [sys.executable, "tools/sens_a6_drag.py", *sys.argv[1:]],
        "git_commit": _git_commit(),
        "seeds": [SEED],
        "scan_delays_s": [round(t * ct.CONTROL_DT_S, 10) for t in SCAN_TICKS],
        "areas_m2": list(areas),
        "inputs_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in inputs},
        "result_sha256": _sha256(output_dir / "result.json"),
        "cpu_s": round(time.process_time() - started, 1),
        "workers": 1,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# --long-check: the registered 80 s long check, with drag patched in.
#
# Flights use ``fly()`` from tools/a_make_dcrit_long_check.py unchanged; only
# that module's ``ActuatedQuadrotor`` is swapped, the same way as above.  The
# control reruns the registered job list with drag off and must reproduce the
# registered long_check.json byte for byte before any drag level runs.  Each
# drag level is saved to its own file as soon as it finishes, and a rerun
# into the same folder reuses finished levels.
# ---------------------------------------------------------------------------
REGISTERED_LONG = ROOT / "sim" / "runs" / "a6-long-check" / "long_check.json"
REGISTRY_PATH = ROOT / "tools" / "reproduce_registry.json"
# 0.32 s to 0.42 s on the 10 ms grid, written as the same literals the
# registered job list uses, so drag-off rows at 0.32 to 0.36 can be compared.
LONG_DELAYS_S = tuple(round(tick / 100, 2) for tick in range(32, 43))
LONG_STEPS_M = (1.0, 0.01)
# (name, drag coefficient, area); "off" keeps the profile values (Cd 0).
LONG_LEVELS = (("off", None, None),) + tuple((f"{a:g}", DRAG_COEFFICIENT, a) for a in AREAS_M2)
# Tilt and floor limits copied from the failed-to-fly rule (45 deg, z = 0).
LONG_TILT_LIMIT_DEG = 45.0
LONG_FLOOR_M = 0.0


def _long_module():
    from tools import a_make_dcrit_long_check as lc
    return lc


@contextlib.contextmanager
def long_drag_patch(drag_coefficient: float | None, area_m2: float | None) -> Iterator[list[tuple[float, float]]]:
    """Like ``drag_patch``, but for the vehicles the long-check ``fly()`` builds."""
    lc = _long_module()
    built: list[tuple[float, float]] = []
    kwargs: dict[str, float] = {}
    if drag_coefficient is not None:
        kwargs["drag_coefficient"] = float(drag_coefficient)
    if area_m2 is not None:
        kwargs["cross_sectional_area"] = float(area_m2)
    make = functools.partial(ActuatedQuadrotor, **kwargs)

    def logged(*args: Any, **kw: Any) -> ActuatedQuadrotor:
        vehicle = make(*args, **kw)
        built.append((float(vehicle.drone.drag_coefficient), float(vehicle.drone.cross_sectional_area)))
        return vehicle

    original = lc.ActuatedQuadrotor
    lc.ActuatedQuadrotor = logged
    try:
        yield built
    finally:
        lc.ActuatedQuadrotor = original


def long_report_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize rows exactly the way the registered long check writes long_check.json."""
    lc = _long_module()
    rows = sorted(rows, key=lambda r: (r["profile"], r["arm"], r["delay_s"], -r["step_m"]))
    report = lc._round({"schema": "zephyr-exp-021-long-check-1", "duration_s": lc.DURATION_S,
                        "window_s": lc.WINDOW_S,
                        "note": "3-D position error; growing means the last 10 s window peak exceeds the third window peak, or the run left the 5 m box (5 x step size for steps above 1 m)",
                        "rows": rows})
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def registered_long_sha256() -> str:
    """The long_check.json hash recorded in the reproduce registry (read only)."""
    for entry in json.loads(REGISTRY_PATH.read_text(encoding="utf-8")):
        if entry.get("run_dir") == "sim/runs/a6-long-check":
            return entry["outputs"]["long_check.json"]
    raise KeyError("sim/runs/a6-long-check is not in the reproduce registry")


def long_fail(row: dict[str, Any]) -> tuple[bool, list[str]]:
    """The pre-registered long-run fail flag: growing, too much tilt, or below the floor."""
    reasons = []
    if row["growing"]:
        reasons.append("left_box" if row["stopped_at_s"] is not None else "growing")
    if any(t is not None and t > LONG_TILT_LIMIT_DEG for t in row["window_peak_tilt_deg"]):
        reasons.append("tilt_over_45_deg")
    if any(z is not None and z < LONG_FLOOR_M for z in row["window_lowest_z_m"]):
        reasons.append("below_floor")
    return bool(reasons), reasons


def long_limit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Smallest grid delay with a long fail, and whether every larger delay also fails."""
    rows = sorted(rows, key=lambda r: r["delay_s"])
    failing = [r["delay_s"] for r in rows if r["long_fail"]]
    if not failing:
        return {"limit_s": None, "monotone": True, "passes_after_limit_s": []}
    passes_after = [r["delay_s"] for r in rows if r["delay_s"] > failing[0] and not r["long_fail"]]
    return {"limit_s": failing[0], "monotone": not passes_after, "passes_after_limit_s": passes_after}


def _fly_rows(jobs: list[tuple[str, str, float, float]], drag_coefficient: float | None,
              area_m2: float | None) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
    lc = _long_module()
    with long_drag_patch(drag_coefficient, area_m2) as built:
        rows = [lc.fly(job) for job in jobs]
    effective = sorted(set(built))
    if len(effective) != 1:
        raise RuntimeError(f"expected one drag setting across the jobs, got {effective}")
    return rows, effective


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def run_long_check(output_dir: Path, levels: tuple = LONG_LEVELS,
                   profiles: tuple[str, ...] = tuple(PROFILES),
                   delays: tuple[float, ...] = LONG_DELAYS_S,
                   log=print) -> dict[str, Any]:
    """Control first, then each drag level, checkpointing each level to its own file."""
    lc = _long_module()
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to((ROOT / "sim" / "runs").resolve()):
        raise ValueError("side-run output must not go under sim/runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.process_time()

    # Control: the registered job list, drag off, must match the registered file exactly.
    control_path = output_dir / "control_long_check.json"
    if not control_path.exists():
        jobs = [(p, arm, d, a) for p in sorted(PROFILES) for arm, d, a in lc.JOBS]
        rows, _ = _fly_rows(jobs, None, None)
        _write_atomic(control_path, long_report_bytes(rows))
    control_bytes = control_path.read_bytes()
    control_sha = hashlib.sha256(control_bytes).hexdigest()
    registered_rows = json.loads(REGISTERED_LONG.read_text(encoding="utf-8"))["rows"]
    control_rows = json.loads(control_bytes)["rows"]
    control = {
        "sha256": control_sha,
        "registry_sha256": registered_long_sha256(),
        "registered_file_sha256": _sha256(REGISTERED_LONG),
        "rows_equal_registered": control_rows == registered_rows,
    }
    control["matches_registered"] = (control["rows_equal_registered"]
                                     and control_sha == control["registry_sha256"])
    log(f"long control: sha256 {control_sha[:12]}, registry {control['registry_sha256'][:12]}, "
        f"rows equal {control['rows_equal_registered']}")
    result: dict[str, Any] = {
        "schema": "zephyr-mesh-sens-a6-drag-long-1",
        "simulated_only": True,
        "arm": "b", "loss_p": 0.0, "seed": 0, "duration_s": lc.DURATION_S, "window_s": lc.WINDOW_S,
        "delays_s": list(delays), "steps_m": list(LONG_STEPS_M),
        "fail_rule": "registered growing flag, or any 10 s window peak tilt above 45 deg, "
                     "or any window lowest z below 0 m",
        "control": control, "levels": {},
    }
    if control["matches_registered"]:
        control_by_key = {(r["profile"], r["arm"], r["delay_s"], r["step_m"]): r for r in control_rows}
        for name, cd, area in levels:
            level_path = output_dir / f"level_{name}.json"
            config = {"level": name, "drag_coefficient": cd, "area_m2": area,
                      "profiles": list(profiles), "delays_s": list(delays), "steps_m": list(LONG_STEPS_M)}
            if level_path.exists() and json.loads(level_path.read_text(encoding="utf-8"))["config"] == config:
                level = json.loads(level_path.read_text(encoding="utf-8"))
                log(f"long level {name}: reused checkpoint")
            else:
                jobs = [(p, "b", d, a) for p in profiles for d in delays for a in LONG_STEPS_M]
                rows, effective = _fly_rows(jobs, cd, area)
                rows = lc._round(rows)
                for row in rows:
                    row["long_fail"], row["long_fail_reasons"] = long_fail(row)
                level = {"config": config, "effective_drag_coefficient": effective[0][0],
                         "effective_area_m2": effective[0][1], "rows": rows}
                _write_atomic(level_path, (json.dumps(level, indent=2, sort_keys=True) + "\n").encode("utf-8"))
                log(f"long level {name}: done")
            # Drag-off rows that overlap the registered grid must equal the control rows.
            overlap = [r for r in level["rows"] if (r["profile"], r["arm"], r["delay_s"], r["step_m"]) in control_by_key]
            consistent = all(
                {k: v for k, v in r.items() if k not in ("long_fail", "long_fail_reasons")}
                == control_by_key[(r["profile"], r["arm"], r["delay_s"], r["step_m"])] for r in overlap)
            summary = {}
            for pid in profiles:
                for step in LONG_STEPS_M:
                    sel = [r for r in level["rows"] if r["profile"] == pid and r["step_m"] == step]
                    summary[f"{pid} step {step:g}"] = long_limit(sel)
            result["levels"][name] = {
                "effective_drag_coefficient": level["effective_drag_coefficient"],
                "effective_area_m2": level["effective_area_m2"],
                "overlap_rows": len(overlap),
                "overlap_equals_control": consistent if name == "off" else None,
                "limits": summary,
            }
    result["controls_pass"] = control["matches_registered"] and all(
        lv["overlap_equals_control"] is not False for lv in result["levels"].values())
    (output_dir / "long_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8")
    inputs = [Path(__file__).resolve(), Path(lc.__file__).resolve(), REGISTERED_LONG, REGISTRY_PATH,
              *(PROFILES[p] for p in profiles)]
    outputs = sorted(output_dir.glob("level_*.json")) + [control_path, output_dir / "long_summary.json"]
    manifest = {
        "command": [sys.executable, "tools/sens_a6_drag.py", *sys.argv[1:]],
        "git_commit": _git_commit(),
        "seeds": [0],
        "inputs_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in inputs},
        "outputs_sha256": {p.name: _sha256(p) for p in outputs},
        "cpu_s_this_invocation": round(time.process_time() - started, 1),
        "workers": 1,
    }
    (output_dir / "long_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--long-check", action="store_true",
                        help="run the 80 s long check with drag levels instead of the 20 s scan")
    args = parser.parse_args(argv)
    if args.long_check:
        result = run_long_check(args.output_dir, log=lambda msg: print(msg, flush=True))
        for name, level in result["levels"].items():
            for key, lim in level["limits"].items():
                print(f"long {name} {key}: limit {lim['limit_s']} s (monotone {lim['monotone']})")
        return 0 if result["controls_pass"] else 1
    result = run(args.output_dir)
    for pid, controls in result["controls"].items():
        for name, answer in controls.items():
            first = answer.get("d_crit_s", answer.get("exp021", {}).get("first_fail_delay_s"))
            print(f"control {pid} {name}: d_crit {first} s, matches registered {answer['matches_registered']}")
        print(f"control {pid} profile_drag_scan: failed-to-fly first fail "
              f"{controls['profile_drag_scan']['failed_to_fly']['first_fail_delay_s']} s")
    for pid, variants in result["variants"].items():
        for answer in variants.values():
            print(f"drag {pid} Cd {answer['effective_drag_coefficient']:g} area {answer['effective_area_m2']:g}: "
                  f"EXP-021 first fail {answer['exp021']['first_fail_delay_s']} "
                  f"(monotone {answer['exp021']['monotone']}), "
                  f"failed-to-fly first fail {answer['failed_to_fly']['first_fail_delay_s']} "
                  f"(monotone {answer['failed_to_fly']['monotone']}), "
                  f"EXP-021 stable below ground at {answer['exp021_stable_below_ground_s']}")
    return 0 if result["controls_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
