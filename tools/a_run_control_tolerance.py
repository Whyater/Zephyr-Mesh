"""Generate the deterministic EXP-021 control delay and loss sweep."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sim.physics.run_manifest import write_manifest
from sim.s6_recovery import median_interval, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "tools" / "a_control_tolerance_config.json"
CONTROL_DT_S = 0.01


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _profile_id(path: str) -> str:
    profile_path = ROOT / path if not Path(path).is_absolute() else Path(path)
    try:
        profile_id = json.loads(profile_path.read_text(encoding="utf-8")).get("profile_id")
    except (OSError, json.JSONDecodeError):
        profile_id = None
    return str(profile_id or Path(path).stem.replace("_class", "-class"))


def build_cells(pilot: bool = False) -> list[dict]:
    config = load_config()
    jobs = config["pilot"]["jobs"] if pilot else config["jobs"]
    cells = []
    for job in jobs:
        cell = dict(job)
        cell["profile_id"] = _profile_id(cell["profile"])
        cell["delay_s"] = float(cell["delay_s"])
        cell["loss_probability"] = float(cell["loss_probability"])
        if pilot and config.get("pilot", {}).get("duration_s") is not None:
            cell["duration_s"] = float(config["pilot"]["duration_s"])
        cells.append(cell)
    keys = ("profile_id", "reference", "arm", "delay_s", "loss_probability", "profile")
    return sorted(cells, key=lambda c: tuple(c[k] for k in keys))


def _cell_id(cell: dict) -> str:
    text = json.dumps(cell, sort_keys=True, separators=(",", ":"))
    return f"{cell['profile_id']}_{cell['reference']}_{cell['arm']}_{hashlib.sha256(text.encode()).hexdigest()[:16]}"


def _effective_delay_s(delay_s: float) -> float:
    """Transport delay realized on the 10 ms control grid."""
    return math.ceil(float(delay_s) / CONTROL_DT_S - 1e-9) * CONTROL_DT_S


def _run_one(args: tuple[dict, int]) -> tuple[str, dict]:
    """Run one cell and seed. Exceptions intentionally propagate to the pool."""
    cell, seed = args
    from sim.physics.control_tolerance import run_episode

    config = load_config()
    result = run_episode(
        cell["profile"], cell["arm"], cell["reference"],
        cell["delay_s"], cell["loss_probability"], int(seed),
        duration_s=cell["duration_s"] if "duration_s" in cell else None,
        scoring_windows=config.get("scoring_windows"),
    )
    row = dict(cell)
    protected = {"profile", "arm", "reference", "delay_s", "loss_p", "loss_probability", "seed"}
    row.update({k: v for k, v in result.items() if k not in protected and (np.isscalar(v) or v is None)})
    row["seed"] = int(seed)
    row["rms_error_m"] = float(row["rms_error_m"])
    row["unstable"] = bool(row["unstable"])
    row["effective_delay_s"] = _effective_delay_s(cell["delay_s"])
    return _cell_id(cell), row


def _cell_rows(cell: dict, seeds: list[int]) -> list[tuple[dict, int]]:
    # A zero-loss cell is a single reference run. Loss cells use all common seeds.
    run_seeds = [0] if cell["loss_probability"] == 0.0 else list(seeds)
    return [(cell, int(seed)) for seed in run_seeds]


def _median_interval(values: np.ndarray) -> tuple[float, float, float]:
    return median_interval(values)


def _summary(rows: list[dict]) -> list[dict]:
    keys = ("profile_id", "reference", "arm", "delay_s", "loss_probability")
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key in sorted(groups):
        group = groups[key]
        values = np.asarray([r["rms_error_m"] for r in group], dtype=float)
        values = values[np.isfinite(values)]
        failures = sum(bool(r["unstable"]) for r in group)
        if values.size:
            median, low, high = _median_interval(values)
            med, interval = float(median), [float(low), float(high)]
        else:
            med, interval = None, [None, None]
        fail_low, fail_high = wilson_interval(failures, len(group))
        result.append(
            dict(
                zip(keys, key), median_rms_error_m=med, median_interval_m=interval,
                unstable_rate=float(failures / len(group)),
                unstable_interval=[float(fail_low), float(fail_high)],
                unstable_count=int(failures), n=len(group),
            )
        )
    return result


def _dcrit(rows: list[dict], pilot: bool) -> dict:
    from sim.hardware import load_hardware_profile
    from sim.physics.control_tolerance import find_d_crit
    from tools.a_dcrit_hand import estimate

    config = load_config()
    dcrit_duration = float(config.get("pilot", {}).get("duration_s", 6.0)) if pilot else 20.0
    answer = {
        "schema": "zephyr-mesh-exp-021-d-crit-1",
        "pilot": bool(pilot),
        "episode_duration_s": dcrit_duration,
        "effective_delay_grid_s": CONTROL_DT_S,
        "effective_delay_method": "10 ms-rounded transport delay for all arms",
        "profiles": {},
    }
    for profile in sorted({r["profile"] for r in rows}):
        pid = _profile_id(profile)
        loaded = load_hardware_profile(profile)
        hand = estimate(tau=loaded.motor_time_constant_s)
        entry = {
            "hand_xy": hand["xy"],
            "hand_xy_sim_expected_s": hand["xy"].get("d_crit_sim_expected_s"),
            "arm_b": {}, "slice_c": {"hand_attitude_whole_state": hand["attitude_whole_state"]}, "arm_a": {},
        }
        dcrit_cfg = config.get("d_crit", {})
        b_cfg = dcrit_cfg.get("arm_b", {"resolution_s": CONTROL_DT_S, "bracket": [0.0, 0.5]})
        c_cfg = dcrit_cfg.get("slice_c", {"resolution_s": CONTROL_DT_S, "bracket": [0.0, 0.08]})
        run_kwargs = {
            "duration_s": dcrit_duration,
            "scoring_windows": config.get("scoring_windows"),
        }
        entry["arm_b"]["p0"] = find_d_crit(
            profile, arm="b", resolution_s=CONTROL_DT_S,
            bracket=tuple(b_cfg["bracket"]), **run_kwargs,
        )
        entry["arm_b"]["p0"]["episode_duration_s"] = dcrit_duration
        entry["slice_c"].update(find_d_crit(
            profile, arm="c", resolution_s=CONTROL_DT_S,
            bracket=tuple(c_cfg["bracket"]), **run_kwargs,
        ))
        entry["slice_c"]["episode_duration_s"] = dcrit_duration

        # For p > 0, d_crit is the smallest grid delay with at least half of
        # the common-random-number seeds unstable. Keep any-first separately.
        for loss in sorted(float(x) for x in config.get("loss_probabilities", [0.1, 0.3, 0.5])):
            if loss <= 0.0:
                continue
            candidates = [
                r for r in rows
                if r["profile"] == profile and r["reference"] == "step"
                and r["arm"] == "b" and float(r["loss_probability"]) == loss
            ]
            first_any = None
            dcrit_delay = None
            dcrit_interval = [None, None]
            for delay in sorted({float(r["delay_s"]) for r in candidates}):
                at_delay = [r for r in candidates if float(r["delay_s"]) == delay]
                bad = sum(bool(r["unstable"]) for r in at_delay)
                if bad and first_any is None:
                    first_any = delay
                if dcrit_delay is None and bad * 2 >= len(at_delay):
                    dcrit_delay = delay
                    lo, hi = wilson_interval(bad, len(at_delay))
                    dcrit_interval = [float(lo), float(hi)]
            loss_key = str(loss)
            entry["arm_b"].setdefault("p_gt_0", {})[loss_key] = {
                "smallest_grid_delay_half_unstable_s": dcrit_delay,
                "first_delay_any_unstable_s": first_any,
                "unstable_rate_interval": dcrit_interval,
            }

        sim = entry["arm_b"].get("p0", {}).get("d_crit_s")
        expected = hand["xy"].get("d_crit_sim_expected_s")
        entry["percent_difference_vs_hand_sim_expected"] = (
            None if sim is None or not expected else float(100 * (sim - expected) / expected)
        )
        ramp = [
            r for r in rows if r["profile"] == profile and r["arm"] == "a"
            and r["reference"] == "ramp" and float(r["loss_probability"]) == 0.0
        ]
        if len(ramp) >= 2:
            x = np.asarray([r["delay_s"] for r in ramp]); y = np.asarray([r["rms_error_m"] for r in ramp])
            slope, offset = np.polyfit(x, y, 1)
            entry["arm_a"].update(steady_ramp_rms_slope_m_per_s=float(slope), offset_m=float(offset), hand_slope_m_per_s=0.5)
        unstable_a = [r for r in rows if r["profile"] == profile and r["arm"] == "a" and r["unstable"]]
        entry["arm_a"].update(arm_a_any_unstable=bool(unstable_a), unstable_count=len(unstable_a))
        answer["profiles"][pid] = entry
    return answer


def _write_plot(output: Path, summaries: list[dict], dcrit: dict) -> None:
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="zephyr-mplconfig-")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools import horizon_mpl

    colors = horizon_mpl.apply_style("light")
    profile_styles = {profile: style for profile, style in zip(
        sorted({r["profile_id"] for r in summaries}), ("-", "--")
    )}
    # Four non-status tokens that are easy to tell apart, plus a marker per loss
    # level. Red is kept for the d_crit lines only.
    loss_colors = {0.0: colors["text"], 0.1: colors["accent"], 0.3: colors["keepOut"], 0.5: colors["lineStrong"]}
    loss_markers = {0.0: "o", 0.1: "s", 0.3: "^", 0.5: "D"}
    hand_color = colors["textSecondary"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    profiles = sorted(profile_styles)

    left = axes[0]
    for profile in profiles:
        style = profile_styles[profile]
        for loss in (0.0, 0.3):
            points = sorted(
                [r for r in summaries if r["profile_id"] == profile and r["reference"] == "ramp"
                 and r["arm"] == "a" and float(r["loss_probability"]) == loss],
                key=lambda r: r["delay_s"],
            )
            if not points:
                continue
            x = np.asarray([r["delay_s"] * 1000 for r in points], dtype=float)
            y = np.asarray([r["median_rms_error_m"] for r in points], dtype=float)
            color = loss_colors[loss]
            line = left.plot(x, y, linestyle=style, color=color, marker=loss_markers[loss], markersize=3,
                             label=f"{profile} p={loss:g}")[0]
            low = np.asarray([r["median_interval_m"][0] for r in points], dtype=float)
            high = np.asarray([r["median_interval_m"][1] for r in points], dtype=float)
            if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                left.fill_between(x, low, high, color=line.get_color(), alpha=0.14)
    hand_x = np.linspace(0.0, 500.0, 100)
    left.plot(hand_x, 0.5 * (0.75 + hand_x / 1000.0), ":", color=hand_color,
              label="hand: 0.5(0.75+d)")
    left.set_title("arm a ramp simulated")
    left.set_xlabel("delay (ms)")
    left.set_ylabel("median RMS error (m)")
    left.legend(fontsize=7)

    right = axes[1]
    for profile in profiles:
        style = profile_styles[profile]
        for loss in (0.0, 0.1, 0.3, 0.5):
            points = sorted(
                [r for r in summaries if r["profile_id"] == profile and r["reference"] == "step"
                 and r["arm"] == "b" and float(r["loss_probability"]) == loss],
                key=lambda r: r["delay_s"],
            )
            if not points:
                continue
            x = np.asarray([r["delay_s"] * 1000 for r in points], dtype=float)
            y = np.asarray([r["unstable_rate"] for r in points], dtype=float)
            right.plot(x, y, linestyle=style, color=loss_colors[loss], marker=loss_markers[loss], markersize=3,
                       label=f"{profile} p={loss:g}")
        entry = dcrit.get("profiles", {}).get(profile, {})
        marker = entry.get("arm_b", {}).get("p0", {}).get("d_crit_s")
        hand = entry.get("hand_xy_sim_expected_s")
        if marker is not None:
            right.axvline(float(marker) * 1000, color=colors.get("alert", "#C62828"), linewidth=1.2,
                          label=f"{profile} sim d_crit")
        if hand is not None:
            right.axvline(float(hand) * 1000, color=hand_color, linestyle=":",
                          linewidth=1.0, label=f"{profile} hand")
    right.set_title("arm b step simulated")
    right.set_xlabel("delay (ms)")
    right.set_ylabel("unstable rate")
    right.set_ylim(0.0, 1.0)
    right.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "a6_control_tolerance.png", dpi=150, metadata={"Software": None})
    plt.close(fig)


def _measure_full_run_cost(config: dict) -> float:
    """Measure two full-length episodes for each arm for the runtime estimate."""
    from sim.physics.control_tolerance import run_episode

    profiles = list(config.get("profiles", []))
    if not profiles:
        return 0.0
    cases = []
    for arm in ("a", "b", "c"):
        reference = "ramp" if arm == "a" else "step"
        for seed in (0, 1):
            cases.append((profiles[0], arm, reference, seed))
    started = time.perf_counter()
    for profile, arm, reference, seed in cases:
        run_episode(
            profile, arm, reference, 0.0, 0.0, seed,
            scoring_windows=config.get("scoring_windows"),
        )
    return (time.perf_counter() - started) / len(cases)


def run_sweep(output: str | Path, pilot: bool = False, workers: int | None = None, resume: bool = False) -> dict:
    started = time.perf_counter()
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"output directory is non-empty: {output}; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    cells_dir = output / "cells"
    cells_dir.mkdir(exist_ok=True)
    workers = max(1, (os.cpu_count() or 1) // 3) if workers is None else max(1, int(workers))
    config = load_config()
    cells = build_cells(pilot)
    seeds = list(config.get("pilot", {}).get("seeds", [0])) if pilot else list(config["seeds"])
    rows: list[dict] = []
    pending: list[tuple[dict, int]] = []
    pending_cells: dict[str, dict] = {}
    expected_counts: dict[str, int] = {}
    for cell in cells:
        path = cells_dir / f"{_cell_id(cell)}.json"
        if resume and path.is_file():
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
            continue
        pending.extend(_cell_rows(cell, seeds))
        pending_cells[_cell_id(cell)] = cell
        expected_counts[_cell_id(cell)] = len(_cell_rows(cell, seeds))

    completed: dict[str, list[dict]] = {key: [] for key in pending_cells}
    if pending:
        # One process pool covers every cell. map preserves input order, and
        # the final sort makes output independent of scheduling and worker count.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for key, row in pool.map(_run_one, pending):
                rows.append(row)
                completed[key].append(row)
                if len(completed[key]) == expected_counts[key]:
                    cell_rows = sorted(completed[key], key=lambda r: (r["seed"], r["profile_id"]))
                    (cells_dir / f"{key}.json").write_text(
                        json.dumps({"cell": pending_cells[key], "rows": cell_rows}, sort_keys=True, indent=2) + "\n", encoding="utf-8"
                    )
    for key, cell in pending_cells.items():
        if (cells_dir / f"{key}.json").is_file():
            continue
        cell_rows = sorted(completed[key], key=lambda r: (r["seed"], r["profile_id"]))
        (cells_dir / f"{key}.json").write_text(
            json.dumps({"cell": cell, "rows": cell_rows}, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    keys = ("profile_id", "reference", "arm", "delay_s", "loss_probability", "seed")
    rows.sort(key=lambda r: tuple(r[k] for k in keys))
    fields = ["profile_id", "profile", "reference", "arm", "delay_s", "effective_delay_s", "loss_probability", "seed", "rms_error_m", "max_error_m", "whole_run_max_error_m", "whole_run_max_tilt_deg", "whole_run_min_z_m", "unstable", "unstable_reason", "unstable_time_s"]
    with (output / "per_run.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            encoded = {}
            for field in fields:
                value = row.get(field)
                encoded[field] = "" if value is None else (f"{value:.9g}" if isinstance(value, (float, np.floating)) else value)
            writer.writerow(encoded)
    summaries = _summary(rows)
    (output / "summary.json").write_text(
        json.dumps({"schema": "zephyr-a6-control-tolerance-summary-1", "pilot": pilot,
                    "median_interval_method": "order-statistic (binomial) interval, sim.s6_recovery.median_interval",
                    "rows": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dcrit = _dcrit(rows, pilot)
    schema_path = ROOT / "sim" / "physics" / "d_crit_schema.json"
    import jsonschema
    jsonschema.validate(dcrit, json.loads(schema_path.read_text(encoding="utf-8")))
    (output / "d_crit.json").write_text(json.dumps(dcrit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_plot(output, summaries, dcrit)
    elapsed = time.perf_counter() - started
    manifest = write_manifest(
        output / "manifest.json", schema="zephyr-a6-control-tolerance-manifest-1", config=config,
        seeds=seeds, limits=["simulated only", "literature-derived profiles", "linear hover hand model",
                        "d_crit_s uses a 20 s growth rule that can miss slow growth; runs/a6-long-check (80 s) shows arm b decaying at 0.33 s and growing at 0.34 s for both profiles.",
                        "esp-fly-class slice c at 0.03 s passes the 20 s rule but falls 3.2 m with a 68 degree tilt; treat the whole-state delay limit as 20 to 30 ms for both profiles."],
        outputs={"per_run.csv": output / "per_run.csv", "summary.json": output / "summary.json",
                 "d_crit.json": output / "d_crit.json", "a6_control_tolerance.png": output / "a6_control_tolerance.png"},
        output_dir=output, runtime_s=elapsed, n_processes=workers,
    )
    full_runs = sum(1 if float(c["loss_probability"]) == 0.0 else len(config["seeds"]) for c in build_cells(False))
    if pilot:
        measured_cost = _measure_full_run_cost(config)
        estimate = measured_cost * full_runs / workers
        print(f"pilot runtime: {elapsed:.3f} s")
        print(f"measured full-length per-run cost: {measured_cost:.6f} s")
        print(f"full-grid run count: {full_runs}")
        print(f"full-grid runtime estimate: {estimate:.3f} s")
    else:
        print(f"run count: {len(rows)}")
    return {"rows": len(rows), "runtime_s": elapsed, "manifest": manifest, "full_run_count": full_runs}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_sweep(args.output_dir, pilot=args.pilot, workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
