from __future__ import annotations

import csv
import json
import subprocess
import os
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from sim.coop.allocation import slots_for_alive
from sim.coop.simulator import CoopLinkConfig, CoopSwarmConfig, CoopSwarmSimulator
from tools.d_run_dropout import run


DROP_TIME = 10.0
HOLD = 1.0
WINDOWS = ((5.0, 10.0), (10.0, 15.0))


def _test_config(mode: str, loss: float, seed: int, timeout: float) -> CoopSwarmConfig:
    """Build a direct simulator config without importing the runner's config helper."""
    return CoopSwarmConfig(
        n_agents=4,
        seed=seed,
        duration_s=25.0,
        control_dt_s=0.02,
        sensor_rate_hz=10.0,
        noise_std_m=0.05,
        estimator="cv",
        fusion="ci",
        decision="onboard",
        target="ct",
        target_params=(("radius", 3.0), ("speed", 0.8)),
        standoff_radius_m=1.5,
        slot_altitude_m=1.5,
        link=CoopLinkConfig(
            delay_s=0.05,
            loss_probability=loss,
            loss_model="independent",
            share_rate_hz=10.0,
        ),
        heartbeat_timeout_s=timeout,
        dropouts=tuple(
            (f"drone-{index:03d}", DROP_TIME, mode) for index in range(4, 5)
        ),
        switch_penalty_m=0.3,
        log_every=1,
        metric_windows=WINDOWS,
        ci_mode="fast",
    )


def _direct_measures(raw: dict, mode: str) -> dict:
    """Recompute runner fields from logged simulator data using test-owned loops."""
    steps = raw["steps"]
    survivors = tuple(f"drone-{index:03d}" for index in range(1, 4))
    dropped = {"drone-004"}
    detection = None
    for step in steps:
        if step["time_s"] < DROP_TIME:
            continue
        rows = {item["agent_id"]: item for item in step["agents"]}
        if all(dropped.isdisjoint(rows[agent]["believed_alive"]) for agent in survivors):
            detection = step["time_s"] - DROP_TIME
            break

    def new_plan_good(step: dict) -> bool:
        rows = {item["agent_id"]: item for item in step["agents"]}
        slots = slots_for_alive(
            np.asarray(step["target_truth_position_m"]), len(survivors), 1.5, 1.5, 0.0
        )
        for agent_id in survivors:
            row = rows[agent_id]
            if len(row["believed_alive"]) != len(survivors):
                return False
            slot_index = row["believed_assignment"][agent_id]
            distance = np.linalg.norm(np.asarray(row["position_m"]) - slots[slot_index])
            if distance > 0.3:
                return False
        return True

    post = [step for step in steps if step["time_s"] >= DROP_TIME]
    recovery = None
    recovery_start = None
    for index, step in enumerate(post):
        end = step["time_s"] + HOLD
        if end > post[-1]["time_s"] + 1e-12:
            continue
        end_index = max(
            candidate
            for candidate, candidate_step in enumerate(post)
            if candidate_step["time_s"] <= end + 1e-12
        )
        if post[end_index]["time_s"] >= end - 1e-12 and all(
            new_plan_good(candidate) for candidate in post[index : end_index + 1]
        ):
            recovery_start = step["time_s"]
            recovery = recovery_start - DROP_TIME
            break

    def error_stats(start: float, end: float) -> tuple[float | None, float | None]:
        values = [
            step["gs_group_error_m"]
            for step in steps
            if start <= step["time_s"] < end and step["gs_group_error_m"] is not None
        ]
        if not values:
            return None, None
        return float(np.percentile(values, 95)), float(max(values))

    before_p95, before_max = error_stats(*WINDOWS[0])
    gap_p95, gap_max = error_stats(*WINDOWS[1])
    scored = [
        step["gs_group_error_m"]
        for step in steps
        if step["time_s"] >= 2.0 and step["gs_group_error_m"] is not None
    ]
    separation = None
    separation_violations = 0
    for step in steps:
        positions = [
            np.asarray(row["position_m"])
            for row in step["agents"]
            if row["active"]
        ]
        for index, first in enumerate(positions):
            for second in positions[index + 1 :]:
                distance = float(np.linalg.norm(first - second))
                separation = distance if separation is None else min(separation, distance)
                separation_violations += int(distance < 0.35 - 1e-12)

    previous = {}
    role_changes = 0
    for step in steps:
        if not 10.0 <= step["time_s"] < 15.0:
            continue
        rows = {item["agent_id"]: item for item in step["agents"]}
        for agent_id in survivors:
            role = (len(rows[agent_id]["believed_alive"]), rows[agent_id]["slot_index"])
            role_changes += int(agent_id in previous and role != previous[agent_id])
            previous[agent_id] = role

    covered_fraction = None
    if recovery_start is not None:
        covered = [new_plan_good(step) for step in steps if step["time_s"] >= recovery_start]
        covered_fraction = sum(covered) / len(covered)
    return {
        "detection_time_s": detection,
        "recoverage_time_s": recovery,
        "recovery_censored": recovery is None,
        "gs_before_p95_m": before_p95,
        "gs_before_max_m": before_max,
        "gs_gap_p95_m": gap_p95,
        "gs_gap_max_m": gap_max,
        "gs_overall_p95_m": float(np.percentile(scored, 95)),
        "gs_overall_max_m": float(max(scored)),
        "role_changes_after_drop": role_changes,
        "min_pairwise_true_separation_m": separation,
        "separation_violations": separation_violations,
        "covered_fraction_after_recovery": covered_fraction,
        "mode": mode,
    }


def _float_or_none(value: str) -> float | None:
    return None if value == "" else float(value)


def _typed_rows(path: Path) -> list[dict]:
    """Parse the per-seed CSV without relying on a runner parser."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ints = {"n", "k", "seed", "role_changes_after_drop", "plan_disagreement_steps"}
    floats = {
        "link_loss",
        "detection_time_s",
        "recoverage_time_s",
        "gs_before_p95_m",
        "gs_before_max_m",
        "gs_gap_p95_m",
        "gs_gap_max_m",
        "gs_overall_p95_m",
        "gs_overall_max_m",
        "min_pairwise_true_separation_m",
        "hand_detection_registered_s",
        "hand_detection_brief_s",
        "covered_fraction_after_recovery",
        "survivor_gap_deg",
    }
    booleans = {"never_detected", "recoverage_censored", "seed_failed", "failed_in_gap", "violation"}
    for row in rows:
        for key in ints:
            row[key] = int(row[key])
        for key in floats:
            row[key] = _float_or_none(row[key])
        for key in booleans:
            row[key] = row[key] == "True"
    return rows


def _assert_same(actual: object, expected: object) -> None:
    if actual is None or expected is None:
        assert actual is None and expected is None
    else:
        assert math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)


def test_thresholds_and_independent_measures_for_silent_and_landed_jobs():
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "pilot"
        run(output, ["--pilot"], pilot=True, processes=2)
        rows = _typed_rows(output / "per_seed.csv")
        assert {row["mode"] for row in rows} == {"silent", "landed"}
        for mode in ("silent", "landed"):
            row = next(
                item
                for item in rows
                if item["n"] == 4
                and item["k"] == 1
                and item["mode"] == mode
                and item["link_loss"] == 0.1
            )
            direct = CoopSwarmSimulator(
                _test_config(mode, 0.1, row["seed"], 0.5)
            ).run().as_dict()
            expected = _direct_measures(direct, mode)
            for key in (
                "detection_time_s",
                "recoverage_time_s",
                "gs_before_p95_m",
                "gs_before_max_m",
                "gs_gap_p95_m",
                "gs_gap_max_m",
                "gs_overall_p95_m",
                "gs_overall_max_m",
                "role_changes_after_drop",
                "min_pairwise_true_separation_m",
                "separation_violations",
                "covered_fraction_after_recovery",
            ):
                _assert_same(row[key], expected[key])
            assert row["recoverage_censored"] == expected["recovery_censored"]


def test_pilot_is_reproducible_provenance_and_figures_change_with_data():
    with tempfile.TemporaryDirectory() as temp:
        first = Path(temp) / "first"
        second = Path(temp) / "second"
        run(first, ["--pilot"], pilot=True, processes=2)
        run(second, ["--pilot"], pilot=True, processes=2)
        manifest_a = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        manifest_b = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
        volatile = {"runtime_s", "git_revision", "git_tree_state", "platform", "n_processes", "argv"}
        for manifest in (manifest_a, manifest_b):
            for key in volatile:
                manifest.pop(key, None)
        assert manifest_a == manifest_b
        for name in manifest_a["outputs"]:
            assert (first / name).read_bytes() == (second / name).read_bytes()
        required = {
            "sim/tracker.py",
            "sim/s6_recovery.py",
            "sim/estimation/sweep.py",
            "tools/d_common.py",
            "sim/estimation/cv.py",
            "tools/d_run_dropout.py",
            "tools/d_make_dropout_figure.py",
        }
        assert required <= set(manifest_a["code_sha256"])
        for name in ("dropout_times.png", "dropout_timeline.png"):
            assert (first / name).stat().st_size > 10_000

        rows = _typed_rows(first / "per_seed.csv")
        for row in rows:
            row["timeline"] = [(5.0, 100.0), (10.0, 180.0), (25.0, 180.0)]
        original = (first / "dropout_timeline.png").read_bytes()
        from tools.d_make_dropout_figure import make_figures

        make_figures(first, rows, json.loads((first / "summary.json").read_text(encoding="utf-8")))
        assert (first / "dropout_timeline.png").read_bytes() != original

        root = Path(__file__).resolve().parents[1]
        loaded = set()
        for module in sys.modules.values():
            filename = getattr(module, "__file__", None)
            if not filename:
                continue
            try:
                relative = Path(filename).resolve().relative_to(root)
            except ValueError:
                continue
            if relative.parts and relative.parts[0] in {"sim", "tools"}:
                loaded.add(relative.as_posix())
        assert loaded <= set(manifest_a["code_sha256"])

        # A fresh process that runs one job, as a pool worker does, must load nothing the
        # manifest leaves out (lazily imported modules only appear inside workers).
        worker_script = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "import tools.d_run_dropout as runner\n"
            "runner._job({'n': 3, 'k': 1, 'mode': 'silent', 'link_loss': 0.0,\n"
            "             'seed': 12345, 'timeout_s': 0.5})\n"
            "root = Path.cwd().resolve()\n"
            "names = set()\n"
            "for module in list(sys.modules.values()):\n"
            "    filename = getattr(module, '__file__', None)\n"
            "    if not filename:\n"
            "        continue\n"
            "    try:\n"
            "        relative = Path(filename).resolve().relative_to(root)\n"
            "    except ValueError:\n"
            "        continue\n"
            "    if relative.parts and relative.parts[0] in {'sim', 'tools'}:\n"
            "        names.add(relative.as_posix())\n"
            "print(json.dumps(sorted(names)))\n"
        )
        env = dict(os.environ, PYTHONPATH=str(root))
        result = subprocess.run(
            [sys.executable, "-c", worker_script],
            cwd=root, env=env, capture_output=True, text=True, check=True,
        )
        worker_loaded = set(json.loads(result.stdout.strip().splitlines()[-1]))
        assert worker_loaded <= set(manifest_a["code_sha256"]), worker_loaded - set(manifest_a["code_sha256"])
