from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from sim.coop.fusion import covariance_intersection_fast, covariance_intersection_many, information_fusion
from sim.coop.net import LinkSpec
from sim.coop.observers import ARMS, CoopCell, run_seed
from tools.d_run_fusion import run
from tools.d_make_fusion_figure import make_figures


def _stable_manifest(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in value.get("volatile_fields", []):
        value.pop(key, None)
    value.pop("argv", None)
    return json.dumps(value, sort_keys=True).encode("utf-8")


def _float_or_none(value):
    return None if value == "" else float(value)


def _row_map(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_row_matches(row, outcome):
    for name in ("p95_error_m", "max_error_m", "rmse_m", "mean_nees",
                 "frac_windows_above_band"):
        actual = _float_or_none(row[name])
        expected_value = getattr(outcome, name)
        expected = None if expected_value is None or not np.isfinite(expected_value) else float(expected_value)
        assert actual == expected
    assert row["failed"] == ("true" if outcome.failed else "false")
    assert int(row["fused_count"]) == outcome.fused_count
    diverged = bool((outcome.extras or {}).get("diverged", False))
    assert row["diverged"] == ("true" if diverged else "false")
    expected_time = (outcome.extras or {}).get("diverged_time_s")
    assert _float_or_none(row["diverged_time_s"]) == expected_time


def _direct_cell(row):
    return CoopCell(
        target="ct", target_params=(), n_observers=int(row["n_observers"]),
        noise_std_m=float(row["noise_std_m"]),
        link=LinkSpec(mode="abstract", delay_s=0.05,
                      loss_probability=float(row["loss_probability"])),
        phase="random", ci=row["ci"],
    )


def test_pilot_reproducible_independent_rows_math_figures_and_manifest(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    run(first, pilot=True, processes=2, argv=["--pilot", "--processes", "2"])
    run(second, pilot=True, processes=2, argv=["--pilot", "--processes", "2"])
    manifest_a = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert _stable_manifest(first / "manifest.json") == _stable_manifest(second / "manifest.json")
    for name in manifest_a["outputs"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()

    rows = _row_map(first / "per_seed.csv")
    selected = {}
    for row in rows:
        selected.setdefault(row["arm"], row)
    for arm in ARMS:
        row = selected[arm]
        outcome = run_seed(_direct_cell(row), int(row["seed"]), arms=(arm,))[arm]
        _assert_row_matches(row, outcome)
    exact_row = next(row for row in rows if row["ci"] == "exact")
    exact_outcome = run_seed(_direct_cell(exact_row), int(exact_row["seed"]),
                             arms=("peer_ci",))["peer_ci"]
    _assert_row_matches(exact_row, exact_outcome)

    checks = json.loads((first / "math_checks.json").read_text(encoding="utf-8"))
    reported = next(item for item in checks["sigma2_over_n"] if item["n"] == 3)
    rng = np.random.default_rng(np.random.SeedSequence([918273, 44, 3]))
    draws = rng.normal(0.0, 0.7, (12000, 3, 3))
    independent_means = np.mean(draws, axis=1)
    measured = float(np.var(independent_means[:, 0], ddof=1))
    assert abs(measured - reported["measured_variance"]) < 0.08
    assert abs(reported["ratio"] - 1.0) < 0.04
    assert np.allclose(checks["ci_hand"]["information_covariance"], 0.5 * np.eye(3))
    assert checks["ci_consistency"]["fast"]["min_eigenvalue"] >= -1e-10
    assert checks["ci_consistency"]["exact"]["min_eigenvalue"] >= -1e-10
    assert checks["ci_consistency"]["information"]["claims_too_little_fraction"] > 0.0

    rng = np.random.default_rng(np.random.SeedSequence([918273, 44, 31]))
    fast_min = float("inf")
    exact_min = float("inf")
    info_too_little = 0
    for _ in range(12):
        a0 = rng.normal(size=(3, 3))
        b0 = rng.normal(size=(3, 3))
        a = a0 @ a0.T + 0.2 * np.eye(3)
        b = b0 @ b0.T + 0.2 * np.eye(3)
        cross = 0.35 * np.linalg.cholesky(a) @ np.linalg.cholesky(b).T
        _, fast_p, fast_w = covariance_intersection_fast([np.zeros(3)] * 2, [a, b])
        _, exact_p, exact_w = covariance_intersection_many([np.zeros(3)] * 2, [a, b])
        _, info_p = information_fusion([np.zeros(3)] * 2, [a, b])
        ia = np.linalg.solve(a, np.eye(3))
        ib = np.linalg.solve(b, np.eye(3))
        def true_cov(claimed, weights):
            ga, gb = claimed @ (weights[0] * ia), claimed @ (weights[1] * ib)
            return (ga @ a @ ga.T + gb @ b @ gb.T + ga @ cross @ gb.T +
                    gb @ cross.T @ ga.T)
        fast_min = min(fast_min, float(np.linalg.eigvalsh(fast_p - true_cov(fast_p, fast_w)).min()))
        exact_min = min(exact_min, float(np.linalg.eigvalsh(exact_p - true_cov(exact_p, exact_w)).min()))
        info_true = true_cov(info_p, [1.0, 1.0])
        info_too_little += int(float(np.linalg.eigvalsh(info_p - info_true).min()) < -1e-12)
    assert fast_min >= -1e-10
    assert exact_min >= -1e-10
    assert info_too_little > 0

    assert (first / "fusion_error.png").stat().st_size > 10_000
    assert (first / "fusion_consistency.png").stat().st_size > 10_000
    changed = tmp_path / "changed"
    shutil.copytree(first, changed)
    changed_summary = json.loads((changed / "summary.json").read_text(encoding="utf-8"))
    plotted = next(cell for cell in changed_summary["cells"]
                   if cell["target"] == "turn" and cell["loss_probability"] == 0.3
                   and cell["arm"] == "single")
    plotted["p95_error_m"]["median"] += 0.001
    plotted["mean_nees"]["median"] += 0.1
    (changed / "summary.json").write_text(
        json.dumps(changed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figures(changed)
    assert (changed / "fusion_error.png").read_bytes() != (first / "fusion_error.png").read_bytes()
    assert (changed / "fusion_consistency.png").read_bytes() != (first / "fusion_consistency.png").read_bytes()

    script = """
import json
import sys
from pathlib import Path
import tools.d_run_fusion as runner
runner._job({"n_observers": 2, "noise_std_m": 0.05, "target_label": "turn",
            "target": "ct", "target_params": (), "loss_probability": 0.0,
            "seed": 12345, "ci": "fast"})
root = Path.cwd().resolve()
paths = []
for module in sys.modules.values():
    path = getattr(module, "__file__", None)
    if path:
        try:
            rel = Path(path).resolve().relative_to(root)
        except ValueError:
            continue
        if rel.suffix == ".py":
            paths.append(rel.as_posix())
print(json.dumps(sorted(set(paths))))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env,
                               check=True, capture_output=True, text=True)
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert loaded <= set(manifest_a["code_sha256"])
