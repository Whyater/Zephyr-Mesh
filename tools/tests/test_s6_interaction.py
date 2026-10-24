from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools.s6_interaction import analyze_rows


def test_known_additive_and_interaction_shares_are_recovered():
    # Independent synthetic construction. Effects are chosen orthogonal with
    # known squared contributions: main delay=4, main loss=1, delay*loss=9.
    rows = []
    for arm in ("naive", "recovery"):
        for motion in ("cv", "ct"):
            for delay in ("0", "1"):
                for loss in ("0", "1"):
                    for noise in ("0", "1"):
                        for seed in range(4):
                            x_delay = -1 if delay == "0" else 1
                            x_loss = -1 if loss == "0" else 1
                            value = 10 + 2 * x_delay + x_loss + 3 * x_delay * x_loss
                            rows.append({"slice": "main", "arm": arm, "motion": motion, "delay_s": delay,
                                         "loss_probability": loss, "noise_std_m": noise, "seed": str(seed),
                                         "p95_error_m": str(value)})
    result = analyze_rows(rows, bootstrap_samples=20, bootstrap_seed=3)
    shares = result["arms"]["naive"]["p95"]["shares"]
    assert shares["main:delay_s"] == pytest.approx(4 / 14)
    assert shares["main:loss_probability"] == pytest.approx(1 / 14)
    assert shares["interaction:delay_s*loss_probability"] == pytest.approx(9 / 14)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_residual_is_documented_as_unmodeled_higher_order_and_seed_variance():
    rows = []
    for arm in ("naive", "recovery"):
        for motion in ("cv", "ct"):
            for delay in ("0", "1"):
                for loss in ("0", "1"):
                    for noise in ("0", "1"):
                        for seed in range(2):
                            rows.append({"slice": "main", "arm": arm, "motion": motion,
                                         "delay_s": delay, "loss_probability": loss,
                                         "noise_std_m": noise, "seed": str(seed),
                                         "p95_error_m": str(1 + seed +
                                             (delay == "1") * (loss == "1") * (motion == "ct"))})
    result = analyze_rows(rows, bootstrap_samples=2, bootstrap_seed=3)
    expected = "within-cell seed variance plus all 3-way and 4-way interactions"
    assert expected in result["residual_definition"]


def test_real_slice_is_balanced_and_shares_sum_to_one():
    root = Path(__file__).resolve().parents[2]
    path = root / "sim/runs/s6-recovery/per_seed.csv"
    rows = [row for row in __import__("csv").DictReader(path.open(newline="")) if row["slice"] == "main"]
    result = analyze_rows(rows, bootstrap_samples=8, bootstrap_seed=1)
    for arm in ("naive", "recovery"):
        for metric in ("p95", "log_p95"):
            shares = result["arms"][arm][metric]["shares"]
            assert sum(shares.values()) == pytest.approx(1.0)


def test_cli_is_byte_reproducible(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = root / "sim/runs/s6-recovery/per_seed.csv"
    outputs = []
    for name in ("a.json", "b.json"):
        output = tmp_path / name
        subprocess.run([sys.executable, "tools/s6_interaction.py", "--input", str(source), "--output", str(output), "--bootstrap-samples", "16"], cwd=root, check=True)
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]
