from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from tools import horizon_mpl

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "tools" / "make_s6_figure.py"
TOKENS = ROOT / "tools" / "design" / "horizon_tokens.json"

DELAYS = (0.0, 0.05, 0.15, 0.30)
LOSSES = (0.0, 0.10, 0.30, 0.50)
NOISES = (0.0, 0.02, 0.05, 0.10)
MOTIONS = ("constant_velocity", "constant_turn")


def _arm(median: float) -> dict:
    # Synthetic shape for the test only: not simulation output.
    return {
        "n_seeds": 5,
        "p95_median_m": median,
        "p95_ci_low_m": median * 0.9,
        "p95_ci_high_m": median * 1.1,
        "label": "passes",
    }


def _synthetic_summary(schema: str = "zephyr-s6-recovery-1") -> dict:
    cells = []
    for motion in MOTIONS:
        for delay in DELAYS:
            for loss in LOSSES:
                for noise in NOISES:
                    cells.append(
                        {
                            "cell_id": f"{motion}-{delay}-{loss}-{noise}",
                            "slice": "main",
                            "motion": motion,
                            "delay_s": delay,
                            "loss_probability": loss,
                            "noise_std_m": noise,
                            "arms": {
                                "naive": _arm(delay + noise + 0.05 * loss),
                                "recovery": _arm(0.3 * noise + 0.02 * loss + 0.1 * delay),
                            },
                            "paired": {},
                        }
                    )
    return {
        "schema": schema,
        "config": {
            "failure_p95_m": 0.25,
            "target_speed_mps": 1.0,
            "turn_radius_m": 1.0,
            "observation_period_s": 0.1,
            "tracker_measurement_std_m": 0.05,
        },
        "design": {"n_seeds": 5},
        "cells": cells,
    }


def _run(summary: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(summary), "--output-dir", str(out_dir)],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_figures_are_byte_stable(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(_synthetic_summary(), indent=2, sort_keys=True) + "\n")
    first, second = tmp_path / "a", tmp_path / "b"
    assert _run(summary, first).returncode == 0
    assert _run(summary, second).returncode == 0
    for name in ("s6_recovery.png", "s6_recovery_all_noise.png"):
        data = (first / name).read_bytes()
        assert data.startswith(b"\x89PNG")
        assert data == (second / name).read_bytes()


def test_wrong_schema_exits_nonzero(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(_synthetic_summary(schema="something-else")))
    result = _run(summary, tmp_path / "out")
    assert result.returncode != 0
    assert "schema" in result.stderr
    assert not (tmp_path / "out" / "s6_recovery.png").exists()


def test_figure_script_has_no_hex_color_literals():
    text = SCRIPT.read_text(encoding="utf-8")
    assert re.search(r"#[0-9A-Fa-f]{6}", text) is None
    assert chr(0x2014) not in text


def test_palette_matches_tokens_file():
    tokens = json.loads(TOKENS.read_text(encoding="utf-8"))["colors"]["light"]
    light = horizon_mpl.palette("light")

    def flat(entry):
        return entry["value"] if isinstance(entry, dict) else entry

    assert light["accent"] == flat(tokens["accent"]) == "#1F6FD1"
    assert light["alert"] == flat(tokens["alert"]) == "#C62828"
    assert light["surface"] == tokens["surface"]["value"]
