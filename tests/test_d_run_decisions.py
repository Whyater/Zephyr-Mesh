"""Checks for the simulated decision-authority runner."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _run_pilot(path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "tools/d_run_decisions.py",
            "--pilot",
            "--processes",
            "2",
            "--output",
            str(path),
        ],
        check=True,
    )


@pytest.fixture(scope="module")
def pilot_output(tmp_path_factory):
    """Run one real pilot for tests that inspect runner outputs."""
    output = tmp_path_factory.mktemp("pilot") / "output"
    _run_pilot(output)
    return output


def test_pilot_is_byte_stable_and_figures_are_png(tmp_path, pilot_output):
    outputs = []
    outputs.append(pilot_output)
    second = tmp_path / "b"
    _run_pilot(second)
    outputs.append(second)
    names = [
        "per_seed.csv",
        "timeline.csv",
        "summary.json",
        "decisions_outage.png",
        "decisions_traffic.png",
    ]
    for name in names:
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    manifest0 = json.loads((outputs[0] / "manifest.json").read_text(encoding="utf-8"))
    manifest1 = json.loads((outputs[1] / "manifest.json").read_text(encoding="utf-8"))
    volatile = {
        "runtime_s",
        "git_revision",
        "git_tree_state",
        "platform",
        "n_processes",
        "argv",
    }
    for manifest in (manifest0, manifest1):
        for key in volatile:
            manifest.pop(key, None)
    assert manifest0 == manifest1
    for source in ("sim/estimation/cv.py", "sim/tracker.py", "tools/d_common.py"):
        assert source in manifest0["code_sha256"]
    for name in ("decisions_outage.png", "decisions_traffic.png"):
        data = (outputs[0] / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(data) > 10_000


def test_figures_change_when_data_change(tmp_path, pilot_output):
    output = pilot_output
    before = (output / "decisions_traffic.png").read_bytes()
    changed = tmp_path / "changed"
    shutil.copytree(output, changed)
    csv_path = changed / "per_seed.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    byte_fields = [
        field
        for field in fields
        if "bytes_per_second" in field and field != "bytes_per_second"
    ]
    field = byte_fields[0] if byte_fields else "bytes_per_second"
    for row in rows:
        row[field] = repr(float(row[field]) + 123.0)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    from tools.d_make_decisions_figure import make_figures

    make_figures(changed)
    assert (changed / "decisions_traffic.png").read_bytes() != before


def test_ground_outage_row_matches_independent_simulation(pilot_output):
    output = pilot_output
    with (output / "per_seed.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    from tools.d_common import seeds_for
    from sim.coop.simulator import CoopLinkConfig, CoopSwarmConfig, run

    seed = seeds_for("decisions", 50)[0]
    row = next(
        item
        for item in rows
        if item["mode"] == "ground"
        and float(item["loss"]) == 0.3
        and item["outage"] == "10_15"
        and int(item["seed"]) == seed
    )
    config = CoopSwarmConfig(
        n_agents=4,
        seed=seed,
        duration_s=20.0,
        control_dt_s=0.02,
        sensor_rate_hz=10.0,
        noise_std_m=0.05,
        estimator="cv",
        fusion="central",
        decision="ground",
        target="ct",
        target_params=(("radius", 3.0), ("speed", 0.8)),
        target_altitude_m=1.5,
        standoff_radius_m=1.5,
        slot_altitude_m=1.5,
        keep_out_radius_m=0.75,
        link=CoopLinkConfig(
            delay_s=0.05,
            loss_probability=0.3,
            loss_model="independent",
            share_rate_hz=10.0,
        ),
        heartbeat_timeout_s=0.5,
        log_every=1,
        gs_outage=((10.0, 15.0),),
        dropouts=(),
        ci_mode="fast",
    )
    raw = run(config).as_dict()
    steps = raw["steps"]
    scored = [step for step in steps if 2.0 <= float(step["time_s"]) <= 20.0]
    decision_errors = [
        float(step["gs_fuser_error_m"])
        for step in scored
        if step.get("gs_fuser_error_m") is not None
    ]
    standoff = [
        float(agent["standoff_error_m"])
        for step in steps
        if 10.0 <= float(step["time_s"]) < 15.0
        for agent in step["agents"]
        if agent.get("active") and agent.get("standoff_error_m") is not None
    ]
    active_steps = sum(
        1 for step in steps for agent in step["agents"] if agent.get("active")
    )
    timeout_steps = sum(
        "command_timeout" in agent.get("flags", [])
        for step in steps
        for agent in step["agents"]
        if agent.get("active")
    )
    traffic = raw["summary"]["traffic"]
    total_bytes_per_second = sum(
        float(bucket["bytes_per_second"]) for bucket in traffic.values()
    )
    assert np.isclose(
        float(row["decision_estimate_error_p95_m"]),
        np.percentile(decision_errors, 95),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(row["standoff_error_10_15_m"]),
        np.median(standoff),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(row["command_timeout_fraction"]),
        timeout_steps / max(1, active_steps),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(row["bytes_per_second"]),
        total_bytes_per_second,
        rtol=0.0,
        atol=1e-9,
    )
