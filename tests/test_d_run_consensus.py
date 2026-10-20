from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sim.coop.consensus import complete, laplacian, ring, run_consensus, spectral_rate, steps_to_agreement
from tools import d_run_consensus


def test_consensus_math_independent_of_runner():
    adjacency = ring(4)
    initial = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    result = run_consensus(initial, adjacency, 0.5, 50)
    assert spectral_rate(laplacian(adjacency), 0.5) == pytest.approx(1.0, abs=1e-12)
    assert np.isclose(result.disagreement[-1], result.disagreement[0])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pilot_is_byte_stable_and_rows_match_simulation(tmp_path: Path):
    one, two = tmp_path / "one", tmp_path / "two"
    d_run_consensus.main(["--output", str(one), "--pilot", "--processes", "1"])
    d_run_consensus.main(["--output", str(two), "--pilot", "--processes", "1"])
    m1 = json.loads((one / "manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((two / "manifest.json").read_text(encoding="utf-8"))
    volatile = {"runtime_s", "git_revision", "git_tree_state", "platform", "n_processes", "argv"}
    for key in volatile:
        m1.pop(key, None)
        m2.pop(key, None)
    assert m1 == m2
    for name in m1["outputs"]:
        assert (one / name).read_bytes() == (two / name).read_bytes()

    row = next(r for r in _read_csv(one / "loss_delay.csv") if r["graph"] == "ring" and r["p"] == "0.5" and r["delay_steps"] == "2" and r["symmetric_loss"].lower() == "false")
    seed = int(row["seed"])
    gi = d_run_consensus.GRAPH_NAMES.index("ring")
    initial = np.random.default_rng(np.random.SeedSequence([seed, gi, 31])).uniform(-1, 1, size=(6, 3))
    result = run_consensus(initial, ring(6), 1 / 3, 400, loss_probability=float(row["p"]),
                           delay_steps=int(row["delay_steps"]), symmetric_loss=row["symmetric_loss"].lower() == "true", seed=seed)
    assert int(row["steps_to_agreement"]) == steps_to_agreement(result, 0.01)
    spread = float(np.max(np.ptp(initial, axis=0)))
    drift = float(np.linalg.norm(result.average[-1] - result.average[0]) / max(spread, 1e-15))
    assert float(row["final_average_drift"]) == pytest.approx(drift)
    assert float(row["messages_lost_fraction"]) == pytest.approx(result.messages_lost / result.messages_sent)

    acting = next(r for r in _read_csv(one / "acting.csv") if r["graph"] == "complete" and r["p"] == "0.5" and r["delay_steps"] == "1" and r["updates"] == "2")
    seed = int(acting["seed"])
    truth = np.asarray([0.0, 0.0, 1.5])
    streams = np.random.SeedSequence([seed, 2, 41]).spawn(4)
    initial = np.asarray([truth + np.random.default_rng(stream).normal(0.0, 0.05, size=3)
                          for stream in streams])
    result = run_consensus(initial, complete(4), 1 / 4, int(acting["updates"]), loss_probability=float(acting["p"]),
                           delay_steps=int(acting["delay_steps"]), symmetric_loss=True, seed=seed + 991)
    states = result.states[-1]
    largest = float(np.max(np.linalg.norm(states[:, None, :] - states[None, :, :], axis=2)))
    assert float(acting["largest_pairwise_distance_m"]) == pytest.approx(largest)
    assert (acting["still_disagreeing"].lower() == "true") == (largest > 0.10)

    for name in ("consensus_loss.png", "consensus_acting.png"):
        data = (one / name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 10_000
    assert (one / "consensus_loss.png").read_bytes() != (one / "consensus_acting.png").read_bytes()


def test_counterexample_starts_disagreeing_and_never_shrinks(tmp_path: Path):
    out = tmp_path / "pilot"
    d_run_consensus.main(["--output", str(out), "--pilot", "--processes", "1"])
    data = json.loads((out / "counterexample.json").read_text(encoding="utf-8"))
    assert data["initial_disagreement"] == pytest.approx(2.0 * 3**0.5, abs=1e-12)
    assert data["disagreement_after_50"] == pytest.approx(data["initial_disagreement"], abs=1e-12)
    assert abs(data["rho"] - 1.0) <= 1e-12
    spectral = json.loads((out / "spectral_check.json").read_text(encoding="utf-8"))
    assert all(r["rate_check"] in ("ok", "not applicable (rho = 0)") for r in spectral)


def test_manifest_records_real_argv_and_config(tmp_path: Path):
    out = tmp_path / "pilot"
    d_run_consensus.main(["--output", str(out), "--pilot", "--processes", "1"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["argv"][-1] == "1"
    assert manifest["replay_command"].startswith("PYTHONPATH=. ")
    assert manifest["config"]["agreement_tolerance"] == 0.01
    assert {"graphs", "loss_probabilities", "delay_steps", "consensus_steps", "part_c", "part_c_note"} <= set(manifest["config"])


def test_median_is_null_when_half_the_seeds_never_agree():
    rows = []
    for i in range(50):
        never = i >= 25
        rows.append({"graph": "ring", "p": 0.5, "delay_steps": 0, "symmetric_loss": True,
                     "steps_to_agreement": None if never else 100, "did_not_agree": never,
                     "final_average_drift": 0.0, "random_seed": None})
    (cell,) = d_run_consensus._summarize(rows)
    assert cell["median_steps"] is None
    rows[25]["steps_to_agreement"], rows[25]["did_not_agree"] = 100, False
    (cell,) = d_run_consensus._summarize(rows)
    assert cell["median_steps"] == 100
