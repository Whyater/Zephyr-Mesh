import json
from pathlib import Path

import numpy as np

from sim.estimation.base import ArmSpec, RECOVERY_V1
from sim.estimation.engine import CellSpec, deliver_cell, run_arm, run_cell
from sim.estimation.sweep import compare_outputs
from sim.s6_recovery import seed_list
from tools.run_s6_clocks import build_cells, job_cell, run


def test_grid_counts_and_unique_keys():
    cells = build_cells()
    assert len(cells) == 348
    assert sum(c["slice"] == "U" for c in cells) == 132
    assert sum(c["slice"] == "N" for c in cells) == 216
    keys = [CellSpec(**{k: v for k, v in c.items() if k not in {"slice", "sync_loss_per_exchange"}}).key() for c in cells]
    assert len(keys) == len(set(keys))


def test_small_run_is_byte_stable(tmp_path):
    cells = build_cells(True)[:2]
    seeds = seed_list(2)
    assert json.dumps(job_cell({"cell": cells[0], "seeds": seeds}))
    a, b = tmp_path / "a", tmp_path / "b"
    run(a, seeds_count=2, processes=1, no_checkpoint=True, cells_override=cells)
    run(b, seeds_count=2, processes=1, no_checkpoint=True, cells_override=cells)
    assert all(compare_outputs(a, b, ["summary.json", "cells.csv", "per_seed.csv.gz"]).values())


def test_zero_offset_matches_perfect_and_naive_offset_invariant():
    base = next(c for c in build_cells(True) if c["slice"] == "U" and c["target"] == "cv" and dict(c["clock"])["offset_s"] == 0.0)
    perfect = dict(base); perfect["clock"] = ()
    seeds = seed_list(2)
    base_result = job_cell({"cell": base, "seeds": seeds})
    perfect_result = job_cell({"cell": perfect, "seeds": seeds})
    base_rows = [{k: v for k, v in row.items() if k != "cell_key"} for row in base_result["per_seed"]]
    perfect_rows = [{k: v for k, v in row.items() if k != "cell_key"} for row in perfect_result["per_seed"]]
    assert base_rows == perfect_rows
    a = base_result
    shifted = dict(base); shifted["clock"] = tuple((k, (v + 0.05 if k == "offset_s" else v)) for k, v in base["clock"])
    b = job_cell({"cell": shifted, "seeds": seed_list(2)})
    pa = [x["p95_error_m"] for x in a["per_seed"] if x["arm"] == "naive"]
    pb = [x["p95_error_m"] for x in b["per_seed"] if x["arm"] == "naive"]
    assert np.allclose(pa, pb)


def test_ntp_asymmetry_sign():
    """With a = d_up - d_down, positive a makes the corrected stamp early by -a/2."""
    c = next(c for c in build_cells() if c["slice"] == "N" and dict(c["clock"])["sync_asymmetry_s"] == 0.04 and dict(c["clock"])["drift_ppm"] == 0.0 and dict(c["clock"])["sync_loss_probability"] == 0.0)
    assert dict(c["clock"])["sync_delay_s"] == 0.025
    r = job_cell({"cell": c, "seeds": seed_list(2)})
    assert abs(r["stamp_error_stats"]["post_sync_mean"] + 0.02) < 1e-6


def test_retro_matches_recovery_on_zero_jitter_u_cell():
    c = next(c for c in build_cells() if c["slice"] == "U" and c["target"] == "cv" and c["delay_s"] == 0.1 and c["loss_probability"] == 0.0)
    cell = CellSpec(**{k: v for k, v in c.items() if k not in {"slice", "sync_loss_per_exchange"}})
    seeds = seed_list(2)
    retro = ArmSpec("retro", stamping="capture", late_policy="retrodict")
    result = run_cell(cell, [RECOVERY_V1, retro], seeds)
    for seed in seeds:
        rows = [row for row in result["per_seed"] if row["seed"] == seed]
        recovery = next(row for row in rows if row["arm"] == "recovery")
        retro_row = next(row for row in rows if row["arm"] == "retro")
        assert {k: v for k, v in recovery.items() if k != "arm"} == {k: v for k, v in retro_row.items() if k != "arm"}
