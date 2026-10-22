"""The git state in each manifest must be read before the run writes anything.

If it is read afterwards, the run's own new output folder makes a clean tree
look dirty. Each runner is stopped at its sweep call; by then the provenance
must already have been captured while the output folder did not exist.
"""

import importlib

import pytest


class _Stop(Exception):
    pass


def _entry(name, out):
    mod = importlib.import_module(f"tools.run_s6_{name}")
    calls = {
        "retro": lambda: mod.run_sweep(out, pilot=True, seeds_count=1, processes=1, no_checkpoint=True, argv=[]),
        "clocks": lambda: mod.run(out, pilot=True, seeds_count=1, processes=1, no_checkpoint=True, argv=[]),
        "turn": lambda: mod.run(out, pilot=True, seeds=1, processes=1, checkpoint_dir=None, no_checkpoint=True, argv=[]),
        "sending": lambda: mod.run_sweep(out, pilot=True, seeds_count=1, processes=1, no_checkpoint=True, argv=[]),
        "tuning": lambda: mod.main(["--output", str(out), "--pilot", "--seeds", "1", "--processes", "1", "--no-checkpoint"]),
    }
    return mod, calls[name]


@pytest.mark.parametrize("name", ["retro", "clocks", "turn", "sending", "tuning"])
def test_provenance_is_captured_before_outputs_exist(name, tmp_path, monkeypatch):
    out = tmp_path / "new-run"
    mod, call = _entry(name, out)
    seen = []

    def fake_provenance():
        seen.append(out.exists())
        return {"git_revision": None, "git_tree_state": "unknown"}

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(mod, "capture_provenance", fake_provenance)
    monkeypatch.setattr(mod, "run_jobs", stop)
    with pytest.raises(_Stop):
        call()
    assert seen == [False]
