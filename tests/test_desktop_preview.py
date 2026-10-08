"""Pure-model checks for the cross-platform desktop replay preview."""
from pathlib import Path

import pytest

from desktop.replay import ReplayFormatError, ReplayModel


ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs/s7-swarm/run.json"


def test_replay_model_reads_canonical_s7_fixture():
    model = ReplayModel.from_path(RUN)
    assert model.schema == "zephyr-s7-swarm-run-1"
    assert model.scenario_id == "s7-ring-50-seed-17"
    assert model.frame_count == 6
    assert len(model.frame.agents) == 50
    assert model.frame.active_count == 50
    assert model.command_authority == "replay only"
    assert model.failsafe.startswith("synthetic constraints")


def test_replay_cursor_clamps_and_reset_is_deterministic():
    model = ReplayModel.from_path(RUN)
    first = model.frame
    assert model.step(-10) == first
    assert model.step(100).step_index == 5
    assert model.progress == 1.0
    assert model.reset() == first
    assert model.index == 0
    assert model.summary()["frame"] == "1/6"


def test_replay_model_rejects_missing_steps(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema":"zephyr-s7-swarm-run-1","scenario_id":"bad","seed":1,"dt_s":0.05,"status":"synthetic","command_authority":"replay only","failsafe":"synthetic","evidence_boundary":"fixture"}', encoding="utf-8")
    with pytest.raises(ReplayFormatError, match="steps"):
        ReplayModel.from_path(path)


def test_replay_preserves_optional_telemetry_without_inventing_defaults():
    model = ReplayModel.from_path(RUN)
    agent = model.frame.agents[0]
    assert agent.battery_pct is None
    assert agent.link_delay_ms is None


def test_windows_preview_constructs_when_tk_display_is_available():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no desktop display in this test environment")
    root.withdraw()
    from desktop.windows_preview import WindowsReplayApp
    app = WindowsReplayApp(root, ReplayModel.from_path(RUN))
    assert app.selected_id == "drone-001"
    root.destroy()
