"""Checks for the long-run delay limit tool."""
import pytest

from tools import a_make_dcrit_long_check as tool


def test_zero_delay_small_step_decays(monkeypatch):
    monkeypatch.setattr(tool, "DURATION_S", 30.0)
    row = tool.fly(("crazyflie-class", "b", 0.0, 0.01))
    peaks = row["window_peak_error_m"]
    # The 1 cm step itself sets the first window; later windows ring down.
    assert peaks[0] == pytest.approx(0.01, rel=0.05)
    assert peaks[2] < 0.1 * peaks[0]
    assert row["growing"] is False


def test_refuses_non_empty_output_directory(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    with pytest.raises(SystemExit):
        tool.main(["--output-dir", str(tmp_path)])
