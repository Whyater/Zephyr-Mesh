"""The screen refresh rate recorded in run_meta.json (no OpenCV, no camera).

A run folder can say which refresh rate its screen had (``display_refresh_hz``).
Recording it changes no default: ``analyze`` still adds the refresh-phase term
only when ``--refresh-hz`` is passed. What the recorded rate adds is a guard:
a flag that disagrees with the recording is refused, and a forgotten flag is
warned about by name.
"""
import json

import pytest

from tools import vision_latency as cli


def _synth(run_dir, *extra):
    assert cli.main(["synth", str(run_dir), "--latency-ms", "90", "--duration-s", "12", "--seed", "4", *extra]) == 0
    return json.loads((run_dir / "run_meta.json").read_text())


def _set_meta(run_dir, **fields):
    meta = json.loads((run_dir / "run_meta.json").read_text())
    meta.update(fields)
    (run_dir / "run_meta.json").write_text(json.dumps(meta))


def test_synth_records_the_refresh_rate_only_when_it_simulates_one(tmp_path):
    assert _synth(tmp_path / "grid", "--vsync-hz", "60")["display_refresh_hz"] == 60.0
    assert "display_refresh_hz" not in _synth(tmp_path / "plain")  # default run_meta unchanged


def test_a_flag_that_disagrees_with_the_recorded_rate_is_refused(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _synth(run_dir, "--vsync-hz", "60")
    assert cli.main(["analyze", str(run_dir), "--refresh-hz", "50"]) == 2
    err = capsys.readouterr().err
    assert "50" in err and "60" in err and "run_meta.json" in err
    assert not (run_dir / "latency.json").exists()


def test_a_flag_that_matches_the_recorded_rate_is_used(tmp_path):
    run_dir = tmp_path / "run"
    _synth(run_dir, "--vsync-hz", "60")
    assert cli.main(["analyze", str(run_dir), "--display-lag-ms", "0,20", "--refresh-hz", "60"]) == 0
    result = json.loads((run_dir / "latency.json").read_text())
    # Hand check: half of one refresh step at 60 Hz is 0.5 / 60 s = 8.33 ms.
    assert result["apparent_latency_s"]["stated_error_parts"]["refresh_phase"] == pytest.approx(0.5 / 60)


def test_a_forgotten_flag_is_warned_about_but_the_default_is_kept(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _synth(run_dir, "--vsync-hz", "60")
    capsys.readouterr()
    assert cli.main(["analyze", str(run_dir), "--display-lag-ms", "0,20"]) == 0
    out = capsys.readouterr().out
    assert "--refresh-hz 60" in out and "8.3 ms" in out
    result = json.loads((run_dir / "latency.json").read_text())
    assert "refresh_phase" not in result["apparent_latency_s"]["stated_error_parts"]  # default kept
    assert "display_refresh_hz" not in result["settings"]


@pytest.mark.parametrize("bad", ["60", True, -60, 0, float("nan")])
def test_a_recorded_rate_that_is_not_a_positive_number_is_refused(tmp_path, capsys, bad):
    run_dir = tmp_path / "run"
    _synth(run_dir)
    _set_meta(run_dir, display_refresh_hz=bad)
    assert cli.main(["analyze", str(run_dir)]) == 2
    assert "display_refresh_hz" in capsys.readouterr().err
    assert not (run_dir / "latency.json").exists()
