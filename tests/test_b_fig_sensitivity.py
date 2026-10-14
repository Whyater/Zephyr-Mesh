"""Build the sensitivity figures from the pilot output and check the files."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "make_s6_sensitivity_figure.py"
PILOT = Path("/tmp/runb-pilots/sensitivity")
NAMES = ("s6_sensitivity_morris.png", "s6_sensitivity_oat.png")

pytestmark = pytest.mark.skipif(
    not PILOT.is_dir(), reason=f"pilot output {PILOT} is missing (run the sensitivity pilot first)"
)


def _build(run_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir), "--output-dir", str(out_dir)],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )


def _copy_pilot(dest: Path) -> Path:
    shutil.copytree(PILOT, dest)
    return dest


def test_figures_are_built_deterministically_and_hashed(tmp_path):
    run_dir = _copy_pilot(tmp_path / "run")
    first = _build(run_dir, run_dir)
    assert first.returncode == 0, first.stderr
    hashes = {}
    for name in NAMES:
        path = run_dir / name
        assert path.exists(), f"{name} was not written"
        assert path.stat().st_size > 20_000, f"{name} is suspiciously small"
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in NAMES:
        assert manifest["outputs"][name] == hashes[name]
    # The other manifest entries survive and the file is sorted, indented, newline-ended.
    text = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text == json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    original = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    for key, value in original["outputs"].items():
        assert manifest["outputs"][key] == value

    second_dir = tmp_path / "second"
    second = _build(run_dir, second_dir)
    assert second.returncode == 0, second.stderr
    for name in NAMES:
        assert (second_dir / name).read_bytes() == (run_dir / name).read_bytes(), (
            f"{name} differs between two builds"
        )


def test_bad_input_gives_a_clear_error(tmp_path):
    run_dir = _copy_pilot(tmp_path / "run")
    (run_dir / "morris.json").write_text("{not json", encoding="utf-8")
    result = _build(run_dir, tmp_path / "out")
    assert result.returncode == 2
    assert "morris.json" in result.stderr


# ---------------------------------------------------------------- Morris labels

FULL = ROOT / "runs" / "s6-sensitivity"


def test_morris_labels_sit_on_their_own_points_in_the_full_run(tmp_path):
    """Read the label anchors back from the figure and compare them with morris.json."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.text import Annotation, Text

    from tools import horizon_mpl
    from tools import make_s6_sensitivity_figure as fig_mod

    if not FULL.is_dir():
        pytest.skip(f"full run {FULL} is missing")
    run_dir = tmp_path / "full"
    shutil.copytree(FULL, run_dir)
    run = fig_mod.load_run(run_dir)
    pal = horizon_mpl.apply_style("light")
    fig = fig_mod.build_morris_figure(run, pal, fig_mod.arm_colors(pal))
    try:
        fig.canvas.draw()
        axes = fig.axes
        panels = [(cls, arm) for cls, _case in fig_mod.CLASSES for arm in fig_mod.ARMS]
        assert len(axes) == len(panels) == 4
        names = run["morris"]["inputs"]["names"]
        for ax, (cls, arm) in zip(axes, panels):
            effects = run["morris"]["classes"][cls]["effects"][arm]
            expected = {
                fig_mod.input_label(n): (effects["mu_star"][i] * 100.0, effects["sigma"][i] * 100.0)
                for i, n in enumerate(names)
            }
            labels = [t for t in ax.texts if isinstance(t, Annotation) and t.arrow_patch is not None]
            assert len(labels) == len(names), (cls, arm)
            offsets = ax.collections[0].get_offsets()
            assert len(offsets) == len(names)
            pixels = {tuple(round(float(v), 9) for v in o) for o in offsets}
            seen = set()
            renderer = fig.canvas.get_renderer()
            for t in labels:
                key = t.get_text().replace(" (no effect)", "")
                assert key in expected, f"unexpected label {t.get_text()!r}"
                seen.add(key)
                want = expected[key]
                assert t.xy == pytest.approx(want, abs=1e-9), (cls, arm, key)
                assert tuple(round(v, 9) for v in want) in pixels
                # The leader runs from the label box to its own point and must not pass
                # through any other point, so the label cannot be read as a neighbour's.
                box = Text.get_window_extent(t, renderer)  # the text only, not its leader
                px, py = ax.transData.transform(want)
                seg = (px, py, min(max(px, box.x0), box.x1), min(max(py, box.y0), box.y1))

                def dist_to_segment(point):
                    qx, qy = ax.transData.transform(point)
                    dx, dy = seg[2] - seg[0], seg[3] - seg[1]
                    length2 = dx * dx + dy * dy
                    u = 0.0 if length2 == 0 else max(
                        0.0, min(1.0, ((qx - seg[0]) * dx + (qy - seg[1]) * dy) / length2))
                    return math.hypot(qx - (seg[0] + u * dx), qy - (seg[1] + u * dy))

                for other_key, other in expected.items():
                    ox, oy = ax.transData.transform(other)
                    twin = math.hypot(ox - px, oy - py) < 9.0  # markers drawn on top of each other
                    if other_key != key and not twin:
                        assert dist_to_segment(other) > 6.0, (cls, arm, key, other_key)
            assert seen == set(expected)
            # No two label texts overlap, and no label text covers a marker.
            boxes = {tt.get_text(): Text.get_window_extent(tt, renderer) for tt in labels}
            keys = sorted(boxes)
            for a_i, ka in enumerate(keys):
                for kb in keys[a_i + 1:]:
                    assert not boxes[ka].overlaps(boxes[kb]), (cls, arm, ka, kb)
                for point in expected.values():
                    qx, qy = ax.transData.transform(point)
                    covered = (boxes[ka].x0 - 3 <= qx <= boxes[ka].x1 + 3
                               and boxes[ka].y0 - 3 <= qy <= boxes[ka].y1 + 3)
                    assert not covered, (cls, arm, ka, point)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
