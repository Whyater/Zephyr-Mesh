from __future__ import annotations

import json
import math
from pathlib import Path

from tools.desktop.scene_projection import (
    OrbitCamera,
    clamp_distance,
    clamp_pitch,
    ease_out,
    fit_all,
    painter_order,
    project_point,
    quadcopter_glyph,
)


ROOT = Path(__file__).resolve().parents[1]


def test_camera_clamps_and_top_view():
    camera = OrbitCamera(pitch=-5, distance=100, swarm_radius=2)
    assert camera.pitch == 10
    assert camera.distance == 12, "zoom out is bounded at FAR_LIMIT_RADII (6) swarm radii"
    assert clamp_pitch(100) == 85
    assert clamp_distance(0, 2) == 0.5
    assert camera.top_view().pitch == 85


def test_fit_all_bounds_ring_fixture():
    run = json.loads((ROOT / "runs/s7-swarm/run.json").read_text())
    points = [agent["position_m"] for agent in run["steps"][0]["agents"]]
    camera = fit_all(points)
    assert camera.target[0] == 0
    assert camera.target[1] == 0
    assert camera.distance <= camera.swarm_radius * 3


def test_projection_center_and_depth():
    camera = OrbitCamera(yaw=0, pitch=35, distance=10, target=(0, 0, 0), swarm_radius=4)
    center = project_point(camera, camera.target, 800, 600)
    assert center.x == 400
    assert center.y == 300
    near = project_point(camera, (5, 0, 0), 800, 600)
    far = project_point(camera, (-5, 0, 0), 800, 600)
    assert near.depth < far.depth


def test_sorting_ease_and_glyph_counts():
    assert painter_order([("near", 2), ("far", 1)]) == [("far", 1), ("near", 2)]
    assert ease_out(0, 10, 0.5) == 7.5
    assert ease_out(0, 10, 0.5, reduced_motion=True) == 10
    camera = fit_all([(0, 0, 0), (1, 1, 0)])
    glyph = quadcopter_glyph(camera, (0, 0, 0), math.pi / 4, 1, 400, 300)
    assert [primitive.kind for primitive in glyph].count("body") == 1
    assert [primitive.kind for primitive in glyph].count("arm") == 4
    assert [primitive.kind for primitive in glyph].count("rotor") == 4
    assert [primitive.kind for primitive in glyph].count("heading") == 1
    assert all(primitive.radius > 0 for primitive in glyph if primitive.kind == "rotor")
