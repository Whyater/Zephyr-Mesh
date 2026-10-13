"""Canvas 3D scene for the Windows replay: orbit camera, X-frame drones, target, floor grid.

Geometry lives in ``scene_projection`` (pure, unit tested). This module owns
the Canvas: it keeps one fixed set of items per drone and only moves or
recolors them, raises drones far to near once per draw, and coalesces redraws
so at most one draw runs per Tk idle cycle or animation frame.
"""
from __future__ import annotations

import math
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .design_tokens import MOTION_MS, SPACE, UI_TEXT, WIN_LAYOUT
from .scene_projection import (
    CameraAnimator,
    finite_point,
    fit_distance,
    OrbitCamera,
    Projector,
    ViewPose,
    circle_points,
    clamp_distance,
    far_to_near,
    framing_camera,
    grid_lines,
    hit_test,
    top_pose,
    xframe_glyph,
)
from .ui_theme import Theme, mix


FADED_LEVELS = {"caution", "alert"}
# Late and lost drones also get a small status-colored marker above them: fading alone is easy to miss.
MARKED_LEVELS = FADED_LEVELS
MARKER_PX = SPACE["xxs"] + 1.5
FOCUS_DISTANCE_M = 4.0
FIT_MARGIN_PX = 34
TOP_LIMIT_RADII = 6.0
ARM_M = 0.2
ROTOR_M = 0.1
FRAME_MS = 16


HUD_AXIS = SPACE["lg"] + SPACE["sm"] - SPACE["xxs"]  # axis arrow length; the HUD sits SPACE["lg"] from the corner


def _agents(frame: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    agents = frame.get("agents", frame.get("drones", [])) if isinstance(frame, Mapping) else []
    return [agent for agent in agents if isinstance(agent, Mapping)] if isinstance(agents, list) else []


def _heading(velocity: Any) -> float | None:
    """Heading in radians from a recorded velocity; ``None`` at rest or when the vector is malformed."""
    point = finite_point(velocity)
    if point is None or math.hypot(point[0], point[1]) < 0.01:
        return None
    return math.atan2(point[1], point[0])


def _agent_position(agent: Mapping[str, Any]) -> tuple[float, float, float] | None:
    return finite_point(agent.get("position_m", agent.get("position")))


class SceneView:
    """Projected 3D and Top views of one replay frame."""

    def __init__(
        self,
        parent: Any,
        *,
        scale: float,
        fonts: Any,
        all_points: Sequence[Sequence[float]],
        on_select: Callable[[str], None],
        on_focus: Callable[[str], None],
        on_context: Callable[[str, int, int], None],
        reduced_motion: bool = False,
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.scale = scale
        self.fonts = fonts
        ring = max(1, int(round(WIN_LAYOUT["hairline"] * scale)))
        self.canvas = tk.Canvas(parent, highlightthickness=ring, borderwidth=0, takefocus=1)
        self.on_select = on_select
        self.on_focus = on_focus
        self.on_context = on_context
        self.reduced_motion = reduced_motion
        self.theme: Theme | None = None
        self.frame: Mapping[str, Any] = {}
        self.records: dict[str, Mapping[str, Any]] = {}
        self.order: list[str] = []
        self.selected = ""
        self.view_mode = "3d"
        self.show_keepout = True
        self.draw_count = 0
        # Malformed or non-finite points would poison the framing (NaN on the Canvas), so they are left out.
        self._points = [point for point in (finite_point(item) for item in all_points) if point is not None]
        self._home = framing_camera(self._points)
        self._fitted_size: tuple[int, int] | None = None
        self._top_fit = self._home.distance
        self._user_moved = False
        self.camera: OrbitCamera = self._home
        self.animator = CameraAnimator(ViewPose.of(self.camera))
        self._grid_items: list[tuple[int, bool, tuple]] = []
        self._drone_items: list[dict[str, Any]] = []
        self._static: dict[str, Any] = {}
        self._keepout_items: list[int] = []
        self._screen_points: list[tuple[float, float] | None] = []
        self._draw_job: Any = None
        self._anim_job: Any = None
        self._press: tuple[int, int] | None = None
        self._dragging = False
        self._hover_cursor = ""
        radius = max(self._home.swarm_radius, 1.0)
        self._grid = grid_lines(self._home.target, math.ceil(radius * 1.6 + 2.0))
        canvas = self.canvas
        canvas.bind("<Configure>", lambda _event: self.request_draw())
        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        canvas.bind("<Double-Button-1>", self._on_double)
        canvas.bind("<MouseWheel>", self._on_wheel)
        canvas.bind("<Button-4>", lambda event: self._zoom_steps(1))
        canvas.bind("<Button-5>", lambda event: self._zoom_steps(-1))
        canvas.bind("<Motion>", self._on_motion)
        canvas.bind("<Button-3>", self._on_context_click)
        canvas.bind("<Control-Button-1>", self._on_context_click)
        if sys.platform == "darwin":
            canvas.bind("<Button-2>", self._on_context_click)  # Aqua reports the secondary click as Button-2

    # ---- state from the app -------------------------------------------------
    def set_frame(self, frame: Mapping[str, Any], records: Sequence[Mapping[str, Any]], selected: str) -> None:
        self.frame = frame
        self.records = {str(record["drone_id"]): record for record in records}
        self.order = [str(agent.get("agent_id", agent.get("id", ""))) for agent in _agents(frame)]
        self.selected = selected
        self.request_draw()

    def set_selected(self, drone_id: str) -> None:
        if drone_id != self.selected:
            self.selected = drone_id
            self.request_draw()

    def set_keepout_visible(self, visible: bool) -> None:
        self.show_keepout = bool(visible)
        self.request_draw()

    def set_view_mode(self, mode: str) -> None:
        if mode == self.view_mode:
            return
        self.view_mode = mode
        self._animate_to(self._goal_pose(), MOTION_MS["standard"])

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.canvas.configure(background=theme.scene.floor.hex, highlightbackground=theme.scene.floor.hex,
                              highlightcolor=theme.p.accent)
        self._recolor_static()
        self.request_draw()

    # ---- camera ------------------------------------------------------------
    def _goal_pose(self) -> ViewPose:
        if self.view_mode != "top":
            return ViewPose.of(self.camera)
        pose = top_pose(self.camera)
        ratio = self.camera.distance / max(self._home.distance, 1e-6)
        return ViewPose(pose.yaw, pose.pitch, min(self._top_fit * ratio, self._top_limit(self.camera)), pose.target)

    @staticmethod
    def _top_limit(camera: OrbitCamera) -> float:
        """Farthest top-view distance: 6x the swarm radius.

        The orbit view's 3x limit works because an oblique view foreshortens
        the swarm by sin(pitch). Looking straight down, the full swarm diameter
        lands on the short screen axis, so a wide, short canvas needs up to
        about twice the distance to show every drone. It stays bounded.
        """
        return max(camera.swarm_radius * TOP_LIMIT_RADII, camera.near_limit)

    def _animate_to(self, goal: ViewPose, duration_ms: float) -> None:
        now = time.monotonic() * 1000.0
        self.animator.go(goal, now, duration_ms, self.reduced_motion)
        if self.reduced_motion or not self.animator.active:
            self.request_draw()
            return
        if self._anim_job is None:
            self._anim_job = self.canvas.after(FRAME_MS, self._animate)

    def _animate(self) -> None:
        self._anim_job = None
        self.animator.advance(time.monotonic() * 1000.0)
        self._draw_now()
        if self.animator.active:
            self._anim_job = self.canvas.after(FRAME_MS, self._animate)

    def _refit(self, width: int, height: int) -> None:
        """Frame the whole swarm for this canvas size, unless the user has moved the camera."""
        if self._fitted_size == (width, height):
            return
        self._fitted_size = (width, height)
        distance = fit_distance(self._home, self._points, width, height, FIT_MARGIN_PX * self.scale)
        top = OrbitCamera(-90.0, 85.0, self._home.distance, self._home.target, self._home.near_limit, self._home.swarm_radius)
        self._top_fit = self._fit_top(top, width, height)
        self._home = OrbitCamera(self._home.yaw, self._home.pitch, distance, self._home.target,
                                 self._home.near_limit, self._home.swarm_radius)
        if not self._user_moved:
            self.camera = self._home
            self.animator.go(self._goal_pose(), 0, 0, True)

    def _fit_top(self, camera: OrbitCamera, width: int, height: int) -> float:
        """Distance that frames the swarm in the straight-down view."""
        margin = FIT_MARGIN_PX * self.scale
        low, high = camera.near_limit, self._top_limit(camera)
        for _ in range(30):
            middle = (low + high) / 2.0
            projector = Projector(ViewPose(-90.0, 90.0, middle, camera.target), width, height)
            inside = all(
                margin <= point.x <= width - margin and margin <= point.y <= height - margin and point.depth > 0.2
                for point in (projector.project(item) for item in self._points)
            )
            if inside:
                high = middle
            else:
                low = middle
        return high

    def refit(self) -> None:
        """Frame the swarm again for the current canvas size (after a side panel is shown or hidden)."""
        self._fitted_size = None
        self.request_draw()

    def reset_camera(self) -> None:
        self._user_moved = False
        self.camera = self._home
        self._animate_to(self._goal_pose(), MOTION_MS["standard"])

    def focus_drone(self, drone_id: str) -> None:
        position = self._position(drone_id)
        if position is None:
            return
        distance = clamp_distance(min(self.camera.distance, FOCUS_DISTANCE_M), self.camera.swarm_radius, self.camera.near_limit)
        self._user_moved = True
        self.camera = OrbitCamera(self.camera.yaw, self.camera.pitch, distance, tuple(position),
                                  self.camera.near_limit, self.camera.swarm_radius)
        self._animate_to(self._goal_pose(), MOTION_MS["standard"])

    def _zoom_steps(self, steps: float) -> None:
        self._user_moved = True
        self.camera = self.camera.zoom(0.88 ** steps)
        self._animate_to(self._goal_pose(), MOTION_MS["fast"])

    # ---- input -------------------------------------------------------------
    def _drone_at(self, x: float, y: float) -> str | None:
        index = hit_test(self._screen_points, x, y, 14 * self.scale)
        return self.order[index] if index is not None and index < len(self.order) else None

    def _on_press(self, event: Any) -> None:
        self.canvas.focus_set()
        self._press = (event.x, event.y)
        self._dragging = False

    def _on_drag(self, event: Any) -> None:
        if self._press is None:
            return
        dx, dy = event.x - self._press[0], event.y - self._press[1]
        if not self._dragging and math.hypot(dx, dy) < 3 * self.scale:
            return
        self._dragging = True
        self._press = (event.x, event.y)
        if self.view_mode == "top":
            return
        self._user_moved = True
        self.camera = self.camera.drag_delta(dx / self.scale, dy / self.scale)
        self._animate_to(self._goal_pose(), MOTION_MS["fast"])

    def _on_release(self, event: Any) -> None:
        if self._press is not None and not self._dragging:
            drone = self._drone_at(event.x, event.y)
            if drone:
                self.on_select(drone)
        self._press = None
        self._dragging = False

    def _on_double(self, event: Any) -> None:
        drone = self._drone_at(event.x, event.y)
        if drone:
            self.on_focus(drone)

    def _on_wheel(self, event: Any) -> str:
        delta = float(getattr(event, "delta", 0))
        if delta:
            self._zoom_steps(delta / 120.0 if abs(delta) >= 120 else (0.35 if delta > 0 else -0.35))
        return "break"

    def wheel(self, delta: float) -> None:
        """Zoom from a routed wheel event (the app routes TouchpadScroll here too)."""
        if delta:
            self._zoom_steps(delta / 120.0 if abs(delta) >= 120 else (0.35 if delta > 0 else -0.35))

    def _on_motion(self, event: Any) -> None:
        cursor = "hand2" if self._drone_at(event.x, event.y) else ""
        if cursor != self._hover_cursor:
            self._hover_cursor = cursor
            self.canvas.configure(cursor=cursor)

    def _on_context_click(self, event: Any) -> str | None:
        drone = self._drone_at(event.x, event.y)
        if drone:
            self.on_context(drone, event.x_root, event.y_root)
            return "break"
        return None

    # ---- drawing -----------------------------------------------------------
    def request_draw(self) -> None:
        if self._draw_job is None and self._anim_job is None:
            self._draw_job = self.canvas.after_idle(self._draw_now)

    def _position(self, drone_id: str) -> tuple[float, float, float] | None:
        for agent in _agents(self.frame):
            if str(agent.get("agent_id", agent.get("id", ""))) == drone_id:
                return _agent_position(agent)
        return None

    def _ensure_items(self) -> None:
        canvas = self.canvas
        if not self._grid_items:
            for start, end, major in self._grid:
                self._grid_items.append((canvas.create_line(0, 0, 0, 0, tags=("grid",)), major, (start, end)))
            self._static["target_stem"] = canvas.create_line(0, 0, 0, 0, tags=("target",))
            self._static["target_foot"] = canvas.create_oval(0, 0, 0, 0, tags=("target",))
            self._static["target_ring"] = canvas.create_polygon(0, 0, 0, 0, 0, 0, tags=("target",))
            self._static["axis_x"] = canvas.create_line(0, 0, 0, 0, tags=("hud",))
            self._static["axis_y"] = canvas.create_line(0, 0, 0, 0, tags=("hud",))
            self._static["axis_x_label"] = canvas.create_text(0, 0, text="x", anchor="center", tags=("hud",))
            self._static["axis_y_label"] = canvas.create_text(0, 0, text="y", anchor="center", tags=("hud",))
            self._static["grid_label"] = canvas.create_text(0, 0, text=UI_TEXT["windowsText"]["gridLabel"], anchor="w",
                                                            tags=("hud",))
            self._static["label"] = canvas.create_text(0, 0, text="", anchor="w", tags=("hud",))
            self._recolor_static()
        while len(self._drone_items) < len(self.order):
            index = len(self._drone_items)
            tag = f"drone{index}"
            items = {
                "foot": canvas.create_oval(0, 0, 0, 0, width=0, tags=("foot",)),
                "discs": [canvas.create_oval(0, 0, 0, 0, tags=(tag,)) for _ in range(4)],
                "arms": [canvas.create_line(0, 0, 0, 0, capstyle="round", tags=(tag,)) for _ in range(2)],
                "body": canvas.create_oval(0, 0, 0, 0, tags=(tag,)),
                "motors": [canvas.create_oval(0, 0, 0, 0, width=0, tags=(tag,)) for _ in range(4)],
                "nose": canvas.create_polygon(0, 0, 0, 0, 0, 0, width=0, tags=(tag,)),
                "marker": canvas.create_oval(0, 0, 0, 0, width=0, tags=(tag,)),
                "tag": tag,
                "style": None,
            }
            self._drone_items.append(items)

    def _recolor_static(self) -> None:
        if self.theme is None or not self._static:
            return
        theme = self.theme
        p, scene = theme.p, theme.scene
        floor = scene.floor.hex
        if theme.dark:
            minor = scene.grid.hex
            major = mix(p.lineStrong, scene.grid.hex, 0.45)
        else:
            # On the light floor the token grid reads as heavy rules through the ring; keep both quieter.
            minor = mix(scene.grid.hex, floor, 0.55)
            major = mix(p.lineStrong, floor, 0.35)
        for item, is_major, _ends in self._grid_items:
            self.canvas.itemconfigure(item, fill=major if is_major else minor, width=1)
        ring = scene.target_ring.hex
        self.canvas.itemconfigure(self._static["target_ring"], outline=ring, fill="", width=max(2, int(round(2 * self.scale))))
        self.canvas.itemconfigure(self._static["target_stem"], fill=mix(ring, floor, 0.55), width=1)
        self.canvas.itemconfigure(self._static["target_foot"], outline=mix(ring, floor, 0.55), fill="", width=1)
        for name in ("axis_x", "axis_y"):
            self.canvas.itemconfigure(self._static[name], fill=p.textSecondary, width=max(1, int(round(1.5 * self.scale))), arrow="last",
                                      arrowshape=(6 * self.scale, 7 * self.scale, 2.5 * self.scale))
        for name in ("axis_x_label", "axis_y_label", "grid_label"):
            self.canvas.itemconfigure(self._static[name], fill=p.textSecondary, font=self.fonts["caption"])
        self.canvas.itemconfigure(self._static["label"], fill=p.text, font=self.fonts["caption"])
        for items in self._drone_items:
            items["style"] = None

    def _drone_style(self, record: Mapping[str, Any] | None, selected: bool) -> tuple[str, ...]:
        theme = self.theme
        assert theme is not None
        floor = theme.scene.floor.hex
        if selected:
            body = arms = disc = theme.p.accent
        else:
            body, arms, disc = theme.scene.drone_body.hex, theme.scene.drone_arms.hex, theme.scene.rotor_disc.hex
        if record is not None and record.get("level") in FADED_LEVELS and not selected:
            # Faded, but still legible on the dark floor: keep about 60 percent of the color.
            body, arms, disc = mix(body, floor, 0.6), mix(arms, floor, 0.6), mix(disc, floor, 0.7)
        level = str(record.get("level", "")) if record is not None else ""
        marker = theme.status_color(level) if level in MARKED_LEVELS else ""
        return (
            body,
            arms,
            mix(disc, floor, 0.3),
            mix(disc, floor, 0.62),
            mix(theme.p.textSecondary, floor, 0.22),
            marker,
        )

    def _draw_now(self) -> None:
        self._draw_job = None
        if self.theme is None:
            return
        canvas = self.canvas
        width = max(int(canvas.winfo_width()), 2)
        height = max(int(canvas.winfo_height()), 2)
        if width < 10 or height < 10:
            return
        self._ensure_items()
        self._refit(width, height)
        self.draw_count += 1
        pose = self.animator.pose
        projector = Projector(pose, width, height)
        hidden = (-10, -10, -10, -10)
        for item, _major, (start, end) in self._grid_items:
            segment = projector.segment(start, end)
            canvas.coords(item, *(hidden if segment is None else (segment[0].x, segment[0].y, segment[1].x, segment[1].y)))

        target = finite_point(self.frame.get("target_position_m", self.frame.get("target_position")))
        entities: list[tuple[float, str, Any]] = []
        if target is not None:
            tx, ty, tz = target
            ring = [projector.project(point) for point in circle_points((tx, ty, tz), 0.32, 48)]
            stem = projector.segment((tx, ty, tz), (tx, ty, 0.0))
            foot_center = projector.project((tx, ty, 0.0))
            if all(point.depth > 0.2 for point in ring):
                canvas.coords(self._static["target_ring"], *[value for point in ring for value in (point.x, point.y)])
            else:
                canvas.coords(self._static["target_ring"], *hidden, *hidden[:2])
            canvas.coords(self._static["target_stem"], *(hidden if stem is None else (stem[0].x, stem[0].y, stem[1].x, stem[1].y)))
            foot_r = 0.1 * projector.scale_at(max(foot_center.depth, 0.2))
            canvas.coords(self._static["target_foot"], foot_center.x - foot_r, foot_center.y - foot_r * projector.ellipse_ratio,
                          foot_center.x + foot_r, foot_center.y + foot_r * projector.ellipse_ratio)
            entities.append((projector.project((tx, ty, tz)).depth, "target", None))
            canvas.itemconfigure("target", state="normal")
        else:
            # No target in this frame: hide the previous frame's ring and stem rather than leave them stale.
            canvas.itemconfigure("target", state="hidden")

        self._draw_keepout(projector, entities)

        agents = {str(agent.get("agent_id", agent.get("id", ""))): agent for agent in _agents(self.frame)}
        screen_points: list[tuple[float, float] | None] = []
        label_at = None
        for index, drone_id in enumerate(self.order):
            items = self._drone_items[index]
            agent = agents.get(drone_id, {})
            position = _agent_position(agent)
            glyph = None
            if position is not None:
                glyph = xframe_glyph(projector, position, _heading(agent.get("velocity_mps")), ARM_M, ROTOR_M)
            if glyph is None:
                screen_points.append(None)
                canvas.itemconfigure(items["tag"], state="hidden")
                canvas.itemconfigure(items["foot"], state="hidden")
                continue
            canvas.itemconfigure(items["tag"], state="normal")
            canvas.itemconfigure(items["foot"], state="hidden" if pose.pitch > 80 else "normal")
            selected = drone_id == self.selected
            style = self._drone_style(self.records.get(drone_id), selected)
            if items["style"] != style:
                body, arms, disc_fill, disc_line, foot, marker = style
                for disc in items["discs"]:
                    canvas.itemconfigure(disc, fill=disc_fill, outline=disc_line, width=1)
                for arm in items["arms"]:
                    canvas.itemconfigure(arm, fill=arms, width=max(2, int(round(2.2 * self.scale))))
                canvas.itemconfigure(items["body"], fill=body, outline=body)
                for motor in items["motors"]:
                    canvas.itemconfigure(motor, fill=arms)
                canvas.itemconfigure(items["nose"], fill=body)
                canvas.itemconfigure(items["foot"], fill=foot)
                canvas.itemconfigure(items["marker"], fill=marker or "", state="normal" if marker else "hidden")
                items["style"] = style
            tips = glyph.arm_tips
            for disc, (cx, cy, rx, ry) in zip(items["discs"], glyph.rotors):
                canvas.coords(disc, cx - rx, cy - ry, cx + rx, cy + ry)
            canvas.coords(items["arms"][0], tips[0][0], tips[0][1], tips[2][0], tips[2][1])
            canvas.coords(items["arms"][1], tips[1][0], tips[1][1], tips[3][0], tips[3][1])
            canvas.coords(items["body"], *glyph.body)
            radius = glyph.motor_radius
            for motor, (mx, my) in zip(items["motors"], tips):
                canvas.coords(motor, mx - radius, my - radius * max(projector.ellipse_ratio, 0.6), mx + radius,
                        my + radius * max(projector.ellipse_ratio, 0.6))
            if glyph.heading is None:
                canvas.coords(items["nose"], *hidden[:2], *hidden[:2], *hidden[:2])
            else:
                canvas.coords(items["nose"], *[value for point in glyph.heading for value in point])
            if style[-1]:
                lowest_top = min(cy - ry for cx, cy, rx, ry in glyph.rotors)
                marker_r = MARKER_PX * self.scale
                marker_y = lowest_top - marker_r - SPACE["xs"] * self.scale
                canvas.coords(items["marker"], glyph.center.x - marker_r, marker_y - marker_r, glyph.center.x + marker_r,
                              marker_y + marker_r)
            foot = projector.project((position[0], position[1], 0.0))
            foot_r = 0.07 * projector.scale_at(max(foot.depth, 0.2))
            canvas.coords(items["foot"], foot.x - foot_r, foot.y - foot_r * projector.ellipse_ratio,
                          foot.x + foot_r, foot.y + foot_r * projector.ellipse_ratio)
            screen_points.append((glyph.center.x, glyph.center.y))
            entities.append((glyph.center.depth, "drone", items["tag"]))
            if selected:
                label_at = (glyph.rotors[0][0] + glyph.rotors[0][2] + SPACE["xs"] * self.scale, glyph.center.y,
                            self.records.get(drone_id, {}).get("compact_name", drone_id))
        for index in range(len(self.order), len(self._drone_items)):
            canvas.itemconfigure(self._drone_items[index]["tag"], state="hidden")
            canvas.itemconfigure(self._drone_items[index]["foot"], state="hidden")
        self._screen_points = screen_points

        canvas.tag_raise("foot", "grid")
        canvas.tag_raise("target", "foot")
        for _depth, kind, handle in sorted(entities, key=lambda entity: -entity[0]):
            if kind == "drone":
                canvas.tag_raise(handle)
            elif kind == "target":
                canvas.tag_raise("target")
            elif kind == "keepout":
                for item in handle:
                    canvas.tag_raise(item)
        if label_at is not None:
            canvas.coords(self._static["label"], label_at[0], label_at[1])
            canvas.itemconfigure(self._static["label"], text=label_at[2], state="normal")
        else:
            canvas.itemconfigure(self._static["label"], state="hidden")
        self._draw_hud(pose, height)
        canvas.tag_raise("hud")

    def _draw_keepout(self, projector: Projector, entities: list[tuple[float, str, Any]]) -> None:
        canvas = self.canvas
        for item in self._keepout_items:
            canvas.delete(item)
        self._keepout_items = []
        if not self.show_keepout or self.theme is None:
            return
        spheres = self.frame.get("keep_out_spheres") or []
        color = self.theme.scene.keep_out_volume
        floor = self.theme.scene.floor.hex
        for sphere in spheres:
            try:
                center_point = finite_point(sphere["center_m"])
                radius = float(sphere["radius_m"])
            except (KeyError, TypeError, ValueError):
                continue
            if center_point is None or not math.isfinite(radius) or radius <= 0:
                continue
            cx, cy, cz = center_point
            center = projector.project((cx, cy, cz))
            if center.depth <= radius + 0.2:
                continue
            r = radius * projector.scale_at(center.depth)
            ry = r * projector.ellipse_ratio
            fill = mix(color.hex, floor, color.alpha)
            items = [
                canvas.create_oval(center.x - r, center.y - r, center.x + r, center.y + r, fill=fill, outline=color.hex,
                                   width=max(1, int(round(1.5 * self.scale)))),
                canvas.create_oval(center.x - r, center.y - ry, center.x + r, center.y + ry, outline=mix(color.hex, floor, 0.6), width=1),
            ]
            self._keepout_items.extend(items)
            entities.append((center.depth, "keepout", items))

    def _draw_hud(self, pose: ViewPose, height: int) -> None:
        canvas = self.canvas
        origin_x = (SPACE["lg"] + SPACE["md"]) * self.scale
        origin_y = height - (SPACE["lg"] + SPACE["md"]) * self.scale
        length = HUD_AXIS * self.scale
        yaw = math.radians(pose.yaw)
        pitch_factor = max(0.25, math.sin(math.radians(pose.pitch)))
        right = (-math.sin(yaw), math.cos(yaw))

        def screen(dx: float, dy: float) -> tuple[float, float]:
            horizontal = dx * right[0] + dy * right[1]
            depth_axis = dx * math.cos(yaw) + dy * math.sin(yaw)
            return origin_x + horizontal * length, origin_y + depth_axis * length * pitch_factor

        for axis, label, (dx, dy) in (("axis_x", "axis_x_label", (1, 0)), ("axis_y", "axis_y_label", (0, 1))):
            end = screen(dx, dy)
            canvas.coords(self._static[axis], origin_x, origin_y, *end)
            label_point = screen(dx * 1.6, dy * 1.6)
            canvas.coords(self._static[label], *label_point)
        canvas.coords(self._static["grid_label"], origin_x + length + SPACE["xl"] * self.scale, origin_y)


__all__ = ["SceneView"]
