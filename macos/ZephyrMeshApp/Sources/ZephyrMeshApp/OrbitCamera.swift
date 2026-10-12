import Foundation
import simd

/// Pure orbit camera state in simulator coordinates (z up). Orbit pitch is clamped to
/// 10...85 degrees and distance to nearLimit...3x the swarm radius; there is no roll and no
/// free translation. Top view is a separate straight-down mode that keeps the same target.
struct OrbitCamera: Equatable {
    static let nearLimit: Float = 0.35
    static let minimumPitch: Float = 10
    static let maximumPitch: Float = 85
    static let topPitch: Float = 90
    static let homeYaw: Float = 30
    static let homePitch: Float = 32

    var yaw: Float
    var pitch: Float
    var distance: Float
    var target: SIMD3<Float>
    var swarmRadius: Float
    var isTopView = false

    init(yaw: Float = 45, pitch: Float = 35, distance: Float = 8, target: SIMD3<Float> = .zero, swarmRadius: Float = 1) {
        self.yaw = yaw
        self.pitch = pitch
        self.distance = distance
        self.target = target
        self.swarmRadius = max(Self.nearLimit, swarmRadius)
        clamp()
    }

    var maximumDistance: Float { max(Self.nearLimit, swarmRadius * 3) }

    mutating func applyDrag(delta: SIMD2<Float>) {
        yaw += delta.x
        if !isTopView { pitch += delta.y }
        clamp()
    }

    mutating func applyDrag(deltaX: Float, deltaY: Float) {
        applyDrag(delta: SIMD2(deltaX, deltaY))
    }

    /// Multiplies closeness by `factor`: 2 halves the distance, 0.5 doubles it.
    mutating func applyZoom(factor: Float) {
        guard factor.isFinite, factor > 0 else { return }
        distance /= factor
        clamp()
    }

    mutating func clamp() {
        if !yaw.isFinite { yaw = 0 }
        if !pitch.isFinite { pitch = Self.homePitch }
        if !distance.isFinite { distance = maximumDistance }
        pitch = isTopView ? Self.topPitch : min(max(pitch, Self.minimumPitch), Self.maximumPitch)
        distance = min(max(distance, Self.nearLimit), maximumDistance)
        if !(target.x.isFinite && target.y.isFinite && target.z.isFinite) { target = .zero }
    }

    mutating func clamp(swarmRadius: Float) {
        self.swarmRadius = max(Self.nearLimit, swarmRadius.isFinite ? swarmRadius : 1)
        clamp()
    }

    /// Centers on the points and sets a distance that fits them, keeping yaw and pitch.
    mutating func frame(points: [SIMD3<Float>], padding: Float = 1.25) {
        guard !points.isEmpty else {
            target = .zero
            swarmRadius = 1
            clamp()
            return
        }
        let center = points.reduce(SIMD3<Float>.zero, +) / Float(points.count)
        let radius = points.map { simd_distance($0, center) }.max() ?? 1
        target = center
        swarmRadius = max(Self.nearLimit, radius * max(padding, 1))
        distance = swarmRadius * 2.1
        clamp()
    }

    static func framed(points: [SIMD3<Float>], padding: Float = 1.25) -> OrbitCamera {
        var camera = OrbitCamera(yaw: homeYaw, pitch: homePitch)
        camera.frame(points: points, padding: padding)
        return camera
    }

    mutating func focus(point: SIMD3<Float>) {
        guard point.x.isFinite, point.y.isFinite, point.z.isFinite else { return }
        target = point
        clamp()
    }

    mutating func focus(on point: SIMD3<Float>) {
        focus(point: point)
    }

    /// Straight down, north up. Yaw resets so the plan view always has the same orientation.
    mutating func topView() {
        isTopView = true
        yaw = 0
        clamp()
    }

    mutating func orbitView(pitch newPitch: Float = OrbitCamera.homePitch) {
        isTopView = false
        pitch = newPitch
        clamp()
    }

    static func interpolate(from: OrbitCamera, to: OrbitCamera, progress: Float, reducedMotion: Bool = false) -> OrbitCamera {
        if reducedMotion { return progress >= 1 ? to : from }
        let t = min(max(progress, 0), 1)
        if t == 0 { return from }
        if t == 1 { return to }
        let eased = 1 - pow(1 - t, 2)
        var result = from
        result.yaw = from.yaw + (to.yaw - from.yaw) * eased
        result.pitch = from.pitch + (to.pitch - from.pitch) * eased
        result.distance = from.distance + (to.distance - from.distance) * eased
        result.target = from.target + (to.target - from.target) * eased
        result.swarmRadius = from.swarmRadius + (to.swarmRadius - from.swarmRadius) * eased
        result.isTopView = false
        result.pitch = min(max(result.pitch, Self.minimumPitch), Self.topPitch)
        result.distance = min(max(result.distance, Self.nearLimit), max(from.maximumDistance, to.maximumDistance))
        return result
    }

    func interpolated(to other: OrbitCamera, progress: Float, reducedMotion: Bool = false) -> OrbitCamera {
        Self.interpolate(from: self, to: other, progress: progress, reducedMotion: reducedMotion)
    }

    /// Frame-rate independent ease-out toward `goal`: after `elapsed` seconds the remaining
    /// gap shrinks by exp(-elapsed / timeConstant). Snaps once the gap is negligible.
    func smoothed(toward goal: OrbitCamera, elapsed: Float, timeConstant: Float = 0.09) -> OrbitCamera {
        guard elapsed > 0, timeConstant > 0 else { return goal }
        let progress = 1 - exp(-elapsed / timeConstant)
        var next = self
        next.yaw += (goal.yaw - yaw) * progress
        next.pitch += (goal.pitch - pitch) * progress
        next.distance += (goal.distance - distance) * progress
        next.target += (goal.target - target) * progress
        next.swarmRadius = goal.swarmRadius
        next.isTopView = goal.isTopView
        let settled = abs(goal.yaw - next.yaw) < 0.01 && abs(goal.pitch - next.pitch) < 0.01
            && abs(goal.distance - next.distance) < 0.0005 && simd_distance(goal.target, next.target) < 0.0005
        return settled ? goal : next
    }

    // MARK: SceneKit pose

    /// Simulator (x, y, z up) to SceneKit (x, y up, z toward the viewer): (x, z, -y).
    static func sceneVector(_ point: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(point.x, point.z, -point.y)
    }

    /// Camera position in SceneKit coordinates.
    var sceneEye: SIMD3<Float> {
        let yawRadians = yaw * .pi / 180
        let pitchRadians = pitch * .pi / 180
        let offset = SIMD3<Float>(cos(pitchRadians) * sin(yawRadians), sin(pitchRadians), cos(pitchRadians) * cos(yawRadians))
        return Self.sceneVector(target) + offset * distance
    }

    /// Camera orientation: pitch down about the local x axis, then yaw about the world up axis.
    var sceneOrientation: simd_quatf {
        let yawRotation = simd_quatf(angle: yaw * .pi / 180, axis: SIMD3(0, 1, 0))
        let pitchRotation = simd_quatf(angle: -pitch * .pi / 180, axis: SIMD3(1, 0, 0))
        return yawRotation * pitchRotation
    }
}
