import AppKit
import SceneKit
import simd

/// Everything the scene needs for one update, gathered from the model by the SwiftUI view.
struct SceneState {
    let frame: FixtureFrame
    let styles: [String: StatusStyle]
    let headings: [String: Double]
    let selectedID: String?
    let hoveredID: String?
    let showKeepOut: Bool
    let viewMode: CockpitViewMode
    let palette: Theme.Palette
    let scenePalette: ScenePalette
    let isDark: Bool
    let reduceMotion: Bool
    let cameraCommand: CameraCommand?
}

/// Moves the camera toward its goal on SceneKit's render loop. Mouse events write the goal on
/// the main thread; the render thread eases the current pose toward it, so orbit and zoom
/// stay smooth at the display rate without SwiftUI updates.
final class CameraDriver: NSObject, SCNSceneRendererDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var goalCamera: OrbitCamera
    private var currentCamera: OrbitCamera
    private var lastTime: TimeInterval?
    private var snapNext = true
    private var reducedMotion = false
    private var settleCountdown: Int?
    private var resumeClock = false
    private var settleHandler: (@MainActor () -> Void)?
    let cameraNode: SCNNode

    init(camera: OrbitCamera, cameraNode: SCNNode) {
        goalCamera = camera
        currentCamera = camera
        self.cameraNode = cameraNode
        super.init()
        apply(camera)
    }

    var goal: OrbitCamera {
        lock.lock(); defer { lock.unlock() }
        return goalCamera
    }

    func update(_ change: (inout OrbitCamera) -> Void) {
        lock.lock()
        change(&goalCamera)
        goalCamera.clamp()
        if reducedMotion { snapNext = true }
        lock.unlock()
    }

    /// Settled frames to draw after a request before `settleHandler` runs. More than one, so
    /// scene changes committed just after the request are on screen before drawing stops.
    static let settledFramesBeforeNotice = 3

    /// Runs `handler` on the main queue once the camera has rested at its goal for a few drawn
    /// frames, so the view can stop drawing. One call per `requestSettleNotice()`.
    func onSettle(_ handler: @escaping @MainActor () -> Void) {
        lock.lock(); settleHandler = handler; lock.unlock()
    }

    /// `resumingFromIdle` marks the first frame after the view stopped drawing: its clock gap
    /// is idle time, not motion, so the camera takes one display frame's step, as it did when
    /// the view drew every frame.
    func requestSettleNotice(resumingFromIdle: Bool = false) {
        lock.lock()
        settleCountdown = Self.settledFramesBeforeNotice
        if resumingFromIdle { resumeClock = true }
        lock.unlock()
    }

    var isSettleNoticePending: Bool {
        lock.lock(); defer { lock.unlock() }
        return settleCountdown != nil
    }

    func setReducedMotion(_ value: Bool) {
        lock.lock()
        reducedMotion = value
        if value { snapNext = true }
        lock.unlock()
    }

    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        lock.lock()
        if resumeClock, lastTime != nil { lastTime = time - 1.0 / 60 }
        resumeClock = false
        let elapsed = Float(lastTime.map { min(max(time - $0, 0), 0.1) } ?? 0)
        lastTime = time
        currentCamera = snapNext ? goalCamera : currentCamera.smoothed(toward: goalCamera, elapsed: elapsed)
        snapNext = false
        let pose = currentCamera
        var handler: (@MainActor () -> Void)?
        if let countdown = settleCountdown {
            // A frame where the camera still moves restarts the count.
            let next = currentCamera == goalCamera ? countdown - 1 : Self.settledFramesBeforeNotice
            settleCountdown = next > 0 ? next : nil
            if next <= 0 { handler = settleHandler }
        }
        lock.unlock()
        apply(pose)
        if let handler {
            DispatchQueue.main.async { MainActor.assumeIsolated { handler() } }
        }
    }

    private func apply(_ camera: OrbitCamera) {
        cameraNode.simdPosition = camera.sceneEye
        cameraNode.simdOrientation = camera.sceneOrientation
    }
}

private final class DroneNode {
    let root: SCNNode
    let body: SCNNode
    let blades: [SCNNode]
    let marker: SCNNode
    let dimensions: DroneDimensions
    var look: DroneLook = .normal
    var markerLevel: String?
    var spinning = false

    init(root: SCNNode, body: SCNNode, blades: [SCNNode], marker: SCNNode, dimensions: DroneDimensions) {
        self.root = root
        self.body = body
        self.blades = blades
        self.marker = marker
        self.dimensions = dimensions
    }
}

/// Keeps overlay marks (status markers, the selection ring, the target ring) at a fixed size in
/// points, whatever the camera distance. SceneKit evaluates the constraints on its render
/// thread after the camera driver has moved the camera, so the size is right on every frame.
final class ScreenSizer: @unchecked Sendable {
    private let lock = NSLock()
    private var height: Float = 400
    private let tanHalfFOV: Float
    private let camera: SCNNode

    init(verticalFieldOfView: CGFloat, camera: SCNNode) {
        tanHalfFOV = Float(tan(verticalFieldOfView / 2 * .pi / 180))
        self.camera = camera
    }

    var viewHeight: Float {
        get { lock.lock(); defer { lock.unlock() }; return height }
        set { lock.lock(); height = max(newValue, 1); lock.unlock() }
    }

    /// World size that projects to `points` at `distance` from the camera.
    func worldSize(points: CGFloat, distance: Float) -> Float {
        Float(points) * distance * 2 * tanHalfFOV / viewHeight
    }

    /// Scales the node so a mark `baseSize` metres across appears `points` across. With
    /// `allowsShrink` false the mark never gets smaller than its true size when close up.
    /// The block only computes a transform; it never touches the scene graph, because SceneKit
    /// evaluates it on the render thread.
    func constraint(points: CGFloat, baseSize: CGFloat, allowsShrink: Bool) -> SCNTransformConstraint {
        SCNTransformConstraint(inWorldSpace: true) { [weak self] _, transform in
            guard let self else { return transform }
            let world = simd_float4x4(transform)
            let position = SIMD3(world.columns.3.x, world.columns.3.y, world.columns.3.z)
            let distance = simd_distance(position, self.camera.presentation.simdWorldPosition)
            var scale = self.worldSize(points: points, distance: distance) / Float(baseSize)
            if !allowsShrink { scale = max(scale, 1) }
            guard scale.isFinite, scale > 0 else { return transform }
            var result = matrix_identity_float4x4
            result.columns.0.x = scale
            result.columns.1.y = scale
            result.columns.2.z = scale
            result.columns.3 = world.columns.3
            return SCNMatrix4(result)
        }
    }
}

@MainActor
final class SceneController {
    static let hitCategory = 1 << 1
    static let verticalFieldOfView: CGFloat = 38

    /// Distance at which the swarm radius fills the vertical field of view, for top view.
    private func topDistance(for radius: Float) -> Float {
        radius / Float(tan(Self.verticalFieldOfView / 2 * .pi / 180))
    }

    let scene = SCNScene()
    let cameraNode = SCNNode()
    let driver: CameraDriver
    private let fixture: Fixture
    /// One model factory per distinct (arm length, propeller diameter) in the fixture; drones
    /// with the same dimensions share geometry.
    private let factories: [DroneDimensions: DroneModelFactory]
    private let dimensionsByDrone: [String: DroneDimensions]
    private var drones: [String: DroneNode] = [:]
    private var bodyGeometry: [DroneDimensions: [DroneLook: SCNGeometry]] = [:]
    private var bladeGeometry: [DroneDimensions: SCNGeometry] = [:]
    private var hitGeometry: [DroneDimensions: SCNGeometry] = [:]
    private var ringDimensions: DroneDimensions?
    private let targetNode = SCNNode()
    private let keepOutRoot = SCNNode()
    private let gridNode = SCNNode()
    private let floorNode = SCNNode()
    /// Positioned on the selected drone and sized in points; its child faces the camera.
    private let selectionRing = SCNNode()
    private let selectionRingShape = SCNNode()
    private let sizer: ScreenSizer
    private var selectedScenePosition: SIMD3<Float>?
    private var markerGeometry: [String: SCNGeometry] = [:]
    /// Swarm center and unpadded radius from the home framing, for the fog band.
    private var swarmCenter: SIMD3<Float> = .zero
    private var swarmRawRadius: Float = 1
    private let keyLight = SCNNode()
    private var keepOutVolumes: [KeepOutVolume] = []
    private var themeKey: Bool?
    private var lastCommandSerial = 0
    private var lastViewMode: CockpitViewMode = .threeD
    private var savedOrbit: OrbitCamera?
    private var reduceMotion = false
    private let gridHalfExtent: Int
    private var didFrameInitially = false
    /// Set by the view: keep drawing frames until the camera settles. Called after every
    /// change the view must show, because the view otherwise draws only while something moves.
    var onNeedsFrames: (() -> Void)?
    /// True while any propeller spins. Its action moves the scene every frame, so the view must
    /// draw continuously at the display rate, exactly as before.
    private(set) var hasSpinningPropellers = false

    init(fixture: Fixture) {
        self.fixture = fixture
        // The decoder guarantees every drone's profile exists, so no drone falls back to a guess.
        var byDrone: [String: DroneDimensions] = [:]
        for drone in fixture.drones {
            if let profile = fixture.profile(id: drone.profileID) { byDrone[drone.id] = DroneDimensions(profile: profile) }
        }
        dimensionsByDrone = byDrone
        factories = Dictionary(Set(byDrone.values).map { ($0, DroneModelFactory(dimensions: $0)) }, uniquingKeysWith: { first, _ in first })
        let points = fixture.frames.first?.drones.map(\.position) ?? []
        let home = OrbitCamera.framed(points: points)
        // The decoder bounds coordinates; the clamp keeps the Int conversion safe regardless.
        let extent = home.swarmRadius.isFinite ? min(home.swarmRadius * 1.5, fixtureCoordinateLimitM) : 10
        gridHalfExtent = max(10, Int(extent.rounded(.up)))
        let camera = SCNCamera()
        camera.fieldOfView = Self.verticalFieldOfView
        camera.zNear = 0.02
        camera.zFar = 160
        cameraNode.camera = camera
        driver = CameraDriver(camera: home, cameraNode: cameraNode)
        sizer = ScreenSizer(verticalFieldOfView: Self.verticalFieldOfView, camera: cameraNode)
        buildStaticScene()
    }

    /// Fits the whole swarm to the current view: centered on the swarm, home yaw and pitch, and
    /// the closest distance at which every drone projects inside the viewport, keeping a margin
    /// of `sceneFitMarginPoints` on every side so the selection ring and the status markers of
    /// the outermost drones stay in frame.
    var homeCamera: OrbitCamera {
        var camera = OrbitCamera.framed(points: currentPositions)
        let margin = Float(LayoutMetrics.sceneFitMarginPoints) * 2
        let fillY = max(0.5, 1 - margin / viewSize.y)
        let fillX = max(0.5, 1 - margin / viewSize.x)
        camera.distance = camera.fitDistance(points: currentPositions, aspect: viewSize.x / viewSize.y,
                                             verticalFOV: Float(Self.verticalFieldOfView), fill: fillY, horizontalFill: fillX)
        camera.clamp()
        return camera
    }

    private var viewSize = SIMD2<Float>(640, 400)
    private var userMovedCamera = false

    /// Keeps the home framing matched to the view until the user orbits, zooms or focuses.
    func viewDidResize(_ size: CGSize) {
        guard size.width > 1, size.height > 1 else { return }
        onNeedsFrames?()
        viewSize = SIMD2(Float(size.width), Float(size.height))
        sizer.viewHeight = Float(size.height)
        updateRingVisibility()
        guard !userMovedCamera, lastViewMode == .threeD else { return }
        let home = homeCamera
        setGoal { $0 = home }
    }

    /// Moves the camera goal and keeps the fog band matched to it.
    private func setGoal(_ change: (inout OrbitCamera) -> Void) {
        driver.update(change)
        updateFog(for: driver.goal)
        updateRingVisibility()
        onNeedsFrames?()
    }

    /// Linear fog that starts just past the far side of the swarm and is nearly opaque at the
    /// far grid edge, so the grid fades into the floor instead of ending in a hard line. It is
    /// measured from the swarm center, so focusing a near drone never fogs the far side.
    private func updateFog(for camera: OrbitCamera) {
        let centerDistance = simd_distance(camera.sceneEye, OrbitCamera.sceneVector(swarmCenter))
        let start = centerDistance + swarmRawRadius
        let end = max(start + 1, centerDistance + Float(gridHalfExtent) * 0.8)
        scene.fogStartDistance = CGFloat(start)
        scene.fogEndDistance = CGFloat(end)
    }

    private var currentPositions: [SIMD3<Float>] = []

    // MARK: Building

    private func buildStaticScene() {
        scene.rootNode.addChildNode(cameraNode)

        let floor = SCNFloor()
        floor.reflectivity = 0
        let floorMaterial = SCNMaterial()
        floorMaterial.lightingModel = .shadowOnly
        floor.materials = [floorMaterial]
        floorNode.geometry = floor
        scene.rootNode.addChildNode(floorNode)
        scene.rootNode.addChildNode(gridNode)

        let key = SCNLight()
        key.type = .directional
        key.intensity = 1100
        key.castsShadow = true
        key.shadowMode = .forward
        key.shadowSampleCount = 16
        key.shadowRadius = 2.5
        key.shadowMapSize = CGSize(width: 4096, height: 4096)
        key.automaticallyAdjustsShadowProjection = true
        key.maximumShadowDistance = 40
        keyLight.light = key
        // Nearly overhead, so each shadow sits under its drone instead of reading as a ghost drone.
        keyLight.eulerAngles = SCNVector3(-CGFloat.pi / 2.2, CGFloat.pi / 5, 0)
        scene.rootNode.addChildNode(keyLight)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 220
        ambient.name = "ambient"
        scene.rootNode.addChildNode(ambient)

        buildTarget()
        scene.rootNode.addChildNode(keepOutRoot)

        // The selection ring circles the drone at its altitude and faces the camera, so it reads
        // as part of the drone, never as a mark on the floor.
        let ring = SCNTorus(ringRadius: 1, pipeRadius: 0.09)
        ring.ringSegmentCount = 64
        ring.pipeSegmentCount = 8
        selectionRingShape.geometry = ring
        selectionRingShape.eulerAngles.x = .pi / 2
        selectionRingShape.castsShadow = false
        selectionRingShape.renderingOrder = 30
        let ringFacing = SCNNode()
        ringFacing.constraints = [SCNBillboardConstraint()]
        ringFacing.addChildNode(selectionRingShape)
        selectionRing.addChildNode(ringFacing)
        selectionRing.name = "selection-ring"
        selectionRing.constraints = [sizer.constraint(points: LayoutMetrics.sceneSelectionRingPoints * 2, baseSize: 2,
                                                      allowsShrink: true)]
        selectionRing.isHidden = true
        scene.rootNode.addChildNode(selectionRing)

        buildDrones()
    }

    private static let targetRingRadius: CGFloat = 0.14

    private func buildTarget() {
        let ring = SCNTorus(ringRadius: Self.targetRingRadius, pipeRadius: 0.012)
        ring.ringSegmentCount = 64
        let ringNode = SCNNode(geometry: ring)
        ringNode.name = "target-ring"
        // Never smaller on screen than `sceneTargetRingPoints` in radius, so the target stays
        // findable next to the keep-out sphere when the whole swarm is in view.
        ringNode.constraints = [sizer.constraint(points: LayoutMetrics.sceneTargetRingPoints * 2,
                                                 baseSize: Self.targetRingRadius * 2, allowsShrink: false)]
        let stem = SCNCylinder(radius: 0.005, height: 1)
        let stemNode = SCNNode(geometry: stem)
        stemNode.name = "target-stem"
        let foot = SCNTorus(ringRadius: 0.07, pipeRadius: 0.004)
        let footNode = SCNNode(geometry: foot)
        footNode.name = "target-foot"
        [ringNode, stemNode, footNode].forEach {
            $0.castsShadow = false
            targetNode.addChildNode($0)
        }
        scene.rootNode.addChildNode(targetNode)
    }

    private func buildDrones() {
        guard let frame = fixture.frames.first else { return }
        for (index, telemetry) in frame.drones.enumerated() {
            guard let dimensions = dimensionsByDrone[telemetry.id], let factory = factories[dimensions] else { continue }
            let root = SCNNode()
            root.name = telemetry.id
            root.simdPosition = OrbitCamera.sceneVector(telemetry.position)
            let body = SCNNode()
            body.castsShadow = true
            root.addChildNode(body)
            var blades: [SCNNode] = []
            for (motorIndex, tip) in factory.motorPositions().enumerated() {
                let blade = SCNNode()
                blade.position = SCNVector3(tip.x, dimensions.propHeight + 0.001, tip.z)
                blade.eulerAngles.y = CGFloat(index * 37 + motorIndex * 53) * .pi / 180
                blade.castsShadow = false
                root.addChildNode(blade)
                blades.append(blade)
            }
            let hit = SCNNode(geometry: hitProxy(for: dimensions))
            hit.name = telemetry.id
            hit.categoryBitMask = Self.hitCategory
            // Not hidden: SceneKit skips hidden nodes in hit tests. Its material writes no color or
            // depth, so it is invisible and never occludes anything.
            hit.castsShadow = false
            hit.position = SCNVector3(0, dimensions.propHeight / 2, 0)
            root.addChildNode(hit)
            // Lifted clear of the propellers, so close up the marker floats above the drone.
            let marker = statusMarker(height: dimensions.propHeight + dimensions.hitRadius * 0.5)
            root.addChildNode(marker)
            scene.rootNode.addChildNode(root)
            drones[telemetry.id] = DroneNode(root: root, body: body, blades: blades, marker: marker, dimensions: dimensions)
        }
    }

    /// A late or lost marker above the drone: the status shape (triangle or square) in the alert
    /// color with a floor-colored outline, a fixed size in points, facing the camera and drawn
    /// over everything so it survives zoom-out, the keep-out sphere and the top view.
    /// Hierarchy: sized holder at the drone top, billboard, then the shape one marker up.
    private func statusMarker(height: CGFloat) -> SCNNode {
        let holder = SCNNode()
        holder.name = "status-marker"
        holder.position = SCNVector3(0, height, 0)
        holder.constraints = [sizer.constraint(points: LayoutMetrics.sceneStatusMarkerPoints, baseSize: 1, allowsShrink: true)]
        let facing = SCNNode()
        facing.constraints = [SCNBillboardConstraint()]
        holder.addChildNode(facing)
        let outline = SCNNode()
        outline.name = "status-marker-outline"
        outline.position = SCNVector3(0, CGFloat(Self.markerLift), 0)
        outline.scale = SCNVector3(1.45, 1.45, 1)
        outline.renderingOrder = 40
        let shape = SCNNode()
        shape.name = "status-marker-shape"
        shape.position = SCNVector3(0, CGFloat(Self.markerLift), 0.001)
        shape.renderingOrder = 41
        facing.addChildNode(outline)
        facing.addChildNode(shape)
        holder.isHidden = true
        holder.castsShadow = false
        outline.castsShadow = false
        shape.castsShadow = false
        return holder
    }

    /// Marker centre above its holder, in marker sizes: just above the drone, or above the
    /// selection ring (`sceneSelectionRingPoints` plus a small gap) on the selected drone.
    static let markerLift: Float = 1.05
    static let selectedMarkerLift = Float((LayoutMetrics.sceneSelectionRingPoints + Space.xs)
                                          / LayoutMetrics.sceneStatusMarkerPoints) + 0.5

    /// Unit-size status shape centred on its origin: a triangle for caution, a square for alert.
    private static func markerPath(level: String) -> NSBezierPath {
        let path = NSBezierPath()
        if level == "caution" {
            path.move(to: NSPoint(x: 0, y: 0.5))
            path.line(to: NSPoint(x: 0.56, y: -0.45))
            path.line(to: NSPoint(x: -0.56, y: -0.45))
            path.close()
        } else {
            path.appendRect(NSRect(x: -0.42, y: -0.42, width: 0.84, height: 0.84))
        }
        return path
    }

    private func markerShape(level: String, color: NSColor) -> SCNGeometry {
        let shape = SCNShape(path: Self.markerPath(level: level), extrusionDepth: 0)
        let material = constant(color)
        material.readsFromDepthBuffer = false
        material.writesToDepthBuffer = false
        material.isDoubleSided = true
        shape.materials = [material]
        return shape
    }

    private func hitProxy(for dimensions: DroneDimensions) -> SCNGeometry {
        if let existing = hitGeometry[dimensions] { return existing }
        let sphere = SCNSphere(radius: dimensions.hitRadius)
        sphere.segmentCount = 12
        let material = SCNMaterial()
        material.colorBufferWriteMask = []
        material.writesToDepthBuffer = false
        material.readsFromDepthBuffer = false
        sphere.materials = [material]
        hitGeometry[dimensions] = sphere
        return sphere
    }

    // MARK: Theme

    private func applyTheme(_ state: SceneState) {
        guard themeKey != state.isDark else { return }
        themeKey = state.isDark
        let palette = state.palette
        let scenePalette = state.scenePalette
        scene.background.contents = scenePalette.floorNS
        scene.fogColor = scenePalette.floorNS
        updateFog(for: driver.goal)
        scene.lightingEnvironment.contents = palette.textSecondaryNS
        scene.lightingEnvironment.intensity = state.isDark ? 0.9 : 1.3
        scene.rootNode.childNode(withName: "ambient", recursively: false)?.light?.intensity = state.isDark ? 220 : 380
        keyLight.light?.shadowColor = (scenePalette.floorNS.shadow(withLevel: state.isDark ? 0.75 : 0.35) ?? scenePalette.floorNS)
            .withAlphaComponent(state.isDark ? 0.7 : 0.2)

        for (dimensions, factory) in factories {
            var looks: [DroneLook: SCNGeometry] = [:]
            for look in DroneLook.allCases {
                looks[look] = factory.staticBody(look: look, palette: palette, scene: scenePalette).geometry
            }
            bodyGeometry[dimensions] = looks
            bladeGeometry[dimensions] = factory.propellerBlades(scene: scenePalette)
        }
        for drone in drones.values {
            drone.body.geometry = bodyGeometry[drone.dimensions]?[drone.look]
            let blades = bladeGeometry[drone.dimensions]
            drone.blades.forEach { $0.geometry = blades }
        }
        gridNode.geometry = nil
        gridNode.childNodes.forEach { $0.removeFromParentNode() }
        gridNode.addChildNode(gridLines(step: 1, skipEvery: 5, color: palette.lineStrongNS.withAlphaComponent(state.isDark ? 0.28 : 0.42)))
        gridNode.addChildNode(gridLines(step: 5, skipEvery: 0, color: palette.lineStrongNS.withAlphaComponent(state.isDark ? 0.6 : 0.72)))

        let targetColor = scenePalette.targetRingNS
        let targetParts: [(String, CGFloat)] = [("target-ring", 1), ("target-stem", 0.55), ("target-foot", 0.7)]
        for (name, alpha) in targetParts {
            let color = alpha < 1 ? targetColor.withAlphaComponent(alpha) : targetColor
            targetNode.childNode(withName: name, recursively: false)?.geometry?.materials = [constant(color)]
        }
        let ringMaterial = constant(palette.accentNS)
        ringMaterial.readsFromDepthBuffer = false
        selectionRingShape.geometry?.materials = [ringMaterial]
        // Red and amber appear only here, as alerts: caution for late, alert for lost.
        markerGeometry = [
            "caution": markerShape(level: "caution", color: palette.cautionNS),
            "alert": markerShape(level: "alert", color: palette.alertNS),
            "caution-outline": markerShape(level: "caution", color: scenePalette.floorNS),
            "alert-outline": markerShape(level: "alert", color: scenePalette.floorNS)
        ]
        for drone in drones.values {
            drone.markerLevel = nil
        }
        keepOutVolumes = []
    }

    private func constant(_ color: NSColor) -> SCNMaterial {
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = color
        if color.alphaComponent < 1 {
            material.transparency = color.alphaComponent
            material.diffuse.contents = color.withAlphaComponent(1)
            material.writesToDepthBuffer = false
        }
        return material
    }

    /// Floor grid with 1 m spacing; the 5 m lines are drawn by a second, stronger pass.
    private func gridLines(step: Int, skipEvery: Int, color: NSColor) -> SCNNode {
        var vertices: [SCNVector3] = []
        let extent = CGFloat(gridHalfExtent)
        for value in stride(from: -gridHalfExtent, through: gridHalfExtent, by: step) {
            if skipEvery > 0 && value % skipEvery == 0 { continue }
            let offset = CGFloat(value)
            vertices.append(SCNVector3(offset, 0, -extent))
            vertices.append(SCNVector3(offset, 0, extent))
            vertices.append(SCNVector3(-extent, 0, offset))
            vertices.append(SCNVector3(extent, 0, offset))
        }
        let source = SCNGeometrySource(vertices: vertices)
        let indices = (0..<Int32(vertices.count)).map { $0 }
        let element = SCNGeometryElement(indices: indices, primitiveType: .line)
        let geometry = SCNGeometry(sources: [source], elements: [element])
        geometry.materials = [constant(color)]
        let node = SCNNode(geometry: geometry)
        node.position = SCNVector3(0, skipEvery == 0 ? 0.0025 : 0.002, 0)
        node.castsShadow = false
        return node
    }

    // MARK: Updates

    func apply(_ state: SceneState) {
        applyTheme(state)
        if reduceMotion != state.reduceMotion {
            reduceMotion = state.reduceMotion
            driver.setReducedMotion(state.reduceMotion)
        }
        currentPositions = state.frame.drones.map(\.position)
        if !didFrameInitially {
            didFrameInitially = true
            let framed = OrbitCamera.framed(points: currentPositions, padding: 1)
            swarmCenter = framed.target
            swarmRawRadius = framed.swarmRadius
            let home = homeCamera
            setGoal { $0 = home }
        }
        updateDrones(state)
        updateTarget(state.frame.targetPosition)
        updateKeepOut(state.frame.keepOutVolumes, visible: state.showKeepOut, state: state)
        updateSelectionRing(state)
        handleViewMode(state.viewMode)
        handle(state.cameraCommand, state: state)
        onNeedsFrames?()
    }

    private func updateDrones(_ state: SceneState) {
        for telemetry in state.frame.drones {
            guard let drone = drones[telemetry.id] else { continue }
            drone.root.simdPosition = OrbitCamera.sceneVector(telemetry.position)
            if let heading = state.headings[telemetry.id] {
                drone.root.eulerAngles.y = CGFloat(heading * .pi / 180)
            }
            let look: DroneLook = telemetry.id == state.selectedID ? .selected : (telemetry.id == state.hoveredID ? .hovered : .normal)
            if drone.look != look || drone.body.geometry == nil {
                drone.look = look
                drone.body.geometry = bodyGeometry[drone.dimensions]?[look]
            }
            // Status is a marker, never opacity: a selected late or lost drone keeps its marker,
            // lifted clear of the selection ring.
            let style = state.styles[telemetry.id]
            let level = style?.hasSceneMarker == true ? style?.level : nil
            let lift = telemetry.id == state.selectedID ? Self.selectedMarkerLift : Self.markerLift
            for part in drone.marker.childNodes.first?.childNodes ?? [] where Float(part.position.y) != lift {
                part.position.y = CGFloat(lift)
            }
            if drone.markerLevel != level || drone.marker.isHidden != (level == nil) {
                drone.markerLevel = level
                drone.marker.isHidden = level == nil
                if let level {
                    drone.marker.childNodes.first?.childNode(withName: "status-marker-shape", recursively: false)?
                        .geometry = markerGeometry[level]
                    drone.marker.childNodes.first?.childNode(withName: "status-marker-outline", recursively: false)?
                        .geometry = markerGeometry["\(level)-outline"]
                }
            }
            let shouldSpin = telemetry.active && !state.reduceMotion
            if shouldSpin != drone.spinning {
                drone.spinning = shouldSpin
                for (index, blade) in drone.blades.enumerated() {
                    if shouldSpin {
                        let direction: CGFloat = index.isMultiple(of: 2) ? 1 : -1
                        blade.runAction(.repeatForever(.rotateBy(x: 0, y: direction * .pi * 2, z: 0, duration: 0.11)), forKey: "spin")
                    } else {
                        blade.removeAction(forKey: "spin")
                    }
                }
            }
        }
        hasSpinningPropellers = drones.values.contains { $0.spinning }
    }

    private func updateTarget(_ position: SIMD3<Float>?) {
        guard let position else {
            targetNode.isHidden = true
            return
        }
        targetNode.isHidden = false
        let scenePosition = OrbitCamera.sceneVector(position)
        targetNode.simdPosition = SIMD3(scenePosition.x, 0, scenePosition.z)
        let height = CGFloat(max(scenePosition.y, 0))
        targetNode.childNode(withName: "target-ring", recursively: false)?.position = SCNVector3(0, height, 0)
        if let stem = targetNode.childNode(withName: "target-stem", recursively: false) {
            (stem.geometry as? SCNCylinder)?.height = max(height - 0.007, 0.001)
            stem.position = SCNVector3(0, height / 2, 0)
            stem.isHidden = height < 0.02
        }
        targetNode.childNode(withName: "target-foot", recursively: false)?.position = SCNVector3(0, 0.003, 0)
    }

    private func updateKeepOut(_ volumes: [KeepOutVolume], visible: Bool, state: SceneState) {
        keepOutRoot.isHidden = !visible
        guard volumes != keepOutVolumes else { return }
        keepOutVolumes = volumes
        keepOutRoot.childNodes.forEach { $0.removeFromParentNode() }
        let fill = state.scenePalette.keepOutVolumeNS
        let edge = state.palette.keepOutNS
        for volume in volumes {
            let radius = CGFloat(volume.radiusM)
            let container = SCNNode()
            container.simdPosition = OrbitCamera.sceneVector(volume.center)
            let sphere = SCNSphere(radius: radius)
            sphere.segmentCount = 64
            let material = constant(fill)
            material.transparencyMode = .singleLayer
            sphere.materials = [material]
            let sphereNode = SCNNode(geometry: sphere)
            sphereNode.castsShadow = false
            sphereNode.renderingOrder = 10
            container.addChildNode(sphereNode)
            container.addChildNode(edgeRing(radius: radius, color: edge.withAlphaComponent(0.55), billboard: false))
            container.addChildNode(edgeRing(radius: radius, color: edge.withAlphaComponent(0.9), billboard: true))
            keepOutRoot.addChildNode(container)
        }
    }

    /// A thin ring of the sphere's radius. The billboarded one always faces the camera, so the
    /// sphere keeps a crisp outline from any angle; the other marks the equator.
    private func edgeRing(radius: CGFloat, color: NSColor, billboard: Bool) -> SCNNode {
        let torus = SCNTorus(ringRadius: radius, pipeRadius: 0.0045)
        torus.ringSegmentCount = 96
        torus.pipeSegmentCount = 8
        torus.materials = [constant(color)]
        let ring = SCNNode(geometry: torus)
        ring.castsShadow = false
        ring.renderingOrder = 11
        guard billboard else { return ring }
        ring.eulerAngles.x = .pi / 2
        let holder = SCNNode()
        holder.addChildNode(ring)
        holder.constraints = [SCNBillboardConstraint()]
        return holder
    }

    private func updateSelectionRing(_ state: SceneState) {
        guard let id = state.selectedID, let telemetry = state.frame.drones.first(where: { $0.id == id }) else {
            selectionRing.isHidden = true
            selectedScenePosition = nil
            return
        }
        let position = OrbitCamera.sceneVector(telemetry.position)
        if let dimensions = dimensionsByDrone[id], dimensions != ringDimensions {
            ringDimensions = dimensions
        }
        let lift = Float((ringDimensions?.propHeight ?? 0) / 2)
        let center = SIMD3(position.x, position.y + lift, position.z)
        selectedScenePosition = center
        selectionRing.simdPosition = center
        updateRingVisibility()
    }

    /// The ring is a fixed `sceneSelectionRingPoints` on screen. Once the camera is close enough
    /// that the drone itself is wider than the ring, it hides: the drone's accent shell marks
    /// the selection, and a ring would only sit on top of the model.
    private func updateRingVisibility() {
        guard let position = selectedScenePosition, let dimensions = ringDimensions else {
            selectionRing.isHidden = true
            return
        }
        let distance = simd_distance(driver.goal.sceneEye, position)
        let pointsPerMeter = 1 / sizer.worldSize(points: 1, distance: distance)
        selectionRing.isHidden = Float(dimensions.hitRadius) * pointsPerMeter > Float(LayoutMetrics.sceneSelectionRingPoints)
    }

    // MARK: Camera

    private func handleViewMode(_ mode: CockpitViewMode) {
        guard mode != lastViewMode else { return }
        lastViewMode = mode
        if mode == .top {
            savedOrbit = driver.goal
            let home = homeCamera
            setGoal { camera in
                camera.target = home.target
                camera.swarmRadius = home.swarmRadius
                camera.distance = self.topDistance(for: home.swarmRadius)
                camera.topView()
            }
        } else {
            let restore = savedOrbit ?? homeCamera
            setGoal { camera in
                camera = restore
                camera.orbitView(pitch: restore.isTopView ? OrbitCamera.homePitch : restore.pitch)
            }
        }
    }

    private func handle(_ command: CameraCommand?, state: SceneState) {
        guard let command, command.serial != lastCommandSerial else { return }
        lastCommandSerial = command.serial
        switch command.kind {
        case .reset:
            userMovedCamera = false
            let home = homeCamera
            setGoal { camera in
                let wasTop = camera.isTopView
                camera = home
                if wasTop {
                    camera.distance = self.topDistance(for: home.swarmRadius)
                    camera.topView()
                }
            }
        case .focus(let droneID):
            userMovedCamera = true
            guard let telemetry = state.frame.drones.first(where: { $0.id == droneID }) else { return }
            setGoal { camera in
                camera.focus(point: telemetry.position)
                camera.distance = camera.isTopView ? 2.5 : 0.9
            }
        }
    }

    func orbit(deltaX: CGFloat, deltaY: CGFloat) {
        userMovedCamera = true
        setGoal { $0.applyDrag(deltaX: Float(-deltaX) * 0.3, deltaY: Float(deltaY) * 0.25) }
    }

    func zoom(factor: CGFloat) {
        userMovedCamera = true
        setGoal { $0.applyZoom(factor: Float(factor)) }
    }
}
