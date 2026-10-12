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

    func setReducedMotion(_ value: Bool) {
        lock.lock()
        reducedMotion = value
        if value { snapNext = true }
        lock.unlock()
    }

    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        lock.lock()
        let elapsed = Float(lastTime.map { min(max(time - $0, 0), 0.1) } ?? 0)
        lastTime = time
        currentCamera = snapNext ? goalCamera : currentCamera.smoothed(toward: goalCamera, elapsed: elapsed)
        snapNext = false
        let pose = currentCamera
        lock.unlock()
        apply(pose)
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
    var look: DroneLook = .normal
    var spinning = false

    init(root: SCNNode, body: SCNNode, blades: [SCNNode]) {
        self.root = root
        self.body = body
        self.blades = blades
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
    private let factory: DroneModelFactory
    private var drones: [String: DroneNode] = [:]
    private var bodyGeometry: [DroneLook: SCNGeometry] = [:]
    private var bladeGeometry: SCNGeometry?
    private let targetNode = SCNNode()
    private let keepOutRoot = SCNNode()
    private let gridNode = SCNNode()
    private let floorNode = SCNNode()
    private let selectionRing = SCNNode()
    private let selectionStem = SCNNode()
    private let keyLight = SCNNode()
    private var keepOutVolumes: [KeepOutVolume] = []
    private var themeKey: Bool?
    private var lastCommandSerial = 0
    private var lastViewMode: CockpitViewMode = .threeD
    private var savedOrbit: OrbitCamera?
    private var reduceMotion = false
    private let gridHalfExtent: Int

    init(fixture: Fixture) {
        self.fixture = fixture
        factory = DroneModelFactory(dimensions: DroneDimensions(profile: fixture.profiles.first))
        let points = fixture.frames.first?.drones.map(\.position) ?? []
        let home = OrbitCamera.framed(points: points)
        gridHalfExtent = max(10, Int((home.swarmRadius * 1.5).rounded(.up)))
        let camera = SCNCamera()
        camera.fieldOfView = Self.verticalFieldOfView
        camera.zNear = 0.02
        camera.zFar = 160
        cameraNode.camera = camera
        driver = CameraDriver(camera: home, cameraNode: cameraNode)
        buildStaticScene()
    }

    var homeCamera: OrbitCamera {
        OrbitCamera.framed(points: currentPositions)
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
        keyLight.eulerAngles = SCNVector3(-CGFloat.pi / 2.8, CGFloat.pi / 5, 0)
        scene.rootNode.addChildNode(keyLight)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 220
        ambient.name = "ambient"
        scene.rootNode.addChildNode(ambient)

        buildTarget()
        scene.rootNode.addChildNode(keepOutRoot)

        let ring = SCNTorus(ringRadius: factory.dimensions.hitRadius * 1.1, pipeRadius: 0.006)
        ring.ringSegmentCount = 64
        selectionRing.geometry = ring
        selectionRing.castsShadow = false
        selectionRing.isHidden = true
        scene.rootNode.addChildNode(selectionRing)
        selectionStem.geometry = SCNCylinder(radius: 0.0015, height: 1)
        selectionStem.castsShadow = false
        selectionStem.isHidden = true
        scene.rootNode.addChildNode(selectionStem)

        buildDrones()
    }

    private func buildTarget() {
        let ring = SCNTorus(ringRadius: 0.12, pipeRadius: 0.007)
        ring.ringSegmentCount = 64
        let ringNode = SCNNode(geometry: ring)
        ringNode.name = "target-ring"
        let stem = SCNCylinder(radius: 0.0035, height: 1)
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
            let root = SCNNode()
            root.name = telemetry.id
            root.simdPosition = OrbitCamera.sceneVector(telemetry.position)
            let body = SCNNode()
            body.castsShadow = true
            root.addChildNode(body)
            var blades: [SCNNode] = []
            for (motorIndex, tip) in factory.motorPositions().enumerated() {
                let blade = SCNNode()
                blade.position = SCNVector3(tip.x, factory.dimensions.propHeight + 0.001, tip.z)
                blade.eulerAngles.y = CGFloat(index * 37 + motorIndex * 53) * .pi / 180
                blade.castsShadow = false
                root.addChildNode(blade)
                blades.append(blade)
            }
            let hit = SCNNode(geometry: hitGeometry)
            hit.name = telemetry.id
            hit.categoryBitMask = Self.hitCategory
            // Not hidden: SceneKit skips hidden nodes in hit tests. Its material writes no color or
            // depth, so it is invisible and never occludes anything.
            hit.castsShadow = false
            hit.position = SCNVector3(0, factory.dimensions.propHeight / 2, 0)
            root.addChildNode(hit)
            scene.rootNode.addChildNode(root)
            drones[telemetry.id] = DroneNode(root: root, body: body, blades: blades)
        }
    }

    private lazy var hitGeometry: SCNGeometry = {
        let sphere = SCNSphere(radius: factory.dimensions.hitRadius)
        sphere.segmentCount = 12
        let material = SCNMaterial()
        material.colorBufferWriteMask = []
        material.writesToDepthBuffer = false
        material.readsFromDepthBuffer = false
        sphere.materials = [material]
        return sphere
    }()

    // MARK: Theme

    private func applyTheme(_ state: SceneState) {
        guard themeKey != state.isDark else { return }
        themeKey = state.isDark
        let palette = state.palette
        let scenePalette = state.scenePalette
        scene.background.contents = scenePalette.floorNS
        scene.fogColor = scenePalette.floorNS
        scene.fogStartDistance = 30
        scene.fogEndDistance = 70
        scene.lightingEnvironment.contents = palette.textSecondaryNS
        scene.lightingEnvironment.intensity = state.isDark ? 0.9 : 1.3
        scene.rootNode.childNode(withName: "ambient", recursively: false)?.light?.intensity = state.isDark ? 220 : 380
        keyLight.light?.shadowColor = (scenePalette.floorNS.shadow(withLevel: state.isDark ? 0.75 : 0.35) ?? scenePalette.floorNS)
            .withAlphaComponent(state.isDark ? 0.8 : 0.3)

        for look in DroneLook.allCases {
            bodyGeometry[look] = factory.staticBody(look: look, palette: palette, scene: scenePalette).geometry
        }
        bladeGeometry = factory.propellerBlades(scene: scenePalette)
        for drone in drones.values {
            drone.body.geometry = bodyGeometry[drone.look]
            drone.blades.forEach { $0.geometry = bladeGeometry }
        }
        gridNode.geometry = nil
        gridNode.childNodes.forEach { $0.removeFromParentNode() }
        gridNode.addChildNode(gridLines(step: 1, skipEvery: 5, color: palette.lineStrongNS.withAlphaComponent(state.isDark ? 0.28 : 0.3)))
        gridNode.addChildNode(gridLines(step: 5, skipEvery: 0, color: palette.lineStrongNS.withAlphaComponent(state.isDark ? 0.6 : 0.6)))

        let targetColor = scenePalette.targetRingNS
        let targetParts: [(String, CGFloat)] = [("target-ring", 1), ("target-stem", 0.55), ("target-foot", 0.7)]
        for (name, alpha) in targetParts {
            let color = alpha < 1 ? targetColor.withAlphaComponent(alpha) : targetColor
            targetNode.childNode(withName: name, recursively: false)?.geometry?.materials = [constant(color)]
        }
        selectionRing.geometry?.materials = [constant(palette.accentNS)]
        selectionStem.geometry?.materials = [constant(palette.accentNS.withAlphaComponent(0.35))]
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
        updateDrones(state)
        updateTarget(state.frame.targetPosition)
        updateKeepOut(state.frame.keepOutVolumes, visible: state.showKeepOut, state: state)
        updateSelectionRing(state)
        handleViewMode(state.viewMode)
        handle(state.cameraCommand, state: state)
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
                drone.body.geometry = bodyGeometry[look]
            }
            let faded = state.styles[telemetry.id]?.fadesInScene ?? false
            drone.root.opacity = faded && look != .selected ? 0.38 : 1
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
            selectionStem.isHidden = true
            return
        }
        let position = OrbitCamera.sceneVector(telemetry.position)
        selectionRing.isHidden = false
        // In top view the ring sits at drone height so it does not drift by parallax.
        let ringHeight: Float = state.viewMode == .top ? position.y + 0.01 : 0.004
        selectionRing.simdPosition = SIMD3(position.x, ringHeight, position.z)
        let height = max(CGFloat(position.y) - 0.02, 0.001)
        (selectionStem.geometry as? SCNCylinder)?.height = height
        selectionStem.simdPosition = SIMD3(position.x, Float(height / 2), position.z)
        selectionStem.isHidden = height < 0.05 || state.viewMode == .top
    }

    // MARK: Camera

    private func handleViewMode(_ mode: CockpitViewMode) {
        guard mode != lastViewMode else { return }
        lastViewMode = mode
        if mode == .top {
            savedOrbit = driver.goal
            let home = homeCamera
            driver.update { camera in
                camera.target = home.target
                camera.swarmRadius = home.swarmRadius
                camera.distance = self.topDistance(for: home.swarmRadius)
                camera.topView()
            }
        } else {
            let restore = savedOrbit ?? homeCamera
            driver.update { camera in
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
            let home = homeCamera
            driver.update { camera in
                let wasTop = camera.isTopView
                camera = home
                if wasTop {
                    camera.distance = self.topDistance(for: home.swarmRadius)
                    camera.topView()
                }
            }
        case .focus(let droneID):
            guard let telemetry = state.frame.drones.first(where: { $0.id == droneID }) else { return }
            driver.update { camera in
                camera.focus(point: telemetry.position)
                camera.distance = camera.isTopView ? 2.5 : 0.9
            }
        }
    }

    func orbit(deltaX: CGFloat, deltaY: CGFloat) {
        driver.update { $0.applyDrag(deltaX: Float(-deltaX) * 0.3, deltaY: Float(deltaY) * 0.25) }
    }

    func zoom(factor: CGFloat) {
        driver.update { $0.applyZoom(factor: Float(factor)) }
    }
}
