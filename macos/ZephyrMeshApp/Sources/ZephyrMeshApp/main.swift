import SwiftUI
import SceneKit

@main
struct ZephyrMeshApp: App {
    @StateObject private var model = CockpitModel()

    var body: some Scene {
        WindowGroup("Zephyr Mesh") {
            CockpitView(model: model)
                .frame(minWidth: 1120, minHeight: 720)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Reset camera") { model.resetCamera.toggle() }
                    .keyboardShortcut("0", modifiers: [])
                Button("Play / pause") { model.isPlaying.toggle() }
                    .keyboardShortcut(.space, modifiers: [])
            }
        }
    }
}

struct DroneTelemetry: Identifiable, Codable {
    let id: String
    let name: String
    let role: String
    let color: String
    var position: SIMD3<Float>
    var velocity: SIMD3<Float>
    var battery: Double
    var linkDelayMs: Double
    var linkLossPct: Double
    var state: String
    var confidence: Double

    enum CodingKeys: String, CodingKey { case id, name, role, color, position, velocity, battery, linkDelayMs = "link_delay_ms", linkLossPct = "link_loss_pct", state, confidence }
}

struct Fixture: Codable {
    let schema: String
    let runID: String
    let source: String
    let seed: Int
    let model: String
    let unavailable: [String]
    let commandAuthority: String
    let failsafe: String
    var drones: [DroneTelemetry]
    var frames: [FixtureFrame]

    enum CodingKeys: String, CodingKey { case schema, runID = "run_id", source, seed, model, unavailable, commandAuthority = "command_authority", failsafe, drones, frames }

    init(schema: String, runID: String, source: String, seed: Int, model: String, unavailable: [String], commandAuthority: String = "replay only", failsafe: String = "unavailable without a live adapter", drones: [DroneTelemetry], frames: [FixtureFrame] = []) {
        self.schema = schema; self.runID = runID; self.source = source; self.seed = seed; self.model = model; self.unavailable = unavailable; self.commandAuthority = commandAuthority; self.failsafe = failsafe; self.drones = drones; self.frames = frames
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schema = try values.decode(String.self, forKey: .schema)
        runID = try values.decode(String.self, forKey: .runID)
        source = try values.decode(String.self, forKey: .source)
        seed = try values.decode(Int.self, forKey: .seed)
        model = try values.decode(String.self, forKey: .model)
        unavailable = try values.decode([String].self, forKey: .unavailable)
        commandAuthority = try values.decodeIfPresent(String.self, forKey: .commandAuthority) ?? "replay only"
        failsafe = try values.decodeIfPresent(String.self, forKey: .failsafe) ?? "unavailable without a live adapter"
        drones = try values.decode([DroneTelemetry].self, forKey: .drones)
        frames = try values.decodeIfPresent([FixtureFrame].self, forKey: .frames) ?? []
    }
}

struct FixtureFrame: Codable {
    let time: Double
    let drones: [DroneTelemetry]
}

@MainActor
final class CockpitModel: ObservableObject {
    @Published var fixture: Fixture
    @Published var selectedID = "D1"
    @Published var isPlaying = true
    @Published var time: Double = 0
    @Published var resetCamera = false
    @Published var showObstacles = true
    @Published var selectedTab = "Scene"

    private var timer: Timer?
    private var basePositions: [String: SIMD3<Float>] = [:]
    private var frameIndex = 0

    init() {
        fixture = FixtureLoader.load() ?? Fixture(
            schema: "zephyr-s7-native-fixture-1", runID: "local-fallback", source: "synthetic", seed: 17,
            model: "multi-agent kinematic preview", unavailable: ["live radio", "camera", "flight performance"], drones: FixtureLoader.fallbackDrones())
        selectedID = fixture.drones.first?.id ?? "D1"
        basePositions = Dictionary(uniqueKeysWithValues: fixture.drones.map { ($0.id, $0.position) })
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in self.tick() }
        }
    }

    var selected: DroneTelemetry? { fixture.drones.first(where: { $0.id == selectedID }) }

    func tick() {
        guard isPlaying else { return }
        advanceOneFrame()
    }

    func stepFrame() {
        isPlaying = false
        advanceOneFrame()
    }

    private func advanceOneFrame() {
        if !fixture.frames.isEmpty {
            frameIndex = min(frameIndex + 1, fixture.frames.count - 1)
            fixture.drones = fixture.frames[frameIndex].drones
            time = fixture.frames[frameIndex].time
            if frameIndex == fixture.frames.count - 1 { isPlaying = false }
            return
        }
        time += 1.0 / 30.0
    }

    func reset() {
        time = 0
        frameIndex = 0
        if let first = fixture.frames.first { fixture.drones = first.drones }
        else { fixture.drones = fixture.drones.map { drone in var value = drone; value.position = basePositions[drone.id] ?? value.position; value.velocity = .zero; return value } }
    }
}

enum FixtureLoader {
    static func load() -> Fixture? {
        guard let url = Bundle.module.url(forResource: "demo_swarm", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let fixture = try? JSONDecoder().decode(Fixture.self, from: data) else { return nil }
        return fixture
    }

    static func fallbackDrones() -> [DroneTelemetry] {
        var drones: [DroneTelemetry] = []
        for index in 0..<6 {
            let id = "D\(index + 1)"
            let position = SIMD3<Float>(Float(index % 3) * 1.4 - 1.4, Float(index / 3) * 1.3 - 0.65, 1.6)
            let isLead = index == 0
            let isWeak = index == 4
            drones.append(DroneTelemetry(id: id, name: "Scout \(index + 1)", role: isLead ? "lead" : "cooperative", color: isLead ? "sky" : "aurora", position: position, velocity: .zero, battery: 86 - Double(index * 4), linkDelayMs: 18 + Double(index * 3), linkLossPct: isWeak ? 9.0 : 1.0, state: isWeak ? "Holding" : "Flying", confidence: 0.94 - Double(index) * 0.02))
        }
        return drones
    }
}

struct CockpitView: View {
    @ObservedObject var model: CockpitModel
    private let horizon = Horizon()

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider().overlay(horizon.line)
            VStack(spacing: 0) {
                topBar
                ZStack(alignment: .bottom) {
                    SceneViewRepresentable(model: model)
                    timeline
                }
            }
            .background(horizon.night)
        }
        .background(horizon.night)
        .foregroundStyle(horizon.cloud)
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Text("Z").font(.system(size: 22, weight: .black, design: .rounded)).foregroundStyle(horizon.night).frame(width: 38, height: 38).background(horizon.sky, in: RoundedRectangle(cornerRadius: 11))
                VStack(alignment: .leading, spacing: 2) { Text("ZEPHYR MESH").font(.system(size: 12, weight: .bold, design: .rounded)); Text("3D SIMULATOR").font(.system(size: 10, weight: .medium, design: .rounded)).foregroundStyle(horizon.mist) }
            }
            .padding(.bottom, 24)
            Text("FLEET / \(model.fixture.drones.count)").font(.system(size: 11, weight: .bold, design: .rounded)).foregroundStyle(horizon.mist).tracking(1.1)
            ScrollView {
                VStack(spacing: 8) {
                    ForEach(model.fixture.drones) { drone in FleetRow(drone: drone, isSelected: drone.id == model.selectedID, horizon: horizon) { model.selectedID = drone.id } }
                }
                .padding(.vertical, 10)
            }
            Spacer()
            VStack(alignment: .leading, spacing: 10) {
                Text("SAFETY GATE").font(.system(size: 10, weight: .bold, design: .rounded)).foregroundStyle(horizon.mist).tracking(1.1)
                Label("Keep-out constraints in fixture", systemImage: "checkmark.shield.fill").foregroundStyle(horizon.aurora)
                Label("Authority: \(model.fixture.commandAuthority)", systemImage: "pause.circle.fill").foregroundStyle(horizon.apricot)
                Label("Failsafe: \(model.fixture.failsafe)", systemImage: "exclamationmark.triangle.fill").foregroundStyle(horizon.apricot)
                Text("Replay surface only. Hardware authority is unavailable in this build.").font(.system(size: 10)).foregroundStyle(horizon.mist).fixedSize(horizontal: false, vertical: true)
            }
            .padding(14)
            .background(horizon.surface, in: RoundedRectangle(cornerRadius: 14))
        }
        .padding(22)
        .frame(width: 250)
        .background(horizon.night)
    }

    private var topBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) { Text("COOPERATIVE TEST RANGE").font(.system(size: 11, weight: .bold, design: .rounded)).foregroundStyle(horizon.sky).tracking(1.2); Text("Fleet replay / \(model.fixture.runID)").font(.system(size: 18, weight: .semibold, design: .rounded)) }
            Spacer()
            HStack(spacing: 8) {
                Tag(text: "S7 FOUNDATION", color: horizon.lavender)
                Tag(text: model.isPlaying ? "● RUNNING" : "Ⅱ PAUSED", color: model.isPlaying ? horizon.aurora : horizon.apricot)
                Button { model.isPlaying = false; model.reset() } label: { Label("Reset", systemImage: "arrow.counterclockwise") }.buttonStyle(.bordered)
            }
        }
        .padding(.horizontal, 24).padding(.vertical, 18)
        .background(horizon.surface.opacity(0.66))
    }

    private var timeline: some View {
        VStack(spacing: 10) {
            HStack { Text("T+\(model.time, specifier: "%.1f") s").font(.system(size: 12, weight: .semibold, design: .monospaced)); Spacer(); Text("SIMULATED · SEE PROVENANCE").font(.system(size: 10, weight: .bold, design: .rounded)).foregroundStyle(horizon.mist) }
            ProgressView(value: model.fixture.frames.isEmpty ? 0 : min(model.time / max(model.fixture.frames.last?.time ?? 1, 1e-6), 1)).tint(horizon.sky)
            HStack { Button(model.isPlaying ? "Pause" : "Play") { model.isPlaying.toggle() }.keyboardShortcut(.space, modifiers: []); Button("Step") { model.stepFrame() }; Spacer(); Toggle("Obstacles", isOn: $model.showObstacles) }
        }
        .padding(16)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(horizon.line, lineWidth: 1))
        .padding(18)
    }
}

struct FleetRow: View {
    let drone: DroneTelemetry
    let isSelected: Bool
    let horizon: Horizon
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Circle().fill(drone.linkLossPct > 8 ? horizon.apricot : horizon.aurora).frame(width: 8, height: 8)
                VStack(alignment: .leading, spacing: 3) { Text(drone.id).font(.system(size: 13, weight: .bold, design: .rounded)); Text(drone.state).font(.system(size: 10)).foregroundStyle(horizon.mist) }
                Spacer()
                VStack(alignment: .trailing, spacing: 3) { Text("\(Int(drone.battery))%").font(.system(size: 11, weight: .semibold, design: .monospaced)); Text("\(Int(drone.linkDelayMs)) ms").font(.system(size: 9, design: .monospaced)).foregroundStyle(drone.linkLossPct > 8 ? horizon.apricot : horizon.mist) }
            }
            .padding(11)
            .background(isSelected ? horizon.sky.opacity(0.16) : horizon.surface.opacity(0.55), in: RoundedRectangle(cornerRadius: 11))
            .overlay(RoundedRectangle(cornerRadius: 11).stroke(isSelected ? horizon.sky.opacity(0.8) : horizon.line, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(drone.id), \(drone.state), \(Int(drone.battery)) percent battery, \(Int(drone.linkDelayMs)) milliseconds link delay")
    }
}

struct Tag: View { let text: String; let color: Color; var body: some View { Text(text).font(.system(size: 9, weight: .bold, design: .rounded)).foregroundStyle(color).padding(.horizontal, 9).padding(.vertical, 6).background(color.opacity(0.12), in: Capsule()) } }

struct Horizon { let night = Color(hex: 0x0F1522); let surface = Color(hex: 0x172033); let line = Color(hex: 0x2B3A55); let cloud = Color(hex: 0xEEF2F7); let mist = Color(hex: 0x9AA8C0); let sky = Color(hex: 0x7CC4FF); let aurora = Color(hex: 0x6EE7C8); let apricot = Color(hex: 0xFFB38A); let lavender = Color(hex: 0xB4A7FF) }
extension Color { init(hex: UInt, alpha: Double = 1) { self.init(.sRGB, red: Double((hex >> 16) & 0xFF)/255, green: Double((hex >> 8) & 0xFF)/255, blue: Double(hex & 0xFF)/255, opacity: alpha) } }

struct SceneViewRepresentable: NSViewRepresentable {
    @ObservedObject var model: CockpitModel
    func makeNSView(context: Context) -> SCNView { let view = SCNView(); view.scene = SceneBuilder.build(model: model); view.allowsCameraControl = true; view.backgroundColor = NSColor(red: 0.04, green: 0.07, blue: 0.12, alpha: 1); view.antialiasingMode = .multisampling4X; view.rendersContinuously = true; return view }
    func updateNSView(_ view: SCNView, context: Context) { SceneBuilder.update(view: view, model: model) }
}

@MainActor
enum SceneBuilder {
    static let dronePrefix = "drone-"
    static func build(model: CockpitModel) -> SCNScene {
        let scene = SCNScene(); scene.rootNode.addChildNode(floor()); scene.rootNode.addChildNode(camera()); scene.rootNode.addChildNode(light());
        for drone in model.fixture.drones { scene.rootNode.addChildNode(droneNode(drone)) }
        if model.showObstacles { scene.rootNode.addChildNode(obstacle(position: SCNVector3(0, 0.8, -1.5), scale: SCNVector3(1.3, 1.6, 1.3))) }
        return scene
    }
    static func update(view: SCNView, model: CockpitModel) {
        for drone in model.fixture.drones { guard let node = view.scene?.rootNode.childNode(withName: dronePrefix + drone.id, recursively: true) else { continue }; node.position = SCNVector3(drone.position.x, drone.position.z, drone.position.y); node.opacity = drone.linkLossPct > 8 ? 0.62 : 1.0; node.childNodes.first?.geometry?.firstMaterial?.emission.contents = drone.id == model.selectedID ? NSColor(calibratedRed: 0.49, green: 0.77, blue: 1, alpha: 1) : NSColor.clear }
        view.scene?.rootNode.childNodes.filter { $0.name == "obstacle" }.forEach { $0.isHidden = !model.showObstacles }
    }
    static func droneNode(_ drone: DroneTelemetry) -> SCNNode { let group = SCNNode(); group.name = dronePrefix + drone.id; group.position = SCNVector3(drone.position.x, drone.position.z, drone.position.y); let body = SCNNode(geometry: SCNBox(width: 0.42, height: 0.12, length: 0.30, chamferRadius: 0.06)); body.geometry?.firstMaterial = material(NSColor(calibratedRed: 0.22, green: 0.43, blue: 0.63, alpha: 1)); group.addChildNode(body); for x in [-0.42, 0.42] { for z in [-0.28, 0.28] { let arm = SCNCylinder(radius: 0.035, height: 0.54); arm.firstMaterial = material(NSColor(calibratedRed: 0.43, green: 0.78, blue: 0.73, alpha: 1)); let n = SCNNode(geometry: arm); n.eulerAngles = SCNVector3(0, 0, Float.pi / 2); n.position = SCNVector3(Float(x) * 0.5, 0, Float(z) * 0.5); group.addChildNode(n); let rotor = SCNCylinder(radius: 0.13, height: 0.025); rotor.firstMaterial = material(NSColor(calibratedWhite: 0.8, alpha: 0.5)); let r = SCNNode(geometry: rotor); r.position = SCNVector3(Float(x), 0.1, Float(z)); group.addChildNode(r) } }; return group }
    static func floor() -> SCNNode { let node = SCNNode(geometry: SCNPlane(width: 12, height: 12)); node.geometry?.firstMaterial = material(NSColor(calibratedRed: 0.07, green: 0.11, blue: 0.17, alpha: 1)); node.eulerAngles.x = -.pi / 2; return node }
    static func camera() -> SCNNode { let node = SCNNode(); node.name = "camera"; node.camera = SCNCamera(); node.camera?.fieldOfView = 54; node.position = SCNVector3(7, 6, 8); node.look(at: SCNVector3(0, 1.1, 0)); return node }
    static func light() -> SCNNode { let node = SCNNode(); node.light = SCNLight(); node.light?.type = .omni; node.light?.intensity = 1100; node.position = SCNVector3(2, 7, 3); return node }
    static func obstacle(position: SCNVector3, scale: SCNVector3) -> SCNNode { let node = SCNNode(geometry: SCNBox(width: 1, height: 1, length: 1, chamferRadius: 0.04)); node.name = "obstacle"; node.position = position; node.scale = scale; node.geometry?.firstMaterial = material(NSColor(calibratedRed: 1, green: 0.7, blue: 0.45, alpha: 0.28)); return node }
    static func material(_ color: NSColor) -> SCNMaterial { let m = SCNMaterial(); m.diffuse.contents = color; m.roughness.contents = 0.78; return m }
}
