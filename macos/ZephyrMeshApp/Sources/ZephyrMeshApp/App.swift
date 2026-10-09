import SwiftUI
import SceneKit
import AppKit

@main
struct ZephyrMeshApp: App {
    @StateObject private var model = CockpitModel()
    @StateObject private var evidenceInspector = EvidenceInspectorController()

    var body: some Scene {
        WindowGroup("Zephyr Mesh") {
            CockpitView(model: model)
                .frame(minWidth: 1380, minHeight: 820)
                .sheet(isPresented: $evidenceInspector.isPresented) {
                    if let report = evidenceInspector.report {
                        EvidenceInspectorView(report: report, filename: evidenceInspector.reportFilename)
                    }
                }
                .alert(item: $evidenceInspector.error) { message in
                    Alert(title: Text("Could not open evidence report"), message: Text(message.message), dismissButton: .default(Text("OK")))
                }
        }
        .commands {
            CommandGroup(after: .newItem) {
                Button("Play / pause") { model.isPlaying.toggle() }
                    .keyboardShortcut(.space, modifiers: [])
                Button("Step replay") { model.stepFrame() }
                    .keyboardShortcut(.rightArrow, modifiers: [])
                Button("Check for updates") { model.checkForUpdates() }
                Button("Open evidence report…") { evidenceInspector.openPanel() }
                    .keyboardShortcut("o", modifiers: [.command])
                Button("Reset camera") { model.resetCamera() }
            }
        }
    }
}

// MARK: - Canonical S7 projection

/// The native surface mirrors the compact projection generated from the
/// canonical Python S7 run. Optional fields let newer projections add agent
/// diagnostics without breaking an older checked-in replay.
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
    var profileID: String
    var active: Bool
    var neighborCount: Int
    var minNeighborDistance: Double
    var constraintFlags: [String]
    var targetEstimate: SIMD3<Float>?
    var fusedTarget: SIMD3<Float>?
    var targetSourceTime: Double?
    var targetAge: Double?

    enum CodingKeys: String, CodingKey {
        case id, name, role, color, position, velocity, battery
        case linkDelayMs = "link_delay_ms"
        case linkLossPct = "link_loss_pct"
        case state, confidence
        case profileID = "profile_id"
        case active
        case neighborCount = "neighbor_count"
        case minNeighborDistance = "min_neighbor_distance_m"
        case constraintFlags = "constraint_flags"
        case targetEstimate = "target_estimate_m"
        case fusedTarget = "fused_target_m"
        case targetSourceTime = "target_source_time_s"
        case targetAge = "target_age_s"
    }

    init(id: String, name: String, role: String, color: String,
         position: SIMD3<Float>, velocity: SIMD3<Float>, battery: Double,
         linkDelayMs: Double, linkLossPct: Double, state: String,
         confidence: Double, profileID: String = "synthetic-profile",
         active: Bool = true, neighborCount: Int = 0,
         minNeighborDistance: Double = 0, constraintFlags: [String] = [],
         targetEstimate: SIMD3<Float>? = nil, fusedTarget: SIMD3<Float>? = nil,
         targetSourceTime: Double? = nil, targetAge: Double? = nil) {
        self.id = id; self.name = name; self.role = role; self.color = color
        self.position = position; self.velocity = velocity; self.battery = battery
        self.linkDelayMs = linkDelayMs; self.linkLossPct = linkLossPct
        self.state = state; self.confidence = confidence; self.profileID = profileID
        self.active = active; self.neighborCount = neighborCount
        self.minNeighborDistance = minNeighborDistance
        self.constraintFlags = constraintFlags; self.targetEstimate = targetEstimate
        self.fusedTarget = fusedTarget; self.targetSourceTime = targetSourceTime
        self.targetAge = targetAge
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(String.self, forKey: .id)
        name = try values.decodeIfPresent(String.self, forKey: .name) ?? id
        role = try values.decodeIfPresent(String.self, forKey: .role) ?? "cooperative"
        color = try values.decodeIfPresent(String.self, forKey: .color) ?? "aurora"
        position = try values.decode(SIMD3<Float>.self, forKey: .position)
        velocity = try values.decodeIfPresent(SIMD3<Float>.self, forKey: .velocity) ?? .zero
        battery = try values.decodeIfPresent(Double.self, forKey: .battery) ?? 0
        linkDelayMs = try values.decodeIfPresent(Double.self, forKey: .linkDelayMs) ?? 0
        linkLossPct = try values.decodeIfPresent(Double.self, forKey: .linkLossPct) ?? 0
        state = try values.decodeIfPresent(String.self, forKey: .state) ?? "Unknown"
        confidence = try values.decodeIfPresent(Double.self, forKey: .confidence) ?? 0
        profileID = try values.decodeIfPresent(String.self, forKey: .profileID) ?? "unidentified"
        active = try values.decodeIfPresent(Bool.self, forKey: .active) ?? true
        neighborCount = try values.decodeIfPresent(Int.self, forKey: .neighborCount) ?? 0
        minNeighborDistance = try values.decodeIfPresent(Double.self, forKey: .minNeighborDistance) ?? 0
        constraintFlags = try values.decodeIfPresent([String].self, forKey: .constraintFlags) ?? []
        targetEstimate = try? values.decode(SIMD3<Float>.self, forKey: .targetEstimate)
        fusedTarget = try? values.decode(SIMD3<Float>.self, forKey: .fusedTarget)
        targetSourceTime = try values.decodeIfPresent(Double.self, forKey: .targetSourceTime)
        targetAge = try values.decodeIfPresent(Double.self, forKey: .targetAge)
    }

    func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(id, forKey: .id); try values.encode(name, forKey: .name)
        try values.encode(role, forKey: .role); try values.encode(color, forKey: .color)
        try values.encode(position, forKey: .position); try values.encode(velocity, forKey: .velocity)
        try values.encode(battery, forKey: .battery); try values.encode(linkDelayMs, forKey: .linkDelayMs)
        try values.encode(linkLossPct, forKey: .linkLossPct); try values.encode(state, forKey: .state)
        try values.encode(confidence, forKey: .confidence); try values.encode(profileID, forKey: .profileID)
        try values.encode(active, forKey: .active); try values.encode(neighborCount, forKey: .neighborCount)
        try values.encode(minNeighborDistance, forKey: .minNeighborDistance)
        try values.encode(constraintFlags, forKey: .constraintFlags)
        try values.encodeIfPresent(targetEstimate, forKey: .targetEstimate)
        try values.encodeIfPresent(fusedTarget, forKey: .fusedTarget)
        try values.encodeIfPresent(targetSourceTime, forKey: .targetSourceTime)
        try values.encodeIfPresent(targetAge, forKey: .targetAge)
    }
}

struct FixtureFrame: Codable {
    let stepIndex: Int
    let time: Double
    let targetPosition: SIMD3<Float>?
    let activeCount: Int
    let drones: [DroneTelemetry]

    enum CodingKeys: String, CodingKey {
        case stepIndex = "step_index", time, targetPosition = "target_position"
        case activeCount = "active_count", drones
    }

    init(stepIndex: Int = 0, time: Double, targetPosition: SIMD3<Float>? = nil,
         activeCount: Int = 0, drones: [DroneTelemetry]) {
        self.stepIndex = stepIndex; self.time = time; self.targetPosition = targetPosition
        self.activeCount = activeCount == 0 ? drones.filter(\.active).count : activeCount
        self.drones = drones
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        stepIndex = try values.decodeIfPresent(Int.self, forKey: .stepIndex) ?? 0
        time = try values.decode(Double.self, forKey: .time)
        targetPosition = try? values.decode(SIMD3<Float>.self, forKey: .targetPosition)
        drones = try values.decode([DroneTelemetry].self, forKey: .drones)
        activeCount = try values.decodeIfPresent(Int.self, forKey: .activeCount) ?? drones.filter(\.active).count
    }

    func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(stepIndex, forKey: .stepIndex); try values.encode(time, forKey: .time)
        try values.encodeIfPresent(targetPosition, forKey: .targetPosition)
        try values.encode(activeCount, forKey: .activeCount); try values.encode(drones, forKey: .drones)
    }
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
    let dt: Double
    let evidenceBoundary: String
    let generationCommand: String
    let status: String
    var drones: [DroneTelemetry]
    var frames: [FixtureFrame]

    enum CodingKeys: String, CodingKey {
        case schema, runID = "run_id", source, seed, model, unavailable
        case commandAuthority = "command_authority", failsafe, dt = "dt_s"
        case evidenceBoundary = "evidence_boundary", generationCommand = "generation_command"
        case status, drones, frames
    }

    init(schema: String, runID: String, source: String, seed: Int, model: String,
         unavailable: [String], commandAuthority: String = "replay only",
         failsafe: String = "unavailable without a live adapter", dt: Double = 0.05,
         evidenceBoundary: String = "Synthetic replay projection; not flight performance.",
         generationCommand: String = "tools/generate_s7_fixture.py --output runs/s7-swarm",
         status: String = "synthetic deterministic replay", drones: [DroneTelemetry],
         frames: [FixtureFrame] = []) {
        self.schema = schema; self.runID = runID; self.source = source; self.seed = seed
        self.model = model; self.unavailable = unavailable; self.commandAuthority = commandAuthority
        self.failsafe = failsafe; self.dt = dt; self.evidenceBoundary = evidenceBoundary
        self.generationCommand = generationCommand; self.status = status; self.drones = drones; self.frames = frames
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schema = try values.decode(String.self, forKey: .schema)
        runID = try values.decode(String.self, forKey: .runID)
        source = try values.decodeIfPresent(String.self, forKey: .source) ?? "unknown source"
        seed = try values.decodeIfPresent(Int.self, forKey: .seed) ?? 0
        model = try values.decodeIfPresent(String.self, forKey: .model) ?? "unspecified model"
        unavailable = try values.decodeIfPresent([String].self, forKey: .unavailable) ?? []
        commandAuthority = try values.decodeIfPresent(String.self, forKey: .commandAuthority) ?? "replay only"
        failsafe = try values.decodeIfPresent(String.self, forKey: .failsafe) ?? "unavailable without a live adapter"
        dt = try values.decodeIfPresent(Double.self, forKey: .dt) ?? 0.05
        evidenceBoundary = try values.decodeIfPresent(String.self, forKey: .evidenceBoundary) ?? "Synthetic replay projection; not flight performance."
        generationCommand = try values.decodeIfPresent(String.self, forKey: .generationCommand) ?? "tools/generate_s7_fixture.py --output runs/s7-swarm"
        status = try values.decodeIfPresent(String.self, forKey: .status) ?? "synthetic deterministic replay"
        drones = try values.decodeIfPresent([DroneTelemetry].self, forKey: .drones) ?? []
        frames = try values.decodeIfPresent([FixtureFrame].self, forKey: .frames) ?? []
    }

    func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(schema, forKey: .schema); try values.encode(runID, forKey: .runID)
        try values.encode(source, forKey: .source); try values.encode(seed, forKey: .seed)
        try values.encode(model, forKey: .model); try values.encode(unavailable, forKey: .unavailable)
        try values.encode(commandAuthority, forKey: .commandAuthority); try values.encode(failsafe, forKey: .failsafe)
        try values.encode(dt, forKey: .dt); try values.encode(evidenceBoundary, forKey: .evidenceBoundary)
        try values.encode(generationCommand, forKey: .generationCommand); try values.encode(status, forKey: .status)
        try values.encode(drones, forKey: .drones); try values.encode(frames, forKey: .frames)
    }
}

struct MissionEvent: Identifiable {
    enum Level { case info, success, attention }
    let id: String
    let time: Double
    let title: String
    let detail: String
    let level: Level
}

@MainActor
final class CockpitModel: ObservableObject {
    @Published var fixture: Fixture
    @Published var selectedID = "drone-001"
    @Published var isPlaying = true
    @Published var time: Double = 0
    @Published var showObstacles = true
    @Published var updateStatus = "Update check idle"
    @Published var cameraResetToken = 0
    @Published var cameraFocusID: String?

    private var timer: Timer?
    private var basePositions: [String: SIMD3<Float>] = [:]
    private var frameIndex = 0

    init() {
        fixture = FixtureLoader.load() ?? Fixture(
            schema: "zephyr-s7-native-fixture-1", runID: "local-fallback", source: "synthetic", seed: 17,
            model: "multi-agent kinematic preview", unavailable: ["live radio", "camera", "flight performance"],
            drones: FixtureLoader.fallbackDrones())
        selectedID = fixture.drones.first?.id ?? "drone-001"
        basePositions = Dictionary(uniqueKeysWithValues: fixture.drones.map { ($0.id, $0.position) })
        if let first = fixture.frames.first { fixture.drones = first.drones; time = first.time }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in self.tick() }
        }
    }


    var selected: DroneTelemetry? { fixture.drones.first(where: { $0.id == selectedID }) }
    var currentFrame: FixtureFrame? { fixture.frames.indices.contains(frameIndex) ? fixture.frames[frameIndex] : nil }
    var activeCount: Int { currentFrame?.activeCount ?? fixture.drones.filter(\.active).count }
    var frameLabel: String { fixture.frames.isEmpty ? "LIVE PREVIEW" : "FRAME \(frameIndex + 1) / \(fixture.frames.count)" }
    var progress: Double {
        guard let last = fixture.frames.last?.time, last > 0 else { return 0 }
        return min(max(time / last, 0), 1)
    }

    var targetPosition: SIMD3<Float> { currentFrame?.targetPosition ?? SIMD3<Float>(0, 0, 1.5) }

    func resetCamera() { cameraFocusID = nil; cameraResetToken += 1 }
    func focusSelectedVehicle() { cameraFocusID = selectedID; cameraResetToken += 1 }

    var missionEvents: [MissionEvent] {
        let selected = self.selected
        var events: [MissionEvent] = [
            MissionEvent(id: "manifest", time: 0, title: "Replay manifest loaded", detail: "\(fixture.runID) · seed \(fixture.seed)", level: .info),
            MissionEvent(id: "authority", time: 0, title: "Command authority locked", detail: fixture.commandAuthority.capitalized, level: .attention),
            MissionEvent(id: "failsafe", time: 0, title: "Failsafe boundary recorded", detail: fixture.failsafe, level: .attention)
        ]
        if let frame = currentFrame {
            events.append(MissionEvent(id: "frame-\(frame.stepIndex)-\(frame.time)", time: frame.time,
                                       title: "Frame \(frame.stepIndex) committed", detail: "\(frame.activeCount) active vehicles · Δt \(decimal(fixture.dt, "%.3f")) s", level: .success))
            let worstLoss = frame.drones.map(\.linkLossPct).max() ?? 0
            if worstLoss >= 8 {
                events.append(MissionEvent(id: "link-\(frame.stepIndex)", time: frame.time,
                                           title: "Link envelope visible", detail: "Peak modeled loss \(decimal(worstLoss, "%.1f"))% · replay metric", level: .attention))
            }
            let lowConfidence = frame.drones.map(\.confidence).min() ?? 1
            if lowConfidence < 0.98 {
                events.append(MissionEvent(id: "confidence-\(frame.stepIndex)", time: frame.time,
                                           title: "Estimator confidence changed", detail: "Minimum cooperative confidence \(decimal(lowConfidence, "%.3f"))", level: .attention))
            }
            if let selected, selected.minNeighborDistance > 0 {
                events.append(MissionEvent(id: "separation-\(frame.stepIndex)-\(selected.id)", time: frame.time,
                                           title: "Separation margin observed", detail: "\(selected.id) nearest neighbor \(decimal(selected.minNeighborDistance, "%.3f")) m", level: .success))
            }
        }
        return events.sorted { $0.time > $1.time }
    }

    func tick() { guard isPlaying else { return }; advanceOneFrame() }

    func stepFrame() { isPlaying = false; advanceOneFrame() }

    private func advanceOneFrame() {
        guard !fixture.frames.isEmpty else { time += fixture.dt; return }
        if frameIndex < fixture.frames.count - 1 { frameIndex += 1 }
        fixture.drones = fixture.frames[frameIndex].drones
        time = fixture.frames[frameIndex].time
        if frameIndex == fixture.frames.count - 1 { isPlaying = false }
    }

    func reset() {
        isPlaying = false; time = fixture.frames.first?.time ?? 0; frameIndex = 0
        if let first = fixture.frames.first { fixture.drones = first.drones }
        else { fixture.drones = fixture.drones.map { drone in var value = drone; value.position = basePositions[drone.id] ?? value.position; value.velocity = .zero; return value } }
    }

    func checkForUpdates() {
        updateStatus = "Checking GitHub release..."
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let check = try await ReleaseUpdater.checkLatest()
                guard check.isNewer else {
                    updateStatus = "Up to date · \(check.release.tagName)"
                    return
                }
                updateStatus = "Downloading \(check.release.tagName)..."
                let tag = try await ReleaseUpdater.installLatest()
                updateStatus = "Restarting with \(tag)..."
                try? await Task.sleep(for: .milliseconds(500))
                NSApp.terminate(nil)
            } catch {
                updateStatus = "Update unavailable · \(error.localizedDescription)"
            }
        }
    }
}

enum FixtureLoader {
    static func load() -> Fixture? {
        guard let url = Bundle.module.url(forResource: "demo_swarm", withExtension: "json"),
              let data = try? Data(contentsOf: url), let fixture = try? JSONDecoder().decode(Fixture.self, from: data) else { return nil }
        return fixture
    }

    static func fallbackDrones() -> [DroneTelemetry] {
        (0..<6).map { index in
            let id = "drone-00\(index + 1)"
            let position = SIMD3<Float>(Float(index % 3) * 1.4 - 1.4, Float(index / 3) * 1.3 - 0.65, 1.6)
            let isLead = index == 0; let isWeak = index == 4
            return DroneTelemetry(id: id, name: "Scout \(index + 1)", role: isLead ? "lead" : "cooperative", color: isLead ? "sky" : "aurora", position: position, velocity: .zero, battery: 86 - Double(index * 4), linkDelayMs: 18 + Double(index * 3), linkLossPct: isWeak ? 9.0 : 1.0, state: isWeak ? "Holding" : "Flying", confidence: 0.94 - Double(index) * 0.02, profileID: "\(id)-synthetic-profile", neighborCount: 5, minNeighborDistance: 1.2)
        }
    }
}

// MARK: - Operator cockpit

struct CockpitView: View {
    @ObservedObject var model: CockpitModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private let horizon = Horizon()
    @State private var fleetFilter = ""
    @State private var showReplayDetails = false
    @State private var showProvenance = false

    private var filteredDrones: [DroneTelemetry] {
        guard !fleetFilter.isEmpty else { return model.fixture.drones }
        return model.fixture.drones.filter { $0.id.localizedCaseInsensitiveContains(fleetFilter) || $0.name.localizedCaseInsensitiveContains(fleetFilter) }
    }

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider().overlay(horizon.line)
            VStack(spacing: 0) {
                topBar
                HStack(spacing: 0) {
                    sceneSurface
                    Divider().overlay(horizon.line)
                    inspector
                }
                .frame(maxHeight: .infinity)
                timeline
            }
            .background(horizon.night)
        }
        .background(horizon.night)
        .foregroundStyle(horizon.cloud)
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: ZephyrDesign.Layout.sectionSpacing) {
                Image(systemName: "dot.radiowaves.left.and.right").font(ZephyrDesign.Typography.icon.weight(.semibold)).foregroundStyle(horizon.sky)
                Text("Zephyr Mesh").font(ZephyrDesign.Typography.brand)
            }
            .padding(.bottom, 20)
            HStack(alignment: .firstTextBaseline) {
                Text("Fleet").font(ZephyrDesign.Typography.heading)
                Spacer()
                Text("\(filteredDrones.count) of \(model.fixture.drones.count)").font(ZephyrDesign.Typography.mono).foregroundStyle(horizon.mist)
            }
            TextField("Filter by ID or name", text: $fleetFilter)
                .textFieldStyle(.roundedBorder)
                .padding(.vertical, 9)
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(filteredDrones) { drone in FleetRow(drone: drone, isSelected: drone.id == model.selectedID, horizon: horizon) { model.selectedID = drone.id } }
                }
                .padding(.bottom, 10)
            }
            Divider().overlay(horizon.line)
            OperationalGate(fixture: model.fixture, horizon: horizon)
        }
        .padding(.horizontal, ZephyrDesign.Spacing.md)
        .padding(.vertical, ZephyrDesign.Spacing.lg)
        .frame(width: ZephyrDesign.Layout.sidebarWidth)
        .background(horizon.grouped)
    }

    private var topBar: some View {
        HStack(alignment: .center, spacing: ZephyrDesign.Layout.panelSpacing) {
            VStack(alignment: .leading, spacing: ZephyrDesign.Layout.tightSpacing) {
                Text("Cooperative test range").font(ZephyrDesign.Typography.heading).foregroundStyle(horizon.mist)
                Text(model.fixture.runID).font(ZephyrDesign.Typography.title.monospaced())
            }
            Spacer()
            Label("Synthetic replay", systemImage: "waveform.path.ecg").font(ZephyrDesign.Typography.secondary).foregroundStyle(horizon.mist)
            Menu {
                Button("Check for updates") { model.checkForUpdates() }
                Button("Focus selected vehicle") { model.focusSelectedVehicle() }
                Button("Reset camera") { model.resetCamera() }
                Button("Reset replay") { model.isPlaying = false; model.reset() }
            } label: {
                Image(systemName: "ellipsis.circle").font(ZephyrDesign.Typography.icon)
            }
            .menuStyle(.borderlessButton)
            .help("More actions")
        }
        .padding(.horizontal, ZephyrDesign.Spacing.lg).padding(.vertical, ZephyrDesign.Spacing.md)
        .background(horizon.surface)
        .overlay(alignment: .bottom) { Divider().overlay(horizon.line) }
    }

    private var sceneSurface: some View {
        ZStack(alignment: .topLeading) {
            SceneViewRepresentable(model: model, reduceMotion: reduceMotion)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            VStack(alignment: .leading, spacing: ZephyrDesign.Layout.compactSpacing) {
                Text(model.frameLabel.capitalized).font(ZephyrDesign.Typography.heading)
                Text("\(model.activeCount) of \(model.fixture.drones.count) vehicles active · target \(vector(model.targetPosition))")
                    .font(ZephyrDesign.Typography.mono).foregroundStyle(horizon.mist)
            }
            .padding(.horizontal, ZephyrDesign.Layout.overlayPadding).padding(.vertical, 11)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: ZephyrDesign.Radius.surface))
            .padding(ZephyrDesign.Layout.panelSpacing)
        }
        .frame(minWidth: ZephyrDesign.Layout.sceneMinimumWidth, maxWidth: .infinity, maxHeight: .infinity)
    }

    private var inspector: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: ZephyrDesign.Layout.panelSpacing) {
                if let selected = model.selected { VehicleInspector(drone: selected, horizon: horizon) }
                DisclosureGroup("Replay notes", isExpanded: $showReplayDetails) {
                    MissionEventList(events: model.missionEvents, horizon: horizon)
                }
                .font(ZephyrDesign.Typography.heading)
                DisclosureGroup("Provenance", isExpanded: $showProvenance) {
                    ProvenancePanel(fixture: model.fixture, horizon: horizon)
                }
                .font(ZephyrDesign.Typography.heading)
            }
            .padding(ZephyrDesign.Layout.panelPadding)
        }
        .frame(width: ZephyrDesign.Layout.inspectorWidth)
        .background(horizon.night.opacity(0.96))
    }

    private var timeline: some View {
        VStack(spacing: ZephyrDesign.Layout.sectionSpacing) {
            HStack { Text("T+\(model.time, specifier: "%.2f") s").font(ZephyrDesign.Typography.mono.weight(.semibold)); Spacer(); Text("Synthetic replay").font(ZephyrDesign.Typography.secondary).foregroundStyle(horizon.mist) }
            ProgressView(value: model.progress).tint(horizon.sky)
            HStack(spacing: ZephyrDesign.Layout.controlSpacing) {
                Button(model.isPlaying ? "Pause" : "Play") { model.isPlaying.toggle() }.keyboardShortcut(.space, modifiers: []).frame(minHeight: ZephyrDesign.Controls.minHeight)
                Button("Step") { model.stepFrame() }.keyboardShortcut(.rightArrow, modifiers: []).frame(minHeight: ZephyrDesign.Controls.minHeight)
                Spacer()
                Toggle("Show keep-out volume", isOn: $model.showObstacles)
            }
        }
        .padding(.horizontal, ZephyrDesign.Spacing.lg).padding(.vertical, ZephyrDesign.Spacing.sm)
        .background(horizon.surface)
        .overlay(alignment: .top) { Divider().overlay(horizon.line) }
    }
}

struct FleetRow: View {
    let drone: DroneTelemetry; let isSelected: Bool; let horizon: Horizon; let action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: ZephyrDesign.Layout.compactSpacing) {
                Circle().fill(statusColor).frame(width: 6, height: 6)
                Text(drone.id).font(ZephyrDesign.Typography.mono.weight(isSelected ? .semibold : .regular))
                if drone.role == "lead" { Text("lead").font(ZephyrDesign.Typography.secondary).foregroundStyle(horizon.mist) }
                Spacer(minLength: 8)
                Text("\(Int(drone.battery))%").font(ZephyrDesign.Typography.mono).foregroundStyle(horizon.mist)
                Text("\(Int(drone.linkDelayMs)) ms").font(ZephyrDesign.Typography.mono).foregroundStyle(horizon.mist)
            }
            .padding(.horizontal, ZephyrDesign.Spacing.xs).padding(.vertical, ZephyrDesign.Layout.rowVertical)
            .background(isSelected ? horizon.accent.opacity(0.12) : .clear)
            .overlay(alignment: .bottom) { Divider().overlay(horizon.line.opacity(0.55)) }
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(drone.id), \(drone.state), \(Int(drone.battery)) percent battery, \(Int(drone.linkDelayMs)) milliseconds link delay")
    }
    private var statusColor: Color { !drone.active ? horizon.warning : drone.linkLossPct > 8 ? horizon.warning : horizon.mist }
}

struct VehicleInspector: View {
    let drone: DroneTelemetry; let horizon: Horizon
    var body: some View {
        VStack(alignment: .leading, spacing: ZephyrDesign.Spacing.md) {
            SectionTitle(title: "Selected vehicle", subtitle: drone.profileID, horizon: horizon)
            HStack(alignment: .firstTextBaseline) {
                Text(drone.name).font(ZephyrDesign.Typography.title)
                Text(drone.id).font(ZephyrDesign.Typography.mono).foregroundStyle(horizon.mist)
                Spacer()
                Text(drone.state).font(ZephyrDesign.Typography.heading).foregroundStyle(drone.active ? horizon.success : horizon.warning)
            }
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: ZephyrDesign.Layout.gridSpacing) {
                MetricTile(label: "Battery", value: "\(decimal(drone.battery, "%.0f"))%", color: horizon.primary, horizon: horizon)
                MetricTile(label: "Confidence", value: "\(decimal(drone.confidence, "%.3f"))", color: horizon.primary, horizon: horizon)
                MetricTile(label: "Link delay", value: "\(decimal(drone.linkDelayMs, "%.0f")) ms", color: horizon.primary, horizon: horizon)
                MetricTile(label: "Packet loss", value: "\(decimal(drone.linkLossPct, "%.1f"))%", color: horizon.primary, horizon: horizon)
            }
            DataSection(title: "Flight state", horizon: horizon) {
                DataRow(label: "Position", value: vector(drone.position), horizon: horizon)
                DataRow(label: "Velocity", value: vector(drone.velocity), horizon: horizon)
                DataRow(label: "Active", value: drone.active ? "Yes" : "No", horizon: horizon)
            }
            DataSection(title: "Cooperative context", horizon: horizon) {
                DataRow(label: "Neighbors", value: "\(drone.neighborCount)", horizon: horizon)
                DataRow(label: "Nearest vehicle", value: drone.minNeighborDistance > 0 ? "\(decimal(drone.minNeighborDistance, "%.3f")) m" : "Not recorded", horizon: horizon)
                DataRow(label: "Constraints", value: drone.constraintFlags.isEmpty ? "Clear" : drone.constraintFlags.joined(separator: ", "), horizon: horizon)
            }
        }
    }
}

struct MissionEventList: View {
    let events: [MissionEvent]; let horizon: Horizon
    var body: some View {
        DataSection(title: "Recent events", horizon: horizon) {
            ForEach(events.prefix(4)) { event in
                HStack(alignment: .top, spacing: ZephyrDesign.Layout.sectionSpacing) {
                    Image(systemName: icon(for: event.level)).foregroundStyle(color(for: event.level)).frame(width: 14)
                    VStack(alignment: .leading, spacing: ZephyrDesign.Layout.microSpacing) { HStack { Text(event.title).font(ZephyrDesign.Typography.secondary.weight(.semibold)); Spacer(); Text("T+\(event.time, specifier: "%.2f")").font(ZephyrDesign.Typography.tinyMono).foregroundStyle(horizon.mist) }; Text(event.detail).font(ZephyrDesign.Typography.caption).foregroundStyle(horizon.mist).fixedSize(horizontal: false, vertical: true) }
                }
                .padding(.vertical, 3)
            }
        }
    }
    private func icon(for level: MissionEvent.Level) -> String { switch level { case .info: return "info.circle.fill"; case .success: return "checkmark.circle.fill"; case .attention: return "exclamationmark.triangle.fill" } }
    private func color(for level: MissionEvent.Level) -> Color { switch level { case .info: return horizon.sky; case .success: return horizon.aurora; case .attention: return horizon.apricot } }
}

struct ProvenancePanel: View {
    let fixture: Fixture; let horizon: Horizon
    var body: some View {
        DataSection(title: "Evidence", horizon: horizon) {
            DataRow(label: "Schema", value: fixture.schema, horizon: horizon)
            DataRow(label: "Source", value: fixture.source, horizon: horizon)
            DataRow(label: "Model", value: fixture.model, horizon: horizon)
            DataRow(label: "Seed / Δt", value: "\(fixture.seed) / \(decimal(fixture.dt, "%.3f")) s", horizon: horizon)
            DataRow(label: "Unavailable", value: fixture.unavailable.joined(separator: ", "), horizon: horizon)
            Text(fixture.evidenceBoundary).font(ZephyrDesign.Typography.caption).foregroundStyle(horizon.mist).fixedSize(horizontal: false, vertical: true).padding(.top, 3)
            Text("Generator: \(fixture.generationCommand)").font(ZephyrDesign.Typography.tinyMono).foregroundStyle(horizon.mist).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct OperationalGate: View {
    let fixture: Fixture; let horizon: Horizon
    var body: some View {
        VStack(alignment: .leading, spacing: ZephyrDesign.Layout.tightSpacing) {
            Text("Replay only").font(ZephyrDesign.Typography.secondary.weight(.semibold))
            Text("No radio, camera, motor, or swarm command path is attached.").font(ZephyrDesign.Typography.caption).foregroundStyle(horizon.mist).fixedSize(horizontal: false, vertical: true)
        }
        .padding(.top, ZephyrDesign.Spacing.sm)
    }
}

struct SectionTitle: View { let title: String; var subtitle: String? = nil; let horizon: Horizon; var body: some View { VStack(alignment: .leading, spacing: ZephyrDesign.Layout.tightSpacing) { Text(title).font(ZephyrDesign.Typography.heading).foregroundStyle(horizon.primary); if let subtitle { Text(subtitle).font(ZephyrDesign.Typography.secondary).foregroundStyle(horizon.mist).lineLimit(2) } } } }
struct DataSection<Content: View>: View { let title: String; var subtitle: String? = nil; let horizon: Horizon; @ViewBuilder let content: () -> Content; var body: some View { VStack(alignment: .leading, spacing: ZephyrDesign.Layout.sectionSpacing) { SectionTitle(title: title, subtitle: subtitle, horizon: horizon); content() }.padding(.vertical, 10).overlay(alignment: .bottom) { Divider().overlay(horizon.line) } } }
struct DataRow: View { let label: String; let value: String; let horizon: Horizon; var body: some View { HStack(alignment: .top, spacing: ZephyrDesign.Layout.compactSpacing) { Text(label).font(ZephyrDesign.Typography.secondary).foregroundStyle(horizon.mist); Spacer(minLength: 6); Text(value).font(ZephyrDesign.Typography.mono).foregroundStyle(horizon.primary).multilineTextAlignment(.trailing).textSelection(.enabled) } } }
struct MetricTile: View { let label: String; let value: String; let color: Color; let horizon: Horizon; var body: some View { VStack(alignment: .leading, spacing: ZephyrDesign.Layout.smallSpacing) { Text(label).font(ZephyrDesign.Typography.secondary).foregroundStyle(horizon.mist); Text(value).font(ZephyrDesign.Typography.metric).foregroundStyle(color) }.frame(maxWidth: .infinity, alignment: .leading) } }

private func decimal(_ value: Double, _ format: String) -> String { String(format: format, value) }
private func vector(_ value: SIMD3<Float>) -> String { "[\(decimal(Double(value.x), "%.2f")), \(decimal(Double(value.y), "%.2f")), \(decimal(Double(value.z), "%.2f"))]" }

// MARK: - SceneKit projection

struct SceneViewRepresentable: NSViewRepresentable {
    @ObservedObject var model: CockpitModel
    let reduceMotion: Bool
    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var resetToken = -1 }
    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = SceneBuilder.build(model: model, reduceMotion: reduceMotion)
        context.coordinator.resetToken = model.cameraResetToken
        view.allowsCameraControl = true
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.defaultCameraController.automaticTarget = false
        view.defaultCameraController.worldUp = SCNVector3(0, 1, 0)
        view.defaultCameraController.target = SCNVector3(0, 1.1, 0)
        view.cameraControlConfiguration.allowsTranslation = false
        view.defaultCameraController.inertiaEnabled = true
        view.defaultCameraController.minimumVerticalAngle = -12
        view.defaultCameraController.maximumVerticalAngle = 62
        view.defaultCameraController.minimumHorizontalAngle = -140
        view.defaultCameraController.maximumHorizontalAngle = 140
        view.backgroundColor = NSColor.windowBackgroundColor
        view.antialiasingMode = .multisampling4X
        view.rendersContinuously = !reduceMotion
        view.setAccessibilityLabel("3D synthetic replay scene")
        return view
    }
    func updateNSView(_ view: SCNView, context: Context) {
        view.rendersContinuously = !reduceMotion
        if let focus = model.cameraFocusID.flatMap({ id in model.fixture.drones.first(where: { $0.id == id })?.position }) {
            view.defaultCameraController.target = SCNVector3(focus.x, focus.z, focus.y)
        } else {
            view.defaultCameraController.target = SCNVector3(0, 1.1, 0)
        }
        if context.coordinator.resetToken != model.cameraResetToken {
            view.scene = SceneBuilder.build(model: model, reduceMotion: reduceMotion)
            context.coordinator.resetToken = model.cameraResetToken
        } else {
            SceneBuilder.update(view: view, model: model, reduceMotion: reduceMotion)
        }
    }
}

@MainActor
enum SceneBuilder {
    static let dronePrefix = "drone-"
    static func build(model: CockpitModel, reduceMotion: Bool = false) -> SCNScene {
        let scene = SCNScene(); scene.rootNode.addChildNode(floor()); scene.rootNode.addChildNode(grid()); scene.rootNode.addChildNode(camera(focus: model.cameraFocusID.flatMap { id in model.fixture.drones.first(where: { $0.id == id })?.position })); scene.rootNode.addChildNode(light()); scene.rootNode.addChildNode(targetNode(position: model.targetPosition))
        for drone in model.fixture.drones { scene.rootNode.addChildNode(droneNode(drone, animate: !reduceMotion)) }
        scene.rootNode.addChildNode(obstacle(position: SCNVector3(0, 0.8, -1.5), scale: SCNVector3(1.3, 1.6, 1.3)))
        return scene
    }

    static func update(view: SCNView, model: CockpitModel, reduceMotion: Bool = false) {
        for drone in model.fixture.drones {
            guard let node = view.scene?.rootNode.childNode(withName: dronePrefix + drone.id, recursively: true) else { continue }
            node.position = SCNVector3(drone.position.x, drone.position.z, drone.position.y)
            let horizontalSpeed = hypot(drone.velocity.x, drone.velocity.y)
            if horizontalSpeed > 0.01 { node.eulerAngles.y = CGFloat(atan2(drone.velocity.x, drone.velocity.y)) }
            updateRotors(node: node, drone: drone, reduceMotion: reduceMotion)
            node.opacity = drone.active ? (drone.linkLossPct >= 8 ? 0.62 : 1.0) : 0.2
            node.childNode(withName: "selection-ring", recursively: true)?.isHidden = drone.id != model.selectedID
            node.childNode(withName: "body", recursively: true)?.geometry?.firstMaterial?.emission.contents = drone.id == model.selectedID ? ZephyrDesign.ScenePalette.selectionEmission : NSColor.clear
        }
        view.scene?.rootNode.childNodes.filter { $0.name == "obstacle" }.forEach { $0.isHidden = !model.showObstacles }
        if let target = view.scene?.rootNode.childNode(withName: "target", recursively: true) { target.position = SCNVector3(model.targetPosition.x, model.targetPosition.z, model.targetPosition.y) }
    }

    private static func updateRotors(node: SCNNode, drone: DroneTelemetry, reduceMotion: Bool) {
        let speed = min(max(sqrt(drone.velocity.x * drone.velocity.x + drone.velocity.y * drone.velocity.y + drone.velocity.z * drone.velocity.z), 0), 4)
        let period = max(0.12, 0.42 - Double(speed) * 0.06)
        let previous = node.value(forKey: "rotorPeriod") as? Double
        let needsRetiming = previous.map { abs($0 - period) > 0.01 } ?? true
        node.setValue(period, forKey: "rotorPeriod")
        for rotor in node.childNodes where rotor.name?.hasPrefix("rotor-") == true {
            if !drone.active || reduceMotion {
                rotor.removeAction(forKey: "spin")
            } else if needsRetiming || rotor.action(forKey: "spin") == nil {
                let spin = SCNAction.repeatForever(SCNAction.rotateBy(x: 0, y: CGFloat.pi * 2, z: 0, duration: period))
                rotor.runAction(spin, forKey: "spin")
            }
        }
    }

    static func droneNode(_ drone: DroneTelemetry, animate: Bool = true) -> SCNNode {
        let group = SCNNode(); group.name = dronePrefix + drone.id; group.position = SCNVector3(drone.position.x, drone.position.z, drone.position.y)
        let horizontalSpeed = hypot(drone.velocity.x, drone.velocity.y)
        if horizontalSpeed > 0.01 { group.eulerAngles.y = CGFloat(atan2(drone.velocity.x, drone.velocity.y)) }
        let body = SCNNode(geometry: SCNBox(width: 0.30, height: 0.10, length: 0.22, chamferRadius: 0.04)); body.name = "body"; body.geometry?.firstMaterial = material(ZephyrDesign.ScenePalette.body); group.addChildNode(body)
        for angle in [Float.pi / 4, -Float.pi / 4] {
            let arm = SCNNode(geometry: SCNBox(width: 0.86, height: 0.045, length: 0.045, chamferRadius: 0.018))
            arm.eulerAngles.y = CGFloat(angle)
            arm.geometry?.firstMaterial = material(ZephyrDesign.ScenePalette.arm)
            group.addChildNode(arm)
        }
        let ring = SCNNode(geometry: SCNTorus(ringRadius: 0.33, pipeRadius: 0.009)); ring.name = "selection-ring"; ring.geometry?.firstMaterial = material(ZephyrDesign.ScenePalette.selection); ring.position.y = 0.12; ring.isHidden = true; group.addChildNode(ring)
        var rotorIndex = 0
        for x in [-0.30, 0.30] { for z in [-0.30, 0.30] {
            let rotor = SCNCylinder(radius: 0.105, height: 0.018)
            rotor.firstMaterial = material(ZephyrDesign.ScenePalette.rotor)
            let r = SCNNode(geometry: rotor); r.name = "rotor-\(rotorIndex)"; r.position = SCNVector3(Float(x), 0.06, Float(z)); group.addChildNode(r)
            let motionSpeed = min(max(sqrt(drone.velocity.x * drone.velocity.x + drone.velocity.y * drone.velocity.y + drone.velocity.z * drone.velocity.z), 0), 4)
            let rotorPeriod = max(0.12, 0.42 - Double(motionSpeed) * 0.06)
            if drone.active && animate { let spin = SCNAction.repeatForever(SCNAction.rotateBy(x: 0, y: CGFloat.pi * 2, z: 0, duration: rotorPeriod)); r.runAction(spin) }
            rotorIndex += 1
        } }
        return group
    }

    static func targetNode(position: SIMD3<Float>) -> SCNNode { let node = SCNNode(geometry: SCNTorus(ringRadius: 0.16, pipeRadius: 0.018)); node.name = "target"; node.position = SCNVector3(position.x, position.z, position.y); node.geometry?.firstMaterial = material(ZephyrDesign.ScenePalette.target); return node }
    static func floor() -> SCNNode { let node = SCNNode(geometry: SCNPlane(width: 14, height: 14)); node.geometry?.firstMaterial = material(ZephyrDesign.ScenePalette.floor); node.eulerAngles.x = -.pi / 2; return node }
    static func grid() -> SCNNode { let group = SCNNode(); for i in -7...7 { let a = SCNNode(geometry: SCNBox(width: 0.006, height: 0.003, length: 14, chamferRadius: 0)); a.position = SCNVector3(Float(i), 0.005, 0); a.geometry?.firstMaterial = material(ZephyrDesign.ScenePalette.grid); group.addChildNode(a); let b = SCNNode(geometry: SCNBox(width: 14, height: 0.003, length: 0.006, chamferRadius: 0)); b.position = SCNVector3(0, 0.006, Float(i)); b.geometry?.firstMaterial = material(ZephyrDesign.ScenePalette.grid); group.addChildNode(b) }; return group }
    static func camera(focus: SIMD3<Float>? = nil) -> SCNNode { let node = SCNNode(); node.name = "camera"; node.camera = SCNCamera(); node.camera?.fieldOfView = 48; let target = focus.map { SCNVector3($0.x, $0.z, $0.y) } ?? SCNVector3(0, 1.1, 0); node.position = SCNVector3(target.x + 7.5, target.y + 5.1, target.z + 8.5); node.look(at: target); return node }
    static func light() -> SCNNode { let node = SCNNode(); node.light = SCNLight(); node.light?.type = .omni; node.light?.intensity = 850; node.position = SCNVector3(2, 7, 3); return node }
    static func obstacle(position: SCNVector3, scale: SCNVector3) -> SCNNode { let node = SCNNode(geometry: SCNBox(width: 1, height: 1, length: 1, chamferRadius: 0.04)); node.name = "obstacle"; node.position = position; node.scale = scale; node.geometry?.firstMaterial = material(ZephyrDesign.ScenePalette.obstacle); return node }
    static func material(_ color: NSColor) -> SCNMaterial { let m = SCNMaterial(); m.diffuse.contents = color; m.roughness.contents = 0.78; return m }
}

struct Horizon {
    let night = ZephyrDesign.window
    let grouped = ZephyrDesign.groupedSurface
    let surface = ZephyrDesign.surface
    let line = ZephyrDesign.separator
    let primary = ZephyrDesign.primary
    let cloud = ZephyrDesign.primary
    let mist = ZephyrDesign.secondary
    let sky = ZephyrDesign.accent
    let accent = ZephyrDesign.accent
    let success = ZephyrDesign.success
    let warning = ZephyrDesign.warning
    let apricot = ZephyrDesign.warning
    let aurora = ZephyrDesign.success
    let lavender = ZephyrDesign.secondary
}
