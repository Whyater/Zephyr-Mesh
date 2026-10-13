import AppKit
import Foundation
import SwiftUI

enum PlaybackSpeed: Double, CaseIterable, Identifiable {
    case tenth = 0.1
    case quarter = 0.25
    case realtime = 1.0
    var id: Double { rawValue }
    var label: String {
        switch self {
        case .tenth: return "0.1x"
        case .quarter: return "0.25x"
        case .realtime: return "1x"
        }
    }
}

enum CockpitViewMode: String, CaseIterable, Identifiable {
    case threeD = "3d"
    case top
    var id: String { rawValue }
    var label: String { self == .threeD ? "3D" : "Top" }
}

enum CockpitAppearance: String, CaseIterable, Identifiable {
    case system
    case dark
    case light
    var id: String { rawValue }
    var label: String {
        switch self {
        case .system: return "System"
        case .dark: return "Dark"
        case .light: return "Light"
        }
    }
    static let storageKey = "zephyr.mesh.appearance"
}

/// A one-shot camera instruction for the scene. The serial makes repeated presses distinct.
struct CameraCommand: Equatable {
    enum Kind: Equatable {
        case reset
        case focus(droneID: String)
    }
    let kind: Kind
    let serial: Int
}

/// Update messages from the shared UI text (`UIText.update`), with `{tag}` and `{reason}` filled in.
enum UpdateText {
    static func text(_ key: String, _ values: [String: String] = [:]) -> String {
        Format.fill(UIText.update[key] ?? key, values)
    }
}

struct UpdateNotice: Identifiable {
    let id = UUID()
    let title: String
    let message: String
    /// True for the "installing" notice: the app quits moments later, so it must never block.
    var precedesQuit = false
}

/// Replay state for one loaded fixture. It never computes a displayed metric: every value the
/// views show is read from the fixture's embedded Python presentation for the current frame.
@MainActor
final class CockpitModel: ObservableObject {
    let fixture: Fixture

    @Published private(set) var frameIndex = 0
    @Published private(set) var isPlaying = false
    @Published var playbackSpeed: PlaybackSpeed = .quarter
    @Published var selectedID: String?
    @Published var hoveredID: String?
    @Published var teamFilter = ""
    @Published var isTextFieldFocused = false
    /// True while the team list has keyboard focus; its arrow keys move the list selection.
    @Published var isTeamListFocused = false
    @Published var viewMode: CockpitViewMode = .threeD
    @Published var showKeepOut = true
    @Published var showInspector = true
    @Published private(set) var cameraCommand: CameraCommand?
    @Published var updateNotice: UpdateNotice?
    @Published private(set) var isCheckingForUpdates = false
    @Published var filterFocusRequest = 0

    private var timer: Timer?
    private var playbackAccumulator = 0.0
    private var commandSerial = 0
    nonisolated static let tickInterval = 1.0 / 60.0

    init(fixture: Fixture, automaticallyPlay: Bool = false) {
        self.fixture = fixture
        self.selectedID = fixture.drones.first?.id
        if let failure = ReleaseUpdater.takeLastResult() {
            updateNotice = UpdateNotice(title: UpdateText.text("keptCurrent"), message: failure)
        }
        if automaticallyPlay { play() }
    }

    // MARK: Current frame

    var frame: FixtureFrame { fixture.frames[frameIndex] }
    var presentation: FixturePresentation { frame.presentation }
    var frameCount: Int { fixture.frames.count }
    var time: Double { frame.time }
    var duration: Double { fixture.frames.last?.time ?? 0 }
    var isAtEnd: Bool { frameIndex >= frameCount - 1 }
    var isAtStart: Bool { frameIndex == 0 }

    var selectedDrone: PresentationDrone? {
        guard let selectedID else { return nil }
        return presentation.perDrone.first { $0.droneID == selectedID }
    }

    var selectedTelemetry: DroneTelemetry? {
        guard let selectedID else { return nil }
        return frame.drones.first { $0.id == selectedID }
    }

    func presentationDrone(id: String) -> PresentationDrone? {
        presentation.perDrone.first { $0.droneID == id }
    }

    /// Events from the first frame through the current one, oldest first.
    var eventsSoFar: [PresentationEvent] {
        fixture.frames.prefix(frameIndex + 1).flatMap(\.presentation.events)
    }

    /// Events in the whole run, for sizing the events pane once.
    private(set) lazy var totalEventCount = fixture.frames.reduce(0) { $0 + $1.presentation.events.count }

    /// Frame indexes that carry at least one event, for the timeline tick marks.
    var eventFrameIndexes: [Int] {
        fixture.frames.indices.filter { !fixture.frames[$0].presentation.events.isEmpty }
    }

    // MARK: Playback

    func play() {
        guard !isPlaying else { return }
        if isAtEnd { seek(toFrame: 0) }
        isPlaying = true
        startTimer()
    }

    func pause() {
        guard isPlaying else { return }
        isPlaying = false
        stopTimer()
    }

    func togglePlayback() { isPlaying ? pause() : play() }

    func stepFrame() {
        pause()
        seek(toFrame: frameIndex + 1)
    }

    func stepBackward() {
        pause()
        seek(toFrame: frameIndex - 1)
    }

    func restart() {
        pause()
        seek(toFrame: 0)
    }

    func seek(toFrame index: Int) {
        let clamped = min(max(index, 0), frameCount - 1)
        guard clamped != frameIndex else { return }
        frameIndex = clamped
    }

    /// Seeks to the last frame at or before `target` seconds.
    func seek(toTime target: Double) {
        let safeTarget = target.isFinite ? max(0, target) : 0
        let tolerance = 1e-9
        let index = fixture.frames.lastIndex { $0.time <= safeTarget + tolerance } ?? 0
        seek(toFrame: index)
    }

    // MARK: Selection and camera

    func select(_ droneID: String?) {
        selectedID = droneID
    }

    func focusSelected() {
        guard let selectedID else { return }
        focus(on: selectedID)
    }

    func focus(on droneID: String) {
        selectedID = droneID
        issue(.focus(droneID: droneID))
    }

    func resetCamera() { issue(.reset) }

    func toggleViewMode() { viewMode = viewMode == .threeD ? .top : .threeD }

    func requestFilterFocus() { filterFocusRequest += 1 }

    private func issue(_ kind: CameraCommand.Kind) {
        commandSerial += 1
        cameraCommand = CameraCommand(kind: kind, serial: commandSerial)
    }

    // MARK: Shortcuts

    /// Runs a generated shortcut action. Single-key actions are refused while a text field is
    /// being edited so typing in the team filter never moves the replay.
    func perform(actionID: String, router: ShortcutRouter = ShortcutRouter()) {
        guard router.shouldFire(actionID: actionID, textFieldFocused: isTextFieldFocused,
                                listFocused: isTeamListFocused) else { return }
        switch actionID {
        case "playPause": togglePlayback()
        case "stepForward": stepFrame()
        case "stepBack": stepBackward()
        case "restart": restart()
        case "focusSelected": focusSelected()
        case "resetCamera": resetCamera()
        case "toggleTopView": toggleViewMode()
        case "toggleKeepOut": showKeepOut.toggle()
        case "toggleInspector": showInspector.toggle()
        case "focusFilter": requestFilterFocus()
        default: break
        }
    }

    // MARK: Updates

    func checkForUpdates() {
        guard !isCheckingForUpdates else { return }
        isCheckingForUpdates = true
        Task { @MainActor [weak self] in
            guard let self else { return }
            defer { isCheckingForUpdates = false }
            do {
                let check = try await ReleaseUpdater.checkLatest()
                guard check.isNewer else {
                    updateNotice = UpdateNotice(title: UpdateText.text("upToDate", ["tag": check.release.tagName]), message: "")
                    return
                }
                let tag = try await ReleaseUpdater.installLatest()
                updateNotice = UpdateNotice(title: UpdateText.text("installing"), message: UpdateText.text("available", ["tag": tag]),
                                            precedesQuit: true)
                try? await Task.sleep(for: .milliseconds(800))
                NSApp.terminate(nil)
            } catch {
                updateNotice = UpdateNotice(title: UpdateText.text("failed", ["reason": error.localizedDescription]), message: "")
            }
        }
    }

    // MARK: Timer

    private func startTimer() {
        guard timer == nil else { return }
        let timer = Timer(timeInterval: Self.tickInterval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
        playbackAccumulator = 0
    }

    /// Advances replay time by the wall-clock tick scaled by the playback speed, one fixture
    /// frame per `dt` of accumulated replay time. Stops at the last frame.
    func tick(elapsed: Double = CockpitModel.tickInterval) {
        guard isPlaying, fixture.dt > 0 else { return }
        playbackAccumulator += elapsed * playbackSpeed.rawValue
        while playbackAccumulator >= fixture.dt {
            playbackAccumulator -= fixture.dt
            if isAtEnd {
                pause()
                return
            }
            frameIndex += 1
        }
    }
}
