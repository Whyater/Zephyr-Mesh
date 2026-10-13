import SwiftUI

/// Menu bar commands in their standard places. Single-key shortcuts are attached here and the
/// items are disabled while a text field is being edited, so typing in the team filter never
/// triggers them (AppKit does not perform the key equivalent of a disabled menu item, and the
/// key goes to the field instead).
struct CockpitCommands: Commands {
    @ObservedObject var store: AppStore
    @Environment(\.openWindow) private var openWindow

    private var model: CockpitModel? { store.cockpit }
    private var typing: Bool { model?.isTextFieldFocused ?? false }
    /// Arrow-key items also give way to the focused team list, which uses the arrows itself.
    private var arrowsTaken: Bool { typing || (model?.isTeamListFocused ?? false) }
    private var unavailable: Bool { model == nil }

    var body: some Commands {
        CommandGroup(after: .appInfo) {
            Button(UIText.update["check"] ?? "Check for Updates...") { model?.checkForUpdates() }
                .disabled(unavailable || model?.isCheckingForUpdates == true)
        }
        CommandGroup(replacing: .newItem) {
            Button("Open Evidence Report...") {
                if let id = store.chooseEvidenceReport() { openWindow(id: WindowID.evidenceReport, value: id) }
            }
            .shortcut("openEvidenceReport")
            Button("Open Scenario Run...") {
                if let id = store.chooseScenarioRun() { openWindow(id: WindowID.scenarioRun, value: id) }
            }
            .shortcut("openScenarioRun")
        }
        CommandGroup(before: .toolbar) {
            Button(model?.viewMode == .top ? "Show 3D View" : "Show Top View") { model?.toggleViewMode() }
                .shortcut("toggleTopView")
                .disabled(unavailable || typing)
            Toggle("Show Keep-out Volume", isOn: binding(\.showKeepOut))
                .shortcut("toggleKeepOut")
                .disabled(unavailable || typing)
            Toggle("Show Inspector", isOn: binding(\.showInspector))
                .shortcut("toggleInspector")
                .disabled(unavailable || typing)
            Divider()
            Button("Focus Selected Drone") { model?.focusSelected() }
                .shortcut("focusSelected")
                .disabled(unavailable || typing || model?.selectedID == nil)
            Button("Reset Camera") { model?.resetCamera() }
                .shortcut("resetCamera")
                .disabled(unavailable || typing)
            Button("Filter Drones") { model?.requestFilterFocus() }
                .shortcut("focusFilter")
                .disabled(unavailable)
            Divider()
        }
        CommandMenu("Playback") {
            Button(model?.isPlaying == true ? "Pause" : "Play") { model?.togglePlayback() }
                .shortcut("playPause")
                .disabled(unavailable || typing)
            Button("Step Forward") { model?.stepFrame() }
                .shortcut("stepForward")
                .disabled(unavailable || arrowsTaken || model?.isAtEnd == true)
            Button("Step Back") { model?.stepBackward() }
                .shortcut("stepBack")
                .disabled(unavailable || arrowsTaken || model?.isAtStart == true)
            Button("Restart") { model?.restart() }
                .shortcut("restart")
                .disabled(unavailable || typing)
            Divider()
            Picker("Speed", selection: speedBinding) {
                ForEach(PlaybackSpeed.allCases) { speed in
                    Text(speed.label).tag(speed)
                }
            }
            .disabled(unavailable)
        }
        CommandGroup(replacing: .help) {
            Button("About This Replay") { openWindow(id: WindowID.about) }
                .shortcut("aboutReplay")
                .disabled(unavailable)
        }
    }

    private func binding(_ keyPath: ReferenceWritableKeyPath<CockpitModel, Bool>) -> Binding<Bool> {
        Binding(get: { model?[keyPath: keyPath] ?? false }, set: { model?[keyPath: keyPath] = $0 })
    }

    private var speedBinding: Binding<PlaybackSpeed> {
        Binding(get: { model?.playbackSpeed ?? .quarter }, set: { model?.playbackSpeed = $0 })
    }
}
