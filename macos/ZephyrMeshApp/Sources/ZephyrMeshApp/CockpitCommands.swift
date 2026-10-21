import SwiftUI

/// Menu bar commands in their standard places. Single-key shortcuts are attached here and the
/// items are disabled while a text field is being edited, so typing in the team filter never
/// triggers them (AppKit does not perform the key equivalent of a disabled menu item, and the
/// key goes to the field instead).
struct CockpitCommands: Commands {
    @ObservedObject var store: AppStore
    @Environment(\.openWindow) private var openWindow

    private var model: CockpitModel? { store.cockpit }
    /// Typing in a field, or working in the Results window, which uses the arrows and letters itself.
    private var typing: Bool { (model?.isTextFieldFocused ?? false) || store.isResultsWindowKey }
    /// Arrow-key items also give way to the focused team list, which uses the arrows itself.
    private var arrowsTaken: Bool { typing || (model?.isTeamListFocused ?? false) }
    private var unavailable: Bool { model == nil }

    var body: some Commands {
        CommandGroup(after: .appInfo) {
            Button(UIText.Menus.checkForUpdates) { model?.checkForUpdates() }
                .disabled(unavailable || model?.isCheckingForUpdates == true)
        }
        CommandGroup(replacing: .newItem) {
            Button(UIText.Menus.openEvidenceReport) {
                Task { if let id = await store.chooseEvidenceReport() { openWindow(id: WindowID.evidenceReport, value: id) } }
            }
            .shortcut("openEvidenceReport")
            Button(UIText.Menus.openScenarioRun) {
                Task { if let id = await store.chooseScenarioRun() { openWindow(id: WindowID.scenarioRun, value: id) } }
            }
            .shortcut("openScenarioRun")
        }
        CommandGroup(before: .toolbar) {
            Button(model?.viewMode == .top ? UIText.Menus.show3DView : UIText.Menus.showTopView) { model?.toggleViewMode() }
                .shortcut("toggleTopView")
                .disabled(unavailable || typing)
            Toggle(UIText.Menus.showKeepOut, isOn: binding(\.showKeepOut))
                .shortcut("toggleKeepOut")
                .disabled(unavailable || typing)
            Toggle(UIText.Menus.showInspector, isOn: binding(\.showInspector))
                .shortcut("toggleInspector")
                .disabled(unavailable || typing)
            Divider()
            Button(UIText.Menus.focusSelected) { model?.focusSelected() }
                .shortcut("focusSelected")
                .disabled(unavailable || typing || model?.selectedID == nil)
            Button(UIText.Menus.resetCamera) { model?.resetCamera() }
                .shortcut("resetCamera")
                .disabled(unavailable || typing)
            Button(UIText.Menus.filterDrones) { model?.requestFilterFocus() }
                .shortcut("focusFilter")
                .disabled(unavailable || store.isResultsWindowKey)
            Divider()
            Button(UIText.Menus.showResults) { openWindow(id: WindowID.results) }
                .shortcut("showResults")
            Divider()
        }
        CommandMenu(UIText.Menus.playback) {
            Button(model?.isPlaying == true ? UIText.Transport.pause.label : UIText.Transport.play.label) { model?.togglePlayback() }
                .shortcut("playPause")
                .disabled(unavailable || typing)
            Button(UIText.Menus.stepForward) { model?.stepFrame() }
                .shortcut("stepForward")
                .disabled(unavailable || arrowsTaken || model?.isAtEnd == true)
            Button(UIText.Menus.stepBack) { model?.stepBackward() }
                .shortcut("stepBack")
                .disabled(unavailable || arrowsTaken || model?.isAtStart == true)
            Button(UIText.Transport.restart.label) { model?.restart() }
                .shortcut("restart")
                .disabled(unavailable || typing)
            Divider()
            Picker(UIText.Menus.speed, selection: speedBinding) {
                ForEach(PlaybackSpeed.allCases) { speed in
                    Text(speed.label).tag(speed)
                }
            }
            .disabled(unavailable)
        }
        CommandGroup(replacing: .help) {
            Button(UIText.Menus.aboutReplay) { openWindow(id: WindowID.about) }
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
