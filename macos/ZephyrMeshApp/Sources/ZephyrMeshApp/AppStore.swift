import AppKit
import Combine
import SwiftUI
import UniformTypeIdentifiers

/// App-wide state: the loaded replay (or why it failed) and the documents open in viewer
/// windows. Files are validated by the strict loaders before any window opens.
@MainActor
final class AppStore: ObservableObject {
    enum ViewerDocument: Sendable {
        case scenarioRun(ScenarioRunDocument, filename: String)
        case evidenceReport(EvidenceReport, filename: String)
    }

    struct OpenError: Identifiable {
        let id = UUID()
        let title: String
        let message: String
    }

    let cockpit: CockpitModel?
    let loadError: String?
    @Published private(set) var documents: [UUID: ViewerDocument] = [:]
    @Published var openError: OpenError?
    /// Files being read and checked right now, so a second open of the same file while the
    /// first is still running does not create a second document.
    private var pendingURLs: Set<String> = []
    /// Open documents by file path, so opening the same file again brings its window forward.
    private var openURLs: [String: UUID] = [:]
    private var forwarding: AnyCancellable?
    private var alerts: Set<AnyCancellable> = []

    /// `presentsAlerts` false leaves open errors and update notices in place instead of showing
    /// them, for tests that must not put up a modal alert.
    init(presentsAlerts: Bool = true) {
        switch FixtureLoader.loadBundled() {
        case .success(let fixture):
            let setup = ScreenshotSetup.current
            let model = CockpitModel(fixture: fixture, automaticallyPlay: setup.frame == nil)
            setup.apply(to: model)
            cockpit = model
            loadError = nil
        case .failure(let error):
            cockpit = nil
            loadError = error.localizedDescription
        }
        forwardCommandState()
        if presentsAlerts { presentAlerts() }
    }

    /// Update results and file-open errors are shown by the app, not by the main window, so
    /// they appear even when the main window is closed: as a sheet on the key window when there
    /// is one, otherwise as an app-modal alert. Each notice is cleared once it is shown.
    private func presentAlerts() {
        $openError.compactMap { $0 }
            .receive(on: RunLoop.main)
            .sink { [weak self] error in
                self?.openError = nil
                AlertPresenter.show(title: error.title, message: error.message)
            }
            .store(in: &alerts)
        cockpit?.$updateNotice.compactMap { $0 }
            .receive(on: RunLoop.main)
            .sink { [weak self] notice in
                self?.cockpit?.updateNotice = nil
                AlertPresenter.show(title: notice.title, message: notice.message, allowsModal: !notice.precedesQuit)
            }
            .store(in: &alerts)
    }

    /// Menus depend on a handful of model values; forward only those so the menu bar does not
    /// rebuild on every hover or frame tick.
    private func forwardCommandState() {
        guard let model = cockpit else { return }
        let signals: [AnyPublisher<Void, Never>] = [
            model.$isPlaying.removeDuplicates().map { _ in () }.eraseToAnyPublisher(),
            model.$isTextFieldFocused.removeDuplicates().map { _ in () }.eraseToAnyPublisher(),
            model.$isTeamListFocused.removeDuplicates().map { _ in () }.eraseToAnyPublisher(),
            model.$viewMode.removeDuplicates().map { _ in () }.eraseToAnyPublisher(),
            model.$showKeepOut.removeDuplicates().map { _ in () }.eraseToAnyPublisher(),
            model.$showInspector.removeDuplicates().map { _ in () }.eraseToAnyPublisher(),
            model.$playbackSpeed.removeDuplicates().map { _ in () }.eraseToAnyPublisher(),
            model.$frameIndex.removeDuplicates().map { _ in () }.eraseToAnyPublisher(),
            model.$selectedID.removeDuplicates().map { _ in () }.eraseToAnyPublisher()
        ]
        forwarding = Publishers.MergeMany(signals)
            .receive(on: RunLoop.main)
            .sink { [weak self] in self?.objectWillChange.send() }
    }

    func document(_ id: UUID) -> ViewerDocument? { documents[id] }

    /// Called when a viewer window closes, so its parsed document is released.
    func close(_ id: UUID) {
        documents[id] = nil
        openURLs = openURLs.filter { $0.value != id }
    }

    /// Shows an open panel, validates the chosen scenario run, and returns its window ID.
    func chooseScenarioRun() async -> UUID? {
        guard let url = choose(title: "Open Scenario Run", message: "Choose a generated scenario run JSON file") else { return nil }
        return await loadScenarioRun(url)
    }

    /// Reads, scans, hashes and decodes off the main actor (a run may be up to 64 MiB), then
    /// registers the document here. A file that is already open returns its existing window ID.
    func loadScenarioRun(_ url: URL) async -> UUID? {
        await load(url, kind: "scenario", errorTitle: "Could not open scenario run") { url in
            .scenarioRun(try ScenarioRunDocument.load(from: url), filename: url.lastPathComponent)
        }
    }

    /// Shows an open panel, validates the chosen evidence report, and returns its window ID.
    func chooseEvidenceReport() async -> UUID? {
        guard let url = choose(title: "Open Evidence Report", message: "Choose a Zephyr Mesh evidence report JSON file") else {
            return nil
        }
        return await loadEvidenceReport(url)
    }

    func loadEvidenceReport(_ url: URL) async -> UUID? {
        await load(url, kind: "evidence", errorTitle: "Could not open evidence report") { url in
            .evidenceReport(try EvidenceReport.load(from: url), filename: url.lastPathComponent)
        }
    }

    /// Open documents are tracked by kind and path, so the same file opened as a scenario run
    /// and as an evidence report is validated by each loader on its own and never shares a window.
    private func load(_ url: URL, kind: String, errorTitle: String,
                      decode: @escaping @Sendable (URL) throws -> ViewerDocument) async -> UUID? {
        let key = "\(kind):\(url.standardizedFileURL.resolvingSymlinksInPath().path)"
        if let existing = openURLs[key], documents[existing] != nil { return existing }
        guard !pendingURLs.contains(key) else { return nil }
        pendingURLs.insert(key)
        defer { pendingURLs.remove(key) }
        let result = await Task.detached(priority: .userInitiated) { Result { try decode(url) } }.value
        switch result {
        case .success(let document):
            let id = register(document)
            openURLs[key] = id
            return id
        case .failure(let error):
            openError = OpenError(title: errorTitle, message: error.localizedDescription)
            return nil
        }
    }

    private func register(_ document: ViewerDocument) -> UUID {
        let id = UUID()
        documents[id] = document
        return id
    }

    private func choose(title: String, message: String) -> URL? {
        let panel = NSOpenPanel()
        panel.title = title
        panel.message = message
        panel.prompt = "Open"
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK else { return nil }
        return panel.url
    }
}

@MainActor
enum AlertPresenter {
    /// Wait between checks for a window at launch.
    static let retryDelay: TimeInterval = 0.36

    /// At launch the main window may not be on screen yet; wait briefly for it before falling
    /// back to an app-modal alert.
    /// `allowsModal` false (a notice shown just before the app quits to install an update) never
    /// falls back to a blocking app-modal alert; with no window it is skipped, since the app is
    /// about to restart anyway.
    static func show(title: String, message: String, retries: Int = 3, allowsModal: Bool = true) {
        let window = NSApp.keyWindow ?? NSApp.mainWindow ?? NSApp.windows.first { $0.isVisible && $0.canBecomeMain }
        if window == nil, retries > 0 {
            DispatchQueue.main.asyncAfter(deadline: .now() + retryDelay) {
                show(title: title, message: message, retries: retries - 1, allowsModal: allowsModal)
            }
            return
        }
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        if let window, window.isVisible, window.attachedSheet == nil {
            alert.beginSheetModal(for: window)
        } else if allowsModal {
            alert.runModal()
        }
    }
}
