import AppKit
import Combine
import SwiftUI
import UniformTypeIdentifiers

/// App-wide state: the loaded replay (or why it failed) and the documents open in viewer
/// windows. Files are validated by the strict loaders before any window opens.
@MainActor
final class AppStore: ObservableObject {
    enum ViewerDocument {
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
    private var forwarding: AnyCancellable?

    init() {
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
    }

    /// Menus depend on a handful of model values; forward only those so the menu bar does not
    /// rebuild on every hover or frame tick.
    private func forwardCommandState() {
        guard let model = cockpit else { return }
        let signals: [AnyPublisher<Void, Never>] = [
            model.$isPlaying.removeDuplicates().map { _ in () }.eraseToAnyPublisher(),
            model.$isTextFieldFocused.removeDuplicates().map { _ in () }.eraseToAnyPublisher(),
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

    func close(_ id: UUID) { documents[id] = nil }

    /// Shows an open panel, validates the chosen scenario run, and returns its window ID.
    func chooseScenarioRun() -> UUID? {
        guard let url = choose(title: "Open Scenario Run", message: "Choose a generated scenario run JSON file") else { return nil }
        return loadScenarioRun(url)
    }

    func loadScenarioRun(_ url: URL) -> UUID? {
        do {
            let run = try ScenarioRunDocument.load(from: url)
            return register(.scenarioRun(run, filename: url.lastPathComponent))
        } catch {
            openError = OpenError(title: "Could not open scenario run", message: error.localizedDescription)
            return nil
        }
    }

    /// Shows an open panel, validates the chosen evidence report, and returns its window ID.
    func chooseEvidenceReport() -> UUID? {
        guard let url = choose(title: "Open Evidence Report", message: "Choose a Zephyr Mesh evidence report JSON file") else {
            return nil
        }
        return loadEvidenceReport(url)
    }

    func loadEvidenceReport(_ url: URL) -> UUID? {
        do {
            let report = try EvidenceReport.load(from: url)
            return register(.evidenceReport(report, filename: url.lastPathComponent))
        } catch {
            openError = OpenError(title: "Could not open evidence report", message: error.localizedDescription)
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
