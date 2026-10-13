import AppKit
import SwiftUI

enum WindowID {
    static let main = "main"
    static let about = "about-replay"
    static let scenarioRun = "scenario-run"
    static let evidenceReport = "evidence-report"
}

@main
struct ZephyrMeshApp: App {
    @StateObject private var store = AppStore()

    init() {
        if CommandLine.arguments.contains("--self-test") {
            switch FixtureLoader.loadBundled() {
            case .success:
                print("self-test ok")
                exit(0)
            case .failure(let error):
                print("self-test failed: \(error.localizedDescription)")
                exit(1)
            }
        }
        if ScreenshotSetup.current.isActive {
            // A capture launched from a background shell can start hidden, and SwiftUI then never
            // builds the window. Unhide and activate as soon as the run loop starts.
            // Repeated because an unhide sent before launch finishes is ignored.
            for delay in [0.0, 0.4, 1.0, 2.0] {
                DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                    NSApp.unhide(nil)
                    NSApp.activate(ignoringOtherApps: true)
                }
            }
        }
    }

    var body: some Scene {
        Window("Zephyr Mesh", id: WindowID.main) {
            AppearanceRoot {
                MainWindowRoot(store: store)
            }
        }
        .defaultSize(width: LayoutMetrics.windowDefaultWidth, height: LayoutMetrics.windowDefaultHeight)
        .commands { CockpitCommands(store: store) }

        Window(UIText.Menus.aboutReplay, id: WindowID.about) {
            AppearanceRoot {
                if let fixture = store.cockpit?.fixture {
                    AboutReplayView(fixture: fixture)
                        .background(NotRestorable())
                }
            }
        }
        .defaultSize(width: LayoutMetrics.aboutWindowWidth, height: LayoutMetrics.aboutWindowHeight)
        .windowResizability(.contentMinSize)

        WindowGroup("Scenario Run", id: WindowID.scenarioRun, for: UUID.self) { $id in
            AppearanceRoot {
                ViewerWindowRoot(store: store, id: id)
            }
        }
        .defaultSize(width: LayoutMetrics.scenarioWindowWidth, height: LayoutMetrics.scenarioWindowHeight)

        WindowGroup("Evidence Report", id: WindowID.evidenceReport, for: UUID.self) { $id in
            AppearanceRoot {
                ViewerWindowRoot(store: store, id: id)
            }
        }
        .defaultSize(width: LayoutMetrics.evidenceWindowWidth, height: LayoutMetrics.evidenceWindowHeight)

        Settings {
            AppearanceRoot {
                SettingsView()
            }
        }
    }
}

/// Applies the saved appearance (default Dark) to the whole app and resolves the palette.
struct AppearanceRoot<Content: View>: View {
    @AppStorage(CockpitAppearance.storageKey) private var stored = CockpitAppearance.dark.rawValue
    @ViewBuilder let content: () -> Content

    private var appearance: CockpitAppearance {
        ScreenshotSetup.current.appearance ?? CockpitAppearance(rawValue: stored) ?? .dark
    }

    var body: some View {
        ThemedRoot(content: content)
            .onAppear(perform: apply)
            .onChange(of: stored) { _, _ in apply() }
    }

    private func apply() {
        switch appearance {
        case .system: NSApp.appearance = nil
        case .dark: NSApp.appearance = NSAppearance(named: .darkAqua)
        case .light: NSApp.appearance = NSAppearance(named: .aqua)
        }
    }
}

private struct MainWindowRoot: View {
    @ObservedObject var store: AppStore
    @Environment(\.openWindow) private var openWindow
    @Environment(\.palette) private var palette
    @State private var didRunSetup = false

    var body: some View {
        Group {
            if let model = store.cockpit {
                CockpitView(model: model, store: store)
            } else {
                LoadFailureView(message: store.loadError ?? "Unknown error")
            }
        }
        .frame(minWidth: LayoutMetrics.windowMinWidth, minHeight: LayoutMetrics.windowMinHeight)
        .onAppear(perform: runScreenshotSetup)
    }

    private func runScreenshotSetup() {
        guard !didRunSetup else { return }
        didRunSetup = true
        if let size = ScreenshotSetup.current.windowSize {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                NSApp.windows.first { $0.identifier?.rawValue == WindowID.main || $0.title == "Zephyr Mesh" }?
                    .setContentSize(size)
            }
        }
        guard let request = ScreenshotSetup.current.open else { return }
        if request == "about" {
            openWindow(id: WindowID.about)
            return
        }
        Task {
            if request.hasPrefix("scenario:"),
               let id = await store.loadScenarioRun(URL(fileURLWithPath: String(request.dropFirst("scenario:".count)))) {
                openWindow(id: WindowID.scenarioRun, value: id)
            } else if request.hasPrefix("evidence:"),
                      let id = await store.loadEvidenceReport(URL(fileURLWithPath: String(request.dropFirst("evidence:".count)))) {
                openWindow(id: WindowID.evidenceReport, value: id)
            }
        }
    }
}

private struct LoadFailureView: View {
    let message: String
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(spacing: Space.md) {
            StatusShape(style: StatusStyle(level: "alert"), font: TypeScale.title)
            Text("The replay could not be loaded")
                .font(TypeScale.title)
            Text(message)
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
                .multilineTextAlignment(.center)
                .textSelection(.enabled)
                .frame(maxWidth: LayoutMetrics.messageMaxWidth)
            Text("Rebuild the replay fixture and the app, then open it again.")
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
        }
        .padding(Space.xxl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(palette.background)
    }
}

/// Hosts a scenario run or evidence report that was validated before its window opened.
private struct ViewerWindowRoot: View {
    private static let notOpenTitle = Format.label("file_not_open")
    private static let notOpenHint = Format.label("file_not_open_hint")
    @ObservedObject var store: AppStore
    let id: UUID?
    @Environment(\.palette) private var palette
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Group {
            switch id.flatMap(store.document) {
            case .scenarioRun(let run, let filename):
                ScenarioRunView(run: run, filename: filename)
            case .evidenceReport(let report, let filename):
                EvidenceReportView(report: report, filename: filename)
            case .none:
                // Only reachable if a window outlives its document; it fills the window in the
                // app background and closes itself.
                VStack(spacing: Space.sm) {
                    Text(Self.notOpenTitle)
                        .font(TypeScale.headline)
                    Text(Self.notOpenHint)
                        .font(TypeScale.body)
                        .foregroundStyle(palette.textSecondary)
                }
                .frame(minWidth: LayoutMetrics.emptyWindowMinWidth, minHeight: LayoutMetrics.emptyWindowMinHeight)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .onAppear { dismiss() }
            }
        }
        // Viewer documents live only in memory, so macOS must not reopen these windows at launch.
        .background(NotRestorable())
        // Closing the window releases its parsed document.
        .onDisappear { if let id { store.close(id) } }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(palette.background)
        .toolbarBackground(palette.background, for: .windowToolbar)
        .toolbarBackground(.visible, for: .windowToolbar)
    }
}

/// Turns off state restoration for the hosting window, so macOS does not reopen it empty at the
/// next launch. (`restorationBehavior(.disabled)` needs macOS 15; the app supports macOS 14.)
private struct NotRestorable: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async { view.window?.isRestorable = false }
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        view.window?.isRestorable = false
    }
}
