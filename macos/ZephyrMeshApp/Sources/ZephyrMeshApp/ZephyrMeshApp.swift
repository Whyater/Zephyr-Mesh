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
    @ObservedObject var store: AppStore
    let id: UUID?
    @Environment(\.palette) private var palette

    var body: some View {
        Group {
            switch id.flatMap(store.document) {
            case .scenarioRun(let run, let filename):
                ScenarioRunView(run: run, filename: filename)
            case .evidenceReport(let report, let filename):
                EvidenceReportView(report: report, filename: filename)
            case .none:
                VStack(spacing: Space.sm) {
                    Text("This file is no longer open")
                        .font(TypeScale.headline)
                    Text("Open it again from the File menu.")
                        .font(TypeScale.body)
                        .foregroundStyle(palette.textSecondary)
                }
                .frame(minWidth: LayoutMetrics.emptyWindowMinWidth, minHeight: LayoutMetrics.emptyWindowMinHeight)
            }
        }
        // Closing the window releases its parsed document.
        .onDisappear { if let id { store.close(id) } }
        .background(palette.background)
        .toolbarBackground(palette.background, for: .windowToolbar)
        .toolbarBackground(.visible, for: .windowToolbar)
    }
}
