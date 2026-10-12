import Foundation

/// Hidden launch setup for reproducible screenshots and manual checks. Every option is read
/// from an environment variable that a normal launch from Finder or `swift run` never sets, so
/// with none of them present this has no effect at all.
///
///   ZEPHYR_SCREENSHOT_FRAME=3      start paused on frame 3 (0-based) instead of auto-playing
///   ZEPHYR_APPEARANCE=light        force light, dark, or system for this launch only (not saved)
///   ZEPHYR_SELECT=drone-009        select a drone
///   ZEPHYR_FOCUS=1                 focus the camera on the selected drone
///   ZEPHYR_VIEW=top                start in top view
///   ZEPHYR_FILTER=D0               prefill the team filter
///   ZEPHYR_INSPECTOR=0             start with the inspector hidden
///   ZEPHYR_OPEN=about              also open: about, scenario:<path>, or evidence:<path>
///   ZEPHYR_WINDOW=1180x760         resize the main window, for example to the minimum size
///   ZEPHYR_SCENE_STATS=1           show SceneKit frame statistics (read in SceneView.swift)
struct ScreenshotSetup {
    let frame: Int?
    let appearance: CockpitAppearance?
    let selectID: String?
    let focus: Bool
    let viewMode: CockpitViewMode?
    let filter: String?
    let showInspector: Bool?
    let open: String?
    let windowSize: CGSize?

    static let current = ScreenshotSetup(environment: ProcessInfo.processInfo.environment)

    init(environment: [String: String]) {
        frame = environment["ZEPHYR_SCREENSHOT_FRAME"].flatMap(Int.init)
        appearance = environment["ZEPHYR_APPEARANCE"].flatMap(CockpitAppearance.init(rawValue:))
        selectID = environment["ZEPHYR_SELECT"]
        focus = environment["ZEPHYR_FOCUS"] == "1"
        viewMode = environment["ZEPHYR_VIEW"].flatMap(CockpitViewMode.init(rawValue:))
        filter = environment["ZEPHYR_FILTER"]
        showInspector = environment["ZEPHYR_INSPECTOR"].map { $0 != "0" }
        open = environment["ZEPHYR_OPEN"]
        let size = environment["ZEPHYR_WINDOW"]?.split(separator: "x").compactMap { Double($0) } ?? []
        windowSize = size.count == 2 ? CGSize(width: size[0], height: size[1]) : nil
    }

    var isActive: Bool {
        frame != nil || appearance != nil || selectID != nil || focus || viewMode != nil || filter != nil
            || showInspector != nil || open != nil || windowSize != nil
    }

    @MainActor
    func apply(to model: CockpitModel) {
        if let frame { model.seek(toFrame: frame) }
        if let selectID { model.select(selectID) }
        if let viewMode { model.viewMode = viewMode }
        if let filter { model.teamFilter = filter }
        if let showInspector { model.showInspector = showInspector }
        if focus { model.focusSelected() }
    }
}
