import SwiftUI

/// Main window: team list sidebar, scene detail with status strip on top and timeline plus
/// events below, and the selected-drone inspector as a trailing inspector column.
struct CockpitView: View {
    @ObservedObject var model: CockpitModel
    @ObservedObject var store: AppStore
    @Environment(\.openWindow) private var openWindow
    @Environment(\.palette) private var palette

    var body: some View {
        NavigationSplitView {
            TeamListView(model: model)
                .navigationSplitViewColumnWidth(min: LayoutMetrics.sidebarWidth, ideal: LayoutMetrics.sidebarIdealWidth,
                                                max: LayoutMetrics.sidebarIdealWidth + 40)
        } detail: {
            VStack(spacing: 0) {
                StatusStripView(model: model)
                SceneContainer(model: model)
                TimelineView(model: model)
                EventListView(model: model)
            }
            .background(palette.background)
            .inspector(isPresented: $model.showInspector) {
                InspectorView(model: model)
                    .inspectorColumnWidth(min: LayoutMetrics.inspectorWidth - 20, ideal: LayoutMetrics.inspectorWidth,
                                          max: LayoutMetrics.inspectorMaxWidth)
            }
        }
        .navigationTitle("Zephyr Mesh")
        .navigationSubtitle(model.fixture.runName)
        .toolbar { toolbarContent }
        .toolbarBackground(palette.surface, for: .windowToolbar)
        .toolbarBackground(.visible, for: .windowToolbar)
        .alert(item: $model.updateNotice) { notice in
            Alert(title: Text(notice.title), message: Text(notice.message), dismissButton: .default(Text("OK")))
        }
    }

    @ToolbarContentBuilder private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            HonestyBadge()
        }
        ToolbarItem(placement: .primaryAction) {
            SegmentedChoice(options: CockpitViewMode.allCases.map { ($0, $0.label) }, selection: $model.viewMode,
                            accessibilityLabel: "View")
                .help("3D orbit view or straight-down top view (T)")
        }
        ToolbarItem(placement: .primaryAction) {
            Button {
                openWindow(id: WindowID.about)
            } label: {
                Label(UIText.aboutButton, systemImage: "info.circle")
                    .foregroundStyle(palette.text)
            }
            .help(UIText.aboutButton)
        }
        ToolbarItem(placement: .primaryAction) {
            Button {
                model.showInspector.toggle()
            } label: {
                Label("Inspector", systemImage: "sidebar.trailing")
                    .foregroundStyle(palette.text)
            }
            .help(model.showInspector ? "Hide inspector" : "Show inspector")
        }
    }
}

