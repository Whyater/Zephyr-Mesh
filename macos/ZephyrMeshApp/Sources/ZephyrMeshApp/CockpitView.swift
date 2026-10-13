import SwiftUI

/// Main window: team list sidebar, scene detail with status strip on top and timeline plus
/// events below, and the selected-drone inspector as a trailing inspector column.
/// Playback lives in one place, the timeline row under the scene; the toolbar holds only view
/// controls (3D or Top, About this replay, the inspector), grouped with one even gap.
struct CockpitView: View {
    @ObservedObject var model: CockpitModel
    @ObservedObject var store: AppStore
    @Environment(\.openWindow) private var openWindow
    @Environment(\.palette) private var palette
    @State private var detailSize: CGSize = .zero

    var body: some View {
        NavigationSplitView {
            TeamListView(model: model)
                .navigationSplitViewColumnWidth(min: LayoutMetrics.sidebarWidth, ideal: LayoutMetrics.sidebarIdealWidth,
                                                max: LayoutMetrics.sidebarIdealWidth + 40)
        } detail: {
            VStack(spacing: 0) {
                StatusStripView(model: model, availableWidth: detailSize.width)
                SceneContainer(model: model)
                TimelineView(model: model)
                EventListView(model: model, isCompact: detailSize.height < LayoutMetrics.compactDetailHeight)
            }
            .onSizeChange { detailSize = $0 }
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
    }

    @ToolbarContentBuilder private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            HonestyBadge()
        }
        // One item, so AppKit cannot space the controls by glyph width: every gap is Space.sm.
        ToolbarItem(placement: .primaryAction) {
            HStack(spacing: Space.sm) {
                SegmentedChoice(options: CockpitViewMode.allCases.map { ($0, $0.label) }, selection: $model.viewMode,
                                accessibilityLabel: "View", segmentWidth: LayoutMetrics.viewSegmentWidth)
                    .help("3D orbit view or straight-down top view (T)")
                toolbarButton(UIText.aboutButton, symbol: "info.circle", help: UIText.aboutButton) {
                    openWindow(id: WindowID.about)
                }
                toolbarButton("Inspector", symbol: "sidebar.trailing",
                              help: model.showInspector ? "Hide inspector (I)" : "Show inspector (I)",
                              isActive: model.showInspector) {
                    model.showInspector.toggle()
                }
            }
        }
    }

    private func toolbarButton(_ title: String, symbol: String, help: String, isActive: Bool = false,
                               action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: symbol)
                .labelStyle(.iconOnly)
                .imageScale(.medium)
        }
        .buttonStyle(QuietButtonStyle(isActive: isActive, font: TypeScale.headline, tone: .secondary))
        .help(help)
    }
}
