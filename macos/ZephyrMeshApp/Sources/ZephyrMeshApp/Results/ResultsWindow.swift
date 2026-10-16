import AppKit
import SwiftUI

/// The Results window: a sidebar with the two views, the selected view in the detail column,
/// and the inspector for the selected point or cell. One "Simulated results" badge, in the
/// toolbar. Opened from View > Show Results (Command-R); a single window.
struct ResultsWindowRoot: View {
    @ObservedObject var store: AppStore
    @StateObject private var model = ResultsModel()
    @Environment(\.palette) private var palette

    var body: some View {
        Group {
            if let data = model.data {
                ResultsSplitView(model: model, data: data)
            } else {
                ResultsLoadFailure(message: model.loadError ?? ResultsFormat.notComputed)
            }
        }
        .frame(minWidth: LayoutMetrics.resultsWindowMinWidth, minHeight: LayoutMetrics.resultsWindowMinHeight)
        .background(WindowKeyObserver { store.isResultsWindowKey = $0 })
        .onAppear(perform: applyScreenshotSize)
    }

    /// ZEPHYR_WINDOW resizes this window too when ZEPHYR_OPEN opened it.
    private func applyScreenshotSize() {
        let setup = ScreenshotSetup.current
        guard ResultsPage(openRequest: setup.open) != nil, let size = setup.windowSize else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            NSApp.windows.first { $0.identifier?.rawValue == WindowID.results || $0.title == UIText.Results.windowTitle }?
                .setContentSize(size)
        }
    }
}

private struct ResultsSplitView: View {
    @ObservedObject var model: ResultsModel
    let data: ResultsData
    @Environment(\.palette) private var palette

    var body: some View {
        NavigationSplitView {
            List(selection: Binding(get: { Optional(model.page) }, set: { if let page = $0 { model.page = page } })) {
                ForEach(ResultsPage.allCases) { page in
                    Label(page.title, systemImage: page.symbol)
                        .font(TypeScale.body)
                        .tag(page)
                }
            }
            .listStyle(.sidebar)
            .navigationSplitViewColumnWidth(LayoutMetrics.resultsSidebarWidth)
            .accessibilityLabel(UIText.Results.windowTitle)
        } detail: {
            ResultsDetail(model: model, data: data)
                .background(palette.background)
                .inspector(isPresented: $model.showInspector) {
                    ResultsInspector(model: model, data: data)
                        .inspectorColumnWidth(min: LayoutMetrics.inspectorWidth - Space.lg - Space.xs,
                                              ideal: LayoutMetrics.inspectorWidth, max: LayoutMetrics.inspectorMaxWidth)
                }
        }
        .navigationTitle(UIText.Results.windowTitle)
        .toolbar {
            // Trailing, next to the inspector toggle, so the window title reads first.
            ToolbarItemGroup(placement: .primaryAction) {
                ResultsBadge()
                Button {
                    model.showInspector.toggle()
                } label: {
                    Label(UIText.Menus.showInspector, systemImage: "sidebar.trailing")
                        .labelStyle(.iconOnly)
                        .imageScale(.medium)
                }
                .buttonStyle(QuietButtonStyle(isActive: model.showInspector, font: TypeScale.headline, tone: .secondary))
                .help(UIText.Menus.showInspector)
            }
        }
        .toolbarBackground(palette.surface, for: .windowToolbar)
        .toolbarBackground(.visible, for: .windowToolbar)
    }
}

/// Title, finding, controls, the visual and its legend, then the Limits and Source drawers.
private struct ResultsDetail: View {
    @ObservedObject var model: ResultsModel
    let data: ResultsData
    @Environment(\.palette) private var palette

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Space.xl) {
                VStack(alignment: .leading, spacing: Space.md) {
                    Text(model.page.question)
                        .font(TypeScale.title)
                        .foregroundStyle(palette.text)
                        .accessibilityAddTraits(.isHeader)
                    // The delay view leads with the recovery sentence; the other two sit in a drawer.
                    FindingBlock(findings: model.page == .delay ? data.s6.findings : data.atlas.findings,
                                 lead: model.page == .delay ? 1 : 0)
                        .id(model.page)
                }
                switch model.page {
                case .delay: DelayLossContent(model: model, data: data)
                case .atlas: AtlasContent(model: model, data: data)
                }
                VStack(alignment: .leading, spacing: Space.xs) {
                    HairlineDivider()
                        .padding(.bottom, Space.xs)
                    if model.page == .delay {
                        LimitsDisclosure(limits: data.s6.limits, notes: [
                            ResultsFormat.capitalizedFirst(data.s6.status) + ".",
                            ResultsFormat.capitalizedFirst(data.s6.criterion) + ".",
                            ResultsFormat.cellRule(data.atlas.failureRule)
                        ])
                        SourceDisclosure(source: data.s6.source)
                    } else {
                        LimitsDisclosure(limits: data.atlas.limits, notes: [ResultsFormat.cellRule(data.atlas.failureRule)])
                        SourceDisclosure(source: data.atlas.source)
                    }
                }
                // A new page gets fresh drawers, collapsed again.
                .id(model.page)
            }
            .padding(LayoutMetrics.contentInset + Space.xs)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// "Delay and loss": path and loss type, the four panels, and the legend.
private struct DelayLossContent: View {
    @ObservedObject var model: ResultsModel
    let data: ResultsData
    @Environment(\.palette) private var palette

    /// The shared top of the error axis for the chosen path and loss type.
    private var scaleTop: Double { data.errorScale(motion: model.delay.motion, lossModel: model.delay.lossModel).top }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.lg) {
            HStack(spacing: Space.xl) {
                ControlGroup(label: UIText.Results.pathLabel) {
                    SegmentedChoice(options: [("constant_velocity", UIText.Results.pathStraight),
                                              ("constant_turn", UIText.Results.pathTurn)],
                                    selection: $model.delay.motion, accessibilityLabel: UIText.Results.pathLabel)
                }
                ControlGroup(label: UIText.Results.lossTypeLabel) {
                    SegmentedChoice(options: [("independent", UIText.Results.lossRandom), ("burst", UIText.Results.lossBursty)],
                                    selection: $model.delay.lossModel, accessibilityLabel: UIText.Results.lossTypeLabel)
                }
                // The four panels share one error axis per path and loss type, so say where it ends.
                Text(ResultsFormat.scaleCaption(top: scaleTop))
                    .font(TypeScale.caption.monospacedDigit())
                    .foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .layoutPriority(-1)
            }
            .frame(height: LayoutMetrics.controlHeight)
            // The legend sits above the panels, so it is on screen without scrolling.
            DelayLossLegend(seeds: seedsText)
            DelayLossGrid(model: model, data: data)
        }
    }

    /// The seed count for the legend: one number when every point used the same count.
    private var seedsText: String {
        let counts = Set(data.s6.series.flatMap(\.nSeeds))
        guard let low = counts.min(), let high = counts.max() else { return ResultsFormat.notComputed }
        return low == high ? "\(low)" : "\(low) to \(high)"
    }
}
