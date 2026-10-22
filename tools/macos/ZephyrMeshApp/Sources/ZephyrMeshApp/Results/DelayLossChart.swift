import SwiftUI

/// Panel insets around the plot area: tick labels on the left, the y title above, tick labels
/// and the x title below. Every panel reserves the same room, so all four plots line up.
enum ChartInsets {
    static let leading = LayoutMetrics.chartPlotInset
    static let top = Space.xl
    static let trailing = Space.md
    static let bottom = Space.xl + Space.lg
    static let delayTicks: [Double] = [0, 0.1, 0.2, 0.3]
}

/// The pure drawing inputs for one panel: both trackers' series, aligned with the delays.
struct PanelSeries {
    let delays: [Double]
    let recovery: S6Series?
    let naive: S6Series?

    func series(arm: Int) -> S6Series? { DelayLossLayout.arms[arm] == "recovery" ? recovery : naive }

    /// Marker centers for one tracker, or none when its series is missing.
    func points(arm: Int, scale: ChartScale) -> [CGPoint] {
        guard let series = series(arm: arm) else { return [] }
        return zip(delays, series.p95MedianM).map { scale.point($0, $1) }
    }

    /// The point under the pointer: the nearest marker of either tracker within the hit radius.
    func hit(_ location: CGPoint, scale: ChartScale) -> (arm: Int, index: Int)? {
        var best: (arm: Int, index: Int, distance: CGFloat)?
        for arm in DelayLossLayout.arms.indices {
            let points = points(arm: arm, scale: scale)
            guard let index = ChartGeometry.nearest(to: location, in: points, radius: LayoutMetrics.chartHoverRing) else { continue }
            let distance = hypot(points[index].x - location.x, points[index].y - location.y)
            if let current = best, distance >= current.distance { continue }
            best = (arm, index, distance)
        }
        return best.map { ($0.arm, $0.index) }
    }
}

/// One small-multiple panel: title, then the plot with interval band, caps, both lines, the
/// failure line, and hover and selection rings. Drawn with Canvas; no chart framework.
/// `displayedTop` is animatable, so when Path or Loss type changes the y range the plot glides
/// to the new range over `Motion.standard` (instantly with Reduce Motion).
struct DelayLossPanel: View, Animatable {
    let panel: LossPanel
    let index: Int
    let data: PanelSeries
    let yScale: NiceScale
    var displayedTop: Double
    let failureLine: Double
    let showYTitle: Bool
    let showXTitle: Bool
    let selection: ChartSelection?
    let summary: String
    let onSelect: (ChartSelection) -> Void
    let onAdjust: (ChartSelection.Move) -> Void
    @Environment(\.palette) private var palette
    @State private var hover: (arm: Int, index: Int)?
    @State private var size: CGSize = .zero

    var animatableData: Double {
        get { displayedTop }
        set { displayedTop = newValue }
    }

    private var selectedHere: ChartSelection? { selection?.panel == index ? selection : nil }

    /// Ticks of the target scale that fit the range drawn right now (all of them once settled).
    private var visibleTicks: [Double] { yScale.ticks.filter { $0 <= displayedTop + 1e-9 } }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            Text(panel.title)
                .font(TypeScale.bodyStrong)
                .foregroundStyle(palette.text)
            Canvas { context, size in draw(context, size: size) }
                .frame(height: LayoutMetrics.chartPanelHeight)
                .onSizeChange { size = $0 }
                .overlay(alignment: .topLeading) { tooltip }
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let location): hover = data.hit(location, scale: scale(in: size))
                    case .ended: hover = nil
                    }
                }
                .gesture(SpatialTapGesture().onEnded { value in
                    if let hit = data.hit(value.location, scale: scale(in: size)) {
                        onSelect(ChartSelection(panel: index, arm: hit.arm, delayIndex: hit.index))
                    }
                })
        }
        .padding(Space.md)
        .background(palette.surface, in: RoundedRectangle(cornerRadius: Radius.small))
        .overlay { RoundedRectangle(cornerRadius: Radius.small).strokeBorder(palette.line, lineWidth: 1) }
        .frame(minWidth: LayoutMetrics.chartPanelMinWidth)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(panel.title). \(summary)")
        .accessibilityValue(accessibilityValue)
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: onAdjust(.right)
            case .decrement: onAdjust(.left)
            @unknown default: break
            }
        }
    }

    private var accessibilityValue: String {
        guard let selected = selectedHere, let series = data.series(arm: selected.arm) else { return "" }
        return ResultsFormat.hoverText(series: series, index: selected.delayIndex, delays: data.delays)
    }

    private func scale(in size: CGSize) -> ChartScale {
        let plot = ChartScale.plotRect(in: size, leading: ChartInsets.leading, top: ChartInsets.top,
                                       trailing: ChartInsets.trailing, bottom: ChartInsets.bottom)
        return ChartScale(plot: plot, xMax: ChartInsets.delayTicks.last ?? 0.3, yMax: displayedTop)
    }

    @ViewBuilder private var tooltip: some View {
        if let hover, let series = data.series(arm: hover.arm), size != .zero {
            let point = data.points(arm: hover.arm, scale: scale(in: size))[hover.index]
            ChartTooltip(text: ResultsFormat.hoverText(series: series, index: hover.index, delays: data.delays),
                         anchor: point, bounds: size)
        }
    }

    // MARK: Drawing

    private func draw(_ context: GraphicsContext, size: CGSize) {
        let scale = scale(in: size)
        drawAxes(context, scale: scale, size: size)
        if let recovery = data.recovery {
            let band = ChartGeometry.bandPolygon(delays: data.delays, lows: recovery.p95CILowM, highs: recovery.p95CIHighM, scale: scale)
            if let first = band.first {
                var path = Path()
                path.move(to: first)
                band.dropFirst().forEach { path.addLine(to: $0) }
                path.closeSubpath()
                context.fill(path, with: .color(palette.bandFill))
            }
        }
        drawFailureLine(context, scale: scale)
        if let naive = data.naive {
            drawCaps(context, series: naive, scale: scale)
            drawLine(context, points: zip(data.delays, naive.p95MedianM).map { scale.point($0, $1) }, color: palette.lineStrong,
                     dashed: true)
        }
        if let recovery = data.recovery {
            drawLine(context, points: zip(data.delays, recovery.p95MedianM).map { scale.point($0, $1) }, color: palette.accent,
                     dashed: false)
        }
        drawMarkers(context, scale: scale)
        if let hover {
            ring(context, at: data.points(arm: hover.arm, scale: scale)[hover.index], diameter: LayoutMetrics.chartHoverRing,
                 color: palette.textSecondary, width: 1)
        }
        if let selected = selectedHere, data.series(arm: selected.arm) != nil {
            ring(context, at: data.points(arm: selected.arm, scale: scale)[selected.delayIndex],
                 diameter: LayoutMetrics.chartHoverRing + Space.xs, color: palette.accent, width: Space.xxs)
        }
    }

    private func drawAxes(_ context: GraphicsContext, scale: ChartScale, size: CGSize) {
        let plot = scale.plot
        var grid = Path()
        for tick in visibleTicks.dropFirst() {
            let y = scale.y(tick)
            grid.move(to: CGPoint(x: plot.minX, y: y))
            grid.addLine(to: CGPoint(x: plot.maxX, y: y))
        }
        for tick in ChartInsets.delayTicks.dropFirst() {
            let x = scale.x(tick)
            grid.move(to: CGPoint(x: x, y: plot.minY))
            grid.addLine(to: CGPoint(x: x, y: plot.maxY))
        }
        context.stroke(grid, with: .color(palette.line), lineWidth: 1)

        var axes = Path()
        axes.move(to: CGPoint(x: plot.minX, y: plot.minY))
        axes.addLine(to: CGPoint(x: plot.minX, y: plot.maxY))
        axes.addLine(to: CGPoint(x: plot.maxX, y: plot.maxY))
        for tick in visibleTicks {
            let y = scale.y(tick)
            axes.move(to: CGPoint(x: plot.minX - Space.xs, y: y))
            axes.addLine(to: CGPoint(x: plot.minX, y: y))
            label(context, yScale.label(tick), at: CGPoint(x: plot.minX - Space.xs - Space.xxs, y: y), anchor: .trailing)
        }
        for tick in ChartInsets.delayTicks {
            let x = scale.x(tick)
            axes.move(to: CGPoint(x: x, y: plot.maxY))
            axes.addLine(to: CGPoint(x: x, y: plot.maxY + Space.xs))
            label(context, "\(ResultsFormat.milliseconds(tick))", at: CGPoint(x: x, y: plot.maxY + Space.xs + Space.xxs), anchor: .top)
        }
        context.stroke(axes, with: .color(palette.lineStrong), lineWidth: 1)
        if showYTitle {
            label(context, UIText.Results.axisError, at: CGPoint(x: 0, y: 0), anchor: .topLeading)
        }
        if showXTitle {
            label(context, UIText.Results.axisDelay, at: CGPoint(x: plot.midX, y: size.height), anchor: .bottom)
        }
    }

    /// The red line carries the color; the caption beside it carries the meaning. The caption takes
    /// the first free spot: right end above, right end below, left end above, left end below.
    private func drawFailureLine(_ context: GraphicsContext, scale: ChartScale) {
        let y = scale.y(failureLine)
        var line = Path()
        line.move(to: CGPoint(x: scale.plot.minX, y: y))
        line.addLine(to: CGPoint(x: scale.plot.maxX, y: y))
        context.stroke(line, with: .color(palette.alert), lineWidth: LayoutMetrics.failureLineWidth)
        let text = ResultsFormat.failureLineLabel(failureLine)
        let resolved = context.resolve(Text(text).font(TypeScale.caption).foregroundColor(palette.textSecondary))
        let measured = resolved.measure(in: scale.plot.size)
        let origin = ChartGeometry.captionOrigin(size: measured, lineY: y, plot: scale.plot, gap: Space.xs,
                                                 clearance: LayoutMetrics.chartMarkerSize / 2, segments: drawnSegments(scale))
        context.draw(resolved, at: origin, anchor: .topLeading)
    }

    /// Every line the panel draws for the data: both curves, the band edges and the caps.
    private func drawnSegments(_ scale: ChartScale) -> [(CGPoint, CGPoint)] {
        var segments: [(CGPoint, CGPoint)] = []
        for series in [data.recovery, data.naive].compactMap({ $0 }) {
            for values in [series.p95MedianM, series.p95CILowM, series.p95CIHighM] {
                segments += ChartGeometry.segments(zip(data.delays, values).map { scale.point($0, $1) })
            }
            for (index, delay) in data.delays.enumerated() {
                segments.append((scale.point(delay, series.p95CILowM[index]), scale.point(delay, series.p95CIHighM[index])))
            }
        }
        return segments
    }

    private func drawCaps(_ context: GraphicsContext, series: S6Series, scale: ChartScale) {
        var caps = Path()
        let half = LayoutMetrics.chartCapWidth / 2
        for (index, delay) in data.delays.enumerated() {
            let x = scale.x(delay)
            let low = scale.y(series.p95CILowM[index]), high = scale.y(series.p95CIHighM[index])
            caps.move(to: CGPoint(x: x, y: low))
            caps.addLine(to: CGPoint(x: x, y: high))
            for y in [low, high] {
                caps.move(to: CGPoint(x: x - half, y: y))
                caps.addLine(to: CGPoint(x: x + half, y: y))
            }
        }
        context.stroke(caps, with: .color(palette.lineStrong), lineWidth: 1)
    }

    private func drawLine(_ context: GraphicsContext, points: [CGPoint], color: Color, dashed: Bool) {
        guard let first = points.first else { return }
        var path = Path()
        path.move(to: first)
        points.dropFirst().forEach { path.addLine(to: $0) }
        let style = StrokeStyle(lineWidth: LayoutMetrics.chartLineWidth, lineCap: .round, lineJoin: .round,
                                dash: dashed ? [Space.xs, Space.xs - 1] : [])
        context.stroke(path, with: .color(color), style: style)
    }

    private func drawMarkers(_ context: GraphicsContext, scale: ChartScale) {
        let size = LayoutMetrics.chartMarkerSize
        for point in data.points(arm: DelayLossLayout.arms.firstIndex(of: "naive") ?? 1, scale: scale) {
            let rect = CGRect(x: point.x - size / 2, y: point.y - size / 2, width: size, height: size)
            context.fill(Path(ellipseIn: rect), with: .color(palette.surface))
            context.stroke(Path(ellipseIn: rect.insetBy(dx: 0.5, dy: 0.5)), with: .color(palette.lineStrong), lineWidth: 1.5)
        }
        for point in data.points(arm: DelayLossLayout.arms.firstIndex(of: "recovery") ?? 0, scale: scale) {
            let rect = CGRect(x: point.x - size / 2, y: point.y - size / 2, width: size, height: size)
            context.fill(Path(ellipseIn: rect), with: .color(palette.accent))
        }
    }

    private func ring(_ context: GraphicsContext, at point: CGPoint, diameter: CGFloat, color: Color, width: CGFloat) {
        let rect = CGRect(x: point.x - diameter / 2, y: point.y - diameter / 2, width: diameter, height: diameter)
        context.stroke(Path(ellipseIn: rect), with: .color(color), lineWidth: width)
    }

    private func label(_ context: GraphicsContext, _ text: String, at point: CGPoint, anchor: UnitPoint) {
        context.draw(Text(text).font(TypeScale.caption.monospacedDigit()).foregroundColor(palette.textSecondary),
                     at: point, anchor: anchor)
    }
}

/// The compact hover readout, placed above the point and kept inside the panel.
private struct ChartTooltip: View {
    let text: String
    let anchor: CGPoint
    let bounds: CGSize
    @Environment(\.palette) private var palette
    @State private var tipSize: CGSize = .zero

    var body: some View {
        Text(text)
            .font(TypeScale.caption.monospacedDigit())
            .foregroundStyle(palette.text)
            .lineLimit(1)
            .fixedSize()
            .padding(.horizontal, Space.sm)
            .padding(.vertical, Space.xs)
            .background(palette.background, in: RoundedRectangle(cornerRadius: Radius.small))
            .overlay { RoundedRectangle(cornerRadius: Radius.small).strokeBorder(palette.lineStrong, lineWidth: 1) }
            .onSizeChange { tipSize = $0 }
            .offset(x: offsetX, y: offsetY)
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }

    private var offsetX: CGFloat {
        min(max(anchor.x - tipSize.width / 2, 0), max(bounds.width - tipSize.width, 0))
    }

    /// Above the point when it fits, otherwise below it.
    private var offsetY: CGFloat {
        let above = anchor.y - LayoutMetrics.chartHoverRing - tipSize.height
        return above >= 0 ? above : anchor.y + LayoutMetrics.chartHoverRing
    }
}

/// The 2 x 2 grid of panels. It takes keyboard focus as one control: Left and Right step the
/// delay, Up and Down switch tracker, [ and ] (or Tab and Shift-Tab) move between panels, and
/// Escape clears the selection.
struct DelayLossGrid: View {
    @ObservedObject var model: ResultsModel
    let data: ResultsData
    @FocusState private var focused: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let panels = model.panels
        let yScale = data.errorScale(motion: model.delay.motion, lossModel: model.delay.lossModel)
        let summary = Format.fill(UIText.Results.chartSummary, [
            "path": ResultsFormat.pathName(model.delay.motion).lowercased(),
            "lossType": (model.delay.lossModel == "burst" ? UIText.Results.lossBursty : UIText.Results.lossRandom).lowercased()
        ])
        Grid(horizontalSpacing: Space.lg, verticalSpacing: Space.lg) {
            ForEach(Array(stride(from: 0, to: panels.count, by: 2)), id: \.self) { row in
                GridRow {
                    ForEach(row..<min(row + 2, panels.count), id: \.self) { index in
                        DelayLossPanel(
                            panel: panels[index], index: index,
                            data: PanelSeries(delays: data.s6.delaysS,
                                              recovery: model.series(panel: panels[index], arm: "recovery"),
                                              naive: model.series(panel: panels[index], arm: "naive")),
                            yScale: yScale, displayedTop: yScale.top, failureLine: data.s6.failureP95M,
                            showYTitle: index % 2 == 0, showXTitle: index >= panels.count - 2,
                            selection: model.delay.selection, summary: summary,
                            onSelect: { model.delay.selection = $0 },
                            onAdjust: { move in
                                if model.delay.selection?.panel != index {
                                    model.delay.selection = ChartSelection(panel: index, arm: 0, delayIndex: 0)
                                } else {
                                    model.moveChart(move)
                                }
                            })
                    }
                }
            }
        }
        .animation(.easeInOut(duration: Motion.duration(Motion.standard, reduceMotion: reduceMotion)), value: yScale.top)
        .focusable()
        .focused($focused)
        .focusEffectDisabled()
        .modifier(FocusRing(visible: focused))
        .onKeyPress(keys: [.leftArrow, .rightArrow, .upArrow, .downArrow, .escape, .tab, "[", "]"]) { press in
            handle(press)
        }
    }

    private func handle(_ press: KeyPress) -> KeyPress.Result {
        let move: ChartSelection.Move
        switch press.key {
        case .leftArrow: move = .left
        case .rightArrow: move = .right
        case .upArrow: move = .up
        case .downArrow: move = .down
        case "[": move = .previousPanel
        case "]": move = .nextPanel
        case .tab: move = press.modifiers.contains(.shift) ? .previousPanel : .nextPanel
        case .escape:
            guard model.delay.selection != nil else { return .ignored }
            model.delay.selection = nil
            return .handled
        default: return .ignored
        }
        return model.moveChart(move) ? .handled : .ignored
    }
}

/// Legend above the charts: shape and line style plus a word, never color alone.
struct DelayLossLegend: View {
    let seeds: String
    @Environment(\.palette) private var palette

    var body: some View {
        HStack(spacing: Space.lg) {
            LegendItem(title: UIText.Results.armRecovery) { LineSwatch(color: palette.accent, dashed: false, filled: true) }
            LegendItem(title: UIText.Results.armNaive) { LineSwatch(color: palette.lineStrong, dashed: true, filled: false) }
            LegendItem(title: Format.fill(UIText.Results.legendInterval, ["n": seeds])) {
                RoundedRectangle(cornerRadius: Space.xxs).fill(palette.bandFill)
            }
            LegendItem(title: UIText.Results.legendFailure) {
                Rectangle().fill(palette.alert).frame(height: LayoutMetrics.failureLineWidth)
            }
        }
        .foregroundStyle(palette.text)
    }
}

/// A short line with a marker in the middle, as drawn on the charts.
private struct LineSwatch: View {
    let color: Color
    let dashed: Bool
    let filled: Bool
    @Environment(\.palette) private var palette

    var body: some View {
        Canvas { context, size in
            var line = Path()
            line.move(to: CGPoint(x: 0, y: size.height / 2))
            line.addLine(to: CGPoint(x: size.width, y: size.height / 2))
            context.stroke(line, with: .color(color), style: StrokeStyle(lineWidth: LayoutMetrics.chartLineWidth,
                                                                          dash: dashed ? [Space.xs, Space.xs - 1] : []))
            let marker = LayoutMetrics.chartMarkerSize
            let rect = CGRect(x: size.width / 2 - marker / 2, y: size.height / 2 - marker / 2, width: marker, height: marker)
            if filled {
                context.fill(Path(ellipseIn: rect), with: .color(color))
            } else {
                context.fill(Path(ellipseIn: rect), with: .color(palette.background))
                context.stroke(Path(ellipseIn: rect.insetBy(dx: 0.5, dy: 0.5)), with: .color(color), lineWidth: 1.5)
            }
        }
    }
}
