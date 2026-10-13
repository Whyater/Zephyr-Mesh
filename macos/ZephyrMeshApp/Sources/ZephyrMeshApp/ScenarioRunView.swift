import SwiftUI

/// Scenario run window: the run status and the plot of the selected frame first, then a plain
/// summary of the frame and the run parameters, then collapsed Details with hashes, provenance
/// and the raw parameter and provenance trees. No per-frame text dump. When the window is wide
/// the plot keeps the left column and the summary scrolls in a column beside it, so nothing
/// important sits below the fold.
struct ScenarioRunView: View {
    private static let runStatusLabel = Format.label("run_status")
    let run: ScenarioRunDocument
    let filename: String
    @Environment(\.palette) private var palette
    @State private var cursor: ScenarioRunCursor
    @State private var showDetails = false

    init(run: ScenarioRunDocument, filename: String) {
        self.run = run
        self.filename = filename
        _cursor = State(initialValue: ScenarioRunCursor(frameCount: run.frames.count))
    }

    private var frame: ScenarioRunFrame { run.frames[cursor.index] }

    var body: some View {
        GeometryReader { proxy in
            if proxy.size.width >= LayoutMetrics.viewerTwoColumnWidth {
                HStack(alignment: .top, spacing: Space.xl) {
                    plotColumn(fillsHeight: true)
                        .padding([.leading, .vertical], Space.xl)
                    ScrollView {
                        summary
                            .padding(.vertical, Space.xl)
                            .padding(.trailing, Space.xl)
                    }
                    .frame(width: LayoutMetrics.scenarioSideWidth)
                }
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: Space.xl) {
                        plotColumn(fillsHeight: false)
                        summary
                    }
                    .modifier(ViewerContent())
                }
            }
        }
        .frame(minWidth: LayoutMetrics.scenarioWindowMinWidth, minHeight: LayoutMetrics.scenarioWindowMinHeight)
        .background(palette.background)
        .navigationTitle("Scenario Run")
        .navigationSubtitle(filename)
        .id(run.runIdentity)
    }

    private func plotColumn(fillsHeight: Bool) -> some View {
        VStack(alignment: .leading, spacing: Space.sm) {
            Text(heading)
                .font(TypeScale.headline)
                .accessibilityAddTraits(.isHeader)
            boundaryStrip
            ScenarioPlotView(geometry: ScenarioPlotGeometry.make(frame: frame))
                .frame(minHeight: LayoutMetrics.scenarioPlotHeight, maxHeight: fillsHeight ? .infinity : LayoutMetrics.scenarioPlotHeight)
            if run.frames.count > 1 {
                frameControls
            }
            PlotLegend()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var heading: String {
        let time = Format.seconds(frame.time)
        if run.frames.count < 2 { return "Positions at \(time), \(Format.label("single_frame").lowercased())" }
        return "Positions at \(time), frame \(cursor.index + 1) of \(run.frames.count)"
    }

    /// The run's own declared status, above the plot, so what this is never sits below the fold.
    private var boundaryStrip: some View {
        HStack(alignment: .firstTextBaseline, spacing: Space.sm) {
            StatusShape(style: StatusStyle(level: "info"))
            Text(Self.runStatusLabel)
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
            Text(sentence(run.status))
                .font(TypeScale.bodyStrong)
                .foregroundStyle(palette.text)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, Space.md)
        .padding(.vertical, Space.xs + Space.xxs)
        .frame(minHeight: LayoutMetrics.controlHeight)
        .background(palette.surface, in: RoundedRectangle(cornerRadius: Radius.small))
        .overlay { RoundedRectangle(cornerRadius: Radius.small).strokeBorder(palette.line, lineWidth: 1) }
        .accessibilityElement(children: .combine)
    }

    private var frameControls: some View {
        HStack(spacing: Space.sm) {
            HStack(spacing: Space.xs) {
                Button { cursor.step(-1) } label: {
                    Label("Previous frame", systemImage: "backward.frame.fill").labelStyle(.iconOnly)
                }
                .buttonStyle(QuietButtonStyle(font: TypeScale.headline))
                .disabled(cursor.index == 0)
                .help("Previous frame")
                Button { cursor.step() } label: {
                    Label("Next frame", systemImage: "forward.frame.fill").labelStyle(.iconOnly)
                }
                .buttonStyle(QuietButtonStyle(font: TypeScale.headline))
                .disabled(cursor.index >= run.frames.count - 1)
                .help("Next frame")
            }
            // Glyph ink, not the hit square, lines up with the plot's left edge.
            .padding(.leading, -QuietButtonStyle.glyphInset)
            IndexScrubber(index: Binding(get: { cursor.index }, set: { cursor.seek($0) }), count: run.frames.count)
        }
    }

    private var summary: some View {
        VStack(alignment: .leading, spacing: Space.xl) {
            SectionBlock(title: "Selected frame") {
                ValueRow(label: Format.label("time_s"), value: Format.number(frame.time, decimals: 2))
                ValueRow(label: Format.label("target_position_m"),
                         value: frame.target.map { Format.number($0, decimals: 2) }.joined(separator: ", "))
                ValueRow(label: Format.label("active_count"), value: "\(frame.activeCount) of \(frame.agentCount)")
                ValueRow(label: Format.label("keep_out_volumes"), value: "\(frame.keepOutSpheres.count)")
            }
            SectionBlock(title: "Run") {
                VStack(alignment: .leading, spacing: Space.sm) {
                    StackedValue(label: Format.label("status"), value: sentence(run.status))
                    StackedValue(label: Format.label("evidence_boundary"), value: sentence(run.evidenceBoundary))
                }
                .padding(.bottom, Space.xs)
                ForEach(["agent_count", "steps", "dt_s", "seed", "radius_m", "altitude_m", "target_velocity_mps"],
                        id: \.self) { key in
                    ValueRow(label: Format.label(key), value: run.parameters[key]?.plainText ?? Format.notRecorded)
                }
                ValueRow(label: Format.label("keep_out_spheres"), value: run.parameters["keep_out_spheres"]?.branchSummary
                         ?? Format.notRecorded)
            }
            linkSection("target_link")
            linkSection("neighbor_link")
            DetailsSection(isExpanded: $showDetails) {
                ValueRow(label: "File", value: filename)
                ValueRow(label: Format.label("schema"), value: "\(run.schema), version \(run.version)", style: .identifier)
                ValueRow(label: Format.label("scenario_id"), value: run.scenarioID, style: .identifier)
                StackedValue(label: Format.label("scenario_hash"), value: run.scenarioHash, style: .identifier)
                StackedValue(label: Format.label("payload_sha256"), value: run.payloadSHA256, style: .identifier)
                ValueRow(label: Format.label("link_event_count"), value: "\(run.linkEventCount)")
                ValueRow(label: Format.label("profile_count"), value: "\(run.profileCount)")
                ValueRow(label: Format.label("generator"), value: run.generatorText, style: .code)
                ValueRow(label: Format.label("python"), value: run.pythonText)
                ValueRow(label: Format.label("numpy"), value: run.numpyText)
                ValueRow(label: Format.label("code_revision"), value: run.codeRevisionText, style: .identifier)
                Text("A -dirty suffix on the revision means uncommitted changes; -unchecked means the check could not run.")
                    .font(TypeScale.caption)
                    .foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                // Frames, link events and profiles stay out: they can be huge and their counts are above.
                RawTreeBlock(title: Format.label("raw_parameters"), value: .object(run.parameters))
                RawTreeBlock(title: Format.label("raw_provenance"), value: .object(run.provenance))
            }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// A link block, or nothing when the run has no such link object.
    @ViewBuilder private func linkSection(_ key: String) -> some View {
        if case .object(let link)? = run.parameters[key], !link.isEmpty {
            SectionBlock(title: Format.label(key)) {
                ForEach(link.keys.sorted(), id: \.self) { field in
                    ValueRow(label: Format.label(field), value: sentence(link[field]?.plainText ?? Format.notRecorded))
                }
            }
        }
    }

    private func sentence(_ text: String) -> String {
        text.prefix(1).uppercased() + text.dropFirst()
    }
}

/// Frame scrubber drawn like the main window's: full-width track, accent fill and thumb.
private struct IndexScrubber: View {
    @Binding var index: Int
    let count: Int
    @Environment(\.palette) private var palette

    var body: some View {
        GeometryReader { geometry in
            let thumb = LayoutMetrics.scrubberThumbSize
            let track = LayoutMetrics.scrubberTrackHeight
            let travel = max(geometry.size.width - thumb, 1)
            let progress = count > 1 ? CGFloat(index) / CGFloat(count - 1) : 0
            let midY = geometry.size.height / 2
            ZStack(alignment: .topLeading) {
                Capsule().fill(palette.line)
                    .frame(width: geometry.size.width, height: track)
                    .offset(y: midY - track / 2)
                Capsule().fill(palette.accent)
                    .frame(width: thumb / 2 + travel * progress, height: track)
                    .offset(y: midY - track / 2)
                Circle().fill(palette.accent)
                    .overlay { Circle().strokeBorder(palette.surface, lineWidth: Space.xxs) }
                    .frame(width: thumb, height: thumb)
                    .offset(x: travel * progress, y: midY - thumb / 2)
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                let fraction = min(max((value.location.x - thumb / 2) / travel, 0), 1)
                index = Int((fraction * CGFloat(count - 1)).rounded())
            })
        }
        .frame(height: LayoutMetrics.controlHeight)
        .accessibilityElement()
        .accessibilityLabel("Scenario frame")
        .accessibilityValue("Frame \(index + 1) of \(count)")
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: index = min(index + 1, count - 1)
            case .decrement: index = max(index - 1, 0)
            @unknown default: break
            }
        }
    }
}

/// Top-down positions on the left, with a faint grid and a scale bar, and an altitude rail on
/// the right with its end values, for one frame.
struct ScenarioPlotView: View {
    let geometry: ScenarioPlotGeometry
    @Environment(\.palette) private var palette

    private static let railInset = Double(Space.xl + Space.sm - Space.xxs)
    private static let altitudeTitle = Format.label("altitude_m")

    var body: some View {
        Canvas { context, size in
            let railSpace = LayoutMetrics.scenarioRailWidth
            let plotWidth = max(size.width - railSpace, 1)
            let bounds = geometry.bounds
            let scale = ScenarioPlotGeometry.planScale(bounds: bounds, width: plotWidth, height: size.height)
            drawGrid(context, bounds: bounds, plotWidth: plotWidth, height: size.height, scale: scale)
            for sphere in geometry.keepOutSpheres {
                let center = ScenarioPlotGeometry.project(sphere.center, bounds: bounds, width: plotWidth, height: size.height)
                let radius = ScenarioPlotGeometry.projectRadius(sphere.radius, bounds: bounds, width: plotWidth, height: size.height)
                let rect = CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2)
                context.fill(Path(ellipseIn: rect), with: .color(palette.keepOut.opacity(0.14)))
                context.stroke(Path(ellipseIn: rect), with: .color(palette.keepOut), lineWidth: 1.5)
            }
            let target = ScenarioPlotGeometry.project(geometry.target, bounds: bounds, width: plotWidth, height: size.height)
            let targetMark = LayoutMetrics.plotTargetMark
            context.stroke(Path(ellipseIn: CGRect(x: target.x - targetMark / 2, y: target.y - targetMark / 2,
                                                  width: targetMark, height: targetMark)),
                           with: .color(palette.target), lineWidth: 2)
            // Drones last, with a surface ring, so a drone on the target stays visible.
            let droneMark = LayoutMetrics.plotDroneMark
            for agent in geometry.agents {
                let point = ScenarioPlotGeometry.project(agent.point, bounds: bounds, width: plotWidth, height: size.height)
                drawDrone(context, at: CGPoint(x: point.x, y: point.y), size: droneMark, active: agent.active)
            }
            drawRail(context, size: size, railSpace: railSpace)
        }
        .background(palette.surface, in: RoundedRectangle(cornerRadius: Radius.small))
        .overlay { RoundedRectangle(cornerRadius: Radius.small).strokeBorder(palette.lineStrong.opacity(0.5), lineWidth: 1) }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Plot of the selected frame seen from above, with an altitude rail")
        .accessibilityValue("\(geometry.agents.count) drones, \(geometry.keepOutSpheres.count) keep-out volumes, target altitude "
                            + "\(Format.number(geometry.targetAltitude, decimals: 2)) meters")
    }

    private func drawDrone(_ context: GraphicsContext, at point: CGPoint, size: CGFloat, active: Bool) {
        let rect = CGRect(x: point.x - size / 2, y: point.y - size / 2, width: size, height: size)
        context.stroke(Path(ellipseIn: rect.insetBy(dx: -1.5, dy: -1.5)), with: .color(palette.surface), lineWidth: 2)
        if active {
            context.fill(Path(ellipseIn: rect), with: .color(palette.textSecondary))
        } else {
            // Inactive is a state, not an alarm in this plot: a hollow neutral marker, named in the legend.
            context.stroke(Path(ellipseIn: rect.insetBy(dx: 0.75, dy: 0.75)), with: .color(palette.textSecondary), lineWidth: 1.5)
        }
    }

    /// Grid lines at the scale-bar step, then the bar itself in the bottom-left corner.
    private func drawGrid(_ context: GraphicsContext, bounds: ScenarioPlotBounds, plotWidth: Double, height: Double,
                          scale: Double) {
        let step = ScenarioPlotGeometry.scaleBarLength(pointsPerMeter: scale, maxPoints: plotWidth / 4)
        let centerX = (bounds.minX + bounds.maxX) / 2
        let centerY = (bounds.minY + bounds.maxY) / 2
        let minX = centerX - plotWidth / 2 / scale, maxX = centerX + plotWidth / 2 / scale
        let minY = centerY - height / 2 / scale, maxY = centerY + height / 2 / scale
        var grid = Path()
        for index in Int((minX / step).rounded(.up))...max(Int((minX / step).rounded(.up)), Int((maxX / step).rounded(.down))) {
            let x = plotWidth / 2 + (Double(index) * step - centerX) * scale
            grid.move(to: CGPoint(x: x, y: 0))
            grid.addLine(to: CGPoint(x: x, y: height))
        }
        for index in Int((minY / step).rounded(.up))...max(Int((minY / step).rounded(.up)), Int((maxY / step).rounded(.down))) {
            let y = height / 2 - (Double(index) * step - centerY) * scale
            grid.move(to: CGPoint(x: 0, y: y))
            grid.addLine(to: CGPoint(x: plotWidth, y: y))
        }
        context.stroke(grid, with: .color(palette.line), lineWidth: 1)

        let inset = Double(LayoutMetrics.contentInset)
        let barY = height - inset
        let barLength = step * scale
        var bar = Path()
        bar.move(to: CGPoint(x: inset, y: barY - Space.xs))
        bar.addLine(to: CGPoint(x: inset, y: barY))
        bar.addLine(to: CGPoint(x: inset + barLength, y: barY))
        bar.addLine(to: CGPoint(x: inset + barLength, y: barY - Space.xs))
        context.stroke(bar, with: .color(palette.textSecondary), lineWidth: 1.5)
        let label = step >= 1 ? Format.number(step, decimals: 0) : Format.number(step, decimals: 1)
        context.draw(Text("\(label) m").font(TypeScale.caption).foregroundColor(palette.textSecondary),
                     at: CGPoint(x: inset + barLength / 2, y: barY - Space.sm), anchor: .bottom)
    }

    private func drawRail(_ context: GraphicsContext, size: CGSize, railSpace: CGFloat) {
        let inset = Self.railInset
        let railX = ScenarioPlotGeometry.clampRailX(width: size.width, preferred: size.width - railSpace / 2 + Space.sm)
        let rail = ScenarioPlotGeometry.railEndpoints(height: size.height, topInset: inset, bottomInset: inset)
        context.stroke(Path { path in
            path.move(to: CGPoint(x: railX, y: rail.top))
            path.addLine(to: CGPoint(x: railX, y: rail.bottom))
        }, with: .color(palette.lineStrong), lineWidth: 1)
        context.draw(Text(Self.altitudeTitle).font(TypeScale.caption).foregroundColor(palette.textSecondary),
                     at: CGPoint(x: size.width - Space.sm, y: Space.md), anchor: .trailing)
        // End values: the rail spans the bounds' middle altitude plus or minus half a span of at least 1 m.
        let bounds = geometry.altitudeBounds
        let half = max(bounds.maxZ - bounds.minZ, 1) / 2
        let middle = (bounds.minZ + bounds.maxZ) / 2
        for (value, y) in [(middle + half, rail.top), (middle - half, rail.bottom)] {
            var tick = Path()
            tick.move(to: CGPoint(x: railX - Space.xs, y: y))
            tick.addLine(to: CGPoint(x: railX, y: y))
            context.stroke(tick, with: .color(palette.lineStrong), lineWidth: 1)
            context.draw(Text(Format.number(value, decimals: 2)).font(TypeScale.caption).foregroundColor(palette.textSecondary),
                         at: CGPoint(x: railX - Space.sm, y: y), anchor: .trailing)
        }
        for sphere in geometry.keepOutSpheres {
            let centerY = ScenarioPlotGeometry.projectAltitude(sphere.centerZ, bounds: bounds, height: size.height, inset: inset)
            let radiusY = ScenarioPlotGeometry.projectAltitudeRadius(sphere.radius, bounds: bounds, height: size.height, inset: inset)
            context.stroke(Path { path in
                path.move(to: CGPoint(x: railX + Space.sm, y: centerY - radiusY))
                path.addLine(to: CGPoint(x: railX + Space.sm, y: centerY + radiusY))
            }, with: .color(palette.keepOut), lineWidth: 3)
        }
        // The target tick sits left of the rail, so a drone at the same altitude does not hide it.
        let targetY = ScenarioPlotGeometry.projectAltitude(geometry.targetAltitude, bounds: bounds, height: size.height, inset: inset)
        context.stroke(Path { path in
            path.move(to: CGPoint(x: railX - Space.md - Space.xs, y: targetY))
            path.addLine(to: CGPoint(x: railX - Space.xs, y: targetY))
        }, with: .color(palette.target), lineWidth: 2)
        for (index, altitude) in geometry.agentAltitudes.enumerated() {
            let y = ScenarioPlotGeometry.projectAltitude(altitude, bounds: bounds, height: size.height, inset: inset)
            drawDrone(context, at: CGPoint(x: railX, y: y), size: LayoutMetrics.plotDroneMark - Space.xxs,
                      active: geometry.agents[index].active)
        }
    }
}

private struct PlotLegend: View {
    @Environment(\.palette) private var palette

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: Space.lg) {
                items
                Spacer(minLength: Space.sm)
                note
            }
            VStack(alignment: .leading, spacing: Space.xs) {
                HStack(spacing: Space.lg) { items }
                note
            }
        }
        .font(TypeScale.caption)
    }

    @ViewBuilder private var items: some View {
        item(Circle().stroke(palette.target, lineWidth: 2), size: LayoutMetrics.plotTargetMark, "Target")
        item(Circle().fill(palette.textSecondary), size: LayoutMetrics.plotDroneMark, "Drone")
        item(Circle().stroke(palette.textSecondary, lineWidth: 1.5), size: LayoutMetrics.plotDroneMark, "Inactive")
        item(Circle().stroke(palette.keepOut, lineWidth: 1.5), size: LayoutMetrics.plotDroneMark, "Keep-out volume")
    }

    private var note: some View {
        Text("Seen from above; the rail shows altitude")
            .foregroundStyle(palette.textSecondary)
            .lineLimit(1)
    }

    private func item<Mark: View>(_ mark: Mark, size: CGFloat, _ title: String) -> some View {
        HStack(spacing: Space.xs) {
            mark.frame(width: size, height: size)
            Text(title)
                .lineLimit(1)
                .fixedSize()
        }
    }
}
