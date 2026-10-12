import SwiftUI

/// Scenario run window: the plot of the selected frame first, then a plain summary of the run
/// parameters, then collapsed Details with hashes and provenance. No per-frame text dump.
struct ScenarioRunView: View {
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
        ScrollView {
            VStack(alignment: .leading, spacing: Space.xl) {
                VStack(alignment: .leading, spacing: Space.sm) {
                    Text("Positions at \(Format.seconds(frame.time)), frame \(cursor.index + 1) of \(run.frames.count)")
                        .font(TypeScale.headline)
                        .accessibilityAddTraits(.isHeader)
                    ScenarioPlotView(geometry: ScenarioPlotGeometry.make(frame: frame))
                        .frame(height: 340)
                    PlotLegend()
                    frameControls
                }
                section("Selected frame") {
                    ValueRow(label: Format.label("time_s"), value: Format.number(frame.time, decimals: 2))
                    ValueRow(label: Format.label("target_position_m"),
                             value: frame.target.map { Format.number($0, decimals: 2) }.joined(separator: ", "))
                    ValueRow(label: Format.label("active_count"), value: "\(frame.activeCount) of \(frame.agentCount)")
                    ValueRow(label: Format.label("keep_out_volumes"), value: "\(frame.keepOutSpheres.count)")
                }
                section("Run") {
                    StackedValue(label: Format.label("status"), value: sentence(run.status))
                    StackedValue(label: Format.label("evidence_boundary"), value: sentence(run.evidenceBoundary))
                    ForEach(["agent_count", "steps", "dt_s", "seed", "radius_m", "altitude_m", "target_velocity_mps"],
                            id: \.self) { key in
                        ValueRow(label: Format.label(key), value: run.parameters[key]?.plainText ?? Format.notRecorded)
                    }
                    ValueRow(label: Format.label("keep_out_spheres"), value: run.parameters["keep_out_spheres"]?.branchSummary
                             ?? Format.notRecorded)
                }
                linkSection("target_link")
                linkSection("neighbor_link")
                details
            }
            .padding(Space.xl)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .frame(minWidth: 620, minHeight: 520)
        .background(palette.background)
        .navigationTitle("Scenario Run")
        .navigationSubtitle(filename)
        .id(run.runIdentity)
    }

    private var frameControls: some View {
        HStack(spacing: Space.sm) {
            Button { cursor.step(-1) } label: { Label("Previous frame", systemImage: "backward.frame.fill").labelStyle(.iconOnly) }
                .buttonStyle(QuietButtonStyle())
                .disabled(cursor.index == 0)
                .help("Previous frame")
            Button { cursor.step() } label: { Label("Next frame", systemImage: "forward.frame.fill").labelStyle(.iconOnly) }
                .buttonStyle(QuietButtonStyle())
                .disabled(cursor.index >= run.frames.count - 1)
                .help("Next frame")
            Slider(value: Binding(get: { Double(cursor.index) }, set: { cursor.seek(Int($0.rounded())) }),
                   in: 0...Double(max(1, run.frames.count - 1)))
                .disabled(run.frames.count < 2)
                .accessibilityLabel("Scenario frame")
                .accessibilityValue("Frame \(cursor.index + 1) of \(run.frames.count)")
            Text("Frame \(cursor.index + 1) of \(run.frames.count)")
                .font(TypeScale.body.monospacedDigit())
                .foregroundStyle(palette.textSecondary)
                .fixedSize()
        }
    }

    private func linkSection(_ key: String) -> some View {
        section(Format.label(key)) {
            if case .object(let link)? = run.parameters[key] {
                ForEach(link.keys.sorted(), id: \.self) { field in
                    ValueRow(label: Format.label(field), value: sentence(link[field]?.plainText ?? Format.notRecorded))
                }
            }
        }
    }

    private var details: some View {
        DisclosureGroup(isExpanded: $showDetails) {
            VStack(alignment: .leading, spacing: Space.sm) {
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
            }
            .padding(.top, Space.sm)
        } label: {
            Text("Details").font(TypeScale.headline)
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            Text(title)
                .font(TypeScale.headline)
                .padding(.bottom, Space.xxs)
                .accessibilityAddTraits(.isHeader)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func sentence(_ text: String) -> String {
        text.prefix(1).uppercased() + text.dropFirst()
    }
}

/// Top-down positions on the left and an altitude rail on the right, for one frame.
struct ScenarioPlotView: View {
    let geometry: ScenarioPlotGeometry
    @Environment(\.palette) private var palette

    var body: some View {
        Canvas { context, size in
            let railSpace: CGFloat = 72
            let plotWidth = max(size.width - railSpace, 1)
            let bounds = geometry.bounds
            for sphere in geometry.keepOutSpheres {
                let center = ScenarioPlotGeometry.project(sphere.center, bounds: bounds, width: plotWidth, height: size.height)
                let radius = ScenarioPlotGeometry.projectRadius(sphere.radius, bounds: bounds, width: plotWidth, height: size.height)
                let rect = CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2)
                context.fill(Path(ellipseIn: rect), with: .color(palette.keepOut.opacity(0.14)))
                context.stroke(Path(ellipseIn: rect), with: .color(palette.keepOut), lineWidth: 1.5)
            }
            for agent in geometry.agents {
                let point = ScenarioPlotGeometry.project(agent.point, bounds: bounds, width: plotWidth, height: size.height)
                if agent.active {
                    context.fill(Path(ellipseIn: CGRect(x: point.x - 3.5, y: point.y - 3.5, width: 7, height: 7)),
                                 with: .color(palette.textSecondary))
                } else {
                    // Inactive is a state, not an alarm in this plot: a hollow neutral marker, named in the legend.
                    context.stroke(Path(ellipseIn: CGRect(x: point.x - 3.5, y: point.y - 3.5, width: 7, height: 7)),
                                   with: .color(palette.textSecondary), lineWidth: 1.5)
                }
            }
            let target = ScenarioPlotGeometry.project(geometry.target, bounds: bounds, width: plotWidth, height: size.height)
            context.stroke(Path(ellipseIn: CGRect(x: target.x - 6, y: target.y - 6, width: 12, height: 12)),
                           with: .color(palette.target), lineWidth: 2)

            let railX = ScenarioPlotGeometry.clampRailX(width: size.width, preferred: size.width - railSpace / 2)
            let rail = ScenarioPlotGeometry.railEndpoints(height: size.height)
            context.stroke(Path { path in
                path.move(to: CGPoint(x: railX, y: rail.top))
                path.addLine(to: CGPoint(x: railX, y: rail.bottom))
            }, with: .color(palette.lineStrong), lineWidth: 1)
            context.draw(Text("Altitude").font(TypeScale.caption).foregroundColor(palette.textSecondary),
                         at: CGPoint(x: railX, y: 14), anchor: .center)
            for sphere in geometry.keepOutSpheres {
                let centerY = ScenarioPlotGeometry.projectAltitude(sphere.centerZ, bounds: geometry.altitudeBounds,
                                                                   height: size.height, inset: 30)
                let radiusY = ScenarioPlotGeometry.projectAltitudeRadius(sphere.radius, bounds: geometry.altitudeBounds,
                                                                         height: size.height, inset: 30)
                context.stroke(Path { path in
                    path.move(to: CGPoint(x: railX + 10, y: centerY - radiusY))
                    path.addLine(to: CGPoint(x: railX + 10, y: centerY + radiusY))
                }, with: .color(palette.keepOut), lineWidth: 3)
            }
            for (index, altitude) in geometry.agentAltitudes.enumerated() {
                let y = ScenarioPlotGeometry.projectAltitude(altitude, bounds: geometry.altitudeBounds, height: size.height, inset: 30)
                let marker = Path(ellipseIn: CGRect(x: railX - 3, y: y - 3, width: 6, height: 6))
                if geometry.agents[index].active {
                    context.fill(marker, with: .color(palette.textSecondary))
                } else {
                    context.stroke(marker, with: .color(palette.textSecondary), lineWidth: 1.5)
                }
            }
            let targetY = ScenarioPlotGeometry.projectAltitude(geometry.targetAltitude, bounds: geometry.altitudeBounds,
                                                               height: size.height, inset: 30)
            context.stroke(Path { path in
                path.move(to: CGPoint(x: railX - 8, y: targetY))
                path.addLine(to: CGPoint(x: railX + 8, y: targetY))
            }, with: .color(palette.target), lineWidth: 2)
        }
        .background(palette.surface, in: RoundedRectangle(cornerRadius: Radius.small))
        .overlay { RoundedRectangle(cornerRadius: Radius.small).strokeBorder(palette.line, lineWidth: 1) }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Plot of the selected frame seen from above, with an altitude rail")
        .accessibilityValue("\(geometry.agents.count) drones, \(geometry.keepOutSpheres.count) keep-out volumes, target altitude "
                            + "\(Format.number(geometry.targetAltitude, decimals: 2)) meters")
    }
}

private struct PlotLegend: View {
    @Environment(\.palette) private var palette

    var body: some View {
        HStack(spacing: Space.lg) {
            item(Circle().stroke(palette.target, lineWidth: 2), "Target")
            item(Circle().fill(palette.textSecondary), "Drone")
            item(Circle().stroke(palette.textSecondary, lineWidth: 1.5), "Inactive")
            item(Circle().stroke(palette.keepOut, lineWidth: 1.5), "Keep-out volume")
            Spacer()
            Text("Seen from above; the rail shows altitude")
                .foregroundStyle(palette.textSecondary)
        }
        .font(TypeScale.caption)
    }

    private func item<Mark: View>(_ mark: Mark, _ title: String) -> some View {
        HStack(spacing: Space.xs) {
            mark.frame(width: 9, height: 9)
            Text(title)
        }
    }
}
