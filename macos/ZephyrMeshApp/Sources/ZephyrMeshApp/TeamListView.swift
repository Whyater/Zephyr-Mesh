import AppKit
import SwiftUI

/// Sidebar: one row per status group (level and word), non-OK groups first, stalest drone first.
struct TeamListView: View {
    @ObservedObject var model: CockpitModel
    @Environment(\.palette) private var palette
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @FocusState private var filterFocused: Bool
    @FocusState private var listFocused: Bool
    @State private var expansionOverrides: [String: Bool] = [:]

    private var rows: [TeamPresentation.Row] {
        TeamPresentation.rows(groups: model.presentation.groups, drones: model.presentation.perDrone,
                              query: model.teamFilter)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: Space.sm) {
                Text(model.presentation.teamSummary)
                    .font(TypeScale.bodyStrong)
                    .foregroundStyle(palette.text)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                filterField
            }
            .padding(.horizontal, Space.md)
            .padding(.top, Space.md)
            .padding(.bottom, Space.sm)
            columnHeader
            HairlineDivider()
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: Space.xxs) {
                        ForEach(rows) { row in
                            groupSection(row)
                        }
                    }
                    .padding(.horizontal, Space.sm)
                    .padding(.vertical, Space.sm)
                }
                .focusable()
                .focused($listFocused)
                .focusEffectDisabled()
                .overlay {
                    if listFocused {
                        RoundedRectangle(cornerRadius: Radius.small)
                            .strokeBorder(palette.accent, lineWidth: 1)
                            .padding(Space.xxs)
                            .allowsHitTesting(false)
                    }
                }
                .onMoveCommand(perform: move)
                .accessibilityLabel("Team list")
                .accessibilityHint("Up and down arrows move the selection; left and right collapse or expand its group")
                .onChange(of: model.selectedID) { _, newValue in reveal(newValue, proxy: proxy, animated: true) }
                .onChange(of: model.frameIndex) { _, _ in reveal(model.selectedID, proxy: proxy, animated: false) }
                .onAppear { reveal(model.selectedID, proxy: proxy, animated: false) }
            }
        }
        .background(palette.surface)
        .onChange(of: filterFocused) { _, focused in model.isTextFieldFocused = focused }
        .onChange(of: listFocused) { _, focused in model.isTeamListFocused = focused }
        .onChange(of: model.filterFocusRequest) { _, _ in filterFocused = true }
    }

    private var filterField: some View {
        HStack(spacing: Space.xs) {
            Image(systemName: "line.3.horizontal.decrease")
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
                .accessibilityHidden(true)
            TextField("Filter drones", text: $model.teamFilter)
                .textFieldStyle(.plain)
                .font(TypeScale.body)
                .focused($filterFocused)
                .onExitCommand {
                    model.teamFilter = ""
                    filterFocused = false
                }
                .accessibilityLabel("Filter drones by name")
            if !model.teamFilter.isEmpty {
                Button {
                    model.teamFilter = ""
                } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(palette.textSecondary)
                }
                .buttonStyle(.plain)
                .help("Clear filter")
                .accessibilityLabel("Clear filter")
            }
        }
        .padding(.horizontal, Space.sm)
        .frame(height: LayoutMetrics.rowHeight)
        .background(palette.background, in: RoundedRectangle(cornerRadius: Radius.small))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.small)
                .strokeBorder(filterFocused ? palette.accent : palette.line, lineWidth: 1)
        }
        .help("Filter by name, for example D07. Drones that need attention always stay listed.")
    }

    private var columnHeader: some View {
        HStack(spacing: TeamColumns.spacing) {
            Text("Status")
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.leading, Space.md + Space.xxs + Space.lg + TeamColumns.spacing)
            Text("Age")
                .frame(width: TeamColumns.age, alignment: .trailing)
            Text("Loss")
                .frame(width: TeamColumns.loss, alignment: .trailing)
        }
        .font(TypeScale.caption)
        .foregroundStyle(palette.textSecondary)
        .padding(.horizontal, Space.sm + Space.sm)
        .padding(.bottom, Space.xs)
        .accessibilityHidden(true)
    }


    /// Opens the group holding `droneID` (clearing a collapse the user made earlier) and scrolls
    /// its row into view. Runs on selection, on frame change (drones move between groups), and
    /// on appear, so a selection made from the scene or at launch is never hidden.
    private func reveal(_ droneID: String?, proxy: ScrollViewProxy, animated: Bool) {
        guard let droneID, let row = rows.first(where: { $0.group.droneIDs.contains(droneID) }) else { return }
        if expansionOverrides[expansionKey(row.group)] == false { expansionOverrides[expansionKey(row.group)] = nil }
        DispatchQueue.main.async {
            withAnimation(animated ? .easeOut(duration: Motion.duration(Motion.standard, reduceMotion: reduceMotion)) : nil) {
                proxy.scrollTo(droneID, anchor: .center)
            }
        }
    }

    /// Drones in the order they are listed, counting only open groups.
    private var visibleDroneIDs: [String] {
        rows.flatMap { isExpanded($0) ? $0.members.map(\.droneID) : [] }
    }

    private func move(_ direction: MoveCommandDirection) {
        let ids = visibleDroneIDs
        switch direction {
        case .up, .down:
            guard !ids.isEmpty else { return }
            let current = model.selectedID.flatMap { ids.firstIndex(of: $0) }
            let next = current.map { min(max($0 + (direction == .down ? 1 : -1), 0), ids.count - 1) } ?? 0
            model.select(ids[next])
        case .left, .right:
            guard let selected = model.selectedID,
                  let row = rows.first(where: { $0.group.droneIDs.contains(selected) }) else { return }
            expansionOverrides[expansionKey(row.group)] = direction == .right
        @unknown default:
            break
        }
    }

    private func expansionKey(_ group: PresentationGroup) -> String {
        "\(group.level)-\(group.word)"
    }

    /// Filter matches are always open. Otherwise an explicit user toggle wins; with no toggle, the
    /// group holding the selected drone opens, and so do groups that need attention.
    private func isExpanded(_ row: TeamPresentation.Row) -> Bool {
        if row.isFilteredMatch { return true }
        if let override = expansionOverrides[expansionKey(row.group)] { return override }
        if let selected = model.selectedID, row.group.droneIDs.contains(selected) { return true }
        return row.group.level != "ok"
    }

    @ViewBuilder private func groupSection(_ row: TeamPresentation.Row) -> some View {
        let expanded = isExpanded(row)
        GroupRow(row: row, isExpanded: expanded, model: model) {
            withAnimation(.easeOut(duration: Motion.duration(Motion.fast, reduceMotion: reduceMotion))) {
                expansionOverrides[expansionKey(row.group)] = !expanded
            }
        }
        if expanded {
            ForEach(row.members) { drone in
                DroneRow(drone: drone, model: model) { listFocused = true }
                    .id(drone.droneID)
            }
        }
    }
}

enum TeamColumns {
    static let age: CGFloat = 42
    static let loss: CGFloat = 58
    static let spacing: CGFloat = 6
}

/// The Age and Loss cells of a row. When neither value exists the row reads "Not recorded"
/// across both columns; a single missing cell shows a dash with the same words as its tooltip.
private struct MetricColumns: View {
    /// Formatted age, or `nil` when the presentation has no age for the row.
    let age: String?
    let loss: String
    let lossMissing: Bool
    @Environment(\.palette) private var palette

    var body: some View {
        if age == nil && lossMissing {
            Text(Format.notRecorded)
                .foregroundStyle(palette.textSecondary)
                .frame(width: TeamColumns.age + TeamColumns.loss + TeamColumns.spacing, alignment: .trailing)
        } else {
            cell(age, width: TeamColumns.age)
            cell(lossMissing ? nil : loss, width: TeamColumns.loss)
        }
    }

    private func cell(_ value: String?, width: CGFloat) -> some View {
        Text(value ?? "\u{2013}")
            .foregroundStyle(value == nil ? palette.textSecondary : palette.text)
            .frame(width: width, alignment: .trailing)
    }
}

private struct GroupRow: View {
    let row: TeamPresentation.Row
    let isExpanded: Bool
    @ObservedObject var model: CockpitModel
    let toggle: () -> Void
    @Environment(\.palette) private var palette
    @State private var hovering = false

    var body: some View {
        let group = row.group
        let style = StatusStyle(group: group)
        Button(action: toggle) {
            HStack(spacing: TeamColumns.spacing) {
                HStack(spacing: Space.xxs) {
                    Image(systemName: "chevron.right")
                        .font(TypeScale.caption)
                        .foregroundStyle(palette.textSecondary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        .frame(width: Space.md)
                    StatusShape(style: style)
                }
                Text(style.word)
                    .font(TypeScale.bodyStrong)
                    .foregroundStyle(palette.text)
                    .fixedSize()
                Text(Format.count(group.count, noun: "drone"))
                    .font(TypeScale.body)
                    .foregroundStyle(palette.textSecondary)
                    .lineLimit(1)
                Spacer(minLength: 0)
                MetricColumns(age: group.targetAgeS.map { Format.seconds($0) }, loss: Format.percent(group.targetPacketLossPct),
                              lossMissing: group.targetPacketLossPct == nil)
            }
            .font(TypeScale.body.monospacedDigit())
            .foregroundStyle(palette.text)
            .padding(.horizontal, Space.sm)
            .frame(height: LayoutMetrics.rowHeight)
            .background(RowBackground(isSelected: false, isHovered: hovering))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .contextMenu { TeamContextMenu(droneIDs: group.droneIDs, model: model) }
        .accessibilityLabel("\(style.word), \(Format.count(group.count, noun: "drone"))")
        .accessibilityValue("\(UIText.targetAge) \(Format.seconds(group.targetAgeS))")
        .accessibilityHint(isExpanded ? "Collapse the group" : "Expand the group")
    }
}

private struct DroneRow: View {
    let drone: PresentationDrone
    @ObservedObject var model: CockpitModel
    /// Gives the team list keyboard focus when a row is clicked.
    let focusList: () -> Void
    @Environment(\.palette) private var palette
    @State private var hovering = false

    var body: some View {
        let selected = model.selectedID == drone.droneID
        HStack(spacing: TeamColumns.spacing) {
            Text(drone.compactName)
                .font(selected ? TypeScale.bodyStrong : TypeScale.body)
                .foregroundStyle(palette.text)
                .padding(.leading, Space.md + Space.xxs + Space.lg + TeamColumns.spacing)
            Spacer(minLength: Space.xs)
            MetricColumns(age: drone.targetAgeS.map { Format.seconds($0) }, loss: Format.percent(drone.targetPacketLossPct),
                          lossMissing: drone.targetPacketLossPct == nil)
        }
        .font(TypeScale.body.monospacedDigit())
        .foregroundStyle(selected ? palette.text : palette.textSecondary)
        .padding(.horizontal, Space.sm)
        .frame(height: LayoutMetrics.rowHeight)
        .background(RowBackground(isSelected: selected, isHovered: hovering || model.hoveredID == drone.droneID))
        .contentShape(Rectangle())
        .onHover { inside in
            hovering = inside
            model.hoveredID = inside ? drone.droneID : (model.hoveredID == drone.droneID ? nil : model.hoveredID)
        }
        .onTapGesture(count: 2) { model.focus(on: drone.droneID) }
        .onTapGesture {
            model.select(drone.droneID)
            focusList()
        }
        .contextMenu { TeamContextMenu(droneIDs: [drone.droneID], model: model) }
        .accessibilityElement(children: .ignore)
        .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
        .accessibilityLabel(drone.headingName)
        .accessibilityValue("\(drone.word), \(UIText.targetAge) \(Format.seconds(drone.targetAgeS))")
        .accessibilityAction { model.select(drone.droneID) }
    }
}

/// Focus camera, Show in inspector, Copy drone ID. Groups act on their first drone and copy all IDs.
struct TeamContextMenu: View {
    let droneIDs: [String]
    @ObservedObject var model: CockpitModel

    var body: some View {
        if let first = droneIDs.first {
            let name = model.presentationDrone(id: first)?.compactName ?? first
            Button(droneIDs.count == 1 ? "Focus Camera" : "Focus Camera on \(name)") { model.focus(on: first) }
            Button(droneIDs.count == 1 ? "Show in Inspector" : "Show \(name) in Inspector") {
                model.select(first)
                model.showInspector = true
            }
            Divider()
            Button(droneIDs.count == 1 ? "Copy Drone ID" : "Copy \(droneIDs.count) Drone IDs") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(droneIDs.joined(separator: ", "), forType: .string)
            }
        }
    }
}
