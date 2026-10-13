import AppKit
import SwiftUI

/// Sidebar: one row per status group (level and word), non-OK groups first, stalest drone first.
/// The selection pill, the filter field and the summary share one left edge (`Space.md`), and
/// the column headers sit over the row values.
struct TeamListView: View {
    @ObservedObject var model: CockpitModel
    @Environment(\.palette) private var palette
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @FocusState private var filterFocused: Bool
    @FocusState private var listFocused: Bool
    @State private var expansionOverrides: [String: Bool] = [:]
    @State private var listHeight: CGFloat = 0

    private var rows: [TeamPresentation.Row] {
        TeamPresentation.rows(groups: model.presentation.groups, drones: model.presentation.perDrone,
                              query: model.teamFilter, keeping: model.selectedID)
    }

    var body: some View {
        let rows = rows
        let counts = TeamPresentation.filterCounts(rows: rows, drones: model.presentation.perDrone, query: model.teamFilter)
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: Space.sm) {
                // No vertical fixedSize here or on the caption: a wrapped line must not raise the
                // sidebar's minimum height, which shortened the whole split view.
                Text(model.presentation.teamSummary)
                    .font(TypeScale.bodyStrong)
                    .foregroundStyle(palette.text)
                    .lineLimit(2)
                filterField
                if let caption = TeamPresentation.filterCaption(counts: counts, rows: rows) {
                    Text(caption)
                        .font(TypeScale.caption)
                        .foregroundStyle(palette.textSecondary)
                        .lineLimit(2)
                        .accessibilityLabel(caption)
                }
            }
            .padding(.horizontal, Space.md)
            .padding(.top, Space.md)
            .padding(.bottom, Space.sm)
            columnHeader
            HairlineDivider()
            if rows.isEmpty, let empty = counts.emptyText {
                emptyState(empty)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                list(rows: rows, emptyText: counts.emptyText)
            }
        }
        .background(palette.surface)
        .onChange(of: filterFocused) { _, focused in model.isTextFieldFocused = focused }
        .onChange(of: listFocused) { _, focused in model.isTeamListFocused = focused }
        .onChange(of: model.filterFocusRequest) { _, _ in filterFocused = true }
    }

    private func list(rows: [TeamPresentation.Row], emptyText: String?) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: Space.xxs) {
                    if let emptyText {
                        // Nothing matches, but late or lost drones are still listed below.
                        emptyState(emptyText)
                            .padding(.vertical, Space.sm)
                    }
                    ForEach(Array(rows.enumerated()), id: \.element.id) { index, row in
                        groupSection(row)
                            .padding(.top, index == 0 ? 0 : Space.sm)
                    }
                }
                .padding(.horizontal, Space.md)
                .padding(.vertical, Space.sm)
            }
            .onSizeChange { listHeight = $0.height }
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
            .onChange(of: model.selectedID) { old, new in reveal(new, previous: old, proxy: proxy, animated: true) }
            .onChange(of: model.frameIndex) { _, _ in
                reveal(model.selectedID, previous: model.selectedID, proxy: proxy, animated: false)
            }
            // At launch the list starts at the top; it scrolls only if the selection is out of view.
            .onAppear { reveal(model.selectedID, previous: model.selectedID, proxy: proxy, animated: false) }
        }
    }

    private func emptyState(_ text: String) -> some View {
        VStack(spacing: Space.sm) {
            Text(text)
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            Button(UIText.TeamFilter.clearFilter) { model.teamFilter = "" }
                .buttonStyle(QuietButtonStyle(font: TypeScale.body, bordered: true))
        }
        .padding(Space.md)
        .frame(maxWidth: .infinity)
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
                .help(UIText.TeamFilter.clearFilter)
                .accessibilityLabel(UIText.TeamFilter.clearFilter)
            }
        }
        .padding(.horizontal, Space.sm)
        .frame(height: LayoutMetrics.controlHeight)
        .background(palette.background, in: RoundedRectangle(cornerRadius: Radius.small))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.small)
                .strokeBorder(filterFocused ? palette.accent : palette.line, lineWidth: 1)
        }
        .help("Filter by name, for example D07. \(UIText.TeamFilter.pinnedNote)")
    }

    private var columnHeader: some View {
        HStack(spacing: TeamColumns.spacing) {
            Text(TeamColumns.statusTitle)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.leading, TeamColumns.nameIndent)
            Text(TeamColumns.ageTitle)
                .frame(width: TeamColumns.age, alignment: .trailing)
            Text(TeamColumns.lossTitle)
                .frame(width: TeamColumns.loss, alignment: .trailing)
        }
        .font(TypeScale.caption)
        .foregroundStyle(palette.textSecondary)
        .padding(.horizontal, Space.md + Space.sm)
        .padding(.bottom, Space.xs)
        .accessibilityHidden(true)
    }

    /// Opens the group holding `droneID` (clearing a collapse the user made earlier) and scrolls
    /// it into view. When a new selection opens a closed group, the row is placed so its group
    /// header lands at the top of the list and both show (if the row is within one list height
    /// of the header); otherwise the row scrolls the minimum amount. Runs on selection, on frame
    /// change (drones move between groups), and on appear, so a selection is never hidden.
    private func reveal(_ droneID: String?, previous: String?, proxy: ScrollViewProxy, animated: Bool) {
        let rows = rows
        guard let droneID, let row = rows.first(where: { $0.group.droneIDs.contains(droneID) }) else { return }
        let wasClosed = !isExpanded(row, selectedID: previous)
        if expansionOverrides[expansionKey(row.group)] == false { expansionOverrides[expansionKey(row.group)] = nil }
        let index = row.members.firstIndex { $0.droneID == droneID } ?? 0
        let pitch = LayoutMetrics.rowHeight + Space.xxs
        let rowTop = CGFloat(index + 1) * pitch + Space.sm
        let showHeader = droneID != previous && wasClosed && listHeight > 0 && rowTop + pitch <= listHeight
        let atLaunch = previous == droneID
        DispatchQueue.main.async {
            // At appear and on frame changes the list may not have moved: leave it alone when the
            // row already fits in the first screen.
            if atLaunch, let offset = listOffset(of: droneID), listHeight > 0, offset + pitch <= listHeight { return }
            withAnimation(animated ? .easeOut(duration: Motion.duration(Motion.standard, reduceMotion: reduceMotion)) : nil) {
                if showHeader {
                    // Row top at `rowTop` from the viewport top puts the header just above it.
                    proxy.scrollTo(droneID, anchor: UnitPoint(x: 0, y: rowTop / max(listHeight - pitch, 1)))
                } else {
                    proxy.scrollTo(droneID)
                }
            }
        }
    }

    /// Distance from the list top to the row's top when the list is scrolled to the top.
    private func listOffset(of droneID: String) -> CGFloat? {
        let pitch = LayoutMetrics.rowHeight + Space.xxs
        var offset = Space.sm
        for (index, row) in rows.enumerated() {
            if index > 0 { offset += Space.sm }
            offset += pitch
            guard isExpanded(row) else { continue }
            if let position = row.members.firstIndex(where: { $0.droneID == droneID }) {
                return offset + CGFloat(position) * pitch
            }
            offset += CGFloat(row.members.count) * pitch
        }
        return nil
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
        isExpanded(row, selectedID: model.selectedID)
    }

    private func isExpanded(_ row: TeamPresentation.Row, selectedID: String?) -> Bool {
        if row.isFilteredMatch { return true }
        if let override = expansionOverrides[expansionKey(row.group)] { return override }
        if let selectedID, row.group.droneIDs.contains(selectedID) { return true }
        return row.group.level != "ok"
    }

    @ViewBuilder private func groupSection(_ row: TeamPresentation.Row) -> some View {
        let expanded = isExpanded(row)
        VStack(alignment: .leading, spacing: Space.xxs) {
            GroupRow(row: row, isExpanded: expanded, model: model) {
                withAnimation(.easeOut(duration: Motion.duration(Motion.fast, reduceMotion: reduceMotion))) {
                    expansionOverrides[expansionKey(row.group)] = !expanded
                }
            }
            .id(TeamColumns.headerID(row.group))
            if expanded {
                ForEach(row.members) { drone in
                    DroneRow(drone: drone, isKeptSelection: row.keptSelectionID == drone.droneID, model: model) {
                        listFocused = true
                    }
                        .id(drone.droneID)
                }
            }
        }
    }
}

/// Column geometry and cell text. Units live in the column headers ("Age (s)", "Loss (%)"), so
/// a group's range ("0.15\u{2013}0.20") fits the same column as a single value.
enum TeamColumns {
    static let age = LayoutMetrics.teamAgeWidth
    static let loss = LayoutMetrics.teamLossWidth
    static let spacing = LayoutMetrics.teamColumnSpacing
    static let statusTitle = Format.label("status_column")
    static let ageTitle = Format.label("age_s")
    static let lossTitle = Format.label("loss_pct")
    /// Where drone names and the "Status" header start: past the chevron and the status shape.
    static let nameIndent = Space.md + Space.xxs + Space.lg + spacing

    static func headerID(_ group: PresentationGroup) -> String { "group-\(group.id)" }

    /// Group size for the header: "47 drones", or "1 of 47 drones" when only some are listed.
    static func countText(listed: Int, total: Int, noun: Bool) -> String {
        let whole = noun ? Format.count(total, noun: "drone") : "\(total)"
        return listed < total ? "\(listed) of \(whole)" : whole
    }

    static func age(_ value: Double?) -> String? {
        value.map { Format.number($0, decimals: 2) }
    }

    static func loss(_ value: Double?) -> String? {
        value.map { Format.number($0, decimals: 1) }
    }

    static func age(_ metric: PresentationMetric?) -> String? {
        switch metric {
        case .none: return nil
        case .value(let value): return age(value)
        case .range(let low, let high):
            return low == high ? age(low) : "\(Format.number(low, decimals: 2))\u{2013}\(Format.number(high, decimals: 2))"
        }
    }

    static func loss(_ metric: PresentationMetric?) -> String? {
        switch metric {
        case .none: return nil
        case .value(let value): return loss(value)
        case .range(let low, let high):
            return low == high ? loss(low) : "\(Format.number(low, decimals: 0))\u{2013}\(Format.number(high, decimals: 0))"
        }
    }
}

/// The Age and Loss cells of a row. A missing value is a secondary dash with "Not recorded" as
/// its tooltip and spoken label, so 50 waiting drones do not repeat the words 50 times.
private struct MetricColumns: View {
    let age: String?
    let loss: String?
    @Environment(\.palette) private var palette

    var body: some View {
        cell(age, width: TeamColumns.age)
        cell(loss, width: TeamColumns.loss)
    }

    @ViewBuilder private func cell(_ value: String?, width: CGFloat) -> some View {
        if let value {
            Text(value)
                .foregroundStyle(palette.text)
                .lineLimit(1)
                .frame(width: width, alignment: .trailing)
        } else {
            Text("\u{2013}")
                .foregroundStyle(palette.textSecondary)
                .frame(width: width, alignment: .trailing)
                .help(Format.notRecorded)
                .accessibilityLabel(Format.notRecorded)
        }
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
                // "50 drones", or "1 of 47 drones" while a filter lists only some members; the
                // noun drops when the sidebar is too narrow for it.
                ViewThatFits(in: .horizontal) {
                    Text(TeamColumns.countText(listed: row.members.count, total: group.count, noun: true)).fixedSize()
                    Text(TeamColumns.countText(listed: row.members.count, total: group.count, noun: false)).fixedSize()
                }
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
                .layoutPriority(1)
                Spacer(minLength: 0)
                MetricColumns(age: TeamColumns.age(group.targetAgeS), loss: TeamColumns.loss(group.targetPacketLossPct))
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
        .accessibilityLabel("\(style.word), \(TeamColumns.countText(listed: row.members.count, total: group.count, noun: true))")
        .accessibilityValue("\(UIText.targetAge) \(Format.seconds(group.targetAgeS))")
        .accessibilityHint(isExpanded ? "Collapse the group" : "Expand the group")
    }
}

private struct DroneRow: View {
    let drone: PresentationDrone
    /// Listed only because it is selected; it does not match the filter.
    var isKeptSelection = false
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
                .padding(.leading, TeamColumns.nameIndent)
                .overlay(alignment: .leading) {
                    if isKeptSelection {
                        // Sits in the indent, under the group's status shape column.
                        Image(systemName: "pin.fill")
                            .font(TypeScale.caption)
                            .foregroundStyle(palette.textSecondary)
                            .frame(width: Space.lg)
                            .padding(.leading, Space.md + Space.xxs)
                            .help("Selected. Listed although it does not match the filter.")
                    }
                }
            Spacer(minLength: Space.xs)
            MetricColumns(age: TeamColumns.age(drone.targetAgeS), loss: TeamColumns.loss(drone.targetPacketLossPct))
        }
        .font(TypeScale.body.monospacedDigit())
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
        .accessibilityValue("\(drone.word), \(UIText.targetAge) \(Format.seconds(drone.targetAgeS))"
                            + (isKeptSelection ? ", selected, does not match the filter" : ""))
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
