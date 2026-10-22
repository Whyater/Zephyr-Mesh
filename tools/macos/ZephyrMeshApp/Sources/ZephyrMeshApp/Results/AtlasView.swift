import SwiftUI

/// "Tolerance atlas": for one set of conditions and one loss setting, the most delay each
/// tracker tolerates on each path, with the row's winner or tie at the end.
struct AtlasContent: View {
    @ObservedObject var model: ResultsModel
    let data: ResultsData
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(alignment: .leading, spacing: Space.lg) {
            controls
            if let stratum = model.stratum {
                Text(ResultsFormat.summaryLine(stratum.counts))
                    .font(TypeScale.body.monospacedDigit())
                    .foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                // The legend sits above the grid, as in the delay view.
                AtlasLegend()
                AtlasMatrix(model: model, data: data, stratum: stratum)
            } else {
                Text(ResultsFormat.notComputed)
                    .font(TypeScale.body)
                    .foregroundStyle(palette.textSecondary)
            }
        }
    }

    /// One row when it fits; otherwise the conditions menu above the two loss controls.
    private var controls: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: Space.xl) {
                conditions
                lossControls
            }
            VStack(alignment: .leading, spacing: Space.md) {
                conditions
                HStack(spacing: Space.xl) { lossControls }
            }
        }
    }

    private var conditions: some View {
        ControlGroup(label: UIText.Atlas.conditionsLabel) {
            Picker(UIText.Atlas.conditionsLabel, selection: Binding(get: { model.atlas.stratumID },
                                                                    set: { model.selectStratum($0) })) {
                ForEach(data.atlas.strata, id: \.id) { stratum in
                    Text(stratum.title).tag(stratum.id)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .fixedSize()
            .help(model.stratum?.description ?? "")
        }
        .fixedSize()
    }

    @ViewBuilder private var lossControls: some View {
        ControlGroup(label: UIText.Results.lossTypeLabel) {
            SegmentedChoice(options: [("independent", UIText.Results.lossRandom), ("burst", UIText.Results.lossBursty)],
                            selection: $model.atlas.lossModel, accessibilityLabel: UIText.Results.lossTypeLabel)
        }
        .fixedSize()
        ControlGroup(label: UIText.Results.lossLevelLabel) {
            SegmentedChoice(options: (model.stratum.map(AtlasLayout.levels) ?? []).map { ($0, ResultsFormat.levelLabel($0)) },
                            selection: $model.atlas.level, accessibilityLabel: UIText.Results.lossLevelLabel)
        }
        .fixedSize()
    }
}

/// The matrix: one row per path, one column per tracker, then "Most delay". It takes keyboard
/// focus as one control; the arrow keys move the selection and Escape clears it.
private struct AtlasMatrix: View {
    @ObservedObject var model: ResultsModel
    let data: ResultsData
    let stratum: AtlasStratum
    @FocusState private var focused: Bool
    @State private var availableWidth: CGFloat = 0
    @State private var mostDelayWidth: CGFloat = 0
    @Environment(\.palette) private var palette

    var body: some View {
        let loss = model.atlas.lossCondition
        let cellWidth = AtlasGeometry.cellWidth(available: availableWidth, arms: stratum.arms.count, mostDelayWidth: mostDelayWidth)
        ScrollView(.horizontal) {
            Grid(alignment: .leading, horizontalSpacing: AtlasGeometry.columnSpacing, verticalSpacing: AtlasGeometry.columnSpacing) {
                GridRow {
                    Color.clear.frame(width: LayoutMetrics.atlasRowHeaderWidth, height: 0)
                    ForEach(stratum.arms, id: \.self) { arm in
                        Text(ResultsFormat.armLabel(arm, labels: data.atlas.armLabels))
                            .font(TypeScale.caption)
                            .foregroundStyle(palette.textSecondary)
                            .lineLimit(1)
                            // "Recovery gated" and the other long names only fit the 8-tracker set when scaled slightly.
                            .minimumScaleFactor(0.85)
                            .frame(width: cellWidth)
                    }
                    Text(UIText.Atlas.mostDelayColumn)
                        .font(TypeScale.caption)
                        .foregroundStyle(palette.textSecondary)
                        .padding(.leading, Space.sm)
                }
                ForEach(Array(stratum.variants.enumerated()), id: \.element) { row, variant in
                    GridRow {
                        Text(data.atlas.variantTitles[variant] ?? Format.label(variant))
                            .font(TypeScale.body)
                            .foregroundStyle(palette.text)
                            .lineLimit(1)
                            .frame(width: LayoutMetrics.atlasRowHeaderWidth, alignment: .leading)
                        ForEach(Array(stratum.arms.enumerated()), id: \.element) { column, arm in
                            let selection = AtlasSelection(row: row, column: column)
                            AtlasCellButton(
                                entry: data.entry(stratum: stratum.id, variant: variant, loss: loss, arm: arm),
                                isWinner: isWinner(arm: arm, variant: variant, loss: loss),
                                isSelected: model.atlas.selection == selection,
                                width: cellWidth,
                                accessibilityName: accessibilityName(arm: arm, variant: variant)
                            ) {
                                model.atlas.selection = selection
                                focused = true
                            }
                        }
                        MostDelayCell(cell: data.cell(stratum: stratum.id, variant: variant, loss: loss), labels: data.atlas.armLabels)
                            .background { GeometryReader { Color.clear.preference(key: MostDelayWidthKey.self, value: $0.size.width) } }
                            .padding(.leading, AtlasGeometry.mostDelayLeading)
                    }
                }
            }
            // Room for the selection ring outside the edge cells.
            .padding(AtlasGeometry.outerPadding)
            .onPreferenceChange(MostDelayWidthKey.self) { mostDelayWidth = $0 }
        }
        .scrollIndicators(.visible)
        .onSizeChange { availableWidth = $0.width }
        .focusable()
        .focused($focused)
        .focusEffectDisabled()
        .modifier(FocusRing(visible: focused))
        .onKeyPress(keys: [.leftArrow, .rightArrow, .upArrow, .downArrow, .escape]) { press in
            switch press.key {
            case .leftArrow: model.moveAtlas(.left)
            case .rightArrow: model.moveAtlas(.right)
            case .upArrow: model.moveAtlas(.up)
            case .downArrow: model.moveAtlas(.down)
            default:
                guard model.atlas.selection != nil else { return .ignored }
                model.atlas.selection = nil
            }
            return .handled
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(stratum.title)
    }

    private func isWinner(arm: String, variant: String, loss: String) -> Bool {
        guard let cell = data.cell(stratum: stratum.id, variant: variant, loss: loss) else { return false }
        return cell.kind == .winner && cell.arms == [arm]
    }

    private func accessibilityName(arm: String, variant: String) -> String {
        "\(ResultsFormat.armLabel(arm, labels: data.atlas.armLabels)), \(data.atlas.variantTitles[variant] ?? Format.label(variant))"
    }
}

/// The widest "Most delay" entry, so the tracker columns can shrink to leave it room.
private struct MostDelayWidthKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}

/// One matrix cell: the ramp tint for the tolerated delay. A clear winner gets bold text and an
/// accent bar along the bottom inside edge; the selection is a separate accent ring outside the
/// cell with a gap, so a selected winner shows both. Text is always the text color.
private struct AtlasCellButton: View {
    let entry: AtlasEntry?
    let isWinner: Bool
    let isSelected: Bool
    let width: CGFloat
    let accessibilityName: String
    let action: () -> Void
    @Environment(\.palette) private var palette
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            content
                .frame(width: width, height: LayoutMetrics.atlasCellHeight)
                .background(fill)
                .overlay(alignment: .bottom) {
                    if isWinner {
                        Rectangle()
                            .fill(palette.accent)
                            .frame(height: LayoutMetrics.atlasWinnerBar)
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: Radius.small))
                .overlay {
                    RoundedRectangle(cornerRadius: Radius.small)
                        .strokeBorder(hovering ? palette.lineStrong : palette.line, lineWidth: 1)
                }
                .overlay {
                    if isSelected {
                        RoundedRectangle(cornerRadius: Radius.small + Space.xxs * 2)
                            .strokeBorder(palette.accent, lineWidth: Space.xxs)
                            .padding(-Space.xxs * 2)
                    }
                }
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .accessibilityLabel("\(accessibilityName): \(ResultsFormat.cellText(entry))")
        .accessibilityValue(isWinner ? UIText.Atlas.clearWinner : "")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    @ViewBuilder private var content: some View {
        if let entry, entry.kind == .failsAtZero {
            HStack(spacing: Space.xxs) {
                StatusShape(style: StatusStyle(level: "alert"), font: TypeScale.caption)
                    .frame(width: Space.md)
                Text(UIText.Atlas.failsAtZero)
                    .font(isWinner ? TypeScale.bodyStrong : TypeScale.body)
                    .foregroundStyle(palette.text)
            }
        } else {
            Text(ResultsFormat.cellText(entry))
                .font((isWinner ? TypeScale.bodyStrong : TypeScale.body).monospacedDigit())
                .foregroundStyle(entry == nil ? palette.textSecondary : palette.text)
        }
    }

    private var fill: Color {
        switch AtlasBin.stop(for: entry) {
        case 1?: return palette.atlasStop1
        case 2?: return palette.atlasStop2
        case 3?: return palette.atlasStop3
        case 4?: return palette.atlasStop4
        default: return palette.surface
        }
    }
}

/// The row's result at the selected loss: the clear winner, the tie, or "Nothing passes",
/// with the delay the fixture recorded for it.
private struct MostDelayCell: View {
    let cell: AtlasCell?
    let labels: [String: String]
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(ResultsFormat.classification(cell, labels: labels))
                .font(cell?.kind == .winner ? TypeScale.bodyStrong : TypeScale.body)
                .foregroundStyle(cell == nil ? palette.textSecondary : palette.text)
                .lineLimit(1)
            if let delay = cell?.delayText {
                Text(delay)
                    .font(TypeScale.caption.monospacedDigit())
                    .foregroundStyle(palette.textSecondary)
            }
        }
        .fixedSize()
        .frame(minHeight: LayoutMetrics.atlasCellHeight, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

/// Ramp swatches with their delay words, then the fails-at-zero shape and the winner border.
private struct AtlasLegend: View {
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(alignment: .leading, spacing: Space.sm) {
            HStack(spacing: Space.lg) {
                swatch(palette.atlasStop1, UIText.Atlas.bin1)
                swatch(palette.atlasStop2, UIText.Atlas.bin2)
                swatch(palette.atlasStop3, UIText.Atlas.bin3)
                swatch(palette.atlasStop4, UIText.Atlas.censoredLong)
            }
            HStack(spacing: Space.lg) {
                LegendItem(title: UIText.Atlas.failsAtZeroLong) {
                    StatusShape(style: StatusStyle(level: "alert"), font: TypeScale.caption)
                }
                LegendItem(title: UIText.Atlas.clearWinner) {
                    VStack {
                        Spacer(minLength: 0)
                        Rectangle()
                            .fill(palette.accent)
                            .frame(height: LayoutMetrics.atlasWinnerBar)
                    }
                }
            }
        }
        .foregroundStyle(palette.text)
    }

    private func swatch(_ color: Color, _ title: String) -> some View {
        LegendItem(title: title) {
            RoundedRectangle(cornerRadius: Space.xxs)
                .fill(color)
                .overlay { RoundedRectangle(cornerRadius: Space.xxs).strokeBorder(palette.line, lineWidth: 1) }
        }
    }
}
