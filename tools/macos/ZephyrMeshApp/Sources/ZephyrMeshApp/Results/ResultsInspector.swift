import SwiftUI

/// The Results inspector: details of the selected chart point or atlas cell, or an empty state.
struct ResultsInspector: View {
    @ObservedObject var model: ResultsModel
    let data: ResultsData
    @Environment(\.palette) private var palette

    var body: some View {
        Group {
            switch model.page {
            case .delay:
                if let selection = model.delay.selection, let point = DelayPoint(model: model, data: data, selection: selection) {
                    scroll { DelayPointInspector(point: point, data: data) }
                } else {
                    ResultsEmptyInspector(hint: ResultsPage.delay.emptyHint)
                }
            case .atlas:
                if let selection = model.atlas.selection, let stratum = model.stratum,
                   selection.row < stratum.variants.count, selection.column < stratum.arms.count {
                    scroll { AtlasCellInspector(model: model, data: data, stratum: stratum, selection: selection) }
                } else {
                    ResultsEmptyInspector(hint: ResultsPage.atlas.emptyHint)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(palette.surface)
    }

    private func scroll<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        ScrollView {
            content()
                .padding(.horizontal, Space.lg)
                .padding(.top, Space.md)
                .padding(.bottom, Space.lg)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// Everything the point inspector shows, resolved from the selection.
struct DelayPoint {
    let panel: LossPanel
    let arm: String
    let motion: String
    let series: S6Series
    let index: Int
    let delay: Double

    @MainActor
    init?(model: ResultsModel, data: ResultsData, selection: ChartSelection) {
        let panels = model.panels
        guard panels.indices.contains(selection.panel), DelayLossLayout.arms.indices.contains(selection.arm),
              data.s6.delaysS.indices.contains(selection.delayIndex) else { return nil }
        panel = panels[selection.panel]
        arm = DelayLossLayout.arms[selection.arm]
        motion = model.delay.motion
        guard let series = model.series(panel: panel, arm: arm) else { return nil }
        self.series = series
        index = selection.delayIndex
        delay = data.s6.delaysS[selection.delayIndex]
    }
}

private struct DelayPointInspector: View {
    let point: DelayPoint
    let data: ResultsData
    @Environment(\.palette) private var palette
    @State private var showDetails = false

    var body: some View {
        let series = point.series
        let index = point.index
        VStack(alignment: .leading, spacing: Space.lg) {
            InspectorHeader(title: ResultsFormat.armName(point.arm), subtitle: Format.fill(UIText.Results.pointSubtitle, [
                "path": ResultsFormat.pathName(point.motion),
                "loss": ResultsFormat.lossTitle(model: point.panel.lossModel, probability: point.panel.probability,
                                                titles: data.atlas.lossTitles),
                "delay": "\(ResultsFormat.milliseconds(point.delay))"
            ]))
            Grid(horizontalSpacing: Space.sm, verticalSpacing: Space.sm) {
                GridRow {
                    ResultTile(label: UIText.Results.tileMedian, value: ResultsFormat.meters(series.p95MedianM[index]))
                    ResultTile(label: UIText.Results.tileInterval,
                               value: ResultsFormat.meterRange(low: series.p95CILowM[index], high: series.p95CIHighM[index]))
                }
                GridRow {
                    ResultTile(label: UIText.Results.tileSeedsFailing,
                               value: ResultsFormat.count(series.failCount[index], of: series.nSeeds[index]))
                    if series.label[index] == "fails" {
                        ResultTile(label: UIText.Results.tileCellResult, value: UIText.Results.cellFails,
                                   status: StatusStyle(level: "alert", word: UIText.Results.cellFails))
                    } else {
                        ResultTile(label: UIText.Results.tileCellResult, value: UIText.Results.cellPasses)
                    }
                }
            }
            SectionBlock(title: UIText.Results.compareTitle) {
                VStack(alignment: .leading, spacing: Space.xs) {
                    ForEach(Array(ResultsFormat.comparison(series: series, index: index, delay: point.delay).enumerated()),
                            id: \.offset) { line, text in
                        Text(text)
                            .font(line == 0 ? TypeScale.body : TypeScale.caption.monospacedDigit())
                            .foregroundStyle(line == 0 ? palette.text : palette.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            DetailsSection(isExpanded: $showDetails) {
                StackedValue(label: UIText.Results.detailCellID, value: series.cellIDs[index], style: .identifier)
                StackedValue(label: UIText.Results.detailRule, value: ruleText)
                StackedValue(label: UIText.Results.detailIntervals, value: ResultsFormat.capitalizedFirst(data.s6.intervalNote))
            }
        }
        .textSelection(.enabled)
    }

    private var ruleText: String {
        "\(ResultsFormat.capitalizedFirst(data.s6.criterion)). \(ResultsFormat.capitalizedFirst(data.s6.cellLabelRule))."
    }
}

private struct AtlasCellInspector: View {
    @ObservedObject var model: ResultsModel
    let data: ResultsData
    let stratum: AtlasStratum
    let selection: AtlasSelection
    @Environment(\.palette) private var palette
    @State private var showDetails = false

    private var variant: String { stratum.variants[selection.row] }
    private var arm: String { stratum.arms[selection.column] }

    var body: some View {
        let loss = model.atlas.lossCondition
        let entry = data.entry(stratum: stratum.id, variant: variant, loss: loss, arm: arm)
        VStack(alignment: .leading, spacing: Space.lg) {
            VStack(alignment: .leading, spacing: Space.sm) {
                InspectorHeader(title: ResultsFormat.armLabel(arm, labels: data.atlas.armLabels),
                                subtitle: ResultsFormat.cellSubtitle(
                                    path: data.atlas.variantTitles[variant] ?? Format.label(variant),
                                    loss: data.atlas.lossTitles[loss] ?? ResultsFormat.notComputed),
                                detail: stratum.title)
                if let help = ResultsFormat.armHelp(arm) {
                    Text(help)
                        .font(TypeScale.body)
                        .foregroundStyle(palette.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            VStack(alignment: .leading, spacing: Space.sm) {
                tiles(entry)
                Text(ResultsFormat.intervalCaption(entry, seeds: data.atlas.nSeeds))
                    .font(TypeScale.caption)
                    .foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            SectionBlock(title: UIText.Atlas.acrossLossTitle) {
                VStack(spacing: Space.xxs) {
                    ForEach(stratum.lossConditions, id: \.self) { condition in
                        lossRow(condition, selected: condition == loss)
                    }
                }
            }
            if let entry, !entry.improvementVsNaive.isEmpty {
                DetailsSection(isExpanded: $showDetails) {
                    differenceTable(entry.improvementVsNaive)
                }
            }
            Text(ResultsFormat.cellRule(data.atlas.failureRule))
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .textSelection(.enabled)
    }

    private func tiles(_ entry: AtlasEntry?) -> some View {
        let missing = ResultsFormat.notComputed
        return Grid(horizontalSpacing: Space.sm, verticalSpacing: Space.sm) {
            GridRow {
                if let entry, entry.kind == .failsAtZero {
                    ResultTile(label: UIText.Atlas.tileMostDelay, value: ResultsFormat.mostDelayText(entry),
                               status: StatusStyle(level: "alert", word: UIText.Atlas.failsAtZero))
                } else {
                    ResultTile(label: UIText.Atlas.tileMostDelay, value: entry.map(ResultsFormat.mostDelayText) ?? missing)
                }
                ResultTile(label: UIText.Atlas.tileInterval, value: entry.map { ResultsFormat.interval($0.intervalS) } ?? missing)
            }
            GridRow {
                ResultTile(label: UIText.Atlas.tileCause, value: entry.map { ResultsFormat.cause($0.failureMode) } ?? missing)
                ResultTile(label: UIText.Atlas.tileSeeds, value: Format.fill(UIText.Atlas.seedsPerCell, ["n": "\(data.atlas.nSeeds)"]))
            }
        }
    }

    /// One loss condition for this path and tracker. Clicking it shows that loss in the atlas.
    private func lossRow(_ condition: String, selected: Bool) -> some View {
        let entry = data.entry(stratum: stratum.id, variant: variant, loss: condition, arm: arm)
        return Button {
            let parts = condition.split(separator: ":")
            if parts.count == 2, let level = Double(parts[1]) {
                if level > 0 { model.atlas.lossModel = String(parts[0]) }
                model.atlas.level = level
            }
        } label: {
            HStack(spacing: Space.sm) {
                Text(data.atlas.lossTitles[condition] ?? condition)
                    .font(selected ? TypeScale.bodyStrong : TypeScale.body)
                    .foregroundStyle(palette.text)
                Spacer(minLength: Space.sm)
                Text(ResultsFormat.cellText(entry))
                    .font(TypeScale.body.monospacedDigit())
                    .foregroundStyle(entry == nil ? palette.textSecondary : palette.text)
            }
            .padding(.horizontal, Space.sm)
            .frame(minHeight: LayoutMetrics.rowHeight - Space.xs)
            .background(selected ? palette.selectionFill : .clear, in: RoundedRectangle(cornerRadius: Radius.small))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
    }

    private func differenceTable(_ rows: [AtlasImprovement]) -> some View {
        VStack(alignment: .leading, spacing: Space.sm) {
            Text(UIText.Atlas.diffTitle)
                .font(TypeScale.bodyStrong)
            Grid(alignment: .leading, horizontalSpacing: Space.md, verticalSpacing: Space.xs) {
                GridRow {
                    ForEach([UIText.Atlas.diffColumnDelay, UIText.Atlas.diffColumnMedian, UIText.Atlas.diffColumnInterval], id: \.self) {
                        Text($0).font(TypeScale.caption).foregroundStyle(palette.textSecondary)
                    }
                }
                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    GridRow {
                        Text("\(ResultsFormat.milliseconds(row.delayS))")
                            .gridColumnAlignment(.trailing)
                        Text(ResultsFormat.metersValue(row.medianP95DiffM))
                            .gridColumnAlignment(.trailing)
                        Text(ResultsFormat.meterRange(low: row.intervalM[0], high: row.intervalM[1]))
                    }
                    .font(TypeScale.body.monospacedDigit())
                }
            }
            Text(UIText.Atlas.diffCaption)
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct InspectorHeader: View {
    let title: String
    let subtitle: String
    /// An optional last line, such as the atlas conditions, kept whole on its own line.
    var detail: String?
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xxs) {
            Text(title)
                .font(TypeScale.headline)
                .foregroundStyle(palette.text)
            Text(subtitle)
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if let detail {
                Text(detail)
                    .font(TypeScale.body)
                    .foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }
}
