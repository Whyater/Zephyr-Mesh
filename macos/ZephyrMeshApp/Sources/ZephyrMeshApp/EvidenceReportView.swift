import SwiftUI

/// Evidence report window: the verdict (kind and declared status) first, then limitations and
/// a plain summary with shared labels, then collapsed Details with digests and the raw tree.
struct EvidenceReportView: View {
    let report: EvidenceReport
    let filename: String
    @Environment(\.palette) private var palette
    @State private var showDetails = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Space.xl) {
                verdict
                SectionBlock(title: "Limitations") {
                    VStack(alignment: .leading, spacing: Space.sm) {
                        ForEach(Array(report.limitations.enumerated()), id: \.offset) { _, limitation in
                            HStack(alignment: .firstTextBaseline, spacing: Space.sm) {
                                StatusShape(style: StatusStyle(level: "unavailable"))
                                Text(limitation)
                                    .font(TypeScale.body)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                }
                SectionBlock(title: "Summary") {
                    SummaryRows(value: report.summary)
                }
                DetailsSection(isExpanded: $showDetails) {
                    ValueRow(label: "File", value: filename)
                    ValueRow(label: Format.label("schema"), value: report.schema, style: .identifier)
                    ValueRow(label: Format.label("source"), value: report.source.name)
                    ValueRow(label: Format.label("source_schema"), value: report.source.schema, style: .identifier)
                    StackedValue(label: "Source SHA-256 (declared)", value: report.source.sha256, style: .identifier)
                    if case .object(let summary) = report.summary, case .string(let schema)? = summary["schema"] {
                        ValueRow(label: "Summary schema", value: schema, style: .identifier)
                    }
                    RawTreeBlock(title: Format.label("raw_summary"), value: report.summary)
                }
            }
            .modifier(ViewerContent())
        }
        .frame(minWidth: LayoutMetrics.evidenceWindowMinWidth, minHeight: LayoutMetrics.evidenceWindowMinHeight)
        .background(palette.background)
        .navigationTitle(report.kind.label)
        .navigationSubtitle(filename)
    }

    /// Kind and declared status as shape plus word (info, never an alert color), then the boundary.
    private var verdict: some View {
        VStack(alignment: .leading, spacing: Space.sm) {
            HStack(alignment: .firstTextBaseline, spacing: Space.sm) {
                Text("Declared status")
                    .font(TypeScale.body)
                    .foregroundStyle(palette.textSecondary)
                StatusLabel(style: StatusStyle(level: "info", word: Format.label(report.status.rawValue)), font: TypeScale.headline)
            }
            Text("A valid report and a matching source digest show which file this summarizes. "
                 + "They do not establish hardware performance; check the underlying evidence and calibration first.")
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(LayoutMetrics.contentInset)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.surface, in: RoundedRectangle(cornerRadius: Radius.small))
        .overlay { RoundedRectangle(cornerRadius: Radius.small).strokeBorder(palette.line, lineWidth: 1) }
    }
}

/// Plain summary: top-level values as labeled rows, one level of nested objects as labeled
/// groups, anything deeper summarized with a pointer to Details. Keys use shared labels.
/// Groups are `Space.xl` apart, pairs inside a group `Space.sm` or more, and a label sits
/// `Space.xxs` above its own value, so each label clearly belongs to its value. One rule decides
/// the layout: prose (a text value with spaces) stacks under its label; everything else sits in
/// the value column beside it.
struct SummaryRows: View {
    let value: JSONValue
    @Environment(\.palette) private var palette

    var body: some View {
        if case .object(let object) = value {
            let plan = Self.plan(object)
            VStack(alignment: .leading, spacing: Space.xl) {
                if !plan.scalars.isEmpty {
                    VStack(alignment: .leading, spacing: Space.xs) {
                        ForEach(plan.scalars, id: \.self) { key in
                            SummaryLine(label: Format.label(key), value: object[key]?.plainText ?? Format.notRecorded,
                                        isProse: object[key]?.isProse ?? false)
                        }
                    }
                }
                ForEach(plan.groups, id: \.self) { key in
                    VStack(alignment: .leading, spacing: Space.xs) {
                        Text(Format.label(key))
                            .font(TypeScale.bodyStrong)
                        nested(object[key] ?? .null)
                    }
                }
                moreNote(plan.hidden)
            }
        } else {
            Text(value.plainText)
                .font(TypeScale.body)
        }
    }

    /// What the summary lists: plain values first, then groups, at most `JSONTree.limit` in
    /// all; the rest is counted and left to the Details raw tree, so a huge shallow report
    /// cannot build thousands of rows.
    struct Plan: Equatable {
        let scalars: [String]
        let groups: [String]
        let hidden: Int
    }

    static func plan(_ object: [String: JSONValue]) -> Plan {
        let keys = object.keys.filter { !detailsOnlyKeys.contains($0) }.sorted { order($0) < order($1) }
        let scalars = keys.filter { !(object[$0]?.isBranch ?? false) }
        let groups = keys.filter { object[$0]?.hasPlainSummary ?? false }
        let shownScalars = Array(scalars.prefix(JSONTree.limit))
        let shownGroups = Array(groups.prefix(max(JSONTree.limit - shownScalars.count, 0)))
        let hidden = scalars.count + groups.count - shownScalars.count - shownGroups.count
        return Plan(scalars: shownScalars, groups: shownGroups, hidden: hidden)
    }

    @ViewBuilder private func moreNote(_ hidden: Int) -> some View {
        if hidden > 0 {
            Text("\(hidden) more in Details")
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
        }
    }

    @ViewBuilder private func nested(_ value: JSONValue) -> some View {
        switch value {
        case .object(let object):
            // Rows that cannot be put in plain words (for example one block per drone) are left out
            // here; they stay complete in the Details raw tree.
            let unit = object["unit"]?.isBranch == false ? object["unit"]?.plainText : nil
            let rows = Self.orderedKeys(object).filter { object[$0]?.summaryText() != nil }
            ForEach(rows.prefix(JSONTree.limit), id: \.self) { key in
                let child = object[key] ?? .null
                let label = Format.label(key)
                let text = child.summaryText(includeUnit: !label.hasSuffix(")")) ?? ""
                SummaryLine(label: label, value: Self.withUnit(text, key: key, value: child, unit: unit), isProse: child.isProse)
            }
            moreNote(rows.count - min(rows.count, JSONTree.limit))
        default:
            ValueRow(label: "Items", value: value.branchSummary)
        }
    }

    /// Schema ids are identifiers, not findings; they are listed under Details instead.
    static let detailsOnlyKeys: Set<String> = ["schema"]

    /// Statistics in reading order (count, then min to max); the unit is appended to the values,
    /// not listed as a row. Everything else alphabetical.
    static let statisticOrder = ["available", "count", "min", "mean", "p50", "p95", "max"]

    static func orderedKeys(_ object: [String: JSONValue]) -> [String] {
        object.keys.filter { $0 != "unit" || object[$0]?.isBranch == true }.sorted { left, right in
            let l = statisticOrder.firstIndex(of: left) ?? statisticOrder.count
            let r = statisticOrder.firstIndex(of: right) ?? statisticOrder.count
            return l == r ? left < right : l < r
        }
    }

    /// "0.012" becomes "0.012 s" for a statistic in a block that declares its unit.
    static func withUnit(_ text: String, key: String, value: JSONValue, unit: String?) -> String {
        guard let unit, ["min", "mean", "p50", "p95", "max"].contains(key), value.isNumber else { return text }
        return "\(text) \(unit)"
    }

    /// Provenance reads last; everything else alphabetical.
    private static func order(_ key: String) -> String { key == "provenance" ? "~" : key }
}

/// A label and value. Prose stacks under its label; other values sit beside it, and stack only
/// if they cannot fit the width at all.
struct SummaryLine: View {
    let label: String
    let value: String
    var isProse = false

    var body: some View {
        let sentence = value.prefix(1).uppercased() + value.dropFirst()
        if isProse {
            stacked(sentence)
        } else {
            ViewThatFits(in: .horizontal) {
                ValueRow(label: label, value: sentence)
                    .fixedSize(horizontal: true, vertical: false)
                stacked(sentence)
            }
        }
    }

    // The gap under a stacked pair is larger than the one inside it.
    private func stacked(_ sentence: String) -> some View {
        StackedValue(label: label, value: sentence)
            .padding(.vertical, Space.xs)
    }
}

extension JSONValue {
    /// A text value with more than one word, laid out as a paragraph.
    var isProse: Bool {
        if case .string(let text) = self { return text.contains(" ") }
        return false
    }

    var isNumber: Bool {
        switch self {
        case .integer, .number: return true
        default: return false
        }
    }

    var isBranch: Bool {
        switch self {
        case .object, .array: return true
        default: return false
        }
    }

    /// Display text for a leaf. Null is "Not recorded"; booleans read Yes and No.
    var plainText: String {
        switch self {
        case .string(let text): return text.isEmpty ? Format.notRecorded : text
        case .integer(let number): return String(number)
        case .number(let number): return number.formatted(.number.precision(.significantDigits(1...4)).grouping(.never))
        case .bool(let flag): return flag ? "Yes" : "No"
        case .null: return Format.notRecorded
        case .object, .array: return branchSummary
        }
    }

    /// Plain words for a nested value: statistics blocks read "Mean 0.753, p95 0.753, max
    /// 0.753 m (300 samples)" from the values present; empty objects read "None"; other
    /// branches give their size and live under Details.
    /// Nil when the value has no plain-words form; the caller then leaves the row out.
    func summaryText(includeUnit: Bool = true) -> String? {
        if case .array = self { return branchSummary.contains(" item") ? nil : branchSummary }
        guard case .object(let object) = self else { return plainText }
        if object.isEmpty { return Format.label("none") }
        let statKeys = ["mean", "p50", "p95", "max", "min", "count"]
        guard object.keys.contains(where: { statKeys.contains($0) }) else { return nil }
        if case .bool(false)? = object["available"] { return Format.notRecorded }
        let unit = includeUnit ? (object["unit"].map { " \($0.plainText)" } ?? "") : ""
        let parts = [("Mean", "mean"), ("p95", "p95"), ("max", "max")].compactMap { name, key -> String? in
            guard let value = object[key], value != .null else { return nil }
            return "\(name) \(value.plainText)"
        }
        let samples = object["count"].map { " (\($0.plainText) samples)" } ?? ""
        return parts.isEmpty ? branchSummary : parts.joined(separator: ", ") + unit + samples
    }

    /// A branch worth a Summary section: an object with at least one row in plain words, or a
    /// short list of plain values.
    var hasPlainSummary: Bool {
        switch self {
        case .object(let object): return object.values.contains { $0.summaryText() != nil }
        case .array: return summaryText() != nil
        default: return false
        }
    }

    var branchSummary: String {
        switch self {
        case .object(let object): return Format.count(object.count, noun: "field")
        case .array(let items):
            let leaves = items.filter { !$0.isBranch }
            if !items.isEmpty, leaves.count == items.count, items.count <= 6 {
                return items.map(\.plainText).joined(separator: ", ")
            }
            return Format.count(items.count, noun: "item")
        default: return plainText
        }
    }
}
