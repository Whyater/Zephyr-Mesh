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
                section("Limitations") {
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
                section("Summary") {
                    SummaryRows(value: report.summary)
                }
                details
            }
            .padding(Space.xl)
            .frame(maxWidth: 720, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .frame(minWidth: 560, minHeight: 440)
        .background(palette.background)
        .navigationTitle(report.kind.label)
        .navigationSubtitle(filename)
    }

    private var verdict: some View {
        VStack(alignment: .leading, spacing: Space.sm) {
            Text(report.kind.label)
                .font(TypeScale.title)
            HStack(alignment: .firstTextBaseline, spacing: Space.sm) {
                Text("Declared status")
                    .font(TypeScale.body)
                    .foregroundStyle(palette.textSecondary)
                Text(Format.label(report.status.rawValue))
                    .font(TypeScale.metricSmall)
            }
            Text("A valid report and a matching source digest show which file this summarizes. "
                 + "They do not establish hardware performance; check the underlying evidence and calibration first.")
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(Space.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.surface, in: RoundedRectangle(cornerRadius: Radius.small))
        .overlay { RoundedRectangle(cornerRadius: Radius.small).strokeBorder(palette.line, lineWidth: 1) }
    }

    private var details: some View {
        DisclosureGroup(isExpanded: $showDetails) {
            VStack(alignment: .leading, spacing: Space.sm) {
                ValueRow(label: "File", value: filename)
                ValueRow(label: Format.label("schema"), value: report.schema, monospaced: true)
                ValueRow(label: Format.label("source"), value: report.source.name)
                ValueRow(label: Format.label("source_schema"), value: report.source.schema, monospaced: true)
                StackedValue(label: "Source SHA-256 (declared)", value: report.source.sha256, monospaced: true)
                Text("Raw summary")
                    .font(TypeScale.caption)
                    .foregroundStyle(palette.textSecondary)
                    .padding(.top, Space.sm)
                JSONTree(value: report.summary)
            }
            .padding(.top, Space.sm)
        } label: {
            Text("Details").font(TypeScale.headline)
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: Space.sm) {
            Text(title)
                .font(TypeScale.headline)
                .accessibilityAddTraits(.isHeader)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Plain summary: top-level values as labeled rows, one level of nested objects as labeled
/// groups, anything deeper summarized with a pointer to Details. Keys use shared labels.
struct SummaryRows: View {
    let value: JSONValue
    @Environment(\.palette) private var palette

    var body: some View {
        if case .object(let object) = value {
            let keys = object.keys.sorted { order($0) < order($1) }
            VStack(alignment: .leading, spacing: Space.md) {
                let scalars = keys.filter { !(object[$0]?.isBranch ?? false) }
                if !scalars.isEmpty {
                    VStack(alignment: .leading, spacing: Space.xxs) {
                        ForEach(scalars, id: \.self) { key in
                            SummaryLine(label: Format.label(key), value: object[key]?.plainText ?? Format.notRecorded)
                        }
                    }
                }
                ForEach(keys.filter { object[$0]?.isBranch ?? false }, id: \.self) { key in
                    VStack(alignment: .leading, spacing: Space.xxs) {
                        Text(Format.label(key))
                            .font(TypeScale.bodyStrong)
                        nested(object[key] ?? .null)
                    }
                }
            }
        } else {
            Text(value.plainText)
                .font(TypeScale.body)
        }
    }

    @ViewBuilder private func nested(_ value: JSONValue) -> some View {
        switch value {
        case .object(let object):
            ForEach(object.keys.sorted(), id: \.self) { key in
                let child = object[key] ?? .null
                let label = Format.label(key)
                SummaryLine(label: label, value: child.summaryText(includeUnit: !label.hasSuffix(")")))
            }
        default:
            ValueRow(label: "Items", value: value.branchSummary)
        }
    }

    /// Provenance reads last; everything else alphabetical.
    private func order(_ key: String) -> String { key == "provenance" ? "~" : key }
}

/// A label and value; long text stacks under its label instead of sitting right-aligned.
struct SummaryLine: View {
    let label: String
    let value: String

    var body: some View {
        if value.count > 36 {
            StackedValue(label: label, value: value.prefix(1).uppercased() + value.dropFirst())
                .padding(.vertical, Space.xxs)
        } else {
            ValueRow(label: label, value: value)
        }
    }
}

/// Full raw tree with every key labeled, collapsed by default.
struct JSONTree: View {
    let value: JSONValue

    var body: some View {
        switch value {
        case .object(let object):
            VStack(alignment: .leading, spacing: Space.xxs) {
                ForEach(object.keys.sorted(), id: \.self) { key in
                    JSONTreeEntry(label: Format.label(key), value: object[key] ?? .null)
                }
            }
        case .array(let items):
            VStack(alignment: .leading, spacing: Space.xxs) {
                ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                    JSONTreeEntry(label: "Item \(index + 1)", value: item)
                }
            }
        default:
            Text(value.plainText).font(TypeScale.mono)
        }
    }
}

private struct JSONTreeEntry: View {
    let label: String
    let value: JSONValue
    @Environment(\.palette) private var palette

    var body: some View {
        if value.isBranch {
            DisclosureGroup {
                JSONTree(value: value)
                    .padding(.leading, Space.sm)
            } label: {
                HStack {
                    Text(label).font(TypeScale.body)
                    Spacer()
                    Text(value.branchSummary)
                        .font(TypeScale.caption)
                        .foregroundStyle(palette.textSecondary)
                }
            }
        } else {
            ValueRow(label: label, value: value.plainText, monospaced: true)
        }
    }
}

extension JSONValue {
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
    func summaryText(includeUnit: Bool = true) -> String {
        guard case .object(let object) = self else { return isBranch ? branchSummary : plainText }
        if object.isEmpty { return Format.label("none") }
        let statKeys = ["mean", "p50", "p95", "max", "min", "count"]
        guard object.keys.contains(where: { statKeys.contains($0) }) else { return "\(branchSummary), see Details" }
        if case .bool(false)? = object["available"] { return Format.notRecorded }
        let unit = includeUnit ? (object["unit"].map { " \($0.plainText)" } ?? "") : ""
        let parts = [("Mean", "mean"), ("p95", "p95"), ("max", "max")].compactMap { name, key -> String? in
            guard let value = object[key], value != .null else { return nil }
            return "\(name) \(value.plainText)"
        }
        let samples = object["count"].map { " (\($0.plainText) samples)" } ?? ""
        return parts.isEmpty ? branchSummary : parts.joined(separator: ", ") + unit + samples
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
