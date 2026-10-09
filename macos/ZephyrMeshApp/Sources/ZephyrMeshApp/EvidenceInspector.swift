import AppKit
import SwiftUI
import Foundation
import UniformTypeIdentifiers

/// Portable, read-only evidence envelope emitted by the shared Python report
/// tool. This surface intentionally does not import raw traces or recompute
/// measurement metrics. The source digest identifies the input artifact; it
/// does not establish hardware performance.
struct EvidenceReport: Identifiable, Decodable {
    static let schema = "zephyr-evidence-report-1"
    static let maxBytes = 8 * 1024 * 1024
    static let maxJSONDepth = 64

    enum Kind: String, CaseIterable, Decodable {
        case espnow, vision, hardware, bench, investigation, swarm

        var label: String {
            switch self {
            case .espnow: return "ESP-NOW trace"
            case .vision: return "Vision trace"
            case .hardware: return "Hardware profile"
            case .bench: return "Bench trace"
            case .investigation: return "S6 investigation"
            case .swarm: return "S7 swarm replay"
            }
        }
    }

    enum Status: String, CaseIterable, Decodable {
        case measured, synthetic, fixture
    }

    struct Source: Decodable {
        let name: String
        let sha256: String
        let schema: String

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: DynamicCodingKey.self)
            try EvidenceReport.requireExactKeys(container, expected: ["name", "sha256", "schema"], context: "source")
            name = try container.decode(String.self, forKey: DynamicCodingKey("name"))
            sha256 = try container.decode(String.self, forKey: DynamicCodingKey("sha256"))
            schema = try container.decode(String.self, forKey: DynamicCodingKey("schema"))
            guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw EvidenceReportError.invalid("source.name must not be empty")
            }
            guard sha256.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil else {
                throw EvidenceReportError.invalid("source.sha256 must be 64 lowercase hexadecimal characters")
            }
            guard !schema.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw EvidenceReportError.invalid("source.schema must not be empty")
            }
        }
    }

    let id: String
    let schema: String
    let kind: Kind
    let source: Source
    let status: Status
    let summary: JSONValue
    let limitations: [String]

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DynamicCodingKey.self)
        try Self.requireExactKeys(container, expected: ["schema", "kind", "source", "status", "summary", "limitations"], context: "envelope")
        schema = try container.decode(String.self, forKey: DynamicCodingKey("schema"))
        guard schema == Self.schema else {
            throw EvidenceReportError.invalid("schema must be \(Self.schema)")
        }
        kind = try container.decode(Kind.self, forKey: DynamicCodingKey("kind"))
        source = try container.decode(Source.self, forKey: DynamicCodingKey("source"))
        status = try container.decode(Status.self, forKey: DynamicCodingKey("status"))
        summary = try container.decode(JSONValue.self, forKey: DynamicCodingKey("summary"))
        guard case .object = summary else {
            throw EvidenceReportError.invalid("summary must be a JSON object")
        }
        limitations = try container.decode([String].self, forKey: DynamicCodingKey("limitations"))
        guard !limitations.isEmpty,
              limitations.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw EvidenceReportError.invalid("limitations entries must not be empty")
        }
        let expectedSourceSchema = Self.sourceSchema(for: kind)
        guard source.schema == expectedSourceSchema else {
            throw EvidenceReportError.invalid("source.schema does not match kind \(kind.rawValue)")
        }
        guard let declaredSummaryStatus = Self.summaryStatus(summary, kind: kind), declaredSummaryStatus == status.rawValue else {
            throw EvidenceReportError.invalid("status must match the status retained in summary provenance")
        }
        id = "\(source.sha256)-\(kind.rawValue)"
    }

    private static func summaryStatus(_ summary: JSONValue, kind: Kind) -> String? {
        guard case .object(let values) = summary else { return nil }
        if kind == .hardware {
            guard case .string(let status) = values["status"] else { return nil }
            return status
        }
        guard case .object(let provenance) = values["provenance"],
              case .string(let status) = provenance["status"] else { return nil }
        return status
    }

    static func load(from url: URL) throws -> EvidenceReport {
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile == true else { throw EvidenceReportError.invalid("the selected item is not a regular file") }
        guard let fileSize = values.fileSize, fileSize <= maxBytes else {
            throw EvidenceReportError.invalid("report exceeds the 8 MiB size limit")
        }
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard data.count <= maxBytes else { throw EvidenceReportError.invalid("report exceeds the 8 MiB size limit") }
        var scanner = JSONStructureScanner(data: data, maxDepth: maxJSONDepth)
        try scanner.scan()
        do {
            return try JSONDecoder().decode(EvidenceReport.self, from: data)
        } catch let error as EvidenceReportError {
            throw error
        } catch {
            throw EvidenceReportError.invalid("invalid evidence report JSON: \(error.localizedDescription)")
        }
    }

    static func sourceSchema(for kind: Kind) -> String {
        switch kind {
        case .espnow: return "zephyr-espnow-trace-1"
        case .vision: return "zephyr-vision-trace-1"
        case .hardware: return "zephyr-hardware-profile-1"
        case .bench: return "zephyr-bench-trace-1"
        case .investigation: return "zephyr-s6-sweep-1"
        case .swarm: return "zephyr-s7-swarm-run-1"
        }
    }

    fileprivate static func requireExactKeys(_ container: KeyedDecodingContainer<DynamicCodingKey>, expected: [String], context: String) throws {
        let actual = Set(container.allKeys.map(\.stringValue))
        let expectedSet = Set(expected)
        guard actual == expectedSet else {
            let missing = expectedSet.subtracting(actual).sorted().joined(separator: ", ")
            let extra = actual.subtracting(expectedSet).sorted().joined(separator: ", ")
            var detail: [String] = []
            if !missing.isEmpty { detail.append("missing \(missing)") }
            if !extra.isEmpty { detail.append("unknown \(extra)") }
            throw EvidenceReportError.invalid("invalid \(context) keys (\(detail.joined(separator: "; ")))" )
        }
    }
}

enum EvidenceReportError: LocalizedError {
    case invalid(String)
    case unreadable(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let detail): return detail
        case .unreadable(let detail): return detail
        }
    }
}

/// A small JSON tree keeps unknown fields and explicit nulls visible without
/// pretending that the report has a fixed measurement shape.
indirect enum JSONValue: Decodable, Equatable {
    case object([String: JSONValue])
    case array([JSONValue])
    case string(String)
    case integer(Int64)
    case number(Double)
    case bool(Bool)
    case null

    init(from decoder: Decoder) throws {
        if let keyed = try? decoder.container(keyedBy: DynamicCodingKey.self) {
            var object: [String: JSONValue] = [:]
            for key in keyed.allKeys { object[key.stringValue] = try keyed.decode(JSONValue.self, forKey: key) }
            self = .object(object)
            return
        }
        if var unkeyed = try? decoder.unkeyedContainer() {
            var values: [JSONValue] = []
            while !unkeyed.isAtEnd { values.append(try unkeyed.decode(JSONValue.self)) }
            self = .array(values)
            return
        }
        let value = try decoder.singleValueContainer()
        if value.decodeNil() { self = .null; return }
        if let bool = try? value.decode(Bool.self) { self = .bool(bool); return }
        if let integer = try? value.decode(Int64.self) {
            guard integer <= JSONValue.safeIntegerMaximum && integer >= -JSONValue.safeIntegerMaximum else {
                throw EvidenceReportError.invalid("summary integer exceeds the exact JSON-safe range")
            }
            self = .integer(integer)
            return
        }
        if let number = try? value.decode(Double.self), number.isFinite { self = .number(number); return }
        if let string = try? value.decode(String.self) { self = .string(string); return }
        throw EvidenceReportError.invalid("summary contains an unsupported or non-finite JSON value")
    }

    static let safeIntegerMaximum: Int64 = 9_007_199_254_740_991
}

/// A bounded structural pass prevents duplicate JSON object keys and deeply
/// nested reports from reaching the decoder or SwiftUI tree. JSONDecoder still
/// performs the authoritative syntax and type validation after this pass.
struct JSONStructureScanner {
    private let bytes: [UInt8]
    private let maxDepth: Int
    private var index = 0

    init(data: Data, maxDepth: Int) {
        bytes = Array(data)
        self.maxDepth = maxDepth
    }

    mutating func scan() throws {
        try skipWhitespace()
        try parseValue(depth: 0)
        try skipWhitespace()
        guard index == bytes.count else { throw EvidenceReportError.invalid("invalid JSON trailing data") }
    }

    private mutating func parseValue(depth: Int) throws {
        try skipWhitespace()
        guard let byte = current else { throw EvidenceReportError.invalid("invalid JSON: expected a value") }
        switch byte {
        case 0x7B: try parseObject(depth: depth + 1) // {
        case 0x5B: try parseArray(depth: depth + 1) // [
        case 0x22: _ = try parseString()
        case 0x74: try parseLiteral("true")
        case 0x66: try parseLiteral("false")
        case 0x6E: try parseLiteral("null")
        default: try parseNumberToken()
        }
    }

    private mutating func parseObject(depth: Int) throws {
        try checkDepth(depth)
        advance()
        try skipWhitespace()
        var keys = Set<String>()
        if current == 0x7D { advance(); return }
        while true {
            try skipWhitespace()
            guard current == 0x22 else { throw EvidenceReportError.invalid("invalid JSON object key") }
            let key = try parseString()
            guard keys.insert(key).inserted else { throw EvidenceReportError.invalid("duplicate JSON object key: \(key)") }
            try skipWhitespace()
            guard current == 0x3A else { throw EvidenceReportError.invalid("invalid JSON object: expected colon") }
            advance()
            try parseValue(depth: depth)
            try skipWhitespace()
            if current == 0x7D { advance(); return }
            guard current == 0x2C else { throw EvidenceReportError.invalid("invalid JSON object: expected comma") }
            advance()
        }
    }

    private mutating func parseArray(depth: Int) throws {
        try checkDepth(depth)
        advance()
        try skipWhitespace()
        if current == 0x5D { advance(); return }
        while true {
            try parseValue(depth: depth)
            try skipWhitespace()
            if current == 0x5D { advance(); return }
            guard current == 0x2C else { throw EvidenceReportError.invalid("invalid JSON array: expected comma") }
            advance()
        }
    }

    private mutating func parseString() throws -> String {
        let start = index
        guard current == 0x22 else { throw EvidenceReportError.invalid("invalid JSON string") }
        advance()
        while let byte = current {
            switch byte {
            case 0x22:
                advance()
                let encoded = Data(bytes[start..<index])
                guard let string = try? JSONDecoder().decode(String.self, from: encoded) else {
                    throw EvidenceReportError.invalid("invalid JSON string")
                }
                return string
            case 0x5C:
                advance()
                guard let escaped = current else { throw EvidenceReportError.invalid("unterminated JSON escape") }
                if escaped == 0x75 {
                    for _ in 0..<4 { advance(); guard current.map(Self.isHex) == true else { throw EvidenceReportError.invalid("invalid JSON unicode escape") } }
                } else { advance() }
            default:
                guard byte >= 0x20 else { throw EvidenceReportError.invalid("unescaped control character in JSON string") }
                advance()
            }
        }
        throw EvidenceReportError.invalid("unterminated JSON string")
    }

    private mutating func parseLiteral(_ literal: String) throws {
        for expected in literal.utf8 {
            guard current == expected else { throw EvidenceReportError.invalid("invalid JSON literal") }
            advance()
        }
    }

    private mutating func parseNumberToken() throws {
        let start = index
        while let byte = current, !Self.delimiters.contains(byte) { advance() }
        guard index > start else { throw EvidenceReportError.invalid("invalid JSON value") }
        let token = String(decoding: bytes[start..<index], as: UTF8.self)
        if !token.contains(".") && !token.contains("e") && !token.contains("E") {
            let digits = token.hasPrefix("-") ? String(token.dropFirst()) : token
            let normalized = String(digits.drop(while: { $0 == "0" }))
            let magnitude = normalized.isEmpty ? "0" : normalized
            let maximum = String(JSONValue.safeIntegerMaximum)
            guard magnitude.count < maximum.count || (magnitude.count == maximum.count && magnitude <= maximum) else {
                throw EvidenceReportError.invalid("summary integer exceeds the exact JSON-safe range")
            }
        }
    }

    private mutating func skipWhitespace() throws {
        while let byte = current, byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D { advance() }
    }

    private mutating func checkDepth(_ depth: Int) throws {
        guard depth <= maxDepth else { throw EvidenceReportError.invalid("JSON nesting exceeds (maxDepth) levels") }
    }

    private var current: UInt8? { index < bytes.count ? bytes[index] : nil }
    private mutating func advance() { index += 1 }
    private static let delimiters: Set<UInt8> = [0x20, 0x09, 0x0A, 0x0D, 0x2C, 0x5D, 0x7D]
    private static func isHex(_ byte: UInt8) -> Bool { (byte >= 48 && byte <= 57) || (byte >= 65 && byte <= 70) || (byte >= 97 && byte <= 102) }
}

struct DynamicCodingKey: CodingKey, Hashable {
    let stringValue: String
    let intValue: Int? = nil
    init(_ string: String) { stringValue = string }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

@MainActor
final class EvidenceInspectorController: ObservableObject {
    @Published var report: EvidenceReport?
    @Published var reportFilename = ""
    @Published var isPresented = false
    @Published var error: EvidenceErrorMessage?

    func openPanel() {
        let panel = NSOpenPanel()
        panel.title = "Open evidence report"
        panel.message = "Choose a validated Zephyr Mesh evidence report"
        panel.prompt = "Open"
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            report = try EvidenceReport.load(from: url)
            reportFilename = url.lastPathComponent
            isPresented = true
        } catch {
            self.error = EvidenceErrorMessage(message: error.localizedDescription)
        }
    }
}

struct EvidenceErrorMessage: Identifiable {
    let id = UUID()
    let message: String
}

struct EvidenceInspectorView: View {
    let report: EvidenceReport
    let filename: String
    @Environment(\.dismiss) private var dismiss
    private let horizon = Horizon()

    var body: some View {
        VStack(alignment: .leading, spacing: ZephyrDesign.Layout.panelSpacing) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: ZephyrDesign.Layout.tightSpacing) {
                    Text("Evidence report").font(ZephyrDesign.Typography.title)
                    Text("Read-only validated envelope").font(ZephyrDesign.Typography.secondary).foregroundStyle(horizon.mist)
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            Divider()
            EvidenceMetadata(report: report, filename: filename, horizon: horizon)
            ScrollView {
                VStack(alignment: .leading, spacing: ZephyrDesign.Layout.panelSpacing) {
                    DataSection(title: "Summary", horizon: horizon) {
                        EvidenceJSONView(value: report.summary, horizon: horizon)
                    }
                    DataSection(title: "Limitations", horizon: horizon) {
                        if report.limitations.isEmpty {
                            Text("No limitations were supplied by the report producer.").font(ZephyrDesign.Typography.body).foregroundStyle(horizon.mist)
                        } else {
                            ForEach(Array(report.limitations.enumerated()), id: \.offset) { _, limitation in
                                Label(limitation, systemImage: "info.circle").font(ZephyrDesign.Typography.body).foregroundStyle(horizon.primary)
                            }
                        }
                    }
                    Text("Validation and a source digest do not establish hardware performance. Review the underlying evidence and calibration record before drawing conclusions.")
                        .font(ZephyrDesign.Typography.caption).foregroundStyle(horizon.mist)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(ZephyrDesign.Spacing.lg)
        .frame(minWidth: 560, idealWidth: 700, minHeight: 420, idealHeight: 600)
        .background(horizon.night)
    }
}

private struct EvidenceMetadata: View {
    let report: EvidenceReport
    let filename: String
    let horizon: Horizon

    var body: some View {
        VStack(alignment: .leading, spacing: ZephyrDesign.Layout.tightSpacing) {
            DataRow(label: "Filename", value: filename, horizon: horizon)
            DataRow(label: "Type", value: report.kind.label, horizon: horizon)
            DataRow(label: "Declared status", value: report.status.rawValue.capitalized, horizon: horizon)
            DataRow(label: "Source schema", value: report.source.schema, horizon: horizon)
            DataRow(label: "Source", value: report.source.name, horizon: horizon)
            DataRow(label: "Source SHA-256 (declared)", value: report.source.sha256, horizon: horizon)
        }
    }
}

private struct EvidenceJSONView: View {
    let value: JSONValue
    let horizon: Horizon

    var body: some View {
        switch value {
        case .object(let object):
            VStack(alignment: .leading, spacing: ZephyrDesign.Layout.tightSpacing) {
                ForEach(object.keys.sorted(), id: \.self) { key in
                    if let value = object[key] { EvidenceJSONEntry(key: key, value: value, horizon: horizon) }
                }
            }
        case .array(let array):
            VStack(alignment: .leading, spacing: ZephyrDesign.Layout.tightSpacing) {
                ForEach(Array(array.enumerated()), id: \.offset) { index, value in
                    EvidenceJSONEntry(key: "[\(index)]", value: value, horizon: horizon)
                }
            }
        default:
            Text(value.displayText).font(ZephyrDesign.Typography.mono).foregroundStyle(horizon.primary).textSelection(.enabled)
        }
    }
}

private struct EvidenceJSONEntry: View {
    let key: String
    let value: JSONValue
    let horizon: Horizon

    var body: some View {
        if value.isContainer {
            DisclosureGroup {
                EvidenceJSONView(value: value, horizon: horizon)
            } label: {
                Text(key).font(ZephyrDesign.Typography.secondary).foregroundStyle(horizon.mist)
            }
        } else {
            HStack(alignment: .top, spacing: ZephyrDesign.Layout.compactSpacing) {
                Text(key).font(ZephyrDesign.Typography.secondary).foregroundStyle(horizon.mist)
                Spacer(minLength: 8)
                Text(value.displayText).font(ZephyrDesign.Typography.mono).foregroundStyle(horizon.primary).multilineTextAlignment(.trailing).textSelection(.enabled)
            }
        }
    }
}

private extension JSONValue {
    var isContainer: Bool {
        switch self { case .object, .array: return true; default: return false }
    }

    var displayText: String {
        switch self {
        case .object(let values): return "{\(values.count) fields}"
        case .array(let values): return "[\(values.count) items]"
        case .string(let value): return value
        case .integer(let value): return String(value)
        case .number(let value): return String(value)
        case .bool(let value): return value ? "true" : "false"
        case .null: return "null"
        }
    }
}
