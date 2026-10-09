import XCTest
@testable import ZephyrMeshApp

final class EvidenceInspectorTests: XCTestCase {
    private func envelope(kind: EvidenceReport.Kind = .espnow) -> [String: Any] {
        var summary: [String: Any] = [
            "observations": 3,
            "unknown": NSNull(),
            "nested": ["flag": false, "items": [NSNull(), "unknown", 1.25]]
        ]
        if kind == .hardware {
            summary["status"] = "fixture"
        } else {
            summary["provenance"] = ["status": "fixture"]
        }
        return [
            "schema": EvidenceReport.schema,
            "kind": kind.rawValue,
            "source": [
                "name": "example.json",
                "sha256": String(repeating: "a", count: 64),
                "schema": EvidenceReport.sourceSchema(for: kind)
            ],
            "status": "fixture",
            "summary": summary,
            "limitations": ["Fixture data is not measured performance."]
        ]
    }

    private func decode(_ document: [String: Any]) throws -> EvidenceReport {
        let data = try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys])
        return try JSONDecoder().decode(EvidenceReport.self, from: data)
    }

    private func replaceSource(_ document: inout [String: Any], key: String, value: Any) {
        var source = document["source"] as! [String: Any]
        source[key] = value
        document["source"] = source
    }

    func testDecodesAllSupportedKindsWithMatchingSourceSchemas() throws {
        for kind in EvidenceReport.Kind.allCases {
            let report = try decode(envelope(kind: kind))
            XCTAssertEqual(report.kind, kind)
            XCTAssertEqual(report.status, .fixture)
            XCTAssertEqual(report.source.schema, EvidenceReport.sourceSchema(for: kind))
            XCTAssertEqual(report.source.sha256, String(repeating: "a", count: 64))
        }
    }

    func testPreservesNestedNullsUnknownStringsAndBooleans() throws {
        let report = try decode(envelope())
        guard case .object(let summary) = report.summary,
              case .object(let nested) = summary["nested"],
              case .array(let items) = nested["items"] else {
            XCTFail("Expected the summary's nested JSON structure to remain available")
            return
        }
        XCTAssertEqual(summary["unknown"], .null)
        XCTAssertEqual(nested["flag"], .bool(false))
        XCTAssertEqual(items, [.null, .string("unknown"), .number(1.25)])
        XCTAssertEqual(summary["observations"], .integer(3))
    }

    func testPreservesSafeIntegersAndRejectsUnsafeIntegerMagnitude() throws {
        var document = envelope()
        var summary = document["summary"] as! [String: Any]
        summary["safe"] = 9_007_199_254_740_991
        document["summary"] = summary
        let report = try decode(document)
        guard case .object(let values) = report.summary else { return XCTFail("Expected object summary") }
        XCTAssertEqual(values["safe"], .integer(9_007_199_254_740_991))

        summary["unsafe"] = 9_007_199_254_740_992
        document["summary"] = summary
        XCTAssertThrowsError(try decode(document))
    }

    func testRejectsDuplicateKeysAndExcessiveNestingBeforeDecoding() throws {
        let duplicate = #"{"schema":"zephyr-evidence-report-1","schema":"zephyr-evidence-report-1","kind":"espnow","source":{"name":"e.json","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","schema":"zephyr-espnow-trace-1"},"status":"fixture","summary":{},"limitations":["fixture"]}"#
        XCTAssertThrowsError(try EvidenceReport.load(from: writeTemporary(duplicate)))

        var nested = "{}"
        for _ in 0...EvidenceReport.maxJSONDepth { nested = "{\"next\":\(nested)}" }
        let path = writeTemporary("{\"schema\":\"zephyr-evidence-report-1\",\"kind\":\"espnow\",\"source\":{\"name\":\"e.json\",\"sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"schema\":\"zephyr-espnow-trace-1\"},\"status\":\"fixture\",\"summary\":\(nested),\"limitations\":[\"fixture\"]}")
        defer { try? FileManager.default.removeItem(at: path) }
        XCTAssertThrowsError(try EvidenceReport.load(from: path))
    }

    func testRejectsWrongEnvelopeAndKindSourceSchemas() {
        var document = envelope()
        document["schema"] = "zephyr-evidence-report-2"
        XCTAssertThrowsError(try decode(document))
        document = envelope()
        replaceSource(&document, key: "schema", value: EvidenceReport.sourceSchema(for: .vision))
        XCTAssertThrowsError(try decode(document))
    }

    func testRejectsUnknownOrMissingEnvelopeAndSourceKeys() {
        var document = envelope()
        document["live_command"] = "arm"
        XCTAssertThrowsError(try decode(document))
        document = envelope()
        document.removeValue(forKey: "limitations")
        XCTAssertThrowsError(try decode(document))
        document = envelope()
        replaceSource(&document, key: "unexpected", value: true)
        XCTAssertThrowsError(try decode(document))
        var source = envelope()["source"] as! [String: Any]
        source.removeValue(forKey: "sha256")
        document = envelope()
        document["source"] = source
        XCTAssertThrowsError(try decode(document))
    }

    func testRejectsInvalidStatusAndKind() {
        for invalidStatus: Any in ["confirmed", "", true, NSNull()] {
            var document = envelope()
            document["status"] = invalidStatus
            XCTAssertThrowsError(try decode(document))
        }
        var document = envelope()
        document["kind"] = "telemetry"
        XCTAssertThrowsError(try decode(document))
    }

    func testRejectsInvalidDigestsAndEmptySourceNames() {
        for invalidDigest in ["", String(repeating: "A", count: 64), String(repeating: "a", count: 63), String(repeating: "g", count: 64)] {
            var document = envelope()
            replaceSource(&document, key: "sha256", value: invalidDigest)
            XCTAssertThrowsError(try decode(document))
        }
        var document = envelope()
        replaceSource(&document, key: "name", value: " \n ")
        XCTAssertThrowsError(try decode(document))
    }

    func testRejectsNonObjectSummaryAndEmptyLimitations() {
        var document = envelope()
        document["summary"] = [1, 2, 3]
        XCTAssertThrowsError(try decode(document))
        document = envelope()
        document["limitations"] = [" \n "]
        XCTAssertThrowsError(try decode(document))
    }

    func testRejectsNonFiniteAndInvalidJSON() throws {
        let valid = try JSONSerialization.data(withJSONObject: envelope(), options: [.sortedKeys])
        let text = String(data: valid, encoding: .utf8)!
        for invalidNumber in ["NaN", "Infinity", "-Infinity", "1e999"] {
            let invalid = text.replacingOccurrences(of: "1.25", with: invalidNumber)
            XCTAssertThrowsError(try JSONDecoder().decode(EvidenceReport.self, from: Data(invalid.utf8)))
        }
        XCTAssertThrowsError(try JSONDecoder().decode(EvidenceReport.self, from: Data()))
        XCTAssertThrowsError(try JSONDecoder().decode(EvidenceReport.self, from: Data("not JSON".utf8)))
    }

    func testLoadRejectsOversizedFileAndDirectory() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("large.json")
        try Data(repeating: 0x20, count: EvidenceReport.maxBytes + 1).write(to: file)
        XCTAssertThrowsError(try EvidenceReport.load(from: file))
        XCTAssertThrowsError(try EvidenceReport.load(from: directory))
    }

    func testLoadAcceptsValidPortableReportFile() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        try JSONSerialization.data(withJSONObject: envelope()).write(to: file)
        let report = try EvidenceReport.load(from: file)
        XCTAssertEqual(report.kind, .espnow)
        XCTAssertEqual(report.source.name, "example.json")
    }

    private func writeTemporary(_ text: String) -> URL {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).json")
        try! Data(text.utf8).write(to: file)
        return file
    }
}
