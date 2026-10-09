import XCTest
import CryptoKit
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

    private func rehashedScenarioData(fixture: URL, mutateFrames: ([[String: Any]]) -> [[String: Any]]) throws -> Data {
        var document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [String: Any])
        let payloadText = try XCTUnwrap(document["payload_canonical"] as? String)
        var payload = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(payloadText.dropLast().utf8)) as? [String: Any])
        let frames = try XCTUnwrap(payload["frames"] as? [[String: Any]])
        payload["frames"] = mutateFrames(frames)
        let canonical = String(data: try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]), encoding: .utf8)! + "\n"
        document["payload_canonical"] = canonical
        document["payload_sha256"] = SHA256.hash(data: Data(canonical.utf8)).map { String(format: "%02x", $0) }.joined()
        return try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys])
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

    func testScenarioRunDecoderShowsSummaryAndFrames() throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/zephyr-s7-scenario-run-1.json")
        let data = try Data(contentsOf: fixture)
        let run = try ScenarioRunDocument.load(from: fixture)
        XCTAssertEqual(run.schema, ScenarioRunDocument.schema)
        XCTAssertEqual(run.frames.count, 1)
        XCTAssertEqual(run.frames[0].agentCount, 1)
        XCTAssertEqual(run.status, "synthetic deterministic replay; read-only")
        var tampered = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        tampered["scenario_hash"] = String(repeating: "0", count: 64)
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: JSONSerialization.data(withJSONObject: tampered)))
        var extraFrame = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var frames = try XCTUnwrap(extraFrame["frames"] as? [[String: Any]])
        frames[0]["unexpected"] = true
        extraFrame["frames"] = frames
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: JSONSerialization.data(withJSONObject: extraFrame)))
        var payloadTampered = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        payloadTampered["payload_sha256"] = String(repeating: "0", count: 64)
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: JSONSerialization.data(withJSONObject: payloadTampered)))
        var linkTampered = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var parameters = try XCTUnwrap(linkTampered["parameters"] as? [String: Any])
        var targetLink = try XCTUnwrap(parameters["target_link"] as? [String: Any])
        targetLink["delay_jitter_s"] = 0.1
        parameters["target_link"] = targetLink
        linkTampered["parameters"] = parameters
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: JSONSerialization.data(withJSONObject: linkTampered)))

    }

    func testScenarioRunDecoderRejectsWrongSchema() throws {
        let document: [String: Any] = ["schema": "wrong"]
        let data = try JSONSerialization.data(withJSONObject: document)
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: data))
    }

    func testScenarioRunFileRejectsDuplicateKeysBeforeDecoding() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        try Data(#"{"schema":"zephyr-s7-scenario-run-1","schema":"zephyr-s7-scenario-run-1"}"#.utf8).write(to: file)
        XCTAssertThrowsError(try ScenarioRunDocument.load(from: file))
    }

    func testLoadsPythonGeneratedScenarioFixture() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/zephyr-s7-scenario-run-1.json")
        let run = try ScenarioRunDocument.load(from: fixture)
        XCTAssertEqual(run.schema, ScenarioRunDocument.schema)
        XCTAssertEqual(run.frames.count, 1)
        XCTAssertEqual(run.parametersCanonical.last, "\n")
        XCTAssertEqual(run.status, "synthetic deterministic replay; read-only")
        XCTAssertEqual(run.scenarioHash.count, 64)
        XCTAssertEqual(run.payloadSHA256.count, 64)
        XCTAssertFalse(run.evidenceBoundary.isEmpty)
        XCTAssertEqual(run.dtText, "0.05")
        XCTAssertEqual(run.generatorText, "sim.scenario_runner")
        XCTAssertFalse(run.pythonText.isEmpty)
        XCTAssertFalse(run.numpyText.isEmpty)
        XCTAssertFalse(run.codeRevisionText.isEmpty)
    }

    func testLoadsMovingKeepOutSnapshotsFromPythonFixture() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/zephyr-s7-moving-scenario-run-1.json")
        let run = try ScenarioRunDocument.load(from: fixture)
        XCTAssertEqual(run.frames.count, 3)
        XCTAssertEqual(run.frames[0].keepOutSpheres.count, 1)
        guard case .array(let center) = run.frames[1].keepOutSpheres[0]["center_m"],
              case .number(let x) = center[0] else {
            return XCTFail("Expected a moving center vector")
        }
        XCTAssertEqual(x, 2.025, accuracy: 1e-12)
        guard case .string(let label) = run.frames[2].keepOutSpheres[0]["label"] else {
            return XCTFail("Expected moving sphere label")
        }
        XCTAssertEqual(label, "moving-wall")
    }

    func testScenarioPlotGeometryTracksMovingFrameAndProjectsInsideCanvas() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/zephyr-s7-moving-scenario-run-1.json")
        let run = try ScenarioRunDocument.load(from: fixture)
        let first = ScenarioPlotGeometry.make(frame: run.frames[0])
        let second = ScenarioPlotGeometry.make(frame: run.frames[1])
        XCTAssertEqual(first.keepOutSpheres[0].center.x, 2.0, accuracy: 1e-12)
        XCTAssertEqual(second.keepOutSpheres[0].center.x, 2.025, accuracy: 1e-12)
        XCTAssertEqual(first.targetAltitude, 1.5, accuracy: 1e-12)
        XCTAssertEqual(first.agentAltitudes, [1.5])
        XCTAssertEqual(first.keepOutSpheres[0].centerZ, 1.5, accuracy: 1e-12)
        let projected = ScenarioPlotGeometry.project(first.target, bounds: first.bounds, width: 300, height: 180)
        XCTAssertGreaterThanOrEqual(projected.x, 0)
        XCTAssertLessThanOrEqual(projected.x, 300)
        XCTAssertGreaterThanOrEqual(projected.y, 0)
        XCTAssertLessThanOrEqual(projected.y, 180)
        let altitudeY = ScenarioPlotGeometry.projectAltitude(first.targetAltitude, bounds: first.altitudeBounds, height: 180)
        XCTAssertGreaterThanOrEqual(altitudeY, 0)
        XCTAssertLessThanOrEqual(altitudeY, 180)
    }

    func testScenarioPlotGeometryPreservesVariedAltitudesAndTinyProfileBounds() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/zephyr-s7-moving-scenario-run-1.json")
        let document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [String: Any])
        var frames = try XCTUnwrap(document["frames"] as? [[String: Any]])
        frames[0]["target_position_m"] = [0.0, 0.0, 3.0]
        var agents = try XCTUnwrap(frames[0]["agents"] as? [[String: Any]])
        agents[0]["position_m"] = [1.0, 2.0, -1.0]
        frames[0]["agents"] = agents
        var spheres = try XCTUnwrap(frames[0]["keep_out_spheres"] as? [[String: Any]])
        spheres[0]["center_m"] = [0.0, 0.0, 2.0]
        spheres[0]["radius_m"] = 0.75
        frames[0]["keep_out_spheres"] = spheres
        let frameData = try JSONSerialization.data(withJSONObject: frames[0], options: [.sortedKeys])
        let frame = try JSONDecoder().decode(ScenarioRunFrame.self, from: frameData)
        let geometry = ScenarioPlotGeometry.make(frame: frame)
        XCTAssertEqual(geometry.targetAltitude, 3.0, accuracy: 1e-12)
        XCTAssertEqual(geometry.agentAltitudes, [-1.0])
        XCTAssertEqual(geometry.keepOutSpheres[0].centerZ, 2.0, accuracy: 1e-12)
        XCTAssertLessThan(geometry.altitudeBounds.minZ, 1.25)
        XCTAssertGreaterThan(geometry.altitudeBounds.maxZ, 2.75)
        let low = ScenarioPlotGeometry.projectAltitude(-1.0, bounds: geometry.altitudeBounds, height: 2, inset: 30)
        let high = ScenarioPlotGeometry.projectAltitude(3.0, bounds: geometry.altitudeBounds, height: 2, inset: 30)
        XCTAssertGreaterThanOrEqual(low, 0)
        XCTAssertLessThanOrEqual(low, 2)
        XCTAssertGreaterThanOrEqual(high, 0)
        XCTAssertLessThanOrEqual(high, 2)
        XCTAssertLessThanOrEqual(ScenarioPlotGeometry.projectAltitudeRadius(0.75, bounds: geometry.altitudeBounds, height: 2, inset: 30), 1)
        XCTAssertEqual(ScenarioPlotGeometry.clampRailX(width: 1, preferred: 12), 1, accuracy: 1e-12)
        let rail = ScenarioPlotGeometry.railEndpoints(height: 1)
        XCTAssertEqual(rail.top, 0.5, accuracy: 1e-12)
        XCTAssertEqual(rail.bottom, 0.5, accuracy: 1e-12)
        XCTAssertEqual(geometry.altitudeBounds.minZ, -1.25, accuracy: 1e-12)
        XCTAssertEqual(geometry.altitudeBounds.maxZ, 3.25, accuracy: 1e-12)
        XCTAssertEqual(ScenarioPlotGeometry.projectAltitude(3.0, bounds: geometry.altitudeBounds, height: 100, inset: 18), 21.555555555555557, accuracy: 1e-12)
        XCTAssertEqual(ScenarioPlotGeometry.projectAltitude(-1.0, bounds: geometry.altitudeBounds, height: 100, inset: 18), 78.44444444444444, accuracy: 1e-12)
        XCTAssertEqual(ScenarioPlotGeometry.projectAltitude(2.0, bounds: geometry.altitudeBounds, height: 100, inset: 18), 35.77777777777778, accuracy: 1e-12)
        XCTAssertEqual(ScenarioPlotGeometry.projectAltitudeRadius(0.75, bounds: geometry.altitudeBounds, height: 100, inset: 18), 10.666666666666666, accuracy: 1e-12)
    }

    func testRejectsRehashedFrameTimeSequence() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/zephyr-s7-moving-scenario-run-1.json")
        var document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [String: Any])
        var frames = try XCTUnwrap(document["frames"] as? [[String: Any]])
        frames[1]["time_s"] = 0.07
        document["frames"] = frames
        var payloadCanonical = try XCTUnwrap(document["payload_canonical"] as? String)
        payloadCanonical = payloadCanonical.replacingOccurrences(of: "\"time_s\":0.05", with: "\"time_s\":0.07")
        document["payload_canonical"] = payloadCanonical
        document["payload_sha256"] = SHA256.hash(data: Data(payloadCanonical.utf8)).map { String(format: "%02x", $0) }.joined()
        let data = try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys])
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: data))
    }

    func testRejectsRehashedStepMutationAndReorderAndChecksTolerance() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/zephyr-s7-moving-scenario-run-1.json")
        let stepMutation = try rehashedScenarioData(fixture: fixture) { original in
            var frames = original
            frames[1]["step_index"] = 0
            return frames
        }
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: stepMutation))

        let reordered = try rehashedScenarioData(fixture: fixture) { original in
            [original[1], original[0], original[2]]
        }
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: reordered))

        var justInsideDocument = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [String: Any])
        var justInsideCanonical = try XCTUnwrap(justInsideDocument["payload_canonical"] as? String)
        justInsideCanonical = justInsideCanonical.replacingOccurrences(of: "\"time_s\":0.05", with: "\"time_s\":0.0500000005")
        justInsideDocument["payload_canonical"] = justInsideCanonical
        justInsideDocument["payload_sha256"] = SHA256.hash(data: Data(justInsideCanonical.utf8)).map { String(format: "%02x", $0) }.joined()
        var justInsideFrames = try XCTUnwrap(justInsideDocument["frames"] as? [[String: Any]])
        justInsideFrames[1]["time_s"] = 0.0500000005
        justInsideDocument["frames"] = justInsideFrames
        let justInside = try JSONSerialization.data(withJSONObject: justInsideDocument, options: [.sortedKeys])
        XCTAssertNoThrow(try JSONDecoder().decode(ScenarioRunDocument.self, from: justInside))

        var justOutsideDocument = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [String: Any])
        var justOutsideCanonical = try XCTUnwrap(justOutsideDocument["payload_canonical"] as? String)
        justOutsideCanonical = justOutsideCanonical.replacingOccurrences(of: "\"time_s\":0.05", with: "\"time_s\":0.0500000015")
        justOutsideDocument["payload_canonical"] = justOutsideCanonical
        justOutsideDocument["payload_sha256"] = SHA256.hash(data: Data(justOutsideCanonical.utf8)).map { String(format: "%02x", $0) }.joined()
        var justOutsideFrames = try XCTUnwrap(justOutsideDocument["frames"] as? [[String: Any]])
        justOutsideFrames[1]["time_s"] = 0.0500000015
        justOutsideDocument["frames"] = justOutsideFrames
        let justOutside = try JSONSerialization.data(withJSONObject: justOutsideDocument, options: [.sortedKeys])
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: justOutside))
    }

    func testRejectsMovingSphereRadiusAboveBoundAfterCanonicalRehash() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/zephyr-s7-moving-scenario-run-1.json")
        var document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [String: Any])
        var parameters = try XCTUnwrap(document["parameters"] as? [String: Any])
        var spheres = try XCTUnwrap(parameters["keep_out_spheres"] as? [[String: Any]])
        spheres[0]["radius_m"] = 100.1
        parameters["keep_out_spheres"] = spheres
        document["parameters"] = parameters
        var canonical = try XCTUnwrap(document["parameters_canonical"] as? String)
        canonical = canonical.replacingOccurrences(of: "\"radius_m\":0.5", with: "\"radius_m\":100.1")
        document["parameters_canonical"] = canonical
        document["scenario_hash"] = SHA256.hash(data: Data(canonical.utf8)).map { String(format: "%02x", $0) }.joined()
        let data = try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys])
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: data))
    }

    func testScenarioRunCursorClampsStepsAndResets() {
        var cursor = ScenarioRunCursor(frameCount: 3)
        XCTAssertEqual(cursor.index, 0)
        cursor.step(-10)
        XCTAssertEqual(cursor.index, 0)
        cursor.step(1)
        XCTAssertEqual(cursor.index, 1)
        cursor.seek(100)
        XCTAssertEqual(cursor.index, 2)
        cursor.step()
        XCTAssertEqual(cursor.index, 2)
        cursor.reset()
        XCTAssertEqual(cursor.index, 0)
        cursor.seek(2)
        cursor.replace(frameCount: 1)
        XCTAssertEqual(cursor.frameCount, 1)
        XCTAssertEqual(cursor.index, 0)
    }

    func testScenarioRunIdentityIncludesPayloadDigest() {
        let first = ScenarioRunIdentity(scenarioHash: "scenario", payloadSHA256: "payload-a")
        let changedPayload = ScenarioRunIdentity(scenarioHash: "scenario", payloadSHA256: "payload-b")
        XCTAssertNotEqual(first, changedPayload)
        var cursor = ScenarioRunCursor(frameCount: 3)
        cursor.seek(2)
        cursor.replace(frameCount: 1)
        XCTAssertEqual(cursor.index, 0)
    }

    func testScenarioRunRejectsSemanticallyEquivalentNoncanonicalIntegrityText() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/zephyr-s7-scenario-run-1.json")
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [String: Any])

        var reordered = original
        let parameters = try XCTUnwrap(reordered["parameters_canonical"] as? String)
        let reorderedParameters = parameters.replacingOccurrences(
            of: "{\"agent_count\":1,\"altitude_m\":1.5,",
            with: "{\"altitude_m\":1.5,\"agent_count\":1,"
        )
        XCTAssertNotEqual(reorderedParameters, parameters)
        reordered["parameters_canonical"] = reorderedParameters
        reordered["scenario_hash"] = sha256Hex(reorderedParameters)
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: JSONSerialization.data(withJSONObject: reordered)))

        var numericSpelling = original
        let numeric = try XCTUnwrap(numericSpelling["parameters_canonical"] as? String)
        let alternateNumber = numeric.replacingOccurrences(of: "\"altitude_m\":1.5", with: "\"altitude_m\":1.50")
        XCTAssertNotEqual(alternateNumber, numeric)
        numericSpelling["parameters_canonical"] = alternateNumber
        numericSpelling["scenario_hash"] = sha256Hex(alternateNumber)
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: JSONSerialization.data(withJSONObject: numericSpelling)))

        var parameterWhitespace = original
        let parameterText = try XCTUnwrap(parameterWhitespace["parameters_canonical"] as? String)
        let spacedParameters = parameterText.replacingOccurrences(of: "{\"agent_count\"", with: "{ \"agent_count\"")
        XCTAssertNotEqual(spacedParameters, parameterText)
        parameterWhitespace["parameters_canonical"] = spacedParameters
        parameterWhitespace["scenario_hash"] = sha256Hex(spacedParameters)
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: JSONSerialization.data(withJSONObject: parameterWhitespace)))

        var whitespace = original
        let payload = try XCTUnwrap(whitespace["payload_canonical"] as? String)
        let spacedPayload = payload.replacingOccurrences(of: "{\"frames\":", with: "{ \"frames\":")
        XCTAssertNotEqual(spacedPayload, payload)
        whitespace["payload_canonical"] = spacedPayload
        whitespace["payload_sha256"] = sha256Hex(spacedPayload)
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: JSONSerialization.data(withJSONObject: whitespace)))
    }

    func testScenarioRunUsesJSONSafeIntegerSeedBoundary() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/zephyr-s7-scenario-run-1.json")
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [String: Any])
        let maximum: Int64 = 9_007_199_254_740_991
        let canonical = try XCTUnwrap(original["parameters_canonical"] as? String)
        let safeText = canonical.replacingOccurrences(of: "\"seed\":7", with: "\"seed\":\(maximum)")
        XCTAssertNotEqual(safeText, canonical)

        var safe = original
        var safeParameters = try XCTUnwrap(safe["parameters"] as? [String: Any])
        safeParameters["seed"] = maximum
        safe["parameters"] = safeParameters
        var safeProvenance = try XCTUnwrap(safe["provenance"] as? [String: Any])
        safeProvenance["seed"] = maximum
        safe["provenance"] = safeProvenance
        safe["parameters_canonical"] = safeText
        safe["scenario_hash"] = sha256Hex(safeText)
        XCTAssertNoThrow(try JSONDecoder().decode(ScenarioRunDocument.self, from: JSONSerialization.data(withJSONObject: safe)))

        var above = safe
        let aboveMaximum = maximum + 1
        var aboveParameters = try XCTUnwrap(above["parameters"] as? [String: Any])
        aboveParameters["seed"] = aboveMaximum
        above["parameters"] = aboveParameters
        var aboveProvenance = try XCTUnwrap(above["provenance"] as? [String: Any])
        aboveProvenance["seed"] = aboveMaximum
        above["provenance"] = aboveProvenance
        let aboveText = safeText.replacingOccurrences(of: "\"seed\":\(maximum)", with: "\"seed\":\(aboveMaximum)")
        above["parameters_canonical"] = aboveText
        above["scenario_hash"] = sha256Hex(aboveText)
        XCTAssertThrowsError(try JSONDecoder().decode(ScenarioRunDocument.self, from: JSONSerialization.data(withJSONObject: above)))
    }

    private func writeTemporary(_ text: String) -> URL {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).json")
        try! Data(text.utf8).write(to: file)
        return file
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
