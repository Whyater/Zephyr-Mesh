import SceneKit
import XCTest
@testable import ZephyrMeshApp

private let fixturesDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")

private let replayURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Sources/ZephyrMeshApp/Resources/demo_swarm.json")

private func loadReplay() throws -> Fixture {
    try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: replayURL))
}

final class EventTextTests: XCTestCase {
    func testShorterVariantsDropWholeNamesAndCountThem() {
        let text = "9 drones newly late: D06, D13, D20, D22, D23 +4 more (0.20 s)"
        let variants = EventText.variants(text)
        XCTAssertEqual(variants.first, text, "the full text is tried first")
        XCTAssertEqual(variants.count, 5, "one variant per name dropped, down to one name")
        XCTAssertEqual(variants[1], "9 drones newly late: D06, D13, D20, D22 +5 more (0.20 s)")
        XCTAssertEqual(variants.last, "9 drones newly late: D06 +8 more (0.20 s)")
        // Every variant still accounts for all nine drones: names shown plus the "+N more" count.
        for variant in variants {
            let names = variant.matches(of: #/D\d{2}/#).count
            let more = variant.firstMatch(of: #/\+(\d+) more/#).map { Int($0.1) ?? 0 } ?? 0
            XCTAssertEqual(names + more, 9, variant)
        }
    }

    func testAListWithoutMoreGainsACount() {
        let variants = EventText.variants("3 drones newly late: D09, D39, D44 (0.15 s)")
        XCTAssertEqual(variants, ["3 drones newly late: D09, D39, D44 (0.15 s)",
                                  "3 drones newly late: D09, D39 +1 more (0.15 s)",
                                  "3 drones newly late: D09 +2 more (0.15 s)"])
    }

    func testTextWithoutAListIsLeftAlone() {
        XCTAssertEqual(EventText.variants("D07 went inactive"), ["D07 went inactive"])
        XCTAssertEqual(EventText.variants("Target packet lost for 6 drones"), ["Target packet lost for 6 drones"])
    }
}

final class ScenarioPlotCentringTests: XCTestCase {
    func testDataSpanningHalfAMetreSitsInTheMiddle() {
        // One drone 2 m east of the target on the same line: the y span is only the 0.5 m margin.
        let bounds = ScenarioPlotBounds(minX: -0.25, maxX: 2.25, minY: -0.25, maxY: 0.25)
        let target = ScenarioPlotGeometry.project(ScenarioPlotPoint(x: 0, y: 0), bounds: bounds, width: 600, height: 340)
        let drone = ScenarioPlotGeometry.project(ScenarioPlotPoint(x: 2, y: 0), bounds: bounds, width: 600, height: 340)
        XCTAssertEqual(target.y, 170, accuracy: 1e-9, "a point on the bounds' middle line is at half height")
        XCTAssertEqual(drone.y, 170, accuracy: 1e-9)
        XCTAssertEqual((target.x + drone.x) / 2, 300, accuracy: 1e-9, "the pair is centred horizontally")
    }

    func testAltitudeSpanningHalfAMetreSitsInTheMiddleOfTheRail() {
        let bounds = ScenarioAltitudeBounds(minZ: 1.25, maxZ: 1.75)
        XCTAssertEqual(ScenarioPlotGeometry.projectAltitude(1.5, bounds: bounds, height: 340, inset: 30), 170, accuracy: 1e-9)
        // Higher is up the rail.
        XCTAssertLessThan(ScenarioPlotGeometry.projectAltitude(1.7, bounds: bounds, height: 340, inset: 30), 170)
    }

    func testTheSingleFrameFixturePlotsBothPointsOnTheCentreLine() throws {
        let run = try ScenarioRunDocument.load(from: fixturesDirectory.appendingPathComponent("zephyr-s7-scenario-run-1.json"))
        let geometry = ScenarioPlotGeometry.make(frame: run.frames[0])
        let points = [geometry.target] + geometry.agents.map(\.point)
        for point in points {
            let projected = ScenarioPlotGeometry.project(point, bounds: geometry.bounds, width: 640, height: 340)
            XCTAssertEqual(projected.y, 170, accuracy: 1e-9)
        }
    }

    func testScaleBarPicksTheLargestRoundLengthThatFits() {
        XCTAssertEqual(ScenarioPlotGeometry.scaleBarLength(pointsPerMeter: 100, maxPoints: 160), 1)
        XCTAssertEqual(ScenarioPlotGeometry.scaleBarLength(pointsPerMeter: 300, maxPoints: 160), 0.5)
        XCTAssertEqual(ScenarioPlotGeometry.scaleBarLength(pointsPerMeter: 500, maxPoints: 160), 0.2)
        XCTAssertEqual(ScenarioPlotGeometry.scaleBarLength(pointsPerMeter: 10, maxPoints: 160), 10)
    }
}

final class TeamFilterPolishTests: XCTestCase {
    func testCaptionAddsThePinnedNoteOnlyWhenLateDronesAreListed() throws {
        let fixture = try loadReplay()
        let late = fixture.frames[3].presentation
        let rows = TeamPresentation.rows(groups: late.groups, drones: late.perDrone, query: "zzz")
        let counts = TeamPresentation.filterCounts(rows: rows, drones: late.perDrone, query: "zzz")
        XCTAssertEqual(TeamPresentation.filterCaption(counts: counts, rows: rows),
                       "Showing 3 of 50. Late and lost drones stay listed.")

        let waiting = fixture.frames[0].presentation
        let one = TeamPresentation.rows(groups: waiting.groups, drones: waiting.perDrone, query: "d07")
        let oneCounts = TeamPresentation.filterCounts(rows: one, drones: waiting.perDrone, query: "d07")
        XCTAssertEqual(TeamPresentation.filterCaption(counts: oneCounts, rows: one), "Showing 1 of 50")

        let none = TeamPresentation.rows(groups: waiting.groups, drones: waiting.perDrone, query: "")
        let noneCounts = TeamPresentation.filterCounts(rows: none, drones: waiting.perDrone, query: "")
        XCTAssertNil(TeamPresentation.filterCaption(counts: noneCounts, rows: none))
    }

    func testTheSelectedDroneStaysListedAndMarkedWhenTheFilterExcludesIt() throws {
        let waiting = try loadReplay().frames[0].presentation
        let rows = TeamPresentation.rows(groups: waiting.groups, drones: waiting.perDrone, query: "d08", keeping: "drone-007")
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0].members.map(\.droneID), ["drone-007", "drone-008"], "kept in its group order")
        XCTAssertEqual(rows[0].keptSelectionID, "drone-007")
        let counts = TeamPresentation.filterCounts(rows: rows, drones: waiting.perDrone, query: "d08")
        XCTAssertEqual(counts.shown, 2, "the caption counts every listed row, the kept selection included")
        XCTAssertEqual(counts.matched, 1, "but only real matches count as matched")

        // A selection that matches is not marked.
        let matching = TeamPresentation.rows(groups: waiting.groups, drones: waiting.perDrone, query: "d07", keeping: "drone-007")
        XCTAssertNil(matching[0].keptSelectionID)
    }

    func testGroupCellsDropUnitsBecauseTheHeadersCarryThem() {
        XCTAssertEqual(TeamColumns.age(.range(min: 0.15, max: 0.2)), "0.15\u{2013}0.20")
        XCTAssertEqual(TeamColumns.age(.value(0.1)), "0.10")
        XCTAssertEqual(TeamColumns.loss(.range(min: 25, max: 66.7)), "25\u{2013}67")
        XCTAssertNil(TeamColumns.age(nil as PresentationMetric?))
        XCTAssertEqual(Format.label("age_s"), "Age (s)")
        XCTAssertEqual(Format.label("loss_pct"), "Loss (%)")
    }
}

final class AuditFixTests: XCTestCase {
    func testEventsThatDifferOnlyInTheirDronesHaveDifferentIdentities() throws {
        func event(_ ids: [String]) throws -> PresentationEvent {
            let json: [String: Any] = ["drone_ids": ids, "level": "info", "text": "Target packet lost for 1 drones", "time_s": 0.1]
            return try JSONDecoder().decode(PresentationEvent.self, from: JSONSerialization.data(withJSONObject: json))
        }
        XCTAssertNotEqual(try event(["drone-001"]).id, try event(["drone-002"]).id)
        XCTAssertEqual(try event(["drone-001"]).id, try event(["drone-001"]).id)
    }

    func testCoordinatesBeyondTheBoundAreRejected() throws {
        func volume(_ center: [Double]) throws -> KeepOutVolume {
            let json: [String: Any] = ["center_m": center, "radius_m": 0.75, "label": "target_keepout"]
            return try JSONDecoder().decode(KeepOutVolume.self, from: JSONSerialization.data(withJSONObject: json))
        }
        XCTAssertNoThrow(try volume([1, 2, 1.5]))
        XCTAssertThrowsError(try volume([Double(fixtureCoordinateLimitM) * 2, 0, 0]))
        XCTAssertThrowsError(try volume([0, 0, 1e30]), "finite but huge must not reach the camera maths")
    }

    func testScenarioFramesMustAgreeWithTheirActiveCount() throws {
        let url = fixturesDirectory.appendingPathComponent("zephyr-s7-moving-scenario-run-1.json")
        var document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var frames = try XCTUnwrap(document["frames"] as? [[String: Any]])
        frames[0]["active_count"] = 999
        document["frames"] = frames
        let edited = FileManager.default.temporaryDirectory.appendingPathComponent("active-count-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: document).write(to: edited)
        defer { try? FileManager.default.removeItem(at: edited) }
        XCTAssertThrowsError(try ScenarioRunDocument.load(from: edited)) { error in
            XCTAssertTrue(error.localizedDescription.contains("active_count"), error.localizedDescription)
        }
        XCTAssertNoThrow(try ScenarioRunDocument.load(from: url), "the unedited fixture still loads")
    }

    @MainActor
    func testOpeningTheSameFileReusesItsDocumentAndClosingReleasesIt() async throws {
        let store = AppStore()
        let url = fixturesDirectory.appendingPathComponent("zephyr-s7-scenario-run-1.json")
        let firstID = await store.loadScenarioRun(url)
        let againID = await store.loadScenarioRun(url)
        let first = try XCTUnwrap(firstID)
        let again = try XCTUnwrap(againID)
        XCTAssertEqual(first, again, "the same file brings its window forward instead of a second copy")
        XCTAssertNotNil(store.document(first))
        store.close(first)
        XCTAssertNil(store.document(first), "closing the window releases the parsed document")
        let reopenedID = await store.loadScenarioRun(url)
        let reopened = try XCTUnwrap(reopenedID)
        XCTAssertNotEqual(reopened, first)
        store.close(reopened)
    }
}

final class RoundTwoTests: XCTestCase {
    func testADroneWithoutRecordedSpeedReadsNotRecorded() throws {
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: replayURL)) as? [String: Any])
        var frames = try XCTUnwrap(object["frames"] as? [[String: Any]])
        var presentation = try XCTUnwrap(frames[0]["presentation"] as? [String: Any])
        var perDrone = try XCTUnwrap(presentation["per_drone"] as? [[String: Any]])
        perDrone[0]["speed_mps"] = NSNull()
        perDrone[0]["heading_deg"] = NSNull()
        perDrone[0]["altitude_m"] = NSNull()
        presentation["per_drone"] = perDrone
        frames[0]["presentation"] = presentation
        object["frames"] = frames
        let fixture = try JSONDecoder().decode(Fixture.self, from: JSONSerialization.data(withJSONObject: object))
        let drone = try XCTUnwrap(fixture.frames[0].presentation.perDrone.first)
        XCTAssertNil(drone.speedMps)
        XCTAssertEqual(Format.metersPerSecond(drone.speedMps), "Not recorded", "a missing speed is never shown as 0")
        XCTAssertEqual(Format.degrees(drone.headingDeg), "Not recorded")
        XCTAssertNil(drone.altitudeM)
        XCTAssertEqual(Format.meters(drone.altitudeM), "Not recorded")
        // A recorded speed still reads normally.
        XCTAssertEqual(Format.metersPerSecond(fixture.frames[0].presentation.perDrone[1].speedMps), "0.00 m/s")
    }

    func testTheCommandWrapsOnlyAtSpaces() {
        let command = "PYTHONPATH=. ./.venv/bin/python tools/generate_s7_fixture.py --output runs/s7-swarm"
        let display = CommandText.display(command)
        XCTAssertEqual(display.replacingOccurrences(of: CommandText.joiner, with: ""), command, "only invisible joiners are added")
        XCTAssertEqual(display.split(separator: " ").count, command.split(separator: " ").count)
        // Every part keeps its slashes and hyphens joined to the next character.
        XCTAssertTrue(display.contains("s\u{2060}/\u{2060}g"))
        XCTAssertTrue(display.contains("-\u{2060}-\u{2060}o"))
    }

    func testGroupHeadersCountListedMembersWhileFiltering() {
        XCTAssertEqual(TeamColumns.countText(listed: 47, total: 47, noun: true), "47 drones")
        XCTAssertEqual(TeamColumns.countText(listed: 1, total: 47, noun: true), "1 of 47 drones")
        XCTAssertEqual(TeamColumns.countText(listed: 1, total: 47, noun: false), "1 of 47")
        XCTAssertEqual(TeamColumns.countText(listed: 1, total: 1, noun: true), "1 drone")
    }

    func testUpdateMessagesComeFromTheSharedText() {
        XCTAssertEqual(UpdateText.text("upToDate", ["tag": "v0.2.3"]), "Up to date: v0.2.3.")
        XCTAssertEqual(UpdateText.text("failed", ["reason": "offline"]), "Update check failed: offline")
        XCTAssertEqual(UpdateText.text("keptCurrent"), UIText.update["keptCurrent"])
    }

    func testOnlyMultiWordTextIsProse() {
        XCTAssertTrue(JSONValue.string("observed trace summary; not radio tolerance").isProse)
        XCTAssertFalse(JSONValue.string("fixture").isProse)
        XCTAssertFalse(JSONValue.number(0.012).isProse)
        XCTAssertFalse(JSONValue.bool(true).isProse)
    }
}

final class PolishFixTests: XCTestCase {
    private func editedScenario(_ edit: (inout [[String: Any]]) -> Void) throws -> URL {
        let url = fixturesDirectory.appendingPathComponent("zephyr-s7-moving-scenario-run-1.json")
        var document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var frames = try XCTUnwrap(document["frames"] as? [[String: Any]])
        edit(&frames)
        document["frames"] = frames
        let edited = FileManager.default.temporaryDirectory.appendingPathComponent("scenario-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: document).write(to: edited)
        return edited
    }

    func testAScenarioWithAHugeFiniteCoordinateIsRejected() throws {
        let url = try editedScenario { frames in
            var agents = frames[0]["agents"] as? [[String: Any]] ?? []
            agents[0]["position_m"] = [1e25, 0.0, 1.5]
            frames[0]["agents"] = agents
        }
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try ScenarioRunDocument.load(from: url)) { error in
            XCTAssertTrue(error.localizedDescription.contains("within"), error.localizedDescription)
        }
    }

    func testTheScaleBarAlwaysGrowsAndTheGridIsBounded() {
        // 160 points at 1e-20 points per metre is 1.6e22 m: the step must reach that size.
        let huge = ScenarioPlotGeometry.scaleBarLength(pointsPerMeter: 1e-20, maxPoints: 160)
        XCTAssertLessThanOrEqual(huge, 1.6e22)
        XCTAssertGreaterThan(huge, 1.6e22 / 5)
        XCTAssertEqual(ScenarioPlotGeometry.scaleBarLength(pointsPerMeter: 1e-3, maxPoints: 160), 100_000)
        XCTAssertNil(ScenarioPlotGeometry.gridLineRange(from: -1e25, to: 1e25, step: 1), "too many lines: skip the grid")
        XCTAssertNil(ScenarioPlotGeometry.gridLineRange(from: 0.1, to: 0.4, step: 0.5), "no line inside the span")
        XCTAssertEqual(ScenarioPlotGeometry.gridLineRange(from: -1.2, to: 1.2, step: 0.5), -2...2)
    }

    func testAWideSummaryListsAtMostTheLimitAndCountsTheRest() {
        var object: [String: JSONValue] = [:]
        for index in 0..<500 { object["extra_\(index)"] = .integer(Int64(index)) }
        for index in 0..<10 { object["block_\(index)"] = .object(["mean": .number(1), "count": .integer(3)]) }
        let plan = SummaryRows.plan(object)
        XCTAssertEqual(plan.scalars.count, JSONTree.limit)
        XCTAssertTrue(plan.groups.isEmpty, "the budget is spent on plain values")
        XCTAssertEqual(plan.hidden, 400 + 10 * 2, "400 values and 10 groups of 2 rows are counted, not drawn")
        let small = SummaryRows.plan(["a": .integer(1), "b": .object(["mean": .number(1)])])
        XCTAssertEqual(small, SummaryRows.Plan(scalars: ["a"], groups: [.init(key: "b", rows: 1, total: 1)], hidden: 0))
    }

    func testOneRowBudgetCoversManyGroupsOfManyValues() {
        // 100 groups x 100 values: 10,000 rows in the file, at most JSONTree.limit drawn in all.
        var object: [String: JSONValue] = [:]
        for group in 0..<100 {
            var values: [String: JSONValue] = [:]
            for value in 0..<100 { values["v\(value)"] = .string("text \(value)") }
            object["group_\(group)"] = .object(values)
        }
        let plan = SummaryRows.plan(object)
        XCTAssertLessThanOrEqual(plan.drawnRows, JSONTree.limit)
        let drawnValues = plan.groups.reduce(0) { $0 + $1.rows }
        let notShownInGroups = plan.groups.reduce(0) { $0 + $1.total - $1.rows }
        XCTAssertEqual(drawnValues + notShownInGroups + plan.hidden, 100 * 100, "every value is drawn or counted")
    }

    @MainActor
    func testTwoOpensOfTheSameFileAtOnceMakeOneDocument() async throws {
        let store = AppStore()
        let url = fixturesDirectory.appendingPathComponent("zephyr-s7-scenario-run-1.json")
        async let first = store.loadScenarioRun(url)
        async let second = store.loadScenarioRun(url)
        let ids = await [first, second].compactMap { $0 }
        XCTAssertEqual(store.documents.count, 1, "the second open waits on nothing and adds nothing")
        XCTAssertEqual(Set(ids).count, 1)
        ids.forEach(store.close)
    }

    func testScreenSizerAgreesWithSceneKitsOwnProjection() {
        // Independent of the sizer's formula: SceneKit's projection matrix gives the vertical
        // scale, 1 / tan(fov / 2), in m22.
        let camera = SCNCamera()
        camera.fieldOfView = 38
        camera.projectionDirection = .vertical
        camera.zNear = 0.02
        camera.zFar = 160
        let m22 = Float(camera.projectionTransform.m22)
        let viewHeight: Float = 400
        let distance: Float = 5
        let expected = 10 * (2 * distance / m22) / viewHeight
        let sizer = ScreenSizer(verticalFieldOfView: 38, camera: SCNNode())
        sizer.viewHeight = viewHeight
        XCTAssertEqual(sizer.worldSize(points: 10, distance: distance), expected, accuracy: 1e-5)
    }
}

final class RereviewFixTests: XCTestCase {
    func testALongGeneratedRunLoads() throws {
        // Made by: python3 tools/run_scenario.py with sim/scenario_example.json changed to
        // agent_count 1, steps 800, dt_s 1.0, target_velocity_mps [20, 0, 0]. The target ends at
        // x = 15,980 m, which the old 10 km bound wrongly rejected.
        let run = try ScenarioRunDocument.load(from: fixturesDirectory.appendingPathComponent("zephyr-s7-long-run-800.json"))
        XCTAssertEqual(run.frames.count, 800)
        XCTAssertEqual(run.frames.last?.target[0] ?? 0, 15_980, accuracy: 1e-6)
        let geometry = ScenarioPlotGeometry.make(frame: run.frames[799])
        let projected = ScenarioPlotGeometry.project(geometry.target, bounds: geometry.bounds, width: 640, height: 340)
        XCTAssertTrue(projected.x.isFinite && projected.y.isFinite)
    }

    func testAMissingAgentVectorIsReportedAsAnInvalidAgent() throws {
        let url = fixturesDirectory.appendingPathComponent("zephyr-s7-moving-scenario-run-1.json")
        var document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var frames = try XCTUnwrap(document["frames"] as? [[String: Any]])
        var agents = try XCTUnwrap(frames[0]["agents"] as? [[String: Any]])
        agents[0].removeValue(forKey: "position_m")
        frames[0]["agents"] = agents
        document["frames"] = frames
        let edited = FileManager.default.temporaryDirectory.appendingPathComponent("missing-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: document).write(to: edited)
        defer { try? FileManager.default.removeItem(at: edited) }
        XCTAssertThrowsError(try ScenarioRunDocument.load(from: edited)) { error in
            XCTAssertEqual(error.localizedDescription, "agent is invalid")
        }
    }

    func testGridIndexesNeverReachTheIntConversionWhenHuge() {
        XCTAssertNil(ScenarioPlotGeometry.gridLineRange(from: 1e20, to: 1e20 + 50, step: 100), "index far beyond 1e15")
        XCTAssertNil(ScenarioPlotGeometry.gridLineRange(from: -1e300, to: -1e300, step: 1e-3))
        XCTAssertNotNil(ScenarioPlotGeometry.gridLineRange(from: 999_000, to: 1_000_000, step: 500))
    }

    @MainActor
    func testAScenarioFileOpenedAsEvidenceIsCheckedAsEvidence() async throws {
        let store = AppStore(presentsAlerts: false)
        let url = fixturesDirectory.appendingPathComponent("zephyr-s7-scenario-run-1.json")
        let scenarioID = await store.loadScenarioRun(url)
        let scenario = try XCTUnwrap(scenarioID)
        let asEvidence = await store.loadEvidenceReport(url)
        XCTAssertNil(asEvidence, "validated as an evidence report, which it is not")
        XCTAssertNotNil(store.openError, "and it fails with a clear error")
        XCTAssertEqual(store.documents.count, 1, "the scenario is never shown in an evidence window")
        if case .scenarioRun? = store.document(scenario) {} else { XCTFail("the scenario document is untouched") }
        store.close(scenario)
    }

    func testHeadingFollowsTheSharedRule() {
        XCTAssertEqual(Format.heading(90, speed: 2), "90\u{00B0}")
        XCTAssertEqual(Format.heading(nil, speed: 0), "At rest", "a recorded speed too small for a heading")
        XCTAssertEqual(Format.heading(nil, speed: nil), "Not recorded", "no velocity recorded")
        XCTAssertEqual(Format.heading(nil, speed: 0.5), "Straight up or down", "moving with no horizontal motion")
    }

    func testMovedStringsComeFromTheSharedText() {
        XCTAssertEqual(UIText.fieldLabels["events_pending"], "Later events appear here as the replay plays.")
        XCTAssertEqual(UIText.fieldLabels["file_not_open"], "This file is no longer open")
        XCTAssertEqual(UIText.fieldLabels["file_not_open_hint"], "Open it again from the File menu.")
        XCTAssertEqual(UIText.fieldLabels["at_rest"], "At rest")
    }
}
