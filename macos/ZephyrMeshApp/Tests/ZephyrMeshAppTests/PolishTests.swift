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
        XCTAssertEqual(counts.shown, 1, "the kept selection is not counted as a filter match")

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
