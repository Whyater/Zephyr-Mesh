import CoreGraphics
import SwiftUI
import XCTest
@testable import ZephyrMeshApp

private let resultsURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Sources/ZephyrMeshApp/Resources/results_fixture.json")

private func fixtureObject() throws -> [String: Any] {
    try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: resultsURL)) as? [String: Any])
}

private func decode(_ object: [String: Any]) throws -> ResultsFixture {
    try ResultsFixture.decode(JSONSerialization.data(withJSONObject: object))
}

private func loadData() throws -> ResultsData {
    try ResultsData(fixture: ResultsFixture.decode(Data(contentsOf: resultsURL)))
}

/// Edits one atlas entry of the fixture and decodes the result.
private func decodeWithEntry(_ edit: (inout [String: Any]) -> Void) throws -> ResultsFixture {
    var object = try fixtureObject()
    var atlas = try XCTUnwrap(object["atlas"] as? [String: Any])
    var entries = try XCTUnwrap(atlas["entries"] as? [[String: Any]])
    edit(&entries[0])
    atlas["entries"] = entries
    object["atlas"] = atlas
    return try decode(object)
}

private func entryJSON(_ json: String) throws -> AtlasEntry {
    try JSONDecoder().decode(AtlasEntry.self, from: Data(json.utf8))
}

private func entry(kind: String, value: String = "null", censored: String = "null") throws -> AtlasEntry {
    try entryJSON("""
    {"arm":"ct","censored":\(censored),"failure_mode":null,"improvement_vs_naive":[],"interval_s":[0.1,0.2],"kind":"\(kind)",
     "loss_condition":"independent:0","loss_model":"independent","loss_probability":0.0,"stratum":"central",
     "value_s":\(value),"variant":"cv_1"}
    """)
}

// MARK: - Fixture decoding

final class ResultsFixtureDecodingTests: XCTestCase {
    func testBundledFixtureDecodesWithExpectedShape() throws {
        let data = try loadData()
        XCTAssertEqual(data.fixture.schema, ResultsFixture.schemaName)
        XCTAssertTrue(data.fixture.simulated)
        XCTAssertEqual(data.s6.delaysS, [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3])
        // 2 paths x (4 random + 3 bursty loss levels) x 2 trackers.
        XCTAssertEqual(data.s6.series.count, 28)
        XCTAssertEqual(data.atlas.strata.first?.id, "central")
        XCTAssertEqual(data.atlas.nSeeds, 20)
        XCTAssertFalse(data.s6.limits.isEmpty)
        XCTAssertFalse(data.atlas.findings.isEmpty)
    }

    func testEveryStratumCellHasAnEntryForEveryArm() throws {
        let data = try loadData()
        for stratum in data.atlas.strata {
            for variant in stratum.variants {
                for loss in stratum.lossConditions {
                    XCTAssertNotNil(data.cell(stratum: stratum.id, variant: variant, loss: loss), "\(stratum.id) \(variant) \(loss)")
                    for arm in stratum.arms {
                        XCTAssertNotNil(data.entry(stratum: stratum.id, variant: variant, loss: loss, arm: arm))
                    }
                }
            }
        }
    }

    func testUnknownTopLevelKeyFails() throws {
        var object = try fixtureObject()
        object["extra"] = 1
        XCTAssertThrowsError(try decode(object)) { error in
            XCTAssertTrue(error.localizedDescription.contains("unknown extra"), error.localizedDescription)
        }
    }

    func testMissingNestedKeyFailsAndNamesIt() throws {
        XCTAssertThrowsError(try decodeWithEntry { $0.removeValue(forKey: "failure_mode") }) { error in
            XCTAssertTrue(error.localizedDescription.contains("missing failure_mode"), error.localizedDescription)
        }
    }

    func testWrongSchemaAndUnsimulatedAreRejected() throws {
        var object = try fixtureObject()
        object["schema"] = "zephyr-results-fixture-2"
        XCTAssertThrowsError(try decode(object))
        object = try fixtureObject()
        object["simulated"] = false
        XCTAssertThrowsError(try decode(object))
    }

    func testIntervalEndsDecodeNumbersAndBothMarkers() throws {
        let ends = try JSONDecoder().decode([IntervalEnd].self, from: Data(#"["<0", 0.15, ">0.3", 0]"#.utf8))
        XCTAssertEqual(ends, [.belowZero, .seconds(0.15), .aboveMax, .seconds(0)])
    }

    func testUnknownIntervalMarkerIsRejected() {
        XCTAssertThrowsError(try JSONDecoder().decode([IntervalEnd].self, from: Data(#"[">0.5"]"#.utf8)))
        XCTAssertThrowsError(try JSONDecoder().decode([IntervalEnd].self, from: Data("[true]".utf8)))
    }

    func testEntryKindMustAgreeWithItsValues() throws {
        XCTAssertNoThrow(try entry(kind: "value", value: "0.2"))
        XCTAssertThrowsError(try entry(kind: "value"), "a value entry needs value_s")
        XCTAssertThrowsError(try entry(kind: "censored"), "a censored entry needs the >0.3 marker")
        XCTAssertThrowsError(try entry(kind: "fails_at_zero", value: "0.1"))
        XCTAssertThrowsError(try entry(kind: "unknown"))
    }

    func testSeriesWithAShortArrayFails() throws {
        var object = try fixtureObject()
        var s6 = try XCTUnwrap(object["s6"] as? [String: Any])
        var series = try XCTUnwrap(s6["series"] as? [[String: Any]])
        series[0]["p95_median_m"] = [0.1, 0.2]
        s6["series"] = series
        object["s6"] = s6
        XCTAssertThrowsError(try decode(object)) { error in
            XCTAssertTrue(error.localizedDescription.contains("p95_median_m"), error.localizedDescription)
        }
    }

    func testBundledLoaderFindsTheResource() {
        // swift test puts the resource bundle beside the test bundle; the loader must find it.
        if case .failure(let error) = ResultsLoader.loadBundled() { XCTFail(error.localizedDescription) }
    }
}

// MARK: - Formatting

final class ResultsFormatTests: XCTestCase {
    func testMetresUseThreeDecimalsAndNoNegativeZero() {
        XCTAssertEqual(ResultsFormat.meters(0.15838), "0.158 m")
        XCTAssertEqual(ResultsFormat.meters(-0.0001), "0.000 m")
        XCTAssertEqual(ResultsFormat.meters(-0.0), "0.000 m")
        XCTAssertEqual(ResultsFormat.metersValue(-0.0249), "-0.025")
    }

    func testFailureLineLabelUsesThreeDecimals() throws {
        XCTAssertEqual(ResultsFormat.failureLineLabel(0.25), "Failure line 0.250 m")
        let data = try loadData()
        XCTAssertEqual(ResultsFormat.failureLineLabel(data.s6.failureP95M), "Failure line 0.250 m")
    }

    func testDelaysRoundHalfToEvenLikePython() {
        XCTAssertEqual(ResultsFormat.delay(0), "0 ms")
        XCTAssertEqual(ResultsFormat.delay(0.025), "25 ms")
        XCTAssertEqual(ResultsFormat.delay(0.3), "300 ms")
        // Python round(0.5) == 0 and round(2.5) == 2.
        XCTAssertEqual(ResultsFormat.milliseconds(0.0005), 0)
        XCTAssertEqual(ResultsFormat.milliseconds(0.0025), 2)
    }

    func testRangesCountsAndPanelTitles() {
        XCTAssertEqual(ResultsFormat.meterRange(low: 0.149, high: 0.1704), "0.149 to 0.170 m")
        XCTAssertEqual(ResultsFormat.count(0, of: 50), "0 of 50")
        XCTAssertEqual(ResultsFormat.panelTitle(probability: 0), "No loss")
        XCTAssertEqual(ResultsFormat.panelTitle(probability: 0.3), "30 % loss", "int(0.3 * 100) is 30")
        XCTAssertEqual(ResultsFormat.levelLabel(0), "None")
        XCTAssertEqual(ResultsFormat.levelLabel(0.1), "10 %")
    }

    func testLossKeysAndTitlesMatchTheFixture() throws {
        let titles = try loadData().atlas.lossTitles
        XCTAssertEqual(ResultsFormat.lossKey(model: "independent", probability: 0), "independent:0")
        XCTAssertEqual(ResultsFormat.lossKey(model: "burst", probability: 0.3), "burst:0.3")
        XCTAssertEqual(ResultsFormat.lossTitle(model: "independent", probability: 0, titles: titles), "No loss")
        XCTAssertEqual(ResultsFormat.lossTitle(model: "burst", probability: 0.5, titles: titles), "50 % bursty")
        XCTAssertEqual(ResultsFormat.lossTitle(model: "burst", probability: 0.9, titles: titles), "Not computed")
    }

    func testAtlasIntervalsGiveEveryEndItsUnit() {
        XCTAssertEqual(ResultsFormat.interval([.seconds(0.1), .seconds(0.2)]), "100 ms to 200 ms")
        XCTAssertEqual(ResultsFormat.interval([.belowZero, .seconds(0.2)]), "Below 0 ms to 200 ms")
        XCTAssertEqual(ResultsFormat.interval([.seconds(0.1), .aboveMax]), "100 ms to above 300 ms")
        XCTAssertEqual(ResultsFormat.interval([.belowZero, .aboveMax]), "Below 0 ms to above 300 ms")
        XCTAssertEqual(ResultsFormat.interval([.aboveMax, .aboveMax]), "Above 300 ms")
        XCTAssertEqual(ResultsFormat.interval([.belowZero, .belowZero]), "Below 0 ms")
        XCTAssertEqual(ResultsFormat.interval([.seconds(0), .seconds(0)]), "0 ms")
        XCTAssertEqual(ResultsFormat.interval([.aboveMax]), "Not computed")
    }

    func testFindingsDropBackticksForDisplayOnly() throws {
        let data = try loadData()
        let verbatim = data.s6.findings[0].text
        XCTAssertTrue(verbatim.contains("(`runs/s6-recovery`)"), "the fixture keeps the write-up text verbatim")
        let shown = ResultsFormat.displayText(verbatim)
        XCTAssertTrue(shown.hasSuffix("(runs/s6-recovery)."), shown)
        XCTAssertFalse(shown.contains("`"))
    }

    func testDelayViewLeadsWithTheRecoveryFinding() throws {
        let findings = try loadData().s6.findings
        let split = ResultsFormat.splitFindings(findings, lead: 1)
        XCTAssertTrue(split.lead?.text.hasPrefix("Stamping each measurement") ?? false)
        XCTAssertEqual(split.more, [findings[0], findings[2]], "the other two keep their order in the drawer")
        let atlas = try loadData().atlas.findings
        XCTAssertEqual(ResultsFormat.splitFindings(atlas, lead: 0).more, [], "the atlas has a single finding and no drawer")
        XCTAssertEqual(ResultsFormat.splitFindings(atlas, lead: 5).lead, atlas[0], "an out-of-range lead falls back to the first")
        XCTAssertNil(ResultsFormat.splitFindings([], lead: 1).lead)
    }

    func testCellTextsAndCauses() throws {
        XCTAssertEqual(ResultsFormat.cellText(try entry(kind: "censored", censored: #"">0.3""#)), ">300 ms")
        XCTAssertEqual(ResultsFormat.cellText(try entry(kind: "value", value: "0.2")), "200 ms")
        XCTAssertEqual(ResultsFormat.cellText(try entry(kind: "value", value: "0")), "0 ms", "a tolerated 0 ms is a value, not missing")
        XCTAssertEqual(ResultsFormat.cellText(try entry(kind: "fails_at_zero")), "Fails at 0")
        XCTAssertEqual(ResultsFormat.mostDelayText(try entry(kind: "fails_at_zero")), "Fails at 0 ms")
        XCTAssertEqual(ResultsFormat.cellText(nil), "Not computed")
        XCTAssertEqual(ResultsFormat.cause(nil), "No failure seen")
        XCTAssertEqual(ResultsFormat.cause("gap-dominated"), "Lost packets")
        XCTAssertEqual(ResultsFormat.cause("model-mismatch"), "Path model")
    }

    func testClassificationListsShortTiesAndCountsLongOnes() throws {
        let data = try loadData()
        let labels = data.atlas.armLabels
        let cells = data.atlas.cells
        let winner = try XCTUnwrap(cells.first { $0.kind == .winner })
        XCTAssertEqual(ResultsFormat.classification(winner, labels: labels), labels[winner.arms[0]])
        let longTie = try XCTUnwrap(cells.first { $0.kind == .tie && $0.arms.count > 3 })
        XCTAssertEqual(ResultsFormat.classification(longTie, labels: labels), "Tie: \(longTie.arms.count) trackers")
        let shortTie = try XCTUnwrap(cells.first { $0.kind == .tie && $0.arms.count == 2 })
        let first = try XCTUnwrap(labels[shortTie.arms[0]])
        let second = try XCTUnwrap(labels[shortTie.arms[1]])
        XCTAssertEqual(ResultsFormat.classification(shortTie, labels: labels), "Tie: \(first), \(second)")
        let none = try XCTUnwrap(cells.first { $0.kind == .none })
        XCTAssertEqual(ResultsFormat.classification(none, labels: labels), "Nothing passes")
    }

    func testScaleCaptionNamesTheSharedAxisTop() throws {
        XCTAssertEqual(ResultsFormat.scaleCaption(top: 0.7), "Error axis 0 to 0.700 m for this path and loss type")
        XCTAssertEqual(ResultsFormat.scaleCaption(top: 1.5), "Error axis 0 to 1.500 m for this path and loss type")
        let data = try loadData()
        let top = data.errorScale(motion: "constant_turn", lossModel: "independent").top
        XCTAssertTrue(ResultsFormat.scaleCaption(top: top).contains("0 to 0.700 m"))
    }

    func testIntervalCaptionExplainsAFailsAtZeroCellWhoseIntervalReachesAboveZero() throws {
        let data = try loadData()
        let seeds = data.atlas.nSeeds
        let reaching = try XCTUnwrap(data.atlas.entries.first {
            $0.kind == .failsAtZero && $0.intervalS.count == 2 && $0.intervalS[1] == .seconds(0.2)
        })
        XCTAssertEqual(ResultsFormat.intervalCaption(reaching, seeds: seeds),
                       "Fails at 0 ms because at least half its seeds fail, but with \(seeds) seeds per cell "
                       + "the limit could be as high as 200 ms.")
        let closed = try XCTUnwrap(data.atlas.entries.first {
            $0.kind == .failsAtZero && $0.intervalS.count == 2 && $0.intervalS[1] == .belowZero
        })
        XCTAssertEqual(ResultsFormat.intervalCaption(closed, seeds: seeds), UIText.Atlas.intervalCaption, "<0 to <0 keeps the general note")
        let plain = try XCTUnwrap(data.atlas.entries.first { $0.kind == .value })
        XCTAssertEqual(ResultsFormat.intervalCaption(plain, seeds: seeds), UIText.Atlas.intervalCaption)
        XCTAssertEqual(ResultsFormat.intervalCaption(nil, seeds: seeds), UIText.Atlas.intervalCaption)
    }

    func testCellSubtitleKeepsTheConditionsForTheirOwnLine() {
        XCTAssertEqual(ResultsFormat.cellSubtitle(path: "Figure 8", loss: "30 % bursty"), "Figure 8, 30 % bursty")
    }

    func testNaiveTrackerNameComesFromTheUIText() throws {
        XCTAssertEqual(ResultsFormat.armName("naive"), "Naive tracker")
        XCTAssertFalse(UIText.Results.armNaive.contains("Basic"))
    }

    func testComparisonSignAndTieWording() throws {
        let data = try loadData()
        // Straight line, no loss: the paired difference is naive minus recovery, so a positive
        // median at 300 ms means recovery had the lower error.
        let series = try XCTUnwrap(data.series(motion: "constant_velocity", lossModel: "independent", probability: 0, arm: "recovery"))
        let lines = ResultsFormat.comparison(series: series, index: 6, delay: 0.3)
        XCTAssertTrue(lines[0].hasPrefix("Recovery is lower by 0.201 m"), lines[0])
        XCTAssertEqual(lines.count, 3)
        XCTAssertEqual(ResultsFormat.comparison(series: series, index: 0, delay: 0),
                       ["Both trackers are identical here (no delay, no loss)"])
        let lossy = try XCTUnwrap(data.series(motion: "constant_velocity", lossModel: "independent", probability: 0.3, arm: "naive"))
        XCTAssertEqual(ResultsFormat.comparison(series: lossy, index: 0, delay: 0), ["Tied in every seed"])
    }

    func testHoverTextReadsFromTheSeries() throws {
        let data = try loadData()
        let series = try XCTUnwrap(data.series(motion: "constant_velocity", lossModel: "independent", probability: 0, arm: "naive"))
        XCTAssertEqual(ResultsFormat.hoverText(series: series, index: 6, delays: data.s6.delaysS),
                       "Naive tracker, 300 ms: 0.359 m (0.355 to 0.363), 50 of 50 fail")
    }
}

// MARK: - Geometry

final class ResultsGeometryTests: XCTestCase {
    func testNiceScaleCeilings() {
        let fixtureScale = NiceScale(maximum: 1.483924471944304)
        XCTAssertEqual(fixtureScale.top, 1.5, accuracy: 1e-12)
        XCTAssertEqual(fixtureScale.step, 0.5, accuracy: 1e-12)
        XCTAssertEqual(fixtureScale.ticks.map(fixtureScale.label), ["0.0", "0.5", "1.0", "1.5"])
        XCTAssertEqual(NiceScale(maximum: 0.38).top, 0.4, accuracy: 1e-12)
        XCTAssertEqual(NiceScale(maximum: 1).top, 1, accuracy: 1e-12, "an exact step is its own ceiling")
        XCTAssertEqual(NiceScale(maximum: 0.9).step, 0.25, accuracy: 1e-12)
        XCTAssertEqual(NiceScale(maximum: 0).top, 1, "an empty axis still has a range")
    }

    func testNiceScaleMatchesTheExpectedCeilingsAndReadableSteps() {
        let cases: [(Double, Double, Double)] = [(0.389, 0.4, 0.1), (0.446, 0.5, 0.1), (0.606, 0.7, 0.1), (1.484, 1.5, 0.5)]
        for (maximum, top, step) in cases {
            let scale = NiceScale(maximum: maximum)
            XCTAssertEqual(scale.top, top, accuracy: 1e-12, "\(maximum)")
            XCTAssertEqual(scale.step, step, accuracy: 1e-12, "\(maximum)")
            XCTAssertTrue((3...7).contains(scale.ticks.count - 1), "\(maximum) has \(scale.ticks.count - 1) intervals")
        }
        XCTAssertEqual(NiceScale(maximum: 0.446).ticks.map(NiceScale(maximum: 0.446).label), ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    }

    /// The intended behaviour (SPEC section 9): the four panels share one error axis chosen per
    /// path and loss type, not one global axis, so the straight-line curves are not squeezed into the
    /// bottom of a 0 to 1.5 m axis. The caption beside the controls says where each axis ends.
    func testEachPathAndLossTypeGetsItsOwnScaleFromTheFixture() throws {
        let data = try loadData()
        let expected: [(String, String, Double)] = [("constant_velocity", "independent", 0.4), ("constant_velocity", "burst", 0.5),
                                                    ("constant_turn", "independent", 0.7), ("constant_turn", "burst", 1.5)]
        for (motion, loss, top) in expected {
            let panels = DelayLossLayout.panels(lossModel: loss, levels: data.s6.lossLevels)
            let shown = panels.flatMap { panel in
                DelayLossLayout.arms.compactMap {
                    data.series(motion: motion, lossModel: panel.lossModel, probability: panel.probability, arm: $0)
                }
            }
            XCTAssertEqual(shown.count, 8, "four panels, two trackers each")
            let maximum = try XCTUnwrap(shown.flatMap(\.p95CIHighM).max())
            let scale = data.errorScale(motion: motion, lossModel: loss)
            XCTAssertEqual(scale.top, top, accuracy: 1e-12, "\(motion) \(loss)")
            XCTAssertGreaterThanOrEqual(scale.top, maximum, "every interval stays on the plot")
        }
    }

    func testChartScaleMapsCornersAndClampsAboveTheTop() {
        let scale = ChartScale(plot: CGRect(x: 10, y: 20, width: 300, height: 150), xMax: 0.3, yMax: 1.5)
        XCTAssertEqual(scale.point(0, 0), CGPoint(x: 10, y: 170))
        XCTAssertEqual(scale.point(0.3, 1.5), CGPoint(x: 310, y: 20))
        XCTAssertEqual(scale.y(3), 20, "values above the top stay on the plot edge")
        XCTAssertEqual(scale.x(0.15), 160, accuracy: 1e-9)
    }

    func testBandPolygonGoesOutAlongHighsAndBackAlongLows() {
        let scale = ChartScale(plot: CGRect(x: 0, y: 0, width: 100, height: 100), xMax: 1, yMax: 1)
        let band = ChartGeometry.bandPolygon(delays: [0, 1], lows: [0.1, 0.2], highs: [0.3, 0.4], scale: scale)
        XCTAssertEqual(band.count, 4)
        XCTAssertEqual(band[0].y, 70, accuracy: 1e-9)
        XCTAssertEqual(band[1].y, 60, accuracy: 1e-9)
        XCTAssertEqual(band[2], CGPoint(x: 100, y: 80))
        XCTAssertEqual(band[3].y, 90, accuracy: 1e-9)
        XCTAssertTrue(ChartGeometry.bandPolygon(delays: [], lows: [], highs: [], scale: scale).isEmpty)
    }

    func testNearestPointRespectsTheHitRadius() {
        let points = [CGPoint(x: 0, y: 0), CGPoint(x: 20, y: 0), CGPoint(x: 40, y: 0)]
        XCTAssertEqual(ChartGeometry.nearest(to: CGPoint(x: 18, y: 3), in: points, radius: 10), 1)
        XCTAssertNil(ChartGeometry.nearest(to: CGPoint(x: 30, y: 15), in: points, radius: 10))
        XCTAssertEqual(ChartGeometry.nearest(to: CGPoint(x: 10, y: 0), in: points, radius: 10), 0, "ties go to the earlier point")
    }

    func testPanelHitPicksTheCloserTracker() throws {
        let data = try loadData()
        let recovery = data.series(motion: "constant_velocity", lossModel: "independent", probability: 0, arm: "recovery")
        let naive = data.series(motion: "constant_velocity", lossModel: "independent", probability: 0, arm: "naive")
        let series = PanelSeries(delays: data.s6.delaysS, recovery: recovery, naive: naive)
        let scale = ChartScale(plot: CGRect(x: 0, y: 0, width: 300, height: 300), xMax: 0.3, yMax: 1.5)
        let naiveTop = series.points(arm: 1, scale: scale)[6]
        let hit = try XCTUnwrap(series.hit(CGPoint(x: naiveTop.x - 2, y: naiveTop.y + 2), scale: scale))
        XCTAssertEqual(hit.arm, 1)
        XCTAssertEqual(hit.index, 6)
        XCTAssertNil(series.hit(CGPoint(x: 150, y: 10), scale: scale))
    }

    func testChartSelectionStopsAtEveryEdge() {
        let start = ChartSelection(panel: 0, arm: 0, delayIndex: 0)
        XCTAssertNil(start.moved(.left, panels: 4, arms: 2, delays: 7))
        XCTAssertNil(start.moved(.up, panels: 4, arms: 2, delays: 7))
        XCTAssertNil(start.moved(.previousPanel, panels: 4, arms: 2, delays: 7))
        XCTAssertEqual(start.moved(.right, panels: 4, arms: 2, delays: 7), ChartSelection(panel: 0, arm: 0, delayIndex: 1))
        XCTAssertEqual(start.moved(.down, panels: 4, arms: 2, delays: 7), ChartSelection(panel: 0, arm: 1, delayIndex: 0))
        let end = ChartSelection(panel: 3, arm: 1, delayIndex: 6)
        XCTAssertNil(end.moved(.right, panels: 4, arms: 2, delays: 7))
        XCTAssertNil(end.moved(.down, panels: 4, arms: 2, delays: 7))
        XCTAssertNil(end.moved(.nextPanel, panels: 4, arms: 2, delays: 7))
    }

    func testAtlasSelectionClampsInsideTheGrid() {
        let corner = AtlasSelection(row: 0, column: 0)
        XCTAssertEqual(corner.moved(.up, rows: 9, columns: 5), corner)
        XCTAssertEqual(corner.moved(.left, rows: 9, columns: 5), corner)
        XCTAssertEqual(corner.moved(.down, rows: 9, columns: 5), AtlasSelection(row: 1, column: 0))
        let far = AtlasSelection(row: 8, column: 4)
        XCTAssertEqual(far.moved(.down, rows: 9, columns: 5), far)
        XCTAssertEqual(far.moved(.right, rows: 9, columns: 5), far)
        XCTAssertEqual(far.clamped(rows: 7, columns: 3), AtlasSelection(row: 6, column: 2), "a smaller stratum keeps the nearest cell")
    }

    func testAtlasBinsEachKind() throws {
        XCTAssertEqual(AtlasBin.stop(for: try entry(kind: "censored", censored: #"">0.3""#)), 4)
        XCTAssertNil(AtlasBin.stop(for: try entry(kind: "fails_at_zero")), "fails at zero has no tint")
        XCTAssertNil(AtlasBin.stop(for: nil))
        let stops = try [0, 0.025, 0.05, 0.1, 0.15, 0.2].map { try AtlasBin.stop(for: entry(kind: "value", value: "\($0)")) }
        XCTAssertEqual(stops, [1, 1, 2, 2, 3, 3])
    }

    func testSegmentRectangleIntersection() {
        let rect = CGRect(x: 10, y: 10, width: 10, height: 10)
        XCTAssertTrue(ChartGeometry.segment(CGPoint(x: 0, y: 15), CGPoint(x: 30, y: 15), intersects: rect), "passes through")
        XCTAssertTrue(ChartGeometry.segment(CGPoint(x: 12, y: 12), CGPoint(x: 14, y: 14), intersects: rect), "inside")
        XCTAssertFalse(ChartGeometry.segment(CGPoint(x: 0, y: 0), CGPoint(x: 30, y: 0), intersects: rect), "above")
        XCTAssertFalse(ChartGeometry.segment(CGPoint(x: 0, y: 30), CGPoint(x: 5, y: 0), intersects: rect), "beside")
    }

    func testAtlasCellsShrinkToFitThenStopAtTheMinimum() {
        // Central at the default window: 5 trackers and a 225 pt Most delay column in 900 pt.
        let fit = AtlasGeometry.cellWidth(available: 900, arms: 5, mostDelayWidth: 225)
        XCTAssertGreaterThanOrEqual(fit, LayoutMetrics.atlasCellMinWidth)
        XCTAssertLessThanOrEqual(fit, LayoutMetrics.atlasCellWidth)
        let used = LayoutMetrics.atlasRowHeaderWidth + 5 * fit + AtlasGeometry.columnSpacing * 6 + AtlasGeometry.mostDelayLeading + 225
            + AtlasGeometry.outerPadding * 2
        XCTAssertLessThanOrEqual(used, 900, "the whole matrix fits without scrolling")
        XCTAssertEqual(AtlasGeometry.cellWidth(available: 2000, arms: 5, mostDelayWidth: 225), LayoutMetrics.atlasCellWidth)
        XCTAssertEqual(AtlasGeometry.cellWidth(available: 700, arms: 8, mostDelayWidth: 225), LayoutMetrics.atlasCellMinWidth,
                       "too narrow: cells stop at the minimum and the matrix scrolls")
    }

    func testFailureCaptionTakesTheFirstFreeSpotInOrder() {
        let plot = CGRect(x: 0, y: 0, width: 300, height: 200)
        let size = CGSize(width: 100, height: 14)
        func origin(_ segments: [(CGPoint, CGPoint)], lineY: CGFloat = 100) -> CGPoint {
            ChartGeometry.captionOrigin(size: size, lineY: lineY, plot: plot, gap: 4, clearance: 3, segments: segments)
        }
        func bar(_ x0: CGFloat, _ x1: CGFloat, _ y: CGFloat) -> (CGPoint, CGPoint) { (CGPoint(x: x0, y: y), CGPoint(x: x1, y: y)) }
        let rightAbove = bar(200, 300, 90), rightBelow = bar(200, 300, 110), leftAbove = bar(0, 100, 90), leftBelow = bar(0, 100, 110)
        XCTAssertEqual(origin([]), CGPoint(x: 196, y: 82), "right end above the line when nothing is there")
        XCTAssertEqual(origin([bar(0, 100, 90)]), CGPoint(x: 196, y: 82), "data at the left end does not move it")
        XCTAssertEqual(origin([rightAbove]), CGPoint(x: 196, y: 104), "right below when a curve runs through right above")
        XCTAssertEqual(origin([rightAbove, rightBelow]), CGPoint(x: 4, y: 82), "left above when both right spots collide")
        XCTAssertEqual(origin([rightAbove, rightBelow, leftAbove]), CGPoint(x: 4, y: 104), "left below is the last free spot")
        XCTAssertEqual(origin([], lineY: 10), CGPoint(x: 196, y: 14), "the panel top rules out the spots above")
        XCTAssertEqual(origin([], lineY: 195), CGPoint(x: 196, y: 177), "near the bottom edge it stays above")
        XCTAssertEqual(origin([bar(200, 300, 185)], lineY: 195), CGPoint(x: 4, y: 177),
                       "below is off the plot, so a blocked right-above leaves left above")
        XCTAssertEqual(origin([bar(200, 300, 20)], lineY: 10), CGPoint(x: 4, y: 14),
                       "above is off the plot and right below is blocked: left below")
        XCTAssertEqual(origin([rightAbove, rightBelow]), origin([rightAbove, rightBelow]), "same input, same place")

        // All four collide: the least overlap wins, and the earlier spot wins a tie.
        let all = [rightAbove, rightBelow, leftAbove, leftBelow]
        XCTAssertEqual(origin(all), CGPoint(x: 196, y: 82), "equal overlap everywhere: right above, the first in order")
        let heavyRightAbove = [rightAbove, bar(200, 300, 86), bar(200, 300, 94)]
        XCTAssertEqual(origin(heavyRightAbove + [rightBelow, leftAbove, leftBelow]), CGPoint(x: 196, y: 104),
                       "right below touches the fewest segments")
        XCTAssertEqual(origin(heavyRightAbove + [rightBelow, bar(200, 300, 114), leftAbove, bar(0, 100, 86), bar(0, 100, 94),
                                                 leftBelow]), CGPoint(x: 4, y: 104), "left below has one hit, the others have more")
    }

    func testNearestPointNeverNeedsAForceUnwrapAndKeepsTheEarlierTie() {
        let points = [CGPoint(x: 0, y: 0), CGPoint(x: 20, y: 0), CGPoint(x: 10, y: 10)]
        XCTAssertNil(ChartGeometry.nearest(to: CGPoint(x: 100, y: 100), in: points, radius: 5))
        XCTAssertNil(ChartGeometry.nearest(to: .zero, in: [], radius: 5))
        XCTAssertEqual(ChartGeometry.nearest(to: CGPoint(x: 10, y: 0), in: Array(points.prefix(2)), radius: 10), 0)
        XCTAssertEqual(ChartGeometry.nearest(to: CGPoint(x: 19, y: 1), in: points, radius: 10), 1)
    }
}

// MARK: - Layout, model and screenshot setup

@MainActor
final class ResultsModelTests: XCTestCase {
    func testBurstyPanelsStartWithTheSharedNoLossRun() throws {
        let levels = try loadData().s6.lossLevels
        let random = DelayLossLayout.panels(lossModel: "independent", levels: levels)
        let bursty = DelayLossLayout.panels(lossModel: "burst", levels: levels)
        XCTAssertEqual(random.map(\.probability), [0, 0.1, 0.3, 0.5])
        XCTAssertEqual(bursty.map(\.lossModel), ["independent", "burst", "burst", "burst"])
        XCTAssertEqual(bursty.map(\.title), ["No loss", "10 % loss", "30 % loss", "50 % loss"])
    }

    func testAtlasNoneLevelMapsToTheNoLossRunForBothTypes() throws {
        XCTAssertEqual(AtlasLayout.lossCondition(lossModel: "burst", level: 0), "independent:0")
        XCTAssertEqual(AtlasLayout.lossCondition(lossModel: "burst", level: 0.3), "burst:0.3")
        let central = try XCTUnwrap(loadData().stratum("central"))
        XCTAssertEqual(AtlasLayout.levels(for: central), [0, 0.1, 0.3, 0.5])
    }

    func testSelectSpecParsing() {
        XCTAssertEqual(ResultsSelectSpec.parse("path=turn, loss=burst,arm=ct,bad,=x"), ["path": "turn", "loss": "burst", "arm": "ct"])
        XCTAssertEqual(ResultsSelectSpec.parse(nil), [:])
    }

    func testDelayScreenshotSpecSelectsThePoint() throws {
        let spec = ResultsSelectSpec.parse("path=turn,loss=independent,level=0.3,arm=recovery,delay=0.2")
        let state = ResultsSelectSpec.delayState(spec, data: try loadData())
        XCTAssertEqual(state.motion, "constant_turn")
        XCTAssertEqual(state.selection, ChartSelection(panel: 2, arm: 0, delayIndex: 5))
        let bad = ResultsSelectSpec.delayState(ResultsSelectSpec.parse("path=turn,level=0.7,arm=recovery,delay=0.2"),
                                               data: try loadData())
        XCTAssertNil(bad.selection, "an unknown level selects nothing rather than a wrong point")
    }

    func testAtlasScreenshotSpecSelectsTheCell() throws {
        let spec = ResultsSelectSpec.parse("stratum=central,loss=burst,level=0.3,variant=figure8,arm=ct")
        let state = ResultsSelectSpec.atlasState(spec, data: try loadData())
        XCTAssertEqual(state.lossCondition, "burst:0.3")
        XCTAssertEqual(state.selection, AtlasSelection(row: 7, column: 3))
    }

    func testModelOpensTheRequestedPageWithSelectionFromTheEnvironment() throws {
        let setup = ScreenshotSetup(environment: ["ZEPHYR_OPEN": "results:atlas",
                                                  "ZEPHYR_RESULTS_SELECT": "stratum=noise10,variant=jink,arm=adaptive",
                                                  "ZEPHYR_INSPECTOR": "0"])
        XCTAssertTrue(setup.isActive)
        let model = ResultsModel(load: .success(try loadData()), setup: setup)
        XCTAssertEqual(model.page, .atlas)
        XCTAssertFalse(model.showInspector)
        XCTAssertEqual(model.atlas.stratumID, "noise10")
        XCTAssertNotNil(model.atlas.selection)
        XCTAssertNil(ResultsPage(openRequest: "results:missions"))
        XCTAssertNil(ResultsPage(openRequest: "about"))
    }

    func testKeyboardMovesSelectionAndLetsTabLeaveAtTheEnd() throws {
        let model = ResultsModel(load: .success(try loadData()), setup: ScreenshotSetup(environment: [:]))
        XCTAssertFalse(model.moveChart(.nextPanel), "Tab with nothing selected moves focus on")
        XCTAssertTrue(model.moveChart(.right))
        XCTAssertEqual(model.delay.selection, ChartSelection(panel: 0, arm: 0, delayIndex: 0), "the first arrow selects the first point")
        XCTAssertTrue(model.moveChart(.right))
        XCTAssertEqual(model.delay.selection?.delayIndex, 1)
        model.delay.selection = ChartSelection(panel: 3, arm: 0, delayIndex: 1)
        XCTAssertFalse(model.moveChart(.nextPanel), "Tab past the last panel is not swallowed")
        XCTAssertTrue(model.moveChart(.right), "arrow keys at an edge are still handled")
        model.clearSelection()
        XCTAssertNil(model.delay.selection)
    }

    func testChangingStratumKeepsTheSelectionInsideIt() throws {
        let model = ResultsModel(load: .success(try loadData()), setup: ScreenshotSetup(environment: ["ZEPHYR_OPEN": "results:atlas"]))
        model.atlas.selection = AtlasSelection(row: 8, column: 4)
        model.selectStratum("noise0")
        XCTAssertEqual(model.atlas.selection, AtlasSelection(row: 6, column: 2))
    }

    func testLoadFailureLeavesNoDataAndKeepsTheMessage() {
        let model = ResultsModel(load: .failure(ResultsDecodeError(message: "broken file")), setup: ScreenshotSetup(environment: [:]))
        XCTAssertNil(model.data)
        XCTAssertEqual(model.loadError, "broken file")
        XCTAssertTrue(model.panels.isEmpty)
        XCTAssertFalse(model.moveChart(.right))
    }

    func testEmptyInspectorHintsAndShortcutAreGenerated() {
        XCTAssertEqual(ResultsPage.delay.emptyHint, UIText.Results.inspectorEmptyHintDelay)
        XCTAssertEqual(ResultsPage.atlas.emptyHint, UIText.Results.inspectorEmptyHintAtlas)
        let shortcut = ShortcutRouter.keyboardShortcut(for: "showResults")
        XCTAssertEqual(shortcut?.key, KeyEquivalent("r"))
        XCTAssertEqual(shortcut?.modifiers, .command)
    }
}
