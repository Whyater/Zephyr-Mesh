import Metal
import SceneKit
import SwiftUI
import XCTest
@testable import ZephyrMeshApp

private func sourceURL(_ relative: String) -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent(relative)
}

private let fixtureURL = sourceURL("Sources/ZephyrMeshApp/Resources/demo_swarm.json")

private func loadFixture() throws -> Fixture {
    try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
}

/// Re-encodes the bundled fixture after `edit` changes its JSON object tree.
private func decodeEdited(_ edit: (inout [String: Any]) -> Void) throws -> Fixture {
    var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL)) as? [String: Any])
    edit(&object)
    return try JSONDecoder().decode(Fixture.self, from: JSONSerialization.data(withJSONObject: object))
}

final class ReplayFixtureTests: XCTestCase {
    func testNativeFixtureDecodesSixFramesAndFiftyDrones() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.schema, "zephyr-s7-native-fixture-4")
        XCTAssertEqual(fixture.frames.count, 6)
        XCTAssertEqual(fixture.drones.count, 50)
        XCTAssertTrue(fixture.frames.allSatisfy { $0.drones.count == 50 })
        XCTAssertEqual(fixture.runName, "50-drone ring, seed 17")
        XCTAssertEqual(fixture.profiles.count, 50)
        XCTAssertEqual(fixture.notInReplay, [
            "Live radio link", "Camera or vision input", "Battery state",
            "Flight controller commands", "Measured flight performance",
        ])
    }

    func testRunNameAndNotInReplayComeFromTheGoldenPresentation() throws {
        // The fixture copies these from desktop/presentation.py; Swift never builds them.
        let goldenURL = sourceURL("../../design/golden/s7_presentation.json")
        let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: goldenURL)) as? [String: Any])
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.runName, raw["run_name"] as? String)
        XCTAssertEqual(fixture.notInReplay, raw["not_in_replay"] as? [String])
    }

    func testMissingNeighborCountDecodesAsNotRecorded() throws {
        let fixture = try decodeEdited { object in
            var frames = object["frames"] as! [[String: Any]]
            var drones = frames[1]["drones"] as! [[String: Any]]
            drones[2]["neighbor_count"] = NSNull()
            drones[2]["min_neighbor_distance_m"] = NSNull()
            frames[1]["drones"] = drones
            var presentation = frames[1]["presentation"] as! [String: Any]
            var perDrone = presentation["per_drone"] as! [[String: Any]]
            perDrone[2]["neighbor_count"] = NSNull()
            perDrone[2]["nearest_neighbor_m"] = NSNull()
            presentation["per_drone"] = perDrone
            frames[1]["presentation"] = presentation
            object["frames"] = frames
        }
        XCTAssertNil(fixture.frames[1].drones[2].neighborCount)
        XCTAssertNil(fixture.frames[1].drones[2].minNeighborDistance)
        let drone = fixture.frames[1].presentation.perDrone[2]
        XCTAssertNil(drone.neighborCount)
        XCTAssertEqual(Format.integer(drone.neighborCount), "Not recorded")
        XCTAssertEqual(Format.meters(drone.nearestNeighborM), "Not recorded")
        XCTAssertEqual(Format.integer(0), "0", "a recorded zero stays zero")
    }

    func testFrameThreePresentationGroupsAndEventsMatchGolden() throws {
        let goldenURL = sourceURL("../../design/golden/s7_presentation.json")
        let fixture = try loadFixture()
        let golden = try JSONDecoder().decode(GoldenPresentation.self, from: Data(contentsOf: goldenURL))
        XCTAssertEqual(fixture.frames[3].presentation.groups, golden.frames[3].groups)
        XCTAssertEqual(fixture.frames[3].presentation.events, golden.frames[3].events)
    }

    func testMalformedFixtureFailsWithClearError() throws {
        XCTAssertThrowsError(try decodeEdited { $0["unexpected"] = true }) { error in
            XCTAssertTrue(error.localizedDescription.contains("invalid fixture keys"))
            XCTAssertTrue(error.localizedDescription.contains("unknown unexpected"))
        }
    }

    func testUnknownKeyInsideAFrameIsRejected() throws {
        XCTAssertThrowsError(try decodeEdited { object in
            var frames = object["frames"] as! [[String: Any]]
            frames[2]["extra_field"] = 1
            object["frames"] = frames
        }) { error in
            XCTAssertTrue(error.localizedDescription.contains("invalid frame keys (unknown extra_field)"), "\(error)")
        }
    }

    func testUnknownKeyInsideADroneIsRejected() throws {
        XCTAssertThrowsError(try decodeEdited { object in
            var frames = object["frames"] as! [[String: Any]]
            var drones = frames[0]["drones"] as! [[String: Any]]
            drones[4]["battery"] = 90
            frames[0]["drones"] = drones
            object["frames"] = frames
        }) { error in
            XCTAssertTrue(error.localizedDescription.contains("invalid drone keys (unknown battery)"), "\(error)")
        }
    }

    func testMissingKeyIsRejected() throws {
        XCTAssertThrowsError(try decodeEdited { $0.removeValue(forKey: "run_name") }) { error in
            XCTAssertTrue(error.localizedDescription.contains("missing run_name"), "\(error)")
        }
    }

    func testKeepOutVolumeIsTheTargetKeepOutFromTheManifest() throws {
        // runs/s7-swarm/manifest.json: target_keepout_radius_m 0.75, target at (0, 0, 1.5).
        let fixture = try loadFixture()
        for frame in fixture.frames {
            XCTAssertEqual(frame.keepOutVolumes.count, 1)
            XCTAssertEqual(frame.keepOutVolumes[0].label, "target_keepout")
            XCTAssertEqual(frame.keepOutVolumes[0].radiusM, 0.75)
            XCTAssertEqual(frame.keepOutVolumes[0].center, SIMD3<Float>(0, 0, 1.5))
        }
    }
}

private struct GoldenPresentation: Decodable {
    let frames: [FixturePresentation]
}

final class PresentationFormattingTests: XCTestCase {
    func testMissingValuesReadNotRecordedNeverZero() {
        XCTAssertEqual(Format.meters(nil), "Not recorded")
        XCTAssertEqual(Format.milliseconds(nil), "Not recorded")
        XCTAssertEqual(Format.percent(Optional<Double>.none), "Not recorded")
        XCTAssertEqual(Format.secondsRange(newest: nil, oldest: 0.2), "Not recorded")
    }

    func testPrecisionRules() {
        XCTAssertEqual(Format.meters(1.5), "1.50 m")
        XCTAssertEqual(Format.milliseconds(80.0), "80 ms")
        XCTAssertEqual(Format.percent(19.13), "19.1%")
        XCTAssertEqual(Format.seconds(0.1), "0.10 s")
        XCTAssertEqual(Format.percent(PresentationMetric.range(min: 50, max: 66.67)), "50\u{2013}67%")
        XCTAssertEqual(Format.secondsRange(newest: 0.1, oldest: 0.1), "0.10 s")
        XCTAssertEqual(Format.secondsRange(newest: 0.1, oldest: 0.15), "0.10\u{2013}0.15 s")
    }

    func testLabelsNeverShowSnakeCase() {
        XCTAssertEqual(Format.label("neighbor_count"), "Neighbors in range")
        XCTAssertEqual(Format.label("target_keepout"), "Target keep-out")
        XCTAssertEqual(Format.label("made_up_key"), "Made up key")
    }

    func testStatusStylesComeFromGeneratedText() {
        XCTAssertEqual(StatusStyle(level: "caution").symbol, "triangle.fill")
        XCTAssertEqual(StatusStyle(level: "caution").word, "Late")
        XCTAssertEqual(StatusStyle(level: "info").symbol, "info.circle")
        XCTAssertTrue(StatusStyle(level: "alert").hasSceneMarker)
        XCTAssertTrue(StatusStyle(level: "caution").hasSceneMarker)
        XCTAssertFalse(StatusStyle(level: "waiting").hasSceneMarker)
        XCTAssertFalse(StatusStyle(level: "ok").hasSceneMarker)
    }
}

final class TeamPresentationTests: XCTestCase {
    func testTeamSummaryIsReadFromThePresentation() throws {
        // Golden frame 3: Late 3 and OK 47. Frame 0: 50 waiting. Hand-checked against the group counts.
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.frames[3].presentation.teamSummary, "50 drones: 47 OK, 3 late")
        XCTAssertEqual(fixture.frames[0].presentation.teamSummary, "50 drones: 50 waiting")
        XCTAssertEqual(fixture.frames[5].presentation.teamSummary, "50 drones: 44 OK, 6 late")
        for frame in fixture.frames {
            let total = frame.presentation.groups.reduce(0) { $0 + $1.count }
            XCTAssertTrue(frame.presentation.teamSummary.hasPrefix("\(total) drones"))
        }
    }

    func testOneGroupPerStatusWithAnAgeRangeAndStalestFirst() throws {
        // Frame 5 has six late drones with ages 0.15 to 0.20 s: one Late group, not two.
        let presentation = try loadFixture().frames[5].presentation
        let late = presentation.groups.filter { $0.level == "caution" }
        XCTAssertEqual(late.count, 1)
        XCTAssertEqual(late[0].count, 6)
        XCTAssertEqual(late[0].targetAgeS, .range(min: 0.15, max: 0.2))
        XCTAssertEqual(Format.seconds(late[0].targetAgeS), "0.15\u{2013}0.20 s")
        let byID = Dictionary(uniqueKeysWithValues: presentation.perDrone.map { ($0.droneID, $0) })
        let ages = late[0].droneIDs.compactMap { byID[$0]?.targetAgeS }
        XCTAssertEqual(ages, ages.sorted(by: >), "stalest drone first")
    }

    func testFilterHidesWaitingButKeepsLateDrones() throws {
        let fixture = try loadFixture()
        let waiting = fixture.frames[0].presentation
        let hidden = TeamPresentation.rows(groups: waiting.groups, drones: waiting.perDrone, query: "zzz")
        XCTAssertTrue(hidden.isEmpty, "a waiting group is filtered like an OK group")
        let counts = TeamPresentation.filterCounts(rows: hidden, drones: waiting.perDrone, query: "zzz")
        XCTAssertEqual(counts, TeamPresentation.FilterCounts(
            shown: 0, matched: 0, total: 50, caption: "Showing 0 of 50", emptyText: "No drones match \"zzz\""))

        let presentation = fixture.frames[3].presentation
        let none = TeamPresentation.rows(groups: presentation.groups, drones: presentation.perDrone, query: "zzz")
        XCTAssertEqual(none.map(\.group.level), ["caution"])
        XCTAssertEqual(Set(none[0].members.map(\.compactName)), ["D09", "D39", "D44"])
        let pinned = TeamPresentation.filterCounts(rows: none, drones: presentation.perDrone, query: "zzz")
        XCTAssertEqual(pinned.shown, 3)
        XCTAssertEqual(pinned.matched, 0)

        let one = TeamPresentation.rows(groups: presentation.groups, drones: presentation.perDrone, query: "d07")
        XCTAssertEqual(one.map(\.group.level), ["caution", "ok"])
        XCTAssertEqual(one[1].members.map(\.droneID), ["drone-007"])
        XCTAssertTrue(one[1].isFilteredMatch)

        XCTAssertEqual(TeamPresentation.filterCounts(rows: one, drones: presentation.perDrone, query: "d07").caption,
                       "Showing 4 of 50")

        let all = TeamPresentation.rows(groups: presentation.groups, drones: presentation.perDrone, query: "  ")
        XCTAssertEqual(all.reduce(0) { $0 + $1.members.count }, 50)
        XCTAssertNil(TeamPresentation.filterCounts(rows: all, drones: presentation.perDrone, query: "  ").caption)
    }
}

@MainActor
final class CockpitModelTests: XCTestCase {
    func testEventsAccumulateOldestFirst() throws {
        let fixture = try loadFixture()
        let model = CockpitModel(fixture: fixture)
        model.seek(toFrame: 3)
        let expected = (0...3).flatMap { fixture.frames[$0].presentation.events }
        XCTAssertEqual(model.eventsSoFar, expected)
        XCTAssertEqual(model.eventsSoFar.first?.text, "Replay loaded: 50 drones")
        XCTAssertEqual(model.eventFrameIndexes, [0, 1, 2, 3, 4, 5])
    }

    func testSeekByTimeAndClamping() throws {
        let model = CockpitModel(fixture: try loadFixture())
        model.seek(toTime: 0.15)
        XCTAssertEqual(model.frameIndex, 3)
        model.seek(toFrame: 99)
        XCTAssertEqual(model.frameIndex, 5)
        model.stepFrame()
        XCTAssertEqual(model.frameIndex, 5)
        model.restart()
        XCTAssertEqual(model.frameIndex, 0)
    }

    func testPlaybackAdvancesOneFramePerDtOfScaledTime() throws {
        // dt is 0.05 s; at 0.25x a 0.2 s wall-clock tick is 0.05 s of replay, so one frame.
        let model = CockpitModel(fixture: try loadFixture())
        model.play()
        model.tick(elapsed: 0.2)
        XCTAssertEqual(model.frameIndex, 1)
        model.playbackSpeed = .realtime
        model.tick(elapsed: 0.1)
        XCTAssertEqual(model.frameIndex, 3)
        model.tick(elapsed: 10)
        XCTAssertEqual(model.frameIndex, 5)
        XCTAssertFalse(model.isPlaying)
    }

    func testSingleKeyShortcutsAreRefusedWhileTyping() throws {
        let model = CockpitModel(fixture: try loadFixture())
        model.isTextFieldFocused = true
        model.perform(actionID: "stepForward")
        model.perform(actionID: "toggleKeepOut")
        XCTAssertEqual(model.frameIndex, 0)
        XCTAssertTrue(model.showKeepOut)
        let before = model.filterFocusRequest
        model.perform(actionID: "focusFilter")
        XCTAssertEqual(model.filterFocusRequest, before + 1)
        model.isTextFieldFocused = false
        model.perform(actionID: "stepForward")
        XCTAssertEqual(model.frameIndex, 1)
    }

    func testFocusSelectsAndIssuesDistinctCameraCommands() throws {
        let model = CockpitModel(fixture: try loadFixture())
        model.focus(on: "drone-009")
        XCTAssertEqual(model.selectedID, "drone-009")
        let first = try XCTUnwrap(model.cameraCommand)
        model.focusSelected()
        XCTAssertEqual(model.cameraCommand?.kind, .focus(droneID: "drone-009"))
        XCTAssertNotEqual(model.cameraCommand?.serial, first.serial)
    }
}

final class OrbitCameraTests: XCTestCase {
    func testClampAndTopView() {
        var camera = OrbitCamera(pitch: -20, distance: 99, swarmRadius: 2)
        camera.clamp()
        XCTAssertEqual(camera.pitch, 10)
        XCTAssertEqual(camera.distance, 6)
        camera.applyZoom(factor: 1000)
        XCTAssertEqual(camera.distance, OrbitCamera.nearLimit)
        camera.applyDrag(deltaX: 0, deltaY: 500)
        XCTAssertEqual(camera.pitch, 85)
        camera.topView()
        XCTAssertEqual(camera.pitch, 90)
        XCTAssertEqual(camera.yaw, 0)
        camera.applyDrag(deltaX: 0, deltaY: -40)
        XCTAssertEqual(camera.pitch, 90, "drag never tilts out of top view")
        camera.orbitView()
        XCTAssertEqual(camera.pitch, OrbitCamera.homePitch)
    }

    func testFitAllRingOfFiftyDrones() throws {
        let fixture = try loadFixture()
        var camera = OrbitCamera()
        camera.frame(points: fixture.drones.map(\.position))
        XCTAssertEqual(camera.target.z, 1.5, accuracy: 0.001)
        XCTAssertEqual(camera.swarmRadius, 7.5, accuracy: 0.01)
        XCTAssertLessThanOrEqual(camera.distance, camera.swarmRadius * 3)
    }

    func testFocusAndInterpolationEndpoints() {
        var camera = OrbitCamera()
        camera.focus(point: SIMD3<Float>(1, 2, 3))
        XCTAssertEqual(camera.target, SIMD3<Float>(1, 2, 3))
        let target = OrbitCamera(yaw: 90, pitch: 70, distance: 3, target: SIMD3<Float>(2, 3, 4), swarmRadius: 2)
        XCTAssertEqual(camera.interpolated(to: target, progress: 0), camera)
        XCTAssertEqual(camera.interpolated(to: target, progress: 1), target)
        XCTAssertEqual(camera.interpolated(to: target, progress: 0.5, reducedMotion: true), camera)
    }

    func testSmoothingEasesOutAndSettles() {
        let start = OrbitCamera(yaw: 0, pitch: 30, distance: 10, swarmRadius: 10)
        let goal = OrbitCamera(yaw: 60, pitch: 30, distance: 10, swarmRadius: 10)
        // One time constant closes 1 - 1/e (about 63%) of the gap, independent of frame rate.
        let oneStep = start.smoothed(toward: goal, elapsed: 0.09)
        XCTAssertEqual(oneStep.yaw, 60 * (1 - exp(-1)), accuracy: 0.01)
        var twoSteps = start.smoothed(toward: goal, elapsed: 0.045)
        twoSteps = twoSteps.smoothed(toward: goal, elapsed: 0.045)
        XCTAssertEqual(twoSteps.yaw, oneStep.yaw, accuracy: 0.01)
        var camera = start
        for _ in 0..<200 { camera = camera.smoothed(toward: goal, elapsed: 1.0 / 60.0) }
        XCTAssertEqual(camera, goal)
    }

    func testScenePoseMapsSimulatorZUpToSceneKitYUp() {
        XCTAssertEqual(OrbitCamera.sceneVector(SIMD3(1, 2, 3)), SIMD3(1, 3, -2))
        var camera = OrbitCamera(distance: 5, target: SIMD3(0, 0, 1.5), swarmRadius: 5)
        camera.topView()
        let eye = camera.sceneEye
        XCTAssertEqual(eye.x, 0, accuracy: 1e-4)
        XCTAssertEqual(eye.y, 6.5, accuracy: 1e-4)
        XCTAssertEqual(eye.z, 0, accuracy: 1e-4)
        let forward = camera.sceneOrientation.act(SIMD3<Float>(0, 0, -1))
        XCTAssertEqual(forward.y, -1, accuracy: 1e-4, "top view looks straight down")
        let up = camera.sceneOrientation.act(SIMD3<Float>(0, 1, 0))
        XCTAssertEqual(up.z, -1, accuracy: 1e-4, "screen up is simulator +y (north)")
    }
}

final class ShortcutRouterTests: XCTestCase {
    func testGeneratedShortcutsRouteAndRespectTextFields() {
        let router = ShortcutRouter()
        XCTAssertEqual(router.action(forKey: "f"), "focusSelected")
        XCTAssertTrue(router.shouldFire(key: "f", textFieldFocused: false))
        XCTAssertFalse(router.shouldFire(key: "f", textFieldFocused: true))
        XCTAssertTrue(router.shouldFire(key: "o", modifiers: ["primary"], textFieldFocused: true))
        XCTAssertFalse(router.shouldFire(key: "z", textFieldFocused: false))
    }

    func testMenuKeyEquivalentsMatchUIText() {
        XCTAssertEqual(ShortcutRouter.keyboardShortcut(for: "playPause"), KeyboardShortcut(.space, modifiers: []))
        XCTAssertEqual(ShortcutRouter.keyboardShortcut(for: "stepBack"), KeyboardShortcut(.leftArrow, modifiers: []))
        XCTAssertEqual(ShortcutRouter.keyboardShortcut(for: "restart"), KeyboardShortcut(.home, modifiers: []))
        XCTAssertEqual(ShortcutRouter.keyboardShortcut(for: "openScenarioRun"), KeyboardShortcut("o", modifiers: [.command, .shift]))
        XCTAssertEqual(ShortcutRouter.keyboardShortcut(for: "aboutReplay"), KeyboardShortcut("i", modifiers: [.command]))
        XCTAssertNil(ShortcutRouter.keyboardShortcut(for: "notAnAction"))
        for shortcut in UIText.shortcuts {
            XCTAssertNotNil(ShortcutRouter.keyboardShortcut(for: shortcut.actionID), shortcut.actionID)
        }
    }
}

final class ScreenshotSetupTests: XCTestCase {
    func testNormalLaunchHasNoSetup() {
        XCTAssertFalse(ScreenshotSetup(environment: [:]).isActive)
        XCTAssertFalse(ScreenshotSetup(environment: ["HOME": "/Users/someone", "PATH": "/usr/bin"]).isActive)
        let setup = ScreenshotSetup(environment: ["ZEPHYR_SCREENSHOT_FRAME": "3", "ZEPHYR_APPEARANCE": "light"])
        XCTAssertEqual(setup.frame, 3)
        XCTAssertEqual(setup.appearance, .light)
    }
}

@MainActor
final class SceneControllerTests: XCTestCase {
    /// Renders once offscreen, then hit tests at the projected screen point of a drone, the
    /// same path InteractiveSceneView uses for hover and click.
    func testHitProxiesFindTheDroneUnderThePointer() throws {
        let fixture = try loadFixture()
        let controller = SceneController(fixture: fixture)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let renderer = SCNRenderer(device: device, options: nil)
        renderer.scene = controller.scene
        renderer.pointOfView = controller.cameraNode
        renderer.delegate = controller.driver
        let size = CGSize(width: 800, height: 600)
        _ = renderer.snapshot(atTime: 0, with: size, antialiasingMode: .none)
        let options: [SCNHitTestOption: Any] = [
            .categoryBitMask: SceneController.hitCategory,
            .searchMode: SCNHitTestSearchMode.closest.rawValue
        ]
        for index in [0, 8, 30] {
            let drone = fixture.frames[0].drones[index]
            let center = OrbitCamera.sceneVector(drone.position)
            let projected = renderer.projectPoint(SCNVector3(center.x, center.y, center.z))
            let point = CGPoint(x: projected.x, y: projected.y)
            XCTAssertEqual(renderer.hitTest(point, options: options).first?.node.name, drone.id, "drone \(index)")
        }
        let target = renderer.projectPoint(SCNVector3(0, 0, 0))
        XCTAssertTrue(renderer.hitTest(CGPoint(x: target.x, y: target.y), options: options).isEmpty)
    }
}

final class RepairRoundTests: XCTestCase {
    func testDroneWithMissingProfileIsRejected() throws {
        XCTAssertThrowsError(try decodeEdited { object in
            var profiles = object["profiles"] as! [[String: Any]]
            profiles.removeFirst()
            object["profiles"] = profiles
        }) { error in
            XCTAssertTrue(error.localizedDescription.contains("references hardware profile drone-001-synthetic-profile"), "\(error)")
        }
    }

    func testNonPositivePropellerIsRejected() throws {
        XCTAssertThrowsError(try decodeEdited { object in
            var profiles = object["profiles"] as! [[String: Any]]
            profiles[3]["propeller_diameter_m"] = 0.0
            object["profiles"] = profiles
        }) { error in
            XCTAssertTrue(error.localizedDescription.contains("positive arm length and propeller diameter"), "\(error)")
        }
    }

    func testArrowShortcutsGiveWayToAFocusedList() {
        let router = ShortcutRouter()
        XCTAssertFalse(router.shouldFire(actionID: "stepForward", textFieldFocused: false, listFocused: true))
        XCTAssertFalse(router.shouldFire(actionID: "stepBack", textFieldFocused: false, listFocused: true))
        XCTAssertTrue(router.shouldFire(actionID: "playPause", textFieldFocused: false, listFocused: true))
        XCTAssertTrue(router.shouldFire(actionID: "stepForward", textFieldFocused: false, listFocused: false))
        XCTAssertTrue(router.usesArrowKey("stepBack"))
        XCTAssertFalse(router.usesArrowKey("restart"))
    }

    func testEvidenceSummaryPutsStatsInWordsAndDropsOpaqueBlocks() {
        let stats = JSONValue.object([
            "available": .bool(true), "count": .integer(300), "mean": .number(0.75), "p95": .number(0.8),
            "max": .number(0.9), "min": .number(0.7), "p50": .number(0.75), "unit": .string("m")
        ])
        XCTAssertEqual(stats.summaryText(), "Mean 0.75, p95 0.8, max 0.9 m (300 samples)")
        XCTAssertEqual(stats.summaryText(includeUnit: false), "Mean 0.75, p95 0.8, max 0.9 (300 samples)")
        let unavailable = JSONValue.object(["available": .bool(false), "count": .integer(0), "mean": .null])
        XCTAssertEqual(unavailable.summaryText(), "Not recorded")
        let perDrone = JSONValue.object(["drone-001": .object(["censored_outages": .integer(1)])])
        XCTAssertNil(perDrone.summaryText())
        XCTAssertEqual(JSONValue.object([:]).summaryText(), "None")
        XCTAssertTrue(SummaryRows.detailsOnlyKeys.contains("schema"))
    }
}

@MainActor
final class SceneGeometryTests: XCTestCase {
    private func state(_ fixture: Fixture, frame: Int, selected: String?, command: CameraCommand? = nil) -> SceneState {
        let presentation = fixture.frames[frame].presentation
        return SceneState(
            frame: fixture.frames[frame],
            styles: Dictionary(uniqueKeysWithValues: presentation.perDrone.map { ($0.droneID, StatusStyle(drone: $0)) }),
            headings: [:], selectedID: selected, hoveredID: nil, showKeepOut: true, viewMode: .threeD,
            palette: Theme.dark, scenePalette: Theme.darkScene3D, isDark: true, reduceMotion: true, cameraCommand: command)
    }

    func testEachDroneIsSizedFromItsOwnProfile() throws {
        // Independent of the Swift formula's inputs: read the raw profile JSON and expect
        // hit radius = arm length + propeller diameter / 2 + 0.04 m for every drone.
        let fixture = try loadFixture()
        let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL)) as? [String: Any])
        let profiles = Dictionary(uniqueKeysWithValues: (raw["profiles"] as! [[String: Any]]).map { ($0["profile_id"] as! String, $0) })
        let controller = SceneController(fixture: fixture)
        var radii = Set<Double>()
        for drone in fixture.drones {
            let profile = try XCTUnwrap(profiles[drone.profileID])
            let expected = (profile["arm_length_m"] as! Double) + (profile["propeller_diameter_m"] as! Double) / 2 + 0.04
            let node = try XCTUnwrap(controller.scene.rootNode.childNode(withName: drone.id, recursively: false))
            let hit = try XCTUnwrap(node.childNodes.first { $0.categoryBitMask == SceneController.hitCategory })
            let radius = Double(try XCTUnwrap(hit.geometry as? SCNSphere).radius)
            XCTAssertEqual(radius, expected, accuracy: 1e-9, drone.id)
            radii.insert((radius * 1e6).rounded())
        }
        XCTAssertEqual(radii.count, 5, "the fixture has five propeller sizes")
    }

    func testLateDronesCarryAStatusMarkerInsteadOfFading() throws {
        // Golden frame 3: D09, D39 and D44 are late (caution); the other 47 are OK.
        let fixture = try loadFixture()
        let controller = SceneController(fixture: fixture)
        controller.apply(state(fixture, frame: 3, selected: "drone-009"))
        let root = controller.scene.rootNode
        func marker(_ id: String) throws -> SCNNode {
            let drone = try XCTUnwrap(root.childNode(withName: id, recursively: false))
            return try XCTUnwrap(drone.childNode(withName: "status-marker", recursively: false))
        }
        for id in ["drone-009", "drone-039", "drone-044"] {
            XCTAssertFalse(try marker(id).isHidden, "\(id) is late and must show its marker")
            XCTAssertEqual(try XCTUnwrap(root.childNode(withName: id, recursively: false)).opacity, 1,
                           "status is a marker, never opacity alone")
        }
        XCTAssertTrue(try marker("drone-001").isHidden)
        let late = fixture.frames[3].presentation.perDrone.filter { $0.level == "caution" || $0.level == "alert" }.count
        let shown = fixture.drones.filter { (try? marker($0.id).isHidden) == false }.count
        XCTAssertEqual(shown, late)
    }

    func testSelectionRingShowsFarAwayAndStepsAsideWhenFocused() throws {
        let fixture = try loadFixture()
        let controller = SceneController(fixture: fixture)
        controller.viewDidResize(CGSize(width: 640, height: 400))
        controller.apply(state(fixture, frame: 3, selected: "drone-009"))
        let ring = try XCTUnwrap(controller.scene.rootNode.childNode(withName: "selection-ring", recursively: false))
        XCTAssertFalse(ring.isHidden, "at the home framing the drone is a few points wide, so the ring marks it")
        // The ring sits at drone altitude (1.5 m plus half the propeller height), not on the floor.
        XCTAssertGreaterThan(ring.simdPosition.y, 1.5)
        let focused = state(fixture, frame: 3, selected: "drone-009",
                            command: CameraCommand(kind: .focus(droneID: "drone-009"), serial: 1))
        controller.apply(focused)
        XCTAssertTrue(ring.isHidden, "close up the drone is wider than the ring, so the ring steps aside")
    }

}

final class CameraFitTests: XCTestCase {
    func testFitKeepsTheRingInsideAndUsesTheAspect() throws {
        let points = try loadFixture().drones.map(\.position)
        let camera = OrbitCamera.framed(points: points)
        let wide = camera.fitDistance(points: points, aspect: 2.0, verticalFOV: 38)
        let narrow = camera.fitDistance(points: points, aspect: 1.0, verticalFOV: 38)
        XCTAssertGreaterThan(narrow, wide, "a narrower view must back the camera away")
        XCTAssertLessThanOrEqual(wide, camera.maximumDistance)
        // Every drone at the fitted distance is in front of the camera and inside the frustum.
        var fitted = camera
        fitted.distance = wide
        let inverse = fitted.sceneOrientation.inverse
        let tanY = tan(Float(19) * .pi / 180)
        for point in points {
            let local = inverse.act(OrbitCamera.sceneVector(point) - fitted.sceneEye)
            XCTAssertGreaterThan(-local.z, 0)
            XCTAssertLessThanOrEqual(abs(local.y), -local.z * tanY)
            XCTAssertLessThanOrEqual(abs(local.x), -local.z * tanY * 2)
        }
        // A single point at the target needs no distance beyond the near limit.
        XCTAssertEqual(camera.fitDistance(points: [camera.target], aspect: 1.5, verticalFOV: 38), OrbitCamera.nearLimit, accuracy: 1e-3)
    }
}
