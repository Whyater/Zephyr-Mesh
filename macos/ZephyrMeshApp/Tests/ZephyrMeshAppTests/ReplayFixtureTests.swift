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
        XCTAssertEqual(fixture.schema, "zephyr-s7-native-fixture-3")
        XCTAssertEqual(fixture.frames.count, 6)
        XCTAssertEqual(fixture.drones.count, 50)
        XCTAssertTrue(fixture.frames.allSatisfy { $0.drones.count == 50 })
        XCTAssertEqual(fixture.runName, "50-drone ring, seed 17")
        XCTAssertEqual(fixture.profiles.count, 50)
        XCTAssertFalse(fixture.unavailable.isEmpty)
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
        XCTAssertTrue(StatusStyle(level: "alert").fadesInScene)
        XCTAssertFalse(StatusStyle(level: "waiting").fadesInScene)
    }
}

final class TeamPresentationTests: XCTestCase {
    func testSummaryMatchesGoldenCounts() throws {
        // Golden frame 3: groups Late 3 and OK 47. Frame 0: one Waiting group of 50.
        let fixture = try loadFixture()
        XCTAssertEqual(TeamPresentation.summary(groups: fixture.frames[3].presentation.groups), "50 drones: 47 OK, 3 late")
        XCTAssertEqual(TeamPresentation.summary(groups: fixture.frames[0].presentation.groups), "50 drones: 50 waiting")
        XCTAssertEqual(TeamPresentation.summary(groups: fixture.frames[5].presentation.groups), "50 drones: 44 OK, 6 late")
    }

    func testFilterNeverHidesNonOKDrones() throws {
        let presentation = try loadFixture().frames[3].presentation
        let none = TeamPresentation.rows(groups: presentation.groups, drones: presentation.perDrone, query: "zzz")
        XCTAssertEqual(none.map(\.group.level), ["caution"])
        XCTAssertEqual(none[0].members.map(\.compactName), ["D09", "D39", "D44"])

        let one = TeamPresentation.rows(groups: presentation.groups, drones: presentation.perDrone, query: "d07")
        XCTAssertEqual(one.map(\.group.level), ["caution", "ok"])
        XCTAssertEqual(one[1].members.map(\.droneID), ["drone-007"])
        XCTAssertTrue(one[1].isFilteredMatch)

        let all = TeamPresentation.rows(groups: presentation.groups, drones: presentation.perDrone, query: "  ")
        XCTAssertEqual(all.reduce(0) { $0 + $1.members.count }, 50)
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
