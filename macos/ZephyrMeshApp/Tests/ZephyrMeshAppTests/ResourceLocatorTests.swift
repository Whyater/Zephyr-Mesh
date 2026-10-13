import Foundation
import XCTest
@testable import ZephyrMeshApp

final class ResourceLocatorTests: XCTestCase {
    private var temporaryDirectory: URL!

    override func setUpWithError() throws {
        temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("resource-locator-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let temporaryDirectory { try? FileManager.default.removeItem(at: temporaryDirectory) }
    }

    func testFindsResourceInFlatGeneratedBundle() throws {
        let bundle = try makeDirectory("ZephyrMeshApp_ZephyrMeshApp.bundle")
        let expected = bundle.appendingPathComponent("demo_swarm.json")
        try Data("flat".utf8).write(to: expected)

        XCTAssertEqual(
            ResourceLocator.url(forResource: "demo_swarm", withExtension: "json", searchRoots: [temporaryDirectory]),
            expected
        )
    }

    func testFindsResourceInContentsResourcesBundle() throws {
        let bundle = try makeDirectory("ZephyrMeshApp_ZephyrMeshApp.bundle/Contents/Resources")
        let expected = bundle.appendingPathComponent("demo_swarm.json")
        try Data("nested".utf8).write(to: expected)

        XCTAssertEqual(
            ResourceLocator.url(forResource: "demo_swarm", withExtension: "json", searchRoots: [temporaryDirectory]),
            expected
        )
    }

    func testFindsBareResourceInContentsResources() throws {
        let resources = try makeDirectory("Contents/Resources")
        let expected = resources.appendingPathComponent("demo_swarm.json")
        try Data("bare".utf8).write(to: expected)

        XCTAssertEqual(
            ResourceLocator.url(forResource: "demo_swarm", withExtension: "json", searchRoots: [temporaryDirectory]),
            expected
        )
    }

    func testFindsBundleBesideExecutableContentsDirectory() throws {
        let executableDirectory = try makeDirectory("App/Contents/MacOS")
        let bundle = try makeDirectory("App/Contents/Resources/ZephyrMeshApp_ZephyrMeshApp.bundle")
        let expected = bundle.appendingPathComponent("demo_swarm.json")
        try Data("beside executable".utf8).write(to: expected)

        XCTAssertEqual(
            ResourceLocator.url(
                forResource: "demo_swarm",
                withExtension: "json",
                searchRoots: [executableDirectory]
            ),
            expected
        )
    }

    func testNeverLooksInTheFolderThatHoldsTheApp() throws {
        // A stray bundle or file beside the app (for example in ~/Applications) must not be used.
        let app = try makeDirectory("Applications/Zephyr Mesh.app")
        let strayBundle = try makeDirectory("Applications/Resources/ZephyrMeshApp_ZephyrMeshApp.bundle")
        try Data("stray".utf8).write(to: strayBundle.appendingPathComponent("demo_swarm.json"))
        try Data("stray".utf8).write(to: strayBundle.deletingLastPathComponent().appendingPathComponent("demo_swarm.json"))

        XCTAssertNil(ResourceLocator.url(forResource: "demo_swarm", withExtension: "json", searchRoots: [app]))
    }

    func testSearchesRootsInOrder() throws {
        let firstRoot = try makeDirectory("first/ZephyrMeshApp_ZephyrMeshApp.bundle")
        let secondRoot = try makeDirectory("second/ZephyrMeshApp_ZephyrMeshApp.bundle")
        let first = firstRoot.appendingPathComponent("demo_swarm.json")
        let second = secondRoot.appendingPathComponent("demo_swarm.json")
        try Data("first".utf8).write(to: first)
        try Data("second".utf8).write(to: second)

        XCTAssertEqual(
            ResourceLocator.url(
                forResource: "demo_swarm",
                withExtension: "json",
                searchRoots: [firstRoot.deletingLastPathComponent(), secondRoot.deletingLastPathComponent()]
            ),
            first
        )
    }

    func testMissingResourceReturnsNil() {
        XCTAssertNil(ResourceLocator.url(forResource: "demo_swarm", withExtension: "json", searchRoots: [temporaryDirectory]))
    }

    func testBundledFixtureLoadsWithoutSwiftPMModuleBundleAccessor() {
        switch FixtureLoader.loadBundled() {
        case .success(let fixture):
            XCTAssertEqual(fixture.runName, "50-drone ring, seed 17")
        case .failure(let error):
            XCTFail("fixture should be available during swift test: \(error)")
        }
    }

    private func makeDirectory(_ relativePath: String) throws -> URL {
        let directory = temporaryDirectory.appendingPathComponent(relativePath, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
