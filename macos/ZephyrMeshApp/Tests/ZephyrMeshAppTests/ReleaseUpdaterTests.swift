import XCTest
@testable import ZephyrMeshApp

final class ReleaseUpdaterTests: XCTestCase {
    private func release(_ names: [String]) -> ZephyrReleaseInfo {
        ZephyrReleaseInfo(
            tagName: "v0.2.0",
            name: "Zephyr Mesh 0.2.0",
            htmlURL: "https://example.invalid/release",
            assets: names.map { ZephyrReleaseAsset(name: $0, browserDownloadURL: "https://example.invalid/\($0)", digest: String(repeating: "a", count: 64)) }
        )
    }

    func testSelectsExactArchitecture() throws {
        let selected = try ReleaseUpdater.select(release(["ZephyrMesh-windows-arm64.zip", "ZephyrMesh-windows-x86_64.zip"]), platform: "windows-arm64")
        XCTAssertTrue(selected.name.contains("arm64"))
    }

    func testUsesExplicitUniversalAsset() throws {
        let selected = try ReleaseUpdater.select(release(["ZephyrMesh-macos-universal.zip"]), platform: "macos-arm64")
        XCTAssertTrue(selected.name.contains("universal"))
    }

    func testRejectsMismatchedArchitecture() {
        XCTAssertThrowsError(try ReleaseUpdater.select(release(["ZephyrMesh-windows-x86_64.zip"]), platform: "windows-arm64"))
    }

    func testRelaunchTargetIsTheApplicationBundle() {
        let bundle = URL(fileURLWithPath: "/tmp/ZephyrMesh.app")
        XCTAssertEqual(ReleaseUpdater.launchTarget(for: bundle), bundle)
        XCTAssertEqual(bundle.pathExtension, "app")
    }
}
