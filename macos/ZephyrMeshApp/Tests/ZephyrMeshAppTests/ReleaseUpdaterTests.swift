import XCTest
import CryptoKit
@testable import ZephyrMeshApp

final class ReleaseUpdaterTests: XCTestCase {
    // MARK: - Shared cases (the Python tests read the same file)

    private static let cases: [String: Any] = {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/updater-cases.json")
        let data = try! Data(contentsOf: url)
        return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    }()

    private func release(tag: String = "v0.2.0", _ names: [String]) -> ZephyrReleaseInfo {
        ZephyrReleaseInfo(
            tagName: tag,
            name: "Zephyr Mesh \(tag)",
            htmlURL: "https://github.com/Whyater/Zephyr-Mesh/releases/tag/\(tag)",
            assets: names.map { ZephyrReleaseAsset(name: $0, browserDownloadURL: "https://github.com/Whyater/Zephyr-Mesh/releases/download/\(tag)/\($0)", digest: String(repeating: "a", count: 64)) }
        )
    }

    func testVersionOrderingMatchesSharedCases() throws {
        let ordering = Self.cases["ordering"] as! [String: Any]
        let compare = ordering["compare"] as! [[String: String]]
        XCTAssertGreaterThan(compare.count, 5)
        for item in compare {
            let a = try XCTUnwrap(ReleaseUpdater.versionKey(item["a"]!), "\(item["a"]!) should parse")
            let b = try XCTUnwrap(ReleaseUpdater.versionKey(item["b"]!), "\(item["b"]!) should parse")
            let actual = ReleaseUpdater.isNewer(a, than: b) ? "greater" : ReleaseUpdater.isNewer(b, than: a) ? "less" : "equal"
            XCTAssertEqual(actual, item["expect"], "\(item["a"]!) vs \(item["b"]!)")
        }
    }

    func testUnparseableVersionsAreRejectedNotGuessed() {
        let ordering = Self.cases["ordering"] as! [String: Any]
        let invalid = ordering["invalid"] as! [String]
        XCTAssertGreaterThan(invalid.count, 5)
        for value in invalid {
            XCTAssertNil(ReleaseUpdater.versionKey(value), "\(value.debugDescription) should be rejected")
        }
    }

    func testEqualVersionsAndPreReleaseOrdering() throws {
        let v020 = try XCTUnwrap(ReleaseUpdater.versionKey("v0.2.0"))
        let v02 = try XCTUnwrap(ReleaseUpdater.versionKey("0.2"))
        XCTAssertFalse(ReleaseUpdater.isNewer(v020, than: v02))
        XCTAssertFalse(ReleaseUpdater.isNewer(v02, than: v020))
        let rc = try XCTUnwrap(ReleaseUpdater.versionKey("v0.3.0-rc1"))
        let final = try XCTUnwrap(ReleaseUpdater.versionKey("0.3.0"))
        XCTAssertTrue(ReleaseUpdater.isNewer(final, than: rc))
        XCTAssertFalse(ReleaseUpdater.isNewer(rc, than: final))
    }

    func testLocalDevelopmentVersionIsOlderThanRelease() throws {
        let release = try XCTUnwrap(ReleaseUpdater.versionKey("0.2.3"))
        let development = try XCTUnwrap(ReleaseUpdater.versionKey("0.2.3-dev"))
        XCTAssertTrue(ReleaseUpdater.isNewer(release, than: development))
        XCTAssertFalse(ReleaseUpdater.isNewer(development, than: release))
    }

    func testFreshLocalBuildIsNotOfferedAnOlderOrEqualRelease() throws {
        // build_app.py stamps local builds as <VERSION>-dev, so a fresh 0.2.3-dev
        // build must never be "updated" to the published v0.2.2 or to itself.
        let local = try XCTUnwrap(ReleaseUpdater.versionKey("0.2.3-dev"))
        for older in ["v0.2.2", "0.2.2", "v0.1.0", "v0.2.3-dev"] {
            let release = try XCTUnwrap(ReleaseUpdater.versionKey(older))
            XCTAssertFalse(ReleaseUpdater.isNewer(release, than: local), older)
        }
        for malformed in ["", "v", "-rc1", "-dev", "v-dev"] {
            XCTAssertNil(ReleaseUpdater.versionKey(malformed), malformed.debugDescription)
        }
    }

    func testAssetSelectionMatchesSharedCases() throws {
        let cases = Self.cases["select"] as! [[String: Any]]
        XCTAssertGreaterThan(cases.count, 5)
        for item in cases {
            let name = item["name"] as! String
            let info = release(tag: item["tag"] as! String, item["assets"] as! [String])
            let platform = item["platform"] as! String
            if let expected = item["expect"] as? String {
                XCTAssertEqual(try ReleaseUpdater.select(info, platform: platform).name, expected, name)
            } else {
                XCTAssertThrowsError(try ReleaseUpdater.select(info, platform: platform), name)
            }
        }
    }

    func testDownloadURLAllowlistMatchesSharedCases() {
        let urls = Self.cases["download_urls"] as! [String: Any]
        let repository = urls["repository"] as! String
        for url in urls["allowed"] as! [String] {
            XCTAssertTrue(ReleaseUpdater.isOfficialDownloadURL(url, repository: repository), url)
        }
        let rejected = urls["rejected"] as! [String]
        XCTAssertGreaterThan(rejected.count, 8)
        for url in rejected {
            XCTAssertFalse(ReleaseUpdater.isOfficialDownloadURL(url, repository: repository), url.debugDescription)
        }
    }

    func testRelaunchTargetIsTheApplicationBundle() {
        let bundle = URL(fileURLWithPath: "/tmp/ZephyrMesh.app")
        XCTAssertEqual(ReleaseUpdater.launchTarget(for: bundle), bundle)
        XCTAssertEqual(bundle.pathExtension, "app")
    }

    // MARK: - verify (SHA-256 integrity check)

    private func asset(digest: String?) -> ZephyrReleaseAsset {
        ZephyrReleaseAsset(name: "ZephyrMesh-macos-arm64-v0.2.0.zip", browserDownloadURL: "https://github.com/Whyater/Zephyr-Mesh/releases/download/v0.2.0/x.zip", digest: digest)
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    func testVerifyAcceptsTheMatchingDigestInEitherSpelling() throws {
        let data = Data("synthetic release bytes".utf8)
        // Independent reference: the well known digest of the empty input, plus a computed one.
        XCTAssertEqual(sha256Hex(Data()), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        let hex = sha256Hex(data)
        XCTAssertNoThrow(try ReleaseUpdater.verify(data, against: asset(digest: hex)))
        XCTAssertNoThrow(try ReleaseUpdater.verify(data, against: asset(digest: "sha256:" + hex.uppercased())))
        XCTAssertNoThrow(try ReleaseUpdater.verify(data, against: asset(digest: "  sha256:\(hex)\n")))
    }

    func testVerifyRejectsChangedBytesMissingAndMalformedDigests() {
        let data = Data("synthetic release bytes".utf8)
        let hex = sha256Hex(data)
        XCTAssertThrowsError(try ReleaseUpdater.verify(Data("tampered".utf8), against: asset(digest: hex))) { error in
            guard case ReleaseUpdaterError.digestMismatch = error else { return XCTFail("wrong error \(error)") }
        }
        XCTAssertThrowsError(try ReleaseUpdater.verify(data, against: asset(digest: nil))) { error in
            guard case ReleaseUpdaterError.missingDigest = error else { return XCTFail("wrong error \(error)") }
        }
        for bad in ["", "sha256:", "sha256:abc", String(repeating: "z", count: 64), hex + "00"] {
            XCTAssertThrowsError(try ReleaseUpdater.verify(data, against: asset(digest: bad)), bad)
        }
    }

    // MARK: - The generated install script, run for real in a temporary folder

    private struct Sandbox {
        let base: URL
        var applications: URL { base.appendingPathComponent("Applications") }
        var install: URL { applications.appendingPathComponent("ZephyrMesh.app") }
        var backupDirectory: URL { base.appendingPathComponent("Library/Caches/org.zephyrmesh.cockpit") }
        var root: URL { base.appendingPathComponent("update-root") }
        var extract: URL { root.appendingPathComponent("payload") }
        var result: URL { base.appendingPathComponent("logs/last-update.txt") }
        var openLog: URL { base.appendingPathComponent("open.log") }
        var openStub: URL { base.appendingPathComponent("open-stub.sh") }
        var shim: URL { base.appendingPathComponent("shim") }
    }

    private func makeSandbox() throws -> Sandbox {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("zephyr-updater-test-\(UUID().uuidString)", isDirectory: true)
        let box = Sandbox(base: base)
        let manager = FileManager.default
        try manager.createDirectory(at: box.extract, withIntermediateDirectories: true)
        try manager.createDirectory(at: box.shim, withIntermediateDirectories: true)
        try writeApp(at: box.install, marker: "old", sign: true)
        try "#!/bin/sh\necho \"$@\" >> '\(box.openLog.path)'\n".write(to: box.openStub, atomically: true, encoding: .utf8)
        try manager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: box.openStub.path)
        addTeardownBlock { try? manager.removeItem(at: base) }
        return box
    }

    /// A minimal app bundle. `sign` gives it an ad hoc signature, which is what the real releases carry.
    private func writeApp(at app: URL, marker: String, sign: Bool, executableBody: String = "exit 0") throws {
        let manager = FileManager.default
        let macOS = app.appendingPathComponent("Contents/MacOS", isDirectory: true)
        try manager.createDirectory(at: macOS, withIntermediateDirectories: true)
        let executable = macOS.appendingPathComponent("ZephyrMeshApp")
        try "#!/bin/sh\n\(executableBody)\n".write(to: executable, atomically: true, encoding: .utf8)
        try manager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        let plist = """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0"><dict><key>CFBundleExecutable</key><string>ZephyrMeshApp</string><key>CFBundleIdentifier</key><string>test.zephyr.updater</string><key>CFBundlePackageType</key><string>APPL</string></dict></plist>
        """
        try plist.write(to: app.appendingPathComponent("Contents/Info.plist"), atomically: true, encoding: .utf8)
        try marker.write(to: app.appendingPathComponent("Contents/marker.txt"), atomically: true, encoding: .utf8)
        if sign { try run("/usr/bin/codesign", ["--force", "--deep", "-s", "-", app.path]) }
    }

    @discardableResult
    private func run(_ tool: String, _ arguments: [String], environment: [String: String]? = nil) throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: tool)
        process.arguments = arguments
        if let environment { process.environment = environment }
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        return process.terminationStatus
    }

    private func zipDirectory(_ directory: URL, into archive: URL) throws {
        XCTAssertEqual(try run("/usr/bin/ditto", ["-c", "-k", "--keepParent", directory.path, archive.path]), 0)
    }

    /// A ZIP whose entries are empty files with the given names. Enough to test the path guard,
    /// which reads only the entry names, and the only way to write a name such as `../x`.
    private func zipWithEntryNames(_ names: [String], to archive: URL) throws {
        var body = Data(), central = Data()
        func bytes(_ values: [(Int, Int)]) -> Data {
            // (value, width in bytes), little endian
            var out = Data()
            for (value, width) in values { for index in 0..<width { out.append(UInt8((value >> (8 * index)) & 0xff)) } }
            return out
        }
        for name in names {
            let nameBytes = Data(name.utf8)
            let offset = body.count
            body.append(bytes([(0x04034b50, 4), (20, 2), (0, 2), (0, 2), (0, 2), (0x21, 2), (0, 4), (0, 4), (0, 4), (nameBytes.count, 2), (0, 2)]))
            body.append(nameBytes)
            central.append(bytes([(0x02014b50, 4), (20, 2), (20, 2), (0, 2), (0, 2), (0, 2), (0x21, 2), (0, 4), (0, 4), (0, 4), (nameBytes.count, 2), (0, 2), (0, 2), (0, 2), (0, 2), (0, 4), (offset, 4)]))
            central.append(nameBytes)
        }
        var end = bytes([(0x06054b50, 4), (0, 2), (0, 2), (names.count, 2), (names.count, 2), (central.count, 4), (body.count, 4), (0, 2)])
        end = Data(end)
        try (body + central + end).write(to: archive)
    }

    enum MoveFailure { case none, clean, halfway }

    /// Runs the real generated script with shims on PATH. `move` makes moving the new
    /// app into place fail (`halfway` first leaves a partial copy, like a full disk
    /// during a cross-disk copy). Only that move fails; restoring the backup works.
    /// `ps` is always shimmed: by default it reports the new app as running, and
    /// `newAppCrashes` makes it report nothing, as if the app died at launch.
    private func runUpdate(_ box: Sandbox, archive: URL, move: MoveFailure = .none, failFind: Bool = false,
                           newAppCrashes: Bool = false, archiveSHA256: String? = nil) throws -> Int32 {
        let script = ReleaseUpdater.helperScript(root: box.root, archive: archive, extract: box.extract, install: box.install,
                                                 resultFile: box.result, archiveSHA256: archiveSHA256,
                                                 openTool: box.openStub.path, startupCheckSeconds: 1)
        let scriptURL = box.base.appendingPathComponent("apply-update.sh")
        try script.write(to: scriptURL, atomically: true, encoding: .utf8)
        func shim(_ name: String, _ body: String) throws {
            let url = box.shim.appendingPathComponent(name)
            try "#!/bin/sh\n\(body)\n".write(to: url, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        }
        switch move {
        case .none: break
        case .clean:
            try shim("mv", """
            case "$1" in '\(box.extract.path)'/*) echo "simulated disk full" >&2; exit 1 ;; esac
            exec /bin/mv "$@"
            """)
        case .halfway:
            try shim("mv", """
            case "$1" in '\(box.extract.path)'/*) /bin/mkdir -p "$2/Contents"; echo partial > "$2/Contents/marker.txt"; echo "simulated disk full" >&2; exit 1 ;; esac
            exec /bin/mv "$@"
            """)
        }
        if failFind { try shim("find", "echo \"simulated find failure\" >&2\nexit 1") }
        try shim("ps", newAppCrashes ? "exit 0" : "cd '\(box.install.path)/Contents/MacOS' 2>/dev/null && echo \"$(pwd -P)/ZephyrMeshApp\"")
        // PID 99999999 does not exist, so the script's wait loop ends at once.
        return try run("/bin/sh", [scriptURL.path, "99999999"], environment: ["PATH": box.shim.path + ":/usr/bin:/bin", "HOME": box.base.path])
    }

    private func leftoverBackups(_ box: Sandbox) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: box.backupDirectory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.hasPrefix("ZephyrMesh.app.previous.") }
    }

    private func marker(_ app: URL) -> String? {
        try? String(contentsOf: app.appendingPathComponent("Contents/marker.txt"), encoding: .utf8)
    }

    private func buildSignedUpdateArchive(_ box: Sandbox, executableBody: String = "exit 0") throws -> URL {
        let staging = box.base.appendingPathComponent("new-staging", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        let app = staging.appendingPathComponent("ZephyrMesh.app")
        try writeApp(at: app, marker: "new", sign: true, executableBody: executableBody)
        let archive = box.base.appendingPathComponent("update.zip")
        try zipDirectory(app, into: archive)
        return archive
    }

    private func assertOldAppStillInPlaceAndRelaunched(_ box: Sandbox, reason: String, file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertEqual(marker(box.install), "old", "the installed app must be untouched", file: file, line: line)
        let note = try String(contentsOf: box.result, encoding: .utf8)
        XCTAssertTrue(note.contains("current version was kept"), note, file: file, line: line)
        XCTAssertTrue(note.contains(reason), "expected '\(reason)' in: \(note)", file: file, line: line)
        let opened = (try? String(contentsOf: box.openLog, encoding: .utf8)) ?? ""
        XCTAssertTrue(opened.contains(box.install.path), "the old app should be reopened: \(opened)", file: file, line: line)
        XCTAssertNil(takeNothing(box.root), file: file, line: line)
    }

    private func takeNothing(_ root: URL) -> String? {
        FileManager.default.fileExists(atPath: root.path) ? "update folder was not cleaned up" : nil
    }

    func testScriptInstallsASignedUpdateDeletesTheBackupAndClearsAnyOldFailure() throws {
        let box = try makeSandbox()
        let archive = try buildSignedUpdateArchive(box)
        try FileManager.default.createDirectory(at: box.result.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "stale failure".write(to: box.result, atomically: true, encoding: .utf8)
        XCTAssertEqual(try runUpdate(box, archive: archive), 0)
        XCTAssertEqual(marker(box.install), "new")
        XCTAssertTrue(FileManager.default.fileExists(atPath: box.backupDirectory.path), "the backup folder is the cache, not the install folder")
        XCTAssertTrue(leftoverBackups(box).isEmpty, "the backup is deleted once the new app is running")
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: box.applications.path), ["ZephyrMesh.app"], "nothing is left beside the app")
        XCTAssertFalse(FileManager.default.fileExists(atPath: box.result.path))
        XCTAssertTrue(((try? String(contentsOf: box.openLog, encoding: .utf8)) ?? "").contains(box.install.path))
        XCTAssertNil(takeNothing(box.root))
    }

    func testScriptRestoresTheBackupWhenTheFinalMoveFails() throws {
        let box = try makeSandbox()
        let archive = try buildSignedUpdateArchive(box)
        XCTAssertEqual(try runUpdate(box, archive: archive, move: .clean), 1)
        try assertOldAppStillInPlaceAndRelaunched(box, reason: "could not move the new app into place")
        XCTAssertTrue(leftoverBackups(box).isEmpty, "backup should have been moved back")
    }

    func testScriptRestoresTheBackupWhenTheFinalMoveFailsHalfway() throws {
        // The old script skipped the restore here because something existed at the
        // install path, then reopened the half-copied app and left the old one in Caches.
        let box = try makeSandbox()
        let archive = try buildSignedUpdateArchive(box)
        XCTAssertEqual(try runUpdate(box, archive: archive, move: .halfway), 1)
        try assertOldAppStillInPlaceAndRelaunched(box, reason: "could not move the new app into place")
        XCTAssertTrue(leftoverBackups(box).isEmpty, "backup should have been moved back")
    }

    func testScriptRefusesAPayloadWhoseSelfTestFails() throws {
        // A stand-in for the v0.2.2 build that crashed at launch: it exits 1 from --self-test.
        let box = try makeSandbox()
        let archive = try buildSignedUpdateArchive(box, executableBody: "echo 'self-test failed: The bundled replay file is missing from the app.'; exit 1")
        XCTAssertEqual(try runUpdate(box, archive: archive), 1)
        try assertOldAppStillInPlaceAndRelaunched(box, reason: "self-test failed: The bundled replay file is missing")
        XCTAssertTrue(leftoverBackups(box).isEmpty, "nothing was moved, so there is no backup")
    }

    func testScriptPutsTheOldAppBackWhenTheNewOneQuitsAtLaunch() throws {
        let box = try makeSandbox()
        let archive = try buildSignedUpdateArchive(box)
        XCTAssertEqual(try runUpdate(box, archive: archive, newAppCrashes: true), 1)
        XCTAssertEqual(marker(box.install), "old")
        let note = try String(contentsOf: box.result, encoding: .utf8)
        XCTAssertTrue(note.contains("current version was kept") && note.contains("did not start"), note)
        let opened = try String(contentsOf: box.openLog, encoding: .utf8).split(separator: "\n")
        XCTAssertEqual(opened.count, 2, "opens the new app, then the restored old one: \(opened)")
        XCTAssertTrue(leftoverBackups(box).isEmpty)
    }

    func testScriptChecksTheArchiveDigestAgainBeforeExtracting() throws {
        let box = try makeSandbox()
        let archive = try buildSignedUpdateArchive(box)
        XCTAssertEqual(try runUpdate(box, archive: archive, archiveSHA256: String(repeating: "0", count: 64)), 1)
        try assertOldAppStillInPlaceAndRelaunched(box, reason: "changed after it was checked")
        let digest = SHA256.hash(data: try Data(contentsOf: archive)).map { String(format: "%02x", $0) }.joined()
        let second = try makeSandbox()
        let secondArchive = second.base.appendingPathComponent("update.zip")
        try FileManager.default.copyItem(at: archive, to: secondArchive)
        XCTAssertEqual(try runUpdate(second, archive: secondArchive, archiveSHA256: digest), 0)
        XCTAssertEqual(marker(second.install), "new")
    }

    func testScriptFailsClosedWhenTheArchiveScanCannotRun() throws {
        let box = try makeSandbox()
        let archive = try buildSignedUpdateArchive(box)
        XCTAssertEqual(try runUpdate(box, archive: archive, failFind: true), 1)
        try assertOldAppStillInPlaceAndRelaunched(box, reason: "could not scan the archive")
    }

    func testScriptRejectsAnUnsignedBundleAndLeavesTheInstalledAppAlone() throws {
        let box = try makeSandbox()
        let staging = box.base.appendingPathComponent("new-staging", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        let app = staging.appendingPathComponent("ZephyrMesh.app")
        try writeApp(at: app, marker: "new", sign: false)
        let archive = box.base.appendingPathComponent("update.zip")
        try zipDirectory(app, into: archive)
        XCTAssertEqual(try runUpdate(box, archive: archive), 1)
        try assertOldAppStillInPlaceAndRelaunched(box, reason: "code signature")
    }

    func testScriptRejectsPathTraversalAndBothBackslashSpellings() throws {
        for (names, reason) in [
            (["ZephyrMesh.app/", "../escape.txt"], "unsafe path"),
            (["ZephyrMesh.app/", "ZephyrMesh.app/a/../../b"], "unsafe path"),
            (["/etc/evil"], "unsafe path"),
            (["ZephyrMesh.app/", "ZephyrMesh.app\\evil"], "unsafe path"),       // one backslash
            (["ZephyrMesh.app/", "ZephyrMesh.app\\\\evil"], "unsafe path"),     // two backslashes
        ] {
            let box = try makeSandbox()
            let archive = box.base.appendingPathComponent("hostile.zip")
            try zipWithEntryNames(names, to: archive)
            XCTAssertEqual(try runUpdate(box, archive: archive), 1, "\(names)")
            try assertOldAppStillInPlaceAndRelaunched(box, reason: reason)
        }
    }

    func testScriptRejectsSymlinksAndMoreThanOneApplication() throws {
        // symlink inside the app
        var box = try makeSandbox()
        var staging = box.base.appendingPathComponent("new-staging", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        var app = staging.appendingPathComponent("ZephyrMesh.app")
        try writeApp(at: app, marker: "new", sign: false)
        try FileManager.default.createSymbolicLink(atPath: app.appendingPathComponent("Contents/link").path, withDestinationPath: "/tmp")
        var archive = box.base.appendingPathComponent("update.zip")
        try zipDirectory(app, into: archive)
        XCTAssertEqual(try runUpdate(box, archive: archive), 1)
        try assertOldAppStillInPlaceAndRelaunched(box, reason: "symlink")

        // two application folders at the archive root
        box = try makeSandbox()
        staging = box.base.appendingPathComponent("two", isDirectory: true)
        try FileManager.default.createDirectory(at: staging.appendingPathComponent("A.app"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging.appendingPathComponent("B.app"), withIntermediateDirectories: true)
        archive = box.base.appendingPathComponent("two.zip")
        XCTAssertEqual(try run("/usr/bin/ditto", ["-c", "-k", staging.path, archive.path]), 0)
        // `ditto -c -k` without --keepParent puts A.app and B.app at the archive root
        XCTAssertEqual(try runUpdate(box, archive: archive), 1)
        try assertOldAppStillInPlaceAndRelaunched(box, reason: "exactly one application")
        app = box.install
        XCTAssertEqual(marker(app), "old")
    }

    func testTakeLastResultReturnsTheNoteOnceThenNothing() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("zephyr-result-\(UUID().uuidString).txt")
        XCTAssertNil(ReleaseUpdater.takeLastResult(from: file))
        try "The last update to Zephyr Mesh did not install: disk full\n".write(to: file, atomically: true, encoding: .utf8)
        XCTAssertEqual(ReleaseUpdater.takeLastResult(from: file), "The last update to Zephyr Mesh did not install: disk full")
        XCTAssertNil(ReleaseUpdater.takeLastResult(from: file))
    }
}
