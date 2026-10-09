import Foundation
import CryptoKit

/// The subset of GitHub's latest-release response used by the native updater.
/// A release is installable only when it contains a platform ZIP with a
/// SHA-256 digest. The updater never pulls from a branch or executes a script
/// downloaded from GitHub.
struct ZephyrReleaseAsset: Codable, Sendable {
    let name: String
    let browserDownloadURL: String
    let digest: String?

    enum CodingKeys: String, CodingKey {
        case name, browserDownloadURL = "browser_download_url", digest
    }
}

struct ZephyrReleaseInfo: Codable, Sendable {
    let tagName: String
    let name: String
    let htmlURL: String
    let assets: [ZephyrReleaseAsset]

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name", name, htmlURL = "html_url", assets
    }
}

struct ZephyrUpdateCheck: Sendable {
    let release: ZephyrReleaseInfo
    let asset: ZephyrReleaseAsset
    let isNewer: Bool
}

enum ReleaseUpdaterError: Error, LocalizedError {
    case invalidReleaseJSON
    case network(String)
    case httpStatus(Int)
    case noAsset(String)
    case missingDigest(String)
    case unsafeFilename
    case digestMismatch(expected: String, actual: String)
    case notAnApplicationBundle
    case updateProcessUnavailable

    var errorDescription: String? {
        switch self {
        case .invalidReleaseJSON: return "GitHub release JSON is invalid"
        case .network(let message): return "GitHub request failed: \(message)"
        case .httpStatus(let status): return "GitHub returned HTTP \(status)"
        case .noAsset(let platform): return "No release ZIP asset for \(platform)"
        case .missingDigest(let name): return "Release asset has no SHA-256 digest: \(name)"
        case .unsafeFilename: return "Release filename must be a plain file name"
        case .digestMismatch(let expected, let actual): return "Release digest mismatch: expected \(expected), got \(actual)"
        case .notAnApplicationBundle: return "Updates require a Finder-launched ZephyrMesh.app bundle"
        case .updateProcessUnavailable: return "Could not start the update helper"
        }
    }
}

enum ReleaseUpdater {
    static let repository = "Whyater/Zephyr-Mesh"
    static let currentVersion: String = {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "0.1.0"
    }()

    static func platformKey() -> String {
        #if arch(arm64)
        return "macos-arm64"
        #else
        return "macos-x86_64"
        #endif
    }

    static func decodeRelease(_ data: Data) throws -> ZephyrReleaseInfo {
        do { return try JSONDecoder().decode(ZephyrReleaseInfo.self, from: data) }
        catch { throw ReleaseUpdaterError.invalidReleaseJSON }
    }

    static func select(_ release: ZephyrReleaseInfo, platform: String = platformKey()) throws -> ZephyrReleaseAsset {
        let tokens: [String]
        let base = platform.split(separator: "-", maxSplits: 1).first.map(String.init) ?? platform
        let architecture = platform.split(separator: "-", maxSplits: 1).dropFirst().first.map(String.init)
        switch base {
        case "macos": tokens = ["macos", "darwin", "mac"]
        case "windows": tokens = ["windows", "win"]
        case "linux": tokens = ["linux"]
        default: throw ReleaseUpdaterError.noAsset(platform)
        }
        let candidates = release.assets.filter { asset in
            let lower = asset.name.lowercased()
            return lower.hasSuffix(".zip") && tokens.contains(where: lower.contains)
        }.sorted { $0.name < $1.name }
        let preferred: [ZephyrReleaseAsset]
        if let architecture {
            let archTokens = architecture == "arm64" ? ["arm64", "aarch64"] : architecture == "x86_64" ? ["x86_64", "amd64", "x64"] : [architecture]
            preferred = candidates.filter { asset in archTokens.contains(where: asset.name.lowercased().contains) }
        } else {
            preferred = candidates
        }
        let universal = candidates.filter { asset in
            let lower = asset.name.lowercased()
            return ["universal", "any-arch", "any_arch"].contains(where: lower.contains)
        }
        guard let asset = (preferred.first ?? universal.first) else { throw ReleaseUpdaterError.noAsset(platform) }
        return asset
    }

    static func verify(_ data: Data, against asset: ZephyrReleaseAsset) throws {
        guard let digest = asset.digest else { throw ReleaseUpdaterError.missingDigest(asset.name) }
        let expected = normalizedDigest(digest)
        guard expected.count == 64, expected.allSatisfy({ $0.isHexDigit }) else { throw ReleaseUpdaterError.digestMismatch(expected: expected, actual: "invalid SHA-256") }
        let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard actual == expected else { throw ReleaseUpdaterError.digestMismatch(expected: expected, actual: actual) }
    }

    static func checkLatest(currentVersion: String = Self.currentVersion) async throws -> ZephyrUpdateCheck {
        guard let url = URL(string: "https://api.github.com/repos/\(repository)/releases/latest") else {
            throw ReleaseUpdaterError.network("invalid GitHub URL")
        }
        var request = URLRequest(url: url)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("Zephyr-Mesh-Updater/0.1", forHTTPHeaderField: "User-Agent")
        let data: Data
        let response: URLResponse
        do { (data, response) = try await URLSession.shared.data(for: request) }
        catch { throw ReleaseUpdaterError.network(error.localizedDescription) }
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw ReleaseUpdaterError.httpStatus(http.statusCode)
        }
        let release = try decodeRelease(data)
        let asset = try select(release)
        return ZephyrUpdateCheck(release: release, asset: asset, isNewer: isNewer(versionKey(release.tagName), than: versionKey(currentVersion)))
    }

    /// Downloads, verifies, extracts, and starts a detached helper that swaps
    /// the app after this process exits. Returns the release tag that was
    /// scheduled. A bundle launched with `swift run` is intentionally rejected
    /// because it has no stable install directory to replace.
    static func installLatest(currentBundle: URL = Bundle.main.bundleURL, currentVersion: String = Self.currentVersion) async throws -> String {
        guard currentBundle.pathExtension == "app" else { throw ReleaseUpdaterError.notAnApplicationBundle }
        let check = try await checkLatest(currentVersion: currentVersion)
        guard check.isNewer else { return check.release.tagName }
        guard let downloadURL = URL(string: check.asset.browserDownloadURL) else { throw ReleaseUpdaterError.network("invalid release asset URL") }
        var request = URLRequest(url: downloadURL)
        request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")
        request.setValue("Zephyr-Mesh-Updater/0.1", forHTTPHeaderField: "User-Agent")
        let data: Data
        let response: URLResponse
        do { (data, response) = try await URLSession.shared.data(for: request) }
        catch { throw ReleaseUpdaterError.network(error.localizedDescription) }
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw ReleaseUpdaterError.httpStatus(http.statusCode)
        }
        try verify(data, against: check.asset)
        let filename = URL(fileURLWithPath: check.asset.name).lastPathComponent
        guard filename == check.asset.name, !filename.isEmpty else { throw ReleaseUpdaterError.unsafeFilename }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("zephyr-update-\(UUID().uuidString)", isDirectory: true)
        let archive = root.appendingPathComponent(filename)
        let extract = root.appendingPathComponent("payload", isDirectory: true)
        try FileManager.default.createDirectory(at: extract, withIntermediateDirectories: true)
        try data.write(to: archive, options: [.atomic])
        let helper = try writeHelper(root: root, archive: archive, extract: extract, install: currentBundle)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = [helper.path, String(ProcessInfo.processInfo.processIdentifier)]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { throw ReleaseUpdaterError.updateProcessUnavailable }
        return check.release.tagName
    }

    /// Copy a verified local release artifact into a staging directory. This
    /// helper is retained for offline tests and release tooling.
    static func stage(localURL: URL, asset: ZephyrReleaseAsset, stagingDirectory: URL) throws -> URL {
        let filename = localURL.lastPathComponent
        guard !filename.isEmpty, filename == asset.name, !filename.contains("/") && !filename.contains("\\") else { throw ReleaseUpdaterError.unsafeFilename }
        let data = try Data(contentsOf: localURL)
        try verify(data, against: asset)
        let manager = FileManager.default
        try manager.createDirectory(at: stagingDirectory, withIntermediateDirectories: true)
        let destination = stagingDirectory.appendingPathComponent(asset.name, isDirectory: false)
        let temporary = stagingDirectory.appendingPathComponent(".\(asset.name).\(UUID().uuidString).part", isDirectory: false)
        try data.write(to: temporary, options: [.atomic])
        if manager.fileExists(atPath: destination.path) { try manager.removeItem(at: destination) }
        try manager.moveItem(at: temporary, to: destination)
        return destination
    }

    private static func normalizedDigest(_ digest: String) -> String {
        let lower = digest.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        return lower.hasPrefix("sha256:") ? String(lower.dropFirst(7)) : lower
    }

    private static func versionKey(_ value: String) -> [Int] {
        value.split { !$0.isNumber }.compactMap { Int($0) }
    }

    private static func isNewer(_ lhs: [Int], than rhs: [Int]) -> Bool {
        let width = max(lhs.count, rhs.count)
        for index in 0..<width {
            let left = index < lhs.count ? lhs[index] : 0
            let right = index < rhs.count ? rhs[index] : 0
            if left != right { return left > right }
        }
        return false
    }

    private static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    private static func writeHelper(root: URL, archive: URL, extract: URL, install: URL) throws -> URL {
        let helper = root.appendingPathComponent("apply-update.sh")
        let backup = install.deletingLastPathComponent().appendingPathComponent(install.lastPathComponent + ".previous")
        let launch = install.appendingPathComponent("Contents/MacOS/ZephyrMeshApp")
        let script = """
        #!/bin/sh
        set -eu
        pid="$1"
        while kill -0 "$pid" 2>/dev/null; do sleep 0.25; done
        entries_file=\(shellQuote(root.appendingPathComponent("entries.txt").path))
        /usr/bin/unzip -Z1 \(shellQuote(archive.path)) > "$entries_file"
        while IFS= read -r entry; do
            case "$entry" in /*|../*|*/../*|*/..|..|*'\\\\'*) exit 1 ;; esac
        done < "$entries_file"
        /usr/bin/ditto -x -k \(shellQuote(archive.path)) \(shellQuote(extract.path))
        symlink=$(find \(shellQuote(extract.path)) -type l -print -quit)
        test -z "$symlink"
        payload_count=$(find \(shellQuote(extract.path)) -maxdepth 1 -type d ! -path \(shellQuote(extract.path)) ! -name '__MACOSX' -print | /usr/bin/wc -l | /usr/bin/tr -d ' ')
        test "$payload_count" = "1"
        payload=$(find \(shellQuote(extract.path)) -maxdepth 1 -type d ! -path \(shellQuote(extract.path)) ! -name '__MACOSX' -print -quit)
        case "$payload" in *.app) ;; *) exit 1 ;; esac
        test -x "$payload/Contents/MacOS/ZephyrMeshApp"
        /usr/bin/codesign --verify --deep --strict "$payload"
        rm -rf \(shellQuote(backup.path))
        if [ -d \(shellQuote(install.path)) ]; then mv \(shellQuote(install.path)) \(shellQuote(backup.path)); fi
        mv "$payload" \(shellQuote(install.path))
        /usr/bin/open \(shellQuote(launch.path))
        rm -rf \(shellQuote(root.path))
        """
        try script.write(to: helper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: helper.path)
        return helper
    }
}
