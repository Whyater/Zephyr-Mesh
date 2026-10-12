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

/// An ordered release version. See `ReleaseUpdater.versionKey` for the rules.
struct ZephyrVersion: Equatable, Sendable {
    /// A numeric chunk sorts before a word, so `rc1` is older than `rcfinal`.
    enum Chunk: Equatable, Comparable, Sendable {
        case number(Int)
        case word(String)

        static func < (lhs: Chunk, rhs: Chunk) -> Bool {
            switch (lhs, rhs) {
            case let (.number(a), .number(b)): return a < b
            case let (.word(a), .word(b)): return a < b
            case (.number, .word): return true
            case (.word, .number): return false
            }
        }
    }

    /// Dotted numbers with trailing zeros removed.
    let numbers: [Int]
    /// nil for a normal release.
    let preRelease: [Chunk]?
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
    case invalidVersion(String)
    case untrustedDownloadURL(String)

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
        case .invalidVersion(let value): return "Unrecognized version: \(value)"
        case .untrustedDownloadURL(let url): return "Download URL is not an official release asset of \(ReleaseUpdater.repository): \(url)"
        }
    }
}

enum ReleaseUpdater {
    static let repository = "Whyater/Zephyr-Mesh"
    static let currentVersion: String = {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "0.1.0"
    }()

    static func launchTarget(for bundle: URL) -> URL { bundle }

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

    /// Returns the asset named exactly `ZephyrMesh-{platform}-v{version}.zip`,
    /// the name `tools/package_release.py` writes. Matching the whole name (not
    /// a substring such as `win` inside `darwin`) means a release can never
    /// hand a Mac build to a Windows machine or the reverse.
    static func select(_ release: ZephyrReleaseInfo, platform: String = platformKey()) throws -> ZephyrReleaseAsset {
        let parts = platform.split(separator: "-", maxSplits: 1).map(String.init)
        guard parts.count == 2, ["macos", "windows", "linux"].contains(parts[0]) else { throw ReleaseUpdaterError.noAsset(platform) }
        let version = release.tagName.hasPrefix("v") ? String(release.tagName.dropFirst()) : release.tagName
        let wanted = "ZephyrMesh-\(platform)-v\(version).zip"
        guard let asset = release.assets.first(where: { $0.name == wanted }) else { throw ReleaseUpdaterError.noAsset(platform) }
        return asset
    }

    /// True only for `https://github.com/<repository>/releases/download/<something>`.
    /// The asset URL arrives in the same API response as its digest, so the digest
    /// alone cannot tell a real release from a forged one. This narrows where a
    /// download may come from: no other host, repository, port, user info, or
    /// `..` path tricks. `updater-cases.json` holds the shared test cases.
    static func isOfficialDownloadURL(_ string: String, repository: String = ReleaseUpdater.repository) -> Bool {
        guard string.hasPrefix("https://github.com/"), !string.contains("\\"), let parts = URLComponents(string: string) else { return false }
        guard parts.scheme == "https", parts.host == "github.com", parts.user == nil, parts.password == nil, parts.port == nil else { return false }
        let prefix = "/\(repository)/releases/download/"
        let path = parts.percentEncodedPath
        guard path.hasPrefix(prefix) else { return false }
        let tail = String(path.dropFirst(prefix.count))
        let lowered = tail.lowercased()
        guard !tail.isEmpty, !lowered.contains("%2e"), !lowered.contains("%2f"), !lowered.contains("%5c") else { return false }
        return !tail.split(separator: "/", omittingEmptySubsequences: false).contains { $0 == "." || $0 == ".." }
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
        guard let latest = versionKey(release.tagName) else { throw ReleaseUpdaterError.invalidVersion(release.tagName) }
        guard let current = versionKey(currentVersion) else { throw ReleaseUpdaterError.invalidVersion(currentVersion) }
        return ZephyrUpdateCheck(release: release, asset: asset, isNewer: isNewer(latest, than: current))
    }

    /// Downloads, verifies, extracts, and starts a detached helper that swaps
    /// the app after this process exits. Returns the release tag that was
    /// scheduled. A bundle launched with `swift run` is intentionally rejected
    /// because it has no stable install directory to replace.
    static func installLatest(currentBundle: URL = Bundle.main.bundleURL, currentVersion: String = Self.currentVersion) async throws -> String {
        guard currentBundle.pathExtension == "app" else { throw ReleaseUpdaterError.notAnApplicationBundle }
        let check = try await checkLatest(currentVersion: currentVersion)
        guard check.isNewer else { return check.release.tagName }
        guard isOfficialDownloadURL(check.asset.browserDownloadURL) else { throw ReleaseUpdaterError.untrustedDownloadURL(check.asset.browserDownloadURL) }
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
        let helper = try writeHelper(root: root, archive: archive, extract: extract, install: currentBundle, resultFile: updateResultFile)
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

    /// Turns a release tag into an ordered key, or nil if it is not a version.
    /// Same rules as `version_key` in `tools/release_updater.py` (checked by the
    /// shared `updater-cases.json`): optional leading `v`, dotted numbers where
    /// trailing zeros do not count (0.2 equals v0.2.0), a hyphen suffix marks a
    /// pre-release that sorts below its own release (0.3.0-rc1 is older than
    /// 0.3.0), and `+build` metadata is ignored.
    static func versionKey(_ value: String) -> ZephyrVersion? {
        var text = Substring(value)
        if text.hasPrefix("v") { text = text.dropFirst() }
        let identifierCharacters: (Character) -> Bool = { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "." || $0 == "-") }
        if let plus = text.firstIndex(of: "+") {
            let build = text[text.index(after: plus)...]
            guard !build.isEmpty, build.allSatisfy(identifierCharacters) else { return nil }
            text = text[..<plus]
        }
        var preRelease: Substring?
        if let dash = text.firstIndex(of: "-") {
            preRelease = text[text.index(after: dash)...]
            text = text[..<dash]
        }
        var numbers: [Int] = []
        for part in text.split(separator: ".", omittingEmptySubsequences: false) {
            guard (1...9).contains(part.count), part.allSatisfy({ $0.isASCII && $0.isNumber }), let number = Int(part) else { return nil }
            numbers.append(number)
        }
        while numbers.last == 0 { numbers.removeLast() }
        guard let pre = preRelease else { return ZephyrVersion(numbers: numbers, preRelease: nil) }
        guard !pre.isEmpty, pre.allSatisfy(identifierCharacters) else { return nil }
        var chunks: [ZephyrVersion.Chunk] = []
        var word = ""
        var digits = ""
        func flush() -> Bool {
            if !word.isEmpty { chunks.append(.word(word)); word = "" }
            if !digits.isEmpty {
                guard digits.count <= 9, let number = Int(digits) else { return false }
                chunks.append(.number(number)); digits = ""
            }
            return true
        }
        for character in pre {
            if character.isNumber {
                if !word.isEmpty { chunks.append(.word(word)); word = "" }
                digits.append(character)
            } else if character == "." || character == "-" {
                guard flush() else { return nil }
            } else {
                if !digits.isEmpty { guard flush() else { return nil } }
                word.append(character)
            }
        }
        guard flush(), !chunks.isEmpty else { return nil }
        return ZephyrVersion(numbers: numbers, preRelease: chunks)
    }

    static func isNewer(_ lhs: ZephyrVersion, than rhs: ZephyrVersion) -> Bool {
        compare(lhs, rhs) > 0
    }

    /// -1, 0, or 1.
    static func compare(_ lhs: ZephyrVersion, _ rhs: ZephyrVersion) -> Int {
        if lhs.numbers != rhs.numbers {
            for (left, right) in zip(lhs.numbers, rhs.numbers) where left != right { return left < right ? -1 : 1 }
            return lhs.numbers.count < rhs.numbers.count ? -1 : 1
        }
        switch (lhs.preRelease, rhs.preRelease) {
        case (nil, nil): return 0
        case (nil, _?): return 1
        case (_?, nil): return -1
        case let (left?, right?):
            for (a, b) in zip(left, right) where a != b { return a < b ? -1 : 1 }
            return left.count == right.count ? 0 : (left.count < right.count ? -1 : 1)
        }
    }

    private static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    /// Where the detached helper records a failed update. The helper has no
    /// window and the app has already quit, so this file is the only way to tell
    /// the user why nothing changed.
    static var updateResultFile: URL {
        let library = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library")
        return library.appendingPathComponent("Logs/ZephyrMesh/last-update.txt")
    }

    /// Returns the failure note left by the last update attempt, once, and deletes it.
    static func takeLastResult(from file: URL = updateResultFile) -> String? {
        guard let text = try? String(contentsOf: file, encoding: .utf8) else { return nil }
        try? FileManager.default.removeItem(at: file)
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// The shell script that swaps the app after this process exits.
    ///
    /// Every check runs before anything is moved. If one fails, the installed app
    /// is untouched. If the final move fails, the backup is moved back. In both
    /// cases the script writes the reason to `resultFile`, reopens the old app,
    /// and stops. `openTool` is a parameter only so tests do not launch real apps.
    static func helperScript(root: URL, archive: URL, extract: URL, install: URL, resultFile: URL, openTool: String = "/usr/bin/open") -> String {
        let backup = install.deletingLastPathComponent().appendingPathComponent(install.lastPathComponent + ".previous")
        let launch = launchTarget(for: install)
        return """
        #!/bin/sh
        set -u
        pid="$1"
        result=\(shellQuote(resultFile.path))
        install=\(shellQuote(install.path))
        backup=\(shellQuote(backup.path))
        root=\(shellQuote(root.path))
        mkdir -p "$(dirname "$result")"
        fail() {
            printf '%s\n' "The last update to Zephyr Mesh did not install: $1" > "$result"
            if [ ! -d "$install" ] && [ -d "$backup" ]; then mv "$backup" "$install"; fi
            if [ -d "$install" ]; then \(shellQuote(openTool)) \(shellQuote(launch.path)); fi
            rm -rf "$root"
            exit 1
        }
        while kill -0 "$pid" 2>/dev/null; do sleep 0.25; done
        entries_file="$root/entries.txt"
        /usr/bin/unzip -Z1 \(shellQuote(archive.path)) > "$entries_file" || fail "could not list the archive"
        while IFS= read -r entry; do
            case "$entry" in /*|../*|*/../*|*/..|..|*'\\'*) fail "unsafe path in archive: $entry" ;; esac
        done < "$entries_file"
        /usr/bin/ditto -x -k \(shellQuote(archive.path)) \(shellQuote(extract.path)) || fail "could not extract the archive"
        find \(shellQuote(extract.path)) -type l -print > "$root/symlinks.txt" || fail "could not scan the archive for symlinks"
        [ ! -s "$root/symlinks.txt" ] || fail "the archive contains a symlink"
        find \(shellQuote(extract.path)) -maxdepth 1 -type d ! -path \(shellQuote(extract.path)) ! -name '__MACOSX' -print > "$root/payloads.txt" || fail "could not list the extracted files"
        payload_count=$(/usr/bin/wc -l < "$root/payloads.txt" | /usr/bin/tr -d ' ')
        [ "$payload_count" = "1" ] || fail "the archive must contain exactly one application"
        payload=$(/usr/bin/head -n 1 "$root/payloads.txt")
        case "$payload" in *.app) ;; *) fail "the archive root is not an .app" ;; esac
        [ -x "$payload/Contents/MacOS/ZephyrMeshApp" ] || fail "the app executable is missing"
        /usr/bin/codesign --verify --deep --strict "$payload" || fail "the code signature check failed"
        rm -rf "$backup"
        if [ -d "$install" ]; then mv "$install" "$backup" || fail "could not move the current app aside"; fi
        mv "$payload" "$install" || fail "could not move the new app into place"
        rm -f "$result"
        \(shellQuote(openTool)) \(shellQuote(launch.path))
        rm -rf "$root"
        """
    }

    private static func writeHelper(root: URL, archive: URL, extract: URL, install: URL, resultFile: URL) throws -> URL {
        let helper = root.appendingPathComponent("apply-update.sh")
        let script = helperScript(root: root, archive: archive, extract: extract, install: install, resultFile: resultFile)
        try script.write(to: helper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: helper.path)
        return helper
    }
}
