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
    case installLocationUnavailable
    case updateProcessUnavailable
    case invalidVersion(String)
    case untrustedDownloadURL(String)
    case archiveTooLarge(Int64)
    case alreadyUpToDate(String)
    case stagingFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidReleaseJSON: return "GitHub release JSON is invalid"
        case .network(let message): return "GitHub request failed: \(message)"
        case .httpStatus(let status): return "GitHub returned HTTP \(status)"
        case .noAsset(let platform): return "No release ZIP asset for \(platform)"
        case .missingDigest(let name): return "Release asset has no SHA-256 digest: \(name)"
        case .unsafeFilename: return "Release filename must be a plain file name"
        case .digestMismatch(let expected, let actual): return "Release digest mismatch: expected \(expected), got \(actual)"
        case .notAnApplicationBundle: return "Updates require a Finder-launched Zephyr Mesh.app bundle"
        case .installLocationUnavailable: return "The app folder is read-only"
        case .updateProcessUnavailable: return "Could not start the update helper"
        case .invalidVersion(let value): return "Unrecognized version: \(value)"
        case .untrustedDownloadURL(let url): return "Download URL is not an official release asset of \(ReleaseUpdater.repository): \(url)"
        case .archiveTooLarge(let limit): return "The update download is larger than \(limit / 1_048_576) MB, so it was stopped"
        case .alreadyUpToDate(let tag): return "Zephyr Mesh is already up to date (\(tag))"
        case .stagingFailed(let reason): return "Could not save the update: \(reason)"
        }
    }
}

enum ReleaseUpdater {
    static let repository = "Whyater/Zephyr-Mesh"
    static let currentVersion: String = {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "0.2.3-dev"
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
        try verify(sha256: hex(SHA256.hash(data: data)), against: asset)
    }

    /// Compares an already computed lowercase SHA-256 hex string with the asset's digest.
    /// Returns the expected digest so callers can hand it to the install helper.
    @discardableResult
    static func verify(sha256 actual: String, against asset: ZephyrReleaseAsset) throws -> String {
        guard let digest = asset.digest else { throw ReleaseUpdaterError.missingDigest(asset.name) }
        let expected = normalizedDigest(digest)
        guard expected.count == 64, expected.allSatisfy({ $0.isHexDigit }) else { throw ReleaseUpdaterError.digestMismatch(expected: expected, actual: "invalid SHA-256") }
        guard actual == expected else { throw ReleaseUpdaterError.digestMismatch(expected: expected, actual: actual) }
        return expected
    }

    private static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    /// The largest update archive the app will download. Real releases are a few MB.
    static let maximumArchiveBytes: Int64 = 512 * 1_048_576
    /// Limits on what an archive may unpack to, checked from its directory before
    /// extracting. The same numbers as `MAX_EXTRACTED_BYTES` and
    /// `MAX_ARCHIVE_MEMBERS` in `tools/release_updater.py`.
    static let maximumExtractedBytes: Int64 = 2 * 1024 * 1_048_576
    static let maximumArchiveEntries = 50_000

    /// Streams a download into `destination` (never holding the whole file in memory),
    /// hashing it on the way, and stops once more than `maximumBytes` arrive.
    /// Returns the lowercase SHA-256 hex digest of what was written. A partial file is
    /// removed on every failure.
    static func download(_ request: URLRequest, to destination: URL, maximumBytes: Int64 = maximumArchiveBytes,
                         session: URLSession = .shared) async throws -> String {
        let bytes: URLSession.AsyncBytes
        let response: URLResponse
        do { (bytes, response) = try await session.bytes(for: request) }
        catch { throw ReleaseUpdaterError.network(error.localizedDescription) }
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw ReleaseUpdaterError.httpStatus(http.statusCode)
        }
        if response.expectedContentLength > maximumBytes { throw ReleaseUpdaterError.archiveTooLarge(maximumBytes) }
        guard FileManager.default.createFile(atPath: destination.path, contents: nil) else {
            throw ReleaseUpdaterError.stagingFailed("could not create \(destination.lastPathComponent)")
        }
        do {
            let handle = try FileHandle(forWritingTo: destination)
            defer { try? handle.close() }
            var hasher = SHA256()
            var buffer = Data()
            let chunk = 1_048_576
            buffer.reserveCapacity(chunk)
            var total: Int64 = 0
            func flush() throws {
                total += Int64(buffer.count)
                guard total <= maximumBytes else { throw ReleaseUpdaterError.archiveTooLarge(maximumBytes) }
                hasher.update(data: buffer)
                try handle.write(contentsOf: buffer)
                buffer.removeAll(keepingCapacity: true)
            }
            do {
                for try await byte in bytes {
                    buffer.append(byte)
                    if buffer.count >= chunk { try flush() }
                }
            } catch let error as ReleaseUpdaterError {
                throw error
            } catch {
                throw ReleaseUpdaterError.network(error.localizedDescription)
            }
            try flush()
            return hex(hasher.finalize())
        } catch {
            try? FileManager.default.removeItem(at: destination)
            if error is ReleaseUpdaterError { throw error }
            throw ReleaseUpdaterError.stagingFailed(error.localizedDescription)
        }
    }

    static func checkLatest(currentVersion: String = Self.currentVersion) async throws -> ZephyrUpdateCheck {
        guard let url = URL(string: "https://api.github.com/repos/\(repository)/releases/latest") else {
            throw ReleaseUpdaterError.network("invalid GitHub URL")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 20
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
        let installParent = currentBundle.deletingLastPathComponent()
        guard FileManager.default.isWritableFile(atPath: installParent.path) else {
            throw ReleaseUpdaterError.installLocationUnavailable
        }
        let check = try await checkLatest(currentVersion: currentVersion)
        // Returning normally tells the caller to quit for the swap, so "nothing to
        // install" must be an error, never a tag.
        guard check.isNewer else { throw ReleaseUpdaterError.alreadyUpToDate(check.release.tagName) }
        guard isOfficialDownloadURL(check.asset.browserDownloadURL) else { throw ReleaseUpdaterError.untrustedDownloadURL(check.asset.browserDownloadURL) }
        guard let downloadURL = URL(string: check.asset.browserDownloadURL) else { throw ReleaseUpdaterError.network("invalid release asset URL") }
        let filename = URL(fileURLWithPath: check.asset.name).lastPathComponent
        guard filename == check.asset.name, !filename.isEmpty else { throw ReleaseUpdaterError.unsafeFilename }
        var request = URLRequest(url: downloadURL)
        request.timeoutInterval = 20
        request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")
        request.setValue("Zephyr-Mesh-Updater/0.1", forHTTPHeaderField: "User-Agent")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("zephyr-update-\(UUID().uuidString)", isDirectory: true)
        let archive = root.appendingPathComponent(filename)
        let extract = root.appendingPathComponent("payload", isDirectory: true)
        do {
            do { try FileManager.default.createDirectory(at: extract, withIntermediateDirectories: true) }
            catch { throw ReleaseUpdaterError.stagingFailed(error.localizedDescription) }
            let actual = try await download(request, to: archive)
            let expected = try verify(sha256: actual, against: check.asset)
            let helper: URL
            do { helper = try writeHelper(root: root, archive: archive, extract: extract, install: currentBundle, resultFile: updateResultFile, archiveSHA256: expected) }
            catch { throw ReleaseUpdaterError.stagingFailed(error.localizedDescription) }
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/sh")
            process.arguments = [helper.path, String(ProcessInfo.processInfo.processIdentifier)]
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            do { try process.run() } catch { throw ReleaseUpdaterError.updateProcessUnavailable }
        } catch {
            // The helper deletes `root` itself once it runs; before that, this process owns it.
            try? FileManager.default.removeItem(at: root)
            throw error
        }
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
    /// Every check runs before anything is moved: the archive digest is checked
    /// again, entries are screened, the bundle signature is verified, and the new
    /// app must pass `--self-test`. If one fails, the installed app is untouched.
    /// The current app is then moved to a backup in
    /// `~/Library/Caches/org.zephyrmesh.cockpit/`. If moving the new app into
    /// place fails, even halfway, whatever landed at `install` is deleted and the
    /// backup is moved back. The new app is then opened and watched for
    /// `startupCheckSeconds`. If it crashes in that time (a new
    /// `ZephyrMeshApp*.ips` crash report appears) or never shows up, and it is
    /// still not running at that moment, the old app is put back. Quitting it
    /// normally counts as a successful start. Every failure writes the reason to
    /// `resultFile` and reopens the old app.
    ///
    /// `openTool` and `startupCheckSeconds` are parameters only so tests do not
    /// launch real apps (0 skips the running check). `archiveSHA256` is always set
    /// by `installLatest`; nil skips the second digest check.
    static func helperScript(root: URL, archive: URL, extract: URL, install: URL, resultFile: URL,
                             archiveSHA256: String? = nil, openTool: String = "/usr/bin/open",
                             startupCheckSeconds: Int = 5) -> String {
        let launch = launchTarget(for: install)
        return """
        #!/bin/sh
        set -u
        pid="$1"
        result=\(shellQuote(resultFile.path))
        install=\(shellQuote(install.path))
        launch=\(shellQuote(launch.path))
        root=\(shellQuote(root.path))
        archive=\(shellQuote(archive.path))
        extract=\(shellQuote(extract.path))
        open_tool=\(shellQuote(openTool))
        expected_sha=\(shellQuote(archiveSHA256 ?? ""))
        startup_wait=\(max(0, startupCheckSeconds))
        cache="$HOME/Library/Caches/org.zephyrmesh.cockpit"
        backup="$cache/ZephyrMesh.app.previous.$$"
        moved_aside=0
        mkdir -p "$(dirname "$result")"
        # Only after this script moved the old app aside: whatever is at $install
        # is then the new or half-copied app, so it is safe to delete.
        restore_old() {
            [ "$moved_aside" = "1" ] || return 0
            [ -d "$backup" ] || return 1
            if [ -e "$install" ] || [ -L "$install" ]; then rm -rf "$install" || return 1; fi
            mv "$backup" "$install" || return 1
            moved_aside=0
        }
        fail() {
            if restore_old; then
                printf '%s\\n' "The update could not start, so the current version was kept. $1" > "$result"
                if [ -d "$install" ]; then "$open_tool" "$launch"; fi
                rm -rf "$root"
            else
                printf '%s\\n' "The update failed and the previous version could not be put back: $1 The previous version is saved at $backup" > "$result"
            fi
            exit 1
        }
        waited=0
        while kill -0 "$pid" 2>/dev/null; do
            if [ "$waited" -ge 480 ]; then
                # The old app never quit. It is still running, so do not open a second copy.
                printf '%s\\n' "The update could not start, so the current version was kept. Zephyr Mesh did not quit within 2 minutes." > "$result"
                rm -rf "$root"
                exit 1
            fi
            sleep 0.25
            waited=$((waited + 1))
        done
        if [ -n "$expected_sha" ]; then
            actual_sha=$(/usr/bin/shasum -a 256 "$archive" 2>/dev/null | /usr/bin/cut -d ' ' -f 1)
            [ "$actual_sha" = "$expected_sha" ] || fail "the downloaded archive changed after it was checked"
        fi
        entries_file="$root/entries.txt"
        /usr/bin/unzip -Z1 "$archive" > "$entries_file" || fail "could not list the archive"
        # Totals line: "<entries> files, <bytes> bytes uncompressed, ...". Checked before
        # anything is written, so a small archive cannot unpack into a full disk.
        totals=$(/usr/bin/unzip -Zt "$archive" 2>/dev/null) || fail "could not read the archive size"
        entry_count=$(printf '%s\n' "$totals" | /usr/bin/awk '{ print $1 }')
        unpacked=$(printf '%s\n' "$totals" | /usr/bin/awk '{ print $3 }')
        case "$entry_count" in ''|*[!0-9]*) fail "could not read the archive size" ;; esac
        case "$unpacked" in ''|*[!0-9]*) fail "could not read the archive size" ;; esac
        [ "$entry_count" -le \(maximumArchiveEntries) ] || fail "the archive has more than \(maximumArchiveEntries) entries"
        [ "$unpacked" -le \(maximumExtractedBytes) ] || fail "the archive expands to more than \(maximumExtractedBytes / 1_073_741_824) GB"
        while IFS= read -r entry; do
            case "$entry" in /*|../*|*/../*|*/..|..|*'\\'*) fail "unsafe path in archive: $entry" ;; esac
        done < "$entries_file"
        /usr/bin/ditto -x -k "$archive" "$extract" || fail "could not extract the archive"
        find "$extract" -type l -print > "$root/symlinks.txt" || fail "could not scan the archive for symlinks"
        [ ! -s "$root/symlinks.txt" ] || fail "the archive contains a symlink"
        find "$extract" -maxdepth 1 -type d ! -path "$extract" ! -name '__MACOSX' -print > "$root/payloads.txt" || fail "could not list the extracted files"
        payload_count=$(/usr/bin/wc -l < "$root/payloads.txt" | /usr/bin/tr -d ' ')
        [ "$payload_count" = "1" ] || fail "the archive must contain exactly one application"
        payload=$(/usr/bin/head -n 1 "$root/payloads.txt")
        case "$payload" in *.app) ;; *) fail "the archive root is not an .app" ;; esac
        [ -x "$payload/Contents/MacOS/ZephyrMeshApp" ] || fail "the app executable is missing"
        /usr/bin/codesign --verify --deep --strict "$payload" || fail "the code signature check failed"
        "$payload/Contents/MacOS/ZephyrMeshApp" --self-test >"$root/self-test.log" 2>&1 &
        selftest_pid=$!
        selftest_elapsed=0
        while kill -0 "$selftest_pid" 2>/dev/null; do
            if [ "$selftest_elapsed" -ge 20 ]; then
                kill -9 "$selftest_pid" 2>/dev/null || true
                wait "$selftest_pid" 2>/dev/null || true
                fail "self-test timed out after 20 seconds"
            fi
            sleep 1
            selftest_elapsed=$((selftest_elapsed + 1))
        done
        wait "$selftest_pid"
        selftest_status=$?
        if [ "$selftest_status" -ne 0 ]; then
            reason=$(/usr/bin/tr '\\n' ' ' < "$root/self-test.log" | /usr/bin/cut -c 1-400)
            # The app prints "self-test failed: <reason>"; say "self-test failed" once.
            reason=${reason#self-test failed: }
            [ -n "$reason" ] || reason="exit status $selftest_status"
            fail "self-test failed: $reason"
        fi
        mkdir -p "$cache" || fail "could not create the update backup folder"
        rm -rf "$backup"
        if [ -e "$install" ]; then
            if [ "$(/usr/bin/stat -f %d "$cache")" = "$(/usr/bin/stat -f %d "$(dirname "$install")")" ]; then
                # Same disk: a rename, which either fully happens or does not happen.
                mv "$install" "$backup" || fail "could not move the current app aside"
                moved_aside=1
            else
                # Different disks: copy first so a failure leaves the current app whole.
                /usr/bin/ditto "$install" "$backup" || { rm -rf "$backup"; fail "could not copy the current app aside"; }
                moved_aside=1
                rm -rf "$install" || fail "could not remove the current app"
            fi
        fi
        mv "$payload" "$install" || fail "could not move the new app into place"
        rm -f "$result"
        crash_dir="$HOME/Library/Logs/DiagnosticReports"
        : > "$root/launch-marker"
        "$open_tool" "$launch" || fail "the updated app could not be opened"
        if [ "$startup_wait" -gt 0 ]; then
            exe="$(cd "$install/Contents/MacOS" 2>/dev/null && pwd -P)/ZephyrMeshApp"
            app_running() { ps -axo comm= 2>/dev/null | /usr/bin/grep -Fxq -- "$exe"; }
            new_crash() { /usr/bin/find "$crash_dir" -maxdepth 1 -name 'ZephyrMeshApp*.ips' -newer "$root/launch-marker" 2>/dev/null | /usr/bin/grep -q .; }
            # Roll back only if the new app is not running right now, so a slow start
            # is never deleted from under itself.
            launch_failed() { app_running && return 0; fail "$1"; }
            seen=0
            looked=0
            while [ "$looked" -lt $((startup_wait * 3)) ]; do
                if app_running; then seen=1; break; fi
                new_crash && break
                sleep 1
                looked=$((looked + 1))
            done
            if [ "$seen" = "1" ]; then
                alive=0
                while [ "$alive" -lt "$startup_wait" ] && app_running; do
                    sleep 1
                    alive=$((alive + 1))
                done
            fi
            if ! app_running; then
                # Give ReportCrash a moment to write its report before deciding.
                settle=2
                [ "$startup_wait" -lt 2 ] && settle="$startup_wait"
                sleep "$settle"
                if new_crash; then
                    launch_failed "the updated app crashed when it started"
                elif [ "$seen" != "1" ]; then
                    launch_failed "the updated app did not start"
                fi
                # Seen running, then gone with no crash report: it was quit normally.
            fi
        fi
        moved_aside=0
        rm -rf "$cache"/ZephyrMesh.app.previous.*
        rm -rf "$root"
        """
    }

    private static func writeHelper(root: URL, archive: URL, extract: URL, install: URL, resultFile: URL, archiveSHA256: String) throws -> URL {
        let helper = root.appendingPathComponent("apply-update.sh")
        let script = helperScript(root: root, archive: archive, extract: extract, install: install, resultFile: resultFile, archiveSHA256: archiveSHA256)
        try script.write(to: helper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: helper.path)
        return helper
    }
}
