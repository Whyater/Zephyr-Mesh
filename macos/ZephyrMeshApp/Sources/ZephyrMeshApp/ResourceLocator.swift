import Foundation

/// Locates resources without relying on SwiftPM's generated resource accessor assertion.
enum ResourceLocator {
    static let resourceBundleName = "ZephyrMeshApp_ZephyrMeshApp.bundle"

    /// Returns a resource from the app's generated resource bundle, or a bare resource file.
    ///
    /// `searchRoots` exists for deterministic tests. Production callers should use the default
    /// search roots, which cover app bundles, command-line launches, and SwiftPM test bundles.
    static func url(
        forResource resource: String,
        withExtension pathExtension: String? = nil,
        searchRoots: [URL]? = nil
    ) -> URL? {
        let filename = pathExtension.map { "\(resource).\($0)" } ?? resource
        let roots = searchRoots ?? defaultSearchRoots()
        var visited = Set<String>()

        for root in roots {
            let normalizedRoot = root.standardizedFileURL
            guard visited.insert(normalizedRoot.path).inserted else { continue }

            let resourceDirectories = [
                normalizedRoot.appendingPathComponent(resourceBundleName, isDirectory: true),
                normalizedRoot.appendingPathComponent("Contents/Resources", isDirectory: true)
                    .appendingPathComponent(resourceBundleName, isDirectory: true),
                normalizedRoot.deletingLastPathComponent()
                    .appendingPathComponent("Resources", isDirectory: true)
                    .appendingPathComponent(resourceBundleName, isDirectory: true)
            ]
            for directory in resourceDirectories {
                if let url = existingFile(named: filename, in: directory) {
                    return url
                }
                if let url = existingFile(named: filename, in: directory.appendingPathComponent("Contents/Resources", isDirectory: true)) {
                    return url
                }
            }

            // Some release packaging and `swift run` layouts leave the resource beside the
            // executable rather than inside the generated bundle.
            for directory in [
                normalizedRoot,
                normalizedRoot.appendingPathComponent("Contents/Resources", isDirectory: true),
                normalizedRoot.deletingLastPathComponent()
                    .appendingPathComponent("Resources", isDirectory: true)
            ] {
                if let url = existingFile(named: filename, in: directory) {
                    return url
                }
            }
        }
        return nil
    }

    private static func existingFile(named filename: String, in directory: URL) -> URL? {
        let candidate = directory.appendingPathComponent(filename, isDirectory: false)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            return nil
        }
        return candidate
    }

    private static func defaultSearchRoots() -> [URL] {
        var roots: [URL] = []
        if let resourceURL = Bundle.main.resourceURL {
            roots.append(resourceURL)
        }
        roots.append(Bundle.main.bundleURL)
        if let executableDirectory = Bundle.main.executableURL?.deletingLastPathComponent() {
            roots.append(executableDirectory)
        }

        // SwiftPM puts a target's generated resource bundle next to the test or executable
        // bundle. Looking beside the bundle that contains this type covers `swift test` and
        // `swift run` without asking Bundle for a generated module resource accessor.
        // Skipped when that bundle is the app itself, so a stray resource bundle in the
        // folder that holds the app (for example ~/Applications) is never picked up.
        let locatorBundle = Bundle(for: ResourceLocatorAnchor.self)
        if locatorBundle.bundleURL.standardizedFileURL != Bundle.main.bundleURL.standardizedFileURL {
            roots.append(locatorBundle.bundleURL.deletingLastPathComponent())
        }
        return roots
    }
}

private final class ResourceLocatorAnchor: NSObject {}
