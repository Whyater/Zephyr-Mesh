import SwiftUI

// Display formatting for values that come from the fixture's embedded Python presentation.
// Nothing here derives a metric: functions only round for display, pick words and shapes
// from the generated UI text, and arrange presentation records for the views.

enum Format {
    static let notRecorded = UIText.notRecorded

    static func number(_ value: Double, decimals: Int) -> String {
        value.formatted(.number.precision(.fractionLength(decimals)).grouping(.never))
    }

    static func seconds(_ value: Double?) -> String {
        value.map { "\(number($0, decimals: 2)) s" } ?? notRecorded
    }

    static func meters(_ value: Double?) -> String {
        value.map { "\(number($0, decimals: 2)) m" } ?? notRecorded
    }

    static func milliseconds(_ value: Double?) -> String {
        value.map { "\(number($0, decimals: 0)) ms" } ?? notRecorded
    }

    static func percent(_ value: Double?) -> String {
        value.map { "\(number($0, decimals: 1))%" } ?? notRecorded
    }

    static func metersPerSecond(_ value: Double?) -> String {
        value.map { "\(number($0, decimals: 2)) m/s" } ?? notRecorded
    }

    static func degrees(_ value: Double?) -> String {
        value.map { "\(number($0, decimals: 0))\u{00B0}" } ?? notRecorded
    }

    /// A recorded count, or "Not recorded" when the run did not carry it (never 0).
    static func integer(_ value: Int?) -> String {
        value.map { "\($0)" } ?? notRecorded
    }

    /// A group age: one value when every member agrees, otherwise a min to max range.
    static func seconds(_ metric: PresentationMetric?) -> String {
        switch metric {
        case .none: return notRecorded
        case .value(let value): return seconds(value)
        case .range(let low, let high): return secondsRange(newest: low, oldest: high)
        }
    }

    /// Fills a shared UI text template such as "Showing {shown} of {total}".
    static func fill(_ template: String, _ values: [String: String]) -> String {
        values.reduce(template) { $0.replacingOccurrences(of: "{\($1.key)}", with: $1.value) }
    }

    static func count(_ value: Int, noun: String) -> String {
        "\(value) \(noun)\(value == 1 ? "" : "s")"
    }

    /// A group metric is one value when every member agrees, otherwise a min to max range.
    static func percent(_ metric: PresentationMetric?) -> String {
        switch metric {
        case .none: return notRecorded
        case .value(let value): return percent(value)
        case .range(let low, let high): return "\(number(low, decimals: 0))\u{2013}\(number(high, decimals: 0))%"
        }
    }

    static func secondsRange(newest: Double?, oldest: Double?) -> String {
        guard let newest, let oldest else { return notRecorded }
        if newest == oldest { return seconds(newest) }
        return "\(number(newest, decimals: 2))\u{2013}\(number(oldest, decimals: 2)) s"
    }

    /// "6.00, 0.00, 1.50" for x, y, z. The unit lives in the label, for example "Position (m)".
    static func vector(_ value: SIMD3<Float>?) -> String {
        guard let value else { return notRecorded }
        return [value.x, value.y, value.z].map { number(Double($0), decimals: 2) }.joined(separator: ", ")
    }

    static func time(_ value: Double) -> String { number(value, decimals: 2) }

    /// Plain label for a data key: the shared UI text label when one exists, otherwise the key
    /// with underscores turned into spaces, so no raw snake_case reaches the screen.
    static func label(_ key: String) -> String {
        if let label = UIText.fieldLabels[key] { return label }
        let words = key.replacingOccurrences(of: "_", with: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }
}

/// Shape, word and color for one status level, read from the generated UI text.
struct StatusStyle: Equatable {
    let level: String
    let word: String
    let symbol: String
    let colorToken: String

    init(level: String, word: String? = nil, shape: String? = nil, color: String? = nil) {
        let spec = UIText.statusLevels[level] ?? UIText.statusLevels["unavailable"]
        self.level = level
        self.word = word ?? spec?.word ?? Format.notRecorded
        let shapeName = shape ?? spec?.shape ?? "dash"
        self.symbol = UIText.statusShapesMacOS[shapeName] ?? "minus"
        self.colorToken = color ?? spec?.color ?? "textSecondary"
    }

    init(drone: PresentationDrone) {
        self.init(level: drone.level, word: drone.word, shape: drone.shape, color: drone.color)
    }

    init(group: PresentationGroup) {
        self.init(level: group.level, word: group.word, shape: group.shape, color: group.color)
    }

    /// Late and lost drones carry a status marker in the scene (their shape in the alert colors),
    /// so status never depends on opacity alone.
    var hasSceneMarker: Bool { level == "caution" || level == "alert" }

    func color(in palette: Theme.Palette) -> Color {
        switch colorToken {
        case "ok": return palette.ok
        case "caution": return palette.caution
        case "alert": return palette.alert
        case "info": return palette.info
        case "accent": return palette.accent
        default: return palette.textSecondary
        }
    }
}

enum TeamPresentation {
    /// Levels that stay listed whatever the filter says (late and lost), from the shared UI text.
    static let pinnedLevels = Set(UIText.TeamFilter.pinnedLevels)

    /// Names a drone matches in the filter: raw ID, compact name, and heading name.
    static func matches(_ drone: PresentationDrone, query: String) -> Bool {
        let needle = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !needle.isEmpty else { return true }
        return [drone.droneID, drone.compactName, drone.headingName].contains { $0.lowercased().contains(needle) }
    }

    struct Row: Identifiable, Equatable {
        let group: PresentationGroup
        let members: [PresentationDrone]
        let isFilteredMatch: Bool
        /// The selected drone, when it is listed only because it is selected (it does not match).
        var keptSelectionID: String?
        var id: String { group.id }
    }

    /// Mirrors `filter_group_rows` in desktop/presentation.py: a pinned (late or lost) group is
    /// never hidden and keeps every member; any other group stays only when a member matches,
    /// and then lists only the matching members. The selected drone is the one addition (macOS
    /// only so far): it stays listed in its group, marked, even when it does not match, so the
    /// selection the scene and inspector show always has a row and arrow keys start from it.
    static func rows(groups: [PresentationGroup], drones: [PresentationDrone], query: String,
                     keeping selectedID: String? = nil) -> [Row] {
        let byID = Dictionary(uniqueKeysWithValues: drones.map { ($0.droneID, $0) })
        let filtering = !query.trimmingCharacters(in: .whitespaces).isEmpty
        return groups.compactMap { group in
            let members = group.droneIDs.compactMap { byID[$0] }
            guard !pinnedLevels.contains(group.level), filtering else {
                return Row(group: group, members: members, isFilteredMatch: false)
            }
            let keep = selectedID.flatMap { id in group.droneIDs.contains(id) ? id : nil }
            let listed = members.filter { matches($0, query: query) || $0.droneID == keep }
            guard !listed.isEmpty else { return nil }
            let kept = keep.flatMap { id in listed.contains { $0.droneID == id && !matches($0, query: query) } ? id : nil }
            return Row(group: group, members: listed, isFilteredMatch: true, keptSelectionID: kept)
        }
    }

    /// Counts for the filter caption and empty state. Mirrors `team_filter` in
    /// desktop/presentation.py: `shown` counts listed drones (pinned ones included), `matched`
    /// counts drones whose names match, `total` counts every drone in the frame.
    struct FilterCounts: Equatable {
        let shown: Int
        let matched: Int
        let total: Int
        let caption: String?
        let emptyText: String?
    }

    static func filterCounts(rows: [Row], drones: [PresentationDrone], query: String) -> FilterCounts {
        let needle = query.trimmingCharacters(in: .whitespaces)
        // Counts every listed row, including a selected drone kept only because it is selected,
        // so the caption always matches what the list shows.
        let shown = rows.reduce(0) { $0 + $1.members.count }
        guard !needle.isEmpty else {
            return FilterCounts(shown: shown, matched: drones.count, total: drones.count, caption: nil, emptyText: nil)
        }
        let matched = drones.filter { matches($0, query: needle) }.count
        let caption = Format.fill(UIText.TeamFilter.showing, ["shown": "\(shown)", "total": "\(drones.count)"])
        let empty = matched == 0 ? Format.fill(UIText.TeamFilter.noMatches, ["query": needle]) : nil
        return FilterCounts(shown: shown, matched: matched, total: drones.count, caption: caption, emptyText: empty)
    }
}

extension TeamPresentation {
    /// The line under the filter field: "Showing N of M", plus the pinned note when late or
    /// lost drones are listed even though they do not match. Nil when not filtering.
    static func filterCaption(counts: FilterCounts, rows: [Row]) -> String? {
        guard let caption = counts.caption else { return nil }
        let pinnedListed = rows.contains { !$0.isFilteredMatch && pinnedLevels.contains($0.group.level) }
        return pinnedListed ? "\(caption). \(UIText.TeamFilter.pinnedNote)" : caption
    }
}

/// Shorter forms of an event text for narrow rows. The presentation already caps a drone list
/// at `EventDroneList.limit` names plus "+N more"; when even that does not fit, each shorter
/// variant keeps one name fewer and counts the rest, so a row is never cut inside a name.
enum EventText {
    private static let namePattern: String = {
        let prefix = UIText.droneCompactName.components(separatedBy: "{").first ?? ""
        return NSRegularExpression.escapedPattern(for: prefix) + "\\d{2,}"
    }()

    private static let morePattern: String = {
        let parts = UIText.EventDroneList.more.components(separatedBy: "{count}")
        return parts.map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "(\\d+)")
    }()

    private static let nameExpression = try? NSRegularExpression(pattern: namePattern)

    private static let listExpression: NSRegularExpression? = {
        let name = namePattern
        let pattern = "\\b\(name)(?:, \(name))+\\b(?: \(morePattern))?"
        return try? NSRegularExpression(pattern: pattern)
    }()

    /// The full text first, then one variant per name dropped, down to a single name.
    static func variants(_ text: String) -> [String] {
        let whole = NSRange(text.startIndex..., in: text)
        guard let expression = listExpression, let match = expression.firstMatch(in: text, range: whole),
              let range = Range(match.range, in: text) else { return [text] }
        let listText = String(text[range])
        var extra = 0
        if match.numberOfRanges > 1, let extraRange = Range(match.range(at: 1), in: text) {
            extra = Int(text[extraRange]) ?? 0
        }
        let names = nameExpression?
            .matches(in: listText, range: NSRange(listText.startIndex..., in: listText))
            .compactMap { Range($0.range, in: listText).map { String(listText[$0]) } } ?? []
        guard names.count > 1 else { return [text] }
        var result = [text]
        for keep in stride(from: names.count - 1, through: 1, by: -1) {
            let more = Format.fill(UIText.EventDroneList.more, ["count": "\(names.count - keep + extra)"])
            let list = names.prefix(keep).joined(separator: ", ") + " " + more
            result.append(text.replacingCharacters(in: range, with: list))
        }
        return result
    }
}

// MARK: - Palette environment

private struct PaletteKey: EnvironmentKey {
    static let defaultValue = Theme.dark
}

extension EnvironmentValues {
    var palette: Theme.Palette {
        get { self[PaletteKey.self] }
        set { self[PaletteKey.self] = newValue }
    }
}

/// Resolves the token palette from the window's effective appearance and publishes it.
struct ThemedRoot<Content: View>: View {
    @Environment(\.colorScheme) private var colorScheme
    @ViewBuilder let content: () -> Content

    var body: some View {
        let palette = Theme.palette(for: colorScheme)
        content()
            .environment(\.palette, palette)
            .tint(palette.accent)
            .foregroundStyle(palette.text)
    }
}

