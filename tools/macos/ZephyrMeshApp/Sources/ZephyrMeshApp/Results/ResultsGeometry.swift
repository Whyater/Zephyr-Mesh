import CoreGraphics
import Foundation

// Pure chart math, hit testing, atlas binning and selection movement for the Results window.
// No SwiftUI here, so every rule is unit tested without a view.

/// A 0-based axis with a "nice" top and step. The step is 1, 2, 2.5 or 5 times a power of ten;
/// among the steps that give `minIntervals` to `maxIntervals` intervals, the lowest top wins,
/// and on a tie the larger step (fewer ticks). So 0.389 gets 0.4 in steps of 0.1, 0.606 gets
/// 0.7 in steps of 0.1, and 1.484 gets 1.5 in steps of 0.5.
struct NiceScale: Equatable {
    let maximum: Double
    let step: Double
    let top: Double

    init(maximum: Double, minIntervals: Int = 3, maxIntervals: Int = 7) {
        self.maximum = maximum
        guard maximum.isFinite, maximum > 0, minIntervals > 0, maxIntervals >= minIntervals else {
            step = 1
            top = 1
            return
        }
        let exponent = Int(floor(log10(maximum)))
        var best: (step: Double, top: Double)?
        for power in (exponent - 2)...(exponent + 1) {
            for multiple in [1, 2, 2.5, 5] {
                let step = multiple * pow(10, Double(power))
                let intervals = (maximum / step - 1e-9).rounded(.up)
                guard intervals >= Double(minIntervals), intervals <= Double(maxIntervals) else { continue }
                let top = intervals * step
                if let current = best, top > current.top + 1e-12 || (abs(top - current.top) <= 1e-12 && step <= current.step) {
                    continue
                }
                best = (step, top)
            }
        }
        step = best?.step ?? maximum
        top = best?.top ?? maximum
    }

    /// Tick values from 0 to the top, inclusive.
    var ticks: [Double] {
        let count = Int((top / step).rounded())
        return (0...count).map { Double($0) * step }
    }

    /// Decimals that show every tick exactly (0.5 needs 1, 0.25 needs 2).
    var decimals: Int {
        var places = 0
        while places < 6 {
            let scaled = step * pow(10, Double(places))
            if abs(scaled - scaled.rounded()) < 1e-9 { break }
            places += 1
        }
        return places
    }

    func label(_ value: Double) -> String {
        String(format: "%.\(decimals)f", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}

/// Maps delay (seconds) and error (metres) into a plot rectangle; y grows upward.
struct ChartScale: Equatable {
    let plot: CGRect
    let xMax: Double
    let yMax: Double

    func x(_ seconds: Double) -> CGFloat {
        plot.minX + CGFloat(xMax > 0 ? seconds / xMax : 0) * plot.width
    }

    func y(_ meters: Double) -> CGFloat {
        let fraction = yMax > 0 ? min(max(meters / yMax, 0), 1) : 0
        return plot.maxY - CGFloat(fraction) * plot.height
    }

    func point(_ seconds: Double, _ meters: Double) -> CGPoint { CGPoint(x: x(seconds), y: y(meters)) }

    /// The plot area inside a panel of `size`, leaving room for tick labels on the left and below.
    static func plotRect(in size: CGSize, leading: CGFloat, top: CGFloat, trailing: CGFloat, bottom: CGFloat) -> CGRect {
        CGRect(x: leading, y: top, width: max(size.width - leading - trailing, 1), height: max(size.height - top - bottom, 1))
    }
}

enum ChartGeometry {
    /// The closed interval band: along the high ends left to right, back along the low ends.
    static func bandPolygon(delays: [Double], lows: [Double], highs: [Double], scale: ChartScale) -> [CGPoint] {
        let count = min(delays.count, lows.count, highs.count)
        guard count > 0 else { return [] }
        let upper = (0..<count).map { scale.point(delays[$0], highs[$0]) }
        let lower = (0..<count).reversed().map { scale.point(delays[$0], lows[$0]) }
        return upper + lower
    }

    /// The index of the point nearest `location`, if it lies within `radius`. Ties go to the
    /// earlier point, so the result never depends on dictionary order.
    static func nearest(to location: CGPoint, in points: [CGPoint], radius: CGFloat) -> Int? {
        var best: (index: Int, distance: CGFloat)?
        for (index, point) in points.enumerated() {
            let distance = hypot(point.x - location.x, point.y - location.y)
            guard distance <= radius else { continue }
            if let current = best, distance >= current.distance { continue }
            best = (index, distance)
        }
        return best?.index
    }
}

extension ChartGeometry {
    /// True when the segment from `a` to `b` touches `rect` (Liang-Barsky clipping).
    static func segment(_ a: CGPoint, _ b: CGPoint, intersects rect: CGRect) -> Bool {
        var low: CGFloat = 0, high: CGFloat = 1
        let dx = b.x - a.x, dy = b.y - a.y
        let checks: [(CGFloat, CGFloat)] = [(-dx, a.x - rect.minX), (dx, rect.maxX - a.x), (-dy, a.y - rect.minY), (dy, rect.maxY - a.y)]
        for (p, q) in checks {
            if p == 0 {
                if q < 0 { return false }
            } else {
                let t = q / p
                if p < 0 { low = max(low, t) } else { high = min(high, t) }
                if low > high { return false }
            }
        }
        return true
    }

    /// Where the failure-line caption goes, the same rule in every panel. The four spots are tried in
    /// this order: right end above the line, right end below, left end above, left end below. The first
    /// one inside the plot whose box (grown by `clearance`) touches no drawn segment wins. When all four
    /// collide, the one with the least overlap wins: a spot outside the plot counts as worse than any
    /// inside it, then fewer touched segments is better, and the earlier spot wins a tie.
    static func captionOrigin(size: CGSize, lineY: CGFloat, plot: CGRect, gap: CGFloat, clearance: CGFloat,
                              segments: [(CGPoint, CGPoint)]) -> CGPoint {
        let left = plot.minX + gap, right = plot.maxX - gap - size.width
        let above = lineY - gap - size.height, below = lineY + gap
        let candidates = [CGPoint(x: right, y: above), CGPoint(x: right, y: below), CGPoint(x: left, y: above),
                          CGPoint(x: left, y: below)]
        func overlap(_ origin: CGPoint) -> Int {
            let box = CGRect(origin: origin, size: size)
            let grown = box.insetBy(dx: -clearance, dy: -clearance)
            let hits = segments.filter { segment($0.0, $0.1, intersects: grown) }.count
            return (plot.contains(box) ? 0 : segments.count + 1) + hits
        }
        var best = (origin: candidates[0], overlap: Int.max)
        for origin in candidates {
            let score = overlap(origin)
            if score == 0 { return origin }
            if score < best.overlap { best = (origin, score) }
        }
        return best.origin
    }

    /// Consecutive pairs of a polyline, for collision checks.
    static func segments(_ points: [CGPoint]) -> [(CGPoint, CGPoint)] {
        zip(points, points.dropFirst()).map { ($0, $1) }
    }
}

/// A selected point in the delay charts: which panel, which tracker, which delay.
struct ChartSelection: Equatable {
    var panel: Int
    var arm: Int
    var delayIndex: Int

    enum Move { case left, right, up, down, nextPanel, previousPanel }

    /// One step in a grid of `panels` panels, `arms` trackers and `delays` points. Every move
    /// stops at the edge instead of wrapping. Up picks the upper tracker in the legend order
    /// (index 0), down the lower. Returns nil when the move cannot happen.
    func moved(_ move: Move, panels: Int, arms: Int, delays: Int) -> ChartSelection? {
        var next = self
        switch move {
        case .left: next.delayIndex -= 1
        case .right: next.delayIndex += 1
        case .up: next.arm -= 1
        case .down: next.arm += 1
        case .nextPanel: next.panel += 1
        case .previousPanel: next.panel -= 1
        }
        guard (0..<panels).contains(next.panel), (0..<arms).contains(next.arm), (0..<delays).contains(next.delayIndex) else {
            return nil
        }
        return next
    }
}

/// A selected cell in the atlas matrix: a row (path) and a column (tracker).
struct AtlasSelection: Equatable {
    var row: Int
    var column: Int

    enum Move { case left, right, up, down }

    /// One step inside a `rows` by `columns` grid; stops at the edges.
    func moved(_ move: Move, rows: Int, columns: Int) -> AtlasSelection {
        var next = self
        switch move {
        case .left: next.column -= 1
        case .right: next.column += 1
        case .up: next.row -= 1
        case .down: next.row += 1
        }
        return next.clamped(rows: rows, columns: columns)
    }

    /// The same cell, or the nearest one inside a `rows` by `columns` grid.
    func clamped(rows: Int, columns: Int) -> AtlasSelection {
        AtlasSelection(row: min(max(row, 0), max(rows - 1, 0)), column: min(max(column, 0), max(columns - 1, 0)))
    }
}

/// Matrix spacing and column widths: cells shrink from `atlasCellWidth` to `atlasCellMinWidth` so the row
/// header, every tracker and the "Most delay" column fit before the matrix scrolls sideways.
enum AtlasGeometry {
    /// Grid spacing between columns and the padding kept around the grid for the selection ring.
    /// Wide enough for a selection ring 2 pt outside a cell with a 2 pt gap.
    static let columnSpacing = Space.sm
    static let outerPadding = Space.xs
    static let mostDelayLeading = Space.sm

    static func cellWidth(available: CGFloat, arms: Int, mostDelayWidth: CGFloat) -> CGFloat {
        guard arms > 0 else { return LayoutMetrics.atlasCellWidth }
        let fixed = LayoutMetrics.atlasRowHeaderWidth + columnSpacing * CGFloat(arms + 1) + mostDelayLeading + mostDelayWidth
            + outerPadding * 2
        let fit = ((available - fixed) / CGFloat(arms)).rounded(.down)
        return min(max(fit, LayoutMetrics.atlasCellMinWidth), LayoutMetrics.atlasCellWidth)
    }
}

/// Which accent ramp stop a matrix cell is filled with. Fails-at-zero and missing entries are
/// not tinted (`nil`): they use the surface fill.
enum AtlasBin {
    static func stop(for entry: AtlasEntry?) -> Int? {
        guard let entry else { return nil }
        switch entry.kind {
        case .censored: return 4
        case .failsAtZero: return nil
        case .value:
            guard let seconds = entry.valueS else { return nil }
            return stop(milliseconds: ResultsFormat.milliseconds(seconds))
        }
    }

    /// 0 to 25 ms stop 1, 50 to 100 ms stop 2, 150 ms and above stop 3.
    static func stop(milliseconds: Int) -> Int {
        switch milliseconds {
        case ..<26: return 1
        case ..<101: return 2
        default: return 3
        }
    }
}
