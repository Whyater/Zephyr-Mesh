import Foundation

// Pure text for the Results window. Both apps must produce the same strings: metres "%.3f m",
// delays as whole milliseconds rounded half to even (Python round), counts "x of N", loss
// titles from the fixture. Nothing here derives a statistic.

enum ResultsFormat {
    static let notComputed = UIText.Results.notComputed

    /// "0.158" with three decimals; negative zero prints as "0.000".
    static func metersValue(_ value: Double) -> String {
        let text = String(format: "%.3f", locale: Locale(identifier: "en_US_POSIX"), value)
        return text == "-0.000" ? "0.000" : text
    }

    static func meters(_ value: Double) -> String { "\(metersValue(value)) m" }

    /// Whole milliseconds, rounded like Python's round(seconds * 1000).
    static func milliseconds(_ seconds: Double) -> Int {
        Int((seconds * 1000).rounded(.toNearestOrEven))
    }

    static func delay(_ seconds: Double) -> String { "\(milliseconds(seconds)) ms" }

    /// "0.149 to 0.170 m"
    static func meterRange(low: Double, high: Double) -> String {
        Format.fill(UIText.Results.rangeTemplate, ["low": metersValue(low), "high": metersValue(high)])
    }

    /// "0 of 50"
    static func count(_ count: Int, of total: Int) -> String {
        Format.fill(UIText.Results.seedsOf, ["count": "\(count)", "total": "\(total)"])
    }

    /// Whole percent as Python's int(p * 100), so 0.3 reads 30.
    static func percent(_ probability: Double) -> Int { Int(probability * 100) }

    /// The fixture's loss-condition key, for example "independent:0" or "burst:0.3".
    static func lossKey(model: String, probability: Double) -> String {
        "\(model):\(String(format: "%g", locale: Locale(identifier: "en_US_POSIX"), probability))"
    }

    /// The loss title from the fixture, for example "30 % random"; "Not computed" if absent.
    static func lossTitle(model: String, probability: Double, titles: [String: String]) -> String {
        titles[lossKey(model: model, probability: probability)] ?? notComputed
    }

    /// Delay panel title: "No loss" or "30 % loss".
    static func panelTitle(probability: Double) -> String {
        probability == 0 ? UIText.Results.panelNoLoss
            : Format.fill(UIText.Results.panelLoss, ["percent": "\(percent(probability))"])
    }

    /// A write-up sentence for display: markdown code backticks removed. The fixture keeps the
    /// verbatim text; only what is drawn changes.
    static func displayText(_ text: String) -> String {
        text.replacingOccurrences(of: "`", with: "")
    }

    /// The lead finding and the rest, in fixture order. A lead index past the end falls back to
    /// the first finding, so a shorter list never loses its text.
    static func splitFindings(_ findings: [ResultsFinding], lead: Int) -> (lead: ResultsFinding?, more: [ResultsFinding]) {
        guard !findings.isEmpty else { return (nil, []) }
        let index = findings.indices.contains(lead) ? lead : 0
        var more = findings
        let first = more.remove(at: index)
        return (first, more)
    }

    static func capitalizedFirst(_ text: String) -> String {
        text.prefix(1).uppercased() + text.dropFirst()
    }

    /// The first 12 characters of a digest or revision; the full value stays in the tooltip.
    static func shortDigest(_ value: String) -> String { String(value.prefix(12)) }

    // MARK: S6

    static func armName(_ arm: String) -> String {
        switch arm {
        case "naive": return UIText.Results.armNaive
        case "recovery": return UIText.Results.armRecovery
        default: return Format.label(arm)
        }
    }

    static func pathName(_ motion: String) -> String {
        switch motion {
        case "constant_velocity": return UIText.Results.pathStraight
        case "constant_turn": return UIText.Results.pathTurn
        default: return Format.label(motion)
        }
    }

    /// "Recovery (capture time), 200 ms: 0.160 m (0.154 to 0.166), 0 of 50 fail"
    static func hoverText(series: S6Series, index: Int, delays: [Double]) -> String {
        Format.fill(UIText.Results.hoverTemplate, [
            "arm": armName(series.arm),
            "delay": "\(milliseconds(delays[index]))",
            "median": metersValue(series.p95MedianM[index]),
            "low": metersValue(series.p95CILowM[index]),
            "high": metersValue(series.p95CIHighM[index]),
            "failing": "\(series.failCount[index])",
            "total": "\(series.nSeeds[index])"
        ])
    }

    /// The paired comparison for one delay: positive median means recovery had the lower error
    /// (paired p95 difference is basic minus recovery per seed).
    static func comparison(series: S6Series, index: Int, delay: Double) -> [String] {
        let tied = series.tieCount[index]
        let total = series.nSeeds[index]
        if tied == total {
            return [delay == 0 && series.lossProbability == 0 ? UIText.Results.identicalNoDelay : UIText.Results.tiedEverySeed]
        }
        let median = series.pairedDiffMedianM[index]
        let headline: String
        if median > 0 {
            headline = Format.fill(UIText.Results.recoveryLower, ["value": metersValue(median)])
        } else if median < 0 {
            headline = Format.fill(UIText.Results.recoveryHigher, ["value": metersValue(-median)])
        } else {
            headline = UIText.Results.recoveryNoDifference
        }
        let interval = Format.fill(UIText.Results.pairedInterval, [
            "low": metersValue(series.pairedDiffCILowM[index]), "high": metersValue(series.pairedDiffCIHighM[index])
        ])
        let counts = Format.fill(UIText.Results.pairedCounts, [
            "better": "\(series.recoveryBetterCount[index])", "worse": "\(series.recoveryWorseCount[index])",
            "tied": "\(tied)", "total": "\(total)"
        ])
        return [headline, interval, counts]
    }

    // MARK: Atlas

    /// What a matrix cell says: ">300 ms", "200 ms" or "Fails at 0".
    static func cellText(_ entry: AtlasEntry?) -> String {
        guard let entry else { return notComputed }
        switch entry.kind {
        case .censored: return UIText.Atlas.censored
        case .failsAtZero: return UIText.Atlas.failsAtZero
        case .value: return entry.valueS.map(delay) ?? notComputed
        }
    }

    /// The inspector's "Most delay tolerated" tile: ">300 ms", "200 ms" or "Fails at 0 ms".
    static func mostDelayText(_ entry: AtlasEntry) -> String {
        entry.kind == .failsAtZero ? "\(UIText.Atlas.failsAtZero) ms" : cellText(entry)
    }

    /// One interval end with its unit: "200 ms", "below 0 ms" or "above 300 ms".
    static func intervalEnd(_ end: IntervalEnd) -> String {
        switch end {
        case .seconds(let seconds): return delay(seconds)
        case .belowZero: return "\(UIText.Atlas.intervalBelowZero) ms"
        case .aboveMax: return "\(UIText.Atlas.intervalAbove) ms"
        }
    }

    /// "100 ms to 200 ms", "Below 0 ms to 200 ms", "100 ms to above 300 ms", "Above 300 ms".
    static func interval(_ ends: [IntervalEnd]) -> String {
        guard ends.count == 2 else { return notComputed }
        if ends[0] == ends[1] { return capitalizedFirst(intervalEnd(ends[0])) }
        return capitalizedFirst(Format.fill(UIText.Atlas.intervalSpan, ["low": intervalEnd(ends[0]), "high": intervalEnd(ends[1])]))
    }

    static func cause(_ failureMode: String?) -> String {
        switch failureMode {
        case nil: return UIText.Atlas.causeNone
        case "lag-dominated": return UIText.Atlas.causeLag
        case "gap-dominated": return UIText.Atlas.causeGap
        case "noise-dominated": return UIText.Atlas.causeNoise
        case "model-mismatch": return UIText.Atlas.causeModel
        case let other?: return Format.label(other.replacingOccurrences(of: "-", with: " "))
        }
    }

    static func armHelp(_ arm: String) -> String? {
        switch arm {
        case "naive": return UIText.Atlas.armHelpNaive
        case "recovery": return UIText.Atlas.armHelpRecovery
        case "imm": return UIText.Atlas.armHelpImm
        case "ct": return UIText.Atlas.armHelpCt
        case "adaptive": return UIText.Atlas.armHelpAdaptive
        case "retro": return UIText.Atlas.armHelpRetro
        case "recovery_gated": return UIText.Atlas.armHelpRecoveryGated
        case "imm_gated": return UIText.Atlas.armHelpImmGated
        case "ct_gated": return UIText.Atlas.armHelpCtGated
        default: return nil
        }
    }

    static func armLabel(_ arm: String, labels: [String: String]) -> String {
        labels[arm] ?? Format.label(arm)
    }

    /// The "Most delay" column for one row: the winner, the tie, or "Nothing passes". Ties of
    /// more than three trackers are counted, as in the atlas figure.
    static func classification(_ cell: AtlasCell?, labels: [String: String]) -> String {
        guard let cell else { return notComputed }
        switch cell.kind {
        case .winner: return armLabel(cell.arms[0], labels: labels)
        case .none: return UIText.Atlas.nothingPasses
        case .tie:
            if cell.arms.count > tieListedMax {
                return Format.fill(UIText.Atlas.tieCount, ["count": "\(cell.arms.count)"])
            }
            return Format.fill(UIText.Atlas.tieListed, ["arms": cell.arms.map { armLabel($0, labels: labels) }.joined(separator: ", ")])
        }
    }

    /// Mirrors TIE_LISTED_MAX in tools/make_s6_atlas_figure.py: longer ties are counted.
    static let tieListedMax = 3

    static func summaryLine(_ counts: AtlasCounts) -> String {
        Format.fill(UIText.Atlas.summaryLine, ["tie": "\(counts.tie)", "winner": "\(counts.winner)", "none": "\(counts.none)",
                                               "cells": "\(counts.cells)"])
    }

    static func cellRule(_ rule: AtlasFailureRule) -> String {
        Format.fill(UIText.Atlas.cellRule, ["status": rule.cellRuleStatus])
    }

    /// Loss-level segment label: "None" for 0, else "10 %".
    static func levelLabel(_ probability: Double) -> String {
        probability == 0 ? UIText.Results.lossNone : Format.fill(UIText.Atlas.levelPercent, ["percent": "\(percent(probability))"])
    }
}
