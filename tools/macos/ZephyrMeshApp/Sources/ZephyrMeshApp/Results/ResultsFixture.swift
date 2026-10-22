import Foundation

// Strict decoder for results_fixture.json (schema zephyr-results-fixture-1), built by
// tools/build_results_fixture.py from checked-in run files. Every object has an exact key set,
// so a missing or unknown key fails with a message naming the object. Nothing is recomputed:
// the views only format, bin and arrange these values.

struct ResultsDecodeError: LocalizedError, Equatable {
    let message: String
    var errorDescription: String? { message }
}

/// The exact-key container every results object is read through.
private func strictContainer(_ decoder: Decoder, keys: [String], context: String) throws
    -> KeyedDecodingContainer<DynamicCodingKey> {
    let container = try decoder.container(keyedBy: DynamicCodingKey.self)
    let actual = Set(container.allKeys.map(\.stringValue))
    let expected = Set(keys)
    guard actual == expected else {
        var details: [String] = []
        let missing = expected.subtracting(actual).sorted()
        let extra = actual.subtracting(expected).sorted()
        if !missing.isEmpty { details.append("missing \(missing.joined(separator: ", "))") }
        if !extra.isEmpty { details.append("unknown \(extra.joined(separator: ", "))") }
        throw ResultsDecodeError(message: "\(context) keys do not match (\(details.joined(separator: "; ")))")
    }
    return container
}

private extension KeyedDecodingContainer where Key == DynamicCodingKey {
    func value<T: Decodable>(_ key: String, as type: T.Type = T.self) throws -> T {
        try decode(T.self, forKey: DynamicCodingKey(key))
    }

    /// Present but possibly null: the key is still required by the exact key set.
    func optional<T: Decodable>(_ key: String, as type: T.Type = T.self) throws -> T? {
        try decodeNil(forKey: DynamicCodingKey(key)) ? nil : decode(T.self, forKey: DynamicCodingKey(key))
    }

    func finite(_ key: String, context: String) throws -> Double {
        let number = try value(key, as: Double.self)
        guard number.isFinite else { throw ResultsDecodeError(message: "\(context).\(key) must be finite") }
        return number
    }

    func finiteArray(_ key: String, count: Int? = nil, context: String) throws -> [Double] {
        let numbers = try value(key, as: [Double].self)
        guard numbers.allSatisfy(\.isFinite) else { throw ResultsDecodeError(message: "\(context).\(key) must be finite") }
        if let count, numbers.count != count {
            throw ResultsDecodeError(message: "\(context).\(key) needs \(count) values, found \(numbers.count)")
        }
        return numbers
    }
}

// MARK: - Shared parts

struct ResultsSource: Decodable, Equatable {
    let file, gitRevision, gitTreeState, manifestSHA256, runDir, schema, sha256: String

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: ["file", "git_revision", "git_tree_state", "manifest_sha256",
                                                         "run_dir", "schema", "sha256"], context: "source")
        file = try values.value("file")
        gitRevision = try values.value("git_revision")
        gitTreeState = try values.value("git_tree_state")
        manifestSHA256 = try values.value("manifest_sha256")
        runDir = try values.value("run_dir")
        schema = try values.value("schema")
        sha256 = try values.value("sha256")
        for (name, digest) in [("sha256", sha256), ("manifest_sha256", manifestSHA256)]
        where digest.range(of: "^[0-9a-f]{64}$", options: .regularExpression) == nil {
            throw ResultsDecodeError(message: "source.\(name) must be 64 lowercase hexadecimal characters")
        }
    }
}

struct ResultsFinding: Decodable, Equatable {
    let source, text: String

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: ["source", "text"], context: "finding")
        source = try values.value("source")
        text = try values.value("text")
        guard !text.isEmpty else { throw ResultsDecodeError(message: "finding text must not be empty") }
    }
}

// MARK: - S6 (delay and loss)

struct S6Series: Decodable, Equatable {
    let arm, lossModel, motion: String
    let lossProbability: Double
    let cellIDs, label: [String]
    let p95MedianM, p95CILowM, p95CIHighM, failRate: [Double]
    let pairedDiffMedianM, pairedDiffCILowM, pairedDiffCIHighM: [Double]
    let failCount, nSeeds, recoveryBetterCount, recoveryWorseCount, tieCount: [Int]

    static let keys = ["arm", "cell_ids", "fail_count", "fail_rate", "label", "loss_model", "loss_probability", "motion",
                       "n_seeds", "p95_ci_high_m", "p95_ci_low_m", "p95_median_m", "paired_diff_ci_high_m",
                       "paired_diff_ci_low_m", "paired_diff_median_m", "recovery_better_count", "recovery_worse_count",
                       "tie_count"]

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: Self.keys, context: "s6 series")
        let context = "s6 series"
        arm = try values.value("arm")
        lossModel = try values.value("loss_model")
        motion = try values.value("motion")
        lossProbability = try values.finite("loss_probability", context: context)
        cellIDs = try values.value("cell_ids")
        let count = cellIDs.count
        label = try values.value("label")
        p95MedianM = try values.finiteArray("p95_median_m", count: count, context: context)
        p95CILowM = try values.finiteArray("p95_ci_low_m", count: count, context: context)
        p95CIHighM = try values.finiteArray("p95_ci_high_m", count: count, context: context)
        failRate = try values.finiteArray("fail_rate", count: count, context: context)
        pairedDiffMedianM = try values.finiteArray("paired_diff_median_m", count: count, context: context)
        pairedDiffCILowM = try values.finiteArray("paired_diff_ci_low_m", count: count, context: context)
        pairedDiffCIHighM = try values.finiteArray("paired_diff_ci_high_m", count: count, context: context)
        failCount = try values.value("fail_count")
        nSeeds = try values.value("n_seeds")
        recoveryBetterCount = try values.value("recovery_better_count")
        recoveryWorseCount = try values.value("recovery_worse_count")
        tieCount = try values.value("tie_count")
        for (name, list) in [("label", label.count), ("fail_count", failCount.count), ("n_seeds", nSeeds.count),
                             ("recovery_better_count", recoveryBetterCount.count),
                             ("recovery_worse_count", recoveryWorseCount.count), ("tie_count", tieCount.count)]
        where list != count {
            throw ResultsDecodeError(message: "\(context).\(name) needs \(count) values, found \(list)")
        }
        guard label.allSatisfy({ $0 == "passes" || $0 == "fails" }) else {
            throw ResultsDecodeError(message: "\(context).label must be passes or fails")
        }
    }
}

struct S6Results: Decodable, Equatable {
    let arms, limits, lossModels, motions: [String]
    let cellLabelRule, criterion, intervalNote, status: String
    let delaysS: [Double]
    let failureMaxM, failureP95M, noiseStdM, targetSpeedMps, turnRadiusM: Double
    let findings: [ResultsFinding]
    let lossLevels: [String: [Double]]
    let series: [S6Series]
    let source: ResultsSource

    static let keys = ["arms", "cell_label_rule", "criterion", "delays_s", "failure_max_m", "failure_p95_m", "findings",
                       "interval_note", "limits", "loss_levels", "loss_models", "motions", "noise_std_m", "series",
                       "source", "status", "target_speed_mps", "turn_radius_m"]

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: Self.keys, context: "s6")
        arms = try values.value("arms")
        limits = try values.value("limits")
        lossModels = try values.value("loss_models")
        motions = try values.value("motions")
        cellLabelRule = try values.value("cell_label_rule")
        criterion = try values.value("criterion")
        intervalNote = try values.value("interval_note")
        status = try values.value("status")
        delaysS = try values.finiteArray("delays_s", context: "s6")
        failureMaxM = try values.finite("failure_max_m", context: "s6")
        failureP95M = try values.finite("failure_p95_m", context: "s6")
        noiseStdM = try values.finite("noise_std_m", context: "s6")
        targetSpeedMps = try values.finite("target_speed_mps", context: "s6")
        turnRadiusM = try values.finite("turn_radius_m", context: "s6")
        findings = try values.value("findings")
        lossLevels = try values.value("loss_levels")
        series = try values.value("series")
        source = try values.value("source")
        guard !delaysS.isEmpty, zip(delaysS, delaysS.dropFirst()).allSatisfy({ $0 < $1 }) else {
            throw ResultsDecodeError(message: "s6.delays_s must be a non-empty ascending list")
        }
        guard Set(arms) == ["naive", "recovery"] else { throw ResultsDecodeError(message: "s6.arms must be naive and recovery") }
        for item in series where item.cellIDs.count != delaysS.count {
            throw ResultsDecodeError(message: "s6 series \(item.motion) \(item.lossModel) has \(item.cellIDs.count) delays")
        }
    }
}

// MARK: - Atlas

/// One end of a tolerated-delay interval: seconds, "<0" (even 0 s may fail) or ">0.3" (above
/// the largest tested delay). Any other string is a schema error.
enum IntervalEnd: Decodable, Equatable {
    case seconds(Double)
    case belowZero
    case aboveMax

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let number = try? container.decode(Double.self) {
            guard number.isFinite else { throw ResultsDecodeError(message: "interval end must be finite") }
            self = .seconds(number)
            return
        }
        guard let text = try? container.decode(String.self) else {
            throw ResultsDecodeError(message: "interval end must be a number, \"<0\" or \">0.3\"")
        }
        switch text {
        case "<0": self = .belowZero
        case ">0.3": self = .aboveMax
        default: throw ResultsDecodeError(message: "unknown interval end \"\(text)\"")
        }
    }
}

struct AtlasImprovement: Decodable, Equatable {
    let delayS, medianP95DiffM: Double
    let intervalM: [Double]

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: ["delay_s", "interval_m", "median_p95_diff_m"], context: "improvement")
        delayS = try values.finite("delay_s", context: "improvement")
        medianP95DiffM = try values.finite("median_p95_diff_m", context: "improvement")
        intervalM = try values.finiteArray("interval_m", count: 2, context: "improvement")
    }
}

struct AtlasEntry: Decodable, Equatable {
    enum Kind: String, Decodable { case value, censored, failsAtZero = "fails_at_zero" }

    let arm, lossCondition, lossModel, stratum, variant: String
    let kind: Kind
    let lossProbability: Double
    let valueS: Double?
    let censored, failureMode: String?
    let intervalS: [IntervalEnd]
    let improvementVsNaive: [AtlasImprovement]

    static let keys = ["arm", "censored", "failure_mode", "improvement_vs_naive", "interval_s", "kind", "loss_condition",
                       "loss_model", "loss_probability", "stratum", "value_s", "variant"]

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: Self.keys, context: "atlas entry")
        arm = try values.value("arm")
        lossCondition = try values.value("loss_condition")
        lossModel = try values.value("loss_model")
        stratum = try values.value("stratum")
        variant = try values.value("variant")
        kind = try values.value("kind")
        lossProbability = try values.finite("loss_probability", context: "atlas entry")
        valueS = try values.optional("value_s")
        censored = try values.optional("censored")
        failureMode = try values.optional("failure_mode")
        intervalS = try values.value("interval_s")
        improvementVsNaive = try values.value("improvement_vs_naive")
        guard intervalS.count == 2 else { throw ResultsDecodeError(message: "atlas entry interval_s needs 2 ends") }
        switch kind {
        case .value:
            guard let valueS, valueS.isFinite, valueS >= 0 else {
                throw ResultsDecodeError(message: "atlas entry of kind value needs a finite value_s")
            }
        case .censored:
            guard censored == ">0.3", valueS == nil else { throw ResultsDecodeError(message: "censored atlas entry is malformed") }
        case .failsAtZero:
            guard valueS == nil, censored == nil else { throw ResultsDecodeError(message: "fails_at_zero atlas entry is malformed") }
        }
    }
}

struct AtlasCell: Decodable, Equatable {
    enum Kind: String, Decodable { case winner, tie, none }

    let arms: [String]
    let delayText: String?
    let kind: Kind
    let lossCondition, stratum, variant: String

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: ["arms", "delay_text", "kind", "loss_condition", "stratum", "variant"],
                                         context: "atlas cell")
        arms = try values.value("arms")
        delayText = try values.optional("delay_text")
        kind = try values.value("kind")
        lossCondition = try values.value("loss_condition")
        stratum = try values.value("stratum")
        variant = try values.value("variant")
        switch kind {
        case .winner where arms.count != 1, .tie where arms.count < 2, .none where !arms.isEmpty:
            throw ResultsDecodeError(message: "atlas cell \(kind.rawValue) has \(arms.count) arms")
        default: break
        }
    }
}

struct AtlasCounts: Decodable, Equatable {
    let cells, none, survive, tie, winner: Int

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: ["cells", "none", "survive", "tie", "winner"], context: "stratum counts")
        cells = try values.value("cells")
        none = try values.value("none")
        survive = try values.value("survive")
        tie = try values.value("tie")
        winner = try values.value("winner")
    }
}

struct AtlasStratum: Decodable, Equatable {
    let id, title, description: String
    let arms, lossConditions, variants: [String]
    let counts: AtlasCounts

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: ["arms", "counts", "description", "id", "loss_conditions", "title",
                                                         "variants"], context: "stratum")
        id = try values.value("id")
        title = try values.value("title")
        description = try values.value("description")
        arms = try values.value("arms")
        lossConditions = try values.value("loss_conditions")
        variants = try values.value("variants")
        counts = try values.value("counts")
    }
}

struct AtlasFailureRule: Decodable, Equatable {
    struct PerSeed: Decodable, Equatable {
        let maxErrorM, p95ErrorM: Double
        let scoredWindowS: [Double]

        init(from decoder: Decoder) throws {
            let values = try strictContainer(decoder, keys: ["max_error_m", "p95_error_m", "scored_window_s"], context: "per_seed")
            maxErrorM = try values.finite("max_error_m", context: "per_seed")
            p95ErrorM = try values.finite("p95_error_m", context: "per_seed")
            scoredWindowS = try values.finiteArray("scored_window_s", count: 2, context: "per_seed")
        }
    }

    let cellRule, cellRuleStatus: String
    let failRateThreshold: Double
    let perSeed: PerSeed

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: ["cell_rule", "cell_rule_status", "fail_rate_threshold", "per_seed"],
                                         context: "failure_rule")
        cellRule = try values.value("cell_rule")
        cellRuleStatus = try values.value("cell_rule_status")
        failRateThreshold = try values.finite("fail_rate_threshold", context: "failure_rule")
        perSeed = try values.value("per_seed")
    }
}

struct AtlasResults: Decodable, Equatable {
    let armLabels, lossTitles, variantTitles: [String: String]
    let cells: [AtlasCell]
    let entries: [AtlasEntry]
    let delaysS: [Double]
    let failureRule: AtlasFailureRule
    let findings: [ResultsFinding]
    let limits: [String]
    let nSeeds: Int
    let schemaOfSource: String
    let source: ResultsSource
    let strata: [AtlasStratum]

    static let keys = ["arm_labels", "cells", "delays_s", "entries", "failure_rule", "findings", "limits", "loss_titles",
                       "n_seeds", "schema_of_source", "source", "strata", "variant_titles"]

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: Self.keys, context: "atlas")
        armLabels = try values.value("arm_labels")
        lossTitles = try values.value("loss_titles")
        variantTitles = try values.value("variant_titles")
        cells = try values.value("cells")
        entries = try values.value("entries")
        delaysS = try values.finiteArray("delays_s", context: "atlas")
        failureRule = try values.value("failure_rule")
        findings = try values.value("findings")
        limits = try values.value("limits")
        nSeeds = try values.value("n_seeds")
        schemaOfSource = try values.value("schema_of_source")
        source = try values.value("source")
        strata = try values.value("strata")
        guard !strata.isEmpty else { throw ResultsDecodeError(message: "atlas.strata must not be empty") }
    }
}

// MARK: - Top level

struct ResultsFixture: Decodable, Equatable {
    static let schemaName = "zephyr-results-fixture-1"

    let schema, generator: String
    let simulated: Bool
    let s6: S6Results
    let atlas: AtlasResults

    init(from decoder: Decoder) throws {
        let values = try strictContainer(decoder, keys: ["atlas", "generator", "s6", "schema", "simulated"], context: "results fixture")
        schema = try values.value("schema")
        guard schema == Self.schemaName else { throw ResultsDecodeError(message: "unknown results schema \"\(schema)\"") }
        generator = try values.value("generator")
        simulated = try values.value("simulated")
        guard simulated else { throw ResultsDecodeError(message: "the results fixture must be marked simulated") }
        s6 = try values.value("s6")
        atlas = try values.value("atlas")
    }

    static func decode(_ data: Data) throws -> ResultsFixture {
        do {
            return try JSONDecoder().decode(ResultsFixture.self, from: data)
        } catch let error as ResultsDecodeError {
            throw error
        } catch let DecodingError.typeMismatch(_, context), let DecodingError.valueNotFound(_, context),
                let DecodingError.dataCorrupted(context) {
            let path = context.codingPath.map(\.stringValue).joined(separator: ".")
            throw ResultsDecodeError(message: "\(path): \(context.debugDescription)")
        }
    }
}

enum ResultsLoader {
    static let resourceName = "results_fixture"

    /// Loads the bundled results. A failure is returned so the window can say so; the app never
    /// shows invented results in place of a file it could not read.
    static func loadBundled() -> Result<ResultsData, Error> {
        guard let url = ResourceLocator.url(forResource: resourceName, withExtension: "json") else {
            return .failure(ResultsDecodeError(message: "The bundled results file is missing from the app."))
        }
        return Result { try ResultsData(fixture: ResultsFixture.decode(Data(contentsOf: url))) }
    }
}

/// The decoded fixture plus lookups the views need. Built once per window.
struct ResultsData {
    let fixture: ResultsFixture
    private let seriesIndex: [String: S6Series]
    private let entryIndex: [String: AtlasEntry]
    private let cellIndex: [String: AtlasCell]

    var s6: S6Results { fixture.s6 }
    var atlas: AtlasResults { fixture.atlas }

    init(fixture: ResultsFixture) throws {
        self.fixture = fixture
        var series: [String: S6Series] = [:]
        for item in fixture.s6.series {
            let key = Self.seriesKey(motion: item.motion, lossModel: item.lossModel, probability: item.lossProbability, arm: item.arm)
            guard series.updateValue(item, forKey: key) == nil else { throw ResultsDecodeError(message: "duplicate s6 series \(key)") }
        }
        var entries: [String: AtlasEntry] = [:]
        for entry in fixture.atlas.entries {
            let key = Self.entryKey(entry.stratum, entry.variant, entry.lossCondition, entry.arm)
            guard entries.updateValue(entry, forKey: key) == nil else { throw ResultsDecodeError(message: "duplicate atlas entry \(key)") }
        }
        var cells: [String: AtlasCell] = [:]
        for cell in fixture.atlas.cells {
            let key = Self.entryKey(cell.stratum, cell.variant, cell.lossCondition, nil)
            guard cells.updateValue(cell, forKey: key) == nil else { throw ResultsDecodeError(message: "duplicate atlas cell \(key)") }
        }
        seriesIndex = series
        entryIndex = entries
        cellIndex = cells
    }

    static func seriesKey(motion: String, lossModel: String, probability: Double, arm: String) -> String {
        "\(motion)|\(ResultsFormat.lossKey(model: lossModel, probability: probability))|\(arm)"
    }

    private static func entryKey(_ stratum: String, _ variant: String, _ loss: String, _ arm: String?) -> String {
        [stratum, variant, loss, arm ?? ""].joined(separator: "|")
    }

    func series(motion: String, lossModel: String, probability: Double, arm: String) -> S6Series? {
        seriesIndex[Self.seriesKey(motion: motion, lossModel: lossModel, probability: probability, arm: arm)]
    }

    func entry(stratum: String, variant: String, loss: String, arm: String) -> AtlasEntry? {
        entryIndex[Self.entryKey(stratum, variant, loss, arm)]
    }

    func cell(stratum: String, variant: String, loss: String) -> AtlasCell? {
        cellIndex[Self.entryKey(stratum, variant, loss, nil)]
    }

    func stratum(_ id: String) -> AtlasStratum? {
        atlas.strata.first { $0.id == id }
    }

    /// The y range shared by the four panels on screen for one path and loss type: the nice
    /// ceiling of every interval top they show, including the no-loss panel of the bursty set.
    func errorScale(motion: String, lossModel: String) -> NiceScale {
        let panels = DelayLossLayout.panels(lossModel: lossModel, levels: s6.lossLevels)
        let tops = panels.flatMap { panel in
            DelayLossLayout.arms.compactMap {
                series(motion: motion, lossModel: panel.lossModel, probability: panel.probability, arm: $0)?.p95CIHighM.max()
            }
        }
        return NiceScale(maximum: tops.max() ?? 0)
    }
}
