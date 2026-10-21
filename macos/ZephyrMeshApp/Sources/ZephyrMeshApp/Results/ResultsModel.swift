import Foundation
import SwiftUI

/// The two views in the Results window sidebar.
enum ResultsPage: String, CaseIterable, Identifiable, Hashable {
    case delay, atlas

    var id: String { rawValue }

    var title: String { self == .delay ? UIText.Results.sidebarDelay : UIText.Results.sidebarAtlas }

    var symbol: String { self == .delay ? "chart.xyaxis.line" : "square.grid.3x3" }

    var question: String { self == .delay ? UIText.Results.delayTitle : UIText.Results.atlasTitle }

    var emptyHint: String { self == .delay ? UIText.Results.inspectorEmptyHintDelay : UIText.Results.inspectorEmptyHintAtlas }

    /// "results:delay" or "results:atlas" from ZEPHYR_OPEN.
    init?(openRequest: String?) {
        guard let request = openRequest, request.hasPrefix("results:") else { return nil }
        self.init(rawValue: String(request.dropFirst("results:".count)))
    }
}

/// One panel of the delay charts: the loss model and probability whose series it shows.
struct LossPanel: Equatable, Hashable {
    let lossModel: String
    let probability: Double

    var title: String { ResultsFormat.panelTitle(probability: probability) }
}

enum DelayLossLayout {
    /// Recovery first, as in the legend; Up and Down step through this order.
    static let arms = ["recovery", "naive"]
    static let motions = ["constant_velocity", "constant_turn"]
    static let lossModels = ["independent", "burst"]

    /// Four panels: no loss first, then each level of the chosen loss type. The bursty set has
    /// no 0 level of its own, so its first panel is the same no-loss run as the random set.
    static func panels(lossModel: String, levels: [String: [Double]]) -> [LossPanel] {
        let noLoss = (levels["independent"] ?? []).contains(0) ? [LossPanel(lossModel: "independent", probability: 0)] : []
        let rest = (levels[lossModel] ?? []).sorted().filter { $0 > 0 }.map { LossPanel(lossModel: lossModel, probability: $0) }
        return noLoss + rest
    }
}

enum AtlasLayout {
    static let lossModels = ["independent", "burst"]

    /// Loss levels offered by the segmented control: every probability in the stratum's loss
    /// conditions, ascending, with 0 shown as "None".
    static func levels(for stratum: AtlasStratum) -> [Double] {
        let values = stratum.lossConditions.compactMap { $0.split(separator: ":").last.flatMap { Double($0) } }
        return Array(Set(values)).sorted()
    }

    /// The fixture loss condition for the controls. "None" is the one no-loss run for both types.
    static func lossCondition(lossModel: String, level: Double) -> String {
        level == 0 ? ResultsFormat.lossKey(model: "independent", probability: 0)
            : ResultsFormat.lossKey(model: lossModel, probability: level)
    }
}

struct DelayLossState: Equatable {
    var motion = "constant_velocity"
    var lossModel = "independent"
    var selection: ChartSelection?
}

struct AtlasState: Equatable {
    var stratumID = "central"
    var lossModel = "independent"
    var level = 0.0
    var selection: AtlasSelection?

    var lossCondition: String { AtlasLayout.lossCondition(lossModel: lossModel, level: level) }
}

/// "key=value,key=value" from ZEPHYR_RESULTS_SELECT.
enum ResultsSelectSpec {
    static func parse(_ text: String?) -> [String: String] {
        guard let text else { return [:] }
        var values: [String: String] = [:]
        for part in text.split(separator: ",") {
            let pair = part.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            if pair.count == 2, !pair[0].isEmpty { values[pair[0]] = pair[1] }
        }
        return values
    }

    static func motion(_ path: String?) -> String? {
        switch path {
        case "straight": return "constant_velocity"
        case "turn": return "constant_turn"
        default: return nil
        }
    }

    /// Applies a delay-view spec: path, loss, level, arm, delay. Unknown values are ignored.
    static func delayState(_ spec: [String: String], data: ResultsData, base: DelayLossState = DelayLossState()) -> DelayLossState {
        var state = base
        if let motion = motion(spec["path"]) { state.motion = motion }
        if let loss = spec["loss"], DelayLossLayout.lossModels.contains(loss) { state.lossModel = loss }
        let panels = DelayLossLayout.panels(lossModel: state.lossModel, levels: data.s6.lossLevels)
        guard let level = spec["level"].flatMap(Double.init),
              let panel = panels.firstIndex(where: { abs($0.probability - level) < 1e-9 }),
              let arm = spec["arm"].flatMap({ DelayLossLayout.arms.firstIndex(of: $0) }),
              let delay = spec["delay"].flatMap(Double.init),
              let delayIndex = data.s6.delaysS.firstIndex(where: { abs($0 - delay) < 1e-9 }) else { return state }
        state.selection = ChartSelection(panel: panel, arm: arm, delayIndex: delayIndex)
        return state
    }

    /// Applies an atlas spec: stratum, loss, level, variant, arm. Unknown values are ignored.
    static func atlasState(_ spec: [String: String], data: ResultsData, base: AtlasState = AtlasState()) -> AtlasState {
        var state = base
        if let id = spec["stratum"], data.stratum(id) != nil { state.stratumID = id }
        if let loss = spec["loss"], AtlasLayout.lossModels.contains(loss) { state.lossModel = loss }
        guard let stratum = data.stratum(state.stratumID) else { return state }
        if let level = spec["level"].flatMap(Double.init),
           let match = AtlasLayout.levels(for: stratum).first(where: { abs($0 - level) < 1e-9 }) {
            state.level = match
        }
        if let row = spec["variant"].flatMap({ stratum.variants.firstIndex(of: $0) }),
           let column = spec["arm"].flatMap({ stratum.arms.firstIndex(of: $0) }) {
            state.selection = AtlasSelection(row: row, column: column)
        }
        return state
    }
}

/// State for one Results window: the loaded fixture (or why it failed), the page, and each
/// view's controls and selection. Kept per page, so switching pages never loses a selection.
@MainActor
final class ResultsModel: ObservableObject {
    let data: ResultsData?
    let loadError: String?
    @Published var page: ResultsPage
    @Published var showInspector: Bool
    @Published var delay = DelayLossState()
    @Published var atlas = AtlasState()

    init(load: Result<ResultsData, Error> = ResultsLoader.loadBundled(), setup: ScreenshotSetup = .current) {
        switch load {
        case .success(let data):
            self.data = data
            loadError = nil
        case .failure(let error):
            data = nil
            loadError = error.localizedDescription
        }
        page = ResultsPage(openRequest: setup.open) ?? .delay
        showInspector = setup.showInspector ?? true
        if let data, let text = setup.resultsSelect {
            let spec = ResultsSelectSpec.parse(text)
            if page == .delay {
                delay = ResultsSelectSpec.delayState(spec, data: data)
            } else {
                atlas = ResultsSelectSpec.atlasState(spec, data: data)
            }
        }
    }

    // MARK: Delay and loss

    var panels: [LossPanel] {
        guard let data else { return [] }
        return DelayLossLayout.panels(lossModel: delay.lossModel, levels: data.s6.lossLevels)
    }

    func series(panel: LossPanel, arm: String) -> S6Series? {
        data?.series(motion: delay.motion, lossModel: panel.lossModel, probability: panel.probability, arm: arm)
    }

    /// Moves the chart selection. The first arrow press with nothing selected picks the first
    /// point of the first panel. Returns false when the key should go on (for example Tab past
    /// the last panel), so keyboard focus is never trapped.
    @discardableResult
    func moveChart(_ move: ChartSelection.Move) -> Bool {
        guard let data else { return false }
        guard let current = delay.selection else {
            if move == .nextPanel || move == .previousPanel { return false }
            delay.selection = ChartSelection(panel: 0, arm: 0, delayIndex: 0)
            return true
        }
        guard let next = current.moved(move, panels: panels.count, arms: DelayLossLayout.arms.count,
                                       delays: data.s6.delaysS.count) else {
            return move != .nextPanel && move != .previousPanel
        }
        delay.selection = next
        return true
    }

    // MARK: Atlas

    var stratum: AtlasStratum? { data?.stratum(atlas.stratumID) }

    func moveAtlas(_ move: AtlasSelection.Move) {
        guard let stratum else { return }
        let current = atlas.selection ?? AtlasSelection(row: 0, column: 0)
        atlas.selection = atlas.selection == nil ? current
            : current.moved(move, rows: stratum.variants.count, columns: stratum.arms.count)
    }

    /// A new stratum can have fewer rows or columns, so the selection is clamped to it.
    func selectStratum(_ id: String) {
        atlas.stratumID = id
        guard let stratum, let selection = atlas.selection else { return }
        atlas.selection = selection.clamped(rows: stratum.variants.count, columns: stratum.arms.count)
    }

    func clearSelection() {
        if page == .delay { delay.selection = nil } else { atlas.selection = nil }
    }
}
