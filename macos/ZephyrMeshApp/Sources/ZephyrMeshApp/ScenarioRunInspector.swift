import AppKit
import CryptoKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// Native decoder for the same generated scenario-run JSON inspected by the
/// Windows preview. It is a bounded, read-only projection of synthetic data.
struct ScenarioRunDocument: Identifiable, Decodable {
    static let schema = "zephyr-s7-scenario-run-1"
    static let version = 1
    static let maxBytes = 64 * 1024 * 1024
    static let jsonSafeIntegerMaximum: Int64 = 9_007_199_254_740_991

    let id: String
    let schema: String
    let version: Int
    let scenarioHash: String
    let parametersCanonical: String
    let payloadCanonical: String
    let payloadSHA256: String
    let parameters: [String: JSONValue]
    let evidenceBoundary: String
    let frames: [ScenarioRunFrame]
    let linkEvents: [[String: JSONValue]]
    let profiles: [[String: JSONValue]]
    let provenance: [String: JSONValue]
    var linkEventCount: Int { linkEvents.count }
    var profileCount: Int { profiles.count }
    var status: String { provenance["status"]?.stringValue ?? "unknown" }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: DynamicCodingKey.self)
        let expected = Set(["schema", "version", "scenario_hash", "parameters_canonical", "payload_canonical", "payload_sha256", "parameters", "evidence_boundary", "frames", "link_events", "profiles", "provenance"])
        let actual = Set(values.allKeys.map(\.stringValue))
        guard actual == expected else { throw ScenarioRunError.invalid("scenario run has an invalid field set") }
        schema = try values.decode(String.self, forKey: DynamicCodingKey("schema"))
        guard schema == Self.schema else { throw ScenarioRunError.invalid("schema must be \(Self.schema)") }
        version = try values.decode(Int.self, forKey: DynamicCodingKey("version"))
        guard version == Self.version else { throw ScenarioRunError.invalid("scenario run version is unsupported") }
        scenarioHash = try values.decode(String.self, forKey: DynamicCodingKey("scenario_hash"))
        guard scenarioHash.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil else { throw ScenarioRunError.invalid("scenario_hash must be lowercase SHA-256") }
        parametersCanonical = try values.decode(String.self, forKey: DynamicCodingKey("parameters_canonical"))
        guard parametersCanonical.hasSuffix("\n"), !parametersCanonical.hasSuffix("\n\n"), parametersCanonical.utf8.count > 2 else { throw ScenarioRunError.invalid("parameters_canonical is invalid") }
        payloadCanonical = try values.decode(String.self, forKey: DynamicCodingKey("payload_canonical"))
        guard payloadCanonical.hasSuffix("\n"), !payloadCanonical.hasSuffix("\n\n"), payloadCanonical.utf8.count > 2 else { throw ScenarioRunError.invalid("payload_canonical is invalid") }
        payloadSHA256 = try values.decode(String.self, forKey: DynamicCodingKey("payload_sha256"))
        guard payloadSHA256.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil else { throw ScenarioRunError.invalid("payload_sha256 must be lowercase SHA-256") }
        parameters = try values.decode([String: JSONValue].self, forKey: DynamicCodingKey("parameters"))
        guard case .string(let scenarioID) = parameters["scenario_id"], !scenarioID.isEmpty else { throw ScenarioRunError.invalid("parameters.scenario_id is required") }
        evidenceBoundary = try values.decode(String.self, forKey: DynamicCodingKey("evidence_boundary"))
        guard !evidenceBoundary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw ScenarioRunError.invalid("evidence_boundary must not be empty") }
        frames = try values.decode([ScenarioRunFrame].self, forKey: DynamicCodingKey("frames"))
        guard !frames.isEmpty else { throw ScenarioRunError.invalid("frames must not be empty") }
        linkEvents = try values.decode([[String: JSONValue]].self, forKey: DynamicCodingKey("link_events"))
        profiles = try values.decode([[String: JSONValue]].self, forKey: DynamicCodingKey("profiles"))
        provenance = try values.decode([String: JSONValue].self, forKey: DynamicCodingKey("provenance"))
        guard provenance.keys.sorted() == ["code_revision", "dt_s", "generator", "numpy", "python", "seed", "status"] else { throw ScenarioRunError.invalid("provenance has an invalid field set") }
        guard case .string(let declaredStatus) = provenance["status"], declaredStatus == "synthetic deterministic replay; read-only" else {
            throw ScenarioRunError.invalid("provenance status must identify a synthetic read-only replay")
        }
        try Self.validateContract(parameters: parameters, parametersCanonical: parametersCanonical, scenarioHash: scenarioHash, payloadCanonical: payloadCanonical, payloadSHA256: payloadSHA256, frames: frames, linkEvents: linkEvents, profiles: profiles, provenance: provenance)
        id = "\(scenarioHash)-scenario"
    }

    static func load(from url: URL) throws -> ScenarioRunDocument {
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile == true else { throw ScenarioRunError.invalid("the selected item is not a regular file") }
        guard let fileSize = values.fileSize, fileSize <= maxBytes else { throw ScenarioRunError.invalid("scenario run exceeds the 64 MiB size limit") }
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard data.count <= maxBytes else { throw ScenarioRunError.invalid("scenario run exceeds the 64 MiB size limit") }
        var scanner = JSONStructureScanner(data: data, maxDepth: EvidenceReport.maxJSONDepth)
        try scanner.scan()
        do {
            return try JSONDecoder().decode(ScenarioRunDocument.self, from: data)
        } catch let error as ScenarioRunError {
            throw error
        } catch {
            throw ScenarioRunError.invalid("invalid scenario run JSON: \(error.localizedDescription)")
        }
    }

    var scenarioID: String { if case .string(let value) = parameters["scenario_id"] { return value }; return "unknown" }
    var seedText: String { jsonText(parameters["seed"]) }

    private func jsonText(_ value: JSONValue?) -> String {
        guard let value else { return "unknown" }
        switch value { case .string(let text): return text; case .integer(let number): return String(number); case .number(let number): return String(number); default: return "unknown" }
    }

    private static func validateContract(parameters: [String: JSONValue], parametersCanonical: String, scenarioHash: String, payloadCanonical: String, payloadSHA256: String, frames: [ScenarioRunFrame], linkEvents: [[String: JSONValue]], profiles: [[String: JSONValue]], provenance: [String: JSONValue]) throws {
        let parameterKeys = ["agent_count", "altitude_m", "dt_s", "keep_out_spheres", "neighbor_link", "radius_m", "scenario_id", "seed", "steps", "target_link", "target_velocity_mps"].sorted()
        guard parameters.keys.sorted() == parameterKeys else { throw ScenarioRunError.invalid("parameters have an invalid field set") }
        guard let scenarioID = parameters["scenario_id"]?.stringValue,
              !scenarioID.isEmpty, scenarioID.count <= 120, !scenarioID.unicodeScalars.contains(where: { $0.value < 32 }),
              parameters["agent_count"]?.intValue ?? 0 >= 1,
              parameters["agent_count"]?.intValue ?? 257 <= 256,
              parameters["steps"]?.intValue ?? 0 == frames.count,
              parameters["steps"]?.intValue ?? 10001 <= 10000,
              parameters["seed"]?.intValue ?? -1 >= 0,
              parameters["seed"]?.intValue ?? -1 <= Self.jsonSafeIntegerMaximum,
              parameters["radius_m"]?.finiteNumber(min: 0, strict: true) == true,
              parameters["radius_m"]?.numberValue ?? 1001 <= 1000,
              parameters["altitude_m"]?.finiteNumber(min: 0) == true,
              parameters["altitude_m"]?.numberValue ?? 101 <= 100,
              parameters["dt_s"]?.finiteNumber(min: 0, strict: true) == true,
              parameters["dt_s"]?.numberValue ?? 2 <= 1,
              parameters["target_velocity_mps"]?.vector != nil else { throw ScenarioRunError.invalid("parameters contain an invalid value") }
        try validateLink(parameters["target_link"], field: "target_link")
        try validateLink(parameters["neighbor_link"], field: "neighbor_link")
        guard case .array(let spheres) = parameters["keep_out_spheres"], spheres.count <= 64 else { throw ScenarioRunError.invalid("keep_out_spheres is invalid") }
        for sphere in spheres { guard case .object(let object) = sphere, object.keys.sorted() == ["center_m", "label", "radius_m"].sorted(), object["label"]?.stringValue?.isEmpty == false, object["center_m"]?.vector != nil, object["radius_m"]?.finiteNumber(min: 0) == true else { throw ScenarioRunError.invalid("keep_out_spheres entry is invalid") } }
        for frame in frames { for agent in frame.agents { try validateAgent(agent) } }
        for event in linkEvents { try validateLinkEvent(event) }
        for profile in profiles { try validateProfile(profile) }
        guard provenance["generator"]?.stringValue == "sim.scenario_runner", provenance["python"]?.stringValue?.isEmpty == false, provenance["numpy"]?.stringValue?.isEmpty == false, provenance["seed"]?.intValue == parameters["seed"]?.intValue, provenance["dt_s"]?.finiteNumber(min: 0, strict: true) == true, provenance["code_revision"] == .null || (provenance["code_revision"]?.stringValue?.isEmpty == false && provenance["code_revision"]?.stringValue?.count ?? 201 <= 200) else { throw ScenarioRunError.invalid("provenance does not match parameters") }
        guard let canonicalData = parametersCanonical.data(using: .utf8), canonicalData.last == 0x0A,
              (try? JSONDecoder().decode([String: JSONValue].self, from: Data(canonicalData.dropLast()))) != nil else { throw ScenarioRunError.invalid("parameters_canonical does not match parameters") }
        let expectedParameterCanonical = JSONValue.object(parameters).canonicalJSON().data(using: .utf8).map({ $0 + Data([0x0A]) })
        guard expectedParameterCanonical == canonicalData else {
            throw ScenarioRunError.invalid("parameters_canonical is not canonical")
        }
        let digest = SHA256.hash(data: canonicalData).map { String(format: "%02x", $0) }.joined()
        guard digest == scenarioHash else { throw ScenarioRunError.invalid("scenario_hash does not match parameters") }
        guard let payloadData = payloadCanonical.data(using: .utf8), payloadData.last == 0x0A,
              let canonicalPayload = try? JSONDecoder().decode(JSONValue.self, from: Data(payloadData.dropLast())),
              canonicalPayload == .object(["frames": .array(frames.map { .object($0.raw) }), "link_events": .array(linkEvents.map { .object($0) }), "profiles": .array(profiles.map { .object($0) })]) else { throw ScenarioRunError.invalid("payload_canonical does not match payload") }
        let expectedPayload = JSONValue.object(["frames": .array(frames.map { .object($0.raw) }), "link_events": .array(linkEvents.map { .object($0) }), "profiles": .array(profiles.map { .object($0) })])
        let expectedPayloadCanonical = expectedPayload.canonicalJSON().data(using: .utf8).map({ $0 + Data([0x0A]) })
        guard expectedPayloadCanonical == payloadData else {
            throw ScenarioRunError.invalid("payload_canonical is not canonical")
        }
        let payloadDigest = SHA256.hash(data: payloadData).map { String(format: "%02x", $0) }.joined()
        guard payloadDigest == payloadSHA256 else { throw ScenarioRunError.invalid("payload_sha256 does not match payload") }
    }

    private static func validateLink(_ value: JSONValue?, field: String) throws {
        guard case .object(let object) = value,
              object.keys.sorted() == ["burst_end_probability", "burst_start_probability", "contention_mode", "delay_jitter_s", "delay_s", "loss_model", "loss_probability", "packet_duration_s"].sorted(),
              object["loss_model"]?.stringValue == "none" || object["loss_model"]?.stringValue == "independent" || object["loss_model"]?.stringValue == "burst",
              object["contention_mode"]?.stringValue == "none" || object["contention_mode"]?.stringValue == "serialized",
              object["delay_s"]?.finiteNumber(min: 0) == true,
              object["delay_jitter_s"]?.finiteNumber(min: 0) == true,
              object["loss_probability"]?.finiteNumber(min: 0) == true,
              object["loss_probability"]?.numberValue ?? 2 <= 1,
              object["burst_start_probability"]?.finiteNumber(min: 0) == true,
              object["burst_start_probability"]?.numberValue ?? 2 <= 1,
              object["burst_end_probability"]?.finiteNumber(min: 0) == true,
              object["burst_end_probability"]?.numberValue ?? 2 <= 1,
              object["packet_duration_s"]?.finiteNumber(min: 0) == true else { throw ScenarioRunError.invalid("\(field) is invalid") }
        let delay = object["delay_s"]?.numberValue ?? -1
        let jitter = object["delay_jitter_s"]?.numberValue ?? -1
        let lossModel = object["loss_model"]?.stringValue ?? ""
        let loss = object["loss_probability"]?.numberValue ?? -1
        let burstStart = object["burst_start_probability"]?.numberValue ?? -1
        let burstEnd = object["burst_end_probability"]?.numberValue ?? -1
        let contention = object["contention_mode"]?.stringValue ?? ""
        let packetDuration = object["packet_duration_s"]?.numberValue ?? -1
        guard jitter <= delay,
              !(delay == 0 && jitter != 0),
              !((lossModel == "none" || lossModel == "burst") && loss != 0),
              !(lossModel == "none" || lossModel == "independent") || (burstStart == 0 && burstEnd == 1),
              contention != "none" || packetDuration == 0,
              contention != "serialized" || packetDuration > 0 else { throw ScenarioRunError.invalid("\(field) contains an invalid packet duration") }
    }
    private static func validateAgent(_ value: [String: JSONValue]) throws { let keys = ["active", "agent_id", "constraint_flags", "fused_target_m", "min_neighbor_distance_m", "neighbor_count", "position_m", "profile_id", "target_age_s", "target_estimate_m", "target_source_time_s", "velocity_mps"].sorted(); guard value.keys.sorted() == keys, value["agent_id"]?.stringValue?.isEmpty == false, value["profile_id"]?.stringValue?.isEmpty == false, value["active"]?.boolValue != nil, value["position_m"]?.vector != nil, value["velocity_mps"]?.vector != nil, value["neighbor_count"]?.intValue ?? -1 >= 0, value["neighbor_count"]?.intValue ?? -1 <= Self.jsonSafeIntegerMaximum, value["constraint_flags"]?.strings != nil else { throw ScenarioRunError.invalid("agent is invalid") }; for key in ["target_estimate_m", "fused_target_m"] { if case .null = value[key] { } else { guard value[key]?.vector != nil else { throw ScenarioRunError.invalid("agent target vector is invalid") } } }; for key in ["target_source_time_s", "target_age_s", "min_neighbor_distance_m"] { if case .null = value[key] { } else { guard value[key]?.finiteNumber(min: 0) == true else { throw ScenarioRunError.invalid("agent metric is invalid") } } } }
    private static func validateLinkEvent(_ value: [String: JSONValue]) throws { let keys = ["duplicate", "loss_reason", "out_of_order", "outcome", "packet_age", "receive_time", "receiver", "send_time", "sender", "seq"].sorted(); guard value.keys.sorted() == keys, value["sender"]?.stringValue?.isEmpty == false, value["receiver"]?.stringValue?.isEmpty == false, value["seq"]?.intValue ?? -1 >= 0, value["seq"]?.intValue ?? -1 <= Self.jsonSafeIntegerMaximum, value["send_time"]?.finiteNumber(min: 0) == true, value["duplicate"]?.boolValue != nil, value["out_of_order"]?.boolValue != nil else { throw ScenarioRunError.invalid("link event is invalid") }; let outcome = value["outcome"]?.stringValue; guard outcome == "received" || outcome == "lost" else { throw ScenarioRunError.invalid("link event outcome is invalid") }; if outcome == "received" { guard let receive = value["receive_time"]?.numberValue, let age = value["packet_age"]?.numberValue, value["loss_reason"] == .null, receive >= value["send_time"]!.numberValue!, abs((receive - value["send_time"]!.numberValue!) - age) <= 1e-9 else { throw ScenarioRunError.invalid("received link event is invalid") } } else { guard value["receive_time"] == .null, value["packet_age"] == .null, value["loss_reason"]?.stringValue?.isEmpty == false else { throw ScenarioRunError.invalid("lost link event is invalid") } } }
    private static func validateProfile(_ value: [String: JSONValue]) throws { let keys = ["profile_id", "motor", "propeller", "motor_count", "arm_length_m", "frame_mass_kg", "battery_capacity_wh", "status", "source", "calibration_id", "code_revision", "notes", "total_mass_kg", "estimated_thrust_at_rpm_limit_n", "estimated_max_thrust_n"].sorted(); guard value.keys.allSatisfy({ keys.contains($0) }), ["profile_id", "motor", "propeller", "motor_count", "arm_length_m", "frame_mass_kg", "battery_capacity_wh", "total_mass_kg", "estimated_max_thrust_n"].allSatisfy({ value[$0] != nil }), value["profile_id"]?.stringValue?.isEmpty == false, value["motor_count"]?.intValue ?? 0 >= 1, value["motor_count"]?.intValue ?? 0 <= Self.jsonSafeIntegerMaximum else { throw ScenarioRunError.invalid("profile is invalid") }; guard let motor = value["motor"]?.objectValue, motor.keys.sorted() == ["mass_kg", "max_power_w", "max_rpm", "max_torque_nm", "nominal_voltage_v", "part_id"].sorted(), let prop = value["propeller"]?.objectValue, prop.keys.sorted() == ["diameter_m", "mass_kg", "max_rpm", "part_id", "pitch_m", "power_coefficient", "thrust_coefficient"].sorted() else { throw ScenarioRunError.invalid("profile motor or propeller is invalid") }; guard motor["part_id"]?.stringValue?.isEmpty == false, prop["part_id"]?.stringValue?.isEmpty == false else { throw ScenarioRunError.invalid("profile part IDs are invalid") }; for key in ["mass_kg", "max_power_w", "max_rpm", "max_torque_nm", "nominal_voltage_v"] { guard motor[key]?.finiteNumber(min: 0, strict: true) == true else { throw ScenarioRunError.invalid("profile motor metric is invalid") } }; for key in ["diameter_m", "mass_kg", "max_rpm", "pitch_m", "power_coefficient", "thrust_coefficient"] { guard prop[key]?.finiteNumber(min: 0, strict: true) == true else { throw ScenarioRunError.invalid("profile propeller metric is invalid") } }; for key in ["arm_length_m", "frame_mass_kg", "battery_capacity_wh", "total_mass_kg", "estimated_max_thrust_n"] { guard value[key]?.finiteNumber(min: 0, strict: true) == true else { throw ScenarioRunError.invalid("profile metric is invalid") } }; if let status = value["status"], status.stringValue == nil { throw ScenarioRunError.invalid("profile.status is invalid") }; for key in ["source", "calibration_id", "code_revision", "notes"] { if let optional = value[key], optional != .null && optional.stringValue == nil { throw ScenarioRunError.invalid("profile.\(key) is invalid") } }; if let limit = value["estimated_thrust_at_rpm_limit_n"], limit.finiteNumber(min: 0, strict: true) == false { throw ScenarioRunError.invalid("profile estimated thrust is invalid") } }
}

struct ScenarioRunFrame: Identifiable, Decodable {
    let id: String
    let stepIndex: Int
    let time: Double
    let target: [Double]
    let activeCount: Int
    let agentCount: Int
    let agents: [[String: JSONValue]]
    let raw: [String: JSONValue]

    enum CodingKeys: String, CodingKey { case stepIndex = "step_index", time = "time_s", target = "target_position_m", activeCount = "active_count", agents }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: DynamicCodingKey.self)
        let expected = Set(["step_index", "time_s", "target_position_m", "active_count", "agents"])
        guard Set(values.allKeys.map(\.stringValue)) == expected else { throw ScenarioRunError.invalid("frame has an invalid field set") }
        var decodedRaw: [String: JSONValue] = [:]
        for key in values.allKeys { decodedRaw[key.stringValue] = try values.decode(JSONValue.self, forKey: key) }
        raw = decodedRaw
        stepIndex = try values.decode(Int.self, forKey: DynamicCodingKey("step_index"))
        time = try values.decode(Double.self, forKey: DynamicCodingKey("time_s"))
        target = try values.decode([Double].self, forKey: DynamicCodingKey("target_position_m"))
        activeCount = try values.decode(Int.self, forKey: DynamicCodingKey("active_count"))
        agents = try values.decode([[String: JSONValue]].self, forKey: DynamicCodingKey("agents"))
        agentCount = agents.count
        guard stepIndex >= 0, Int64(stepIndex) <= ScenarioRunDocument.jsonSafeIntegerMaximum, time >= 0, target.count == 3, activeCount >= 0, Int64(activeCount) <= ScenarioRunDocument.jsonSafeIntegerMaximum, agentCount > 0 else { throw ScenarioRunError.invalid("frame contains an invalid value") }
        id = "\(stepIndex)-\(time)"
    }
}

enum ScenarioRunError: LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let value) = self { return value }; return nil }
}

private extension JSONValue {
    /// Match Python json.dumps(sort_keys=True, separators=(',', ':'),
    /// ensure_ascii=True) for the JSON values used by the scenario contract.
    /// The envelope appends exactly one LF after this representation.
    func canonicalJSON(forceFloat: Bool = false) -> String {
        switch self {
        case .null: return "null"
        case .bool(let value): return value ? "true" : "false"
        case .integer(let value): return forceFloat ? "\(value).0" : String(value)
        case .number(let value):
            let text = String(value)
            return forceFloat && !text.contains(".") && !text.contains("e") && !text.contains("E") ? text + ".0" : text
        case .string(let value): return Self.canonicalJSONString(value)
        case .array(let values): return "[" + values.map { $0.canonicalJSON(forceFloat: forceFloat) }.joined(separator: ",") + "]"
        case .object(let values):
            return "{" + values.keys.sorted().map { key in
                let childForceFloat = Self.floatKeys.contains(key)
                return Self.canonicalJSONString(key) + ":" + values[key]!.canonicalJSON(forceFloat: childForceFloat)
            }.joined(separator: ",") + "}"
        }
    }

    private static let floatKeys: Set<String> = [
        "altitude_m", "arm_length_m", "battery_capacity_wh", "burst_end_probability",
        "burst_start_probability", "delay_jitter_s", "delay_s", "diameter_m", "dt_s",
        "estimated_max_thrust_n", "estimated_thrust_at_rpm_limit_n", "frame_mass_kg",
        "loss_probability", "mass_kg", "max_power_w", "max_rpm", "max_torque_nm",
        "nominal_voltage_v", "packet_age", "packet_duration_s", "pitch_m", "power_coefficient",
        "radius_m", "receive_time", "send_time", "target_age_s", "time_s", "total_mass_kg",
        "thrust_coefficient", "target_position_m", "target_estimate_m", "fused_target_m",
        "velocity_mps", "position_m", "target_velocity_mps", "center_m", "min_neighbor_distance_m",
        "source_time_s", "target_source_time_s"
    ]

    private static func canonicalJSONString(_ value: String) -> String {
        var result = "\""
        for unit in value.utf16 {
            switch unit {
            case 0x08: result += "\\b"
            case 0x09: result += "\\t"
            case 0x0A: result += "\\n"
            case 0x0C: result += "\\f"
            case 0x0D: result += "\\r"
            case 0x22: result += "\\\""
            case 0x5C: result += "\\\\"
            case 0x00...0x1F: result += String(format: "\\u%04x", unit)
            case 0x20...0x7E: result.append(Character(UnicodeScalar(unit)!))
            default: result += String(format: "\\u%04x", unit)
            }
        }
        result += "\""
        return result
    }

    var stringValue: String? { if case .string(let value) = self { return value }; return nil }
    var intValue: Int64? { if case .integer(let value) = self { return value }; return nil }
    var boolValue: Bool? { if case .bool(let value) = self { return value }; return nil }
    var objectValue: [String: JSONValue]? { if case .object(let value) = self { return value }; return nil }
    var strings: [String]? { if case .array(let values) = self { let strings = values.compactMap(\.stringValue); return strings.count == values.count ? strings : nil }; return nil }
    var vector: [Double]? { if case .array(let values) = self { let vector = values.compactMap(\.numberValue); return vector.count == 3 && values.count == 3 ? vector : nil }; return nil }
    var numberValue: Double? { switch self { case .integer(let value): return Double(value); case .number(let value): return value.isFinite ? value : nil; default: return nil } }
    func finiteNumber(min: Double, strict: Bool = false) -> Bool { guard let value = numberValue else { return false }; return value.isFinite && (strict ? value > min : value >= min) }
    var foundationValue: Any? {
        switch self {
        case .object(let values): return values.reduce(into: [String: Any]()) { if let value = $1.value.foundationValue { $0[$1.key] = value } }
        case .array(let values): return values.compactMap(\.foundationValue)
        case .string(let value): return value
        case .integer(let value): return value
        case .number(let value): return value
        case .bool(let value): return value
        case .null: return NSNull()
        }
    }
}

private extension Dictionary where Key == String, Value == JSONValue {
    var foundationValue: Any? {
        reduce(into: [String: Any]()) { if let value = $1.value.foundationValue { $0[$1.key] = value } }
    }
}

@MainActor
final class ScenarioRunInspectorController: ObservableObject {
    @Published var run: ScenarioRunDocument?
    @Published var filename = ""
    @Published var isPresented = false
    @Published var error: EvidenceErrorMessage?

    func openPanel() {
        let panel = NSOpenPanel()
        panel.title = "Open synthetic scenario run"
        panel.message = "Choose a generated zephyr-s7-scenario-run-1 JSON"
        panel.prompt = "Open"
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            run = try ScenarioRunDocument.load(from: url)
            filename = url.lastPathComponent
            isPresented = true
        } catch { self.error = EvidenceErrorMessage(message: error.localizedDescription) }
    }
}

struct ScenarioRunInspectorView: View {
    let run: ScenarioRunDocument
    let filename: String
    @Environment(\.dismiss) private var dismiss
    private let horizon = Horizon()

    var body: some View {
        VStack(alignment: .leading, spacing: ZephyrDesign.Layout.panelSpacing) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: ZephyrDesign.Layout.tightSpacing) {
                    Text("Synthetic scenario run").font(ZephyrDesign.Typography.title)
                    Text("Read-only validated frames · \(run.scenarioID)").font(ZephyrDesign.Typography.secondary).foregroundStyle(horizon.mist)
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            Divider()
            VStack(alignment: .leading, spacing: ZephyrDesign.Layout.tightSpacing) {
                DataRow(label: "Filename", value: filename, horizon: horizon)
                DataRow(label: "Schema", value: run.schema, horizon: horizon)
                DataRow(label: "Seed", value: run.seedText, horizon: horizon)
                DataRow(label: "Frames", value: String(run.frames.count), horizon: horizon)
                DataRow(label: "Link events", value: String(run.linkEventCount), horizon: horizon)
                DataRow(label: "Profiles", value: String(run.profileCount), horizon: horizon)
                DataRow(label: "Status", value: run.status, horizon: horizon)
            }
            ScrollView {
                VStack(alignment: .leading, spacing: ZephyrDesign.Layout.tightSpacing) {
                    Text("Frames").font(ZephyrDesign.Typography.secondary).foregroundStyle(horizon.mist)
                    ForEach(run.frames) { frame in
                        DataRow(label: "Frame \(frame.stepIndex)", value: String(format: "t=%.3f s · active=%d · agents=%d", frame.time, frame.activeCount, frame.agentCount), horizon: horizon)
                    }
                    Text(run.evidenceBoundary).font(ZephyrDesign.Typography.caption).foregroundStyle(horizon.mist).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(ZephyrDesign.Spacing.lg)
        .frame(minWidth: 620, idealWidth: 760, minHeight: 520, idealHeight: 680)
        .background(horizon.night)
    }
}
