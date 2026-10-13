import Foundation
import simd

// Strict decoder for the bundled replay fixture built by tools/build_native_fixture.py.
// Every object has an exact key set: a missing or unknown key fails with a message that
// names the object, so a schema drift on the Python side cannot be silently ignored here.

struct FixtureDecodeError: LocalizedError, Equatable {
    let message: String
    var errorDescription: String? { message }
}

/// Rejects missing and unknown JSON keys. A typed `KeyedDecodingContainer` drops keys that
/// have no matching `CodingKey` case from `allKeys`, so the raw key set is read through a
/// dynamic key container instead.
private func requireKeys(_ decoder: Decoder, expected: Set<String>, context: String) throws {
    let actual = Set(try decoder.container(keyedBy: DynamicCodingKey.self).allKeys.map(\.stringValue))
    guard actual == expected else {
        let missing = expected.subtracting(actual).sorted().joined(separator: ", ")
        let extra = actual.subtracting(expected).sorted().joined(separator: ", ")
        var details: [String] = []
        if !missing.isEmpty { details.append("missing \(missing)") }
        if !extra.isEmpty { details.append("unknown \(extra)") }
        throw FixtureDecodeError(message: "invalid \(context) keys (\(details.joined(separator: "; ")))")
    }
}

private func requireFinite(_ value: Double, field: String, minimum: Double? = nil) throws {
    guard value.isFinite, minimum.map({ value >= $0 }) ?? true else {
        throw FixtureDecodeError(message: "\(field) must be a finite number")
    }
}

private func requireFinite(_ vector: SIMD3<Float>, field: String) throws {
    guard vector.x.isFinite, vector.y.isFinite, vector.z.isFinite else {
        throw FixtureDecodeError(message: "\(field) must contain finite numbers")
    }
}

/// One drone's raw state in one frame, projected from the canonical run. Used for the scene
/// geometry and the raw vectors under the inspector Details; every displayed metric comes
/// from `PresentationDrone` instead.
struct DroneTelemetry: Identifiable, Decodable, Equatable {
    let id: String
    let position: SIMD3<Float>
    let velocity: SIMD3<Float>
    let profileID: String
    let active: Bool
    /// `nil` when the run did not record it; never read as zero neighbors.
    let neighborCount: Int?
    let minNeighborDistance: Double?
    let constraintFlags: [String]
    let targetEstimate: SIMD3<Float>?
    let fusedTarget: SIMD3<Float>?
    let targetSourceTime: Double?
    let targetAge: Double?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case id, position, velocity, active
        case profileID = "profile_id"
        case neighborCount = "neighbor_count"
        case minNeighborDistance = "min_neighbor_distance_m"
        case constraintFlags = "constraint_flags"
        case targetEstimate = "target_estimate_m"
        case fusedTarget = "fused_target_m"
        case targetSourceTime = "target_source_time_s"
        case targetAge = "target_age_s"
    }

    init(from decoder: Decoder) throws {
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "drone")
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(String.self, forKey: .id)
        position = try values.decode(SIMD3<Float>.self, forKey: .position)
        velocity = try values.decode(SIMD3<Float>.self, forKey: .velocity)
        profileID = try values.decode(String.self, forKey: .profileID)
        active = try values.decode(Bool.self, forKey: .active)
        neighborCount = try values.decodeIfPresent(Int.self, forKey: .neighborCount)
        minNeighborDistance = try values.decodeIfPresent(Double.self, forKey: .minNeighborDistance)
        constraintFlags = try values.decode([String].self, forKey: .constraintFlags)
        targetEstimate = try values.decodeIfPresent(SIMD3<Float>.self, forKey: .targetEstimate)
        fusedTarget = try values.decodeIfPresent(SIMD3<Float>.self, forKey: .fusedTarget)
        targetSourceTime = try values.decodeIfPresent(Double.self, forKey: .targetSourceTime)
        targetAge = try values.decodeIfPresent(Double.self, forKey: .targetAge)
        guard !id.isEmpty, !profileID.isEmpty, (neighborCount ?? 0) >= 0 else {
            throw FixtureDecodeError(message: "drone identity or neighbor count is invalid")
        }
        try requireFinite(position, field: "drone.position")
        try requireFinite(velocity, field: "drone.velocity")
        if let minNeighborDistance { try requireFinite(minNeighborDistance, field: "drone.min_neighbor_distance_m", minimum: 0) }
    }
}

/// A keep-out sphere the simulator enforced in one frame.
struct KeepOutVolume: Decodable, Equatable {
    let center: SIMD3<Float>
    let radiusM: Double
    let label: String

    enum CodingKeys: String, CodingKey, CaseIterable {
        case center = "center_m"
        case radiusM = "radius_m"
        case label
    }

    init(from decoder: Decoder) throws {
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "keep-out volume")
        let values = try decoder.container(keyedBy: CodingKeys.self)
        center = try values.decode(SIMD3<Float>.self, forKey: .center)
        radiusM = try values.decode(Double.self, forKey: .radiusM)
        label = try values.decode(String.self, forKey: .label)
        try requireFinite(center, field: "keep-out center")
        try requireFinite(radiusM, field: "keep-out radius", minimum: 0)
        guard !label.isEmpty else { throw FixtureDecodeError(message: "keep-out label must not be empty") }
    }
}

/// Hardware profile parts listed under the inspector Details. Scenario inputs, not measurements.
struct HardwareProfile: Decodable, Equatable, Identifiable {
    let profileID: String
    let status: String?
    let motorCount: Int
    let armLengthM: Double
    let totalMassKg: Double
    let motorPartID: String
    let propellerPartID: String
    let propellerDiameterM: Double
    var id: String { profileID }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case profileID = "profile_id"
        case status
        case motorCount = "motor_count"
        case armLengthM = "arm_length_m"
        case totalMassKg = "total_mass_kg"
        case motorPartID = "motor_part_id"
        case propellerPartID = "propeller_part_id"
        case propellerDiameterM = "propeller_diameter_m"
    }

    init(from decoder: Decoder) throws {
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "hardware profile")
        let values = try decoder.container(keyedBy: CodingKeys.self)
        profileID = try values.decode(String.self, forKey: .profileID)
        status = try values.decodeIfPresent(String.self, forKey: .status)
        motorCount = try values.decode(Int.self, forKey: .motorCount)
        armLengthM = try values.decode(Double.self, forKey: .armLengthM)
        totalMassKg = try values.decode(Double.self, forKey: .totalMassKg)
        motorPartID = try values.decode(String.self, forKey: .motorPartID)
        propellerPartID = try values.decode(String.self, forKey: .propellerPartID)
        propellerDiameterM = try values.decode(Double.self, forKey: .propellerDiameterM)
        guard !profileID.isEmpty, motorCount > 0 else {
            throw FixtureDecodeError(message: "hardware profile identity or motor count is invalid")
        }
        try requireFinite(armLengthM, field: "profile arm length", minimum: 0)
        try requireFinite(totalMassKg, field: "profile mass", minimum: 0)
        try requireFinite(propellerDiameterM, field: "profile propeller diameter", minimum: 0)
        guard armLengthM > 0, propellerDiameterM > 0 else {
            throw FixtureDecodeError(message: "hardware profile \(profileID) needs a positive arm length and propeller diameter")
        }
    }
}

/// Where the fixture came from: the canonical run file's schema and digest.
struct FixtureSource: Decodable, Equatable {
    let runSchema: String
    let runSHA256: String
    let presentationSchema: String

    enum CodingKeys: String, CodingKey, CaseIterable {
        case runSchema = "run_schema"
        case runSHA256 = "run_sha256"
        case presentationSchema = "presentation_schema"
    }

    init(from decoder: Decoder) throws {
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "source")
        let values = try decoder.container(keyedBy: CodingKeys.self)
        runSchema = try values.decode(String.self, forKey: .runSchema)
        runSHA256 = try values.decode(String.self, forKey: .runSHA256)
        presentationSchema = try values.decode(String.self, forKey: .presentationSchema)
        guard runSHA256.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil else {
            throw FixtureDecodeError(message: "source run_sha256 must be 64 lowercase hexadecimal characters")
        }
    }
}

struct PresentationEvent: Codable, Equatable, Identifiable {
    let droneIDs: [String]
    let level: String
    let text: String
    let timeS: Double
    var id: String { "\(timeS)-\(level)-\(text)" }

    enum CodingKeys: String, CodingKey, CaseIterable { case droneIDs = "drone_ids", level, text, timeS = "time_s" }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "presentation event")
        droneIDs = try values.decode([String].self, forKey: .droneIDs)
        level = try values.decode(String.self, forKey: .level)
        text = try values.decode(String.self, forKey: .text)
        timeS = try values.decode(Double.self, forKey: .timeS)
        try requireFinite(timeS, field: "presentation event time_s", minimum: 0)
    }
}

enum PresentationMetric: Codable, Equatable {
    case value(Double)
    case range(min: Double, max: Double)

    init(from decoder: Decoder) throws {
        if let scalar = try? decoder.singleValueContainer().decode(Double.self) {
            guard scalar.isFinite else { throw FixtureDecodeError(message: "presentation metric must be finite") }
            self = .value(scalar)
            return
        }
        let values = try decoder.container(keyedBy: RangeKeys.self)
        try requireKeys(decoder, expected: Set(RangeKeys.allCases.map(\.stringValue)), context: "presentation metric")
        let minimum = try values.decode(Double.self, forKey: .min)
        let maximum = try values.decode(Double.self, forKey: .max)
        guard minimum.isFinite, maximum.isFinite, minimum <= maximum else {
            throw FixtureDecodeError(message: "presentation metric range is invalid")
        }
        self = .range(min: minimum, max: maximum)
    }

    func encode(to encoder: Encoder) throws {
        switch self {
        case .value(let number):
            var container = encoder.singleValueContainer()
            try container.encode(number)
        case .range(let minimum, let maximum):
            var container = encoder.container(keyedBy: RangeKeys.self)
            try container.encode(minimum, forKey: .min)
            try container.encode(maximum, forKey: .max)
        }
    }

    private enum RangeKeys: String, CodingKey, CaseIterable { case min, max }
}

struct PresentationGroup: Codable, Equatable, Identifiable {
    let color: String
    let count: Int
    let droneID: String
    let droneIDs: [String]
    let level: String
    let shape: String
    /// One age when every member agrees, otherwise the members' min to max range.
    let targetAgeS: PresentationMetric?
    let targetLinkDelayMs: PresentationMetric?
    let targetPacketLossPct: PresentationMetric?
    let word: String
    var id: String { droneID }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case color, count, droneID = "drone_id", droneIDs = "drone_ids", level, shape
        case targetAgeS = "target_age_s", targetLinkDelayMs = "target_link_delay_ms"
        case targetPacketLossPct = "target_packet_loss_pct", word
    }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "presentation group")
        color = try values.decode(String.self, forKey: .color)
        count = try values.decode(Int.self, forKey: .count)
        droneID = try values.decode(String.self, forKey: .droneID)
        droneIDs = try values.decode([String].self, forKey: .droneIDs)
        level = try values.decode(String.self, forKey: .level)
        shape = try values.decode(String.self, forKey: .shape)
        targetAgeS = try values.decodeIfPresent(PresentationMetric.self, forKey: .targetAgeS)
        targetLinkDelayMs = try values.decodeIfPresent(PresentationMetric.self, forKey: .targetLinkDelayMs)
        targetPacketLossPct = try values.decodeIfPresent(PresentationMetric.self, forKey: .targetPacketLossPct)
        word = try values.decode(String.self, forKey: .word)
        guard count >= 0, count == droneIDs.count, !droneID.isEmpty else {
            throw FixtureDecodeError(message: "presentation group count or identity is invalid")
        }
    }
}

struct PresentationHeadline: Codable, Equatable {
    let level: String
    let text: String
    enum CodingKeys: String, CodingKey, CaseIterable { case level, text }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "presentation headline")
        level = try values.decode(String.self, forKey: .level)
        text = try values.decode(String.self, forKey: .text)
    }
}

struct PresentationDrone: Codable, Equatable, Identifiable {
    let active: Bool
    let altitudeM: Double
    let color: String
    let compactName: String
    let constraintFlags: [String]
    let droneID: String
    let estimateErrorM: Double?
    let headingDeg: Double?
    let headingName: String
    let level: String
    let nearestNeighborM: Double?
    /// `nil` when the run did not record it; shown as "Not recorded", never 0.
    let neighborCount: Int?
    let neighborLinkDelayMs: Double?
    let neighborPacketLossPct: Double?
    let profileID: String
    let shape: String
    let speedMps: Double
    let targetAgeS: Double?
    let targetLinkDelayMs: Double?
    let targetPacketLossPct: Double?
    let word: String
    var id: String { droneID }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case active
        case altitudeM = "altitude_m"
        case color
        case compactName = "compact_name"
        case constraintFlags = "constraint_flags"
        case droneID = "drone_id"
        case estimateErrorM = "estimate_error_m"
        case headingDeg = "heading_deg"
        case headingName = "heading_name"
        case level
        case nearestNeighborM = "nearest_neighbor_m"
        case neighborCount = "neighbor_count"
        case neighborLinkDelayMs = "neighbor_link_delay_ms"
        case neighborPacketLossPct = "neighbor_packet_loss_pct"
        case profileID = "profile_id"
        case shape
        case speedMps = "speed_mps"
        case targetAgeS = "target_age_s"
        case targetLinkDelayMs = "target_link_delay_ms"
        case targetPacketLossPct = "target_packet_loss_pct"
        case word
    }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "presentation drone")
        active = try values.decode(Bool.self, forKey: .active)
        altitudeM = try values.decode(Double.self, forKey: .altitudeM)
        color = try values.decode(String.self, forKey: .color)
        compactName = try values.decode(String.self, forKey: .compactName)
        constraintFlags = try values.decode([String].self, forKey: .constraintFlags)
        droneID = try values.decode(String.self, forKey: .droneID)
        estimateErrorM = try values.decodeIfPresent(Double.self, forKey: .estimateErrorM)
        headingDeg = try values.decodeIfPresent(Double.self, forKey: .headingDeg)
        headingName = try values.decode(String.self, forKey: .headingName)
        level = try values.decode(String.self, forKey: .level)
        nearestNeighborM = try values.decodeIfPresent(Double.self, forKey: .nearestNeighborM)
        neighborCount = try values.decodeIfPresent(Int.self, forKey: .neighborCount)
        neighborLinkDelayMs = try values.decodeIfPresent(Double.self, forKey: .neighborLinkDelayMs)
        neighborPacketLossPct = try values.decodeIfPresent(Double.self, forKey: .neighborPacketLossPct)
        profileID = try values.decode(String.self, forKey: .profileID)
        shape = try values.decode(String.self, forKey: .shape)
        speedMps = try values.decode(Double.self, forKey: .speedMps)
        targetAgeS = try values.decodeIfPresent(Double.self, forKey: .targetAgeS)
        targetLinkDelayMs = try values.decodeIfPresent(Double.self, forKey: .targetLinkDelayMs)
        targetPacketLossPct = try values.decodeIfPresent(Double.self, forKey: .targetPacketLossPct)
        word = try values.decode(String.self, forKey: .word)
        guard !droneID.isEmpty, (neighborCount ?? 0) >= 0 else {
            throw FixtureDecodeError(message: "presentation drone identity or neighbor count is invalid")
        }
        try requireFinite(altitudeM, field: "presentation altitude")
        if let nearestNeighborM { try requireFinite(nearestNeighborM, field: "presentation nearest neighbor", minimum: 0) }
        try requireFinite(speedMps, field: "presentation speed", minimum: 0)
    }
}

struct PresentationStrip: Codable, Equatable {
    let neighborLinkDelayMs: Double?
    let neighborPacketLossPct: Double?
    let newestTargetAgeS: Double?
    let oldestTargetAgeS: Double?
    let targetLinkDelayMs: Double?
    let targetPacketLossPct: Double?
    enum CodingKeys: String, CodingKey, CaseIterable {
        case neighborLinkDelayMs = "neighbor_link_delay_ms"
        case neighborPacketLossPct = "neighbor_packet_loss_pct"
        case newestTargetAgeS = "newest_target_age_s"
        case oldestTargetAgeS = "oldest_target_age_s"
        case targetLinkDelayMs = "target_link_delay_ms"
        case targetPacketLossPct = "target_packet_loss_pct"
    }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "presentation strip")
        neighborLinkDelayMs = try values.decodeIfPresent(Double.self, forKey: .neighborLinkDelayMs)
        neighborPacketLossPct = try values.decodeIfPresent(Double.self, forKey: .neighborPacketLossPct)
        newestTargetAgeS = try values.decodeIfPresent(Double.self, forKey: .newestTargetAgeS)
        oldestTargetAgeS = try values.decodeIfPresent(Double.self, forKey: .oldestTargetAgeS)
        targetLinkDelayMs = try values.decodeIfPresent(Double.self, forKey: .targetLinkDelayMs)
        targetPacketLossPct = try values.decodeIfPresent(Double.self, forKey: .targetPacketLossPct)
    }
}

struct FixturePresentation: Codable, Equatable {
    let estimateErrorP95M: Double?
    let events: [PresentationEvent]
    let frame: Int
    let groups: [PresentationGroup]
    let headline: PresentationHeadline
    let perDrone: [PresentationDrone]
    let strip: PresentationStrip
    /// "50 drones: 47 OK, 3 late", computed once in desktop/presentation.py.
    let teamSummary: String
    let timeS: Double
    enum CodingKeys: String, CodingKey, CaseIterable {
        case estimateErrorP95M = "estimate_error_p95_m"
        case events, frame, groups, headline
        case perDrone = "per_drone"
        case strip
        case teamSummary = "team_summary"
        case timeS = "time_s"
    }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "presentation")
        estimateErrorP95M = try values.decodeIfPresent(Double.self, forKey: .estimateErrorP95M)
        events = try values.decode([PresentationEvent].self, forKey: .events)
        frame = try values.decode(Int.self, forKey: .frame)
        groups = try values.decode([PresentationGroup].self, forKey: .groups)
        headline = try values.decode(PresentationHeadline.self, forKey: .headline)
        perDrone = try values.decode([PresentationDrone].self, forKey: .perDrone)
        strip = try values.decode(PresentationStrip.self, forKey: .strip)
        teamSummary = try values.decode(String.self, forKey: .teamSummary)
        timeS = try values.decode(Double.self, forKey: .timeS)
        guard frame >= 0, !teamSummary.isEmpty else {
            throw FixtureDecodeError(message: "presentation frame or team summary is invalid")
        }
        try requireFinite(timeS, field: "presentation time_s", minimum: 0)
    }
}

struct FixtureAbout: Codable, Equatable {
    let commandAuthority: String
    let evidenceBoundary: String
    let failsafe: String
    let generationCommand: String
    let scenarioID: String
    let seed: Int
    let status: String
    enum CodingKeys: String, CodingKey, CaseIterable {
        case commandAuthority = "command_authority"
        case evidenceBoundary = "evidence_boundary"
        case failsafe
        case generationCommand = "generation_command"
        case scenarioID = "scenario_id"
        case seed, status
    }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "about")
        commandAuthority = try values.decode(String.self, forKey: .commandAuthority)
        evidenceBoundary = try values.decode(String.self, forKey: .evidenceBoundary)
        failsafe = try values.decode(String.self, forKey: .failsafe)
        generationCommand = try values.decode(String.self, forKey: .generationCommand)
        scenarioID = try values.decode(String.self, forKey: .scenarioID)
        seed = try values.decode(Int.self, forKey: .seed)
        status = try values.decode(String.self, forKey: .status)
    }
}

struct FixtureFrame: Decodable, Equatable {
    let stepIndex: Int
    let time: Double
    let targetPosition: SIMD3<Float>?
    let activeCount: Int
    let keepOutVolumes: [KeepOutVolume]
    let drones: [DroneTelemetry]
    let presentation: FixturePresentation

    enum CodingKeys: String, CodingKey, CaseIterable {
        case activeCount = "active_count"
        case drones, presentation
        case keepOutVolumes = "keep_out_volumes"
        case stepIndex = "step_index"
        case targetPosition = "target_position"
        case time
    }

    init(from decoder: Decoder) throws {
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "frame")
        let values = try decoder.container(keyedBy: CodingKeys.self)
        stepIndex = try values.decode(Int.self, forKey: .stepIndex)
        time = try values.decode(Double.self, forKey: .time)
        targetPosition = try values.decodeIfPresent(SIMD3<Float>.self, forKey: .targetPosition)
        activeCount = try values.decode(Int.self, forKey: .activeCount)
        keepOutVolumes = try values.decode([KeepOutVolume].self, forKey: .keepOutVolumes)
        drones = try values.decode([DroneTelemetry].self, forKey: .drones)
        presentation = try values.decode(FixturePresentation.self, forKey: .presentation)
        guard stepIndex >= 0, activeCount >= 0, activeCount <= drones.count, presentation.frame == stepIndex else {
            throw FixtureDecodeError(message: "frame index or active count is invalid")
        }
        try requireFinite(time, field: "frame time", minimum: 0)
        if let targetPosition { try requireFinite(targetPosition, field: "frame target position") }
        guard keepOutVolumes.count <= 64 else { throw FixtureDecodeError(message: "frame keep-out volumes are unbounded") }
        guard presentation.perDrone.map(\.droneID) == drones.map(\.id) else {
            throw FixtureDecodeError(message: "presentation per_drone does not match frame drones")
        }
    }
}

struct Fixture: Decodable, Equatable {
    static let schema = "zephyr-s7-native-fixture-4"

    let schema: String
    let runID: String
    let runName: String
    let source: FixtureSource
    /// What the replay does not contain, from the presentation's `not_in_replay`.
    let notInReplay: [String]
    let profiles: [HardwareProfile]
    let seed: Int
    let commandAuthority: String
    let failsafe: String
    let dt: Double
    let evidenceBoundary: String
    let generationCommand: String
    let status: String
    let about: FixtureAbout
    let drones: [DroneTelemetry]
    let frames: [FixtureFrame]

    enum CodingKeys: String, CodingKey, CaseIterable {
        case about
        case commandAuthority = "command_authority"
        case drones
        case dt = "dt_s"
        case evidenceBoundary = "evidence_boundary"
        case failsafe
        case frames
        case generationCommand = "generation_command"
        case notInReplay = "not_in_replay"
        case profiles
        case runID = "run_id"
        case runName = "run_name"
        case schema
        case seed
        case source
        case status
    }

    init(from decoder: Decoder) throws {
        try requireKeys(decoder, expected: Set(CodingKeys.allCases.map(\.stringValue)), context: "fixture")
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schema = try values.decode(String.self, forKey: .schema)
        runID = try values.decode(String.self, forKey: .runID)
        runName = try values.decode(String.self, forKey: .runName)
        source = try values.decode(FixtureSource.self, forKey: .source)
        notInReplay = try values.decode([String].self, forKey: .notInReplay)
        profiles = try values.decode([HardwareProfile].self, forKey: .profiles)
        seed = try values.decode(Int.self, forKey: .seed)
        commandAuthority = try values.decode(String.self, forKey: .commandAuthority)
        failsafe = try values.decode(String.self, forKey: .failsafe)
        dt = try values.decode(Double.self, forKey: .dt)
        evidenceBoundary = try values.decode(String.self, forKey: .evidenceBoundary)
        generationCommand = try values.decode(String.self, forKey: .generationCommand)
        status = try values.decode(String.self, forKey: .status)
        about = try values.decode(FixtureAbout.self, forKey: .about)
        drones = try values.decode([DroneTelemetry].self, forKey: .drones)
        frames = try values.decode([FixtureFrame].self, forKey: .frames)
        guard schema == Self.schema, !runID.isEmpty, !runName.isEmpty, !drones.isEmpty, !frames.isEmpty else {
            throw FixtureDecodeError(message: "fixture schema, run ID, run name, drones, or frames is invalid")
        }
        guard Set(drones.map(\.id)).count == drones.count else {
            throw FixtureDecodeError(message: "fixture contains duplicate drone IDs")
        }
        guard Set(profiles.map(\.profileID)).count == profiles.count else {
            throw FixtureDecodeError(message: "fixture contains duplicate hardware profile IDs")
        }
        let profileIDs = Set(profiles.map(\.profileID))
        if let orphan = drones.first(where: { !profileIDs.contains($0.profileID) }) {
            throw FixtureDecodeError(
                message: "drone \(orphan.id) references hardware profile \(orphan.profileID), which the fixture does not contain")
        }
        let ids = drones.map(\.id)
        let profileOrder = drones.map(\.profileID)
        guard frames.allSatisfy({ $0.drones.map(\.id) == ids && $0.drones.map(\.profileID) == profileOrder }) else {
            throw FixtureDecodeError(message: "frame drones do not match the fixture drone list")
        }
        guard frames.map(\.stepIndex) == Array(frames.indices), frames.map(\.time).sorted() == frames.map(\.time) else {
            throw FixtureDecodeError(message: "fixture frames are not ordered")
        }
        try requireFinite(dt, field: "fixture dt", minimum: 0)
    }

    func profile(id: String) -> HardwareProfile? { profiles.first { $0.profileID == id } }
}

enum FixtureLoader {
    /// Loads the bundled replay. A decode failure is returned to the caller so the window can
    /// show it; the app never substitutes invented drones for a fixture it could not read.
    static func loadBundled() -> Result<Fixture, Error> {
        guard let url = Bundle.module.url(forResource: "demo_swarm", withExtension: "json") else {
            return .failure(FixtureDecodeError(message: "The bundled replay file is missing from the app."))
        }
        return Result { try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url)) }
    }
}
