// GENERATED FILE. Do not edit. Regenerate with: python tools/generate_design.py
import AppKit
import SwiftUI

struct ScenePalette {
    let droneBody, droneArms, rotorDisc, targetRing, keepOutVolume, grid, floor: Color
    var droneBodyNS: NSColor { NSColor(droneBody) }
    var droneArmsNS: NSColor { NSColor(droneArms) }
    var rotorDiscNS: NSColor { NSColor(rotorDisc) }
    var targetRingNS: NSColor { NSColor(targetRing) }
    var keepOutVolumeNS: NSColor { NSColor(keepOutVolume) }
    var gridNS: NSColor { NSColor(grid) }
    var floorNS: NSColor { NSColor(floor) }
}

enum Theme {
    struct Palette {
        let background, surface, line, lineStrong, text, textSecondary: Color
        let accent, target, keepOut, ok, caution, alert, info, onAlert: Color
        let selectionFill, hoverFill: Color
        var backgroundNS: NSColor { NSColor(background) }
        var surfaceNS: NSColor { NSColor(surface) }
        var lineNS: NSColor { NSColor(line) }
        var lineStrongNS: NSColor { NSColor(lineStrong) }
        var textNS: NSColor { NSColor(text) }
        var textSecondaryNS: NSColor { NSColor(textSecondary) }
        var accentNS: NSColor { NSColor(accent) }
        var targetNS: NSColor { NSColor(target) }
        var keepOutNS: NSColor { NSColor(keepOut) }
        var okNS: NSColor { NSColor(ok) }
        var cautionNS: NSColor { NSColor(caution) }
        var alertNS: NSColor { NSColor(alert) }
        var infoNS: NSColor { NSColor(info) }
        var onAlertNS: NSColor { NSColor(onAlert) }
        var selectionFillNS: NSColor { NSColor(selectionFill) }
        var hoverFillNS: NSColor { NSColor(hoverFill) }
    }
    static let dark = Palette(
        background: Color(hex: "0F1522", opacity: 1),
        surface: Color(hex: "172033", opacity: 1),
        line: Color(hex: "2B3A55", opacity: 1),
        lineStrong: Color(hex: "5B6B88", opacity: 1),
        text: Color(hex: "EEF2F7", opacity: 1),
        textSecondary: Color(hex: "9AA8C0", opacity: 1),
        accent: Color(hex: "7CC4FF", opacity: 1),
        target: Color(hex: "FFB38A", opacity: 1),
        keepOut: Color(hex: "B4A7FF", opacity: 1),
        ok: Color(hex: "5FD39B", opacity: 1),
        caution: Color(hex: "FFC857", opacity: 1),
        alert: Color(hex: "FF5C5C", opacity: 1),
        info: Color(hex: "7CC4FF", opacity: 1),
        onAlert: Color(hex: "0F1522", opacity: 1),
        selectionFill: Color(hex: "273A54", opacity: 1),
        hoverFill: Color(hex: "242D3F", opacity: 1),
    )
    static let light = Palette(
        background: Color(hex: "F6F8FB", opacity: 1),
        surface: Color(hex: "FFFFFF", opacity: 1),
        line: Color(hex: "D5DCE6", opacity: 1),
        lineStrong: Color(hex: "7D8AA1", opacity: 1),
        text: Color(hex: "1B2433", opacity: 1),
        textSecondary: Color(hex: "56627A", opacity: 1),
        accent: Color(hex: "1F6FD1", opacity: 1),
        target: Color(hex: "A8501C", opacity: 1),
        keepOut: Color(hex: "5E4BC4", opacity: 1),
        ok: Color(hex: "1E7F4F", opacity: 1),
        caution: Color(hex: "8A5A00", opacity: 1),
        alert: Color(hex: "C62828", opacity: 1),
        info: Color(hex: "1F6FD1", opacity: 1),
        onAlert: Color(hex: "FFFFFF", opacity: 1),
        selectionFill: Color(hex: "E0EBF9", opacity: 1),
        hoverFill: Color(hex: "F4F4F5", opacity: 1),
    )
    static let darkScene3D = ScenePalette(
        droneBody: Color(hex: "9AA8C0", opacity: 1),
        droneArms: Color(hex: "9AA8C0", opacity: 1),
        rotorDisc: Color(hex: "9AA8C0", opacity: 1),
        targetRing: Color(hex: "FFB38A", opacity: 1),
        keepOutVolume: Color(hex: "B4A7FF", opacity: 0.18),
        grid: Color(hex: "2B3A55", opacity: 1),
        floor: Color(hex: "0F1522", opacity: 1),
    )
    static let lightScene3D = ScenePalette(
        droneBody: Color(hex: "56627A", opacity: 1),
        droneArms: Color(hex: "56627A", opacity: 1),
        rotorDisc: Color(hex: "56627A", opacity: 1),
        targetRing: Color(hex: "A8501C", opacity: 1),
        keepOutVolume: Color(hex: "5E4BC4", opacity: 0.18),
        grid: Color(hex: "D5DCE6", opacity: 1),
        floor: Color(hex: "F6F8FB", opacity: 1),
    )
    static func palette(for colorScheme: ColorScheme) -> Palette {
        colorScheme == .light ? light : dark
    }
    static func scene3D(for colorScheme: ColorScheme) -> ScenePalette {
        colorScheme == .light ? lightScene3D : darkScene3D
    }
}

private extension Color {
    init(hex: String, opacity: Double = 1) {
        let value = UInt64(hex, radix: 16) ?? 0
        self.init(.sRGB, red: Double((value >> 16) & 0xff) / 255,
            green: Double((value >> 8) & 0xff) / 255,
            blue: Double(value & 0xff) / 255, opacity: opacity)
    }
}

enum TypeScale {
    static let caption = Font.system(size: 12, weight: .regular)
    static let body = Font.system(size: 13, weight: .regular)
    static let bodyStrong = Font.system(size: 13, weight: .semibold)
    static let headline = Font.system(size: 15, weight: .semibold)
    static let title = Font.system(size: 20, weight: .semibold)
    static let metric = Font.system(size: 22, weight: .semibold).monospacedDigit()
    static let metricSmall = Font.system(size: 15, weight: .semibold).monospacedDigit()
    static let mono = Font.system(size: 12, weight: .regular, design: .monospaced)
}

enum Space {
    static let xxs: CGFloat = 2
    static let xs: CGFloat = 4
    static let sm: CGFloat = 8
    static let md: CGFloat = 12
    static let lg: CGFloat = 16
    static let xl: CGFloat = 24
    static let xxl: CGFloat = 32
}

enum Radius {
    static let small: CGFloat = 4
    static let medium: CGFloat = 8
}

enum Motion {
    static let fast: TimeInterval = 0.12
    static let standard: TimeInterval = 0.18
    static func duration(_ value: TimeInterval, reduceMotion: Bool) -> TimeInterval {
        reduceMotion ? 0 : value
    }
}

enum LayoutMetrics {
    static let windowMinWidth: CGFloat = 1180
    static let windowMinHeight: CGFloat = 760
    static let sidebarWidth: CGFloat = 260
    static let inspectorWidth: CGFloat = 300
    static let rowHeight: CGFloat = 28
    static let sceneMinWidth: CGFloat = 520
    static let windowDefaultWidth: CGFloat = 1440
    static let windowDefaultHeight: CGFloat = 900
    static let eventListHeight: CGFloat = 128
    static let inspectorMaxWidth: CGFloat = 360
    static let sidebarIdealWidth: CGFloat = 300
    static let eventPaneHeight: CGFloat = 192
}

enum UIText {
    static let honestyBadge = "Simulation replay"
    static let aboutButton = "About this replay"
    static let tracking = "Tracking"
    static let estimateErrorP95 = "Estimate error p95"
    static let targetLinkDelay = "Target link delay"
    static let targetPacketLoss = "Target packet loss"
    static let targetAge = "Target age"
    static let targetAgeLateS: Double = 0.15
    static let targetAgeLostS: Double = 0.3
    struct StatusLevel {
        let word: String
        let alternateWord: String?
        let shape: String
        let color: String
    }
    static let statusLevels: [String: StatusLevel] = [
        "ok": StatusLevel(word: "OK", alternateWord: nil, shape: "circle", color: "ok"),
        "caution": StatusLevel(word: "Late", alternateWord: nil, shape: "triangle", color: "caution"),
        "alert": StatusLevel(word: "Lost", alternateWord: "Inactive", shape: "square", color: "alert"),
        "waiting": StatusLevel(word: "Waiting", alternateWord: nil, shape: "hollow circle", color: "textSecondary"),
        "unavailable": StatusLevel(word: "Not recorded", alternateWord: nil, shape: "dash", color: "textSecondary"),
        "info": StatusLevel(word: "Info", alternateWord: nil, shape: "info", color: "info"),
    ]
    static let statusShapesMacOS: [String: String] = [
        "circle": "circle.fill",
        "triangle": "triangle.fill",
        "square": "square.fill",
        "hollow circle": "circle",
        "dash": "minus",
        "info": "info.circle",
    ]
    static let statusShapesWindows: [String: String] = [
        "circle": "●",
        "triangle": "▲",
        "square": "■",
        "hollow circle": "○",
        "dash": "–",
        "info": "ⓘ",
    ]
    static let fieldLabels: [String: String] = [
        "abstract_event_delay_s": "Abstract event delay (s)",
        "active": "Active",
        "active_agent_observations": "Active agent observations",
        "active_count": "Active drones",
        "agent_count": "Drone count",
        "agent_id": "Drone ID",
        "agent_profiles": "Agent profiles",
        "agents": "Agents",
        "aggregation": "Aggregation",
        "altitude_m": "Altitude (m)",
        "app_version": "App version",
        "arm_length_m": "Arm length (m)",
        "attempted": "Attempted",
        "available": "Available",
        "battery": "Battery",
        "battery_capacity_wh": "Battery capacity (Wh)",
        "bench": "Bench",
        "bracketed_intervals": "Bracketed intervals",
        "burst": "Burst",
        "burst_end_probability": "Burst end probability",
        "burst_start_probability": "Burst start probability",
        "calibration_id": "Calibration ID",
        "capture": "Capture",
        "censored_outages": "Censored outages",
        "censoring": "Censoring",
        "center_m": "Center (m)",
        "channel": "Channel",
        "clock_domain": "Clock domain",
        "clock_uncertainty_s": "Clock uncertainty (s)",
        "code_revision": "Code revision",
        "code_revision_semantics": "Code revision semantics",
        "color": "Color",
        "command_authority": "Command authority",
        "complete_outcome_denominator": "Complete outcome denominator",
        "confidence": "Confidence",
        "config": "Config",
        "constraint_flags": "Constraint flags",
        "constraints": "Constraints",
        "contention_mode": "Contention mode",
        "count": "Count",
        "criterion": "Criterion",
        "criterion_note": "Criterion note",
        "definition": "Definition",
        "delay": "Delay",
        "delay_jitter_s": "Delay jitter (s)",
        "delay_population": "Delay population",
        "delay_s": "Delay (s)",
        "denominator_contract": "Denominator contract",
        "device_ids": "Device IDs",
        "diameter_m": "Diameter (m)",
        "dt": "Frame interval",
        "dt_s": "Frame interval (s)",
        "duplicate": "Duplicate",
        "duplicates": "Duplicates",
        "duration_s": "Duration (s)",
        "espnow": "ESP-NOW",
        "estimate_error_m": "Estimate error",
        "estimated_max_thrust_n": "Estimated maximum thrust (N)",
        "estimated_thrust_at_rpm_limit_n": "Estimated thrust at RPM limit (N)",
        "estimation": "Estimation",
        "evidence_boundary": "Evidence boundary",
        "failsafe": "Failsafe",
        "failure_max_m": "Failure max (m)",
        "failure_p95_m": "Failure p95 (m)",
        "firmware": "Firmware",
        "fixture": "Fixture",
        "flag_counts": "Flag counts",
        "frame_count": "Frame count",
        "frame_mass_kg": "Frame mass (kg)",
        "frames": "Frames",
        "fused_target_m": "Fused target (m)",
        "fused_target_present": "Fused target present",
        "generation_command": "Generation command",
        "generator": "Generator",
        "ground": "Ground",
        "hardware": "Hardware",
        "hardware_profile": "Hardware profile",
        "heading_deg": "Heading",
        "held_last_max_error_m": "Held last max error (m)",
        "held_last_p95_error_m": "Held last p95 error (m)",
        "identity_fields": "Identity fields",
        "independent": "Independent",
        "independent_check": "Independent check",
        "interpretation": "Interpretation",
        "investigation": "Investigation",
        "keep_out_spheres": "Keep-out volumes",
        "keep_out_volumes": "Keep-out volumes",
        "kind": "Kind",
        "label": "Label",
        "limitations": "Limitations",
        "link_delay_ms": "Link delay (ms)",
        "link_event_count": "Link event count",
        "link_events": "Link events",
        "link_loss_pct": "Link packet loss (%)",
        "logical_identity_count": "Logical identity count",
        "loss_model": "Loss model",
        "loss_probability": "Loss probability",
        "loss_rate": "Loss rate",
        "loss_rate_bounds": "Loss rate bounds",
        "loss_reason": "Loss reason",
        "lost": "Lost",
        "lower": "Lower",
        "mass_kg": "Mass (kg)",
        "max": "Max",
        "max_error_m": "Max error (m)",
        "max_power_w": "Max power (W)",
        "max_rpm": "Maximum speed (rpm)",
        "max_torque_nm": "Max torque (N·m)",
        "mean": "Mean",
        "mean_age_s": "Mean age (s)",
        "measured": "Measured",
        "min": "Min",
        "min_neighbor_distance_m": "Nearest neighbor (m)",
        "model": "Model",
        "motor": "Motor",
        "motor_count": "Motor count",
        "motor_part_id": "Motor part",
        "name": "Name",
        "nearest_neighbor_m": "Nearest neighbor",
        "neighbor_count": "Neighbors in range",
        "neighbor_link": "Neighbor link",
        "neighbor_link_delay": "Neighbor link delay",
        "neighbor_link_delay_ms": "Neighbor link delay",
        "neighbor_packet_loss": "Neighbor packet loss",
        "neighbor_packet_loss_pct": "Neighbor packet loss",
        "noise_std_m": "Noise std (m)",
        "nominal_voltage_v": "Nominal voltage (V)",
        "none": "None",
        "notes": "Notes",
        "numpy": "NumPy version",
        "observation_period_s": "Observation period (s)",
        "observed_min_neighbor_distance_m": "Observed min neighbor distance (m)",
        "observed_transmission_records": "Observed transmission records",
        "out_of_order": "Out of order",
        "outcome": "Outcome",
        "p50": "P50",
        "p95": "P95",
        "p95_error_m": "P95 error (m)",
        "packet_age": "Packet age",
        "packet_duration_s": "Packet duration (s)",
        "packets": "Packets",
        "parameters": "Parameters",
        "parameters_canonical": "Parameters canonical",
        "part_id": "Part ID",
        "payload_canonical": "Payload canonical",
        "payload_sha256": "Payload SHA-256",
        "phy": "PHY",
        "pitch_m": "Pitch (m)",
        "position_m": "Position (m)",
        "power_coefficient": "Power coefficient",
        "presentation_schema": "Presentation schema",
        "profile_count": "Profile count",
        "profile_id": "Hardware profile",
        "profiles": "Profiles",
        "propeller": "Propeller",
        "propeller_diameter_m": "Propeller diameter (m)",
        "propeller_diameter_mm": "Propeller diameter (mm)",
        "propeller_part_id": "Propeller part",
        "provenance": "Provenance",
        "python": "Python version",
        "quantile_method": "Quantile method",
        "radius_m": "Radius (m)",
        "receive_time": "Receive time",
        "received": "Received",
        "receiver": "Receiver",
        "recovery": "Recovery",
        "report": "Report",
        "role": "Role",
        "routes": "Routes",
        "rows": "Rows",
        "rssi": "RSSI",
        "rssi_population": "RSSI population",
        "run": "Run",
        "run_id": "Run ID",
        "run_name": "Run",
        "run_schema": "Run schema",
        "run_sha256": "Run file SHA-256",
        "sample_count": "Sample count",
        "scenario_hash": "Scenario hash",
        "scenario_id": "Scenario ID",
        "schema": "Schema",
        "seed": "Random seed",
        "send_time": "Send time",
        "sender": "Sender",
        "separation": "Separation",
        "seq": "Sequence",
        "serialized": "Serialized",
        "sha256": "SHA-256",
        "sim": "Sim",
        "source": "Source",
        "source_schema": "Source schema",
        "source_status": "Source status",
        "speed_mps": "Speed",
        "state": "State",
        "status": "Status",
        "step_index": "Frame",
        "steps": "Frames",
        "summary": "Summary",
        "swarm": "Swarm",
        "synthetic": "Synthetic",
        "tag_name": "Tag name",
        "target_acceleration_mps2": "Target acceleration (m/s²)",
        "target_age": "Target age",
        "target_age_s": "Target age (s)",
        "target_estimate_m": "Target estimate (m)",
        "target_estimate_present": "Target estimate present",
        "target_keepout": "Target keep-out",
        "target_link": "Target link",
        "target_link_delay": "Target link delay",
        "target_link_delay_ms": "Target link delay",
        "target_packet_loss": "Target packet loss",
        "target_packet_loss_pct": "Target packet loss",
        "target_position_m": "Target position (m)",
        "target_source_time_s": "Target source time (s)",
        "target_speed_mps": "Target speed (m/s)",
        "target_velocity_mps": "Target velocity (m/s)",
        "thrust_coefficient": "Thrust coefficient",
        "time_s": "Time (s)",
        "time_span_s": "Time span (s)",
        "timestamp_basis": "Timestamp basis",
        "total_mass_kg": "Total mass (kg)",
        "trace_id": "Trace ID",
        "transport": "Transport",
        "truth_dt_s": "Truth dt (s)",
        "unavailable": "Unavailable",
        "unit": "Unit",
        "unknown": "Unknown",
        "unknown_reason_counts": "Unknown reason counts",
        "unknown_unresolved": "Unknown unresolved",
        "upper": "Upper",
        "velocity_mps": "Velocity (m/s)",
        "version": "Version",
        "vision": "Vision",
        "weighting": "Weighting",
    ]
    static let aboutBodyFields: [String] = ["status", "evidence_boundary", "command_authority", "failsafe", "generation_command", "seed", "scenario_id"]
    static let shortcutRule = "Single-key shortcuts never fire while a text field has focus."
    static let droneCompactName = "D{number:02d}"
    static let droneHeadingName = "Drone {number:02d}"
    static let targetAgeLateNote = "display candidates from the vault research (R-P5-UX-02), not safety limits"
    static let targetAgeLostNote = "display candidates from the vault research (R-P5-UX-02), not safety limits"
    struct Shortcut {
        let actionID: String
        let label: String
        let key: String
        let modifiers: [String]
        let singleKey: Bool
    }
    static let shortcuts: [Shortcut] = [
        Shortcut(actionID: "playPause", label: "Play/Pause", key: "space", modifiers: [], singleKey: true),
        Shortcut(actionID: "stepForward", label: "Step forward", key: "right", modifiers: [], singleKey: true),
        Shortcut(actionID: "stepBack", label: "Step back", key: "left", modifiers: [], singleKey: true),
        Shortcut(actionID: "restart", label: "Restart", key: "home", modifiers: [], singleKey: true),
        Shortcut(actionID: "focusSelected", label: "Focus selected drone", key: "f", modifiers: [], singleKey: true),
        Shortcut(actionID: "resetCamera", label: "Reset camera", key: "r", modifiers: [], singleKey: true),
        Shortcut(actionID: "toggleTopView", label: "Toggle 3D/top view", key: "t", modifiers: [], singleKey: true),
        Shortcut(actionID: "toggleKeepOut", label: "Show/hide keep-out volume", key: "k", modifiers: [], singleKey: true),
        Shortcut(actionID: "toggleInspector", label: "Show/hide inspector", key: "i", modifiers: [], singleKey: true),
        Shortcut(actionID: "openEvidenceReport", label: "Open evidence report...", key: "o", modifiers: ["primary"], singleKey: false),
        Shortcut(actionID: "openScenarioRun", label: "Open scenario run...", key: "o", modifiers: ["primary", "shift"], singleKey: false),
        Shortcut(actionID: "focusFilter", label: "Focus team filter", key: "l", modifiers: ["primary"], singleKey: false),
        Shortcut(actionID: "aboutReplay", label: "About this replay", key: "i", modifiers: ["primary"], singleKey: false),
    ]
    static let eventReplayLoaded = "Replay loaded: {count} drones"
    static let eventEstimateReached = "Target estimate reached {estimated} of {total} drones"
    static let eventEstimateLate = "{count} drones newly late: {drones} ({age:.2f} s)"
    static let eventBackOnTime = "{count} drones back on time"
    static let eventInactive = "{drone} went inactive"
    static let eventPacketLost = "Target packet lost for {count} drones"
}
