import SwiftUI

/// Selected-drone inspector: header, the four numbers that matter, motion, neighbors, Details.
struct InspectorView: View {
    private static let estimateErrorLabel = Format.label("estimate_error_m")
    @ObservedObject var model: CockpitModel
    @Environment(\.palette) private var palette
    @AppStorage("zephyr.mesh.inspector.details") private var showDetails = false

    var body: some View {
        Group {
            if let drone = model.selectedDrone {
                ScrollView {
                    VStack(alignment: .leading, spacing: Space.lg) {
                        header(drone)
                        tiles(drone)
                        motion(drone)
                        neighbors(drone)
                        details(drone)
                    }
                    // Top edge matches the sidebar summary and the status strip (Space.md).
                    .padding(.horizontal, Space.lg)
                    .padding(.top, Space.md)
                    .padding(.bottom, Space.lg)
                }
            } else {
                emptyState
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(palette.surface)
    }

    private func header(_ drone: PresentationDrone) -> some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            Text(drone.headingName)
                .font(TypeScale.title)
                .foregroundStyle(palette.text)
            StatusLabel(style: StatusStyle(drone: drone))
        }
        .accessibilityElement(children: .combine)
    }

    /// Same order as the status strip: error, delay, loss, age.
    private func tiles(_ drone: PresentationDrone) -> some View {
        Grid(horizontalSpacing: Space.sm, verticalSpacing: Space.sm) {
            GridRow {
                MetricTile(label: Self.estimateErrorLabel, value: Format.meters(drone.estimateErrorM))
                MetricTile(label: UIText.targetLinkDelay, value: Format.milliseconds(drone.targetLinkDelayMs))
            }
            GridRow {
                MetricTile(label: UIText.targetPacketLoss, value: Format.percent(drone.targetPacketLossPct))
                MetricTile(label: UIText.targetAge, value: Format.seconds(drone.targetAgeS))
            }
        }
    }

    private func motion(_ drone: PresentationDrone) -> some View {
        VStack(alignment: .leading, spacing: Space.xxs) {
            SectionHeader(title: UIText.Inspector.positionAndMotion)
            ValueRow(label: UIText.Inspector.altitude, value: Format.meters(drone.altitudeM))
            ValueRow(label: Format.label("speed_mps"), value: Format.metersPerSecond(drone.speedMps))
            ValueRow(label: Format.label("heading_deg"), value: Format.heading(drone.headingDeg, speed: drone.speedMps))
        }
    }

    private func neighbors(_ drone: PresentationDrone) -> some View {
        VStack(alignment: .leading, spacing: Space.xxs) {
            SectionHeader(title: UIText.Inspector.neighbors)
            ValueRow(label: Format.label("neighbor_count"), value: Format.integer(drone.neighborCount))
            ValueRow(label: Format.label("nearest_neighbor_m"), value: Format.meters(drone.nearestNeighborM))
            ValueRow(label: Format.label("neighbor_link_delay_ms"), value: Format.milliseconds(drone.neighborLinkDelayMs))
            ValueRow(label: Format.label("neighbor_packet_loss_pct"), value: Format.percent(drone.neighborPacketLossPct))
        }
    }

    private func details(_ drone: PresentationDrone) -> some View {
        DetailsSection(isExpanded: $showDetails) {
                if let telemetry = model.selectedTelemetry {
                    StackedValue(label: Format.label("position_m"), value: Format.vector(telemetry.position))
                    StackedValue(label: Format.label("velocity_mps"), value: Format.vector(telemetry.velocity))
                    StackedValue(label: Format.label("fused_target_m"), value: Format.vector(telemetry.fusedTarget))
                }
                StackedValue(label: Format.label("constraint_flags"), value: flags(drone.constraintFlags))
                StackedValue(label: Format.label("profile_id"), value: drone.profileID, style: .identifier)
                if let profile = model.fixture.profile(id: drone.profileID) {
                    ProfileParts(profile: profile)
                }
                StackedValue(label: Format.label("agent_id"), value: drone.droneID, style: .identifier)
        }
    }

    private func flags(_ flags: [String]) -> String {
        flags.isEmpty ? Format.label("none") : flags.map(Format.label).joined(separator: ", ")
    }

    private var emptyState: some View {
        VStack(spacing: Space.sm) {
            Image(systemName: "scope")
                .font(TypeScale.title)
                .foregroundStyle(palette.textSecondary)
            Text(UIText.Inspector.emptyTitle)
                .font(TypeScale.bodyStrong)
            Text(UIText.Inspector.emptyHint)
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
                .multilineTextAlignment(.center)
        }
        .padding(Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Hardware profile parts. These are scenario inputs, and the status word says so.
private struct ProfileParts: View {
    private static let title = Format.label("hardware_profile")
    private static let diameterLabel = Format.label("propeller_diameter_mm")
    let profile: HardwareProfile
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xxs) {
            Text(Self.title)
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
            if let status = profile.status {
                ValueRow(label: Format.label("status"), value: status.capitalized)
            }
            ValueRow(label: Format.label("motor_part_id"), value: profile.motorPartID, style: .identifier)
            ValueRow(label: Format.label("propeller_part_id"), value: profile.propellerPartID, style: .identifier)
            ValueRow(label: Format.label("motor_count"), value: Format.integer(profile.motorCount))
            ValueRow(label: Format.label("arm_length_m"), value: Format.number(profile.armLengthM, decimals: 2))
            // Shown in millimetres: at 2 decimals in metres the five profile diameters would all read 0.08.
            ValueRow(label: Self.diameterLabel, value: Format.number(profile.propellerDiameterM * 1000, decimals: 1))
            ValueRow(label: Format.label("total_mass_kg"), value: Format.number(profile.totalMassKg, decimals: 3))
        }
    }
}
