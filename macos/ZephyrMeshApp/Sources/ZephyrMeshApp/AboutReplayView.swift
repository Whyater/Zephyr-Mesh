import AppKit
import SwiftUI

/// What this replay is and is not: evidence boundary, status, authority, failsafe, what the
/// data does not contain, how to regenerate it, and under Details the hashes and versions.
struct AboutReplayView: View {
    let fixture: Fixture
    @Environment(\.palette) private var palette
    @State private var showDetails = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Space.xl) {
                VStack(alignment: .leading, spacing: Space.xs) {
                    Text(UIText.aboutButton)
                        .font(TypeScale.title)
                    Text(fixture.runName)
                        .font(TypeScale.body)
                        .foregroundStyle(palette.textSecondary)
                }
                section("What this is") {
                    StackedValue(label: Format.label("status"), value: sentence(fixture.about.status))
                    StackedValue(label: Format.label("evidence_boundary"), value: sentence(fixture.about.evidenceBoundary))
                }
                section("Control and safety") {
                    StackedValue(label: Format.label("command_authority"), value: sentence(fixture.about.commandAuthority))
                    StackedValue(label: Format.label("failsafe"), value: sentence(fixture.about.failsafe))
                }
                section("Not in this replay") {
                    VStack(alignment: .leading, spacing: Space.xs) {
                        ForEach(fixture.unavailable, id: \.self) { item in
                            HStack(alignment: .firstTextBaseline, spacing: Space.sm) {
                                StatusShape(style: StatusStyle(level: "unavailable"))
                                Text(item)
                                    .font(TypeScale.body)
                            }
                        }
                    }
                }
                section("Reproduce") {
                    HStack(alignment: .top, spacing: Space.sm) {
                        StackedValue(label: Format.label("generation_command"), value: fixture.about.generationCommand,
                                     style: .code)
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(fixture.about.generationCommand, forType: .string)
                        } label: {
                            Label("Copy command", systemImage: "doc.on.doc").labelStyle(.iconOnly)
                        }
                        .buttonStyle(QuietButtonStyle())
                        .help("Copy command")
                    }
                }
                details
            }
            .padding(Space.xl)
            .frame(maxWidth: 640, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minWidth: 480, idealWidth: 560, minHeight: 420, idealHeight: 640)
        .background(palette.background)
        .toolbarBackground(palette.background, for: .windowToolbar)
        .toolbarBackground(.visible, for: .windowToolbar)
    }

    private var details: some View {
        DisclosureGroup(isExpanded: $showDetails) {
            VStack(alignment: .leading, spacing: Space.sm) {
                ValueRow(label: Format.label("scenario_id"), value: fixture.about.scenarioID, style: .identifier)
                ValueRow(label: Format.label("seed"), value: "\(fixture.about.seed)")
                ValueRow(label: Format.label("dt_s"), value: Format.number(fixture.dt, decimals: 2))
                ValueRow(label: Format.label("frame_count"), value: "\(fixture.frames.count)")
                ValueRow(label: Format.label("run_schema"), value: fixture.source.runSchema, style: .identifier)
                ValueRow(label: Format.label("fixture"), value: fixture.schema, style: .identifier)
                ValueRow(label: Format.label("presentation_schema"), value: fixture.source.presentationSchema, style: .identifier)
                StackedValue(label: Format.label("run_sha256"), value: fixture.source.runSHA256, style: .identifier)
                ValueRow(label: Format.label("app_version"), value: appVersion)
            }
            .padding(.top, Space.sm)
        } label: {
            Text("Details")
                .font(TypeScale.headline)
        }
    }

    private var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Development build"
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: Space.sm) {
            Text(title)
                .font(TypeScale.headline)
                .accessibilityAddTraits(.isHeader)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Data strings are stored lowercase ("replay only"); show them as sentences.
    private func sentence(_ text: String) -> String {
        text.prefix(1).uppercased() + text.dropFirst()
    }
}

struct SettingsView: View {
    @AppStorage(CockpitAppearance.storageKey) private var appearance = CockpitAppearance.dark.rawValue
    @Environment(\.palette) private var palette

    var body: some View {
        Form {
            Picker("Appearance", selection: $appearance) {
                ForEach(CockpitAppearance.allCases) { option in
                    Text(option.label).tag(option.rawValue)
                }
            }
            .pickerStyle(.radioGroup)
            Text("Dark is the default. System follows the macOS setting.")
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
        }
        .padding(Space.xl)
        .frame(width: 360)
    }
}
