import AppKit
import SwiftUI

/// What this replay is and is not: evidence boundary, status, authority, failsafe, what the
/// data does not contain, how to regenerate it, and under Details the hashes and versions.
struct AboutReplayView: View {
    private static let commandLabel = Format.label("generation_command")
    let fixture: Fixture
    @Environment(\.palette) private var palette
    @State private var showDetails = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Space.xl) {
                VStack(alignment: .leading, spacing: Space.xs) {
                    Text(UIText.Menus.aboutReplay)
                        .font(TypeScale.title)
                    Text(fixture.runName)
                        .font(TypeScale.body)
                        .foregroundStyle(palette.textSecondary)
                }
                SectionBlock(title: "What this is") {
                    StackedValue(label: Format.label("status"), value: sentence(fixture.about.status))
                    StackedValue(label: Format.label("evidence_boundary"), value: sentence(fixture.about.evidenceBoundary))
                }
                SectionBlock(title: "Control and safety") {
                    StackedValue(label: Format.label("command_authority"), value: sentence(fixture.about.commandAuthority))
                    StackedValue(label: Format.label("failsafe"), value: sentence(fixture.about.failsafe))
                }
                SectionBlock(title: UIText.notInReplayTitle) {
                    VStack(alignment: .leading, spacing: Space.xs) {
                        ForEach(fixture.notInReplay, id: \.self) { item in
                            HStack(alignment: .firstTextBaseline, spacing: Space.sm) {
                                StatusShape(style: StatusStyle(level: "unavailable"))
                                Text(item)
                                    .font(TypeScale.body)
                            }
                        }
                    }
                }
                SectionBlock(title: "Reproduce") {
                    Text(Self.commandLabel)
                        .font(TypeScale.body)
                        .foregroundStyle(palette.textSecondary)
                    CommandBlock(command: fixture.about.generationCommand)
                }
                DetailsSection(isExpanded: $showDetails) {
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
            }
            .modifier(ViewerContent())
        }
        .frame(minWidth: LayoutMetrics.aboutWindowMinWidth, minHeight: LayoutMetrics.aboutWindowMinHeight)
        .background(palette.background)
        .toolbarBackground(palette.background, for: .windowToolbar)
        .toolbarBackground(.visible, for: .windowToolbar)
    }

    private var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Development build"
    }

    /// Data strings are stored lowercase ("replay only"); show them as sentences.
    private func sentence(_ text: String) -> String {
        text.prefix(1).uppercased() + text.dropFirst()
    }
}

/// A copyable command in a code block, with the copy button inside its top-trailing corner. The
/// button shows a checkmark for a moment after copying.
private struct CommandBlock: View {
    private static let copyTitle = Format.label("copy_command")
    private static let copiedTitle = Format.label("copied")
    /// How long the checkmark stays after copying.
    private static let copiedFeedback: TimeInterval = 1.2
    let command: String
    @Environment(\.palette) private var palette
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var copied = false

    var body: some View {
        // Wraps only at spaces, so every part of the command is visible and a path never breaks
        // after a slash. The copy button sits inside the block, beside the text.
        HStack(alignment: .top, spacing: Space.xs) {
            Text(CommandText.display(command))
                .font(ValueStyle.code.font)
                .foregroundStyle(palette.text)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, Space.md)
                .padding(.leading, Space.md)
                .accessibilityLabel(command)
            Button(action: copy) {
                Label(copied ? Self.copiedTitle : Self.copyTitle,
                      systemImage: copied ? "checkmark" : "doc.on.doc")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(QuietButtonStyle(tone: .secondary))
            .padding(Space.xs)
            .help(copied ? Self.copiedTitle : Self.copyTitle)
            .accessibilityLabel(copied ? Self.copiedTitle : Self.copyTitle)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.surface, in: RoundedRectangle(cornerRadius: Radius.small))
        .overlay { RoundedRectangle(cornerRadius: Radius.small).strokeBorder(palette.line, lineWidth: 1) }
    }

    private func copy() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(command, forType: .string)
        withAnimation(.easeOut(duration: Motion.duration(Motion.fast, reduceMotion: reduceMotion))) { copied = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.copiedFeedback) { copied = false }
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
        .frame(width: LayoutMetrics.settingsWidth)
    }
}

/// The command as displayed: a word joiner between the characters of each space-separated part,
/// so the line can break only at spaces. The copy button always copies the original command,
/// because the joiners are invisible characters a shell would not accept; for that reason the
/// displayed text is not selectable.
enum CommandText {
    static let joiner = "\u{2060}"

    static func display(_ command: String) -> String {
        command.split(separator: " ", omittingEmptySubsequences: false)
            .map { $0.map(String.init).joined(separator: joiner) }
            .joined(separator: " ")
    }
}
