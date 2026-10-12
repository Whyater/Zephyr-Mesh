import SwiftUI

/// The one honesty badge in the window: a quiet bordered label, never a filled pill.
struct HonestyBadge: View {
    @Environment(\.palette) private var palette

    var body: some View {
        Text(UIText.honestyBadge)
            .font(TypeScale.caption)
            .foregroundStyle(palette.textSecondary)
            .padding(.horizontal, Space.sm)
            .padding(.vertical, Space.xxs + 1)
            .overlay {
                RoundedRectangle(cornerRadius: Radius.small)
                    .strokeBorder(palette.lineStrong, lineWidth: 1)
            }
            .fixedSize()
            .help("Every value on screen is read from a recorded simulation file, not a live connection.")
            .accessibilityLabel(UIText.honestyBadge)
    }
}

/// Status is always shape plus color plus word; this draws the shape in the status color.
struct StatusShape: View {
    let style: StatusStyle
    var font: Font = TypeScale.caption
    @Environment(\.palette) private var palette

    var body: some View {
        Image(systemName: style.symbol)
            .font(font)
            .foregroundStyle(style.color(in: palette))
            .frame(width: Space.lg, alignment: .center)
            .accessibilityHidden(true)
    }
}

/// Shape and word together, with the word in the text color so it stays readable on any fill.
struct StatusLabel: View {
    let style: StatusStyle
    var font: Font = TypeScale.bodyStrong
    @Environment(\.palette) private var palette

    var body: some View {
        HStack(spacing: Space.xs) {
            StatusShape(style: style)
            Text(style.word)
                .font(font)
                .foregroundStyle(palette.text)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Status \(style.word)")
    }
}

struct SectionHeader: View {
    let title: String
    @Environment(\.palette) private var palette

    var body: some View {
        Text(title)
            .font(TypeScale.bodyStrong)
            .foregroundStyle(palette.text)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityAddTraits(.isHeader)
    }
}

/// How a value is set. Plain values and raw numbers use the 13 pt body with tabular digits;
/// the 12 pt mono role is kept for hashes and IDs only (its token use); commands use a 13 pt
/// monospaced body so they stay readable and copyable.
enum ValueStyle {
    case text, identifier, code

    var font: Font {
        switch self {
        case .text: return TypeScale.body.monospacedDigit()
        case .identifier: return TypeScale.mono
        case .code: return TypeScale.body.monospaced()
        }
    }
}

/// One label and value line. Missing values read "Not recorded" in the secondary color.
struct ValueRow: View {
    let label: String
    let value: String
    var style: ValueStyle = .text
    @Environment(\.palette) private var palette

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Space.sm) {
            Text(label)
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
                .lineLimit(1)
            Spacer(minLength: Space.sm)
            Text(value)
                .font(style.font)
                .foregroundStyle(value == Format.notRecorded ? palette.textSecondary : palette.text)
                .multilineTextAlignment(.trailing)
                .textSelection(.enabled)
        }
        .frame(minHeight: LayoutMetrics.rowHeight - Space.xs)
        .accessibilityElement(children: .combine)
    }
}

/// A stacked label over value block for long values (commands, hashes, paragraphs).
struct StackedValue: View {
    let label: String
    let value: String
    var style: ValueStyle = .text
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xxs) {
            Text(label)
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
            Text(value)
                .font(style.font)
                .foregroundStyle(palette.text)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

struct MetricTile: View {
    let label: String
    let value: String
    @Environment(\.palette) private var palette

    var body: some View {
        let missing = value == Format.notRecorded
        VStack(alignment: .leading, spacing: Space.xs) {
            Text(label)
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
                .lineLimit(1)
            Text(value)
                .font(missing ? TypeScale.body : TypeScale.metricSmall)
                .foregroundStyle(missing ? palette.textSecondary : palette.text)
                .lineLimit(1)
                .minimumScaleFactor(0.85)
        }
        .padding(Space.sm + Space.xxs)
        .frame(maxWidth: .infinity, minHeight: 56, alignment: .topLeading)
        .overlay {
            RoundedRectangle(cornerRadius: Radius.small)
                .strokeBorder(palette.line, lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
    }
}

struct HairlineDivider: View {
    var vertical = false
    @Environment(\.palette) private var palette

    var body: some View {
        Rectangle()
            .fill(palette.line)
            .frame(width: vertical ? 1 : nil, height: vertical ? nil : 1)
    }
}

/// Icon button with token hover and press fills, so the app never picks up the system accent.
struct QuietButtonStyle: ButtonStyle {
    var isActive = false

    func makeBody(configuration: Configuration) -> some View {
        QuietButtonBody(configuration: configuration, isActive: isActive)
    }

    private struct QuietButtonBody: View {
        let configuration: ButtonStyleConfiguration
        let isActive: Bool
        @Environment(\.palette) private var palette
        @Environment(\.isEnabled) private var isEnabled
        @State private var hovering = false

        var body: some View {
            configuration.label
                .font(TypeScale.body)
                .foregroundStyle(isEnabled ? palette.text : palette.textSecondary.opacity(0.6))
                .frame(minWidth: LayoutMetrics.rowHeight, minHeight: LayoutMetrics.rowHeight)
                .padding(.horizontal, Space.xxs)
                .background {
                    RoundedRectangle(cornerRadius: Radius.small).fill(fill)
                }
                .contentShape(Rectangle())
                .onHover { hovering = $0 && isEnabled }
        }

        private var fill: Color {
            if configuration.isPressed || isActive { return palette.selectionFill }
            return hovering ? palette.hoverFill : .clear
        }
    }
}

/// Row background for list rows: selection fill, hover fill, or nothing.
struct RowBackground: View {
    let isSelected: Bool
    let isHovered: Bool
    @Environment(\.palette) private var palette

    var body: some View {
        RoundedRectangle(cornerRadius: Radius.small)
            .fill(isSelected ? palette.selectionFill : (isHovered ? palette.hoverFill : .clear))
    }
}

/// Segmented choice drawn from tokens instead of the system control, whose selected segment
/// was too faint on the dark surface. Selected: text on selectionFill. Others: textSecondary on
/// the surface, hoverFill under the pointer. All three pairs pass the 4.5:1 token check.
struct SegmentedChoice<Value: Hashable>: View {
    let options: [(value: Value, label: String)]
    @Binding var selection: Value
    var accessibilityLabel: String
    @Environment(\.palette) private var palette

    var body: some View {
        HStack(spacing: Space.xxs) {
            ForEach(options, id: \.value) { option in
                SegmentButton(label: option.label, isSelected: option.value == selection) {
                    selection = option.value
                }
            }
        }
        .padding(Space.xxs)
        .overlay {
            RoundedRectangle(cornerRadius: Radius.small).strokeBorder(palette.line, lineWidth: 1)
        }
        .fixedSize()
        .accessibilityElement(children: .contain)
        .accessibilityLabel(accessibilityLabel)
    }

    private struct SegmentButton: View {
        let label: String
        let isSelected: Bool
        let action: () -> Void
        @Environment(\.palette) private var palette
        @State private var hovering = false

        var body: some View {
            Button(action: action) {
                Text(label)
                    .font(isSelected ? TypeScale.bodyStrong : TypeScale.body)
                    .foregroundStyle(isSelected ? palette.text : palette.textSecondary)
                    .padding(.horizontal, Space.sm + Space.xxs)
                    .frame(minWidth: 36, minHeight: LayoutMetrics.rowHeight - Space.sm)
                    .background {
                        RoundedRectangle(cornerRadius: Radius.small - 1)
                            .fill(isSelected ? palette.selectionFill : (hovering ? palette.hoverFill : .clear))
                    }
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .onHover { hovering = $0 }
            .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        }
    }
}
