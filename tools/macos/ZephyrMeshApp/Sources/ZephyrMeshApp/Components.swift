import SwiftUI

/// The one honesty badge in the window: a quiet bordered label, never a filled pill.
struct HonestyBadge: View {
    @Environment(\.palette) private var palette

    var body: some View {
        Text(UIText.honestyBadge)
            .font(TypeScale.caption)
            .foregroundStyle(palette.textSecondary)
            .padding(.horizontal, Space.sm)
            .frame(height: LayoutMetrics.badgeHeight)
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

/// A titled block in the viewer windows: headline title, then rows `Space.sm` apart.
struct SectionBlock<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: Space.sm) {
            Text(title)
                .font(TypeScale.headline)
                .accessibilityAddTraits(.isHeader)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
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

private struct ValueRowLabelWidthKey: EnvironmentKey {
    static let defaultValue: CGFloat? = nil
}

extension EnvironmentValues {
    /// When set, `ValueRow` uses a fixed label column and puts the value right after it, so the
    /// eye never crosses a wide gap. The narrow inspector leaves it unset (trailing values).
    var valueRowLabelWidth: CGFloat? {
        get { self[ValueRowLabelWidthKey.self] }
        set { self[ValueRowLabelWidthKey.self] = newValue }
    }
}

/// One label and value line. Missing values read "Not recorded" in the secondary color.
struct ValueRow: View {
    let label: String
    let value: String
    var style: ValueStyle = .text
    @Environment(\.palette) private var palette
    @Environment(\.valueRowLabelWidth) private var labelWidth

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Space.sm) {
            if let labelWidth {
                labelText
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(width: labelWidth, alignment: .leading)
                valueText
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                labelText
                    .lineLimit(1)
                Spacer(minLength: Space.sm)
                valueText
                    .multilineTextAlignment(.trailing)
            }
        }
        .frame(minHeight: LayoutMetrics.controlHeight - Space.xs)
        .accessibilityElement(children: .combine)
    }

    private var labelText: some View {
        Text(label)
            .font(TypeScale.body)
            .foregroundStyle(palette.textSecondary)
    }

    private var valueText: some View {
        Text(value)
            .font(style.font)
            .foregroundStyle(value == Format.notRecorded ? palette.textSecondary : palette.text)
            .textSelection(.enabled)
    }
}

/// A stacked label over value block for long values (commands, hashes, paragraphs). The label
/// uses the same 13 pt secondary role as `ValueRow`, so labels never change size in a section.
struct StackedValue: View {
    let label: String
    let value: String
    var style: ValueStyle = .text
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xxs) {
            Text(label)
                .font(TypeScale.body)
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
            // Same font when missing, so the tile height never changes as playback starts.
            Text(value)
                .font(TypeScale.metricSmall)
                .foregroundStyle(missing ? palette.textSecondary : palette.text)
                .lineLimit(1)
                .minimumScaleFactor(0.85)
        }
        .padding(Space.sm + Space.xxs)
        .frame(maxWidth: .infinity, minHeight: LayoutMetrics.metricTileMinHeight, alignment: .topLeading)
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
/// The hit area is a fixed square (`iconButtonSize`), so a row of these has an even pitch
/// whatever the glyph widths are. `secondary` tones the glyph down for toolbar view controls.
struct QuietButtonStyle: ButtonStyle {
    enum Tone { case primary, secondary }
    var isActive = false
    var font: Font = TypeScale.body
    var tone: Tone = .primary
    /// A text button that needs to read as a button (for example Clear filter): a hairline
    /// outline and `Space.md` side padding.
    var bordered = false

    /// Space between the square's edge and a glyph of `iconGlyphSize`, used to line glyph ink
    /// up with the content inset instead of the invisible hit area.
    static let glyphInset = (LayoutMetrics.iconButtonSize - LayoutMetrics.iconGlyphSize) / 2

    func makeBody(configuration: Configuration) -> some View {
        QuietButtonBody(configuration: configuration, isActive: isActive, font: font, tone: tone, bordered: bordered)
    }

    private struct QuietButtonBody: View {
        let configuration: ButtonStyleConfiguration
        let isActive: Bool
        let font: Font
        let tone: Tone
        let bordered: Bool
        @Environment(\.palette) private var palette
        @Environment(\.isEnabled) private var isEnabled
        @State private var hovering = false

        var body: some View {
            configuration.label
                .font(font)
                .foregroundStyle(foreground)
                .padding(.horizontal, bordered ? Space.md : 0)
                .frame(minWidth: LayoutMetrics.iconButtonSize, minHeight: LayoutMetrics.iconButtonSize)
                .background {
                    RoundedRectangle(cornerRadius: Radius.small).fill(fill)
                }
                .overlay {
                    if bordered {
                        RoundedRectangle(cornerRadius: Radius.small).strokeBorder(palette.lineStrong, lineWidth: 1)
                    }
                }
                .contentShape(Rectangle())
                .onHover { hovering = $0 && isEnabled }
        }

        private var foreground: Color {
            guard isEnabled else { return palette.textSecondary.opacity(0.6) }
            if tone == .secondary && !hovering && !isActive { return palette.textSecondary }
            return palette.text
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
/// Every segment reserves the width of its bold label, and `segmentWidth` makes them equal, so
/// the control never changes width when the selection changes. Its height is `controlHeight`.
struct SegmentedChoice<Value: Hashable>: View {
    let options: [(value: Value, label: String)]
    @Binding var selection: Value
    var accessibilityLabel: String
    var segmentWidth: CGFloat?
    @Environment(\.palette) private var palette

    var body: some View {
        HStack(spacing: Space.xxs) {
            ForEach(options, id: \.value) { option in
                SegmentButton(label: option.label, isSelected: option.value == selection, width: segmentWidth) {
                    selection = option.value
                }
            }
        }
        .padding(Space.xxs)
        .frame(height: LayoutMetrics.controlHeight)
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
        let width: CGFloat?
        let action: () -> Void
        @Environment(\.palette) private var palette
        @State private var hovering = false

        var body: some View {
            Button(action: action) {
                ZStack {
                    // Reserves the bold width so the segment does not grow when selected.
                    Text(label).font(TypeScale.bodyStrong).hidden()
                    Text(label)
                        .font(isSelected ? TypeScale.bodyStrong : TypeScale.body)
                        .foregroundStyle(isSelected ? palette.text : palette.textSecondary)
                }
                .padding(.horizontal, width == nil ? Space.sm + Space.xxs : 0)
                .frame(width: width)
                .frame(maxHeight: .infinity)
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

/// The shared "Details" drawer: a full-width clickable header row with a rotating secondary
/// chevron and a hover fill, used by the inspector, the three viewer windows and the raw tree.
struct DetailsDisclosureStyle: DisclosureGroupStyle {
    var font: Font = TypeScale.bodyStrong

    func makeBody(configuration: Configuration) -> some View {
        DetailsDisclosureBody(configuration: configuration, font: font)
    }

    private struct DetailsDisclosureBody: View {
        let configuration: DisclosureGroupStyleConfiguration
        let font: Font
        @Environment(\.palette) private var palette
        @Environment(\.accessibilityReduceMotion) private var reduceMotion
        @State private var hovering = false

        var body: some View {
            VStack(alignment: .leading, spacing: 0) {
                Button {
                    withAnimation(.easeOut(duration: Motion.duration(Motion.fast, reduceMotion: reduceMotion))) {
                        configuration.isExpanded.toggle()
                    }
                } label: {
                    HStack(spacing: Space.xs) {
                        Image(systemName: "chevron.right")
                            .font(TypeScale.caption)
                            .foregroundStyle(palette.textSecondary)
                            .rotationEffect(.degrees(configuration.isExpanded ? 90 : 0))
                            .frame(width: Space.md)
                            .accessibilityHidden(true)
                        configuration.label
                            .font(font)
                            .foregroundStyle(palette.text)
                        Spacer(minLength: 0)
                    }
                    .frame(minHeight: LayoutMetrics.controlHeight)
                    .contentShape(Rectangle())
                    .background {
                        RoundedRectangle(cornerRadius: Radius.small)
                            .fill(hovering ? palette.hoverFill : .clear)
                            .padding(.horizontal, -Space.xs)
                    }
                }
                .buttonStyle(.plain)
                .onHover { hovering = $0 }
                .accessibilityValue(configuration.isExpanded ? "Expanded" : "Collapsed")
                if configuration.isExpanded {
                    configuration.content
                        .padding(.top, Space.sm)
                }
            }
        }
    }
}

/// A hairline, then the collapsed Details drawer. The divider marks it as a technical drawer
/// rather than one more section.
struct DetailsSection<Content: View>: View {
    @Binding var isExpanded: Bool
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: Space.md) {
            HairlineDivider()
            DisclosureGroup(isExpanded: $isExpanded) {
                VStack(alignment: .leading, spacing: Space.sm) {
                    content()
                }
            } label: {
                Text(UIText.Inspector.details)
            }
            .disclosureGroupStyle(DetailsDisclosureStyle())
        }
    }
}

/// A caption-labelled raw tree inside Details.
struct RawTreeBlock: View {
    let title: String
    let value: JSONValue
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            Text(title)
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
            JSONTree(value: value)
        }
        .padding(.top, Space.sm)
    }
}

/// Raw tree with every key labeled, collapsed by default. Shared by the Evidence and the
/// Scenario Run windows. Each branch builds its children only when opened and lists at most
/// `limit` of them, so a huge shallow file cannot create thousands of views.
struct JSONTree: View {
    /// A data cap, not a layout size: entries listed per branch (the summary uses it too).
    static let limit = 100
    let value: JSONValue
    @Environment(\.palette) private var palette

    @ViewBuilder private func hiddenNote(_ count: Int) -> some View {
        if count > Self.limit {
            Text("\(count - Self.limit) more not shown; the file has every entry")
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
        }
    }

    var body: some View {
        switch value {
        case .object(let object):
            let keys = object.keys.sorted()
            VStack(alignment: .leading, spacing: Space.xxs) {
                ForEach(keys.prefix(Self.limit), id: \.self) { key in
                    JSONTreeEntry(label: Format.label(key), value: object[key] ?? .null)
                }
                hiddenNote(keys.count)
            }
        case .array(let items):
            VStack(alignment: .leading, spacing: Space.xxs) {
                ForEach(Array(items.prefix(Self.limit).enumerated()), id: \.offset) { index, item in
                    JSONTreeEntry(label: "Item \(index + 1)", value: item)
                }
                hiddenNote(items.count)
            }
        default:
            Text(value.plainText)
                .font(TypeScale.mono)
                .textSelection(.enabled)
        }
    }
}

private struct JSONTreeEntry: View {
    let label: String
    let value: JSONValue
    @Environment(\.palette) private var palette

    var body: some View {
        if value.isBranch {
            DisclosureGroup {
                JSONTree(value: value)
                    .padding(.leading, Space.lg)
            } label: {
                HStack {
                    Text(label)
                    Spacer()
                    Text(value.branchSummary)
                        .font(TypeScale.caption)
                        .foregroundStyle(palette.textSecondary)
                }
            }
            .disclosureGroupStyle(DetailsDisclosureStyle(font: TypeScale.body))
        } else {
            ValueRow(label: label, value: value.plainText)
        }
    }
}

/// The shared reading column of the viewer windows: centered, at most `viewerContentMaxWidth`
/// wide, `Space.xl` from the window edge, with label and value columns for every `ValueRow`.
struct ViewerContent: ViewModifier {
    func body(content: Content) -> some View {
        content
            .environment(\.valueRowLabelWidth, LayoutMetrics.viewerLabelWidth)
            .frame(maxWidth: LayoutMetrics.viewerContentMaxWidth, alignment: .leading)
            .padding(Space.xl)
            .frame(maxWidth: .infinity)
    }
}

extension View {
    /// Reports this view's laid-out size, at appear and on every change, without changing its
    /// layout.
    func onSizeChange(_ action: @escaping (CGSize) -> Void) -> some View {
        background {
            GeometryReader { proxy in
                Color.clear
                    .onAppear { action(proxy.size) }
                    .onChange(of: proxy.size) { _, size in action(size) }
            }
        }
    }
}
