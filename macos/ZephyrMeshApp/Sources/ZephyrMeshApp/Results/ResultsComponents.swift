import AppKit
import SwiftUI

/// The one "Simulated results" badge in the Results window, styled like `HonestyBadge`.
struct ResultsBadge: View {
    @Environment(\.palette) private var palette

    var body: some View {
        Text(UIText.Results.badge)
            .font(TypeScale.caption)
            .foregroundStyle(palette.textSecondary)
            .padding(.horizontal, Space.sm)
            .frame(height: LayoutMetrics.badgeHeight)
            .overlay {
                RoundedRectangle(cornerRadius: Radius.small)
                    .strokeBorder(palette.lineStrong, lineWidth: 1)
            }
            .fixedSize()
            .accessibilityLabel(UIText.Results.badge)
    }
}

/// The lead finding quoted from the write-up, its source caption ("From README.md"), and any
/// other findings in a collapsed "More from the write-up" drawer. One message per view.
struct FindingBlock: View {
    let findings: [ResultsFinding]
    var lead = 0
    @State private var showMore = false
    @Environment(\.palette) private var palette

    var body: some View {
        let split = ResultsFormat.splitFindings(findings, lead: lead)
        VStack(alignment: .leading, spacing: Space.sm) {
            if let first = split.lead {
                findingText(first)
                Text(Format.fill(UIText.Results.sourcePrefix, ["source": first.source]))
                    .font(TypeScale.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            if !split.more.isEmpty {
                DisclosureGroup(isExpanded: $showMore) {
                    VStack(alignment: .leading, spacing: Space.sm) {
                        ForEach(Array(split.more.enumerated()), id: \.offset) { _, finding in
                            findingText(finding)
                        }
                        Text(Format.fill(UIText.Results.sourcePrefix, ["source": sources(split.more)]))
                            .font(TypeScale.caption)
                            .foregroundStyle(palette.textSecondary)
                    }
                    .padding(.leading, Space.lg + Space.xs)
                } label: {
                    Text(UIText.Results.moreFindings)
                }
                .disclosureGroupStyle(DetailsDisclosureStyle(font: TypeScale.body))
            }
        }
        .frame(maxWidth: LayoutMetrics.viewerContentMaxWidth, alignment: .leading)
    }

    private func findingText(_ finding: ResultsFinding) -> some View {
        Text(ResultsFormat.displayText(finding.text))
            .font(TypeScale.body)
            .foregroundStyle(palette.text)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
    }

    private func sources(_ findings: [ResultsFinding]) -> String {
        var seen: [String] = []
        for finding in findings where !seen.contains(finding.source) { seen.append(finding.source) }
        return seen.joined(separator: ", ")
    }
}

/// A small caption label above a control in the controls row.
struct ControlGroup<Content: View>: View {
    let label: String
    @ViewBuilder let content: () -> Content
    @Environment(\.palette) private var palette

    var body: some View {
        HStack(spacing: Space.sm) {
            Text(label)
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
            content()
        }
    }
}

/// One legend entry: a mark drawn at the swatch size, then its word.
struct LegendItem<Mark: View>: View {
    let title: String
    @ViewBuilder let mark: () -> Mark

    var body: some View {
        HStack(spacing: Space.xs + Space.xxs) {
            mark()
                .frame(width: LayoutMetrics.legendSwatchWidth, height: LayoutMetrics.legendSwatchHeight)
                .accessibilityHidden(true)
            Text(title)
                .font(TypeScale.caption)
                .lineLimit(1)
                .fixedSize()
        }
        .accessibilityElement(children: .combine)
    }
}

/// A bulleted list inside a disclosure, each bullet a hollow neutral dot.
private struct BulletList: View {
    let items: [String]
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                HStack(alignment: .firstTextBaseline, spacing: Space.sm) {
                    Circle()
                        .fill(palette.textSecondary)
                        .frame(width: Space.xs, height: Space.xs)
                        .alignmentGuide(.firstTextBaseline) { $0[.bottom] + Space.xxs }
                        .accessibilityHidden(true)
                    Text(item)
                        .font(TypeScale.body)
                        .foregroundStyle(palette.text)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }
        }
    }
}

/// "Limits", collapsed by default: the run's limits verbatim, then the rule sentences.
struct LimitsDisclosure: View {
    let limits: [String]
    let notes: [String]
    @State private var expanded = false
    @Environment(\.palette) private var palette

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: Space.md) {
                BulletList(items: limits)
                ForEach(Array(notes.enumerated()), id: \.offset) { _, note in
                    Text(note)
                        .font(TypeScale.caption)
                        .foregroundStyle(palette.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }
            .padding(.leading, Space.lg + Space.xs)
            .frame(maxWidth: LayoutMetrics.viewerContentMaxWidth, alignment: .leading)
        } label: {
            Text(UIText.Results.limitsTitle)
        }
        .disclosureGroupStyle(DetailsDisclosureStyle())
    }
}

/// "Source", collapsed: where the numbers come from. Digests show 12 characters; the full
/// value is in the tooltip and is selectable.
struct SourceDisclosure: View {
    let source: ResultsSource
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: Space.xxs) {
                ValueRow(label: UIText.Results.sourceRunFolder, value: source.runDir, style: .identifier)
                ValueRow(label: UIText.Results.sourceFile, value: source.file, style: .identifier)
                DigestRow(label: Format.label("sha256"), value: source.sha256)
                DigestRow(label: UIText.Results.sourceManifest, value: source.manifestSHA256)
                ValueRow(label: Format.label("schema"), value: source.schema, style: .identifier)
                DigestRow(label: UIText.Results.sourceRevision, value: source.gitRevision)
                ValueRow(label: UIText.Results.sourceTreeState, value: source.gitTreeState)
            }
            .padding(.leading, Space.lg + Space.xs)
            .environment(\.valueRowLabelWidth, LayoutMetrics.atlasRowHeaderWidth)
            .frame(maxWidth: LayoutMetrics.viewerContentMaxWidth, alignment: .leading)
        } label: {
            Text(UIText.Results.sourceTitle)
        }
        .disclosureGroupStyle(DetailsDisclosureStyle())
    }
}

private struct DigestRow: View {
    let label: String
    let value: String

    var body: some View {
        ValueRow(label: label, value: ResultsFormat.shortDigest(value), style: .identifier)
            .help(value)
            .accessibilityValue(value)
            .contextMenu {
                Button(Format.label("copy")) {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(value, forType: .string)
                }
            }
    }
}

/// A metric tile like `MetricTile`, with an optional status shape before the value. Missing
/// values read in the secondary color.
struct ResultTile: View {
    let label: String
    let value: String
    var status: StatusStyle?
    @Environment(\.palette) private var palette

    var body: some View {
        let missing = value == Format.notRecorded || value == ResultsFormat.notComputed
        VStack(alignment: .leading, spacing: Space.xs) {
            Text(label)
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            HStack(spacing: Space.xxs) {
                if let status {
                    StatusShape(style: status, font: TypeScale.caption)
                        .frame(width: Space.md)
                }
                Text(value)
                    .font(TypeScale.metricSmall)
                    .foregroundStyle(missing ? palette.textSecondary : palette.text)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
            }
        }
        .padding(Space.sm + Space.xxs)
        // Fills the grid row, so both tiles in a row share a height when a label wraps.
        .frame(maxWidth: .infinity, minHeight: LayoutMetrics.metricTileMinHeight, maxHeight: .infinity, alignment: .topLeading)
        .overlay {
            RoundedRectangle(cornerRadius: Radius.small)
                .strokeBorder(palette.line, lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
    }
}

/// Inspector placeholder when nothing is selected.
struct ResultsEmptyInspector: View {
    let hint: String
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(spacing: Space.sm) {
            Image(systemName: "scope")
                .font(TypeScale.title)
                .foregroundStyle(palette.textSecondary)
                .accessibilityHidden(true)
            Text(UIText.Results.inspectorEmptyTitle)
                .font(TypeScale.bodyStrong)
            Text(hint)
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Shown instead of the views when the results file cannot be read. Never invented data.
struct ResultsLoadFailure: View {
    let message: String
    @Environment(\.palette) private var palette

    var body: some View {
        VStack(spacing: Space.md) {
            StatusShape(style: StatusStyle(level: "alert"), font: TypeScale.title)
            Text(UIText.Results.loadFailed)
                .font(TypeScale.title)
            Text(message)
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
                .multilineTextAlignment(.center)
                .textSelection(.enabled)
                .frame(maxWidth: LayoutMetrics.messageMaxWidth)
            Text(UIText.Results.loadFailedHint)
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
        }
        .padding(Space.xxl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(palette.background)
    }
}

/// The accent focus ring used by focusable charts and grids, drawn just outside the content.
struct FocusRing: ViewModifier {
    let visible: Bool
    @Environment(\.palette) private var palette

    func body(content: Content) -> some View {
        content.overlay {
            if visible {
                RoundedRectangle(cornerRadius: Radius.small)
                    .strokeBorder(palette.accent, lineWidth: Space.xxs)
                    .padding(-Space.xs)
                    .allowsHitTesting(false)
            }
        }
    }
}

/// Reports whether this view's window is the key window, so the main window's single-key menu
/// shortcuts can stand aside while the Results window has the keyboard.
struct WindowKeyObserver: NSViewRepresentable {
    let onChange: (Bool) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = ObservingView()
        view.onChange = onChange
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        (view as? ObservingView)?.onChange = onChange
    }

    final class ObservingView: NSView {
        var onChange: ((Bool) -> Void)?
        private var tokens: [NSObjectProtocol] = []

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            tokens.forEach { NotificationCenter.default.removeObserver($0) }
            tokens = []
            guard let window else {
                onChange?(false)
                return
            }
            for (name, isKey) in [(NSWindow.didBecomeKeyNotification, true), (NSWindow.didResignKeyNotification, false),
                                  (NSWindow.willCloseNotification, false)] {
                tokens.append(NotificationCenter.default.addObserver(forName: name, object: window, queue: .main) { [weak self] _ in
                    self?.onChange?(isKey)
                })
            }
            onChange?(window.isKeyWindow)
        }

        deinit { tokens.forEach { NotificationCenter.default.removeObserver($0) } }
    }
}
