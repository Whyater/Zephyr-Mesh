import SwiftUI

/// The headline strip: the first thing to read. Tracking state, then the four team numbers,
/// all straight from the frame presentation. The layout is chosen from the column width alone
/// (`stripOneRowMinWidth`), never from the headline text, so the strip keeps its height while
/// the replay plays: one row when wide, otherwise the headline on its own row above the four
/// numbers. The headline is the largest text; the numbers are one step smaller.
struct StatusStripView: View {
    @ObservedObject var model: CockpitModel
    let availableWidth: CGFloat
    @Environment(\.palette) private var palette

    var body: some View {
        let presentation = model.presentation
        let strip = presentation.strip
        let metrics = [
            (UIText.estimateErrorP95, Format.meters(presentation.estimateErrorP95M)),
            (UIText.targetLinkDelay, Format.milliseconds(strip.targetLinkDelayMs)),
            (UIText.targetPacketLoss, Format.percent(strip.targetPacketLossPct)),
            (UIText.targetAge, Format.secondsRange(newest: strip.newestTargetAgeS, oldest: strip.oldestTargetAgeS))
        ]
        Group {
            if availableWidth >= LayoutMetrics.stripOneRowMinWidth {
                HStack(spacing: 0) {
                    TrackingCell(headline: presentation.headline)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    ForEach(metrics, id: \.0) { label, value in
                        StripMetric(label: label, value: value, showsDivider: true)
                            .frame(width: LayoutMetrics.stripMetricWidth, alignment: .leading)
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: Space.sm) {
                    TrackingCell(headline: presentation.headline)
                    HStack(spacing: 0) {
                        ForEach(Array(metrics.enumerated()), id: \.offset) { index, metric in
                            StripMetric(label: metric.0, value: metric.1, showsDivider: index > 0)
                                .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
            }
        }
        .padding(.horizontal, LayoutMetrics.contentInset)
        .padding(.vertical, Space.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.surface)
        .overlay(alignment: .bottom) { HairlineDivider() }
    }
}

/// Shape and headline. The headline already names the subject ("Tracking held"), so there is
/// no caption above it; the accessibility label still says what it is.
private struct TrackingCell: View {
    let headline: PresentationHeadline
    @Environment(\.palette) private var palette

    var body: some View {
        let style = StatusStyle(level: headline.level)
        HStack(alignment: .center, spacing: Space.sm) {
            StatusShape(style: style, font: TypeScale.headline)
            Text(headline.text)
                .font(TypeScale.metric)
                .foregroundStyle(palette.text)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .frame(minHeight: LayoutMetrics.stripDividerHeight, alignment: .leading)
        .padding(.trailing, Space.lg)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(UIText.tracking): \(headline.text)")
    }
}

private struct StripMetric: View {
    let label: String
    let value: String
    let showsDivider: Bool
    @Environment(\.palette) private var palette

    var body: some View {
        let missing = value == Format.notRecorded
        HStack(spacing: 0) {
            if showsDivider {
                HairlineDivider(vertical: true)
                    .frame(height: LayoutMetrics.stripDividerHeight)
                    .padding(.trailing, Space.md)
            }
            VStack(alignment: .leading, spacing: Space.xs) {
                Text(label)
                    .font(TypeScale.caption)
                    .foregroundStyle(palette.textSecondary)
                    .lineLimit(1)
                // A fixed-height box with bottom alignment keeps "Not recorded" on the same
                // baseline line as the digits it replaces.
                Text(value)
                    .font(missing ? TypeScale.headline : TypeScale.title.monospacedDigit())
                    .foregroundStyle(missing ? palette.textSecondary : palette.text)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                    .frame(height: LayoutMetrics.stripValueHeight, alignment: .bottomLeading)
            }
            Spacer(minLength: 0)
        }
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label): \(value)")
    }
}
