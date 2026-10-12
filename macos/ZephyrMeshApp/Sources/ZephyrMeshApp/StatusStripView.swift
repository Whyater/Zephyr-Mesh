import SwiftUI

/// The headline strip: the first thing to read. Tracking state, then the four team numbers,
/// all straight from the frame presentation. One row when it fits; otherwise tracking takes
/// its own row and the four numbers share the row below, so the strip never forces the
/// detail column wider than the window.
struct StatusStripView: View {
    @ObservedObject var model: CockpitModel
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
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 0) {
                TrackingCell(headline: presentation.headline)
                    .frame(minWidth: 240, maxWidth: .infinity, alignment: .leading)
                ForEach(metrics, id: \.0) { label, value in
                    StripMetric(label: label, value: value, showsDivider: true)
                        .frame(width: 136, alignment: .leading)
                }
            }
            VStack(alignment: .leading, spacing: Space.md) {
                TrackingCell(headline: presentation.headline)
                HStack(spacing: 0) {
                    ForEach(Array(metrics.enumerated()), id: \.offset) { index, metric in
                        StripMetric(label: metric.0, value: metric.1, showsDivider: index > 0)
                            .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }
        .padding(.horizontal, Space.lg)
        .padding(.vertical, Space.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.surface)
        .overlay(alignment: .bottom) { HairlineDivider() }
    }
}

private struct TrackingCell: View {
    let headline: PresentationHeadline
    @Environment(\.palette) private var palette

    var body: some View {
        let style = StatusStyle(level: headline.level)
        VStack(alignment: .leading, spacing: Space.xs) {
            Text(UIText.tracking)
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
            HStack(alignment: .center, spacing: Space.sm) {
                StatusShape(style: style, font: TypeScale.headline)
                Text(headline.text)
                    .font(TypeScale.title)
                    .foregroundStyle(palette.text)
                    .lineLimit(1)
                    .fixedSize()
            }
            .frame(minHeight: 30)
        }
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
                    .frame(height: 40)
                    .padding(.trailing, Space.md)
            }
            VStack(alignment: .leading, spacing: Space.xs) {
                Text(label)
                    .font(TypeScale.caption)
                    .foregroundStyle(palette.textSecondary)
                    .lineLimit(1)
                Text(value)
                    .font(missing ? TypeScale.headline : TypeScale.metric)
                    .foregroundStyle(missing ? palette.textSecondary : palette.text)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                    .frame(minHeight: 30, alignment: .leading)
            }
            Spacer(minLength: 0)
        }
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label): \(value)")
    }
}
