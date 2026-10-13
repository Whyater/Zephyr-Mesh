import SwiftUI

/// The media bar and the only home of playback: transport, a seekable scrubber with event
/// ticks, the replay time, and playback speed. Glyph ink, the track ends, the time and the speed
/// control all sit one `Space.lg` apart, and the first glyph lines up with the content inset.
struct TimelineView: View {
    @ObservedObject var model: CockpitModel
    @Environment(\.palette) private var palette

    var body: some View {
        HStack(spacing: Space.lg) {
            HStack(spacing: Space.sm) {
                HStack(spacing: Space.xs) {
                    transportButton(UIText.Transport.restart, symbol: "backward.end.fill",
                                    disabled: model.isAtStart && !model.isPlaying) {
                        model.restart()
                    }
                    transportButton(UIText.Transport.stepBack, symbol: "backward.frame.fill", disabled: model.isAtStart) {
                        model.stepBackward()
                    }
                    transportButton(model.isPlaying ? UIText.Transport.pause : UIText.Transport.play,
                                    symbol: model.isPlaying ? "pause.fill" : "play.fill", disabled: false,
                                    font: TypeScale.title) {
                        model.togglePlayback()
                    }
                    transportButton(UIText.Transport.stepForward, symbol: "forward.frame.fill", disabled: model.isAtEnd) {
                        model.stepFrame()
                    }
                }
                ReplayScrubber(model: model)
                    .frame(minWidth: LayoutMetrics.scrubberMinWidth)
            }
            Text("\(Format.time(model.time)) / \(Format.time(model.duration)) s")
                .font(TypeScale.body.monospacedDigit())
                .foregroundStyle(palette.text)
                .fixedSize()
                .accessibilityLabel("Replay time \(Format.time(model.time)) of \(Format.time(model.duration)) seconds")
            SegmentedChoice(options: PlaybackSpeed.allCases.map { ($0, $0.label) }, selection: $model.playbackSpeed,
                            accessibilityLabel: "Playback speed", segmentWidth: LayoutMetrics.speedSegmentWidth)
            .help("Playback speed. The whole run lasts \(Format.time(model.duration)) s, so 0.25x is the default.")
        }
        // The first button's glyph, not its invisible hit square, starts at the content inset.
        .padding(.leading, LayoutMetrics.contentInset - QuietButtonStyle.glyphInset)
        .padding(.trailing, LayoutMetrics.contentInset)
        .padding(.vertical, Space.sm)
        .background(palette.surface)
        .overlay(alignment: .top) { HairlineDivider() }
    }

    private func transportButton(_ control: UIText.TransportControl, symbol: String, disabled: Bool,
                                 font: Font = TypeScale.headline, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(control.label, systemImage: symbol)
                .labelStyle(.iconOnly)
        }
        .buttonStyle(QuietButtonStyle(font: font))
        .disabled(disabled)
        .help(control.help)
    }
}

/// A frame-snapping scrubber. The fill and thumb use the accent; event ticks are neutral
/// marks under the track whose tooltip lists that frame's events.
struct ReplayScrubber: View {
    @ObservedObject var model: CockpitModel
    @Environment(\.palette) private var palette
    @State private var hovering = false

    private let trackHeight = LayoutMetrics.scrubberTrackHeight
    private let thumbSize = LayoutMetrics.scrubberThumbSize
    private let tickHeight = LayoutMetrics.scrubberTickHeight

    var body: some View {
        GeometryReader { geometry in
            // The track spans the full frame so its visible ends are the frame's ends; the thumb
            // centre travels the track inset by the thumb radius so the thumb never overhangs.
            let width = max(geometry.size.width - thumbSize, 1)
            let midY = geometry.size.height / 2
            let progress = model.duration > 0 ? model.time / model.duration : 0
            ZStack(alignment: .topLeading) {
                Capsule()
                    .fill(palette.line)
                    .frame(width: geometry.size.width, height: trackHeight)
                    .offset(y: midY - trackHeight / 2)
                Capsule()
                    .fill(palette.accent)
                    .frame(width: thumbSize / 2 + max(width * progress, 0), height: trackHeight)
                    .offset(y: midY - trackHeight / 2)
                ForEach(model.eventFrameIndexes, id: \.self) { index in
                    let frame = model.fixture.frames[index]
                    let x = thumbSize / 2 + width * (model.duration > 0 ? frame.time / model.duration : 0)
                    Rectangle()
                        .fill(palette.lineStrong)
                        .frame(width: Space.xxs, height: tickHeight)
                        .offset(x: x - Space.xxs / 2, y: midY + trackHeight / 2 + Space.xxs)
                        .help(frame.presentation.events.map(\.text).joined(separator: "\n"))
                }
                Circle()
                    .fill(palette.accent)
                    .overlay { Circle().strokeBorder(palette.surface, lineWidth: 2) }
                    .frame(width: thumbSize, height: thumbSize)
                    .scaleEffect(hovering ? 1.1 : 1)
                    .offset(x: width * progress, y: midY - thumbSize / 2)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        model.pause()
                        let fraction = min(max((value.location.x - thumbSize / 2) / width, 0), 1)
                        model.seek(toTime: nearestFrameTime(fraction * model.duration))
                    }
            )
            .onHover { hovering = $0 }
        }
        .frame(height: LayoutMetrics.controlHeight)
        .accessibilityElement()
        .accessibilityLabel("Replay position")
        .accessibilityValue("Frame \(model.frameIndex + 1) of \(model.frameCount), \(Format.time(model.time)) seconds")
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: model.stepFrame()
            case .decrement: model.stepBackward()
            @unknown default: break
            }
        }
    }

    private func nearestFrameTime(_ target: Double) -> Double {
        model.fixture.frames.map(\.time).min { abs($0 - target) < abs($1 - target) } ?? 0
    }
}
