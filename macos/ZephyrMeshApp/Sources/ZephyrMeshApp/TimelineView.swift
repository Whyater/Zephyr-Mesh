import SwiftUI

/// Transport, seekable scrubber with event ticks, time label, and playback speed.
struct TimelineView: View {
    @ObservedObject var model: CockpitModel
    @Environment(\.palette) private var palette

    var body: some View {
        HStack(spacing: Space.md) {
            HStack(spacing: Space.xxs) {
                transportButton("Restart", symbol: "backward.end.fill", disabled: model.isAtStart && !model.isPlaying) {
                    model.restart()
                }
                transportButton("Step back", symbol: "backward.frame.fill", disabled: model.isAtStart) {
                    model.stepBackward()
                }
                transportButton(model.isPlaying ? "Pause" : "Play", symbol: model.isPlaying ? "pause.fill" : "play.fill",
                                disabled: false) {
                    model.togglePlayback()
                }
                transportButton("Step forward", symbol: "forward.frame.fill", disabled: model.isAtEnd) {
                    model.stepFrame()
                }
            }
            ReplayScrubber(model: model)
                .frame(minWidth: 160)
            Text("\(Format.time(model.time)) / \(Format.time(model.duration)) s")
                .font(TypeScale.body.monospacedDigit())
                .foregroundStyle(palette.text)
                .fixedSize()
                .accessibilityLabel("Replay time \(Format.time(model.time)) of \(Format.time(model.duration)) seconds")
            SegmentedChoice(options: PlaybackSpeed.allCases.map { ($0, $0.label) }, selection: $model.playbackSpeed,
                            accessibilityLabel: "Playback speed")
            .help("Playback speed. The whole run lasts \(Format.time(model.duration)) s, so 0.25x is the default.")
        }
        .padding(.horizontal, Space.md)
        .padding(.vertical, Space.sm)
        .background(palette.surface)
        .overlay(alignment: .top) { HairlineDivider() }
    }

    private func transportButton(_ title: String, symbol: String, disabled: Bool,
                                 action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: symbol)
                .labelStyle(.iconOnly)
        }
        .buttonStyle(QuietButtonStyle())
        .disabled(disabled)
        .help(title)
    }
}

/// A frame-snapping scrubber. The fill and thumb use the accent; event ticks are neutral
/// marks under the track whose tooltip lists that frame's events.
struct ReplayScrubber: View {
    @ObservedObject var model: CockpitModel
    @Environment(\.palette) private var palette
    @State private var hovering = false

    private let trackHeight: CGFloat = 4
    private let thumbSize: CGFloat = 14

    var body: some View {
        GeometryReader { geometry in
            let width = max(geometry.size.width - thumbSize, 1)
            let midY = geometry.size.height / 2
            let progress = model.duration > 0 ? model.time / model.duration : 0
            ZStack(alignment: .topLeading) {
                Capsule()
                    .fill(palette.line)
                    .frame(width: width, height: trackHeight)
                    .offset(x: thumbSize / 2, y: midY - trackHeight / 2)
                Capsule()
                    .fill(palette.accent)
                    .frame(width: max(width * progress, 0), height: trackHeight)
                    .offset(x: thumbSize / 2, y: midY - trackHeight / 2)
                ForEach(model.eventFrameIndexes, id: \.self) { index in
                    let frame = model.fixture.frames[index]
                    let x = thumbSize / 2 + width * (model.duration > 0 ? frame.time / model.duration : 0)
                    Rectangle()
                        .fill(palette.lineStrong)
                        .frame(width: 2, height: 6)
                        .offset(x: x - 1, y: midY + trackHeight / 2 + 3)
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
        .frame(height: LayoutMetrics.rowHeight)
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
