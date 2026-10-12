import SwiftUI

/// Events from the presentation rules, oldest at the top and newest at the bottom.
/// Events of the current frame use the text color; earlier ones step back to secondary.
struct EventListView: View {
    /// Six full rows fit in the pane; the bottom padding keeps the last one off the window edge.
    static let visibleRows = 6
    @ObservedObject var model: CockpitModel
    @Environment(\.palette) private var palette
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let events = model.eventsSoFar
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Events")
                    .font(TypeScale.caption)
                    .foregroundStyle(palette.textSecondary)
                Spacer()
                Text("Click an event to jump to it")
                    .font(TypeScale.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            .padding(.horizontal, Space.md)
            .padding(.top, Space.sm)
            .padding(.bottom, Space.xxs)
            Spacer(minLength: 0)
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(events) { event in
                            EventRow(event: event, isCurrent: abs(event.timeS - model.time) < 1e-6) {
                                model.pause()
                                model.seek(toTime: event.timeS)
                            }
                            .id(event.id)
                        }
                    }
                    .padding(.horizontal, Space.sm)
                }
                .frame(height: EventRow.height * CGFloat(Self.visibleRows))
                .padding(.bottom, Space.md)
                .onAppear {
                    DispatchQueue.main.async { proxy.scrollTo(events.last?.id, anchor: .bottom) }
                }
                .onChange(of: events.count) { _, _ in
                    guard let last = events.last?.id else { return }
                    withAnimation(.easeOut(duration: Motion.duration(Motion.standard, reduceMotion: reduceMotion))) {
                        proxy.scrollTo(last, anchor: .bottom)
                    }
                }
            }
        }
        .frame(height: LayoutMetrics.eventPaneHeight)
        .background(palette.surface)
        .overlay(alignment: .top) { HairlineDivider() }
    }
}

private struct EventRow: View {
    static let height: CGFloat = 24
    let event: PresentationEvent
    let isCurrent: Bool
    let action: () -> Void
    @Environment(\.palette) private var palette
    @State private var hovering = false

    var body: some View {
        let style = StatusStyle(level: event.level)
        Button(action: action) {
            HStack(alignment: .firstTextBaseline, spacing: Space.sm) {
                StatusShape(style: style)
                Text(style.word)
                    .font(TypeScale.bodyStrong)
                    .frame(width: 44, alignment: .leading)
                Text(Format.seconds(event.timeS))
                    .font(TypeScale.body.monospacedDigit())
                    .foregroundStyle(palette.textSecondary)
                    .frame(width: 52, alignment: .trailing)
                Text(event.text)
                    .font(TypeScale.body)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 0)
            }
            .foregroundStyle(isCurrent ? palette.text : palette.textSecondary)
            .padding(.horizontal, Space.sm)
            .frame(height: Self.height)
            .background(RowBackground(isSelected: false, isHovered: hovering))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .help(event.text)
        .accessibilityLabel("\(style.word) at \(Format.seconds(event.timeS)): \(event.text)")
        .accessibilityHint("Jumps the replay to this event")
    }
}
