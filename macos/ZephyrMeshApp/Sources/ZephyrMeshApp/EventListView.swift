import SwiftUI

/// Events from the presentation rules, oldest at the top and newest at the bottom.
/// Events of the current frame use the text color; earlier ones step back to secondary.
/// The pane shows `eventRowsRegular` rows, or `eventRowsCompact` when the window is short, so
/// the scene keeps most of the height. Text and row edges sit on the content inset.
struct EventListView: View {
    @ObservedObject var model: CockpitModel
    let isCompact: Bool
    @Environment(\.palette) private var palette
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Never more rows than the whole run has events, so a short log leaves no blank block. The
    /// height is fixed for the run, so the scene above does not jump as events arrive.
    private var visibleRows: CGFloat {
        let rows = isCompact ? LayoutMetrics.eventRowsCompact : LayoutMetrics.eventRowsRegular
        return min(rows, CGFloat(max(model.totalEventCount, 1)))
    }

    var body: some View {
        let events = model.eventsSoFar
        VStack(alignment: .leading, spacing: 0) {
            Text("Events")
                .font(TypeScale.caption)
                .foregroundStyle(palette.textSecondary)
                .padding(.horizontal, LayoutMetrics.contentInset)
                .padding(.top, Space.sm)
                .padding(.bottom, Space.xxs)
                .accessibilityAddTraits(.isHeader)
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
                        if CGFloat(events.count) < visibleRows && !model.isAtEnd {
                            Text("Later events appear here as the replay plays.")
                                .font(TypeScale.caption)
                                .foregroundStyle(palette.textSecondary)
                                .padding(.horizontal, Space.sm)
                                .frame(height: LayoutMetrics.eventRowHeight, alignment: .leading)
                        }
                    }
                    // Row padding (Space.sm) plus this lands the row content on the content inset.
                    .padding(.horizontal, LayoutMetrics.contentInset - Space.sm)
                }
                .frame(height: LayoutMetrics.eventRowHeight * visibleRows)
                .padding(.bottom, Space.sm)
                .onAppear {
                    DispatchQueue.main.async { proxy.scrollTo(events.last?.id, anchor: .bottom) }
                }
                .onChange(of: events.count) { _, _ in
                    guard let last = events.last?.id else { return }
                    withAnimation(.easeOut(duration: Motion.duration(Motion.standard, reduceMotion: reduceMotion))) {
                        proxy.scrollTo(last, anchor: .bottom)
                    }
                }
                .onChange(of: isCompact) { _, _ in
                    DispatchQueue.main.async { proxy.scrollTo(events.last?.id, anchor: .bottom) }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.surface)
        .overlay(alignment: .top) { HairlineDivider() }
    }
}

private struct EventRow: View {
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
                    .lineLimit(1)
                    .frame(width: LayoutMetrics.eventWordWidth, alignment: .leading)
                Text(Format.seconds(event.timeS))
                    .font(TypeScale.body.monospacedDigit())
                    .foregroundStyle(palette.textSecondary)
                    .frame(width: LayoutMetrics.eventTimeWidth, alignment: .trailing)
                // The longest variant that fits; shorter ones drop drone IDs at a name boundary
                // and count them in "+N more", so a row never ends in half a drone name.
                ViewThatFits(in: .horizontal) {
                    ForEach(EventText.variants(event.text), id: \.self) { variant in
                        Text(variant)
                            .font(TypeScale.body)
                            .lineLimit(1)
                            .fixedSize()
                    }
                    Text(event.text)
                        .font(TypeScale.body)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                Spacer(minLength: 0)
            }
            .foregroundStyle(isCurrent ? palette.text : palette.textSecondary)
            .padding(.horizontal, Space.sm)
            .frame(height: LayoutMetrics.eventRowHeight)
            .background(RowBackground(isSelected: false, isHovered: hovering))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .help("\(event.text)\nClick to jump to this event.")
        .accessibilityLabel("\(style.word) at \(Format.seconds(event.timeS)): \(event.text)")
        .accessibilityHint("Jumps the replay to this event")
    }
}
