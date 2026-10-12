import AppKit
import SceneKit
import SwiftUI

/// The 3D view plus its only overlays: a hover label and a small grid scale hint.
struct SceneContainer: View {
    @ObservedObject var model: CockpitModel
    @Environment(\.palette) private var palette
    @State private var hover: SceneHover?

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            SceneRepresentable(model: model, hover: $hover)
            if let hover, let drone = model.presentationDrone(id: hover.droneID) {
                HoverLabel(drone: drone)
                    .position(x: hover.location.x + 14, y: hover.location.y - 18)
                    .allowsHitTesting(false)
                    .fixedSize()
            }
            scaleHint
        }
        .frame(minWidth: LayoutMetrics.sceneMinWidth, maxWidth: .infinity, maxHeight: .infinity)
        .background(palette.background)
        .clipped()
    }

    private var scaleHint: some View {
        HStack(spacing: Space.xs) {
            Image(systemName: "square.grid.3x3")
                .font(TypeScale.caption)
            Text(model.viewMode == .top ? "Top view, grid 1 m" : "Grid 1 m, heavy line every 5 m")
                .font(TypeScale.caption)
        }
        .foregroundStyle(palette.textSecondary)
        .padding(Space.md)
        .allowsHitTesting(false)
        .accessibilityElement(children: .combine)
    }
}

struct SceneHover: Equatable {
    let droneID: String
    let location: CGPoint
}

private struct HoverLabel: View {
    let drone: PresentationDrone
    @Environment(\.palette) private var palette

    var body: some View {
        HStack(spacing: Space.xs) {
            StatusShape(style: StatusStyle(drone: drone))
            Text(drone.compactName)
                .font(TypeScale.bodyStrong)
            Text(drone.word)
                .font(TypeScale.body)
                .foregroundStyle(palette.textSecondary)
        }
        .foregroundStyle(palette.text)
        .padding(.horizontal, Space.sm)
        .padding(.vertical, Space.xs)
        .background(palette.surface, in: RoundedRectangle(cornerRadius: Radius.small))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.small).strokeBorder(palette.lineStrong, lineWidth: 1)
        }
    }
}

private struct SceneRepresentable: NSViewRepresentable {
    @ObservedObject var model: CockpitModel
    @Binding var hover: SceneHover?
    @Environment(\.palette) private var palette
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func makeCoordinator() -> Coordinator {
        Coordinator(controller: SceneController(fixture: model.fixture))
    }

    func makeNSView(context: Context) -> InteractiveSceneView {
        let view = InteractiveSceneView(frame: .zero)
        let controller = context.coordinator.controller
        view.scene = controller.scene
        view.pointOfView = controller.cameraNode
        view.delegate = controller.driver
        view.allowsCameraControl = false
        view.antialiasingMode = .multisampling4X
        view.preferredFramesPerSecond = 60
        view.rendersContinuously = true
        view.isPlaying = true
        view.showsStatistics = ProcessInfo.processInfo.environment["ZEPHYR_SCENE_STATS"] == "1"
        view.setAccessibilityLabel("3D view of the replay. Drag to orbit, scroll or pinch to zoom.")
        context.coordinator.attach(view)
        return view
    }

    func updateNSView(_ view: InteractiveSceneView, context: Context) {
        context.coordinator.model = model
        context.coordinator.hover = $hover
        view.backgroundColor = palette.backgroundNS
        let presentation = model.presentation
        var styles: [String: StatusStyle] = [:]
        var headings: [String: Double] = [:]
        for drone in presentation.perDrone {
            styles[drone.droneID] = StatusStyle(drone: drone)
            if let heading = drone.headingDeg { headings[drone.droneID] = heading }
        }
        let state = SceneState(
            frame: model.frame, styles: styles, headings: headings, selectedID: model.selectedID,
            hoveredID: model.hoveredID, showKeepOut: model.showKeepOut, viewMode: model.viewMode,
            palette: palette, scenePalette: Theme.scene3D(for: colorScheme), isDark: colorScheme == .dark,
            reduceMotion: reduceMotion, cameraCommand: model.cameraCommand
        )
        context.coordinator.controller.apply(state)
    }

    @MainActor
    final class Coordinator {
        let controller: SceneController
        var model: CockpitModel?
        var hover: Binding<SceneHover?>?

        init(controller: SceneController) {
            self.controller = controller
        }

        func attach(_ view: InteractiveSceneView) {
            view.onDrag = { [weak self] dx, dy in self?.controller.orbit(deltaX: dx, deltaY: dy) }
            view.onZoom = { [weak self] factor in self?.controller.zoom(factor: factor) }
            view.onResize = { [weak self] size in self?.controller.viewDidResize(size) }
            view.onHover = { [weak self] id, point in
                guard let self else { return }
                let next = id.map { SceneHover(droneID: $0, location: point) }
                if hover?.wrappedValue != next { hover?.wrappedValue = next }
                if model?.hoveredID != id { model?.hoveredID = id }
            }
            view.onClick = { [weak self] id, clickCount in
                guard let self, let model, let id else { return }
                if clickCount >= 2 { model.focus(on: id) } else { model.select(id) }
            }
            view.menuProvider = { [weak self] id in
                guard let self, let model else { return nil }
                return DroneMenu.make(droneID: id, model: model)
            }
        }
    }
}

/// SCNView that turns mouse input into orbit, zoom, hover, click and context-menu events.
final class InteractiveSceneView: SCNView {
    var onDrag: ((CGFloat, CGFloat) -> Void)?
    var onZoom: ((CGFloat) -> Void)?
    var onHover: ((String?, CGPoint) -> Void)?
    var onClick: ((String?, Int) -> Void)?
    var menuProvider: ((String) -> NSMenu?)?
    var onResize: ((CGSize) -> Void)?

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        onResize?(newSize)
    }

    private var dragOrigin: NSPoint?
    private var isDragging = false
    private var trackingArea: NSTrackingArea?
    private static let dragThreshold: CGFloat = 3

    override var acceptsFirstResponder: Bool { true }

    /// The scene, not the team filter, starts as first responder so the generated single-key
    /// shortcuts work from launch. SwiftUI would otherwise focus the first text field.
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        guard let window else { return }
        DispatchQueue.main.async { [weak self, weak window] in
            guard let self, let window, window.firstResponder is NSText || window.firstResponder === window else { return }
            window.makeFirstResponder(self)
        }
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        beginDrag(event)
    }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let trackingArea { removeTrackingArea(trackingArea) }
        let area = NSTrackingArea(rect: bounds, options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
                                  owner: self, userInfo: nil)
        addTrackingArea(area)
        trackingArea = area
    }

    func droneID(at point: NSPoint) -> String? {
        let options: [SCNHitTestOption: Any] = [
            .categoryBitMask: SceneController.hitCategory,
            .searchMode: SCNHitTestSearchMode.closest.rawValue,
            .ignoreHiddenNodes: true
        ]
        return hitTest(point, options: options).first?.node.name
    }

    private func swiftUIPoint(_ point: NSPoint) -> CGPoint {
        CGPoint(x: point.x, y: bounds.height - point.y)
    }

    override func mouseMoved(with event: NSEvent) {
        guard !isDragging else { return }
        let point = convert(event.locationInWindow, from: nil)
        let id = droneID(at: point)
        onHover?(id, swiftUIPoint(point))
        (id == nil ? NSCursor.arrow : NSCursor.pointingHand).set()
    }

    override func mouseExited(with event: NSEvent) {
        onHover?(nil, .zero)
        NSCursor.arrow.set()
    }

    private func beginDrag(_ event: NSEvent) {
        dragOrigin = event.locationInWindow
        isDragging = false
    }

    override func mouseDragged(with event: NSEvent) {
        guard let origin = dragOrigin else { return }
        if !isDragging {
            let moved = hypot(event.locationInWindow.x - origin.x, event.locationInWindow.y - origin.y)
            guard moved >= Self.dragThreshold else { return }
            isDragging = true
            onHover?(nil, .zero)
            NSCursor.closedHand.set()
        }
        onDrag?(event.deltaX, event.deltaY)
    }

    override func mouseUp(with event: NSEvent) {
        defer {
            dragOrigin = nil
            isDragging = false
        }
        if isDragging {
            NSCursor.arrow.set()
            return
        }
        let point = convert(event.locationInWindow, from: nil)
        onClick?(droneID(at: point), event.clickCount)
    }

    override func scrollWheel(with event: NSEvent) {
        let delta = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY * 0.01 : event.scrollingDeltaY * 0.1
        guard delta != 0 else { return }
        onZoom?(CGFloat(exp(Double(delta))))
    }

    override func magnify(with event: NSEvent) {
        onZoom?(1 + event.magnification)
    }

    override func menu(for event: NSEvent) -> NSMenu? {
        let point = convert(event.locationInWindow, from: nil)
        guard let id = droneID(at: point) else { return nil }
        return menuProvider?(id)
    }
}

/// The same three actions as the team list context menu, for a drone under the pointer.
@MainActor
enum DroneMenu {
    static func make(droneID: String, model: CockpitModel) -> NSMenu {
        let menu = NSMenu()
        let name = model.presentationDrone(id: droneID)?.headingName ?? droneID
        let title = NSMenuItem(title: name, action: nil, keyEquivalent: "")
        title.isEnabled = false
        menu.addItem(title)
        menu.addItem(ClosureMenuItem(title: "Focus Camera") { model.focus(on: droneID) })
        menu.addItem(ClosureMenuItem(title: "Show in Inspector") {
            model.select(droneID)
            model.showInspector = true
        })
        menu.addItem(.separator())
        menu.addItem(ClosureMenuItem(title: "Copy Drone ID") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(droneID, forType: .string)
        })
        return menu
    }
}

final class ClosureMenuItem: NSMenuItem {
    private let handler: () -> Void

    init(title: String, handler: @escaping () -> Void) {
        self.handler = handler
        super.init(title: title, action: #selector(run), keyEquivalent: "")
        target = self
    }

    @available(*, unavailable)
    required init(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    @objc private func run() { handler() }
}
