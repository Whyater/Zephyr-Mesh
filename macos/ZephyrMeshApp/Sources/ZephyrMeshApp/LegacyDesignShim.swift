// TEMPORARY compatibility shim. Deleted by the macOS rebuild step.
import AppKit
import SwiftUI

enum ZephyrDesign {
    static let palette = Theme.dark
    static let window = palette.background
    static let surface = palette.surface
    static let groupedSurface = palette.background
    static let separator = palette.line
    static let primary = palette.text
    static let secondary = palette.textSecondary
    static let tertiary = palette.textSecondary.opacity(0.72)
    static let accent = palette.accent
    static let success = palette.ok
    static let warning = palette.caution
    static let critical = palette.alert

    enum Typography {
        static let title = TypeScale.title
        static let brand = TypeScale.headline
        static let icon = Font.system(size: 16)
        static let heading = TypeScale.headline
        static let row = TypeScale.body
        static let body = TypeScale.body
        static let secondary = TypeScale.caption
        static let caption = TypeScale.caption
        static let tiny = Font.system(size: 11)
        static let mono = TypeScale.mono
        static let tinyMono = Font.system(size: 11, design: .monospaced)
        static let metric = TypeScale.metricSmall
    }

    enum Controls { static let minHeight: CGFloat = LayoutMetrics.rowHeight }

    enum Layout {
        static let sidebarWidth = LayoutMetrics.sidebarWidth
        static let inspectorWidth = LayoutMetrics.inspectorWidth
        static let sceneMinimumWidth = LayoutMetrics.sceneMinWidth
        static let panelPadding = Space.lg
        static let overlayPadding = Space.md
        static let sectionSpacing = Space.sm
        static let rowVertical = Space.xs
        static let tightSpacing = Space.xxs
        static let microSpacing = Space.xxs
        static let smallSpacing = Space.xs
        static let compactSpacing = Space.sm
        static let controlSpacing = Space.md
        static let panelSpacing = Space.lg
        static let gridSpacing = Space.md
    }

    enum ScenePalette {
        static let body = palette.textSecondaryNS
        static let arm = palette.textSecondaryNS.withAlphaComponent(0.75)
        static let rotor = palette.textSecondaryNS.withAlphaComponent(0.42)
        static let target = palette.targetNS.withAlphaComponent(0.75)
        static let selection = palette.accentNS
        static let selectionEmission = palette.accentNS
        static let floor = palette.backgroundNS
        static let obstacle = palette.keepOutNS.withAlphaComponent(0.18)
        static let grid = palette.lineNS.withAlphaComponent(0.55)
    }

    enum Spacing {
        static let xxs = Space.xxs
        static let xs = Space.xs
        static let sm = Space.sm
        static let md = Space.md
        static let lg = Space.lg
    }

    enum Radius {
        static let control = CGFloat(RadiusValue.small)
        static let surface = CGFloat(RadiusValue.small)
    }

    private enum RadiusValue { static let small = 4 }
}
