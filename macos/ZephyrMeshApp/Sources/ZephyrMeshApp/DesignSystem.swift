import AppKit
import SwiftUI

/// Shared visual tokens for the native cockpit. The system colors keep the
/// app legible in light and dark appearances and leave color for state only.
enum ZephyrDesign {
    static let window = Color(nsColor: .windowBackgroundColor)
    static let surface = Color(nsColor: .controlBackgroundColor)
    static let groupedSurface = Color(nsColor: .underPageBackgroundColor)
    static let separator = Color(nsColor: .separatorColor)
    static let primary = Color(nsColor: .labelColor)
    static let secondary = Color(nsColor: .secondaryLabelColor)
    static let tertiary = Color(nsColor: .tertiaryLabelColor)
    static let accent = Color.accentColor
    static let success = Color(nsColor: .systemGreen)
    static let warning = Color(nsColor: .systemOrange)
    static let critical = Color(nsColor: .systemRed)

    enum Typography {
        static let title = Font.system(size: 18, weight: .semibold)
        static let brand = Font.system(size: 15, weight: .semibold)
        static let icon = Font.system(size: 16)
        static let heading = Font.system(size: 13, weight: .semibold)
        static let row = Font.system(size: 12)
        static let body = Font.system(size: 13)
        static let secondary = Font.system(size: 11)
        static let caption = Font.system(size: 10)
        static let tiny = Font.system(size: 9)
        static let mono = Font.system(size: 11, design: .monospaced)
        static let tinyMono = Font.system(size: 9, design: .monospaced)
        static let metric = Font.system(size: 15, weight: .semibold, design: .monospaced)
    }

    enum Controls {
        static let minHeight: CGFloat = 28
    }

    enum Layout {
        static let sidebarWidth: CGFloat = 286
        static let inspectorWidth: CGFloat = 350
        static let sceneMinimumWidth: CGFloat = 560
        static let panelPadding: CGFloat = 18
        static let overlayPadding: CGFloat = 14
        static let sectionSpacing: CGFloat = 9
        static let rowVertical: CGFloat = 7
        static let tightSpacing: CGFloat = 3
        static let microSpacing: CGFloat = 2
        static let smallSpacing: CGFloat = 4
        static let compactSpacing: CGFloat = 8
        static let controlSpacing: CGFloat = 10
        static let panelSpacing: CGFloat = 16
        static let gridSpacing: CGFloat = 12
    }

    enum ScenePalette {
        static let body = NSColor.controlAccentColor
        static let arm = NSColor.secondaryLabelColor.withAlphaComponent(0.75)
        static let rotor = NSColor.secondaryLabelColor.withAlphaComponent(0.42)
        static let target = NSColor.controlAccentColor.withAlphaComponent(0.75)
        static let selection = NSColor.controlAccentColor.withAlphaComponent(0.9)
        static let selectionEmission = NSColor(calibratedRed: 0.49, green: 0.77, blue: 1, alpha: 1)
        static let floor = NSColor.underPageBackgroundColor
        static let obstacle = NSColor.systemOrange.withAlphaComponent(0.18)
        static let grid = NSColor.separatorColor.withAlphaComponent(0.28)
    }

    enum Spacing {
        static let xxs: CGFloat = 4
        static let xs: CGFloat = 8
        static let sm: CGFloat = 12
        static let md: CGFloat = 16
        static let lg: CGFloat = 24
    }

    enum Radius {
        static let control: CGFloat = 6
        static let surface: CGFloat = 10
    }
}
