import Foundation
import SwiftUI

struct ShortcutAction: Equatable {
    let actionID: String
    let key: String
    let modifiers: [String]
    let singleKey: Bool
}

struct ShortcutRouter {
    let actions: [ShortcutAction]

    init(shortcuts: [UIText.Shortcut] = UIText.shortcuts) {
        actions = shortcuts.map { ShortcutAction(actionID: $0.actionID, key: $0.key, modifiers: $0.modifiers, singleKey: $0.singleKey) }
    }

    func action(forKey key: String, modifiers: [String] = []) -> String? {
        actions.first { $0.key.caseInsensitiveCompare(key) == .orderedSame && $0.modifiers == modifiers }?.actionID
    }

    func shouldFire(key: String, modifiers: [String] = [], textFieldFocused: Bool) -> Bool {
        guard let action = actions.first(where: {
            $0.key.caseInsensitiveCompare(key) == .orderedSame && $0.modifiers == modifiers
        }) else { return false }
        return !(textFieldFocused && action.singleKey)
    }

    func shouldFire(actionID: String, textFieldFocused: Bool) -> Bool {
        guard let action = actions.first(where: { $0.actionID == actionID }) else { return false }
        return !(textFieldFocused && action.singleKey)
    }
}


extension ShortcutRouter {
    /// The SwiftUI key equivalent for a generated shortcut, so menus show and handle exactly
    /// the keys listed in design/ui_text.json.
    static func keyboardShortcut(for actionID: String, shortcuts: [UIText.Shortcut] = UIText.shortcuts) -> KeyboardShortcut? {
        guard let shortcut = shortcuts.first(where: { $0.actionID == actionID }),
              let key = keyEquivalent(shortcut.key) else { return nil }
        var modifiers: EventModifiers = []
        for modifier in shortcut.modifiers {
            switch modifier {
            case "primary": modifiers.insert(.command)
            case "shift": modifiers.insert(.shift)
            case "option": modifiers.insert(.option)
            case "control": modifiers.insert(.control)
            default: return nil
            }
        }
        return KeyboardShortcut(key, modifiers: modifiers)
    }

    static func keyEquivalent(_ key: String) -> KeyEquivalent? {
        switch key {
        case "space": return .space
        case "right": return .rightArrow
        case "left": return .leftArrow
        case "home": return .home
        default:
            guard key.count == 1, let character = key.lowercased().first else { return nil }
            return KeyEquivalent(character)
        }
    }

    static func label(for actionID: String) -> String {
        UIText.shortcuts.first { $0.actionID == actionID }?.label ?? actionID
    }
}

extension View {
    /// Attaches the generated shortcut for `actionID`, if one exists.
    @ViewBuilder
    func shortcut(_ actionID: String) -> some View {
        if let shortcut = ShortcutRouter.keyboardShortcut(for: actionID) {
            keyboardShortcut(shortcut)
        } else {
            self
        }
    }
}
