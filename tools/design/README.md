# Zephyr Mesh design system

This folder is the single design source for the two desktop replay apps:

- the macOS app (SwiftUI and SceneKit) in `tools/macos/ZephyrMeshApp/`
- the Windows app (Tkinter) in `tools/desktop/`

Both apps replay a recorded simulation. Neither one talks to a radio, camera, motor, or drone. The design system keeps them looking and wording things the same way, and keeps them honest about what the data contains.

The Windows descriptions here come from the source and from running the Windows app on macOS (Tk 9; a few regression tests also on Tk 8.6, see `tools/desktop/README.md`). It has not yet been run on a Windows machine, so treat Windows fonts, scaling and exact rendering as unverified.

A visual version of the tokens (swatches, contrast ratios, type, spacing, status levels, icon) is in `tools/design/reference.html`. Open it in any browser; it needs no network.

Contents: [1 Principles](#1-principles) · [2 How the system works](#2-how-the-system-works) · [3 Color](#3-color) · [4 Status language](#4-status-language) · [5 Type, spacing, radii, motion, layout](#5-type-spacing-radii-motion-layout) · [6 Components](#6-components) · [7 Voice and wording](#7-voice-and-wording) · [8 Iconography and the app icon](#8-iconography-and-the-app-icon) · [9 Platform mapping](#9-platform-mapping) · [10 Updates and the self-test](#10-updates-and-the-self-test) · [11 Keyboard shortcuts](#11-keyboard-shortcuts) · [12 Checklist for a new screen](#12-checklist-for-a-new-screen) · [13 Open decisions](#13-open-decisions)

> Keeping this file current: sections 3, 4, 5 and 11 restate values from `tools/design/horizon_tokens.json` and `tools/design/ui_text.json`. When those files change, regenerate `tools/design/reference.html` (it is always exact) and update the matching tables here. Section 6 describes the views as they are in the source files it cites; re-read those files when a view changes.

## 1. Principles

**The first question is "is the tracker OK?"** The top of the window answers it before anything else: the status strip shows the tracking state as a shape, a color and a word ("Tracking held", "Tracking partial (47 of 50)", "Tracking lost", "Waiting for first target packet"), then four team numbers. Detail (one drone, raw vectors, hashes) is always one step further in, never in the way.

**Show only what the run contains.** Every number on screen comes from the replay file. One shared presentation layer (`tools/desktop/presentation.py`) reads the replay values and derives the display metrics (estimate error, speed, heading, link statistics, status levels); the views only format what it returns, plus raw values the Mac fixture copies straight from the run. A value the run does not hold reads **"Not recorded"** in the secondary text color. It is never shown as 0, as a blank, or as a guess.

**Say what this is, once.** Each main window has exactly one badge, **"Simulation replay"** (`honesty.badge` in `ui_text.json`), plus an **About this replay** window that states the run's status, evidence boundary, command authority and failsafe, lists what is not in the replay (`notInReplay`), and shows the command that regenerates it. Tests limit how often the word "synthetic" can appear in UI strings, so the badge text lives in one place (`tools/tests/test_macos_ui_hygiene.py`, `tools/tests/test_windows_ui_hygiene.py`).

**Color is never the only signal.** Every status pairs a shape, a color and a word (section 4). The 3D scene follows the same rule: a late or lost drone gets a shape above it, not just a different opacity.

**Calm by default.** No shadows or gradients in the app UI (the rule is the `comment` in `horizon_tokens.json`; the app icon is the one exception, section 8), hairline dividers, one accent color, and red or amber only when something is wrong. Motion only follows something the user did.

**One control height.** Buttons, segmented controls, the filter field and the scrubber row are all `layout.controlHeight` = 28 on both apps, so rows of controls line up.

## 2. How the system works

### The two source files

| File | Holds |
|---|---|
| `tools/design/horizon_tokens.json` | Colors for dark and light, 3D scene colors, type scale, spacing, radii, motion, layout sizes (shared `layout`, Windows-only `windowsLayout`), and the contrast rules |
| `tools/design/ui_text.json` | The honesty badge, status strip labels, status levels (word, shape, color), shape names per platform, display thresholds, field labels, drone naming, keyboard shortcuts, event sentence templates, and the shared sections listed below |

Shared sections in `ui_text.json` (both apps read them): `update`, `honesty`, `statusStrip`, `statusLevels`, `statusShapes`, `displayThresholds`, `fieldLabels` (including `at_rest` and `heading_vertical`), `droneNaming`, `shortcuts`, `eventTemplates`, `eventDroneList`, `notRecorded`, `runName`, `teamSummary`, `notInReplay`, `teamFilter`, `transport`, `menus`, `inspector`, `results` and `atlas`.

Two blocks are Windows-only:

- `windowsLayout` in `horizon_tokens.json` (Windows sizes in device-independent pixels, such as `sidebarWidth` 356, stroke widths, timings, and window sizes). It is emitted to Python only, as `WIN_LAYOUT`.
- `windowsText` in `ui_text.json` (Windows-only copy such as the Events hint, open-file errors, and the Loss and Age column titles). It appears in the Python `UI_TEXT` only; the Swift file does not contain it.

The legacy `layout.sidebarWidthWindows` (340) is not read by the Windows app, which uses `windowsLayout.sidebarWidth`. It stays only so the generated Swift file keeps its shape (comment in `tools/generate_design.py`).

Palette values, type sizes, spacing, radii and layout sizes, and the shared status, field, menu, transport, inspector and shortcut labels, reach the views only through the generated modules below. Some view-specific sentences (empty states, hints, section titles) still live in the view source.

### The generator and the drift check

`tools/generate_design.py` reads both JSON files and writes two generated modules:

- `tools/macos/ZephyrMeshApp/Sources/ZephyrMeshApp/DesignSystem.swift` (`Theme`, `ScenePalette`, `TypeScale`, `Space`, `Radius`, `Motion`, `LayoutMetrics`, `UIText`)
- `tools/desktop/design_tokens.py` (`palette()`, `scene3d()`, `TYPE`, `SPACE`, `RADIUS`, `MOTION_MS`, `LAYOUT`, `WIN_LAYOUT`, `SHORTCUTS`, `UI_TEXT`)

Both start with `GENERATED FILE. Do not edit.` Run `python3 tools/generate_design.py --check` to see whether they still match the JSON; it exits 1 and names the stale file if not. `tools/tests/test_design_system.py` runs that check and also proves a JSON edit flows into both outputs.

### The contrast check

`tools/check_contrast.py` computes the WCAG contrast ratio for every text color on the background, the surface, and the `selectionFill` and `hoverFill` fills, in both themes. It also checks `text` on the five Results fills (`bandFill` and `atlasStop1` to `atlasStop4`), and checks that the stored hex of every derived fill really is the blend their `base`, `alpha` and `over` describe. A pair below its target fails the run unless it is listed under `contrast.needs-approval`; a listed pair that now passes is reported as a stale mark. Results are in section 3.

### The shared presentation layer

`tools/desktop/presentation.py` turns a frame of the replay into what the screen shows: each drone's status level, the tracking headline, group rows, the team summary, link statistics, estimate error p95, and event sentences. It reads thresholds, words and templates from the generated `UI_TEXT`. The Windows app calls it live through `tools/desktop/replay_session.py`.

### The generated Mac fixture

The Mac app does not run Python. `tools/build_native_fixture.py` reads `sim/runs/s7-swarm/run.json` (and its `manifest.json`), runs the same presentation code, and writes `tools/macos/ZephyrMeshApp/Sources/ZephyrMeshApp/Resources/demo_swarm.json`, which embeds the presentation records next to the raw positions. Swift code only formats those records (`Presentation.swift`). `tools/build_presentation_golden.py` writes `tools/design/golden/s7_presentation.json`, the checked-in expected output, and `tools/tests/test_presentation.py` and `tools/tests/test_native_fixture.py` check that the golden file, the fixture and the code agree.

### Changing a color or a label in 4 steps

1. **Edit the source.** A color goes in `tools/design/horizon_tokens.json`; a label, word, threshold or template goes in `tools/design/ui_text.json`. If you change a color that a fill is derived from, update the fill's `hex` too (step 3 tells you the right value).
2. **Regenerate the modules.** `python3 tools/generate_design.py`
3. **Check contrast.** `python3 tools/check_contrast.py` must exit 0. If a pair drops below 4.5:1, pick a different value; adding a `needs-approval` entry is a decision for the project owner, not a fix.
4. **Rebuild what depends on it, then test.** If presentation wording changed, run `python3 tools/build_presentation_golden.py` and `python3 tools/build_native_fixture.py`. Always run `python3 tools/build_design_reference.py`. If you changed a dark color the icon uses (background, surface, lineStrong, text, accent, keepOut), run `python3 tools/build_icon.py` on a Mac. Then run `python3 -m pytest -q`.

## 3. Color

The palette is called Horizon: a night-sky navy with a light sky blue accent. Dark is the default theme. Light exists for bright rooms. Values below are from `tools/design/horizon_tokens.json`.

| Token | Dark | Light | Job |
|---|---|---|---|
| `background` | `#0F1522` | `#F6F8FB` | Window background and the 3D floor |
| `surface` | `#172033` | `#FFFFFF` (proposed) | Panels: team list, inspector, timeline, events (and the status strip on macOS) |
| `line` | `#2B3A55` | `#D5DCE6` (proposed) | Hairline dividers, tile borders, scroll track, scene grid |
| `lineStrong` | `#5B6B88` | `#7D8AA1` (proposed) | Badge border (macOS), event ticks on the scrubber, hovered button border, heavier grid lines |
| `text` | `#EEF2F7` | `#1B2433` | Primary text and values |
| `textSecondary` | `#9AA8C0` | `#56627A` | Labels, captions, "Not recorded", earlier events, the neutral drone glyph |
| `accent` | `#7CC4FF` | `#1F6FD1` | Selected drone, focus ring, scrubber fill and thumb |
| `target` | `#FFB38A` | `#A8501C` (proposed) | The target ring in the scene |
| `keepOut` | `#B4A7FF` | `#5E4BC4` (proposed) | Keep-out volume in the scene |
| `ok` | `#5FD39B` | `#1E7F4F` | Status OK only |
| `caution` | `#FFC857` | `#8A5A00` | Status Late only |
| `alert` | `#FF5C5C` | `#C62828` | Status Lost or Inactive, and the load-failure symbol |
| `info` | `#7CC4FF` | `#1F6FD1` | Info events |
| `onAlert` | `#0F1522` | `#FFFFFF` | Text placed on an alert or caution fill |
| `selectionFill` | `#273A54` | `#E0EBF9` | Selected row, pressed button, selected segment (accent at 0.16 over surface in dark, 0.14 in light) |
| `hoverFill` | `#242D3F` | `#F4F4F5` | Row or button under the pointer (text at 0.06 over surface in dark, 0.05 in light) |
| `bandFill` | `#273A54` | `#DBE8F8` | Results window: the 95 % interval band behind the recovery line (accent at 0.16 over surface) |
| `atlasStop1` | `#273A54` | `#E4EEF9` | Tolerance atlas cell, 0 to 25 ms (accent at 0.16 over surface in dark, 0.12 in light) |
| `atlasStop2` | `#324C6A` | `#C9DCF4` | Tolerance atlas cell, 50 to 100 ms (accent at 0.27 over surface in dark, 0.24 in light) |
| `atlasStop3` | `#3D5E81` | `#AAC8EE` | Tolerance atlas cell, 150 to 200 ms (accent at 0.38 over surface in both themes) |
| `atlasStop4` | `#487097` | `#84B0E6` | Tolerance atlas cell, more than 300 ms (accent at 0.49 over surface in dark, 0.55 in light) |

The five Results fills are derived the same way as `selectionFill`: each stores its `base`, `alpha`, `over` and the resulting `hex`. Only `text` is ever drawn on them. The atlas stops are a four step ramp of one hue, so the cell words carry the meaning and the tint only orders the values. The dark top stops use lower alphas than the light ones so `text` stays at or above 4.5:1. Swift reads them as `palette.bandFill` and `palette.atlasStop1` to `palette.atlasStop4`; Python as `palette(theme).bandFill` and so on.

"Proposed" means the token file marks the light value `"status": "proposed"`: it is in use but still waiting for sign-off.

The Windows app also derives a few shades in code (`tools/desktop/ui_theme.py`: `accentHover`, `accentPressed`, `badgeFill`) by mixing the tokens above. They are not tokens in the JSON.

The 3D scene has its own small palette (`scene3D`), built from the tokens above: drone body, arms and rotor discs use `textSecondary`, the target ring uses `target`, the grid uses `line`, and the floor uses `background`. The keep-out volume is `keepOut` at **0.12 alpha in dark and 0.18 in light** (`scene3D.dark.keepOutVolume` and `scene3D.light.keepOutVolume`).

### Red and amber mean alerts only

`alert` (red) and `caution` (amber) are status colors: `caution` marks Late, `alert` marks Lost and Inactive, and `alert` is also the symbol on the "replay could not be loaded" screen. They are never used for decoration, branding, selection, or emphasis. The app icon uses none of them.

### Contrast results

Output of `python3 tools/check_contrast.py` on this branch: 88 pairs checked (44 per theme), **82 PASS, 6 NEEDS-APPROVAL, 0 FAIL**, no stale fill values, exit code 0. Text needs 4.5:1; `lineStrong` needs 3:1 because it is a non-text line.

The closest passes, worth knowing before you nudge a value: dark `text` on `atlasStop4` 4.62:1, dark `lineStrong` on `surface` 3.02:1 (needs 3.0), light `ok` on `hoverFill` 4.54:1, light `target` on `selectionFill` 4.55:1, dark `alert` on `hoverFill` 4.56:1, light `accent` and `info` on `background` 4.64:1.

The six pairs below are too low for text and are avoided. The apps never put accent, info, OK or alert text on these fills; a selected row shows its words in `text` and keeps the status color for the shape only (`StatusLabel` in `Components.swift`).

| Theme | Text color | On fill | Ratio |
|---|---|---|---|
| dark | `alert` | `selectionFill` | 3.81:1 |
| light | `accent` | `selectionFill` | 4.10:1 |
| light | `ok` | `selectionFill` | 4.14:1 |
| light | `info` | `selectionFill` | 4.10:1 |
| light | `accent` | `hoverFill` | 4.49:1 |
| light | `info` | `hoverFill` | 4.49:1 |

These six are marked NEEDS-APPROVAL and are still waiting for the project owner's decision (section 13). These ratios are computed from hex values. Nobody has measured them on a real screen yet.

## 4. Status language

Every status is three things at once: a **shape**, a **color**, and a **word**. People with color blindness, a washed-out projector, or a grayscale screenshot still get the shape and the word, and a screen reader gets the word. Levels come from `ui_text.json` (`statusLevels`, `statusShapes`).

| Level | Word | Shape | Color token | macOS symbol | Windows glyph |
|---|---|---|---|---|---|
| `ok` | OK | filled circle | `ok` | `circle.fill` | ● |
| `caution` | Late | filled triangle | `caution` | `triangle.fill` | ▲ |
| `alert` | Lost (or Inactive) | filled square | `alert` | `square.fill` | ■ |
| `waiting` | Waiting | hollow circle | `textSecondary` | `circle` | ○ |
| `unavailable` | Not recorded | dash | `textSecondary` | `minus` | – |
| `info` | Info | info mark | `info` | `info.circle` | ⓘ |

The Windows app draws these shapes as small antialiased PNGs (`STATUS_SHAPES` in `tools/desktop/ui_icons.py`) rather than font glyphs, so they look the same on every font; the glyphs in the table are the text fallback stored in `ui_text.json`.

**How a drone gets its level** (`status_for_agent` in `tools/desktop/presentation.py`): an inactive drone is Lost with the word "Inactive"; a drone with no fused target estimate, or no target age, is Waiting; otherwise its target age decides.

**Display thresholds** (`displayThresholds` in `ui_text.json`):

| Target age | Level |
|---|---|
| below 0.15 s | OK |
| 0.15 s or more | Late |
| 0.30 s or more | Lost |

These are display choices, not safety limits. The JSON says so in each threshold's `note`.

**Team headline** (`tracking_headline`): "Waiting for first target packet" until any active drone has an estimate; "Tracking held" when every active drone's target age is under the Lost threshold; "Tracking lost" when none is; otherwise "Tracking partial (N of M)".

## 5. Type, spacing, radii, motion, layout

### Type scale

| Role | Size | Weight | Use |
|---|---|---|---|
| `caption` | 12 | regular | Secondary labels only |
| `body` | 13 | regular | Most text and values |
| `bodyStrong` | 13 | semibold | Status words, section headers, row names |
| `headline` | 15 | semibold | Status shape in the strip, missing strip values, transport glyphs, toolbar glyphs, the inspector name on Windows |
| `title` | 20 | semibold | Drone name in the inspector (macOS), window titles, the Play and Pause glyph (macOS) |
| `metric` | 22 | semibold, tabular digits | Status strip numbers and the tracking headline |
| `metricSmall` | 15 | semibold, tabular digits | Inspector metric tiles |
| `mono` | 12 | regular | Hashes and IDs inside Details only |

Fonts: SF Pro through `Font.system` on macOS (SF Mono or Menlo for mono); Segoe UI Variable Text with Segoe UI as fallback on Windows (Cascadia Mono or Consolas for mono). Sizes are points on macOS; on Windows they are scaled by the display DPI (`ui_scale` in `tools/desktop/ui_theme.py`). Numbers use tabular digits so columns do not jitter during playback.

### Spacing

`xxs` 2, `xs` 4, `sm` 8, `md` 12, `lg` 16, `xl` 24, `xxl` 32. Use these only. Small sums such as `sm + xxs` appear where a control needs an in-between value; a brand new raw number does not.

Common uses: 16 around panel content (the content inset), 12 between strip and list sections, 8 inside rows and between related controls, 2 to 4 between buttons in one group.

### Radii

`small` 4 for controls, panels, rows and the badge. `medium` 8 for sheets only. On Windows the same 4 is baked into the button images (section 6, Buttons).

### Motion

| Token | Value |
|---|---|
| `fastMs` | 120 ms (group expand and collapse) |
| `standardMs` | 180 ms (scrolling to a selection, new events) |
| `curve` | ease-out |
| `userTriggeredOnly` | true: nothing moves unless the user (or playback they started) caused it |
| `reducedMotionMs` | 0 |

With Reduce Motion on (macOS) or animations turned off in Windows settings (`system_reduced_motion` in `tools/desktop/ui_settings.py`), durations become 0 and the camera jumps instead of easing (`Motion.duration` in `DesignSystem.swift`, `reduced_motion` in `tools/desktop/scene_view.py`).

### Layout sizes

From the shared `layout` block (points on macOS; Windows scales the same numbers by DPI). Only the tokens a contributor is likely to need are listed; the full set is in `tools/design/reference.html`.

| Token | Value | Used for |
|---|---|---|
| `windowMinWidth` × `windowMinHeight` | 1180 × 760 | Smallest window |
| `windowDefaultWidth` × `windowDefaultHeight` | 1440 × 900 | First launch |
| `contentInset` | 16 | Left and right inset of the strip, scene corner, timeline, events and viewer content (macOS) |
| `controlHeight` | 28 | Every button, segmented control, filter field and scrubber row, both apps |
| `iconButtonSize`, `iconGlyphSize` | 28, 14 | Icon button hit area and the glyph inside it |
| `rowHeight` | 28 | List rows |
| `viewSegmentWidth`, `speedSegmentWidth` | 44, 48 | macOS 3D or Top and speed segments (every segment in one control has the same width) |
| `sidebarWidth`, `sidebarIdealWidth` | 260, 300 | macOS team list (min and ideal) |
| `inspectorWidth`, `inspectorMaxWidth` | 300, 360 | Inspector column |
| `sceneMinWidth` | 520 | 3D view |
| `stripOneRowMinWidth` | 960 | macOS strip: one row at or above this width, otherwise the headline sits above the four numbers |
| `eventRowHeight`, `eventRowsRegular`, `eventRowsCompact` | 24, 6, 4 | macOS events pane; compact below `compactDetailHeight` (820) |
| `sceneStatusMarkerPoints`, `sceneSelectionRingPoints` | 11, 15 | macOS on-screen size of the late and lost marker and the selection ring radius |
| `viewerTwoColumnWidth`, `scenarioSideWidth` | 960, 360 | macOS Scenario Run switches to two columns at 960 wide; the summary column is 360 |

| `resultsWindowWidth` × `resultsWindowHeight` | 1240 × 840 | Results window default size (minimum `resultsWindowMinWidth` × `resultsWindowMinHeight`, 980 × 680) |
| `resultsSidebarWidth` | 200 | Results window sidebar |
| `chartPanelMinWidth`, `chartPanelHeight` | 260, 220 | One small chart panel in "Delay and loss" |
| `chartPlotInset` | 36 | Room left and below a chart's plot area for tick labels |
| `chartLineWidth`, `failureLineWidth` | 1.5, 1.5 | Series line and failure line strokes |
| `chartMarkerSize`, `chartCapWidth`, `chartHoverRing` | 6, 6, 10 | Point marker, interval cap on the basic tracker, hover ring |
| `atlasCellWidth` × `atlasCellHeight` | 104 × 36 | One tolerance atlas cell |
| `atlasRowHeaderWidth`, `atlasWinnerBorder` | 168, 2 | Atlas row header column; inner border on a clear winner |
| `legendSwatchWidth` × `legendSwatchHeight` | 22 × 10 | Legend swatches under the Results visuals |

Windows-only sizes live in `windowsLayout` (`WIN_LAYOUT`): for example `sidebarWidth` 356, `eventRows` 5, `statusMarkerSize` 9, `scaleBarMinLength` 48, and the window sizes of the three viewers. The Results window adds `resultsWindow` (width, height, minWidth, minHeight), `resultsSidebarWidth`, `chartPanelHeight`, `chartPlotInset` (40), `chartMarkerSize`, `chartCapWidth`, `atlasCellWidth`, `atlasCellHeight`, `atlasRowHeaderWidth`, `legendSwatchWidth`, `legendSwatchHeight` and `dashPattern` ([4, 3], the Tk dash for the basic tracker line).

## 6. Components

Each entry: what it is for, what it is made of, its states, and where it lives.

### Status strip
- **Purpose:** answer "is the tracker OK?" at a glance.
- **Anatomy:** the tracking headline (status shape in `headline` size, text in `metric` size), then four metrics with a caption over a value: Estimate error p95, Target link delay, Target packet loss, Target age (labels from `statusStrip`). Vertical hairlines separate the numbers. Hairline below.
- **States:** a missing value shows "Not recorded" in `headline` type and `textSecondary`. Target age shows a newest to oldest range (for example "0.05–0.40 s") when drones differ.
- **Platform differences:** on macOS the headline has no caption (the text already names the subject), the strip sits on `surface`, and it is one row at `stripOneRowMinWidth` or wider, otherwise the headline sits on its own row above the numbers. On Windows every cell, including the headline, has a caption ("Tracking", "Estimate error p95", ...) over its value, and the strip sits on `background`.
- **Where:** `StatusStripView.swift`; Windows: `_build_strip` in `tools/desktop/windows_preview.py`.

### Simulation replay badge
- **Purpose:** say once, everywhere, that this is a recorded simulation.
- **Anatomy:** caption text in `textSecondary`, `small` radius, `badgeHeight` (22) tall on macOS. macOS draws a 1 pt `lineStrong` outline and no fill. Windows draws a quiet tinted pill with no border (fill is `badgeFill`, a mix of `lineStrong` and `surface`). Tooltip on macOS: every value comes from a recorded simulation file, not a live connection.
- **Where:** `HonestyBadge` in `Components.swift` (toolbar, leading); `Badge.TLabel` in `tools/desktop/ui_theme.py` (title row, after the run name).

### Window layout and toolbars
- **macOS:** the toolbar holds view controls only: the 3D or Top segmented control, About this replay, and the inspector toggle, with one `Space.sm` gap between them (`CockpitView.swift`). The "Simulation replay" badge sits at the leading edge. The window title is "Zephyr Mesh" with the run name as subtitle. Playback is not in the toolbar.
- **Windows:** one title row holds the name, run name and badge on the left; a notice area, the 3D and Top segments and an About button on the right (`_build_title`). There is no toolbar toggle for the inspector: it is in the View menu (View > Show Inspector, shortcut I). The menu bar is built from `MENU_LAYOUT` using the shared `menus` labels.
- **Content inset:** on macOS the strip, scene legend, timeline, events and viewer content all start `contentInset` (16 pt) from the edge, and the first glyph's ink, not its invisible hit square, lines up with it (`QuietButtonStyle.glyphInset`). Windows uses `SPACE["lg"]` (also 16) for the same job.

### Team list: group rows and drone rows
- **Purpose:** show 50 drones without 50 rows. There is one group row per status level and word (all late drones form one group, all OK drones another), and inside a group the drones are listed stalest first (oldest target age first).
- **Anatomy:** a summary line on top ("50 drones: 47 OK, 3 late"), the filter field, column captions, then group rows (chevron, status shape, status word, count, age, loss) with their drone rows (compact name such as D07, age, loss) indented under them. Rows are `rowHeight` tall.
- **Columns:** the headers carry the units: "Age (s)" and "Loss (%)" (`fieldLabels` `age_s` and `loss_pct`; on Windows `windowsText` `columnAge` and `columnLoss`), so cells hold bare numbers. A group shows age, delay and loss as a min to max range when its members differ. macOS shows Status, Age, Loss and rounds a loss range to whole percents (a single loss keeps one decimal). Windows shows Drones, **Loss, then Age** (the order of `TEAM_COLUMNS`) and keeps one decimal for loss, range ends included.
- **Missing values:** a missing cell shows a dash. On macOS its tooltip reads "Not recorded". On Windows the tooltip reads the same and a caption under the list ("– means not recorded") appears whenever a visible row shows a dash.
- **States:** groups sort Lost first, then Late, Not recorded, Waiting, and OK last (`LEVEL_ORDER`). On macOS, groups that need attention (not OK) and the group holding the selected drone are open by default, OK groups are closed, and a click on a header overrides that. Hover uses `hoverFill`, selection uses `selectionFill`, the selected drone's name turns `bodyStrong`. A keyboard-focused list gets a 1 pt accent outline. Right-click: Focus Camera, Show in Inspector, Copy Drone ID.
- **Filter rule** (`teamFilter` in `ui_text.json`, `filter_group_rows` and `team_filter` in `tools/desktop/presentation.py`, mirrored in `TeamPresentation` in `Presentation.swift`):
  - Only the late and lost groups are pinned (`pinnedLevels`): they keep every member whatever you type. Every other group keeps only the drones whose name matches (D07, Drone 07, or the raw ID) and disappears when none match.
  - While a filter is active, a caption reads "Showing N of M". When a pinned group is listed, the caption continues "Late and lost drones stay listed."
  - Group headers count the filtered members against the whole group: "1 of 47 drones" on macOS, "OK · 1 of 47" on Windows.
  - When nothing matches, the empty state reads `No drones match “query”` (curly quotes). On macOS it sits in the list with a **Clear filter** button; on Windows it is a centered label in the list and the filter field's clear icon (tooltip "Clear filter (Esc)") is shown.
  - **macOS only:** the selected drone stays listed in its group with a pin mark even when it does not match the filter, so the scene and inspector selection always has a row. The caption counts it. Windows does not do this.
- **Where:** `TeamListView.swift` (`GroupRow`, `DroneRow`, `TeamColumns`, `TeamContextMenu`); Windows: a `ttk.Treeview` styled `Zephyr.Treeview` (`_build_team`, `team_rows` in `tools/desktop/windows_preview.py`, styles in `tools/desktop/ui_theme.py`).

### Filter field
- **Purpose:** find a drone by name (D07, Drone 07, or its raw ID).
- **Anatomy:** filter icon, placeholder "Filter drones" (Windows adds the shortcut), clear button when not empty. `background` fill inside the `surface` panel, `line` border, `small` radius, `controlHeight` (28) tall.
- **States:** focus turns the border `accent`. Escape clears it and leaves the field. While it has focus, single-key shortcuts are off.
- **Where:** `filterField` in `TeamListView.swift`; `Filter.TEntry` plus `Placeholder` in `tools/desktop/ui_theme.py` and `tools/desktop/ui_widgets.py`.

### Inspector
- **Purpose:** everything about one selected drone.
- **Anatomy, top to bottom:** the header (drone name, "Drone 07", with its status label); four metric tiles in a 2 by 2 grid, **in the same order as the status strip**: Estimate error, Target link delay, Target packet loss, Target age; "Position and motion" (altitude, speed, heading); "Neighbors" (count, nearest, neighbor link delay and loss); then a hairline and a collapsed **Details** disclosure with raw position, velocity and fused target vectors, constraint flags, hardware profile parts, and the drone ID in `mono`.
- **Heading text** (`heading_text` in `tools/desktop/ui_format.py`, `Format.heading` in `Presentation.swift`): degrees when a heading exists; "At rest" when a recorded speed is below 0.01 m/s; "Straight up or down" when the drone is moving but has no horizontal motion; "Not recorded" when the run has no velocity for it. The words are `fieldLabels` `at_rest` and `heading_vertical`.
- **Metric tile:** caption over a `metricSmall` value, 1 pt `line` border, `small` radius. On macOS, missing values drop to `body` type in `textSecondary`; on Windows the tile keeps the same font and only the color changes, so a tile never changes height.
- **States:** nothing selected shows an empty state: "No drone selected" and "Click a drone in the team list or the scene." (`inspector` in `ui_text.json`). macOS remembers whether Details was open. The drone name is `title` size on macOS and `headline` size on Windows.
- **Where:** `InspectorView.swift`, `MetricTile` and `ValueRow` in `Components.swift`; `_build_inspector`, `MetricTile`, `KeyValueGrid`, `Disclosure` in `tools/desktop/windows_preview.py` and `tools/desktop/ui_widgets.py`. Toggle with I.

### Timeline and transport controls
- **Purpose:** move through the replay. Playback lives only in this row on macOS (the toolbar has none), and in this row plus the Playback menu on Windows.
- **Anatomy:** Restart, Step back, Play or Pause, Step forward; a scrubber with event ticks; the time ("12.40 / 30.00 s"); the speed choice (0.1x, 0.25x default, 1x).
- **Play and Pause:** macOS draws the glyph larger (`title` size) than the other three (`headline`). Windows keeps the same glyph size and gives Play an accent fill (`Primary.Icon.TButton`, icon in `onAccent`).
- **Scrubber:** 4 pt `line` track, `accent` fill and 14 pt thumb with a surface ring; `lineStrong` ticks under the track mark frames with events. On macOS they are 2 by 6 pt and their tooltip lists that frame's events; on Windows they are scaled to the display and have no tooltip. Dragging pauses and snaps to the nearest frame.
- **Speed choice:** a segmented control with equal-width segments (macOS `speedSegmentWidth` 48 pt; Windows sizes every segment to the longest label). Windows puts a "Speed" caption before it.
- **States:** on macOS, Restart and Step back are disabled at the start and Step forward at the end. Disabled controls use a dimmed `textSecondary`.
- **Where:** `TimelineView.swift` (`ReplayScrubber`, `SegmentedChoice`); Windows: `_build_timeline` in `tools/desktop/windows_preview.py` and `TimelineSlider` in `tools/desktop/ui_widgets.py`.

### Event list
- **Purpose:** a plain-language log of what changed.
- **Anatomy:** caption "Events"; rows with status shape, status word, time, and the sentence (section 7). Oldest at the top, newest at the bottom; it scrolls to the newest as the replay plays. macOS shows 6 rows (4 when the window is shorter than 820 pt) and never more rows than the run has events; Windows shows 5 (`eventRows`).
- **States:** on both apps, events at the current time use `text` and earlier ones use `textSecondary` (Windows tags rows `current` and `past`). Hover uses `hoverFill`. Clicking an event pauses and seeks to it.
- **Hint:** Windows shows the caption "Click an event to jump to it" (`windowsText` `eventsHint`) at the right of the header. macOS has no caption; each row's tooltip says "Click to jump to this event."
- **Drone lists in events:** at most 5 compact names, then "+N more" (`eventDroneList` in `ui_text.json`); the event record still keeps every ID. On macOS, when a row is too narrow, it drops whole names and raises the "+N more" count instead of cutting inside a name (`EventText.variants` in `Presentation.swift`).
- **Where:** `EventListView.swift`; `Events.Treeview` and `_build_events` in `tools/desktop/windows_preview.py`.

### Viewer windows
Evidence reports, scenario runs and About this replay open in their own windows, and they all follow one order:

1. **Verdict or plot first:** the report's declared status and limitations, or the scenario run's top-down plot of the selected frame with an altitude rail.
2. **Summary second:** plain labeled rows using the shared field labels.
3. **Details last, collapsed:** hashes, schemas, versions, provenance, raw data, in `mono` where it is an ID or hash.

Platform notes:

- **Scenario Run, macOS:** two columns at `viewerTwoColumnWidth` (960 pt) or wider, with the plot on the left and a 360 pt summary column on the right; narrower windows stack the plot above the summary. The plot has a scale bar and an altitude rail.
- **Scenario Run, Windows:** a single column (plot, legend, frame controls, summary), also with a scale bar and an altitude rail.
- **Evidence report, macOS:** one budget of 100 rows (`JSONTree.limit`) covers the whole summary: plain values, group headers and group rows. When it runs out, the rest is counted as "N more in Details" and left to the Details raw tree. **Windows has no row cap** in its evidence summary (`evidence_window`).
- **Restoration:** macOS does not reopen About, Scenario Run or Evidence Report windows at launch (`NotRestorable` in `ZephyrMeshApp.swift`), because their documents live only in memory.

Where: `EvidenceReportView.swift`, `ScenarioRunView.swift`, `AboutReplayView.swift`; `evidence_window`, `scenario_window`, `about_window` in `tools/desktop/ui_viewers.py`.

### Results window
One window with two views, "Delay and loss" and "Tolerance atlas", opened from View, Show Results (Primary + R). It reads only the generated results fixture, never raw run files.

- **Colors:** background, surface, `line` and `lineStrong`, `text` and `textSecondary`, `accent` with `bandFill` and the four `atlasStop` tints, and `alert`. `alert` appears only as the failure line in the delay charts and as the small square on atlas entries that fail even at 0 ms. `caution` is not used. No gradients, no shadows.
- **Charts:** the recovery tracker is a solid `accent` line with filled markers and a `bandFill` interval band. The naive tracker is a dashed `lineStrong` line with hollow markers and interval caps. Lines, markers and caps use the `chart*` sizes in section 5. The four panels share one error axis, chosen for the selected path and loss type, and a caption next to the controls names its top.
- **Atlas cells:** `atlasCellWidth` by `atlasCellHeight`, filled with an `atlasStop` tint by value, with the value in `text`. A clear winner gets bold text and an `accent` bar of `atlasWinnerBar` along the bottom edge; the selection ring is drawn outside the cell, so the two never look alike. Legends sit above the charts and the grid.
- **Wording:** all copy comes from the `results` and `atlas` sections of `ui_text.json` (Swift `UIText.Results` and `UIText.Atlas`, Python `UI_TEXT["results"]` and `UI_TEXT["atlas"]`). Templates use `{name}` placeholders that the view fills in. The window carries one "Simulated results" badge.

### Buttons

Every button, segmented control and the filter field are `controlHeight` (28) tall. Icon buttons are 28 by 28.

| Kind | Look | macOS | Windows |
|---|---|---|---|
| Icon button | Icon in `text`, 28 by 28, `hoverFill` on hover, `selectionFill` when pressed or active | `QuietButtonStyle` (`Components.swift`): fixed `iconButtonSize` hit square, no border at rest, `small` rounded fill | `Icon.TButton`: rounded image with a 1 px `line` border at rest |
| Primary icon button | Accent fill, icon in `onAccent` | not used (Play is a larger quiet glyph) | `Primary.Icon.TButton` (Play) |
| Text button | Body text, `controlHeight` tall | `QuietButtonStyle(bordered: true)`, for example Clear filter (1 pt `lineStrong` outline, `Space.md` side padding) | `Tool.TButton`, for example About and Copy |
| Disclosure | Chevron plus `bodyStrong` title, no border | `DisclosureGroup` | `Link.TButton` inside `Disclosure` |
| Segmented choice | Equal-width segments; selected: `text` on `selectionFill`, semibold; others `textSecondary` | `SegmentedChoice`, each segment `segmentWidth` wide, 1 pt `line` outline | `Segment.Toolbutton`, 2 pt gap between segments |

**Windows button images.** `clam` draws square, notched borders, so the Windows app draws buttons, segments, the filter field, tiles and the badge from rounded 9-slice images (`RoundedElements` in `tools/desktop/ui_theme.py`). The corner radius is `RADIUS.small` (4), scaled by DPI, and the images are recolored in place when the theme changes. The widget background stays the panel color in every state (`_same_in_every_state`): the image draws the fill and border, and the corners stay transparent over the panel, so a hovered or pressed button never shows a square light patch. States for the `Quiet` element: rest `surface` with `line`; hover `hoverFill` with `lineStrong`; pressed `selectionFill` with `lineStrong`; keyboard focus `accent` border; disabled dimmed text.

Spacing rules: segments in one segmented control sit 2 pt apart (`Space.xxs`) and transport buttons 4 pt apart (`Space.xs`), on both apps; separate groups are 8 to 16 pt apart (toolbar groups `Space.sm`, timeline groups `Space.lg`). Every icon-only button has a tooltip naming the action; on Windows the tooltip also gives the shortcut. Focus shows as an `accent` border, never the system accent color.

### Empty and error states
- **Empty:** a quiet icon in `textSecondary`, a short `bodyStrong` line saying what is missing, and one `body` line saying what to do. Example: "No drone selected" / "Click a drone in the team list or the scene."
- **Replay failed to load (macOS):** the alert status shape, "The replay could not be loaded", the decoder's message (selectable), and "Rebuild the replay fixture and the app, then open it again." (`LoadFailureView` in `ZephyrMeshApp.swift`). The app never fills in invented drones.
- **File would not open:** "Could not open evidence report" or "Could not open scenario run" with the validation message (`AppStore.swift`). On Windows the message is a plain sentence per cause (missing file, not a file, no permission, not UTF-8, not JSON, not a valid report), from `windowsText` and `open_error_text` in `tools/desktop/windows_preview.py`.
- **Windows start-up failure:** "Zephyr Mesh could not start." with the error, the path of a written diagnostic file, and a hint about extracting the whole ZIP (`show_startup_failure`).

### The 3D scene
- **Drone glyph:** an X-frame quadcopter in one neutral color (`textSecondary`). On macOS it is a procedural model at true scale from the drone's hardware profile (arm length and propeller size, `DroneModel.swift`); on Windows a Canvas drawing with four rotor discs (`tools/desktop/scene_view.py`). The selected drone uses `accent`.
- **Late and lost markers (both apps, 3D and Top):** a late drone gets an amber triangle and a lost drone a red square above it, in the shared status shapes and colors. The marker keeps a fixed on-screen size (11 pt on macOS, 9 px on Windows) and is drawn on top of the drones, so status is never opacity alone. macOS does not fade drones for status; it outlines the marker in the floor color and lifts it above the selection ring on the selected drone. Windows still dims a late or lost drone's body when it is not selected, in addition to the marker. (`SceneController.swift` `statusMarker`; `status_points` and `MARKED_LEVELS` in `tools/desktop/scene_view.py`.)
- **Legend:** macOS shows one chip at the top-left on `surface` with Target, Keep-out, Selected, Late, Lost and Grid 1 m (`SceneLegend` in `SceneView.swift`). Windows shows the Late and Lost shapes with their words at the top-left (`_draw_legend`) and a grid label next to the axis gizmo.
- **Selection:** on macOS a ring in `accent` circles the drone at its altitude, faces the camera and stays 15 pt in radius on screen; it hides once the camera is close enough that the drone is wider than the ring, because the drone's own accent shell then marks the selection. On Windows the selected drone is drawn in `accent` with its compact name beside it.
- **Target:** a ring in `target` with a thin stem down to the floor and a small foot ring.
- **Keep-out sphere:** `keepOut` fill at 0.12 alpha in dark and 0.18 in light, with an outline ring that always faces the camera and a second ring at the equator (`updateKeepOut` in `SceneController.swift`). K shows or hides it. The Mac fixture builds the sphere from the scenario constraints (0.75 m radius); the Windows scene draws keep-out volumes only when the replay frame contains them, and the canonical run's frames have none.
- **Floor grid:** 1 m squares with a heavier line every 5 m, on the `background` floor.
- **Orbit camera:** drag to orbit, scroll or pinch to zoom, both eased and clamped. Pitch stays between 10 and 85 degrees, there is no roll and no panning. T switches to a straight-down top view. F focuses the selected drone, R resets. The Mac scene never uses SceneKit's free camera (`tools/tests/test_macos_ui_hygiene.py`).
- **Zoom-out limit:** macOS stops at **3 times the swarm radius** (`OrbitCamera.maximumDistance`). Windows stops at **6 times** for both the orbit and the top view (`FAR_LIMIT_RADII` in `tools/desktop/scene_projection.py`, chosen because a wide, short canvas needs more than 3x to fit the ring). The two apps disagree; this is a pending decision (section 13).

## 7. Voice and wording

- **Plain sentences.** "Target packet lost for 3 drones", not "PKT_LOSS x3".
- **Units on labels.** The label carries the unit and the value stays a number: "Position (m)" over "6.00, 0.00, 1.50"; "Age (s)" and "Loss (%)" over bare numbers in the team list. Single values elsewhere show their unit: "0.12 s", "18 ms", "2.5%".
- **No snake_case on screen.** Every data key has a label in `fieldLabels` (`agent_id` reads "Drone ID", `seed` reads "Random seed"). A key without a label falls back to spaces and a capital letter, never the raw key (`Format.label` in `Presentation.swift`, `field_label` in `tools/desktop/ui_format.py`). Hygiene tests reject snake_case in visible strings.
- **Drone names.** "D07" in lists, "Drone 07" in headings (`droneNaming`).
- **Missing is "Not recorded".** Never 0, never "N/A", never empty. In narrow table cells a dash stands in for it, and the tooltip or a caption says "Not recorded".
- **Say why there is no heading.** "At rest" and "Straight up or down" explain a missing heading; "Not recorded" is only for a run that has no velocity.
- **Curly quotes** in `No drones match “query”`.
- **No hype and no em dashes.** Describe, do not sell.

Event sentences come from `eventTemplates`:

| Template | Example |
|---|---|
| `replayLoaded` | Replay loaded: 50 drones |
| `estimateReached` | Target estimate reached 48 of 50 drones |
| `estimateLate` | 3 drones newly late: D04, D17, D33 (0.18 s) |
| `backOnTime` | 3 drones back on time |
| `inactive` | D12 went inactive |
| `packetLost` | Target packet lost for 2 drones |

Other shared wording lives in `menus` (menu item titles), `transport` (button labels and tooltips), `inspector` (section titles and empty state), `teamFilter`, `runName` ("50-drone ring, seed 7"), `teamSummary` ("50 drones: 47 OK, 3 late") and `update` (section 10).

| Do | Don't |
|---|---|
| Tracking partial (47 of 50) | Degraded mode |
| Target age 0.32 s, status Lost | Red dot with no word |
| Not recorded | 0.00, N/A, or a blank |
| Simulation replay | Live, Connected, or Online |
| Estimate error p95 | est_err_p95 |
| Could not open evidence report: (reason) | Error |

## 8. Iconography and the app icon

### Interface icons
Status shapes are listed in section 4. Transport and disclosure icons are SF Symbols on macOS (`backward.end.fill`, `backward.frame.fill`, `play.fill`, `pause.fill`, `forward.frame.fill`, `chevron.right`) and matching shapes drawn as PNGs on Windows (`ICONS` in `tools/desktop/ui_icons.py`). Icons take the `text` color, or `textSecondary` when they are hints.

### The app icon

![Zephyr Mesh app icon](icon/ZephyrMesh-256.png)

**Idea.** A zephyr is a gentle west wind. A thin gust comes in from the west (left), thickens, and wraps around into an orbit. Five drones sit on that orbit, and thin links run from each one to a bright point in the middle: the target the team tracks together. A second, shorter breeze runs under the gust so the motion reads as moving air.

**Construction** (`tools/make_icon.swift`, 1024 by 1024 design space):

- Body: an 824 by 824 rounded square centered on the canvas (100 px margin), continuous corners with radius 185, following the macOS Big Sur icon grid. A soft drop shadow (10 px down, 28 px blur) and a subtle top-to-bottom gradient from `surface` to `background`. A faint `lineStrong` inner edge keeps the tile's outline visible on a dark Dock.
- Colors, all read from the dark palette in `horizon_tokens.json`: orbit and drones in `accent` (sky), links, glow and breeze in `keepOut` (lavender), the tracked point in `text`. No red, amber, or green.
- The gust and the orbit are one stroke: a curve that meets the circle at the bottom, tangent to it, then runs most of the way around and ends under the last drone. Each drone has a small gap around it so lines never touch the dot.

**Small-size rule.** At 16 and 32 px, the links, the glow and the breeze are left out; the orbit gets thicker, the center gets larger, and the drones shrink into the ring, so it reads as a gust wrapped around a point. At 48 and 64 px the full design is drawn, and the thin links and the breeze are kept at least one pixel wide.

**Windows frames** use the same drawing on a larger body (40 px margin, no drop shadow), because Windows adds no margin of its own.

**Exception to the token rule.** The app UI uses no shadows or gradients. The icon is the one place with a drop shadow and a gradient, because it follows the macOS app icon grid, which expects depth. This exception is pending the project owner's approval (section 13).

**Files** (all written by `python3 tools/build_icon.py`, which needs macOS with Swift and `iconutil`):

| File | Contents |
|---|---|
| `tools/design/icon/ZephyrMesh.icns` | macOS icon: 16, 32, 64, 128, 256, 512 and 1024 px |
| `tools/design/icon/ZephyrMesh.ico` | Windows icon: 16, 24, 32, 48, 64, 128, 256 px, PNG frames (packed by `tools/pack_ico.py`, standard library only) |
| `tools/design/icon/ZephyrMesh-256.png`, `tools/design/icon/ZephyrMesh-1024.png` | Flat PNGs for docs and the reference page |

The build is deterministic: running it twice gives identical bytes. `tools/tests/test_app_icon.py` checks the sizes inside each file by reading their headers.

**Where the icon is used:**

- **macOS bundle:** `tools/macos/build_app.py` copies `ZephyrMesh.icns` into the app's Resources and sets `CFBundleIconFile` in `Info.plist`. If the file is missing it prints a warning and builds without an icon.
- **Windows executable:** `tools/desktop/build_windows.ps1` passes `--icon design\icon\ZephyrMesh.ico` to PyInstaller and bundles the 256 px PNG under `design\icon`.
- **Windows window:** the app sets the Tk window icon from the 256 px PNG at start-up (`set_window_icon` in `tools/desktop/ui_icons.py`, called from `tools/desktop/windows_preview.py`).

## 9. Platform mapping

| Concept | macOS (SwiftUI) | Windows (Tk) |
|---|---|---|
| Generated tokens | `DesignSystem.swift`: `Theme`, `TypeScale`, `Space`, `Radius`, `Motion`, `LayoutMetrics`, `UIText` | `tools/desktop/design_tokens.py`: `palette()`, `TYPE`, `SPACE`, `RADIUS`, `MOTION_MS`, `LAYOUT`, `WIN_LAYOUT`, `UI_TEXT` |
| Windows-only tokens | none | `WIN_LAYOUT` (from `windowsLayout`), `UI_TEXT["windowsText"]` |
| Theme and appearance | `@Environment(\.palette)`; Settings > Appearance (System, Dark, Light) | `Theme` and `configure_styles` on ttk `clam` (`ui_theme.py`); View > Appearance, saved in `~/.zephyr-mesh/settings.json` |
| Menus | `CockpitCommands.swift` | `MENU_LAYOUT` and `_build_menu` |
| Status shape | SF Symbol from `statusShapesMacOS` (`StatusShape`) | PNG shape from `ui_icons.STATUS_SHAPES` (`ImageBank.status`) |
| Status label | `StatusLabel` | shape image plus word in a label |
| Honesty badge | `HonestyBadge` (outline) | `Badge.TLabel` (tinted pill) |
| Team list | `TeamListView` (`GroupRow`, `DroneRow`) | `ttk.Treeview`, style `Zephyr.Treeview` |
| Filter | `TextField` in `filterField` | `Filter.TEntry` with `Placeholder` |
| Metric tile | `MetricTile` (`Components.swift`) | `MetricTile` (`ui_widgets.py`) |
| Label and value rows | `ValueRow`, `StackedValue` | `KeyValueGrid` |
| Collapsible Details | `DisclosureGroup` in `DetailsSection` | `Disclosure` |
| Buttons | `QuietButtonStyle`, `SegmentedChoice` | `Icon.TButton`, `Primary.Icon.TButton`, `Tool.TButton`, `Link.TButton`, `Segment.Toolbutton` on `RoundedElements` images |
| Scrubber | `ReplayScrubber` | `TimelineSlider` |
| Event list | `EventListView` | `Events.Treeview` |
| Tooltip | `.help(...)` | `Tooltip` |
| 3D scene | SceneKit: `SceneController`, `DroneModel`, `OrbitCamera` | Canvas: `scene_view.py`, `scene_projection.py` |
| Scene legend | `SceneLegend` chip | `_draw_legend` (Late and Lost) |
| Viewer windows | `EvidenceReportView`, `ScenarioRunView`, `AboutReplayView` | `evidence_window`, `scenario_window`, `about_window` |
| Reduced motion | `accessibilityReduceMotion` with `Motion.duration` | Windows animation setting via `system_reduced_motion` |
| Update helper | `ReleaseUpdater.swift` (install script) | `tools/desktop/windows_updater.py`, `tools/release_updater.py` |
| Source rules test | `tools/tests/test_macos_ui_hygiene.py` | `tools/tests/test_windows_ui_hygiene.py`, `tools/tests/test_windows_polish.py` |

Differences that are deliberate or still open are called out in section 6 (strip captions, team list column order, selected drone kept in the list, Play button, zoom limit, row cap).

## 10. Updates and the self-test

Both apps accept `--self-test`. It runs without opening a window, prints `self-test ok` or `self-test failed: <reason>`, and exits with 0 or 1. The macOS app loads its bundled replay (`ZephyrMeshApp.swift`); the Windows app loads the bundled replay and starts a Tcl/Tk interpreter, so a build missing its Tk files fails here and not at the first window (`run_self_test` in `tools/desktop/windows_preview.py`).

Updates use the self-test as a gate (`ReleaseUpdater.swift`, `tools/release_updater.py`):

1. The app downloads the release and checks its digest.
2. A helper waits for the app to quit, checks the staged files again, and runs the new build with `--self-test`. If that fails or times out, the installed app is left untouched.
3. The helper moves the old app aside, puts the new one in place, starts it and watches it briefly. If the swap or the start fails, the old app is put back.

The wording the user sees comes from the `update` section of `ui_text.json`: `check`, `checking`, `upToDate`, `available`, `downloading`, `installing`, `failed` and `keptCurrent` ("The update could not start, so the current version was kept."). macOS reads them through `UpdateText` in `CockpitModel.swift`; Windows through `UI_TEXT["update"]` in `_check_update`. The install scripts also write the `keptCurrent` sentence as a literal string (the shell script in `ReleaseUpdater.swift` and `run_update_helper` in `tools/release_updater.py`), so change those together with the JSON.

## 11. Keyboard shortcuts

From `shortcuts` in `ui_text.json`. "Primary" is Command on macOS and Control on Windows.

| Action | Keys |
|---|---|
| Play/Pause | Space |
| Step forward | Right arrow |
| Step back | Left arrow |
| Restart | Home |
| Focus selected drone | F |
| Reset camera | R |
| Toggle 3D/top view | T |
| Show/hide keep-out volume | K |
| Show/hide inspector | I |
| Open evidence report... | Primary + O |
| Open scenario run... | Primary + Shift + O |
| Focus team filter | Primary + L |
| About this replay | Primary + I |
| Show results | Primary + R |

Single-key shortcuts never fire while a text field has focus (`shortcutRule`). On Windows a focused list or button also keeps its own arrow and Space keys.

## 12. Checklist for a new screen

Before you open a pull request for a new window, panel or view:

- [ ] It answers one question, and the answer is at the top.
- [ ] Every color, size, radius and duration comes from the generated module. No hex values, no raw font sizes, no system accent colors (the hygiene tests check this).
- [ ] Controls are `layout.controlHeight` tall and icon buttons are `iconButtonSize` square; content starts at the 16 pt content inset.
- [ ] Every visible label comes from `ui_text.json` or is a plain sentence; no snake_case, units in labels.
- [ ] Every status uses shape, color and word together, through `StatusStyle` (macOS) or the theme's status helpers (Windows). In a scene, status is a shape, not opacity.
- [ ] Missing values read "Not recorded".
- [ ] Red and amber appear only as status (Late, Lost, Inactive) or a load failure.
- [ ] Nothing claims to be live, and the "Simulation replay" badge is not repeated.
- [ ] Viewer-style windows follow verdict or plot, then summary, then collapsed Details.
- [ ] It works in dark and light; run `python3 tools/check_contrast.py` if you added a new text-on-fill pair.
- [ ] Animations are user-triggered, use `fastMs` or `standardMs`, and drop to 0 with reduced motion.
- [ ] Keyboard: reachable with Tab, a visible accent focus ring, and any new shortcut added to `ui_text.json` (never typed into the view).
- [ ] Both platforms get the same concept (section 9), or the gap is written down.
- [ ] `python3 tools/generate_design.py --check`, `python3 tools/build_design_reference.py --check` and `python3 -m pytest -q` pass.

## 13. Open decisions

These are in the code today but not signed off by the project owner. Do not treat them as settled.

| Item | State |
|---|---|
| Six text-on-fill pairs below 4.5:1 (section 3) | NEEDS-APPROVAL in `contrast.needs-approval`; the apps avoid them |
| Light-theme values marked "proposed" (`surface`, `line`, `lineStrong`, `target`, `keepOut`) | In use, awaiting sign-off |
| Zoom-out limit: macOS 3x swarm radius, Windows 6x for both orbit and top | The apps differ; one limit has not been chosen |
| App icon uses a drop shadow and a gradient | An exception to the "no shadows or gradients" rule, awaiting approval |
| Windows team list orders its columns Loss then Age; macOS uses Age then Loss | Not yet decided whether to match |
