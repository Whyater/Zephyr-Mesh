// Draws the Zephyr Mesh app icon with CoreGraphics and writes PNG frames.
//
// Usage: swift tools/make_icon.swift <outdir> [--tokens design/horizon_tokens.json]
//
// Writes <outdir>/ZephyrMesh.iconset/ (the ten macOS iconset PNGs) and
// <outdir>/windows/ZephyrMesh-<size>.png for the Windows .ico frames.
// Every color is read from the dark palette in design/horizon_tokens.json.
// tools/build_icon.py runs this, then iconutil and tools/pack_ico.py.
import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

// MARK: Tokens

struct IconPalette {
    let background: CGColor
    let surface: CGColor
    let lineStrong: CGColor
    let text: CGColor
    let accent: CGColor
    let keepOut: CGColor
}

func color(fromHex text: String) -> CGColor {
    let clean = text.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
    guard clean.count == 6, let value = UInt32(clean, radix: 16) else {
        fatalError("not a six digit hex color: \(text)")
    }
    let space = CGColorSpace(name: CGColorSpace.sRGB)!
    let parts: [CGFloat] = [
        CGFloat((value >> 16) & 0xFF) / 255,
        CGFloat((value >> 8) & 0xFF) / 255,
        CGFloat(value & 0xFF) / 255,
        1,
    ]
    return CGColor(colorSpace: space, components: parts)!
}

func loadPalette(_ path: URL) -> IconPalette {
    guard let data = try? Data(contentsOf: path),
          let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let colors = root["colors"] as? [String: Any],
          let dark = colors["dark"] as? [String: Any] else {
        fatalError("could not read the dark palette from \(path.path)")
    }
    func token(_ name: String) -> CGColor {
        if let text = dark[name] as? String { return color(fromHex: text) }
        if let item = dark[name] as? [String: Any], let text = item["value"] as? String {
            return color(fromHex: text)
        }
        fatalError("token \(name) is missing from the dark palette")
    }
    return IconPalette(
        background: token("background"), surface: token("surface"),
        lineStrong: token("lineStrong"), text: token("text"), accent: token("accent"),
        keepOut: token("keepOut"))
}

func withAlpha(_ base: CGColor, _ alpha: CGFloat) -> CGColor {
    base.copy(alpha: alpha)!
}

// MARK: Geometry (all in a 1024 x 1024 design space, y pointing down)

/// Body placement. macOS uses Apple's icon grid (824 pt body, 100 pt margin);
/// Windows icons sit closer to the edge because the shell adds no margin.
struct Layout {
    let inset: CGFloat
    let radius: CGFloat
    let shadow: Bool

    static let macOS = Layout(inset: 100, radius: 185, shadow: true)
    static let windows = Layout(inset: 40, radius: 200, shadow: false)
}

/// A rounded square with continuous (squircle style) corners, the shape macOS
/// uses for app icons. The corner is three cubic curves per corner; the
/// control values are the widely used approximation of Apple's curve.
func continuousRoundedRect(_ rect: CGRect, radius: CGFloat) -> CGPath {
    let r = min(radius, min(rect.width, rect.height) / 2 / 1.52866483)
    let path = CGMutablePath()
    // One corner as offsets (along the edge, away from the edge) from the corner point.
    let corner: [(CGFloat, CGFloat)] = [
        (1.52866471, 0), (1.08849323, 0), (0.86840689, 0), (0.66993427, 0.06549600),
        (0.37282997, 0.16404221), (0.16404148, 0.37282972), (0.06549569, 0.66993493),
        (0, 0.86840701), (0, 1.08849299), (0, 1.52866483),
    ]
    // Each corner maps (along, away) to a point; listed clockwise in y-down space.
    let maps: [(CGFloat, CGFloat) -> CGPoint] = [
        { a, b in CGPoint(x: rect.maxX - a * r, y: rect.minY + b * r) },  // top right
        { a, b in CGPoint(x: rect.maxX - b * r, y: rect.maxY - a * r) },  // bottom right
        { a, b in CGPoint(x: rect.minX + a * r, y: rect.maxY - b * r) },  // bottom left
        { a, b in CGPoint(x: rect.minX + b * r, y: rect.minY + a * r) },  // top left
    ]
    for (index, map) in maps.enumerated() {
        let p = corner.map { map($0.0, $0.1) }
        if index == 0 { path.move(to: p[0]) } else { path.addLine(to: p[0]) }
        path.addCurve(to: p[3], control1: p[1], control2: p[2])
        path.addCurve(to: p[6], control1: p[4], control2: p[5])
        path.addCurve(to: p[9], control1: p[7], control2: p[8])
    }
    path.closeSubpath()
    return path
}

/// How much detail a frame of `pixels` size can carry.
enum Detail {
    case small   // 16 to 32 px: ring and center only, no thin links
    case medium  // 48 to 64 px: links drawn at least about one pixel wide
    case full

    init(pixels: Int) {
        if pixels <= 32 { self = .small } else if pixels <= 64 { self = .medium } else { self = .full }
    }
}

struct Mark {
    /// Center of the orbit and of the tracked point.
    static let center = CGPoint(x: 540, y: 482)
    static let ringRadius: CGFloat = 230
    /// Drone angles in degrees (y-down, 0 is 3 o'clock, 90 is 6 o'clock). Five drones,
    /// 60 degrees apart, leave the lower-left slot open for the gust.
    static let nodeDegrees: [CGFloat] = [30, -30, -90, -150, -210]
    /// The orbit starts at the bottom and runs through the right side, the top and the
    /// left side, ending under the last drone (-210 degrees, the same as 150).
    static let orbitStartDegrees: CGFloat = 90
    static let orbitEndDegrees: CGFloat = -210
    /// The gust: a gentle curve from the west that meets the orbit at the bottom, tangent to it.
    static let gustStart = CGPoint(x: 152, y: 770)
    static let gustControl = CGPoint(x: 352, y: 712)
    static let gustFadeEndX: CGFloat = 282
    /// A second, shorter breath of wind under the gust, thin at both ends.
    static let breezeStart = CGPoint(x: 262, y: 828)
    static let breezeControl = CGPoint(x: 410, y: 790)
    static let breezeEnd = CGPoint(x: 560, y: 790)

    static func point(degrees: CGFloat, radius: CGFloat = ringRadius) -> CGPoint {
        let angle = degrees * .pi / 180
        return CGPoint(x: center.x + radius * cos(angle), y: center.y + radius * sin(angle))
    }

    static var nodes: [CGPoint] { nodeDegrees.map { point(degrees: $0) } }

    /// The gust then the orbit as one polyline, plus the gust's share of the points.
    static func streamline(samples: Int = 240) -> (points: [CGPoint], gustCount: Int) {
        let bottom = point(degrees: orbitStartDegrees)
        var points: [CGPoint] = []
        let gustSamples = samples / 4
        for step in 0..<gustSamples {
            let t = CGFloat(step) / CGFloat(gustSamples)
            let u = 1 - t
            points.append(CGPoint(
                x: u * u * gustStart.x + 2 * u * t * gustControl.x + t * t * bottom.x,
                y: u * u * gustStart.y + 2 * u * t * gustControl.y + t * t * bottom.y))
        }
        let orbitSamples = samples - gustSamples
        for step in 0...orbitSamples {
            let t = CGFloat(step) / CGFloat(orbitSamples)
            points.append(point(degrees: orbitStartDegrees + (orbitEndDegrees - orbitStartDegrees) * t))
        }
        return (points, gustSamples)
    }
}

/// A filled shape that follows `points` with a width that changes along the way.
func taperedStroke(_ points: [CGPoint], widths: [CGFloat]) -> CGPath {
    var left: [CGPoint] = []
    var right: [CGPoint] = []
    for index in points.indices {
        let previous = points[max(index - 1, 0)]
        let next = points[min(index + 1, points.count - 1)]
        let dx = next.x - previous.x
        let dy = next.y - previous.y
        let length = max(hypot(dx, dy), 0.0001)
        let normal = CGPoint(x: -dy / length, y: dx / length)
        let half = widths[index] / 2
        left.append(CGPoint(x: points[index].x + normal.x * half, y: points[index].y + normal.y * half))
        right.append(CGPoint(x: points[index].x - normal.x * half, y: points[index].y - normal.y * half))
    }
    let path = CGMutablePath()
    path.addLines(between: left + right.reversed())
    path.closeSubpath()
    return path
}

func quadratic(_ start: CGPoint, _ control: CGPoint, _ end: CGPoint, samples: Int) -> [CGPoint] {
    (0...samples).map { step in
        let t = CGFloat(step) / CGFloat(samples)
        let u = 1 - t
        return CGPoint(x: u * u * start.x + 2 * u * t * control.x + t * t * end.x,
                       y: u * u * start.y + 2 * u * t * control.y + t * t * end.y)
    }
}

/// Mix two colors (t = 0 gives `first`).
func mix(_ first: CGColor, _ second: CGColor, _ t: CGFloat) -> CGColor {
    let a = first.components!
    let b = second.components!
    return CGColor(colorSpace: first.colorSpace!,
                   components: (0..<4).map { a[$0] + (b[$0] - a[$0]) * t })!
}

func circle(_ center: CGPoint, _ radius: CGFloat) -> CGRect {
    CGRect(x: center.x - radius, y: center.y - radius, width: 2 * radius, height: 2 * radius)
}

// MARK: Drawing

func drawIcon(_ ctx: CGContext, palette: IconPalette, layout: Layout, detail: Detail, pixels: Int) {
    let unit = CGFloat(pixels) / 1024  // pixels per design unit
    func atLeastPixels(_ design: CGFloat, _ minimumPixels: CGFloat) -> CGFloat {
        max(design, minimumPixels / unit)
    }
    let body = CGRect(x: layout.inset, y: layout.inset, width: 1024 - 2 * layout.inset,
                      height: 1024 - 2 * layout.inset)
    let shape = continuousRoundedRect(body, radius: layout.radius)
    let space = CGColorSpace(name: CGColorSpace.sRGB)!

    // Soft drop shadow under the body (macOS grid only). Shadow offsets are in device
    // space, where y points up, so a negative height moves the shadow down.
    if layout.shadow {
        ctx.saveGState()
        ctx.setShadow(offset: CGSize(width: 0, height: -10 * unit), blur: 28 * unit,
                      color: CGColor(gray: 0, alpha: 0.3))
        ctx.addPath(shape)
        ctx.setFillColor(palette.background)
        ctx.fillPath()
        ctx.restoreGState()
    }

    // Body: a subtle top-to-bottom gradient from surface to background, and a faint
    // inner edge so the dark tile keeps its outline on a dark Dock.
    ctx.saveGState()
    ctx.addPath(shape)
    ctx.clip()
    let gradient = CGGradient(colorsSpace: space, colors: [palette.surface, palette.background] as CFArray,
                              locations: [0, 1])!
    ctx.drawLinearGradient(gradient, start: CGPoint(x: 512, y: body.minY),
                           end: CGPoint(x: 512, y: body.maxY), options: [])
    ctx.addPath(shape)
    ctx.setStrokeColor(withAlpha(palette.lineStrong, 0.45))
    ctx.setLineWidth(atLeastPixels(8, 1))
    ctx.strokePath()

    let small = detail == .small
    let c = Mark.center
    let nodes = Mark.nodes
    let nodeRadius: CGFloat = small ? 46 : 40
    let nodeGap: CGFloat = small ? 0 : 14
    let core: CGFloat = small ? 88 : 50
    ctx.setLineCap(.round)

    // Soft glow behind the tracked point.
    if !small {
        let glow = CGGradient(colorsSpace: space,
                              colors: [withAlpha(palette.keepOut, 0.26), withAlpha(palette.keepOut, 0)] as CFArray,
                              locations: [0, 1])!
        ctx.drawRadialGradient(glow, startCenter: c, startRadius: 0, endCenter: c,
                               endRadius: Mark.ringRadius * 0.72, options: [])
    }

    // Mesh links: each drone shares its view of the target. Gaps at both ends keep
    // them light; they are left out at 16 and 32 px.
    if !small {
        ctx.setStrokeColor(withAlpha(palette.keepOut, 0.42))
        ctx.setLineWidth(atLeastPixels(6, 1))
        for node in nodes {
            let dx = c.x - node.x
            let dy = c.y - node.y
            let length = hypot(dx, dy)
            let from = nodeRadius + nodeGap
            let to = length - core - 18
            ctx.move(to: CGPoint(x: node.x + dx / length * from, y: node.y + dy / length * from))
            ctx.addLine(to: CGPoint(x: node.x + dx / length * to, y: node.y + dy / length * to))
        }
        ctx.strokePath()
    }

    // The gust and the orbit: one stroke that arrives from the west, thin and faint,
    // and becomes the ring the team flies on.
    let ringWidth: CGFloat = small ? 48 : 20
    let line = Mark.streamline()
    var widths: [CGFloat] = []
    for index in line.points.indices {
        if index < line.gustCount {
            let t = CGFloat(index) / CGFloat(line.gustCount)
            widths.append(ringWidth * (0.25 + 0.75 * t))
        } else {
            widths.append(ringWidth)
        }
    }
    ctx.saveGState()
    ctx.addPath(taperedStroke(line.points, widths: widths))
    ctx.clip()
    let fade = CGGradient(colorsSpace: space, colors: [withAlpha(palette.accent, 0.1), palette.accent] as CFArray,
                          locations: [0, 1])!
    ctx.drawLinearGradient(fade, start: CGPoint(x: Mark.gustStart.x, y: 0), end: CGPoint(x: Mark.gustFadeEndX, y: 0),
                           options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
    ctx.restoreGState()

    // The breeze: drawn from 48 px up, where it is wide enough to see.
    if !small {
        let points = quadratic(Mark.breezeStart, Mark.breezeControl, Mark.breezeEnd, samples: 64)
        let breezeWidths = points.indices.map { index -> CGFloat in
            let t = CGFloat(index) / CGFloat(points.count - 1)
            return atLeastPixels(12, 1) * sin(t * .pi) + 0.5
        }
        ctx.saveGState()
        ctx.addPath(taperedStroke(points, widths: breezeWidths))
        ctx.clip()
        let breeze = CGGradient(colorsSpace: space,
                                colors: [withAlpha(palette.keepOut, 0), withAlpha(palette.keepOut, 0.7),
                                         withAlpha(palette.keepOut, 0)] as CFArray,
                                locations: [0, 0.6, 1])!
        ctx.drawLinearGradient(breeze, start: CGPoint(x: Mark.breezeStart.x, y: 0),
                               end: CGPoint(x: Mark.breezeEnd.x, y: 0),
                               options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
        ctx.restoreGState()
    }

    // Drone nodes, each with a knockout gap so the orbit never touches the dot. The gap
    // takes the body gradient's color at that height so it stays invisible.
    for node in nodes {
        if nodeGap > 0 {
            ctx.setFillColor(mix(palette.surface, palette.background, (node.y - body.minY) / body.height))
            ctx.fillEllipse(in: circle(node, nodeRadius + nodeGap))
        }
        ctx.setFillColor(palette.accent)
        ctx.fillEllipse(in: circle(node, nodeRadius))
    }

    // The tracked point the team agrees on.
    ctx.setFillColor(palette.text)
    ctx.fillEllipse(in: circle(c, core))
    ctx.restoreGState()
}

func render(pixels: Int, palette: IconPalette, layout: Layout) -> CGImage {
    let space = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(data: nil, width: pixels, height: pixels, bitsPerComponent: 8,
                              bytesPerRow: 0, space: space,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        fatalError("could not make a \(pixels) px bitmap")
    }
    ctx.setAllowsAntialiasing(true)
    ctx.setShouldAntialias(true)
    ctx.interpolationQuality = .high
    // Flip to y-down and scale the 1024 design space to the frame size.
    ctx.translateBy(x: 0, y: CGFloat(pixels))
    ctx.scaleBy(x: CGFloat(pixels) / 1024, y: -CGFloat(pixels) / 1024)
    drawIcon(ctx, palette: palette, layout: layout, detail: Detail(pixels: pixels), pixels: pixels)
    return ctx.makeImage()!
}

func writePNG(_ image: CGImage, to url: URL) {
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        fatalError("could not create \(url.path)")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { fatalError("could not write \(url.path)") }
}

// MARK: Main

var arguments = Array(CommandLine.arguments.dropFirst())
let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
var tokensURL = repoRoot.appendingPathComponent("design/horizon_tokens.json")
if let flag = arguments.firstIndex(of: "--tokens"), flag + 1 < arguments.count {
    tokensURL = URL(fileURLWithPath: arguments[flag + 1])
    arguments.removeSubrange(flag...(flag + 1))
}
guard let outdir = arguments.first else {
    FileHandle.standardError.write("usage: swift tools/make_icon.swift <outdir> [--tokens path]\n".data(using: .utf8)!)
    exit(2)
}
let palette = loadPalette(tokensURL)
let out = URL(fileURLWithPath: outdir)
let iconset = out.appendingPathComponent("ZephyrMesh.iconset")
let windows = out.appendingPathComponent("windows")
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
try FileManager.default.createDirectory(at: windows, withIntermediateDirectories: true)

for points in [16, 32, 128, 256, 512] {
    writePNG(render(pixels: points, palette: palette, layout: .macOS),
             to: iconset.appendingPathComponent("icon_\(points)x\(points).png"))
    writePNG(render(pixels: points * 2, palette: palette, layout: .macOS),
             to: iconset.appendingPathComponent("icon_\(points)x\(points)@2x.png"))
}
for pixels in [16, 24, 32, 48, 64, 128, 256] {
    writePNG(render(pixels: pixels, palette: palette, layout: .windows),
             to: windows.appendingPathComponent("ZephyrMesh-\(pixels).png"))
}
print("wrote \(iconset.path) and \(windows.path)")
