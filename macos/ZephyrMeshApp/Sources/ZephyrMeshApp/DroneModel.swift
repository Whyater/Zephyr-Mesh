import AppKit
import SceneKit

/// Procedural X-frame quadcopter at true scale. Arm length and propeller diameter come from
/// the drone's own hardware profile in the fixture (the decoder rejects a fixture whose drones
/// reference a missing profile, or whose profile has a non-positive arm or propeller); the
/// remaining parts are proportions of the arm length for a small 3 inch class frame. The
/// model faces +x, which is heading 0 in the simulator frame.
struct DroneDimensions: Hashable {
    let armLength: CGFloat
    let propellerRadius: CGFloat

    init(profile: HardwareProfile) {
        armLength = CGFloat(profile.armLengthM)
        propellerRadius = CGFloat(profile.propellerDiameterM / 2)
    }

    var motorRadius: CGFloat { armLength * 0.11 }
    var motorHeight: CGFloat { armLength * 0.13 }
    var armWidth: CGFloat { armLength * 0.12 }
    var armThickness: CGFloat { armLength * 0.05 }
    var plateLength: CGFloat { armLength * 0.78 }
    var plateWidth: CGFloat { armLength * 0.48 }
    var plateHeight: CGFloat { armLength * 0.06 }
    var stackHeight: CGFloat { armLength * 0.2 }
    var propHeight: CGFloat { armThickness + motorHeight + armLength * 0.02 }
    /// Radius of the invisible sphere used for hover and click hit testing.
    var hitRadius: CGFloat { armLength + propellerRadius + 0.04 }
}

/// Materials for one look of the drone. Variants share geometry and differ only in materials.
enum DroneLook: CaseIterable {
    case normal, hovered, selected
}

@MainActor
struct DroneModelFactory {
    let dimensions: DroneDimensions

    /// Static parts flattened into one node: plate, stack, arms, motors, camera, rotor discs.
    func staticBody(look: DroneLook, palette: Theme.Palette, scene: ScenePalette) -> SCNNode {
        let root = SCNNode()
        let shell = shellMaterial(look: look, palette: palette, scene: scene)
        let frame = frameMaterial(look: look, palette: palette, scene: scene)
        let metal = metalMaterial(scene: scene)
        let lens = lensMaterial(palette: palette)
        let rotorDisc = discMaterial(scene: scene)
        let d = dimensions

        let plate = SCNBox(width: d.plateLength, height: d.plateHeight, length: d.plateWidth, chamferRadius: d.plateHeight * 0.4)
        plate.chamferSegmentCount = 2
        plate.materials = [frame]
        root.addChildNode(node(plate, at: SCNVector3(0, d.armThickness + d.plateHeight / 2, 0)))

        let stack = SCNBox(width: d.plateLength * 0.72, height: d.stackHeight, length: d.plateWidth * 0.72,
                           chamferRadius: d.stackHeight * 0.3)
        stack.chamferSegmentCount = 3
        stack.materials = [shell]
        root.addChildNode(node(stack, at: SCNVector3(-d.plateLength * 0.04, d.armThickness + d.plateHeight + d.stackHeight / 2, 0)))

        let camera = SCNCylinder(radius: d.plateHeight * 1.6, height: d.plateLength * 0.16)
        camera.radialSegmentCount = 16
        camera.materials = [lens]
        let cameraNode = node(camera, at: SCNVector3(d.plateLength * 0.5, d.armThickness + d.plateHeight + d.stackHeight * 0.45, 0))
        cameraNode.eulerAngles.z = -.pi / 2
        root.addChildNode(cameraNode)

        for (index, tip) in motorPositions().enumerated() {
            let arm = SCNBox(width: d.armLength, height: d.armThickness, length: d.armWidth, chamferRadius: d.armThickness * 0.4)
            arm.chamferSegmentCount = 1
            arm.materials = [frame]
            let armNode = node(arm, at: SCNVector3(tip.x / 2, d.armThickness / 2, tip.z / 2))
            armNode.eulerAngles.y = armAngle(index)
            root.addChildNode(armNode)

            let motor = SCNCylinder(radius: d.motorRadius, height: d.motorHeight)
            motor.radialSegmentCount = 20
            motor.materials = [metal]
            root.addChildNode(node(motor, at: SCNVector3(tip.x, d.armThickness + d.motorHeight / 2, tip.z)))

            let disc = SCNCylinder(radius: d.propellerRadius, height: 0.0006)
            disc.radialSegmentCount = 32
            disc.materials = [rotorDisc]
            root.addChildNode(node(disc, at: SCNVector3(tip.x, d.propHeight, tip.z)))
        }
        let flattened = root.flattenedClone()
        flattened.castsShadow = true
        return flattened
    }

    /// One two-blade propeller, spun by the scene when motion is allowed.
    func propellerBlades(scene: ScenePalette) -> SCNGeometry {
        let d = dimensions
        let blade = SCNBox(width: d.propellerRadius * 1.9, height: 0.0012, length: d.propellerRadius * 0.2,
                           chamferRadius: 0.0006)
        let material = SCNMaterial()
        material.lightingModel = .physicallyBased
        material.diffuse.contents = scene.rotorDiscNS
        material.roughness.contents = 0.6
        material.transparency = 0.55
        material.writesToDepthBuffer = false
        blade.materials = [material]
        return blade
    }

    func motorPositions() -> [SCNVector3] {
        (0..<4).map { index in
            let angle = Double.pi / 4 + Double(index) * Double.pi / 2
            return SCNVector3(dimensions.armLength * CGFloat(cos(angle)), 0, dimensions.armLength * CGFloat(sin(angle)))
        }
    }

    private func armAngle(_ index: Int) -> CGFloat {
        -(CGFloat.pi / 4 + CGFloat(index) * CGFloat.pi / 2)
    }

    private func node(_ geometry: SCNGeometry, at position: SCNVector3) -> SCNNode {
        let node = SCNNode(geometry: geometry)
        node.position = position
        return node
    }

    // MARK: Materials

    private func pbr(_ color: NSColor, metalness: CGFloat, roughness: CGFloat) -> SCNMaterial {
        let material = SCNMaterial()
        material.lightingModel = .physicallyBased
        material.diffuse.contents = color
        material.metalness.contents = metalness
        material.roughness.contents = roughness
        return material
    }

    private func bodyColor(look: DroneLook, palette: Theme.Palette, scene: ScenePalette) -> NSColor {
        switch look {
        case .normal: return scene.droneBodyNS
        case .hovered: return palette.textNS
        case .selected: return palette.accentNS
        }
    }

    private func shellMaterial(look: DroneLook, palette: Theme.Palette, scene: ScenePalette) -> SCNMaterial {
        pbr(bodyColor(look: look, palette: palette, scene: scene), metalness: 0.05, roughness: 0.42)
    }

    private func frameMaterial(look: DroneLook, palette: Theme.Palette, scene: ScenePalette) -> SCNMaterial {
        // The frame stays neutral in every look; selection and hover recolor only the top shell.
        pbr(scene.droneArmsNS, metalness: 0.15, roughness: 0.7)
    }

    private func metalMaterial(scene: ScenePalette) -> SCNMaterial {
        pbr(scene.droneBodyNS, metalness: 0.85, roughness: 0.32)
    }

    private func lensMaterial(palette: Theme.Palette) -> SCNMaterial {
        pbr(palette.textNS, metalness: 0.2, roughness: 0.25)
    }

    private func discMaterial(scene: ScenePalette) -> SCNMaterial {
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = scene.rotorDiscNS
        material.transparency = 0.16
        material.isDoubleSided = true
        material.writesToDepthBuffer = false
        return material
    }
}
