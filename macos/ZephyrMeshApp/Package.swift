// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ZephyrMeshApp",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "ZephyrMeshApp", targets: ["ZephyrMeshApp"])
    ],
    targets: [
        .executableTarget(
            name: "ZephyrMeshApp",
            resources: [.process("Resources")]
        )
    ]
)
